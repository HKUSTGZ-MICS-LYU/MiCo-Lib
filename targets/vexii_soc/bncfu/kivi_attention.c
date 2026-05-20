#ifdef KIVI_BNCFU_INT8_BDOT

#include "nn.h"
#include "profile.h"
#include "mico_qnn.h"
#include "mico_quant.h"

#include <math.h>
#include <string.h>

#ifndef VLEN
#define VLEN 256
#endif

#define KIVI_BDOT_Q8_LANES (VLEN / 8)
#define KIVI_BDOT_LOAD_BYTES (VLEN / 8)
#define EXP_LUT_SIZE 256
#define EXP_LUT_MAX  16.0f
#define EXP_LUT_STEP (EXP_LUT_MAX / (float)EXP_LUT_SIZE)

extern long ATTN_TIMER;
extern long SOFTMAX_TIMER;
extern long EXPF_TIMER;

#ifdef KIVI_ATTN_REF
void MiCo_ViT_kivi_attention_ref_f32(
    Tensor4D_F32 *y,
    const Tensor4D_F32 *q,
    const Tensor4D_F32 *k,
    const Tensor4D_F32 *v,
    const float scale
);
#endif

static float exp_lut[EXP_LUT_SIZE];
static int exp_lut_ready = 0;

static inline size_t idx4(size_t i0, size_t i1, size_t i2, size_t i3, size_t d1, size_t d2, size_t d3){
    return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

static inline int kivi_decode_ternary(uint8_t packed, size_t lane){
    const int bits = (packed >> (2 * lane)) & 0x3;
    return (bits == 1) ? 1 : (bits == 3) ? -1 : 0;
}

static inline float kivi_apply_ternary(float x, uint8_t packed, size_t lane){
    const int bits = (packed >> (2 * lane)) & 0x3;
    return (bits == 1) ? x : (bits == 3) ? -x : 0.0f;
}

static inline void kivi_accum_scores_from_k(float *scores, float q_scaled, const qbyte *k_packed, size_t J){
    if (q_scaled == 0.0f) return;

    size_t j = 0;
    const size_t full_bytes = J / 4;
    for (size_t byte_idx = 0; byte_idx < full_bytes; byte_idx++){
        const uint8_t packed = (uint8_t)k_packed[byte_idx];
        int v0 = kivi_decode_ternary(packed, 0);
        int v1 = kivi_decode_ternary(packed, 1);
        int v2 = kivi_decode_ternary(packed, 2);
        int v3 = kivi_decode_ternary(packed, 3);
        if (v0 > 0) scores[j + 0] += q_scaled;
        else if (v0 < 0) scores[j + 0] -= q_scaled;
        if (v1 > 0) scores[j + 1] += q_scaled;
        else if (v1 < 0) scores[j + 1] -= q_scaled;
        if (v2 > 0) scores[j + 2] += q_scaled;
        else if (v2 < 0) scores[j + 2] -= q_scaled;
        if (v3 > 0) scores[j + 3] += q_scaled;
        else if (v3 < 0) scores[j + 3] -= q_scaled;
        j += 4;
    }

    if (j < J){
        const uint8_t packed = (uint8_t)k_packed[full_bytes];
        for (size_t lane = 0; j < J; lane++, j++){
            int v = kivi_decode_ternary(packed, lane);
            if (v > 0) scores[j] += q_scaled;
            else if (v < 0) scores[j] -= q_scaled;
        }
    }
}

static void MiCo_init_exp_lut(void){
    if (exp_lut_ready) return;
    for (int i = 0; i < EXP_LUT_SIZE; i++){
        exp_lut[i] = expf(-(float)i * EXP_LUT_STEP);
    }
    exp_lut_ready = 1;
}

static float MiCo_expf(float x){
    long start = MiCo_time();
    float res;
#ifdef EXP_ACCEL
    if (x >= 0.0f) return expf(x);
    if (x <= -EXP_LUT_MAX) return 0.0f;
    int idx = (int)(-x * (1.0f / EXP_LUT_STEP));
    if (idx >= EXP_LUT_SIZE) idx = EXP_LUT_SIZE - 1;
    res = exp_lut[idx];
#else
    res = expf(x);
#endif
    EXPF_TIMER += MiCo_time() - start;
    return res;
}

static void MiCo_softmax_vec(float *dst, const float *src, size_t n){
    long start = MiCo_time();
    float max_val = src[0];
    for (size_t i = 1; i < n; i++){
        if (src[i] > max_val) max_val = src[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++){
        dst[i] = MiCo_expf(src[i] - max_val);
        sum += dst[i];
    }
    for (size_t i = 0; i < n; i++){
        dst[i] /= sum;
    }
    SOFTMAX_TIMER += MiCo_time() - start;
}

static inline void bncfu_enable(void){
    __asm__ volatile(
        "li t1, 0x80000000\n\t"
        "csrs 0xBC0, t1"
        ::: "t1"
    );
}

static inline void bncfu_fence(void){
    __asm__ volatile(".word 0x0000100f" ::: "memory");
}

static inline void bncfu_load(unsigned int bank, const void *addr){
    uintptr_t addr_reg = (uintptr_t)addr;
    uintptr_t bank_reg = (uintptr_t)bank;
    __asm__ volatile(
        ".insn r 0x0B, 0x4, 0, x0, %0, %1"
        :: "r"(addr_reg), "r"(bank_reg) : "memory"
    );
}

static inline int32_t bncfu_bdot2(unsigned int int8_bank, unsigned int lowbit_bank){
    uintptr_t int8_reg = (uintptr_t)int8_bank;
    uintptr_t lowbit_reg = (uintptr_t)lowbit_bank;
    int32_t result;
    __asm__ volatile(
        ".insn r 0x0B, 0x1, 0, %0, %1, %2"
        : "=r"(result) : "r"(int8_reg), "r"(lowbit_reg)
    );
    return result;
}

static inline void kivi_pack_v_lane_bdot(qbyte *dst, const qbyte *v_block, size_t base_j, size_t lane, size_t count){
    memset(dst, 0, KIVI_BDOT_LOAD_BYTES);
    for (size_t k = 0; k < count; k++){
        const uint8_t code = ((uint8_t)v_block[base_j + k] >> (2 * lane)) & 0x3u;
        dst[k >> 2] |= (qbyte)(code << (2 * (k & 3)));
    }
}

static inline void kivi_pack_k_token_bdot(qbyte *dst, const qbyte *k_q, size_t packed_J, size_t base_f, size_t j, size_t count){
    const size_t byte_idx = j / 4;
    const size_t lane = j & 3;
    memset(dst, 0, KIVI_BDOT_LOAD_BYTES);
    for (size_t k = 0; k < count; k++){
        const uint8_t code = ((uint8_t)k_q[(base_f + k) * packed_J + byte_idx] >> (2 * lane)) & 0x3u;
        dst[k >> 2] |= (qbyte)(code << (2 * (k & 3)));
    }
}

void MiCo_ViT_kivi_attention_f32(
    Tensor4D_F32 *y,
    const Tensor4D_F32 *q,
    const Tensor4D_F32 *k,
    const Tensor4D_F32 *v,
    const float scale
){
#ifdef KIVI_ATTN_REF
    MiCo_ViT_kivi_attention_ref_f32(y, q, k, v, scale);
    return;
#endif

    const size_t B = q->shape[0];
    const size_t H = q->shape[1];
    const size_t I = q->shape[2];
    const size_t F = q->shape[3];
    const size_t J = k->shape[2];

    MiCo_assert(k->shape[0] == B && k->shape[1] == H && k->shape[3] == F, "[KIVI BNCFU Attention] k shape mismatch");
    MiCo_assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == J && v->shape[3] == F, "[KIVI BNCFU Attention] v shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == I && y->shape[2] == H && y->shape[3] == F, "[KIVI BNCFU Attention] y shape mismatch");
    MiCo_assert(scale != 0.0f, "[KIVI BNCFU Attention] scale must be non-zero");

    MiCo_init_exp_lut();

    const size_t packed_J = (J + 3) / 4;
    const size_t packed_F = (F + 3) / 4;
    const size_t padded_J = packed_J * 4;
    const size_t padded_F = packed_F * 4;
    const size_t padded_J_bdot = ((J + KIVI_BDOT_Q8_LANES - 1) / KIVI_BDOT_Q8_LANES) * KIVI_BDOT_Q8_LANES;
    const size_t padded_F_bdot = ((F + KIVI_BDOT_Q8_LANES - 1) / KIVI_BDOT_Q8_LANES) * KIVI_BDOT_Q8_LANES;
    const size_t j_bdot_chunks = padded_J_bdot / KIVI_BDOT_Q8_LANES;
    const size_t f_bdot_chunks = padded_F_bdot / KIVI_BDOT_Q8_LANES;

    float *scores = (float *)malloc(J * sizeof(float));
    float *output_buf = (float *)malloc(F * sizeof(float));
    float *q_scaled_buf = (float *)MiCo_alloc(F * sizeof(float), MICO_ALIGN);
    float *score_scaled_buf = (float *)MiCo_alloc(J * sizeof(float), MICO_ALIGN);
    qbyte *k_q = (qbyte *)malloc(F * packed_J * sizeof(qbyte));
    float *k_scales = (float *)malloc(F * sizeof(float));
    qbyte *v_q = (qbyte *)malloc(packed_F * J * sizeof(qbyte));
    float *v_scales = (float *)malloc(J * sizeof(float));
    size_t max_buf = padded_J > padded_F ? padded_J : padded_F;
    float *temp_buf = (float *)malloc(max_buf * sizeof(float));
    qbyte *quant_tmp = (qbyte *)malloc((packed_J > packed_F ? packed_J : packed_F) * sizeof(qbyte));
    int8_t *q_i8 = (int8_t *)MiCo_alloc(padded_F_bdot * sizeof(int8_t), MICO_ALIGN);
    int8_t *score_i8 = (int8_t *)MiCo_alloc(padded_J_bdot * sizeof(int8_t), MICO_ALIGN);
    qbyte *k_bdot = (qbyte *)MiCo_alloc(J * f_bdot_chunks * KIVI_BDOT_LOAD_BYTES * sizeof(qbyte), MICO_ALIGN);
    qbyte *v_bdot = (qbyte *)MiCo_alloc(packed_F * 4 * j_bdot_chunks * KIVI_BDOT_LOAD_BYTES * sizeof(qbyte), MICO_ALIGN);

    MiCo_assert(scores != NULL && output_buf != NULL && q_scaled_buf != NULL && score_scaled_buf != NULL &&
                k_q != NULL && k_scales != NULL && v_q != NULL && v_scales != NULL &&
                temp_buf != NULL && quant_tmp != NULL &&
                q_i8 != NULL && score_i8 != NULL && k_bdot != NULL && v_bdot != NULL,
                "[KIVI BNCFU Attention] failed to allocate buffers");

    long start_time = MiCo_time();
#ifdef KIVI_PROFILE_INTERNAL
    long kivi_prof_k_gather = 0;
    long kivi_prof_k_quant = 0;
    long kivi_prof_v_copy = 0;
    long kivi_prof_v_quant = 0;
    long kivi_prof_v_layout = 0;
    long kivi_prof_score_accum = 0;
    long kivi_prof_softmax = 0;
    long kivi_prof_output_accum = 0;
    long kivi_prof_output_store = 0;
#endif

    bncfu_enable();

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            size_t k_base = idx4(b, h, 0, 0, H, J, F);
            size_t v_base = idx4(b, h, 0, 0, H, J, F);

            for (size_t f = 0; f < F; f++){
#ifdef KIVI_PROFILE_INTERNAL
                long prof_start = MiCo_time();
#endif
                for (size_t j = 0; j < J; j++){
                    temp_buf[j] = k->data[k_base + j * F + f];
                }
                for (size_t j = J; j < padded_J; j++){
                    temp_buf[j] = 0.0f;
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_k_gather += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif
                k_scales[f] = __FP32toQ2T(k_q + f * packed_J, temp_buf, J);
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_k_quant += MiCo_time() - prof_start;
#endif
            }

#ifdef KIVI_PROFILE_INTERNAL
            {
                long prof_start = MiCo_time();
#endif
                for (size_t j = 0; j < J; j++){
                    for (size_t fc = 0; fc < f_bdot_chunks; fc++){
                        const size_t f_base = fc * KIVI_BDOT_Q8_LANES;
                        const size_t remain = f_base < F ? F - f_base : 0;
                        const size_t count = remain < KIVI_BDOT_Q8_LANES ? remain : KIVI_BDOT_Q8_LANES;
                        qbyte *dst = k_bdot + (j * f_bdot_chunks + fc) * KIVI_BDOT_LOAD_BYTES;
                        kivi_pack_k_token_bdot(dst, k_q, packed_J, f_base, j, count);
                    }
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_k_quant += MiCo_time() - prof_start;
            }
#endif

            for (size_t j = 0; j < J; j++){
#ifdef KIVI_PROFILE_INTERNAL
                long prof_start = MiCo_time();
#endif
                memcpy(temp_buf, &v->data[v_base + j * F], F * sizeof(float));
                for (size_t f = F; f < padded_F; f++){
                    temp_buf[f] = 0.0f;
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_v_copy += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif
                v_scales[j] = __FP32toQ2T(quant_tmp, temp_buf, F);
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_v_quant += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif
                for (size_t fb = 0; fb < packed_F; fb++){
                    v_q[fb * J + j] = quant_tmp[fb];
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_v_layout += MiCo_time() - prof_start;
#endif
            }

#ifdef KIVI_PROFILE_INTERNAL
            {
                long prof_start = MiCo_time();
#endif
                for (size_t fb = 0; fb < packed_F; fb++){
                    const qbyte *v_block = v_q + fb * J;
                    for (size_t lane = 0; lane < 4; lane++){
                        for (size_t jc = 0; jc < j_bdot_chunks; jc++){
                            const size_t j_base = jc * KIVI_BDOT_Q8_LANES;
                            const size_t remain = j_base < J ? J - j_base : 0;
                            const size_t count = remain < KIVI_BDOT_Q8_LANES ? remain : KIVI_BDOT_Q8_LANES;
                            qbyte *dst = v_bdot + ((fb * 4 + lane) * j_bdot_chunks + jc) * KIVI_BDOT_LOAD_BYTES;
                            kivi_pack_v_lane_bdot(dst, v_block, j_base, lane, count);
                        }
                    }
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_v_layout += MiCo_time() - prof_start;
            }
#endif

            for (size_t i = 0; i < I; i++){
                size_t q_base = idx4(b, h, i, 0, H, I, F);

#ifdef KIVI_PROFILE_INTERNAL
                long prof_start = MiCo_time();
#endif
                for (size_t f = 0; f < F; f++){
                    q_scaled_buf[f] = (q->data[q_base + f] * k_scales[f]) / scale;
                }
                bncfu_fence();
                float q_i8_scale = __FP32toQ8((qbyte *)q_i8, q_scaled_buf, F);
                for (size_t f = F; f < padded_F_bdot; f++){
                    q_i8[f] = 0;
                }
                for (size_t j = 0; j < J; j++){
                    int32_t sum = 0;
                    for (size_t fc = 0; fc < f_bdot_chunks; fc++){
                        const size_t f = fc * KIVI_BDOT_Q8_LANES;
                        const qbyte *lowbit = k_bdot + (j * f_bdot_chunks + fc) * KIVI_BDOT_LOAD_BYTES;
                        bncfu_load(0, q_i8 + f);
                        bncfu_load(1, lowbit);
                        sum += bncfu_bdot2(0, 1);
                    }
                    scores[j] = (float)sum * q_i8_scale;
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_score_accum += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif

                MiCo_softmax_vec(scores, scores, J);
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_softmax += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif

                for (size_t j = 0; j < J; j++){
                    score_scaled_buf[j] = scores[j] * v_scales[j];
                }
                bncfu_fence();

                size_t f = 0;
                float score_i8_scale = __FP32toQ8((qbyte *)score_i8, score_scaled_buf, J);
                for (size_t j = J; j < padded_J_bdot; j++){
                    score_i8[j] = 0;
                }
                for (size_t fb = 0; fb < packed_F; fb++){
                    int32_t o0 = 0;
                    int32_t o1 = 0;
                    int32_t o2 = 0;
                    int32_t o3 = 0;
                    for (size_t jc = 0; jc < j_bdot_chunks; jc++){
                        const size_t j = jc * KIVI_BDOT_Q8_LANES;
                        bncfu_load(0, score_i8 + j);

                        const qbyte *lowbit0 = v_bdot + ((fb * 4 + 0) * j_bdot_chunks + jc) * KIVI_BDOT_LOAD_BYTES;
                        bncfu_load(1, lowbit0);
                        o0 += bncfu_bdot2(0, 1);

                        const qbyte *lowbit1 = v_bdot + ((fb * 4 + 1) * j_bdot_chunks + jc) * KIVI_BDOT_LOAD_BYTES;
                        bncfu_load(1, lowbit1);
                        o1 += bncfu_bdot2(0, 1);

                        const qbyte *lowbit2 = v_bdot + ((fb * 4 + 2) * j_bdot_chunks + jc) * KIVI_BDOT_LOAD_BYTES;
                        bncfu_load(1, lowbit2);
                        o2 += bncfu_bdot2(0, 1);

                        const qbyte *lowbit3 = v_bdot + ((fb * 4 + 3) * j_bdot_chunks + jc) * KIVI_BDOT_LOAD_BYTES;
                        bncfu_load(1, lowbit3);
                        o3 += bncfu_bdot2(0, 1);
                    }
                    output_buf[f++] = (float)o0 * score_i8_scale;
                    if (f < F) output_buf[f++] = (float)o1 * score_i8_scale;
                    if (f < F) output_buf[f++] = (float)o2 * score_i8_scale;
                    if (f < F) output_buf[f++] = (float)o3 * score_i8_scale;
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_output_accum += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif
                for (size_t f = 0; f < F; f++){
                    y->data[idx4(b, i, h, f, I, H, F)] = output_buf[f];
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_output_store += MiCo_time() - prof_start;
#endif
            }
        }
    }

    long total_time = MiCo_time() - start_time;
    ATTN_TIMER += total_time;
#ifdef KIVI_PROFILE_INTERNAL
    printf("KIVI_BNCFU_INTERNAL_PROFILE total=%ld k_gather=%ld k_quant=%ld v_copy=%ld v_quant=%ld v_layout=%ld score_accum=%ld softmax=%ld output_accum=%ld output_store=%ld\n",
           total_time,
           kivi_prof_k_gather,
           kivi_prof_k_quant,
           kivi_prof_v_copy,
           kivi_prof_v_quant,
           kivi_prof_v_layout,
           kivi_prof_score_accum,
           kivi_prof_softmax,
           kivi_prof_output_accum,
           kivi_prof_output_store);
#endif

    free(scores);
    free(output_buf);
    MiCo_free(q_scaled_buf);
    MiCo_free(score_scaled_buf);
    free(k_q);
    free(k_scales);
    free(v_q);
    free(v_scales);
    free(temp_buf);
    free(quant_tmp);
    MiCo_free(q_i8);
    MiCo_free(score_i8);
    MiCo_free(k_bdot);
    MiCo_free(v_bdot);
}

#endif
