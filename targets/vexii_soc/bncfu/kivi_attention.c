#ifdef KIVI_BNCFU_INT8_BDOT

#include "nn.h"
#include "profile.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "bitnet_cfu.h"

#include <math.h>
#include <string.h>
#if defined(KIVI_BNCFU_INT8_VERIFY) || defined(KIVI_PROFILE_INTERNAL)
#include <stdio.h>
#endif

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

static inline size_t idx4(size_t i0, size_t i1, size_t i2, size_t i3, size_t d1, size_t d2, size_t d3){
    return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

static inline int kivi_decode_ternary(uint8_t packed, size_t lane){
    const int bits = (packed >> (2 * lane)) & 0x3;
    return (bits == 1) ? 1 : (bits == 3) ? -1 : 0;
}

static float MiCo_expf(float x){
    long start = MiCo_time();
    float res;
    res = expf(x);
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

static inline void kivi_pack_v_lane_bdot(qbyte *dst, const qbyte *v_block, size_t base_j, size_t lane, size_t count){
    for (size_t k = 0; k < count; k++){
        const uint8_t code = ((uint8_t)v_block[base_j + k] >> (2 * lane)) & 0x3u;
        dst[k >> 2] |= (qbyte)(code << (2 * (k & 3)));
    }
}

static inline void kivi_pack_k_token_bdot(qbyte *dst, const qbyte *k_q, size_t packed_J, size_t base_f, size_t j, size_t count){
    const size_t byte_idx = j / 4;
    const size_t lane = j & 3;
    for (size_t k = 0; k < count; k++){
        const uint8_t code = ((uint8_t)k_q[(base_f + k) * packed_J + byte_idx] >> (2 * lane)) & 0x3u;
        dst[k >> 2] |= (qbyte)(code << (2 * (k & 3)));
    }
}

static inline void kivi_pack_k_contiguous_bdot(qbyte *dst, const qbyte *k_token, size_t base_f, size_t count){
    for (size_t k = 0; k < count; k++){
        const size_t f = base_f + k;
        const uint8_t code = ((uint8_t)k_token[f >> 2] >> (2 * (f & 3))) & 0x3u;
        dst[k >> 2] |= (qbyte)(code << (2 * (k & 3)));
    }
}

static inline void llama_pack_v_channel_bdot(qbyte *dst, const qbyte *v_group, size_t packed_F, size_t base_t, size_t feature, size_t count){
    const size_t byte_idx = feature / 4;
    const size_t lane = feature & 3;
    for (size_t k = 0; k < count; k++){
        const uint8_t code = ((uint8_t)v_group[(base_t + k) * packed_F + byte_idx] >> (2 * lane)) & 0x3u;
        dst[k >> 2] |= (qbyte)(code << (2 * (k & 3)));
    }
}

#ifdef KIVI_BNCFU_INT8_VERIFY
static inline int32_t kivi_bdot2_scalar(const int8_t *q8, const qbyte *q2, size_t count){
    int32_t sum = 0;
    for (size_t i = 0; i < count; i++){
        sum += (int32_t)q8[i] * (int32_t)kivi_decode_ternary((uint8_t)q2[i >> 2], i & 3);
    }
    return sum;
}
#endif

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

    const size_t packed_J = (J + 3) / 4;
    const size_t packed_F = (F + 3) / 4;
    const size_t padded_J = packed_J * 4;
    const size_t padded_F = packed_F * 4;
    const size_t padded_J_bdot = ((J + BNCFU_Q8_ELEMS - 1) / BNCFU_Q8_ELEMS) * BNCFU_Q8_ELEMS;
    const size_t padded_F_bdot = ((F + BNCFU_Q2_FULL_ELEMS - 1) / BNCFU_Q2_FULL_ELEMS) * BNCFU_Q2_FULL_ELEMS;
    const size_t j_bdot_chunks = padded_J_bdot / BNCFU_Q8_ELEMS;
    const size_t f_bdot_chunks = padded_F_bdot / BNCFU_Q2_FULL_ELEMS;

    float *scores = (float *)malloc(J * sizeof(float));
    float *output_buf = (float *)malloc(F * sizeof(float));
    float *q_scaled_buf = (float *)MiCo_alloc(F * sizeof(float), MICO_ALIGN);
    float *score_scaled_buf = (float *)MiCo_alloc(J * sizeof(float), MICO_ALIGN);
    qbyte *k_q = (qbyte *)malloc(
#ifdef KIVI_K_PER_TOKEN
        J * packed_F
#else
        F * packed_J
#endif
        * sizeof(qbyte));
    float *k_scales = (float *)malloc(
#ifdef KIVI_K_PER_TOKEN
        J
#else
        F
#endif
        * sizeof(float));
    qbyte *v_q = (qbyte *)malloc(packed_F * J * sizeof(qbyte));
    float *v_scales = (float *)malloc(J * sizeof(float));
    size_t max_buf = padded_J > padded_F ? padded_J : padded_F;
    float *temp_buf = (float *)malloc(max_buf * sizeof(float));
    qbyte *quant_tmp = (qbyte *)malloc((packed_J > packed_F ? packed_J : packed_F) * sizeof(qbyte));
    int8_t *q_i8 = (int8_t *)MiCo_alloc(padded_F_bdot * sizeof(int8_t), MICO_ALIGN);
    int8_t *score_i8 = (int8_t *)MiCo_alloc(padded_J_bdot * sizeof(int8_t), MICO_ALIGN);
    qbyte *k_bdot = (qbyte *)MiCo_alloc(J * f_bdot_chunks * BNCFU_BYTES * sizeof(qbyte), MICO_ALIGN);
    qbyte *v_bdot = (qbyte *)MiCo_alloc(packed_F * 4 * j_bdot_chunks * BNCFU_BYTES * sizeof(qbyte), MICO_ALIGN);

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
    bncfu_config(BNCFU_QTYPE_15B);

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            size_t k_base = idx4(b, h, 0, 0, H, J, F);
            size_t v_base = idx4(b, h, 0, 0, H, J, F);

#ifdef KIVI_K_PER_TOKEN
            for (size_t j = 0; j < J; j++){
#ifdef KIVI_PROFILE_INTERNAL
                long prof_start = MiCo_time();
#endif
                memcpy(temp_buf, &k->data[k_base + j * F], F * sizeof(float));
                for (size_t f = F; f < padded_F; f++){
                    temp_buf[f] = 0.0f;
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_k_gather += MiCo_time() - prof_start;
                prof_start = MiCo_time();
#endif
                k_scales[j] = __FP32toQ2T(k_q + j * packed_F, temp_buf, F);
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_k_quant += MiCo_time() - prof_start;
#endif
            }
#else
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
#endif

#ifdef KIVI_PROFILE_INTERNAL
            {
                long prof_start = MiCo_time();
#endif
                for (size_t j = 0; j < J; j++){
                    for (size_t fc = 0; fc < f_bdot_chunks; fc++){
                        const size_t f_base = fc * BNCFU_Q2_FULL_ELEMS;
                        const size_t remain = f_base < F ? F - f_base : 0;
                        const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                        qbyte *dst = k_bdot + (j * f_bdot_chunks + fc) * BNCFU_BYTES;
                        memset(dst, 0, BNCFU_BYTES);
#ifdef KIVI_K_PER_TOKEN
                        kivi_pack_k_contiguous_bdot(dst, k_q + j * packed_F, f_base, count);
#else
                        kivi_pack_k_token_bdot(dst, k_q, packed_J, f_base, j, count);
#endif
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
                            const size_t j_base = jc * BNCFU_Q8_ELEMS;
                            const size_t remain = j_base < J ? J - j_base : 0;
                            const size_t count = remain < BNCFU_Q8_ELEMS ? remain : BNCFU_Q8_ELEMS;
                            qbyte *dst = v_bdot + ((fb * 4 + lane) * j_bdot_chunks + jc) * BNCFU_BYTES;
                            memset(dst, 0, BNCFU_BYTES);
                            kivi_pack_v_lane_bdot(dst, v_block, j_base, lane, count);
                        }
                    }
                }
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_v_layout += MiCo_time() - prof_start;
            }
#endif

            bncfu_dma_fence();

            for (size_t i = 0; i < I; i++){
                size_t q_base = idx4(b, h, i, 0, H, I, F);

#ifdef KIVI_PROFILE_INTERNAL
                long prof_start = MiCo_time();
#endif
                for (size_t f = 0; f < F; f++){
#ifdef KIVI_K_PER_TOKEN
                    q_scaled_buf[f] = q->data[q_base + f] / scale;
#else
                    q_scaled_buf[f] = (q->data[q_base + f] * k_scales[f]) / scale;
#endif
                }
                float q_i8_scale = __FP32toQ8((qbyte *)q_i8, q_scaled_buf, F);
                for (size_t f = F; f < padded_F_bdot; f++){
                    q_i8[f] = 0;
                }
                bncfu_dma_fence();
                for (size_t j_base = 0; j_base < J; j_base += BNCFU_REUSE_REGS){
                    const size_t cols = (J - j_base) < BNCFU_REUSE_REGS ? (J - j_base) : BNCFU_REUSE_REGS;
                    int32_t sum[BNCFU_REUSE_REGS] = {0};
                    for (size_t fc = 0; fc < f_bdot_chunks; fc++){
                        const size_t f_base = fc * BNCFU_Q2_FULL_ELEMS;
                        const size_t remain = f_base < F ? F - f_base : 0;
                        const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                        const size_t dots = (count + BNCFU_Q8_ELEMS - 1) / BNCFU_Q8_ELEMS;
                        for (size_t col = 0; col < cols; col++){
                            const qbyte *lowbit = k_bdot + ((j_base + col) * f_bdot_chunks + fc) * BNCFU_BYTES;
                            bncfu_load(col + 1, lowbit);
                        }
                        for (size_t d = 0; d < dots; d++){
                            const size_t f = f_base + d * BNCFU_Q8_ELEMS;
                            bncfu_load(0, q_i8 + f);
                            for (size_t col = 0; col < cols; col++){
                                sum[col] += bncfu_bdot2(0, col + 1);
                            }
                        }
                    }
                    for (size_t col = 0; col < cols; col++){
#ifdef KIVI_BNCFU_INT8_VERIFY
                        int32_t ref_sum = 0;
                        for (size_t ref_fc = 0; ref_fc < f_bdot_chunks; ref_fc++){
                            const size_t f_base = ref_fc * BNCFU_Q2_FULL_ELEMS;
                            const size_t remain = f_base < F ? F - f_base : 0;
                            const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                            const qbyte *lowbit = k_bdot + ((j_base + col) * f_bdot_chunks + ref_fc) * BNCFU_BYTES;
                            ref_sum += kivi_bdot2_scalar(q_i8 + f_base, lowbit, count);
                        }
                        if (sum[col] != ref_sum){
                            printf("KIVI_SCORE_BDOT_MISMATCH i=%d j=%d sum=%d ref=%d first_q=%d first_w=%d\n",
                                   (int)i, (int)(j_base + col), (int)sum[col], (int)ref_sum,
                                   (int)q_i8[0], (int)((uint8_t *)k_bdot)[((j_base + col) * f_bdot_chunks) * BNCFU_BYTES]);
                            MiCo_assert(0, "[KIVI BNCFU Attention] score BDOT mismatch");
                        }
#endif
#ifdef KIVI_K_PER_TOKEN
                        scores[j_base + col] = (float)sum[col] * q_i8_scale * k_scales[j_base + col];
#else
                        scores[j_base + col] = (float)sum[col] * q_i8_scale;
#endif
                    }
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

                size_t f = 0;
                float score_i8_scale = __FP32toQ8((qbyte *)score_i8, score_scaled_buf, J);
                for (size_t j = J; j < padded_J_bdot; j++){
                    score_i8[j] = 0;
                }
                bncfu_dma_fence();
                for (size_t fb = 0; fb < packed_F; fb++){
                    int32_t o0 = 0;
                    int32_t o1 = 0;
                    int32_t o2 = 0;
                    int32_t o3 = 0;
                    for (size_t jc = 0; jc < j_bdot_chunks; jc++){
                        const size_t j = jc * BNCFU_Q8_ELEMS;
                        bncfu_load(0, score_i8 + j);

                        const qbyte *lowbit0 = v_bdot + ((fb * 4 + 0) * j_bdot_chunks + jc) * BNCFU_BYTES;
                        bncfu_load(1, lowbit0);
                        o0 += bncfu_bdot2(0, 1);

                        const qbyte *lowbit1 = v_bdot + ((fb * 4 + 1) * j_bdot_chunks + jc) * BNCFU_BYTES;
                        bncfu_load(1, lowbit1);
                        o1 += bncfu_bdot2(0, 1);

                        const qbyte *lowbit2 = v_bdot + ((fb * 4 + 2) * j_bdot_chunks + jc) * BNCFU_BYTES;
                        bncfu_load(1, lowbit2);
                        o2 += bncfu_bdot2(0, 1);

                        const qbyte *lowbit3 = v_bdot + ((fb * 4 + 3) * j_bdot_chunks + jc) * BNCFU_BYTES;
                        bncfu_load(1, lowbit3);
                        o3 += bncfu_bdot2(0, 1);
                    }
#ifdef KIVI_BNCFU_INT8_VERIFY
                    int32_t ref_o0 = 0;
                    int32_t ref_o1 = 0;
                    int32_t ref_o2 = 0;
                    int32_t ref_o3 = 0;
                    for (size_t jc = 0; jc < j_bdot_chunks; jc++){
                        const size_t j_base = jc * BNCFU_Q8_ELEMS;
                        const size_t remain = j_base < J ? J - j_base : 0;
                        const size_t count = remain < BNCFU_Q8_ELEMS ? remain : BNCFU_Q8_ELEMS;
                        ref_o0 += kivi_bdot2_scalar(score_i8 + j_base, v_bdot + ((fb * 4 + 0) * j_bdot_chunks + jc) * BNCFU_BYTES, count);
                        ref_o1 += kivi_bdot2_scalar(score_i8 + j_base, v_bdot + ((fb * 4 + 1) * j_bdot_chunks + jc) * BNCFU_BYTES, count);
                        ref_o2 += kivi_bdot2_scalar(score_i8 + j_base, v_bdot + ((fb * 4 + 2) * j_bdot_chunks + jc) * BNCFU_BYTES, count);
                        ref_o3 += kivi_bdot2_scalar(score_i8 + j_base, v_bdot + ((fb * 4 + 3) * j_bdot_chunks + jc) * BNCFU_BYTES, count);
                    }
                    MiCo_assert(o0 == ref_o0 && o1 == ref_o1 && o2 == ref_o2 && o3 == ref_o3,
                                "[KIVI BNCFU Attention] output BDOT mismatch");
#endif
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

#ifndef KIVI_BNCFU_LLAMA_DISABLE
void MiCo_llama_pack_kv_group_q2t_bncfu(
    const float* key_cache,
    const float* value_cache,
    qbyte* key_cache_q2t,
    qbyte* value_cache_q2t,
    float* key_scales,
    float* value_scales,
    qbyte* key_cache_bdot,
    qbyte* value_cache_bdot,
    const int group_id,
    const MiCo_MHA_Config* cfg
) {
    MiCo_llama_pack_kv_group_q2t(
        key_cache,
        value_cache,
        key_cache_q2t,
        value_cache_q2t,
        key_scales,
        value_scales,
        group_id,
        cfg
    );

    const int head_size = cfg->head_size;
    const int n_kv_heads = cfg->n_heads / cfg->kv_mul;
    const int group_size = MICO_LLAMA_KV_GROUP_SIZE;
    const size_t packed_group_bytes = MICO_LLAMA_KV_PACKED_GROUP_BYTES(head_size);
    const size_t packed_group_tokens = ((size_t)group_size + 3) / 4;
    const size_t packed_head_size = ((size_t)head_size + 3) / 4;
    const size_t padded_head_bdot = (((size_t)head_size + BNCFU_Q2_FULL_ELEMS - 1) / BNCFU_Q2_FULL_ELEMS) * BNCFU_Q2_FULL_ELEMS;
    const size_t padded_group_bdot = (((size_t)group_size + BNCFU_Q8_ELEMS - 1) / BNCFU_Q8_ELEMS) * BNCFU_Q8_ELEMS;
    const size_t head_bdot_chunks = padded_head_bdot / BNCFU_Q2_FULL_ELEMS;
    const size_t group_bdot_chunks = padded_group_bdot / BNCFU_Q8_ELEMS;
    const size_t k_group_bdot_bytes = MICO_LLAMA_KV_BNCFU_K_GROUP_BYTES(head_size);
    const size_t v_group_bdot_bytes = MICO_LLAMA_KV_BNCFU_V_GROUP_BYTES(head_size);

    MiCo_assert(key_cache_bdot != NULL && value_cache_bdot != NULL,
                "[LLaMa KIVI BNCFU Pack] bdot cache is NULL");

    for (int kv_head = 0; kv_head < n_kv_heads; kv_head++) {
        const size_t group_head_idx = (size_t)group_id * (size_t)n_kv_heads + (size_t)kv_head;
        const qbyte *k_group = key_cache_q2t + group_head_idx * packed_group_bytes;
        const qbyte *v_group = value_cache_q2t + group_head_idx * packed_group_bytes;
        qbyte *k_bdot_group = key_cache_bdot + group_head_idx * k_group_bdot_bytes;
        qbyte *v_bdot_group = value_cache_bdot + group_head_idx * v_group_bdot_bytes;

        for (int local_t = 0; local_t < group_size; local_t++) {
            for (size_t hc = 0; hc < head_bdot_chunks; hc++) {
                const size_t f_base = hc * BNCFU_Q2_FULL_ELEMS;
                const size_t remain = f_base < (size_t)head_size ? (size_t)head_size - f_base : 0;
                const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                qbyte *dst = k_bdot_group + ((size_t)local_t * head_bdot_chunks + hc) * BNCFU_BYTES;
                memset(dst, 0, BNCFU_BYTES);
                kivi_pack_k_token_bdot(dst, k_group, packed_group_tokens, f_base, (size_t)local_t, count);
            }
        }

        for (size_t fb = 0; fb < packed_head_size; fb++) {
            for (size_t lane = 0; lane < 4; lane++) {
                for (size_t gc = 0; gc < group_bdot_chunks; gc++) {
                    const size_t local_base = gc * BNCFU_Q8_ELEMS;
                    const size_t remain = local_base < (size_t)group_size ? (size_t)group_size - local_base : 0;
                    const size_t count = remain < BNCFU_Q8_ELEMS ? remain : BNCFU_Q8_ELEMS;
                    qbyte *dst = v_bdot_group + ((fb * 4 + lane) * group_bdot_chunks + gc) * BNCFU_BYTES;
                    memset(dst, 0, BNCFU_BYTES);
                    llama_pack_v_channel_bdot(dst, v_group, packed_head_size, local_base, fb * 4 + lane, count);
                }
            }
        }
    }
    bncfu_dma_fence();
}

void MiCo_llama_kivi_attention_f32_bncfu(
    Tensor2D_F32* output,
    const Tensor2D_F32* query,
    const float* key_cache,
    const float* value_cache,
    const qbyte* key_cache_q2t,
    const qbyte* value_cache_q2t,
    const float* key_scales,
    const float* value_scales,
    const qbyte* key_cache_bdot,
    const qbyte* value_cache_bdot,
    float* att_buffer,
    const int pos,
    const MiCo_MHA_Config* cfg
) {
    const int n_heads = cfg->n_heads;
    const int head_size = cfg->head_size;
    const int kv_dim = cfg->kv_dim;
    const int kv_mul = cfg->kv_mul;
    const int seq_len = cfg->seq_len;
    const int n_kv_heads = n_heads / kv_mul;
    const int group_size = MICO_LLAMA_KV_GROUP_SIZE;
    const int current_group = pos / group_size;
    const int current_group_start = current_group * group_size;
    const float attn_scale = 1.0f / sqrtf((float)head_size);
    const size_t packed_group_bytes = MICO_LLAMA_KV_PACKED_GROUP_BYTES(head_size);
    const size_t packed_head_size = ((size_t)head_size + 3) / 4;
    const size_t padded_head_bdot = (((size_t)head_size + BNCFU_Q2_FULL_ELEMS - 1) / BNCFU_Q2_FULL_ELEMS) * BNCFU_Q2_FULL_ELEMS;
    const size_t padded_group_bdot = (((size_t)group_size + BNCFU_Q8_ELEMS - 1) / BNCFU_Q8_ELEMS) * BNCFU_Q8_ELEMS;
    const size_t head_bdot_chunks = padded_head_bdot / BNCFU_Q2_FULL_ELEMS;
    const size_t group_bdot_chunks = padded_group_bdot / BNCFU_Q8_ELEMS;
    const size_t k_group_bdot_bytes = MICO_LLAMA_KV_BNCFU_K_GROUP_BYTES(head_size);
    const size_t v_group_bdot_bytes = MICO_LLAMA_KV_BNCFU_V_GROUP_BYTES(head_size);

    MiCo_assert(pos >= 0 && pos < seq_len, "[LLaMa KIVI BNCFU Attention] pos out of range");
    MiCo_assert(output != NULL && query != NULL, "[LLaMa KIVI BNCFU Attention] tensor is NULL");
    MiCo_assert(key_cache != NULL && value_cache != NULL, "[LLaMa KIVI BNCFU Attention] FP32 cache is NULL");
    MiCo_assert(key_cache_q2t != NULL && value_cache_q2t != NULL, "[LLaMa KIVI BNCFU Attention] q2t cache is NULL");
    MiCo_assert(key_scales != NULL && value_scales != NULL, "[LLaMa KIVI BNCFU Attention] scale buffer is NULL");
    MiCo_assert(key_cache_bdot != NULL && value_cache_bdot != NULL, "[LLaMa KIVI BNCFU Attention] bdot cache is NULL");
    MiCo_assert(att_buffer != NULL, "[LLaMa KIVI BNCFU Attention] attention buffer is NULL");
    MiCo_assert(kv_mul > 0 && n_kv_heads > 0, "[LLaMa KIVI BNCFU Attention] invalid GQA config");

    int8_t *q_int8 = (int8_t *)MiCo_alloc(padded_head_bdot * sizeof(int8_t), MICO_ALIGN);
    int8_t *qk_int8 = (int8_t *)MiCo_alloc(padded_head_bdot * sizeof(int8_t), MICO_ALIGN);
    int8_t *score_v_int8 = (int8_t *)MiCo_alloc(padded_group_bdot * sizeof(int8_t), MICO_ALIGN);
    qbyte *score_int8 = (qbyte *)malloc((size_t)seq_len * sizeof(qbyte));
    float *qk_scaled = (float *)malloc((size_t)head_size * sizeof(float));
    float *score_v_scaled = (float *)malloc((size_t)group_size * sizeof(float));

    MiCo_assert(q_int8 != NULL && qk_int8 != NULL && score_v_int8 != NULL &&
                score_int8 != NULL && qk_scaled != NULL && score_v_scaled != NULL,
                "[LLaMa KIVI BNCFU Attention] failed to allocate quant buffers");

    unsigned long start_time = (unsigned long)MiCo_time();
#ifdef KIVI_PROFILE_INTERNAL
    unsigned long prof_q_quant = 0;
    unsigned long prof_hist_score = 0;
    unsigned long prof_float_score = 0;
    unsigned long prof_softmax = 0;
    unsigned long prof_hist_output = 0;
    unsigned long prof_float_output = 0;
#endif

    bncfu_enable();
    bncfu_config(BNCFU_QTYPE_15B);

    for (int h = 0; h < n_heads; h++) {
        const int kv_head = h / kv_mul;
        const float *q = query->data + (size_t)h * (size_t)head_size;
        float *att = att_buffer + (size_t)h * (size_t)seq_len;
        float *out = output->data + (size_t)h * (size_t)head_size;

#ifdef KIVI_PROFILE_INTERNAL
        unsigned long prof_start = (unsigned long)MiCo_time();
#endif
        const float q_scale = __FP32toQ8((qbyte *)q_int8, (float *)q, (size_t)head_size);
        for (size_t f = (size_t)head_size; f < padded_head_bdot; f++){
            q_int8[f] = 0;
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_q_quant += (unsigned long)MiCo_time() - prof_start;
        prof_start = (unsigned long)MiCo_time();
#endif

        for (int group = 0; group < current_group; group++) {
            const size_t group_head_idx = (size_t)group * (size_t)n_kv_heads + (size_t)kv_head;
            const float *k_scale_group = key_scales + group_head_idx * (size_t)head_size;
            const qbyte *k_bdot_group = key_cache_bdot + group_head_idx * k_group_bdot_bytes;

            for (int f = 0; f < head_size; f++) {
                qk_scaled[f] = (float)q_int8[f] * k_scale_group[f];
            }
            const float qk_scale = __FP32toQ8((qbyte *)qk_int8, qk_scaled, (size_t)head_size);
            for (size_t f = (size_t)head_size; f < padded_head_bdot; f++){
                qk_int8[f] = 0;
            }

            for (int local_base = 0; local_base < group_size; local_base += BNCFU_REUSE_REGS) {
                const int cols = (group_size - local_base) < BNCFU_REUSE_REGS ? (group_size - local_base) : BNCFU_REUSE_REGS;
                int32_t sum[BNCFU_REUSE_REGS] = {0};

                for (size_t hc = 0; hc < head_bdot_chunks; hc++) {
                    const size_t f_base = hc * BNCFU_Q2_FULL_ELEMS;
                    const size_t remain = f_base < (size_t)head_size ? (size_t)head_size - f_base : 0;
                    const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                    const size_t dots = (count + BNCFU_Q8_ELEMS - 1) / BNCFU_Q8_ELEMS;

                    for (int col = 0; col < cols; col++) {
                        const qbyte *lowbit = k_bdot_group + ((size_t)(local_base + col) * head_bdot_chunks + hc) * BNCFU_BYTES;
                        bncfu_load((unsigned int)col + 1, lowbit);
                    }
                    for (size_t d = 0; d < dots; d++) {
                        const size_t f = f_base + d * BNCFU_Q8_ELEMS;
                        bncfu_load(0, qk_int8 + f);
                        for (int col = 0; col < cols; col++) {
                            sum[col] += bncfu_bdot2(0, (unsigned int)col + 1);
                        }
                    }
                }

                for (int col = 0; col < cols; col++) {
#ifdef KIVI_BNCFU_INT8_VERIFY
                    int32_t ref_sum = 0;
                    for (size_t hc = 0; hc < head_bdot_chunks; hc++) {
                        const size_t f_base = hc * BNCFU_Q2_FULL_ELEMS;
                        const size_t remain = f_base < (size_t)head_size ? (size_t)head_size - f_base : 0;
                        const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                        const qbyte *lowbit = k_bdot_group + ((size_t)(local_base + col) * head_bdot_chunks + hc) * BNCFU_BYTES;
                        ref_sum += kivi_bdot2_scalar(qk_int8 + f_base, lowbit, count);
                    }
                    if (sum[col] != ref_sum) {
                        printf("LLAMA_KIVI_SCORE_BDOT_MISMATCH h=%d group=%d local=%d sum=%d ref=%d first_q=%d first_w=%d\n",
                               h, group, local_base + col, (int)sum[col], (int)ref_sum,
                               (int)qk_int8[0], (int)((const uint8_t *)k_bdot_group)[0]);
                        MiCo_assert(0, "[LLaMa KIVI BNCFU Attention] score BDOT mismatch");
                    }
#endif
                    const int token = group * group_size + local_base + col;
                    att[token] = (float)sum[col] * q_scale * qk_scale * attn_scale;
                }
            }
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_hist_score += (unsigned long)MiCo_time() - prof_start;
        prof_start = (unsigned long)MiCo_time();
#endif

        for (int token = current_group_start; token <= pos; token++) {
            const int local_t = token - current_group_start;
            const float *k = key_cache + (size_t)local_t * (size_t)kv_dim + (size_t)kv_head * (size_t)head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            att[token] = score * attn_scale;
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_float_score += (unsigned long)MiCo_time() - prof_start;
        prof_start = (unsigned long)MiCo_time();
#endif

        softmax(att, pos + 1);
        const float score_scale = __FP32toQ8(score_int8, att, (size_t)pos + 1);
#ifdef KIVI_PROFILE_INTERNAL
        prof_softmax += (unsigned long)MiCo_time() - prof_start;
        prof_start = (unsigned long)MiCo_time();
#endif

        for (int i = 0; i < head_size; i++) {
            out[i] = 0.0f;
        }

        for (int group = 0; group < current_group; group++) {
            const size_t group_head_idx = (size_t)group * (size_t)n_kv_heads + (size_t)kv_head;
            const float *v_scale_group = value_scales + group_head_idx * (size_t)group_size;
            const qbyte *v_bdot_group = value_cache_bdot + group_head_idx * v_group_bdot_bytes;

            for (int local_t = 0; local_t < group_size; local_t++) {
                const int token = group * group_size + local_t;
                score_v_scaled[local_t] = (float)score_int8[token] * v_scale_group[local_t];
            }
            const float score_v_scale = __FP32toQ8((qbyte *)score_v_int8, score_v_scaled, (size_t)group_size);
            for (size_t t = (size_t)group_size; t < padded_group_bdot; t++){
                score_v_int8[t] = 0;
            }

            for (size_t fb = 0; fb < packed_head_size; fb++) {
                int32_t o0 = 0;
                int32_t o1 = 0;
                int32_t o2 = 0;
                int32_t o3 = 0;
                for (size_t gc = 0; gc < group_bdot_chunks; gc++) {
                    const size_t local_base = gc * BNCFU_Q8_ELEMS;
                    bncfu_load(0, score_v_int8 + local_base);

                    const qbyte *lowbit0 = v_bdot_group + ((fb * 4 + 0) * group_bdot_chunks + gc) * BNCFU_BYTES;
                    bncfu_load(1, lowbit0);
                    o0 += bncfu_bdot2(0, 1);

                    const qbyte *lowbit1 = v_bdot_group + ((fb * 4 + 1) * group_bdot_chunks + gc) * BNCFU_BYTES;
                    bncfu_load(1, lowbit1);
                    o1 += bncfu_bdot2(0, 1);

                    const qbyte *lowbit2 = v_bdot_group + ((fb * 4 + 2) * group_bdot_chunks + gc) * BNCFU_BYTES;
                    bncfu_load(1, lowbit2);
                    o2 += bncfu_bdot2(0, 1);

                    const qbyte *lowbit3 = v_bdot_group + ((fb * 4 + 3) * group_bdot_chunks + gc) * BNCFU_BYTES;
                    bncfu_load(1, lowbit3);
                    o3 += bncfu_bdot2(0, 1);
                }
#ifdef KIVI_BNCFU_INT8_VERIFY
                int32_t ref_o0 = 0;
                int32_t ref_o1 = 0;
                int32_t ref_o2 = 0;
                int32_t ref_o3 = 0;
                for (size_t gc = 0; gc < group_bdot_chunks; gc++) {
                    const size_t local_base = gc * BNCFU_Q8_ELEMS;
                    const size_t remain = local_base < (size_t)group_size ? (size_t)group_size - local_base : 0;
                    const size_t count = remain < BNCFU_Q8_ELEMS ? remain : BNCFU_Q8_ELEMS;
                    ref_o0 += kivi_bdot2_scalar(score_v_int8 + local_base,
                                                 v_bdot_group + ((fb * 4 + 0) * group_bdot_chunks + gc) * BNCFU_BYTES,
                                                 count);
                    ref_o1 += kivi_bdot2_scalar(score_v_int8 + local_base,
                                                 v_bdot_group + ((fb * 4 + 1) * group_bdot_chunks + gc) * BNCFU_BYTES,
                                                 count);
                    ref_o2 += kivi_bdot2_scalar(score_v_int8 + local_base,
                                                 v_bdot_group + ((fb * 4 + 2) * group_bdot_chunks + gc) * BNCFU_BYTES,
                                                 count);
                    ref_o3 += kivi_bdot2_scalar(score_v_int8 + local_base,
                                                 v_bdot_group + ((fb * 4 + 3) * group_bdot_chunks + gc) * BNCFU_BYTES,
                                                 count);
                }
                MiCo_assert(o0 == ref_o0 && o1 == ref_o1 && o2 == ref_o2 && o3 == ref_o3,
                            "[LLaMa KIVI BNCFU Attention] output BDOT mismatch");
#endif
                const size_t feature0 = fb * 4 + 0;
                const size_t feature1 = fb * 4 + 1;
                const size_t feature2 = fb * 4 + 2;
                const size_t feature3 = fb * 4 + 3;
                if (feature0 < (size_t)head_size) out[feature0] += (float)o0 * score_scale * score_v_scale;
                if (feature1 < (size_t)head_size) out[feature1] += (float)o1 * score_scale * score_v_scale;
                if (feature2 < (size_t)head_size) out[feature2] += (float)o2 * score_scale * score_v_scale;
                if (feature3 < (size_t)head_size) out[feature3] += (float)o3 * score_scale * score_v_scale;
            }
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_hist_output += (unsigned long)MiCo_time() - prof_start;
        prof_start = (unsigned long)MiCo_time();
#endif

        for (int token = current_group_start; token <= pos; token++) {
            const int local_t = token - current_group_start;
            const float *v = value_cache + (size_t)local_t * (size_t)kv_dim + (size_t)kv_head * (size_t)head_size;
            const float a = (float)score_int8[token] * score_scale;
            for (int i = 0; i < head_size; i++) {
                out[i] += a * v[i];
            }
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_float_output += (unsigned long)MiCo_time() - prof_start;
#endif
    }

    unsigned long total_time = (unsigned long)MiCo_time() - start_time;
    ATTN_TIMER += (long)total_time;
#ifdef KIVI_PROFILE_INTERNAL
    printf("LLAMA_KIVI_BNCFU_PROFILE total=%lu q_quant=%lu hist_score=%lu float_score=%lu softmax=%lu hist_output=%lu float_output=%lu group_size=%d\n",
           total_time,
           prof_q_quant,
           prof_hist_score,
           prof_float_score,
           prof_softmax,
           prof_hist_output,
           prof_float_output,
           group_size);
#endif

    MiCo_free(q_int8);
    MiCo_free(qk_int8);
    MiCo_free(score_v_int8);
    free(score_int8);
    free(qk_scaled);
    free(score_v_scaled);
}

void MiCo_llama_kivi_attention_f32(
    Tensor2D_F32* output,
    const Tensor2D_F32* query,
    const float* key_cache,
    const float* value_cache,
    const qbyte* key_cache_q2t,
    const qbyte* value_cache_q2t,
    const float* key_scales,
    const float* value_scales,
    float* att_buffer,
    const int pos,
    const MiCo_MHA_Config* cfg
) {
    const int head_size = cfg->head_size;
    const int n_kv_heads = cfg->n_heads / cfg->kv_mul;
    const int n_groups = (cfg->seq_len + MICO_LLAMA_KV_GROUP_SIZE - 1) / MICO_LLAMA_KV_GROUP_SIZE;
    const size_t k_bytes = (size_t)n_groups * (size_t)n_kv_heads * MICO_LLAMA_KV_BNCFU_K_GROUP_BYTES(head_size);
    const size_t v_bytes = (size_t)n_groups * (size_t)n_kv_heads * MICO_LLAMA_KV_BNCFU_V_GROUP_BYTES(head_size);
    qbyte *key_bdot = (qbyte *)MiCo_alloc(k_bytes, MICO_ALIGN);
    qbyte *value_bdot = (qbyte *)MiCo_alloc(v_bytes, MICO_ALIGN);
    MiCo_assert(key_bdot != NULL && value_bdot != NULL,
                "[LLaMa KIVI BNCFU Attention] failed to allocate compatibility bdot cache");
    memset(key_bdot, 0, k_bytes);
    memset(value_bdot, 0, v_bytes);

    for (int group = 0; group <= pos / MICO_LLAMA_KV_GROUP_SIZE; group++) {
        const int group_base = group * MICO_LLAMA_KV_GROUP_SIZE;
        if (group_base >= cfg->seq_len) break;
        for (int kv_head = 0; kv_head < n_kv_heads; kv_head++) {
            const size_t group_head_idx = (size_t)group * (size_t)n_kv_heads + (size_t)kv_head;
            const qbyte *k_group = key_cache_q2t + group_head_idx * MICO_LLAMA_KV_PACKED_GROUP_BYTES(head_size);
            const qbyte *v_group = value_cache_q2t + group_head_idx * MICO_LLAMA_KV_PACKED_GROUP_BYTES(head_size);
            qbyte *k_bdot_group = key_bdot + group_head_idx * MICO_LLAMA_KV_BNCFU_K_GROUP_BYTES(head_size);
            qbyte *v_bdot_group = value_bdot + group_head_idx * MICO_LLAMA_KV_BNCFU_V_GROUP_BYTES(head_size);
            const size_t packed_group_tokens = ((size_t)MICO_LLAMA_KV_GROUP_SIZE + 3) / 4;
            const size_t packed_head_size = ((size_t)head_size + 3) / 4;
            const size_t head_chunks = MICO_LLAMA_KV_BNCFU_HEAD_CHUNKS(head_size);
            const size_t group_chunks = MICO_LLAMA_KV_BNCFU_GROUP_CHUNKS;

            for (int local_t = 0; local_t < MICO_LLAMA_KV_GROUP_SIZE; local_t++) {
                for (size_t hc = 0; hc < head_chunks; hc++) {
                    const size_t f_base = hc * BNCFU_Q2_FULL_ELEMS;
                    const size_t remain = f_base < (size_t)head_size ? (size_t)head_size - f_base : 0;
                    const size_t count = remain < BNCFU_Q2_FULL_ELEMS ? remain : BNCFU_Q2_FULL_ELEMS;
                    qbyte *dst = k_bdot_group + ((size_t)local_t * head_chunks + hc) * BNCFU_BYTES;
                    memset(dst, 0, BNCFU_BYTES);
                    kivi_pack_k_token_bdot(dst, k_group, packed_group_tokens, f_base, (size_t)local_t, count);
                }
            }
            for (size_t fb = 0; fb < packed_head_size; fb++) {
                for (size_t lane = 0; lane < 4; lane++) {
                    for (size_t gc = 0; gc < group_chunks; gc++) {
                        const size_t local_base = gc * BNCFU_Q8_ELEMS;
                        const size_t remain = local_base < MICO_LLAMA_KV_GROUP_SIZE ? MICO_LLAMA_KV_GROUP_SIZE - local_base : 0;
                        const size_t count = remain < BNCFU_Q8_ELEMS ? remain : BNCFU_Q8_ELEMS;
                        qbyte *dst = v_bdot_group + ((fb * 4 + lane) * group_chunks + gc) * BNCFU_BYTES;
                        memset(dst, 0, BNCFU_BYTES);
                        llama_pack_v_channel_bdot(dst, v_group, packed_head_size, local_base, fb * 4 + lane, count);
                    }
                }
            }
        }
    }
    bncfu_dma_fence();
    MiCo_llama_kivi_attention_f32_bncfu(
        output, query, key_cache, value_cache, key_cache_q2t, value_cache_q2t,
        key_scales, value_scales, key_bdot, value_bdot, att_buffer, pos, cfg);
    MiCo_free(key_bdot);
    MiCo_free(value_bdot);
}
#endif

#endif
