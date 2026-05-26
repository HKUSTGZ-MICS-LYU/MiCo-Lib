#include "nn.h"
#include "profile.h"
#include "mico_qnn.h"
#include "mico_quant.h"

#include <math.h>
#include <string.h>

#if defined(KIVI_PROFILE_INTERNAL) && !defined(RISCV_VEXII)
#include <stdio.h>
#endif

extern long ATTN_TIMER;
extern long SOFTMAX_TIMER;

extern long EXPF_TIMER;

static inline size_t idx2(size_t i0, size_t i1, size_t d1){
    return i0 * d1 + i1;
}

static inline size_t idx3(size_t i0, size_t i1, size_t i2, size_t d1, size_t d2){
    return (i0 * d1 + i1) * d2 + i2;
}

static inline size_t idx4(size_t i0, size_t i1, size_t i2, size_t i3, size_t d1, size_t d2, size_t d3){
    return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

static inline int kivi_decode_ternary(uint8_t packed, size_t lane){
    const int bits = (packed >> (2 * lane)) & 0x3;
    return (bits == 1) ? 1 : (bits == 3) ? -1 : 0;
}

static void MiCo_init_exp_lut(void);
static float MiCo_expf(float x);

static inline float kivi_linear_phi(float x){
    return x >= 0.0f ? x + 1.0f : MiCo_expf(x);
}

static inline int32_t kivi_dot_q8_q2t(const qbyte *x_q8, const qbyte *w_q2t, size_t n){
    int32_t acc = 0;
    size_t i = 0;
    const size_t full_bytes = n / 4;
    for (size_t byte_idx = 0; byte_idx < full_bytes; byte_idx++){
        const uint8_t packed = (uint8_t)w_q2t[byte_idx];
        acc += (int32_t)x_q8[i + 0] * kivi_decode_ternary(packed, 0);
        acc += (int32_t)x_q8[i + 1] * kivi_decode_ternary(packed, 1);
        acc += (int32_t)x_q8[i + 2] * kivi_decode_ternary(packed, 2);
        acc += (int32_t)x_q8[i + 3] * kivi_decode_ternary(packed, 3);
        i += 4;
    }
    if (i < n){
        const uint8_t packed = (uint8_t)w_q2t[full_bytes];
        for (size_t lane = 0; i < n; lane++, i++){
            acc += (int32_t)x_q8[i] * kivi_decode_ternary(packed, lane);
        }
    }
    return acc;
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

static inline void kivi_accum_output_from_v(float *out, float score_scaled, const qbyte *v_packed, size_t F){
    if (score_scaled == 0.0f) return;

    size_t f = 0;
    const size_t full_bytes = F / 4;
    for (size_t byte_idx = 0; byte_idx < full_bytes; byte_idx++){
        const uint8_t packed = (uint8_t)v_packed[byte_idx];
        int v0 = kivi_decode_ternary(packed, 0);
        int v1 = kivi_decode_ternary(packed, 1);
        int v2 = kivi_decode_ternary(packed, 2);
        int v3 = kivi_decode_ternary(packed, 3);
        if (v0 > 0) out[f + 0] += score_scaled;
        else if (v0 < 0) out[f + 0] -= score_scaled;
        if (v1 > 0) out[f + 1] += score_scaled;
        else if (v1 < 0) out[f + 1] -= score_scaled;
        if (v2 > 0) out[f + 2] += score_scaled;
        else if (v2 < 0) out[f + 2] -= score_scaled;
        if (v3 > 0) out[f + 3] += score_scaled;
        else if (v3 < 0) out[f + 3] -= score_scaled;
        f += 4;
    }

    if (f < F){
        const uint8_t packed = (uint8_t)v_packed[full_bytes];
        for (size_t lane = 0; f < F; lane++, f++){
            int v = kivi_decode_ternary(packed, lane);
            if (v > 0) out[f] += score_scaled;
            else if (v < 0) out[f] -= score_scaled;
        }
    }
}

#ifdef KIVI_V_LAYOUT_OPT
static inline float kivi_apply_ternary(float x, uint8_t packed, size_t lane){
    const int bits = (packed >> (2 * lane)) & 0x3;
    return (bits == 1) ? x : (bits == 3) ? -x : 0.0f;
}
#endif

// Exp LUT: covers [-EXP_LUT_MAX, 0] with EXP_LUT_SIZE entries
#define EXP_LUT_SIZE 256
#define EXP_LUT_MAX  16.0f
#define EXP_LUT_STEP (EXP_LUT_MAX / (float)EXP_LUT_SIZE)

static float exp_lut[EXP_LUT_SIZE];
static int exp_lut_ready = 0;

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
        if (src[i] > max_val){
            max_val = src[i];
        }
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

/*
    KIVI-style Linear Attention prototype:

    Phase 1 uses the requested INT8 x INT2 path on the token reduction:
        context[d, m] = dot(phi(K[:, d])_q8, V[:, m]_q2t)

    Phase 2 intentionally keeps phi(Q) and context in FP32 for this first
    version, so the error is isolated to the K/V low-bit context path.
*/
__attribute__((weak)) void MiCo_kivi_linear_attention(
    Tensor4D_F32 *y,
    const Tensor4D_F32 *q,
    const Tensor4D_F32 *k,
    const Tensor4D_F32 *v,
    const float eps
){
    const size_t B = q->shape[0];
    const size_t H = q->shape[1];
    const size_t N = q->shape[2];
    const size_t D = q->shape[3];
    const size_t M = v->shape[3];

    MiCo_assert(k->shape[0] == B && k->shape[1] == H && k->shape[2] == N && k->shape[3] == D,
                "[KIVI LinearAttention] k shape mismatch");
    MiCo_assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == N,
                "[KIVI LinearAttention] v shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == N && y->shape[2] == H && y->shape[3] == M,
                "[KIVI LinearAttention] y shape mismatch");

    MiCo_init_exp_lut();

    const size_t packed_N = (N + 3) / 4;
    const size_t padded_N = packed_N * 4;

    long start_time = MiCo_time();

    float *phi_k_col = (float *)malloc(N * sizeof(float));
    qbyte *phi_k_q8 = (qbyte *)malloc(D * N * sizeof(qbyte));
    float *phi_k_scales = (float *)malloc(D * sizeof(float));
    qbyte *v_q2t = (qbyte *)malloc(M * packed_N * sizeof(qbyte));
    float *v_scales = (float *)malloc(M * sizeof(float));
    float *v_col = (float *)malloc(padded_N * sizeof(float));
    float *context = (float *)malloc(H * D * M * sizeof(float));
    float *k_sum = (float *)malloc(H * D * sizeof(float));
    float *phi_q = (float *)malloc(N * D * sizeof(float));
    float *num = (float *)malloc(M * sizeof(float));

    MiCo_assert(phi_k_col != NULL && phi_k_q8 != NULL && phi_k_scales != NULL &&
                v_q2t != NULL && v_scales != NULL && v_col != NULL &&
                context != NULL && k_sum != NULL && phi_q != NULL && num != NULL,
                "[KIVI LinearAttention] failed to allocate buffers");

    for (size_t b = 0; b < B; b++){
        memset(context, 0, H * D * M * sizeof(float));
        memset(k_sum, 0, H * D * sizeof(float));

        for (size_t h = 0; h < H; h++){
            const size_t k_base = idx4(b, h, 0, 0, H, N, D);
            const size_t v_base = idx4(b, h, 0, 0, H, N, M);

            for (size_t d = 0; d < D; d++){
                float sum = 0.0f;
                for (size_t n = 0; n < N; n++){
                    float kp = kivi_linear_phi(k->data[k_base + n * D + d]);
                    phi_k_col[n] = kp;
                    sum += kp;
                }
                k_sum[idx2(h, d, D)] = sum;
                phi_k_scales[d] = __FP32toQ8(phi_k_q8 + d * N, phi_k_col, N);
            }

            for (size_t m = 0; m < M; m++){
                for (size_t n = 0; n < N; n++){
                    v_col[n] = v->data[v_base + n * M + m];
                }
                for (size_t n = N; n < padded_N; n++){
                    v_col[n] = 0.0f;
                }
                v_scales[m] = __FP32toQ2T(v_q2t + m * packed_N, v_col, N);
            }

            for (size_t d = 0; d < D; d++){
                const qbyte *kq = phi_k_q8 + d * N;
                const float k_scale = phi_k_scales[d];
                for (size_t m = 0; m < M; m++){
                    int32_t acc = kivi_dot_q8_q2t(kq, v_q2t + m * packed_N, N);
                    context[idx3(h, d, m, D, M)] = (float)acc * k_scale * v_scales[m];
                }
            }

            for (size_t n = 0; n < N; n++){
                float *phi_q_n = phi_q + n * D;
                for (size_t d = 0; d < D; d++){
                    phi_q_n[d] = kivi_linear_phi(q->data[idx4(b, h, n, d, H, N, D)]);
                }
            }

            for (size_t n = 0; n < N; n++){
                float *phi_q_n = phi_q + n * D;
                float den = 0.0f;
                memset(num, 0, M * sizeof(float));

                for (size_t d = 0; d < D; d++){
                    float qp = phi_q_n[d];
                    den += qp * k_sum[idx2(h, d, D)];
                    float *ctx_d = context + idx3(h, d, 0, D, M);
                    for (size_t m = 0; m < M; m++){
                        num[m] += qp * ctx_d[m];
                    }
                }
                den += eps;

                for (size_t m = 0; m < M; m++){
                    y->data[idx4(b, n, h, m, N, H, M)] = num[m] / den;
                }
            }
        }
    }

    ATTN_TIMER += MiCo_time() - start_time;

    free(phi_k_col);
    free(phi_k_q8);
    free(phi_k_scales);
    free(v_q2t);
    free(v_scales);
    free(v_col);
    free(context);
    free(k_sum);
    free(phi_q);
    free(num);
}

#if defined(KIVI_ATTN_REF) || defined(KIVI_COMPARE_REF)
void MiCo_ViT_kivi_attention_ref_f32(
    Tensor4D_F32 *y,
    const Tensor4D_F32 *q,
    const Tensor4D_F32 *k,
    const Tensor4D_F32 *v,
    const float scale
){
    const size_t B = q->shape[0];
    const size_t H = q->shape[1];
    const size_t I = q->shape[2];
    const size_t F = q->shape[3];
    const size_t J = k->shape[2];
    const size_t packed_J = (J + 3) / 4;
    const size_t packed_F = (F + 3) / 4;
    const size_t padded_J = packed_J * 4;
    const size_t padded_F = packed_F * 4;

    float *scores = (float *)malloc(J * sizeof(float));
    qbyte *k_q = (qbyte *)malloc(F * packed_J * sizeof(qbyte));
    float *k_scales = (float *)malloc(F * sizeof(float));
    qbyte *v_q = (qbyte *)malloc(J * packed_F * sizeof(qbyte));
    float *v_scales = (float *)malloc(J * sizeof(float));
    size_t max_buf = padded_J > padded_F ? padded_J : padded_F;
    float *temp_buf = (float *)malloc(max_buf * sizeof(float));
    MiCo_assert(scores != NULL && k_q != NULL && k_scales != NULL &&
                v_q != NULL && v_scales != NULL && temp_buf != NULL,
                "[KIVI Attention Ref] failed to allocate buffers");

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            size_t k_base = idx4(b, h, 0, 0, H, J, F);
            size_t v_base = idx4(b, h, 0, 0, H, J, F);

            for (size_t f = 0; f < F; f++){
                for (size_t j = 0; j < J; j++){
                    temp_buf[j] = k->data[k_base + j * F + f];
                }
                for (size_t j = J; j < padded_J; j++){
                    temp_buf[j] = 0.0f;
                }
                k_scales[f] = __FP32toQ2T(k_q + f * packed_J, temp_buf, J);
            }

            for (size_t j = 0; j < J; j++){
                memcpy(temp_buf, &v->data[v_base + j * F], F * sizeof(float));
                for (size_t f = F; f < padded_F; f++){
                    temp_buf[f] = 0.0f;
                }
                v_scales[j] = __FP32toQ2T(v_q + j * packed_F, temp_buf, F);
            }

            for (size_t i = 0; i < I; i++){
                size_t q_base = idx4(b, h, i, 0, H, I, F);

                for (size_t j = 0; j < J; j++){
                    float sum = 0.0f;
                    for (size_t f = 0; f < F; f++){
                        const uint8_t packed = (uint8_t)k_q[f * packed_J + j / 4];
                        float k_val = (float)kivi_decode_ternary(packed, j % 4);
                        sum += q->data[q_base + f] * k_val * k_scales[f];
                    }
                    scores[j] = sum / scale;
                }

                MiCo_softmax_vec(scores, scores, J);

                for (size_t f = 0; f < F; f++){
                    float out_sum = 0.0f;
                    for (size_t j = 0; j < J; j++){
                        const uint8_t packed = (uint8_t)v_q[j * packed_F + f / 4];
                        float v_val = (float)kivi_decode_ternary(packed, f % 4);
                        out_sum += scores[j] * v_val * v_scales[j];
                    }
                    y->data[idx4(b, i, h, f, I, H, F)] = out_sum;
                }
            }
        }
    }

    free(scores);
    free(k_q);
    free(k_scales);
    free(v_q);
    free(v_scales);
    free(temp_buf);
}
#endif

/*
    KIVI Attention: 1.58-bit ternary quantization for KV

    Q  stored in FP32
    K  per-channel quantization (one scale per feature channel, computed across all tokens)
    V  per-token quantization   (one scale per token, computed across all features)
*/
__attribute__((weak)) void MiCo_ViT_kivi_attention_f32(
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

    MiCo_assert(k->shape[0] == B && k->shape[1] == H && k->shape[3] == F, "[KIVI Attention] k shape mismatch");
    MiCo_assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == J && v->shape[3] == F, "[KIVI Attention] v shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == I && y->shape[2] == H && y->shape[3] == F, "[KIVI Attention] y shape mismatch");
    MiCo_assert(scale != 0.0f, "[KIVI Attention] scale must be non-zero");

    MiCo_init_exp_lut();

    const size_t packed_J = (J + 3) / 4;
    const size_t packed_F = (F + 3) / 4;
    const size_t padded_J = packed_J * 4;
    const size_t padded_F = packed_F * 4;

    float *scores = (float *)malloc(J * sizeof(float));
    MiCo_assert(scores != NULL, "[KIVI Attention] failed to allocate scores buffer");
    float *output_buf = (float *)malloc(F * sizeof(float));
    MiCo_assert(output_buf != NULL, "[KIVI Attention] failed to allocate output buffer");

    // K per-channel: one packed_J qbytes per channel, one scale per channel
    qbyte *k_q = (qbyte *)malloc(F * packed_J * sizeof(qbyte));
    float *k_scales = (float *)malloc(F * sizeof(float));
    MiCo_assert(k_q != NULL && k_scales != NULL, "[KIVI Attention] failed to allocate K quant buffers");

#ifdef KIVI_V_LAYOUT_OPT
    // Experimental V layout: packed feature block, then token: [packed_F][J].
    qbyte *v_q = (qbyte *)malloc(packed_F * J * sizeof(qbyte));
#else
    // V per-token: one packed_F qbytes per token, one scale per token
    qbyte *v_q = (qbyte *)malloc(J * packed_F * sizeof(qbyte));
#endif
    float *v_scales = (float *)malloc(J * sizeof(float));
    MiCo_assert(v_q != NULL && v_scales != NULL, "[KIVI Attention] failed to allocate V quant buffers");

    // Temp buffer for gathering per-channel K values (non-contiguous across J)
    size_t max_buf = padded_J > padded_F ? padded_J : padded_F;
    float *temp_buf = (float *)malloc(max_buf * sizeof(float));
    MiCo_assert(temp_buf != NULL, "[KIVI Attention] failed to allocate temp buffer");
#ifdef KIVI_V_LAYOUT_OPT
    qbyte *quant_tmp = (qbyte *)malloc((packed_J > packed_F ? packed_J : packed_F) * sizeof(qbyte));
    MiCo_assert(quant_tmp != NULL, "[KIVI Attention] failed to allocate quant temp buffer");
#endif

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

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            size_t k_base = idx4(b, h, 0, 0, H, J, F);
            size_t v_base = idx4(b, h, 0, 0, H, J, F);

            // ---- per-channel quantization of K ----
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

            // ---- per-token quantization of V ----
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
#ifdef KIVI_V_LAYOUT_OPT
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
#else
                v_scales[j] = __FP32toQ2T(v_q + j * packed_F, temp_buf, F);
#ifdef KIVI_PROFILE_INTERNAL
                kivi_prof_v_quant += MiCo_time() - prof_start;
#endif
#endif
            }

            // ---- attention computation ----
            for (size_t i = 0; i < I; i++){
                size_t q_base = idx4(b, h, i, 0, H, I, F);

                // scores = Q @ K^T / scale
#ifdef KIVI_PROFILE_INTERNAL
                long prof_start = MiCo_time();
#endif
                memset(scores, 0, J * sizeof(float));
                for (size_t f = 0; f < F; f++){
                    float q_scaled = (q->data[q_base + f] * k_scales[f]) / scale;
                    kivi_accum_scores_from_k(scores, q_scaled, k_q + f * packed_J, J);
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

                // output = scores @ V
#ifdef KIVI_V_LAYOUT_OPT
                size_t f = 0;
                for (size_t fb = 0; fb < packed_F; fb++){
                    float o0 = 0.0f;
                    float o1 = 0.0f;
                    float o2 = 0.0f;
                    float o3 = 0.0f;
                    const qbyte *v_block = v_q + fb * J;
                    for (size_t j = 0; j < J; j++){
                        const float score_scaled = scores[j] * v_scales[j];
                        const uint8_t packed = (uint8_t)v_block[j];
                        o0 += kivi_apply_ternary(score_scaled, packed, 0);
                        o1 += kivi_apply_ternary(score_scaled, packed, 1);
                        o2 += kivi_apply_ternary(score_scaled, packed, 2);
                        o3 += kivi_apply_ternary(score_scaled, packed, 3);
                    }
                    output_buf[f++] = o0;
                    if (f < F) output_buf[f++] = o1;
                    if (f < F) output_buf[f++] = o2;
                    if (f < F) output_buf[f++] = o3;
                }
#else
                memset(output_buf, 0, F * sizeof(float));
                for (size_t j = 0; j < J; j++){
                    float score_scaled = scores[j] * v_scales[j];
                    kivi_accum_output_from_v(output_buf, score_scaled, v_q + j * packed_F, F);
                }
#endif
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
    printf("KIVI_INTERNAL_PROFILE total=%ld k_gather=%ld k_quant=%ld v_copy=%ld v_quant=%ld v_layout=%ld score_accum=%ld softmax=%ld output_accum=%ld output_store=%ld\n",
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
    free(k_q);
    free(k_scales);
    free(v_q);
    free(v_scales);
    free(temp_buf);
#ifdef KIVI_V_LAYOUT_OPT
    free(quant_tmp);
#endif
}
