#include "nn.h"
#include "profile.h"
#include "mico_qnn.h"
#include "mico_quant.h"

#include <math.h>
#include <string.h>

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
    KIVI Attention: 1.58-bit ternary quantization for KV

    Q  stored in FP32
    K  per-channel quantization (one scale per feature channel, computed across all tokens)
    V  per-token quantization   (one scale per token, computed across all features)
*/
void MiCo_ViT_kivi_attention_f32(
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

    MiCo_assert(k->shape[0] == B && k->shape[1] == H && k->shape[3] == F, "[KIVI Attention] k shape mismatch");
    MiCo_assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == J && v->shape[3] == F, "[KIVI Attention] v shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == I && y->shape[2] == H && y->shape[3] == F, "[KIVI Attention] y shape mismatch");
    MiCo_assert(scale != 0.0f, "[KIVI Attention] scale must be non-zero");

    MiCo_init_exp_lut();

    const size_t packed_J = (J + 3) / 4;
    const size_t packed_F = (F + 3) / 4;

    float *scores = (float *)malloc(J * sizeof(float));
    MiCo_assert(scores != NULL, "[KIVI Attention] failed to allocate scores buffer");

    // K per-channel: one packed_J qbytes per channel, one scale per channel
    qbyte *k_q = (qbyte *)malloc(F * packed_J * sizeof(qbyte));
    float *k_scales = (float *)malloc(F * sizeof(float));
    MiCo_assert(k_q != NULL && k_scales != NULL, "[KIVI Attention] failed to allocate K quant buffers");

    // V per-token: one packed_F qbytes per token, one scale per token
    qbyte *v_q = (qbyte *)malloc(J * packed_F * sizeof(qbyte));
    float *v_scales = (float *)malloc(J * sizeof(float));
    MiCo_assert(v_q != NULL && v_scales != NULL, "[KIVI Attention] failed to allocate V quant buffers");

    // Temp buffer for gathering per-channel K values (non-contiguous across J)
    size_t max_buf = J > F ? J : F;
    float *temp_buf = (float *)malloc(max_buf * sizeof(float));
    MiCo_assert(temp_buf != NULL, "[KIVI Attention] failed to allocate temp buffer");

    long start_time = MiCo_time();

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            size_t k_base = idx4(b, h, 0, 0, H, J, F);
            size_t v_base = idx4(b, h, 0, 0, H, J, F);

            // ---- per-channel quantization of K ----
            for (size_t f = 0; f < F; f++){
                for (size_t j = 0; j < J; j++){
                    temp_buf[j] = k->data[k_base + j * F + f];
                }
                k_scales[f] = __FP32toQ2T(k_q + f * packed_J, temp_buf, J);
            }

            // ---- per-token quantization of V ----
            for (size_t j = 0; j < J; j++){
                v_scales[j] = __FP32toQ2T(v_q + j * packed_F, &v->data[v_base + j * F], F);
            }

            // ---- attention computation ----
            for (size_t i = 0; i < I; i++){
                size_t q_base = idx4(b, h, i, 0, H, I, F);

                // scores = Q @ K^T / scale
                for (size_t j = 0; j < J; j++){
                    float sum = 0.0f;
                    for (size_t f = 0; f < F; f++){
                        size_t k_byte_idx = f * packed_J + j / 4;
                        int shift = 2 * (j % 4);
                        int bits = (k_q[k_byte_idx] >> shift) & 0x3;
                        // ternary decode: 0->0, 1->1, 3->-1
                        float k_val = (bits == 1) ? 1.0f : (bits == 3) ? -1.0f : 0.0f;
                        sum += q->data[q_base + f] * k_val * k_scales[f];
                    }
                    scores[j] = sum / scale;
                }

                MiCo_softmax_vec(scores, scores, J);

                // output = scores @ V
                for (size_t f = 0; f < F; f++){
                    float out_sum = 0.0f;
                    for (size_t j = 0; j < J; j++){
                        size_t v_byte_idx = j * packed_F + f / 4;
                        int shift = 2 * (f % 4);
                        int bits = (v_q[v_byte_idx] >> shift) & 0x3;
                        float v_val = (bits == 1) ? 1.0f : (bits == 3) ? -1.0f : 0.0f;
                        out_sum += scores[j] * v_val * v_scales[j];
                    }
                    y->data[idx4(b, i, h, f, I, H, F)] = out_sum;
                }
            }
        }
    }

    ATTN_TIMER += MiCo_time() - start_time;

    free(scores);
    free(k_q);
    free(k_scales);
    free(v_q);
    free(v_scales);
    free(temp_buf);
}
