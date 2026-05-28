#include "nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "profile.h"
#include <math.h>
#include <string.h>

extern long SOFTMAX_TIMER;
extern long ATTN_TIMER;

#ifndef MICO_LLAMA_KV_GROUP_SIZE
#define MICO_LLAMA_KV_GROUP_SIZE 32
#endif

#ifndef MICO_LLAMA_KV_PACKED_GROUP_BYTES
#define MICO_LLAMA_KV_PACKED_GROUP_BYTES(head_size) \
    (((size_t)MICO_LLAMA_KV_GROUP_SIZE * (size_t)(head_size) + 3) / 4)
#endif

void softmax(float* x, int size) {
    long start = MiCo_time();
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
    long end = MiCo_time();
    SOFTMAX_TIMER += end - start;
}

void MiCo_multihead_attention_f32(
    Tensor2D_F32* output,           // [n_heads, head_size] - output buffer
    const Tensor2D_F32* query,     // [n_heads, head_size] - query vectors
    float* key_cache,         // key cache buffer
    float* value_cache,       // value cache buffer
    float* att_buffer,              // [n_heads, seq_len] - attention scores buffer
    const int pos,                  // current position
    const MiCo_MHA_Config* cfg      // MHA configuration
){
    const int n_heads = cfg->n_heads;
    const int head_size = cfg->head_size;
    const int kv_dim = cfg->kv_dim;
    const int kv_mul = cfg->kv_mul;
    const int seq_len = cfg->seq_len;

    const float scale = 1.0f / sqrtf((float)head_size);

    // Temporary bias (none)
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };
    long start_time = MiCo_time();
    int h;
    for (h = 0; h < n_heads; h++) {
        // get the query vector for this head
        float* q = query->data + h * head_size;
        // attention scores for this head
        float* att = att_buffer + h * seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);

        // weighted sum of the values, store back into xb
        float* xb = output->data + h * head_size;
        for(int i = 0; i < head_size; i++){
            xb[i] = 0.0f;
        }

        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* v = value_cache + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
    ATTN_TIMER += MiCo_time() - start_time;
    return;
}

void MiCo_multihead_attention_f32_kv8(
    Tensor2D_F32* output,           // [n_heads, head_size] - output buffer
    const Tensor2D_F32* query,     // [n_heads, head_size] - query vectors
    int8_t* key_cache,        // key cache buffer (layer offset already applied)
    int8_t* value_cache,      // value cache buffer (layer offset already applied)
    float* key_scales,       // key scales buffer (layer offset already applied), layout: (seq_len,)
    float* value_scales,     // value scales buffer (layer offset already applied), layout: (seq_len,)
    float* att_buffer,             // [n_heads, seq_len] - attention scores buffer
    const int pos,                 // current position
    const MiCo_MHA_Config* cfg     // MHA configuration
){
    const int n_heads = cfg->n_heads;
    const int head_size = cfg->head_size;
    const int kv_dim = cfg->kv_dim;
    const int kv_mul = cfg->kv_mul;
    const int seq_len = cfg->seq_len;

    const float attn_scale = 1.0f / sqrtf((float)head_size);

    // Temporary bias (none)
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };

    long start_time = MiCo_time();

    int h;
    for (h = 0; h < n_heads; h++) {
        int kv_head = h / kv_mul;
        // get the query vector for this head
        float* q = query->data + h * head_size;
        // attention scores for this head
        float* att = att_buffer + h * seq_len;
        float* xb = output->data + h * head_size;
        
        #ifdef USE_INT8_Q
        int8_t q_int8[head_size];
        float q_scale = __FP32toQ8((qbyte*)q_int8, q, head_size);
        for (int t = 0; t <= pos; t++) {
            int8_t* k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
            int32_t acc = 0;
            for (int i = 0; i < head_size; i++) {
                acc += (int32_t)q_int8[i] * (int32_t)k[i];
            }
            att[t] = (float)acc * q_scale * key_scales[t] * attn_scale;
        }
        #else
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            int8_t* k = key_cache + t * kv_dim + kv_head * head_size;
            // get the key scale for this timestep
            float k_scale = key_scales[t];
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += (q[i] * k[i]) * k_scale;
            }
            score *= attn_scale;
            // save the score to the attention buffer
            att[t] = score;
        }
        #endif
        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);

        // weighted sum of the values, store back into xb
        for(int i = 0; i < head_size; i++){
            xb[i] = 0.0f;
        }

        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            int8_t* v = value_cache + t * kv_dim + kv_head * head_size;
            // get the value scale for this timestep
            float v_scale = value_scales[t];
            // get the attention weight for this timestep
            float av = att[t] * value_scales[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += av * v[i];
            }
        }
    }
    ATTN_TIMER += MiCo_time() - start_time;
    return;
}

static inline int llama_decode_ternary(uint8_t packed, size_t lane) {
    const int bits = (packed >> (2 * lane)) & 0x3;
    return (bits == 1) ? 1 : (bits == 3) ? -1 : 0;
}

static inline int llama_decode_q2t_at(const qbyte *packed_q2t, size_t elem) {
    const uint8_t packed = (uint8_t)packed_q2t[elem >> 2];
    return llama_decode_ternary(packed, elem & 0x3);
}

static inline uint8_t llama_encode_ternary(int v) {
    return (v > 0) ? 1 : (v < 0) ? 3 : 0;
}

static inline int q8_q2t_dot_product(
    const qbyte *q8,
    const qbyte *packed_q2t,
    const size_t n
){
    int sum = 0;
    for (int i = 0; i < n; i++){
        sum += q8[i] * llama_decode_q2t_at(packed_q2t, i);
    }
    return sum;
}

static inline void llama_pack_q2t_value(qbyte *packed_q2t, size_t elem, int value) {
    const size_t byte_idx = elem >> 2;
    const size_t lane = elem & 0x3;
    packed_q2t[byte_idx] &= (qbyte)~(0x3u << (2 * lane));
    packed_q2t[byte_idx] |= (qbyte)(llama_encode_ternary(value) << (2 * lane));
}

static inline void llama_accum_output_q2t_offset(
    float *out,
    qbyte score_q8,
    float score_scale,
    const qbyte *packed_q2t,
    float value_scale,
    size_t n
) {
    if (score_q8 == 0) return;
    const float score_scaled = (float)score_q8 * score_scale * value_scale;
    for (size_t i = 0; i < n; i++) {
        const int vv = llama_decode_q2t_at(packed_q2t, i);
        if (vv > 0) out[i] += score_scaled;
        else if (vv < 0) out[i] -= score_scaled;
    }
}

__attribute__((weak)) void MiCo_llama_pack_kv_group_q2t(
    const float* key_cache,
    const float* value_cache,
    qbyte* key_cache_q2t,
    qbyte* value_cache_q2t,
    float* key_scales,
    float* value_scales,
    const int group_id,
    const MiCo_MHA_Config* cfg
) {
    const int head_size = cfg->head_size;
    const int kv_dim = cfg->kv_dim;
    const int n_kv_heads = cfg->n_heads / cfg->kv_mul;
    const int seq_len = cfg->seq_len;
    const int group_size = MICO_LLAMA_KV_GROUP_SIZE;

    MiCo_assert(group_id >= 0, "[LLaMa Groupwise KV] group_id must be non-negative");
    MiCo_assert(group_id * group_size < seq_len, "[LLaMa Groupwise KV] group_id out of range");
    MiCo_assert(key_cache != NULL && value_cache != NULL, "[LLaMa Groupwise KV] FP32 cache is NULL");
    MiCo_assert(key_cache_q2t != NULL && value_cache_q2t != NULL, "[LLaMa Groupwise KV] q2t cache is NULL");
    MiCo_assert(key_scales != NULL && value_scales != NULL, "[LLaMa Groupwise KV] scale buffer is NULL");

    const size_t packed_group_bytes = MICO_LLAMA_KV_PACKED_GROUP_BYTES(head_size);
    const size_t packed_group_tokens = ((size_t)group_size + 3) / 4;
    const size_t padded_group_tokens = packed_group_tokens * 4;
    const size_t packed_head_size = ((size_t)head_size + 3) / 4;
    const size_t padded_head_size = packed_head_size * 4;
    const size_t temp_elems = padded_group_tokens > padded_head_size ? padded_group_tokens : padded_head_size;
    float *temp = (float *)malloc(temp_elems * sizeof(float));
    MiCo_assert(temp != NULL, "[LLaMa Groupwise KV] failed to allocate pack buffer");

    for (int kv_head = 0; kv_head < n_kv_heads; kv_head++) {
        const size_t group_head_idx = (size_t)group_id * (size_t)n_kv_heads + (size_t)kv_head;
        qbyte *k_group = key_cache_q2t + group_head_idx * packed_group_bytes;
        qbyte *v_group = value_cache_q2t + group_head_idx * packed_group_bytes;
        float *k_scale_group = key_scales + group_head_idx * (size_t)head_size;
        float *v_scale_group = value_scales + group_head_idx * (size_t)group_size;

        // K is per-channel quantized: each feature channel packs all tokens in the group.
        for (int f = 0; f < head_size; f++) {
            for (int local_t = 0; local_t < group_size; local_t++) {
                if (group_id * group_size + local_t < seq_len) {
                    temp[local_t] = key_cache[(size_t)local_t * (size_t)kv_dim +
                                              (size_t)kv_head * (size_t)head_size +
                                              (size_t)f];
                } else {
                    temp[local_t] = 0.0f;
                }
            }
            for (size_t i = (size_t)group_size; i < padded_group_tokens; i++) {
                temp[i] = 0.0f;
            }
            k_scale_group[f] = __FP32toQ2T(k_group + (size_t)f * packed_group_tokens, temp, padded_group_tokens);
        }

        // V is per-token quantized: each token packs its feature vector.
        for (int local_t = 0; local_t < group_size; local_t++) {
            if (group_id * group_size + local_t < seq_len) {
                const float *src = value_cache + (size_t)local_t * (size_t)kv_dim + (size_t)kv_head * (size_t)head_size;
                memcpy(temp, src, (size_t)head_size * sizeof(float));
            } else {
                memset(temp, 0, (size_t)head_size * sizeof(float));
            }
            for (size_t i = (size_t)head_size; i < padded_head_size; i++) {
                temp[i] = 0.0f;
            }
            v_scale_group[local_t] = __FP32toQ2T(v_group + (size_t)local_t * packed_head_size, temp, padded_head_size);
        }
    }

    free(temp);
}

__attribute__((weak)) void MiCo_llama_kivi_attention_f32(
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
    const size_t packed_group_tokens = ((size_t)group_size + 3) / 4;
    const size_t packed_head_size = ((size_t)head_size + 3) / 4;

    MiCo_assert(pos >= 0 && pos < seq_len, "[LLaMa Groupwise Attention] pos out of range");
    MiCo_assert(output != NULL && query != NULL, "[LLaMa Groupwise Attention] tensor is NULL");
    MiCo_assert(key_cache != NULL && value_cache != NULL, "[LLaMa Groupwise Attention] FP32 cache is NULL");
    MiCo_assert(key_cache_q2t != NULL && value_cache_q2t != NULL, "[LLaMa Groupwise Attention] q2t cache is NULL");
    MiCo_assert(key_scales != NULL && value_scales != NULL, "[LLaMa Groupwise Attention] scale buffer is NULL");
    MiCo_assert(att_buffer != NULL, "[LLaMa Groupwise Attention] attention buffer is NULL");
    MiCo_assert(kv_mul > 0 && n_kv_heads > 0, "[LLaMa Groupwise Attention] invalid GQA config");

    long start_time = MiCo_time();
#ifdef KIVI_PROFILE_INTERNAL
    long prof_q_quant = 0;
    long prof_hist_score = 0;
    long prof_float_score = 0;
    long prof_softmax = 0;
    long prof_hist_output = 0;
    long prof_float_output = 0;
#endif

    qbyte *q_int8 = (qbyte *)malloc((size_t)head_size * sizeof(qbyte));
    qbyte *score_int8 = (qbyte *)malloc((size_t)seq_len * sizeof(qbyte));
    float *qk_scaled = (float *)malloc((size_t)head_size * sizeof(float));
    qbyte *qk_int8 = (qbyte *)malloc((size_t)head_size * sizeof(qbyte));
    qbyte *k_token_q2t = (qbyte *)malloc(packed_head_size * sizeof(qbyte));
    float *score_v_scaled = (float *)malloc((size_t)group_size * sizeof(float));
    qbyte *score_v_int8 = (qbyte *)malloc((size_t)group_size * sizeof(qbyte));
    qbyte *v_channel_q2t = (qbyte *)malloc(packed_group_tokens * sizeof(qbyte));
    MiCo_assert(q_int8 != NULL && score_int8 != NULL &&
                qk_scaled != NULL && qk_int8 != NULL && k_token_q2t != NULL &&
                score_v_scaled != NULL && score_v_int8 != NULL && v_channel_q2t != NULL,
                "[LLaMa Groupwise Attention] failed to allocate quant buffers");

    for (int h = 0; h < n_heads; h++) {
        const int kv_head = h / kv_mul;
        const float *q = query->data + (size_t)h * (size_t)head_size;
        float *att = att_buffer + (size_t)h * (size_t)seq_len;
        float *out = output->data + (size_t)h * (size_t)head_size;

#ifdef KIVI_PROFILE_INTERNAL
        long prof_start = MiCo_time();
#endif
        const float q_scale = __FP32toQ8(q_int8, (float *)q, (size_t)head_size);
#ifdef KIVI_PROFILE_INTERNAL
        prof_q_quant += MiCo_time() - prof_start;
        prof_start = MiCo_time();
#endif

        for (int group = 0; group < current_group; group++) {
            const size_t group_head_idx = (size_t)group * (size_t)n_kv_heads + (size_t)kv_head;
            const qbyte *k_group = key_cache_q2t + group_head_idx * packed_group_bytes;
            const float *k_scale_group = key_scales + group_head_idx * (size_t)head_size;

            for (int f = 0; f < head_size; f++) {
                qk_scaled[f] = (float)q_int8[f] * k_scale_group[f];
            }
            const float qk_scale = __FP32toQ8(qk_int8, qk_scaled, (size_t)head_size);

            for (int local_t = 0; local_t < group_size; local_t++) {
                const int token = group * group_size + local_t;
                memset(k_token_q2t, 0, packed_head_size * sizeof(qbyte));
                for (int f = 0; f < head_size; f++) {
                    const qbyte *k_channel = k_group + (size_t)f * packed_group_tokens;
                    const int kv = llama_decode_q2t_at(k_channel, (size_t)local_t);
                    llama_pack_q2t_value(k_token_q2t, (size_t)f, kv);
                }
                const int acc = q8_q2t_dot_product(qk_int8, k_token_q2t, (size_t)head_size);
                att[token] = (float)acc * q_scale * qk_scale * attn_scale;
            }
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_hist_score += MiCo_time() - prof_start;
        prof_start = MiCo_time();
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
        prof_float_score += MiCo_time() - prof_start;
        prof_start = MiCo_time();
#endif

        softmax(att, pos + 1);
        const float score_scale = __FP32toQ8(score_int8, att, (size_t)pos + 1);
#ifdef KIVI_PROFILE_INTERNAL
        prof_softmax += MiCo_time() - prof_start;
        prof_start = MiCo_time();
#endif

        for (int i = 0; i < head_size; i++) {
            out[i] = 0.0f;
        }

        for (int group = 0; group < current_group; group++) {
            const size_t group_head_idx = (size_t)group * (size_t)n_kv_heads + (size_t)kv_head;
            const qbyte *v_group = value_cache_q2t + group_head_idx * packed_group_bytes;
            const float *v_scale_group = value_scales + group_head_idx * (size_t)group_size;

            for (int local_t = 0; local_t < group_size; local_t++) {
                const int token = group * group_size + local_t;
                score_v_scaled[local_t] = (float)score_int8[token] * v_scale_group[local_t];
            }
            const float score_v_scale = __FP32toQ8(score_v_int8, score_v_scaled, (size_t)group_size);

            for (int f = 0; f < head_size; f++) {
                memset(v_channel_q2t, 0, packed_group_tokens * sizeof(qbyte));
                for (int local_t = 0; local_t < group_size; local_t++) {
                    const qbyte *v_token = v_group + (size_t)local_t * packed_head_size;
                    const int vv = llama_decode_q2t_at(v_token, (size_t)f);
                    llama_pack_q2t_value(v_channel_q2t, (size_t)local_t, vv);
                }
                const int acc = q8_q2t_dot_product(score_v_int8, v_channel_q2t, (size_t)group_size);
                out[f] += (float)acc * score_scale * score_v_scale;
            }
        }
#ifdef KIVI_PROFILE_INTERNAL
        prof_hist_output += MiCo_time() - prof_start;
        prof_start = MiCo_time();
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
        prof_float_output += MiCo_time() - prof_start;
#endif
    }

    free(q_int8);
    free(score_int8);
    free(qk_scaled);
    free(qk_int8);
    free(k_token_q2t);
    free(score_v_scaled);
    free(score_v_int8);
    free(v_channel_q2t);

    long total_time = MiCo_time() - start_time;
    ATTN_TIMER += total_time;
#ifdef KIVI_PROFILE_INTERNAL
    printf("LLAMA_GROUPWISE_KV_PROFILE total=%ld q_quant=%ld hist_score=%ld float_score=%ld softmax=%ld hist_output=%ld float_output=%ld group_size=%d\n",
           total_time,
           prof_q_quant,
           prof_hist_score,
           prof_float_score,
           prof_softmax,
           prof_hist_output,
           prof_float_output,
           group_size);
#endif
}
