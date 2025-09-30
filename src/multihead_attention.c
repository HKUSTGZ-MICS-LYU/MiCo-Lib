#include "nn.h"

#include <math.h>


void softmax(float* x, int size) {
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

    return;
}

void MiCo_multihead_attention_f32_kv8(
    Tensor2D_F32* output,           // [n_heads, head_size] - output buffer
    const Tensor2D_F32* query,     // [n_heads, head_size] - query vectors
    int8_t* key_cache,        // key cache buffer
    int8_t* value_cache,      // value cache buffer
    float* key_scales,       // key scales buffer
    float* value_scales,     // value scales buffer
    float* att_buffer,             // [n_heads, seq_len] - attention scores buffer
    const int pos,                 // current position
    const MiCo_MHA_Config* cfg     // MHA configuration
){
    const int n_heads = cfg->n_heads;
    const int head_size = cfg->head_size;
    const int kv_dim = cfg->kv_dim;
    const int kv_mul = cfg->kv_mul;
    const int seq_len = cfg->seq_len;

    const float scale = 1.0f / sqrtf((float)head_size);

    // Temporary bias (none)
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };

    int h;
    for (h = 0; h < n_heads; h++) {
        // get the query vector for this head
        float* q = query->data + h * head_size;
        // attention scores for this head
        float* att = att_buffer + h * seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            int8_t* k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i] * key_scales[t];
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
            int8_t* v = value_cache + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i] * value_scales[t];
            }
        }
    }
    return;
}