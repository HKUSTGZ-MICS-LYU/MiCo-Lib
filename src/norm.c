#include "nn.h"
#include <math.h>

void MiCo_batchnorm2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor1D_F32 *weight, Tensor1D_F32 *bias, 
    Tensor1D_F32 *mean, const Tensor1D_F32 *var, 
    const float eps){
    size_t batch_size = x->shape[0];
    #ifdef USE_ALT_LAYOUT
    size_t in_h = x->shape[1];
    size_t in_w = x->shape[2];
    size_t channel_size = x->shape[3];
    #else
    size_t channel_size = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];
    #endif

    #ifdef USE_ALT_LAYOUT
    for (size_t i = 0; i < batch_size; i++){
        for (size_t h = 0; h < in_h; h++){
            for (size_t w = 0; w < in_w; w++){
                for (size_t j = 0; j < channel_size; j++){
                    float scale = weight->data[j] / sqrtf(var->data[j] + eps);
                    size_t idx = OFFSET_4D(i, j, h, w, batch_size, channel_size, in_h, in_w);
                    y->data[idx] = (x->data[idx] - mean->data[j]) * scale + bias->data[j];
                }
            }
        }
    }
    #else
    size_t feature_size = in_h * in_w;
    for (size_t i = 0; i < batch_size; i++){
        for (size_t j = 0; j < channel_size; j++){
            float scale = weight->data[j] / sqrtf(var->data[j] + eps);
            size_t addr = i*channel_size*feature_size + j*feature_size;
            for (size_t k = 0; k < feature_size; k++){
                y->data[addr + k] = (x->data[addr + k] - mean->data[j]) * scale + bias->data[j];
            }
        }
    }
    #endif
    return;
}

void MiCo_rmsnorm2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, 
    const Tensor1D_F32 *weight, const float eps){
    size_t batch_size = x->shape[0];
    size_t dim_size = x->shape[1];

    for(size_t i = 0; i < batch_size; i++){
        float scale = 0;
        for(size_t j = 0; j < dim_size; j++){
            scale += x->data[i*dim_size + j] * x->data[i*dim_size + j];
        }
        scale /= dim_size;
        scale = 1.0 / sqrtf(scale + eps);
        for(size_t j = 0; j < dim_size; j++){
            y->data[i*dim_size + j] = x->data[i*dim_size + j] * scale * weight->data[j];
        }
    }
    return;
}

void MiCo_layernorm2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor1D_F32 *weight, const Tensor1D_F32 *bias,
    const size_t normalized_dim, const float eps){
    const size_t batch_size = x->shape[0];
    MiCo_assert(x->shape[1] == normalized_dim, "[LayerNorm2D] normalized_dim mismatch");
    for (size_t b = 0; b < batch_size; b++){
        const size_t base = b * normalized_dim;
        float mean = 0.0f;
        for (size_t i = 0; i < normalized_dim; i++){
            mean += x->data[base + i];
        }
        mean /= normalized_dim;

        float var = 0.0f;
        for (size_t i = 0; i < normalized_dim; i++){
            float v = x->data[base + i] - mean;
            var += v * v;
        }
        var /= normalized_dim;
        const float inv_std = 1.0f / sqrtf(var + eps);

        for (size_t i = 0; i < normalized_dim; i++){
            float normed = (x->data[base + i] - mean) * inv_std;
            y->data[base + i] = normed * weight->data[i] + bias->data[i];
        }
    }
}

void MiCo_layernorm3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x,
    const Tensor1D_F32 *weight, const Tensor1D_F32 *bias,
    const size_t normalized_dim, const float eps){
    const size_t batch_size = x->shape[0];
    const size_t seq_len = x->shape[1];
    MiCo_assert(x->shape[2] == normalized_dim, "[LayerNorm3D] normalized_dim mismatch");

    for (size_t b = 0; b < batch_size; b++){
        for (size_t s = 0; s < seq_len; s++){
            const size_t base = (b * seq_len + s) * normalized_dim;
            float mean = 0.0f;
            for (size_t i = 0; i < normalized_dim; i++){
                mean += x->data[base + i];
            }
            mean /= normalized_dim;

            float var = 0.0f;
            for (size_t i = 0; i < normalized_dim; i++){
                float v = x->data[base + i] - mean;
                var += v * v;
            }
            var /= normalized_dim;
            const float inv_std = 1.0f / sqrtf(var + eps);

            for (size_t i = 0; i < normalized_dim; i++){
                float normed = (x->data[base + i] - mean) * inv_std;
                y->data[base + i] = normed * weight->data[i] + bias->data[i];
            }
        }
    }
}
