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