#include "nn.h"
#include <math.h>

// TODO: Remove Dummy Implementation
void MiCo_batchnorm2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, const float eps){
    MiCo_CONNECT(y,x);
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