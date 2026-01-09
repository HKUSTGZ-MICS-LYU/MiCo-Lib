#include "nn.h"


void MiCo_flatten2d_f32(Tensor2D_F32 *y, const Tensor4D_F32 *x){
    y->shape[0] = x->shape[0];
    y->shape[1] = x->shape[1] * x->shape[2] * x->shape[3];
    y->data = x->data;
}

void MiCo_NHWC2NCHW_flatten_f32(Tensor2D_F32 *y, const Tensor4D_F32 *x){
    const size_t N = x->shape[0];
    const size_t H = x->shape[1];
    const size_t W = x->shape[2];
    const size_t C = x->shape[3];

    y->shape[0] = N;
    y->shape[1] = C * H * W;
    y->data = x->data;

    // Operate on Y data in place
    if (C == 1 || (H == 1 && W == 1)) return;

    size_t count = N * H * W * C;
    float *temp = (float*)MiCo_alloc(count * sizeof(float), MICO_ALIGN);
    
    for (size_t n = 0; n < N; n++){
        for (size_t h = 0; h < H; h++){
            for (size_t w = 0; w < W; w++){
                for (size_t c = 0; c < C; c++){
                    size_t nhwc_idx = n*H*W*C + h*W*C + w*C + c;
                    size_t nchw_idx = n*C*H*W + c*H*W + h*W + w;
                    temp[nchw_idx] = x->data[nhwc_idx];
                }
            }
        }
    }
    memcpy(y->data, temp, count * sizeof(float));
    MiCo_free(temp);
}