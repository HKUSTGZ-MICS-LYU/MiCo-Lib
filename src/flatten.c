#include "nn.h"


void MiCo_flatten2d_f32(Tensor2D_F32 *y, const Tensor4D_F32 *x){
    y->shape[0] = x->shape[0];
    y->shape[1] = x->shape[1] * x->shape[2] * x->shape[3];
    y->data = x->data;
}