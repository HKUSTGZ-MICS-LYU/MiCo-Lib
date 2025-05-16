#include "nn.h"

void MiCo_mul2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2){
    size_t batch_size = x1->shape[0];
    size_t dims = x1->shape[1];

    for (size_t i = 0; i < batch_size; i++){
        for (size_t j = 0; j < dims; j++){
            y->data[i * dims + j] = x1->data[i * dims + j] * x2->data[i * dims + j];
        }
    }

}
