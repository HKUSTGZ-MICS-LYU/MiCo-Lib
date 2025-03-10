#include "nn.h"

// Adding Functions
void MiCo_add4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x1, const Tensor4D_F32 *x2){
    int data_size = y->shape[1] * y->shape [2] * y->shape[3];
    for (int b = 0; b < y->shape[0]; b++){
        for(int n = 0; n < data_size; n++){
            y->data[b * data_size + n] = x1->data[b * data_size + n] + x2->data[b * data_size + n];
        }
    }
}
void MiCo_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2){

    for (int b = 0; b < y->shape[0]; b++){
        for(int n = 0; n < y->shape[1]; n++){
            y->data[b * y->shape[1] + n] = x1->data[b * x1->shape[1] + n] + x2->data[b * x2->shape[1] + n];
        }
    }
}