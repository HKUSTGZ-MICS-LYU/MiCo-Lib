#include "nn.h"

#define FLOAT_MAX 1e10

void MiCo_argmax2d_f32(size_t *idx, const Tensor2D_F32 *x){

    for (size_t i = 0; i < x->shape[0]; i++){
        float max = -FLOAT_MAX;
        for(size_t j = 0; j < x->shape[1]; j++){
            float data = x->data[i * x->shape[1] + j];
            if(data > max){
                idx[i] = j;
                max = data;
            }
        }
    }
}
