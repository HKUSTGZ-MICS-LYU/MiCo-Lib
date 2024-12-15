#include "mico_quant.h"

float __FP32toQ8(int8_t* qx, float* x, size_t n){
    float absmax = 0.0;
    for (int i = 0; i < n; i++){
        float val = x[i];
        if (val < 0){
            val = -val;
        }
        if (val > absmax){
            absmax = val;
        }
    }
    float scale = 127.0 / absmax;
    for (int i = 0; i < n; i++){
        qx[i] = (int8_t)(x[i] * scale);
    }
    return scale;
}

void MiCo_2D_FP32toQ8(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ8(qx->data+b*n*sizeof(int8_t), 
            x->data+b*n*sizeof(float), 
            n);
    }
}

void MiCo_4D_FP32toQ8(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ8(qx->data+b*n*sizeof(int8_t), 
            x->data+b*n*sizeof(float), 
            n);
    }
}