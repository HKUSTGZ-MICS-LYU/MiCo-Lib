#include "mico_quant.h"

void MiCo_2D_FP32toQ8(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];

    for (int b = 0; b < batch_size; b++){
        float absmax = 0.0;
        for (int i = 0; i < n; i++){
            float val = x->data[b * n + i];
            if (val < 0){
                val = -val;
            }
            if (val > absmax){
                absmax = val;
            }
        }
        float scale = 127.0 / absmax;
        qx->scale = scale;
        for (int i = 0; i < n; i++){
            qx->data[i] = (int8_t)(x->data[i] * scale);
        }
    }
}

void MiCo_4D_FP32toQ8(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    for (int b = 0; b < batch_size; b++){
        float absmax = 0.0;
        for (int i = 0; i < n; i++){
            float val = x->data[b * n + i];
            if (val < 0){
                val = -val;
            }
            if (val > absmax){
                absmax = val;
            }
        }
        float scale = 127.0 / absmax;
        qx->scale = scale;
        for (int i = 0; i < n; i++){
            qx->data[i] = (int8_t)(x->data[i] * scale);
        }
    }
}