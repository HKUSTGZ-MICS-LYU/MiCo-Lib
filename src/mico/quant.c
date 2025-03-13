#include "mico_quant.h"

#include <math.h>

// TODO: roundf seems heavy here...

float __FP32toQ8(qbyte* qx, float* x, size_t n){
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
        qx[i] = (int8_t)(roundf(x[i] * scale));
    }
    return 1.0 / scale;
}

float __FP32toQ4(qbyte* qx, float* x, size_t n){
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
    float scale = 7.0 / absmax;
    for (int i = 0; i < n; i+=2){
        qx[i/2] = ((int8_t)(roundf(x[i] * scale)) & 0xF) | 
        (((int8_t)(roundf(x[i+1] * scale)) & 0xF) << 4);
    }
    return 1.0 / scale;
}

float __FP32toQ2(qbyte* qx, float* x, size_t n){
    float absmean = 0;
    for (int i=0; i<n; i++){
        absmean += x[i] > 0 ? x[i] : -x[i];
    }
    absmean /= n;
    float scale = 1.0 / absmean;

    // Note: AbsMax Quantization works as well
    // float absmax = 0.0;
    // for (int i = 0; i < n; i++){
    //     float val = x[i];
    //     if (val < 0){
    //         val = -val;
    //     }
    //     if (val > absmax){
    //         absmax = val;
    //     }
    // }
    // float scale = 1.0 / absmax;

    for (int i = 0; i < n; i+=4){
        qx[i/4] = (CLAMP_INT2((int8_t)(roundf(x[i] * scale)) & 0x3)) | 
            (CLAMP_INT2(((int8_t)(roundf(x[i+1] * scale)) & 0x3)) << 2) |
            (CLAMP_INT2(((int8_t)(roundf(x[i+2] * scale)) & 0x3)) << 4) |
            (CLAMP_INT2(((int8_t)(roundf(x[i+3] * scale)) & 0x3)) << 6);
    }
    return 1.0 / scale;
}

float __FP32toQ1(qbyte* qx, float* x, size_t n){
    float absmean = 0;
    for (int i=0; i<n; i++){
        absmean += x[i] > 0 ? x[i] : -x[i];
    }
    absmean /= n;
    float scale = absmean;

    for (int i=0; i<n; i+=8){
        qx[i/8] = 0;
        for (int j=0; j<8; j++){
            // TODO: This is quite unstable, especially when 0.0 is involved
            // Dithering might be needed
            
            // 0 for positive, 1 for negative
            qx[i/8] |= (x[i+j] <= 0) << j; 
        }
    }
    return scale;
}

// TODO: Currently batch_size is assumed to be 1, as scale is a scalar
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

void MiCo_2D_FP32toQ4(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ4(qx->data+b*n*sizeof(int8_t)/2, 
            x->data+b*n*sizeof(float), 
            n);
    }
}

void MiCo_4D_FP32toQ4(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ4(qx->data+b*n*sizeof(int8_t)/2, 
            x->data+b*n*sizeof(float), 
            n);
    }
}

void MiCo_2D_FP32toQ2(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ2(qx->data+b*n*sizeof(int8_t)/4, 
            x->data+b*n*sizeof(float), 
            n);
    }
}

void MiCo_4D_FP32toQ2(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ2(qx->data+b*n*sizeof(int8_t)/4, 
            x->data+b*n*sizeof(float), 
            n);
    }
}

void MiCo_2D_FP32toQ1(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ1(qx->data+b*n*sizeof(int8_t)/8, 
            x->data+b*n*sizeof(float), 
            n);
    }
}

void MiCo_4D_FP32toQ1(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    for (int b = 0; b < batch_size; b++){
        qx->scale = __FP32toQ1(qx->data+b*n*sizeof(int8_t)/8, 
            x->data+b*n*sizeof(float), 
            n);
    }
}