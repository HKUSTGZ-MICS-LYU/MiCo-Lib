#include "mico_quant.h"

#include <math.h>


void MiCo_2D_quant(Tensor2D_Q8 *qx, const Tensor2D_F32 *x, const qtype qbits){

    const size_t b = x->shape[0];
    const size_t n = x->shape[1];

    #ifdef QUANT_REUSE
    // Check if the buffer already store the same tensor
    if (MiCo_QX_Buffer_Global.src == x->data && 
        MiCo_QX_Buffer_Global.size == b*n &&
        MiCo_QX_Buffer_Global.qbits == qbits) {
        return;
    }
    #endif

    // Update Global Buffer Info
    MiCo_QX_Buffer_Global.src = x->data;
    MiCo_QX_Buffer_Global.size = b*n;
    MiCo_QX_Buffer_Global.qbits = qbits;
    MiCo_QX_Buffer_Global.dirty = 0;

    switch (qbits)
    {
      case 8:
        MiCo_2D_FP32toQ8(qx, x);
        break;
      case 4:
        MiCo_2D_FP32toQ4(qx, x);
        break;
      case 2:
        MiCo_2D_FP32toQ2(qx, x);
        break;
      case 1:
        MiCo_2D_FP32toQ1(qx, x);
        break;
      default:
        printf("[Warning] Unsupported Weight Quantization - %d\n", qbits);
        break;
    }
    return;
}

void MiCo_4D_quant(Tensor4D_Q8 *qx, const Tensor4D_F32 *x, const qtype qbits){

    const size_t b = x->shape[0];
    const size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    #ifdef QUANT_REUSE
    // Check if the buffer already store the same tensor
    if (MiCo_QX_Buffer_Global.src == x->data &&
        MiCo_QX_Buffer_Global.size == b*n &&
        MiCo_QX_Buffer_Global.qbits == qbits) {
        return;
    }
    #endif

    // Update Global Buffer Info
    MiCo_QX_Buffer_Global.src = x->data;
    MiCo_QX_Buffer_Global.size = b*n;
    MiCo_QX_Buffer_Global.qbits = qbits;
    MiCo_QX_Buffer_Global.dirty = 0;

    switch (qbits)
    {
      case 8:
        MiCo_4D_FP32toQ8(qx, x);
        break;
      case 4:
        MiCo_4D_FP32toQ4(qx, x);
        break;
      case 2:
        MiCo_4D_FP32toQ2(qx, x);
        break;
      case 1:
        MiCo_4D_FP32toQ1(qx, x);
        break;
      default:
        printf("[Warning] Unsupported Weight Quantization - %d\n", qbits);
        break;
    }
    return;
}

#ifdef USE_RVF
int roundf2i(float x){
    int result;
    asm volatile (
        "fcvt.w.s %0, %1, rmm"
        : "=r" (result)
        : "f" (x)
    );
    return result;
}

float roundf2f(float x){
    int result;
    asm volatile (
        "fcvt.w.s %0, %1, rmm"
        : "=r" (result)
        : "f" (x)
    );
    asm volatile (
        "fcvt.s.w %0, %1"
        : "=f" (x)
        : "r" (result)
    );
    return x;
}
#else
#define roundf2i(x) roundf(x)
#endif

__attribute__((weak)) float MiCo_absmax(float* x, size_t n){
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
    return absmax;
}

__attribute__((weak)) float MiCo_absmean(float* x, size_t n){
    float absmean = 0;
    for (int i=0; i<n; i++){
        absmean += x[i] > 0 ? x[i] : -x[i];
    }
    absmean /= n;
    return absmean;
}

// TODO: roundf2i is heavy without RVF support
__attribute__((weak)) float __FP32toQ8(qbyte* qx, float* x, size_t n){
    float scale = 127.0 / MiCo_absmax(x, n);
    for (int i = 0; i < n; i++){
        qx[i] = (int8_t)(roundf2i(x[i] * scale));
    }
    return 1.0 / scale;
}

__attribute__((weak)) float __FP32toQ4(qbyte* qx, float* x, size_t n){
    float scale = 7.0 / MiCo_absmax(x, n);
    for (int i = 0; i < n; i+=2){
        qx[i/2] = ((int8_t)(roundf2i(x[i] * scale)) & 0xF) | 
        (((int8_t)(roundf2i(x[i+1] * scale)) & 0xF) << 4);
    }
    return 1.0 / scale;
}

__attribute__((weak)) float __FP32toQ2(qbyte* qx, float* x, size_t n){

    float scale = 1.0 / MiCo_absmax(x, n);

    for (int i = 0; i < n; i+=4){
        // Unrolled 4 Times
        qx[i/4] = (CLAMP_INT2((int8_t)(roundf2i(x[i] * scale)) & 0x3)) | 
            (CLAMP_INT2(((int8_t)(roundf2i(x[i+1] * scale)) & 0x3)) << 2) |
            (CLAMP_INT2(((int8_t)(roundf2i(x[i+2] * scale)) & 0x3)) << 4) |
            (CLAMP_INT2(((int8_t)(roundf2i(x[i+3] * scale)) & 0x3)) << 6);
    }
    return 1.0 / scale;
}

__attribute__((weak)) float __FP32toQ1(qbyte* qx, float* x, size_t n){

    float scale = MiCo_absmean(x, n);

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

// Note: 
// Currently, quantization is batch-wise, which may affect the accuracy.
// And all the quantization will consider padding, if Q Tensor has larger size.
// than the original tensor, the extra part will be padded with 0.
__attribute__((weak)) void MiCo_2D_FP32toQ8(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];
    size_t qx_b = qx->shape[0];
    size_t qx_n = qx->shape[1];

    MiCo_assert(batch_size == qx_b, 
        "[Quantization] Batch Size Mismatched!");
    float scale = 127.0 / MiCo_absmax(x->data, batch_size*n);
    for (int b = 0; b < qx_b; b++){
        for (int i = 0; i < qx_n; i++){
            if (i >= n){
                // Padding if qx_n > n
                qx->data[b*qx_n + i] = 0;
                continue;
            }
            qx->data[b*qx_n + i] = (int8_t)(roundf2i(x->data[b*n + i] * scale));
        }
    }
    qx->scale = 1.0 / scale;
}

__attribute__((weak)) void MiCo_2D_FP32toQ4(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];
    size_t qx_b = qx->shape[0];
    size_t qx_n = qx->shape[1];

    MiCo_assert(batch_size == qx_b, 
        "[Quantization] Batch Size Mismatched!");
    float scale = 7.0 / MiCo_absmax(x->data, batch_size*n);
    
    for (int b = 0; b < qx_b; b++){
        for (int i = 0; i < qx_n; i+=2){
            if (i >= n-1){
                // Padding if qx_n > n
                qx->data[(b*qx_n + i)/2] = 0;
                continue;
            }
            // Handle case where we have odd number of elements
            int8_t second_val = 0;
            if (i+1 < n) {
                second_val = (int8_t)(roundf2i(x->data[b*n + i+1] * scale));
            }
            
            qx->data[(b*qx_n + i)/2] = ((int8_t)(roundf2i(x->data[b*n + i] * scale)) & 0xF) | 
                                      ((second_val & 0xF) << 4);
        }
    }
    qx->scale = 1.0 / scale;
}

__attribute__((weak)) void MiCo_2D_FP32toQ2(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];
    size_t qx_b = qx->shape[0];
    size_t qx_n = qx->shape[1];

    MiCo_assert(batch_size == qx_b, 
        "[Quantization] Batch Size Mismatched!");
    float scale = 1.0 / MiCo_absmax(x->data, batch_size*n);
    
    for (int b = 0; b < qx_b; b++){
        for (int i = 0; i < qx_n; i+=4){
            if (i >= n-3){
                // Padding if qx_n > n
                qx->data[(b*qx_n + i)/4] = 0;
                continue;
            }
            
            // Handle boundary conditions
            int8_t val0 = (i < n) ? (int8_t)(roundf2i(x->data[b*n + i] * scale)) : 0;
            int8_t val1 = (i+1 < n) ? (int8_t)(roundf2i(x->data[b*n + i+1] * scale)) : 0;
            int8_t val2 = (i+2 < n) ? (int8_t)(roundf2i(x->data[b*n + i+2] * scale)) : 0;
            int8_t val3 = (i+3 < n) ? (int8_t)(roundf2i(x->data[b*n + i+3] * scale)) : 0;
            
            qx->data[(b*qx_n + i)/4] = (CLAMP_INT2(val0 & 0x3)) | 
                                     (CLAMP_INT2((val1 & 0x3)) << 2) |
                                     (CLAMP_INT2((val2 & 0x3)) << 4) |
                                     (CLAMP_INT2((val3 & 0x3)) << 6);
        }
    }
    qx->scale = 1.0 / scale;
}

__attribute__((weak)) void MiCo_2D_FP32toQ1(Tensor2D_Q8 *qx, const Tensor2D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1];
    size_t qx_b = qx->shape[0];
    size_t qx_n = qx->shape[1];

    MiCo_assert(batch_size == qx_b, 
        "[Quantization] Batch Size Mismatched!");
    float scale = MiCo_absmean(x->data, batch_size*n);
    
    for (int b = 0; b < qx_b; b++){
        for (int i = 0; i < qx_n; i+=8){
            if (i >= n-7){
                // Padding if qx_n > n
                qx->data[(b*qx_n + i)/8] = 0;
                continue;
            }
            qx->data[(b*qx_n + i)/8] = 0;
            for (int j = 0; j < 8; j++){
                if (i+j >= n) {
                    // If we're past the original tensor size, we don't set any bits
                    continue;
                }
                // 0 for positive, 1 for negative
                qx->data[(b*qx_n + i)/8] |= (x->data[b*n + i+j] <= 0) << j;
            }
        }
    }
    qx->scale = scale;
}

// TODO: Modify it for the padding
__attribute__((weak)) void MiCo_4D_FP32toQ8(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];
    qx->scale = __FP32toQ8(qx->data, x->data, batch_size*n);
}

__attribute__((weak)) void MiCo_4D_FP32toQ4(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    qx->scale = __FP32toQ4(qx->data, x->data, batch_size*n);
}


__attribute__((weak)) void MiCo_4D_FP32toQ2(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    qx->scale = __FP32toQ2(qx->data, x->data, batch_size*n);
}

__attribute__((weak)) void MiCo_4D_FP32toQ1(Tensor4D_Q8 *qx, const Tensor4D_F32 *x){
    size_t batch_size = x->shape[0];
    size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    qx->scale = __FP32toQ1(qx->data, x->data, batch_size*n);
}