#ifndef __MICO_QUANT_H
#define __MICO_QUANT_H

#include "nn.h"
#include "mico_nn.h"
#include "qtypes.h"

void MiCo_2D_quant(Tensor2D_Q8 *qx, const Tensor2D_F32 *x, const qtype qbits);
void MiCo_4D_quant(Tensor4D_Q8 *qx, const Tensor4D_F32 *x, const qtype qbits);

void MiCo_2D_FP32toQ8(Tensor2D_Q8 *qx, const Tensor2D_F32 *x);
void MiCo_4D_FP32toQ8(Tensor4D_Q8 *qx, const Tensor4D_F32 *x);

void MiCo_2D_FP32toQ4(Tensor2D_Q8 *qx, const Tensor2D_F32 *x);
void MiCo_4D_FP32toQ4(Tensor4D_Q8 *qx, const Tensor4D_F32 *x);

void MiCo_2D_FP32toQ2(Tensor2D_Q8 *qx, const Tensor2D_F32 *x);
void MiCo_4D_FP32toQ2(Tensor4D_Q8 *qx, const Tensor4D_F32 *x);

void MiCo_2D_FP32toQ1(Tensor2D_Q8 *qx, const Tensor2D_F32 *x);
void MiCo_4D_FP32toQ1(Tensor4D_Q8 *qx, const Tensor4D_F32 *x);

// Prim Func
float __FP32toQ8(qbyte* qx, float* x, size_t n);
float __FP32toQ4(qbyte* qx, float* x, size_t n);
float __FP32toQ2(qbyte* qx, float* x, size_t n);
float __FP32toQ1(qbyte* qx, float* x, size_t n);

#define CLAMP(x, l, h) ((x) < (l) ? (l) : ((x) > (h) ? (h) : (x)))
#define CLAMP_INT2(x) CLAMP(x, -2, 1)

#endif // __MICO_QUANT_H