#ifndef __MICO_QNN_H
#define __MICO_QNN_H

#include "mico_nn.h"
#include "qtypes.h"

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

// Quantized Pooling Functions (im2col + matmul based)
// These functions perform pooling on quantized int8 inputs, producing quantized int8 outputs
// Layout: NCHW (batch, channels, height, width)
// Supported: kernel_size 2x2/3x3, stride 1/2, padding 0/1
void MiCo_Q8_AvgPool2D(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);
void MiCo_Q8_MaxPool2D(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);

// Reference implementations for testing
#ifdef REF
void MiCo_Q8_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q4_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q2_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q8x4_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q8x2_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q8x1_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q4x2_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q4x1_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q2x1_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q4x8_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q2x8_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q2x4_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q1x8_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1x4_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1x2_MatMul_Ref(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

// Reference implementations for quantized pooling
void MiCo_Q8_AvgPool2D_Ref(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);
void MiCo_Q8_MaxPool2D_Ref(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);
#endif

// Helper Macros
#define SIGN_EXTEND_TO_INT8(x, n) ((int8_t)((x) << (8-(n))) >> (8-(n)))
#define TWO_BIT_TO_INT8(x) ((x)==1 ? 1 : ((x)== 2 ? -2 : ((x)==3? -1 : 0)))
#define BIT_TO_INT8(x) ((x) ? -1 : 1)
#define AMUX_2BIT(w, a) ((w)==1? (a) : ((w)==2? -((a) << 1) : ((w)==3? -(a) : 0)))
#define AMUX_1BIT(w, a) ((w)? -(a) : (a))
#define EXTRACT_BIT(w, i) (((w) >> (i)) & 0x01)
#define EXTRACT_2BIT(w, i) (((w) >> (2 * (i))) & 0x03)
#define EXTRACT_4BIT(w, i) (((w) >> (4 * (i))) & 0x0F)

#endif // __MICO_QNN_H