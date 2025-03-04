#ifndef __MICO_QNN_H
#define __MICO_QNN_H

#include "mico_nn.h"
#include "qtypes.h"

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

// TODO: implement all these same width functions
void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

// TODO: implement all these mixed width functions
void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);


// Helper Macros
#define SIGN_EXTEND_TO_INT8(x, n) ((int8_t)((x) << (8-(n))) >> (8-(n)))
#define AMUX_2BIT(w, a) (w==1? a : (w==2? -(a << 1) : (w==3? -a : 0)))
#define AMUX_1BIT(w, a) ((w)? -(a) : (a))
#define EXTRACT_BIT(w, i) (((w) >> (i)) & 0x01)

#endif // __MICO_QNN_H