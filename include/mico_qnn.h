#ifndef __MICO_QNN_H
#define __MICO_QNN_H

#include "mico_nn.h"
#include "qtypes.h"

typedef enum {
    MICO_OPT_DEFAULT = 0,
    MICO_OPT_UNROLL = 1,
    MICO_OPT_LUT = 2
} MiCoMatMulOpt;

void MiCo_SetDefaultMatMulOpt(MiCoMatMulOpt opt);
MiCoMatMulOpt MiCo_GetDefaultMatMulOpt(void);

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

// Layer-wise selectable optimization wrappers
void MiCo_Q8_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q4_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q2_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q1_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q8x4_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q8x2_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q8x1_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q4x2_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q4x1_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);
void MiCo_Q2x1_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w, MiCoMatMulOpt opt);

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
