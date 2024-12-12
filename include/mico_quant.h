#ifndef __MICO_QUANT_H
#define __MICO_QUANT_H

#include "nn.h"
#include "mico_nn.h"
#include "qtypes.h"

void MiCo_2D_FP32toQ8(Tensor2D_Q8 *qx, const Tensor2D_F32 *x);
// void MiCo_FP32_to_Q4(Tensor2D_Q8 *qx, Tensor2D_F32 *x);


void MiCo_4D_FP32toQ8(Tensor4D_Q8 *qx, const Tensor4D_F32 *x);

#endif // __MICO_QUANT_H