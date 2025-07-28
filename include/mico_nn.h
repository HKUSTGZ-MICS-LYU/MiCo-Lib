#ifndef __MICO_NN_H
#define __MICO_NN_H

#include <stdint.h>
#include <malloc.h>
#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdlib.h>
#include <string.h>
#endif

#include "nn.h"
#include "qtypes.h"

// TODO: Here Q8 is used for any bit-width like 8, 4, 2, 1. Alter it later.
typedef struct {
    qbyte *data;
    float scale;
} Tensor0D_Q8; // Scalar

typedef struct{
    size_t shape[1];
    qbyte *data;
    float scale;
} Tensor1D_Q8; // 1-D Tensor

typedef struct{
    size_t shape[2];
    qbyte *data;
    float scale;
} Tensor2D_Q8; // 2-D Tensor

typedef struct{
    size_t shape[3];
    qbyte *data;
    float scale;
} Tensor3D_Q8; // 3-D Tensor

typedef struct{
    size_t shape[4];
    qbyte *data;
    float scale;
} Tensor4D_Q8; // 3-D Tensor

// Dense/Linear Functions
void MiCo_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8 *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq, const size_t align);

// Convolution Functions
void MiCo_bitconv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups, const size_t align);

#endif // __MICO_NN_H