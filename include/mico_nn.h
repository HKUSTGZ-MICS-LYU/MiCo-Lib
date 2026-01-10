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

#ifndef QUANTIZE_BUFFER_SIZE
#define QUANTIZE_BUFFER_SIZE 1024*32
#endif

// Global buffer for quantization operations
extern qbyte MiCo_QBuffer[QUANTIZE_BUFFER_SIZE];

typedef struct 
{   qbyte* buffer;
    float* src;
    size_t size;
    char dirty;
    qtype qbits;
} MiCo_QX_Buffer;

extern MiCo_QX_Buffer MiCo_QX_Buffer_Global;

// TODO: Here Q8 is used for any bit-width like 8, 4, 2, 1. Alter it later.
typedef struct {
    qbyte *data;
    float scale;
    qtype wq; // weight quantization bits
} Tensor0D_Q8; // Scalar

typedef struct{
    size_t shape[1];
    qbyte *data;
    float scale;
    qtype wq; // weight quantization bits
} Tensor1D_Q8; // 1-D Tensor

typedef struct{
    size_t shape[2];
    qbyte *data;
    float scale;
    qtype wq; // weight quantization bits
} Tensor2D_Q8; // 2-D Tensor

typedef struct{
    size_t shape[3];
    qbyte *data;
    float scale;
    qtype wq; // weight quantization bits
} Tensor3D_Q8; // 3-D Tensor

typedef struct{
    size_t shape[4];
    qbyte *data;
    float scale;
    qtype wq; // weight quantization bits
} Tensor4D_Q8; // 4-D Tensor

// Group-wise quantized 2D tensor with per-group scales
typedef struct{
    size_t shape[2];
    qbyte *data;
    float *scales;      // per-group scales
    size_t group_size;  // size of each group
    qtype wq; // weight quantization bits
} Tensor2D_Q8_Groupwise; // 2-D Tensor with group-wise quantization

// Dense/Linear Functions
void MiCo_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8 *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq, const size_t align);

void MiCo_groupwise_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8_Groupwise *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq, const size_t align);

// Convolution Functions
void MiCo_bitconv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups, const size_t align);

// 1D Convolution Functions
void MiCo_bitconv1d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, 
    const Tensor3D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups, const size_t align);

#endif // __MICO_NN_H