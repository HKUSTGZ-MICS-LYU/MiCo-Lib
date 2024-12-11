#ifndef __MICO_NN_H
#define __MICO_NN_H

#include <stdint.h>
#ifdef RISCV
#include "sim_stdlib.h"
#else
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#endif

#include "qtypes.h"


typedef struct {
    int8_t *data;
    float scale;
} Tensor0D_Q8; // Scalar

typedef struct{
    size_t shape[1];
    int8_t *data;
    float scale;
} Tensor1D_Q8; // 1-D Tensor

typedef struct{
    size_t shape[2];
    int8_t *data;
    float scale;
} Tensor2D_Q8; // 2-D Tensor

typedef struct{
    size_t shape[3];
    int8_t *data;
    float scale;
} Tensor3D_Q8; // 3-D Tensor

typedef struct{
    size_t shape[4];
    int8_t *data;
    float scale;
} Tensor4D_Q8; // 3-D Tensor

#endif // __MICO_NN_H