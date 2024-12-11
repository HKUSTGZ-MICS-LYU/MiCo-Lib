#ifndef __NN_H
#define __NN_H

#include <stdint.h>
#ifdef RISCV
#include "sim_stdlib.h"
#else
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#endif

// http://elm-chan.org/junk/32bit/binclude.html
#define INCLUDE_FILE(section, filename, symbol) asm (\
    ".section "#section"\n"                   /* Change section */\
    ".balign 4\n"                             /* Word alignment */\
    ".global "#symbol"_start\n"               /* Export the object start address */\
    ".global "#symbol"_data\n"                /* Export the object address */\
    #symbol"_start:\n"                        /* Define the object start address label */\
    #symbol"_data:\n"                         /* Define the object label */\
    ".incbin \""filename"\"\n"                /* Import the file */\
    ".global "#symbol"_end\n"                 /* Export the object end address */\
    #symbol"_end:\n"                          /* Define the object end address label */\
    ".balign 4\n"                             /* Word alignment */\
    ".section \".text\"\n")                   /* Restore section */

typedef struct {
    float *data;
} Tensor0D_F32; // Scalar

typedef struct{
    size_t shape[1];
    float *data;
} Tensor1D_F32; // 1-D Tensor

typedef struct{
    size_t shape[2];
    float *data;
} Tensor2D_F32; // 2-D Tensor

typedef struct{
    size_t shape[3];
    float *data;
} Tensor3D_F32; // 3-D Tensor

typedef struct{
    size_t shape[3];
    float *data;
} Tensor4D_F32; // 3-D Tensor

// Linear/Dense Functions
void MiCo_linear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, 
    const Tensor2D_F32 *weight, const Tensor1D_F32 *bias);


// Convolution Functions
void MiCo_conv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, Tensor4D_F32* weight, Tensor1D_F32* bias, 
    const size_t stride, const size_t padding, const size_t dilation, const size_t groups);

// ReLU Functions
void MiCo_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x);
void MiCo_relu3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x);
void MiCo_relu4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x);

// Flatten Functions
void MiCo_flatten2d_f32(Tensor2D_F32 *y, const Tensor3D_F32 *x);
#endif // __NN_H