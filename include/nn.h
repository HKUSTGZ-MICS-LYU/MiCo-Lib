#ifndef __NN_H
#define __NN_H

#include <stdint.h>
#include <malloc.h>
#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#define FLOAT_MAX 1e10

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
    size_t shape[4];
    float *data;
} Tensor4D_F32; // 3-D Tensor

// Connection Macro
#define MiCo_CONNECT(y, x) (y)->data = (x)->data;
// Linear/Dense Functions
void MiCo_linear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, 
    const Tensor2D_F32 *weight, const Tensor1D_F32 *bias);

// Convolution Functions
void MiCo_conv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_F32* weight, const Tensor1D_F32* bias, 
    const size_t stride, const size_t padding, const size_t dilation, const size_t groups);

void MiCo_im2col_conv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_F32* weight, const Tensor1D_F32* bias, 
    const size_t stride, const size_t padding, const size_t dilation, const size_t groups);

// Adding Functions
void MiCo_add2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);
void MiCo_add4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x1, const Tensor4D_F32 *x2);

// Multiply Functions
void MiCo_mul2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);

// Pooling Functions
void MiCo_avgpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t k_size, const size_t stride, const size_t padding);
void MiCo_maxpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t k_size, const size_t stride, const size_t padding);
void MiCo_adaptive_avgpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t s);

// ReLU Functions
void MiCo_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x);
void MiCo_relu3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x);
void MiCo_relu4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x);

void MiCo_relu62d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x);
void MiCo_relu64d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x);

// Flatten Functions
void MiCo_flatten2d_f32(Tensor2D_F32 *y, const Tensor4D_F32 *x);

// Concat Functions
void MiCo_concat4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x1, const Tensor4D_F32 *x2);
void MiCo_concat2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2);

// Arg Functions
void MiCo_argmax2d_f32(size_t *idx, const Tensor2D_F32 *x);

// BatchNorm Functions
void MiCo_batchnorm2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor1D_F32 *weight, Tensor1D_F32 *bias, 
    Tensor1D_F32 *mean, const Tensor1D_F32 *var, 
    const float eps);

// Simple RMSNorm Functions
void MiCo_rmsnorm2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, 
    const Tensor1D_F32 *weight, const float eps);

// Channel Shuffle Functions (for ShuffleNet)
void MiCo_channel_shuffle(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t channels, const size_t groups);

// Utility Functions
void MiCo_print_tensor2d_f32(const Tensor2D_F32 *x);
void MiCo_print_tensor3d_f32(const Tensor3D_F32 *x);
void MiCo_print_tensor4d_f32(const Tensor4D_F32 *x);

// MatMul Functions
void MiCo_MatMul_f32(float* y, const float* x, const float* w, 
    const size_t m, const size_t n, const size_t p);

// Test Functions
void MiCo_assert(const int condition, const char *message);

// Im2Col Functions
float im2col_get_pixel(float *im, int height, int width,
                        int row, int col, int channel, int pad);
void im2col(float *data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float *data_col);
void im2col_T(float *data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float *data_col);

void im2col_block_T(const float* data_im, const int channels,
   const int height, const int width, const int kernel_size,
   const int stride, const int pad, float* data_col,
   const int row_offset, const int num_rows, const int out_width);
   
void im2col_block_T_aligned(const float* data_im, const int channels,
    const int height, const int width, const int kernel_size,
    const int stride, const int pad, float* data_col,
    const int row_offset, const int num_rows, const int out_width);
#endif // __NN_H