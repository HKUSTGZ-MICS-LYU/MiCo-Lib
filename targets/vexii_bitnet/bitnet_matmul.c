#include "mico_qnn.h"

#ifndef USE_SIMD
#define USE_SIMD 32
#endif

#ifndef BITNET_QUANT
#define BITNET_QUANT 3
#endif

// #define USE_DOTP8X2

// Activation
typedef uint32_t int8x4_t;

// Quantized Weight
typedef uint8_t int2x4_t;
typedef uint16_t int2x8_t;
typedef uint32_t int2x16_t;
typedef uint8_t int1x8_t;
typedef uint16_t int1x16_t;
typedef uint32_t int1x32_t;

#if BITNET_QUANT == 2
extern int32_t __bitnetadd4(int8x4_t a, int1x8_t w);
extern int32_t __bitnetadd8(int8x4_t a1, int8x4_t a2, int1x8_t w);
extern int32_t __bitnetadd16(int8_t *a, int1x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int1x32_t w, int1x32_t dummy);
extern int32_t __bitnetadd64(int8_t *a, int1x32_t w1, int1x32_t w2);
#else
extern int32_t __bitnetadd4(int8x4_t a, int2x4_t w);
extern int32_t __bitnetadd8(int8x4_t a1, int8x4_t a2, int2x8_t w);
extern int32_t __bitnetadd16(int8_t *a, int2x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int2x16_t w1, int2x16_t w2);
#endif

extern int32_t __dotp8(int8x4_t a, int8x4_t b);
extern int32_t __dotp8x2(int8x4_t a, int2x4_t b);

// 8x1, 8x1.5, 8x2 MatMul
void bitnet_qmatmul(int8_t *input, int32_t *output, int8_t *weight, int n, int d){
    
    for (int i=0; i<d; i++){
        output[i] = 0;
        for(int j = 0; j < n; j += USE_SIMD){
            // TODO: Add Padding here when n < USE_SIMD
            #if BITNET_QUANT == 2
            int addr = (i*n + j) >> 3;
            #if USE_SIMD == 4
            int shift = j & 0b111 ? 0 : 4;
            output[i] += __bitnetadd4(*(int8x4_t*)(input + j), 
                weight[addr] >> shift);
            #elif USE_SIMD == 8
            output[i] += __bitnetadd8(*(int8x4_t*)(input + j), 
                *(int8x4_t*)(input + j + 4), weight[addr]);
            #elif USE_SIMD == 16
            output[i] += __bitnetadd16(input + j, *(int1x16_t*)(weight + addr));
            #elif USE_SIMD == 32
            output[i] += __bitnetadd32(input + j, *(int1x32_t*)(weight + addr), 0);
            #elif USE_SIMD == 64
            output[i] += __bitnetadd64(
                input + j, 
                *(int1x32_t*)(weight + addr), 
                *(int1x32_t*)(weight + addr + 4)
            );
            #endif
            #else
            int addr = (i*n + j) >> 2;
            #if USE_SIMD == 4
            output[i] += __bitnetadd4(
                *(int8x4_t*)(input + j), 
                weight[addr]);
            #elif USE_SIMD == 8
            output[i] += __bitnetadd8(
                *(int8x4_t*)(input + j), 
                *(int8x4_t*)(input + j + 4), 
                *(uint16_t*)(weight + addr));
            #elif USE_SIMD == 16
            output[i] += __bitnetadd16(
                input + j, 
                *(int2x16_t*)(weight + addr));
            #elif USE_SIMD == 32
            output[i] += __bitnetadd32(input + j, 
                *(int2x16_t*)(weight+addr), 
                *(int2x16_t*)(weight+addr+4));
            #endif
            #endif
        }
    }
}

// Reversed 1x8, 1.5x8, 2x8 MatMul
void bitnet_qmatmul_rev(
        int8_t *input,  // 1, 1.5, 2-bit
        int32_t *output, 
        int8_t *weight, // 8-bit
        int n, int d){
    
    int addr;

    for (int i=0; i<d; i++){
        output[i] = 0;
        for(int j = 0; j < n; j += USE_SIMD){
            #if BITNET_QUANT == 2
            #else
            addr = i*n + j;
            #if USE_SIMD == 4
            output[i] += __bitnetadd4(
                *(int8x4_t*)(weight + addr), input[j]);
            #elif USE_SIMD == 8
            output[i] += __bitnetadd8(
                *(int8x4_t*)(weight + addr), 
                *(int8x4_t*)(weight + addr + 4), 
                *(int2x8_t*)(input + j));
            #elif USE_SIMD == 16
            output[i] += __bitnetadd16(
                weight + addr, 
                *(int2x16_t*) (input + j));
            #elif USE_SIMD == 32
            output[i] += __bitnetadd32(
                weight + addr, 
                *(int2x16_t*)(input + j), 
                *(int2x16_t*)(input + j + 4));
            #endif
            #endif
        }
    }
}

// 1-bit Quantization SIMD
#if BITNET_QUANT == 2
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (int i = 0; i < batch_size; i++){
        bitnet_qmatmul(
            x->data + i * in_features, // INT8 Activation
            O + i * out_features, // INT32 Output
            w->data, 
            in_features, 
            out_features
        );
    }
}

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (int i = 0; i < batch_size; i++){
        bitnet_qmatmul_rev(
            x->data + (i * in_features) / 8,
            O + i * out_features,
            w->data, 
            in_features, 
            out_features
        );
    }
}

#elif BITNET_QUANT != 0
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    #ifdef USE_DOTP8
    int8_t w_buf[4];
    int8_t temp_w;
    int32_t acc;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            for (size_t k = 0; k < in_features; k+=4) {
                temp_w = w->data[(j * in_features + k)/4];
                for (size_t l=0; l < 4; l++){
                    w_buf[l] = EXTRACT_2BIT(temp_w, l);
                    w_buf[l] = TWO_BIT_TO_INT8(w_buf[l]);
                }
                acc += __dotp8(
                    *(int32_t*) (x->data + i * in_features + k),
                    *(int32_t*) w_buf
                );
            }
            O[i * out_features + j] = acc;
        }
    }
    #else
    #ifdef USE_DOTP8X2
    int8_t temp_w;
    int32_t acc;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            for (size_t k = 0; k < in_features; k+=4) {
                temp_w = w->data[(j * in_features + k)/4];
                acc += __dotp8x2(
                    *(int32_t*) (x->data + i * in_features + k),
                    temp_w
                );
            }
            O[i * out_features + j] = acc;
        }
    }
    #else
    for (int i = 0; i < batch_size; i++){
        bitnet_qmatmul(
            x->data + i * in_features, // INT8 Activation
            O + i * out_features, // INT32 Output
            w->data, 
            in_features, 
            out_features
        );
    }
    #endif
    #endif
}

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (int i = 0; i < batch_size; i++){
        bitnet_qmatmul_rev(
            x->data + (i * in_features) / 4,
            O + i * out_features,
            w->data, 
            in_features, 
            out_features
        );
    }
}
#endif