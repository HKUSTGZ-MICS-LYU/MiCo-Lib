#include "mico_qnn.h"


// Baseline Implementation of MatMuls
// This is the most intensive kernel that you may want to optimize
__attribute__((weak)) void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int32_t acc;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                acc += x->data[i * in_features + k] * \
                    w->data[j * in_features + k];
            }
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                // TODO: What will happen if k is odd?
                temp_w = w->data[(j * in_features + k)/2];
                temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += x->data[i * in_features + k] * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                // TODO: What will happen if k is odd?
                temp_w = w->data[(j * in_features + k)/4];
                temp_w = temp_w >> (((j * in_features + k) & 0b11) << 1) & 0x03;
                acc += AMUX_2BIT(temp_w, x->data[i * in_features + k]);
            }
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                // TODO: What will happen if k is odd?
                temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, (j * in_features + k) & 0b111);
                acc += AMUX_1BIT(temp_w, x->data[i * in_features + k]);
            }
            O[i * out_features + j] = acc;
        }
    }
}