#include "mico_qnn.h"

// Baseline Implementation of 8-bit MatMul
// This is the most intensive kernel that you may want to optimize
__attribute__((weak)) void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            for (size_t k = 0; k < in_features; k++) {
                O[i * out_features + j] += x->data[i * in_features + k] * \
                    w->data[j * in_features + k];
            }
        }
    }
}