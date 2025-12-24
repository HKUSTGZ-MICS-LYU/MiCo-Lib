#define MiCo_Q8_MatMul MiCo_Q8_MatMul_unroll
#define MiCo_Q4_MatMul MiCo_Q4_MatMul_unroll
#include "mico_qnn.h"

// Actually meaningless for now
#define MATMUL_UNROLL_FACTOR 4 

// Unrolled Implementation of 8-bit MatMul
// The most simple optimization to apply on MatMul
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Check if it is possible to unroll
    size_t unrolled_end = (in_features / MATMUL_UNROLL_FACTOR) * MATMUL_UNROLL_FACTOR;
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < unrolled_end; k+=MATMUL_UNROLL_FACTOR) {
                sum += x->data[i * in_features + k] * \
                    w->data[j * in_features + k];
                sum += x->data[i * in_features + k+1] * \
                    w->data[j * in_features + k+1];
                sum += x->data[i * in_features + k+2] * \
                    w->data[j * in_features + k+2];
                sum += x->data[i * in_features + k+3] * \
                    w->data[j * in_features + k+3];
            }
            for (size_t k = unrolled_end; k < in_features; k++) {
                sum += x->data[i * in_features + k] * w->data[j * in_features + k];
            }
            O[i * out_features + j] += sum;
        }
    }
}

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    // Require in_features to be even
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            acc = 0;
            size_t w_base = j * in_features / 2;
            size_t x_base = i * in_features / 2;

            for (size_t k = 0; k < in_features / 2; k++) {

                temp_w = w->data[w_base + k];
                temp_a = x->data[x_base + k];

                int8_t low_w = SIGN_EXTEND_TO_INT8(EXTRACT_4BIT(temp_w, 0), 4);
                int8_t high_w = SIGN_EXTEND_TO_INT8(EXTRACT_4BIT(temp_w, 1), 4);
                int8_t low_a = SIGN_EXTEND_TO_INT8(EXTRACT_4BIT(temp_a, 0), 4);
                int8_t high_a = SIGN_EXTEND_TO_INT8(EXTRACT_4BIT(temp_a, 1), 4);

                acc += low_a * low_w;
                acc += high_a * high_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}
