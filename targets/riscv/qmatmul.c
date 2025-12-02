#include "mico_qnn.h"

// Optimized Mixed Precision MatMul Kernels for Regular RISC-V CPUs
// These implementations use loop unrolling and other software optimizations
// without requiring custom ISA extensions

#define MATMUL_UNROLL_FACTOR 4

// Optimized 8-bit x 8-bit MatMul with loop unrolling
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / MATMUL_UNROLL_FACTOR) * MATMUL_UNROLL_FACTOR;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features];
            const int8_t *w_row = &w->data[j * in_features];
            
            // Unrolled loop for better pipeline utilization
            for (size_t k = 0; k < unrolled_end; k += MATMUL_UNROLL_FACTOR) {
                acc += x_row[k] * w_row[k];
                acc += x_row[k+1] * w_row[k+1];
                acc += x_row[k+2] * w_row[k+2];
                acc += x_row[k+3] * w_row[k+3];
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                acc += x_row[k] * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 8-bit activation x 4-bit weight MatMul
void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 8) * 8;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features];
            const int8_t *w_row = &w->data[j * in_features / 2];
            
            // Process 8 elements at a time (4 bytes of packed weights)
            for (size_t k = 0; k < unrolled_end; k += 8) {
                int8_t w0 = w_row[k/2];
                int8_t w1 = w_row[k/2 + 1];
                int8_t w2 = w_row[k/2 + 2];
                int8_t w3 = w_row[k/2 + 3];
                
                acc += x_row[k]   * SIGN_EXTEND_TO_INT8(w0 & 0x0F, 4);
                acc += x_row[k+1] * SIGN_EXTEND_TO_INT8(w0 >> 4, 4);
                acc += x_row[k+2] * SIGN_EXTEND_TO_INT8(w1 & 0x0F, 4);
                acc += x_row[k+3] * SIGN_EXTEND_TO_INT8(w1 >> 4, 4);
                acc += x_row[k+4] * SIGN_EXTEND_TO_INT8(w2 & 0x0F, 4);
                acc += x_row[k+5] * SIGN_EXTEND_TO_INT8(w2 >> 4, 4);
                acc += x_row[k+6] * SIGN_EXTEND_TO_INT8(w3 & 0x0F, 4);
                acc += x_row[k+7] * SIGN_EXTEND_TO_INT8(w3 >> 4, 4);
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/2];
                temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += x_row[k] * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 8-bit activation x 2-bit weight MatMul
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 16) * 16;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features];
            const int8_t *w_row = &w->data[j * in_features / 4];
            
            // Process 16 elements at a time (4 bytes of packed weights)
            for (size_t k = 0; k < unrolled_end; k += 16) {
                int8_t wb0 = w_row[k/4];
                int8_t wb1 = w_row[k/4 + 1];
                int8_t wb2 = w_row[k/4 + 2];
                int8_t wb3 = w_row[k/4 + 3];
                
                acc += x_row[k]    * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 0));
                acc += x_row[k+1]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 1));
                acc += x_row[k+2]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 2));
                acc += x_row[k+3]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 3));
                acc += x_row[k+4]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 0));
                acc += x_row[k+5]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 1));
                acc += x_row[k+6]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 2));
                acc += x_row[k+7]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 3));
                acc += x_row[k+8]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 0));
                acc += x_row[k+9]  * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 1));
                acc += x_row[k+10] * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 2));
                acc += x_row[k+11] * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 3));
                acc += x_row[k+12] * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 0));
                acc += x_row[k+13] * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 1));
                acc += x_row[k+14] * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 2));
                acc += x_row[k+15] * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 3));
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                acc += x_row[k] * TWO_BIT_TO_INT8(temp_w);
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 8-bit activation x 1-bit weight MatMul
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 32) * 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features];
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Process 32 elements at a time (4 bytes of packed weights)
            for (size_t k = 0; k < unrolled_end; k += 32) {
                qword wb = *(qword*)(w_row + k/8);
                
                for (int b = 0; b < 32; b++) {
                    int8_t bit = (wb >> b) & 1;
                    acc += AMUX_1BIT(bit, x_row[k + b]);
                }
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                acc += AMUX_1BIT(temp_w, x_row[k]);
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 4-bit x 4-bit MatMul
void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 8) * 8;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 2];
            const int8_t *w_row = &w->data[j * in_features / 2];
            
            // Process 8 elements at a time (4 bytes each for x and w)
            for (size_t k = 0; k < unrolled_end; k += 8) {
                int8_t x0 = x_row[k/2];
                int8_t x1 = x_row[k/2 + 1];
                int8_t x2 = x_row[k/2 + 2];
                int8_t x3 = x_row[k/2 + 3];
                int8_t w0 = w_row[k/2];
                int8_t w1 = w_row[k/2 + 1];
                int8_t w2 = w_row[k/2 + 2];
                int8_t w3 = w_row[k/2 + 3];
                
                acc += SIGN_EXTEND_TO_INT8(x0 & 0x0F, 4) * SIGN_EXTEND_TO_INT8(w0 & 0x0F, 4);
                acc += SIGN_EXTEND_TO_INT8(x0 >> 4, 4)   * SIGN_EXTEND_TO_INT8(w0 >> 4, 4);
                acc += SIGN_EXTEND_TO_INT8(x1 & 0x0F, 4) * SIGN_EXTEND_TO_INT8(w1 & 0x0F, 4);
                acc += SIGN_EXTEND_TO_INT8(x1 >> 4, 4)   * SIGN_EXTEND_TO_INT8(w1 >> 4, 4);
                acc += SIGN_EXTEND_TO_INT8(x2 & 0x0F, 4) * SIGN_EXTEND_TO_INT8(w2 & 0x0F, 4);
                acc += SIGN_EXTEND_TO_INT8(x2 >> 4, 4)   * SIGN_EXTEND_TO_INT8(w2 >> 4, 4);
                acc += SIGN_EXTEND_TO_INT8(x3 & 0x0F, 4) * SIGN_EXTEND_TO_INT8(w3 & 0x0F, 4);
                acc += SIGN_EXTEND_TO_INT8(x3 >> 4, 4)   * SIGN_EXTEND_TO_INT8(w3 >> 4, 4);
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 4-bit activation x 2-bit weight MatMul
void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 16) * 16;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 2];
            const int8_t *w_row = &w->data[j * in_features / 4];
            
            // Process 16 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 16) {
                // 8 bytes of 4-bit activations = 16 elements
                int8_t x0 = x_row[k/2];
                int8_t x1 = x_row[k/2 + 1];
                int8_t x2 = x_row[k/2 + 2];
                int8_t x3 = x_row[k/2 + 3];
                int8_t x4 = x_row[k/2 + 4];
                int8_t x5 = x_row[k/2 + 5];
                int8_t x6 = x_row[k/2 + 6];
                int8_t x7 = x_row[k/2 + 7];
                // 4 bytes of 2-bit weights = 16 elements
                int8_t wb0 = w_row[k/4];
                int8_t wb1 = w_row[k/4 + 1];
                int8_t wb2 = w_row[k/4 + 2];
                int8_t wb3 = w_row[k/4 + 3];
                
                acc += SIGN_EXTEND_TO_INT8(x0 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 0));
                acc += SIGN_EXTEND_TO_INT8(x0 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 1));
                acc += SIGN_EXTEND_TO_INT8(x1 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 2));
                acc += SIGN_EXTEND_TO_INT8(x1 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb0, 3));
                acc += SIGN_EXTEND_TO_INT8(x2 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 0));
                acc += SIGN_EXTEND_TO_INT8(x2 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 1));
                acc += SIGN_EXTEND_TO_INT8(x3 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 2));
                acc += SIGN_EXTEND_TO_INT8(x3 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb1, 3));
                acc += SIGN_EXTEND_TO_INT8(x4 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 0));
                acc += SIGN_EXTEND_TO_INT8(x4 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 1));
                acc += SIGN_EXTEND_TO_INT8(x5 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 2));
                acc += SIGN_EXTEND_TO_INT8(x5 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb2, 3));
                acc += SIGN_EXTEND_TO_INT8(x6 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 0));
                acc += SIGN_EXTEND_TO_INT8(x6 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 1));
                acc += SIGN_EXTEND_TO_INT8(x7 & 0x0F, 4) * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 2));
                acc += SIGN_EXTEND_TO_INT8(x7 >> 4, 4)   * TWO_BIT_TO_INT8(EXTRACT_2BIT(wb3, 3));
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 4-bit activation x 1-bit weight MatMul
void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 32) * 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 2];
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Process 32 elements at a time (16 bytes of 4-bit x, 4 bytes of 1-bit w)
            for (size_t k = 0; k < unrolled_end; k += 32) {
                qword wb = *(qword*)(w_row + k/8);
                
                for (int b = 0; b < 32; b++) {
                    int8_t temp_a = x_row[(k + b)/2];
                    temp_a = EXTRACT_4BIT(temp_a, b & 0b1);
                    temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                    int8_t bit = (wb >> b) & 1;
                    acc += temp_a * BIT_TO_INT8(bit);
                }
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit x 2-bit MatMul
void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 16) * 16;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 4];
            const int8_t *w_row = &w->data[j * in_features / 4];
            
            // Process 16 elements at a time (4 bytes each)
            for (size_t k = 0; k < unrolled_end; k += 16) {
                int8_t x0 = x_row[k/4];
                int8_t x1 = x_row[k/4 + 1];
                int8_t x2 = x_row[k/4 + 2];
                int8_t x3 = x_row[k/4 + 3];
                int8_t w0 = w_row[k/4];
                int8_t w1 = w_row[k/4 + 1];
                int8_t w2 = w_row[k/4 + 2];
                int8_t w3 = w_row[k/4 + 3];
                
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 0)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w0, 0));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 1)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w0, 1));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 2)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w0, 2));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 3)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w0, 3));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 0)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w1, 0));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 1)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w1, 1));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 2)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w1, 2));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 3)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w1, 3));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 0)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w2, 0));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 1)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w2, 1));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 2)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w2, 2));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 3)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w2, 3));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 0)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w3, 0));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 1)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w3, 1));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 2)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w3, 2));
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 3)) * TWO_BIT_TO_INT8(EXTRACT_2BIT(w3, 3));
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit activation x 1-bit weight MatMul
void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 32) * 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 4];
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Process 32 elements at a time (8 bytes of 2-bit x, 4 bytes of 1-bit w)
            for (size_t k = 0; k < unrolled_end; k += 32) {
                qword wb = *(qword*)(w_row + k/8);
                
                for (int b = 0; b < 32; b++) {
                    int8_t temp_a = x_row[(k + b)/4];
                    temp_a = EXTRACT_2BIT(temp_a, b & 0b11);
                    temp_a = TWO_BIT_TO_INT8(temp_a);
                    int8_t bit = (wb >> b) & 1;
                    acc += temp_a * BIT_TO_INT8(bit);
                }
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_w = w_row[k/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit x 1-bit MatMul (Binary Neural Network)
void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const qword *x_row = (const qword*)&x->data[i * in_features / 8];
            const qword *w_row = (const qword*)&w->data[j * in_features / 8];
            
            // Use XOR + popcount for binary neural networks
            for (size_t k = 0; k < word_count; k++) {
                qword xnor_result = ~(x_row[k] ^ w_row[k]);
                int popcount = __builtin_popcount(xnor_result);
                // XNOR gives 1 when bits match: +1 for match, -1 for mismatch
                // popcount = number of matches
                // acc += 2*popcount - 32 = (matches - mismatches)
                acc += 2 * popcount - 32;
            }
            
            // Handle remaining bits
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                int8_t temp_a = x->data[(i * in_features + k)/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// -----------------------------------------------------------------------------
// Reversed precision operations (weight precision > activation precision)
// -----------------------------------------------------------------------------

// 4-bit activation x 8-bit weight MatMul
void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 8) * 8;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 2];
            const int8_t *w_row = &w->data[j * in_features];
            
            // Process 8 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 8) {
                int8_t x0 = x_row[k/2];
                int8_t x1 = x_row[k/2 + 1];
                int8_t x2 = x_row[k/2 + 2];
                int8_t x3 = x_row[k/2 + 3];
                
                acc += SIGN_EXTEND_TO_INT8(x0 & 0x0F, 4) * w_row[k];
                acc += SIGN_EXTEND_TO_INT8(x0 >> 4, 4)   * w_row[k+1];
                acc += SIGN_EXTEND_TO_INT8(x1 & 0x0F, 4) * w_row[k+2];
                acc += SIGN_EXTEND_TO_INT8(x1 >> 4, 4)   * w_row[k+3];
                acc += SIGN_EXTEND_TO_INT8(x2 & 0x0F, 4) * w_row[k+4];
                acc += SIGN_EXTEND_TO_INT8(x2 >> 4, 4)   * w_row[k+5];
                acc += SIGN_EXTEND_TO_INT8(x3 & 0x0F, 4) * w_row[k+6];
                acc += SIGN_EXTEND_TO_INT8(x3 >> 4, 4)   * w_row[k+7];
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit activation x 8-bit weight MatMul
void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 16) * 16;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 4];
            const int8_t *w_row = &w->data[j * in_features];
            
            // Process 16 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 16) {
                int8_t x0 = x_row[k/4];
                int8_t x1 = x_row[k/4 + 1];
                int8_t x2 = x_row[k/4 + 2];
                int8_t x3 = x_row[k/4 + 3];
                
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 0)) * w_row[k];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 1)) * w_row[k+1];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 2)) * w_row[k+2];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 3)) * w_row[k+3];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 0)) * w_row[k+4];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 1)) * w_row[k+5];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 2)) * w_row[k+6];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 3)) * w_row[k+7];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 0)) * w_row[k+8];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 1)) * w_row[k+9];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 2)) * w_row[k+10];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 3)) * w_row[k+11];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 0)) * w_row[k+12];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 1)) * w_row[k+13];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 2)) * w_row[k+14];
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 3)) * w_row[k+15];
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc += temp_a * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit activation x 8-bit weight MatMul
void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 32) * 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 8];
            const int8_t *w_row = &w->data[j * in_features];
            
            // Process 32 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 32) {
                qword xb = *(qword*)(x_row + k/8);
                
                for (int b = 0; b < 32; b++) {
                    int8_t bit = (xb >> b) & 1;
                    acc += BIT_TO_INT8(bit) * w_row[k + b];
                }
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_a = x_row[k/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc += temp_a * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit activation x 4-bit weight MatMul
void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 16) * 16;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 4];
            const int8_t *w_row = &w->data[j * in_features / 2];
            
            // Process 16 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 16) {
                int8_t x0 = x_row[k/4];
                int8_t x1 = x_row[k/4 + 1];
                int8_t x2 = x_row[k/4 + 2];
                int8_t x3 = x_row[k/4 + 3];
                int8_t w0 = w_row[k/2];
                int8_t w1 = w_row[k/2 + 1];
                int8_t w2 = w_row[k/2 + 2];
                int8_t w3 = w_row[k/2 + 3];
                int8_t w4 = w_row[k/2 + 4];
                int8_t w5 = w_row[k/2 + 5];
                int8_t w6 = w_row[k/2 + 6];
                int8_t w7 = w_row[k/2 + 7];
                
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 0)) * SIGN_EXTEND_TO_INT8(w0 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 1)) * SIGN_EXTEND_TO_INT8(w0 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 2)) * SIGN_EXTEND_TO_INT8(w1 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x0, 3)) * SIGN_EXTEND_TO_INT8(w1 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 0)) * SIGN_EXTEND_TO_INT8(w2 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 1)) * SIGN_EXTEND_TO_INT8(w2 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 2)) * SIGN_EXTEND_TO_INT8(w3 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x1, 3)) * SIGN_EXTEND_TO_INT8(w3 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 0)) * SIGN_EXTEND_TO_INT8(w4 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 1)) * SIGN_EXTEND_TO_INT8(w4 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 2)) * SIGN_EXTEND_TO_INT8(w5 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x2, 3)) * SIGN_EXTEND_TO_INT8(w5 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 0)) * SIGN_EXTEND_TO_INT8(w6 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 1)) * SIGN_EXTEND_TO_INT8(w6 >> 4, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 2)) * SIGN_EXTEND_TO_INT8(w7 & 0x0F, 4);
                acc += TWO_BIT_TO_INT8(EXTRACT_2BIT(x3, 3)) * SIGN_EXTEND_TO_INT8(w7 >> 4, 4);
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                int8_t temp_w = w_row[k/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit activation x 4-bit weight MatMul
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 32) * 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 8];
            const int8_t *w_row = &w->data[j * in_features / 2];
            
            // Process 32 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 32) {
                qword xb = *(qword*)(x_row + k/8);
                
                for (int b = 0; b < 32; b++) {
                    int8_t temp_w = w_row[(k + b)/2];
                    temp_w = EXTRACT_4BIT(temp_w, b & 0b1);
                    temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                    int8_t bit = (xb >> b) & 1;
                    acc += BIT_TO_INT8(bit) * temp_w;
                }
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_a = x_row[k/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                int8_t temp_w = w_row[k/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit activation x 2-bit weight MatMul
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t unrolled_end = (in_features / 32) * 32;

    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *x_row = &x->data[i * in_features / 8];
            const int8_t *w_row = &w->data[j * in_features / 4];
            
            // Process 32 elements at a time
            for (size_t k = 0; k < unrolled_end; k += 32) {
                qword xb = *(qword*)(x_row + k/8);
                
                for (int b = 0; b < 32; b++) {
                    int8_t temp_w = w_row[(k + b)/4];
                    temp_w = EXTRACT_2BIT(temp_w, b & 0b11);
                    temp_w = TWO_BIT_TO_INT8(temp_w);
                    int8_t bit = (xb >> b) & 1;
                    acc += BIT_TO_INT8(bit) * temp_w;
                }
            }
            
            // Handle remaining elements
            for (size_t k = unrolled_end; k < in_features; k++) {
                int8_t temp_a = x_row[k/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}
