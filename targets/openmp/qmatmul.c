#include "mico_qnn.h"
#include <omp.h>

// Optimized 8-bit matrix multiplication using OpenMP
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Parallelize the batch dimension for best cache coherence
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t* x_row = &x->data[i * in_features];
            const int8_t* w_row = &w->data[j * in_features];
            
            // Loop unrolling for better pipeline utilization
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                acc += x_row[k] * w_row[k];
                acc += x_row[k+1] * w_row[k+1];
                acc += x_row[k+2] * w_row[k+2];
                acc += x_row[k+3] * w_row[k+3];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                acc += x_row[k] * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Parallelize across batches and output features for smaller datasets
    #pragma omp parallel for collapse(2) schedule(guided)
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            
            // Pre-calculate base offsets for better cache performance
            const size_t x_base = i * in_features;
            const size_t w_base = j * in_features;
            
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(w_base + k)/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                
                int8_t temp_a = x->data[(x_base + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Use dynamic scheduling for potentially imbalanced workloads
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < batch_size; i++) {
        // Thread-local temporary storage for results
        int32_t* thread_results = (int32_t*)malloc(out_features * sizeof(int32_t));
        
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            
            // Process multiple elements per iteration when possible
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                for (size_t m = 0; m < 4; m++) {
                    int8_t temp_w = w->data[(j * in_features + k + m)/4];
                    temp_w = EXTRACT_2BIT(temp_w, (k+m) & 0b11);
                    temp_w = TWO_BIT_TO_INT8(temp_w);
                    
                    int8_t temp_a = x->data[(i * in_features + k + m)/4];
                    temp_a = EXTRACT_2BIT(temp_a, (k+m) & 0b11);
                    temp_a = TWO_BIT_TO_INT8(temp_a);
                    
                    acc += temp_a * temp_w;
                }
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                
                int8_t temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                
                acc += temp_a * temp_w;
            }
            
            thread_results[j] = acc;
        }
        
        // Copy thread-local results to output
        #pragma omp critical
        {
            for (size_t j = 0; j < out_features; j++) {
                O[i * out_features + j] = thread_results[j];
            }
        }
        
        free(thread_results);
    }
}

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // BNN benefits from large block processing - use static scheduling with large chunks
    #pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            
            // Block processing - handle 16 elements at once when possible
            size_t k = 0;
            for (; k + 16 <= in_features; k += 16) {
                int32_t block_acc = 0;
                for (size_t m = 0; m < 16; m++) {
                    int8_t temp_w = w->data[(j * in_features + k + m)/8];
                    temp_w = EXTRACT_BIT(temp_w, (k+m) & 0b111);
                    temp_w = BIT_TO_INT8(temp_w);
                    
                    int8_t temp_a = x->data[(i * in_features + k + m)/8];
                    temp_a = EXTRACT_BIT(temp_a, (k+m) & 0b111);
                    temp_a = BIT_TO_INT8(temp_a);
                    
                    // For 1-bit operations, XNOR + popcount would be faster
                    block_acc += temp_a * temp_w;
                }
                acc += block_acc;
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
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