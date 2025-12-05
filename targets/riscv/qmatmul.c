#include "mico_qnn.h"

// Optimized Mixed Precision MatMul Kernels for Regular RISC-V CPUs
// These implementations use loop unrolling and other software optimizations
// without requiring custom ISA extensions

#define MATMUL_UNROLL_FACTOR 4

// Software popcount implementation for portability
// Uses parallel bit counting algorithm (Brian Kernighan's method variant)
static inline int software_popcount(qword x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
}

// Software count trailing zeros (ctz) implementation for portability
// Returns the number of trailing zero bits in x (0-32)
static inline int software_ctz(qword x) {
    if (x == 0) return 32;
    int n = 0;
    if ((x & 0x0000FFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0x000000FF) == 0) { n += 8;  x >>= 8;  }
    if ((x & 0x0000000F) == 0) { n += 4;  x >>= 4;  }
    if ((x & 0x00000003) == 0) { n += 2;  x >>= 2;  }
    if ((x & 0x00000001) == 0) { n += 1; }
    return n;
}

// Use compiler builtin if available, otherwise use software implementation
#ifdef __GNUC__
#define CTZ(x) __builtin_ctz(x)
#else
#define CTZ(x) software_ctz(x)
#endif

// Helper to safely load a qword handling potential alignment issues
static inline qword safe_load_qword(const int8_t *ptr) {
    qword result;
    // Use byte-by-byte copy to avoid unaligned access issues
    const uint8_t *src = (const uint8_t *)ptr;
    result = (qword)src[0] | ((qword)src[1] << 8) | 
             ((qword)src[2] << 16) | ((qword)src[3] << 24);
    return result;
}

// Optimized 8-bit x 8-bit MatMul with 5x row unrolling (inspired by muriscv-nn)
// This technique exposes more instruction-level parallelism by processing
// 5 output rows simultaneously, reducing loop overhead and enabling better
// register utilization.
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t col_unrolled_end = (in_features / MATMUL_UNROLL_FACTOR) * MATMUL_UNROLL_FACTOR;

    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        // Process 5 output rows at a time for better ILP
        const size_t row_loop_cnt = out_features / 5;
        size_t j = 0;
        
        for (size_t j_loop = 0; j_loop < row_loop_cnt; j_loop++) {
            const int8_t *w_row_0 = &w->data[j * in_features];
            const int8_t *w_row_1 = &w->data[(j + 1) * in_features];
            const int8_t *w_row_2 = &w->data[(j + 2) * in_features];
            const int8_t *w_row_3 = &w->data[(j + 3) * in_features];
            const int8_t *w_row_4 = &w->data[(j + 4) * in_features];
            
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
            
            // 4x column unrolling with 5x row unrolling
            for (size_t k = 0; k < col_unrolled_end; k += MATMUL_UNROLL_FACTOR) {
                const int8_t x0 = x_row[k];
                const int8_t x1 = x_row[k+1];
                const int8_t x2 = x_row[k+2];
                const int8_t x3 = x_row[k+3];
                
                acc0 += x0 * w_row_0[k] + x1 * w_row_0[k+1] + x2 * w_row_0[k+2] + x3 * w_row_0[k+3];
                acc1 += x0 * w_row_1[k] + x1 * w_row_1[k+1] + x2 * w_row_1[k+2] + x3 * w_row_1[k+3];
                acc2 += x0 * w_row_2[k] + x1 * w_row_2[k+1] + x2 * w_row_2[k+2] + x3 * w_row_2[k+3];
                acc3 += x0 * w_row_3[k] + x1 * w_row_3[k+1] + x2 * w_row_3[k+2] + x3 * w_row_3[k+3];
                acc4 += x0 * w_row_4[k] + x1 * w_row_4[k+1] + x2 * w_row_4[k+2] + x3 * w_row_4[k+3];
            }
            
            // Handle remaining columns
            for (size_t k = col_unrolled_end; k < in_features; k++) {
                const int8_t xv = x_row[k];
                acc0 += xv * w_row_0[k];
                acc1 += xv * w_row_1[k];
                acc2 += xv * w_row_2[k];
                acc3 += xv * w_row_3[k];
                acc4 += xv * w_row_4[k];
            }
            
            O[i * out_features + j] = acc0;
            O[i * out_features + j + 1] = acc1;
            O[i * out_features + j + 2] = acc2;
            O[i * out_features + j + 3] = acc3;
            O[i * out_features + j + 4] = acc4;
            j += 5;
        }
        
        // Handle remaining output rows
        for (; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t *w_row = &w->data[j * in_features];
            
            for (size_t k = 0; k < col_unrolled_end; k += MATMUL_UNROLL_FACTOR) {
                acc += x_row[k] * w_row[k];
                acc += x_row[k+1] * w_row[k+1];
                acc += x_row[k+2] * w_row[k+2];
                acc += x_row[k+3] * w_row[k+3];
            }
            
            for (size_t k = col_unrolled_end; k < in_features; k++) {
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
// Optimized: For 1-bit weights, w_i is either +1 (bit=0) or -1 (bit=1)
// sum(x_i * w_i) = sum(x_i where bit=0) - sum(x_i where bit=1)
//               = total_sum - 2 * sum(x_i where bit=1)
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        // Pre-compute the sum of all activations for this batch
        int32_t total_sum = 0;
        const int8_t *x_row = &x->data[i * in_features];
        
        // Compute total sum with unrolling
        size_t sum_k;
        for (sum_k = 0; sum_k + 8 <= in_features; sum_k += 8) {
            total_sum += x_row[sum_k] + x_row[sum_k+1] + x_row[sum_k+2] + x_row[sum_k+3];
            total_sum += x_row[sum_k+4] + x_row[sum_k+5] + x_row[sum_k+6] + x_row[sum_k+7];
        }
        for (; sum_k < in_features; sum_k++) {
            total_sum += x_row[sum_k];
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Compute sum of x_i where w_i bit = 1 (meaning w_i = -1)
            int32_t neg_sum = 0;
            
            // Process 32 elements at a time using word operations
            for (size_t wk = 0; wk < word_count; wk++) {
                qword wb = safe_load_qword(w_row + wk * 4);
                size_t base = wk * 32;
                
                // For each set bit in wb, add the corresponding x value
                while (wb) {
                    int bit_pos = CTZ(wb);  // Find lowest set bit
                    neg_sum += x_row[base + bit_pos];
                    wb &= wb - 1;  // Clear lowest set bit
                }
            }
            
            // Handle remaining elements
            size_t remaining_start = word_count * 32;
            for (size_t rem_k = remaining_start; rem_k < in_features; rem_k++) {
                int8_t temp_w = w_row[rem_k/8];
                if (EXTRACT_BIT(temp_w, rem_k & 0b111)) {
                    neg_sum += x_row[rem_k];
                }
            }
            
            // Final result: total_sum - 2 * neg_sum
            O[i * out_features + j] = total_sum - 2 * neg_sum;
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
// Optimized: For 1-bit weights, w_i is either +1 (bit=0) or -1 (bit=1)
// sum(x_i * w_i) = sum(x_i where bit=0) - sum(x_i where bit=1)
//               = total_sum - 2 * sum(x_i where bit=1)
void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        // Pre-compute the sum of all 4-bit activations for this batch
        int32_t total_sum = 0;
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        // Compute total sum - process bytes (2 elements per byte)
        for (size_t k = 0; k < in_features / 2; k++) {
            int8_t byte = x_row[k];
            total_sum += SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
            total_sum += SIGN_EXTEND_TO_INT8(byte >> 4, 4);
        }
        // Handle odd element count
        if (in_features & 1) {
            int8_t byte = x_row[in_features / 2];
            total_sum += SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Compute sum of x_i where w_i bit = 1 (meaning w_i = -1)
            int32_t neg_sum = 0;
            
            // Process 32 elements at a time using word operations
            for (size_t wk = 0; wk < word_count; wk++) {
                qword wb = safe_load_qword(w_row + wk * 4);
                size_t base = wk * 32;
                
                // For each set bit in wb, add the corresponding x value
                while (wb) {
                    int bit_pos = CTZ(wb);  // Find lowest set bit
                    size_t idx = base + bit_pos;
                    int8_t byte = x_row[idx / 2];
                    int8_t val = (idx & 1) ? SIGN_EXTEND_TO_INT8(byte >> 4, 4) 
                                           : SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
                    neg_sum += val;
                    wb &= wb - 1;  // Clear lowest set bit
                }
            }
            
            // Handle remaining elements
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                int8_t temp_w = w_row[k/8];
                if (EXTRACT_BIT(temp_w, k & 0b111)) {
                    int8_t byte = x_row[k / 2];
                    int8_t val = (k & 1) ? SIGN_EXTEND_TO_INT8(byte >> 4, 4) 
                                         : SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
                    neg_sum += val;
                }
            }
            
            // Final result: total_sum - 2 * neg_sum
            O[i * out_features + j] = total_sum - 2 * neg_sum;
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
// Optimized: For 1-bit weights, w_i is either +1 (bit=0) or -1 (bit=1)
// sum(x_i * w_i) = sum(x_i where bit=0) - sum(x_i where bit=1)
//               = total_sum - 2 * sum(x_i where bit=1)
void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        // Pre-compute the sum of all 2-bit activations for this batch
        int32_t total_sum = 0;
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        // Compute total sum - process bytes (4 elements per byte)
        for (size_t k = 0; k < in_features / 4; k++) {
            int8_t byte = x_row[k];
            total_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 0));
            total_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 1));
            total_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 2));
            total_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 3));
        }
        // Handle remaining elements
        size_t remaining = in_features & 3;
        if (remaining) {
            int8_t byte = x_row[in_features / 4];
            for (size_t r = 0; r < remaining; r++) {
                total_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, r));
            }
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Compute sum of x_i where w_i bit = 1 (meaning w_i = -1)
            int32_t neg_sum = 0;
            
            // Process 32 elements at a time using word operations
            for (size_t wk = 0; wk < word_count; wk++) {
                qword wb = safe_load_qword(w_row + wk * 4);
                size_t base = wk * 32;
                
                // For each set bit in wb, add the corresponding x value
                while (wb) {
                    int bit_pos = CTZ(wb);  // Find lowest set bit
                    size_t idx = base + bit_pos;
                    int8_t byte = x_row[idx / 4];
                    int8_t val = TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, idx & 0b11));
                    neg_sum += val;
                    wb &= wb - 1;  // Clear lowest set bit
                }
            }
            
            // Handle remaining elements
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                int8_t temp_w = w_row[k/8];
                if (EXTRACT_BIT(temp_w, k & 0b111)) {
                    int8_t byte = x_row[k / 4];
                    int8_t val = TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, k & 0b11));
                    neg_sum += val;
                }
            }
            
            // Final result: total_sum - 2 * neg_sum
            O[i * out_features + j] = total_sum - 2 * neg_sum;
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
            const int8_t *x_row = &x->data[i * in_features / 8];
            const int8_t *w_row = &w->data[j * in_features / 8];
            
            // Use XOR + popcount for binary neural networks
            for (size_t k = 0; k < word_count; k++) {
                qword x_word = safe_load_qword(x_row + k * 4);
                qword w_word = safe_load_qword(w_row + k * 4);
                qword xnor_result = ~(x_word ^ w_word);
                int popcount = software_popcount(xnor_result);
                // XNOR gives 1 when bits match: +1 for match, -1 for mismatch
                // popcount = number of matches
                // acc += 2*popcount - 32 = (matches - mismatches)
                acc += 2 * popcount - 32;
            }
            
            // Handle remaining bits using row pointers
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                size_t bit_offset = k - remaining_start;
                int8_t temp_w = w_row[(remaining_start + bit_offset)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                int8_t temp_a = x_row[(remaining_start + bit_offset)/8];
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
// Optimized: For 1-bit activations, x_i is either +1 (bit=0) or -1 (bit=1)
// sum(x_i * w_i) = sum(w_i where bit=0) - sum(w_i where bit=1)
//               = total_w_sum - 2 * sum(w_i where bit=1)
void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            
            // Compute sum of all weights and sum of weights where x bit = 1
            int32_t total_w_sum = 0;
            int32_t neg_w_sum = 0;
            
            // Process 32 elements at a time using word operations
            for (size_t wk = 0; wk < word_count; wk++) {
                qword xb = safe_load_qword(x_row + wk * 4);
                size_t base = wk * 32;
                
                // Sum weights with unrolling
                for (int b = 0; b < 32; b += 8) {
                    total_w_sum += w_row[base + b] + w_row[base + b + 1] + 
                                   w_row[base + b + 2] + w_row[base + b + 3];
                    total_w_sum += w_row[base + b + 4] + w_row[base + b + 5] + 
                                   w_row[base + b + 6] + w_row[base + b + 7];
                }
                
                // For each set bit in xb, add the corresponding w value
                while (xb) {
                    int bit_pos = CTZ(xb);  // Find lowest set bit
                    neg_w_sum += w_row[base + bit_pos];
                    xb &= xb - 1;  // Clear lowest set bit
                }
            }
            
            // Handle remaining elements
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                total_w_sum += w_row[k];
                int8_t temp_a = x_row[k/8];
                if (EXTRACT_BIT(temp_a, k & 0b111)) {
                    neg_w_sum += w_row[k];
                }
            }
            
            // Final result: total_w_sum - 2 * neg_w_sum
            O[i * out_features + j] = total_w_sum - 2 * neg_w_sum;
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
// Optimized: For 1-bit activations, x_i is either +1 (bit=0) or -1 (bit=1)
// sum(x_i * w_i) = sum(w_i where bit=0) - sum(w_i where bit=1)
//               = total_w_sum - 2 * sum(w_i where bit=1)
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            
            // Compute sum of all 4-bit weights
            int32_t total_w_sum = 0;
            for (size_t k = 0; k < in_features / 2; k++) {
                int8_t byte = w_row[k];
                total_w_sum += SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
                total_w_sum += SIGN_EXTEND_TO_INT8(byte >> 4, 4);
            }
            if (in_features & 1) {
                int8_t byte = w_row[in_features / 2];
                total_w_sum += SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
            }
            
            // Compute sum of weights where x bit = 1
            int32_t neg_w_sum = 0;
            
            // Process 32 elements at a time using word operations
            for (size_t wk = 0; wk < word_count; wk++) {
                qword xb = safe_load_qword(x_row + wk * 4);
                size_t base = wk * 32;
                
                // For each set bit in xb, add the corresponding w value
                while (xb) {
                    int bit_pos = CTZ(xb);  // Find lowest set bit
                    size_t idx = base + bit_pos;
                    int8_t byte = w_row[idx / 2];
                    int8_t val = (idx & 1) ? SIGN_EXTEND_TO_INT8(byte >> 4, 4)
                                           : SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
                    neg_w_sum += val;
                    xb &= xb - 1;  // Clear lowest set bit
                }
            }
            
            // Handle remaining elements
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                int8_t temp_a = x_row[k/8];
                if (EXTRACT_BIT(temp_a, k & 0b111)) {
                    int8_t byte = w_row[k / 2];
                    int8_t val = (k & 1) ? SIGN_EXTEND_TO_INT8(byte >> 4, 4)
                                         : SIGN_EXTEND_TO_INT8(byte & 0x0F, 4);
                    neg_w_sum += val;
                }
            }
            
            // Final result: total_w_sum - 2 * neg_w_sum
            O[i * out_features + j] = total_w_sum - 2 * neg_w_sum;
        }
    }
}

// 1-bit activation x 2-bit weight MatMul
// Optimized: For 1-bit activations, x_i is either +1 (bit=0) or -1 (bit=1)
// sum(x_i * w_i) = sum(w_i where bit=0) - sum(w_i where bit=1)
//               = total_w_sum - 2 * sum(w_i where bit=1)
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    const size_t word_count = in_features / 32;

    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            
            // Compute sum of all 2-bit weights
            int32_t total_w_sum = 0;
            for (size_t k = 0; k < in_features / 4; k++) {
                int8_t byte = w_row[k];
                total_w_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 0));
                total_w_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 1));
                total_w_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 2));
                total_w_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, 3));
            }
            size_t remaining = in_features & 3;
            if (remaining) {
                int8_t byte = w_row[in_features / 4];
                for (size_t r = 0; r < remaining; r++) {
                    total_w_sum += TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, r));
                }
            }
            
            // Compute sum of weights where x bit = 1
            int32_t neg_w_sum = 0;
            
            // Process 32 elements at a time using word operations
            for (size_t wk = 0; wk < word_count; wk++) {
                qword xb = safe_load_qword(x_row + wk * 4);
                size_t base = wk * 32;
                
                // For each set bit in xb, add the corresponding w value
                while (xb) {
                    int bit_pos = CTZ(xb);  // Find lowest set bit
                    size_t idx = base + bit_pos;
                    int8_t byte = w_row[idx / 4];
                    int8_t val = TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, idx & 0b11));
                    neg_w_sum += val;
                    xb &= xb - 1;  // Clear lowest set bit
                }
            }
            
            // Handle remaining elements
            size_t remaining_start = word_count * 32;
            for (size_t k = remaining_start; k < in_features; k++) {
                int8_t temp_a = x_row[k/8];
                if (EXTRACT_BIT(temp_a, k & 0b111)) {
                    int8_t byte = w_row[k / 4];
                    int8_t val = TWO_BIT_TO_INT8(EXTRACT_2BIT(byte, k & 0b11));
                    neg_w_sum += val;
                }
            }
            
            // Final result: total_w_sum - 2 * neg_w_sum
            O[i * out_features + j] = total_w_sum - 2 * neg_w_sum;
        }
    }
}
