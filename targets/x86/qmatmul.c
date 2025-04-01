#include "mico_qnn.h"
#include "profile.h"
#include <immintrin.h> // For AVX/SSE intrinsics
#include <stdbool.h>

// Helper functions for bit extraction
static inline __m256i extract_1bit_values(const int8_t* data, size_t base_idx, size_t count) {
    __m256i result = _mm256_setzero_si256();
    int8_t extracted[32];
    
    for (size_t i = 0; i < count; i++) {
        size_t bit_idx = base_idx + i;
        size_t byte_idx = bit_idx / 8;
        size_t bit_pos = bit_idx % 8;
        
        int8_t bit_val = (data[byte_idx] >> bit_pos) & 1;
        // Convert to -1/1
        extracted[i] = bit_val ? -1 : 1;
    }
    return _mm256_loadu_si256((__m256i*)extracted);
}

static inline __m256i extract_2bit_values(const int8_t* data, size_t base_idx, size_t count) {
    __m256i result = _mm256_setzero_si256();
    int8_t extracted[32];

    for (size_t i = 0; i < count; i++) {
        size_t bit_idx = base_idx + i;
        size_t byte_idx = bit_idx / 4;
        size_t bit_pos = (bit_idx % 4) * 2;
        
        int8_t two_bits = (data[byte_idx] >> bit_pos) & 0x3;
        // Convert 2-bit to signed int8: 0->1, 1->0, 2->-2, 3->-1
        extracted[i] = (two_bits == 0) ? 1 : 
                       (two_bits == 1) ? 0 : 
                       (two_bits == 2) ? -2 : -1;
    }
    return _mm256_loadu_si256((__m256i*)extracted);
}

static inline __m256i extract_4bit_values(const int8_t* data, size_t base_idx, size_t count) {
    __m256i result = _mm256_setzero_si256();
    int8_t extracted[32];
    

    for (size_t i = 0; i < count; i++) {
        size_t bit_idx = base_idx + i;
        size_t byte_idx = bit_idx / 2;
        size_t is_high = bit_idx % 2;
        
        int8_t nibble = is_high ? ((data[byte_idx] >> 4) & 0xF) : (data[byte_idx] & 0xF);
        // Sign extend from 4 bits
        extracted[i] = (nibble & 0x8) ? (nibble | 0xF0) : nibble;
    }
    return _mm256_loadu_si256((__m256i*)extracted);
}

static inline int32_t horizontal_sum_epi32(__m256i v) {
    // Store to memory and sum manually
    int32_t sum_array[8];
    _mm256_storeu_si256((__m256i*)sum_array, v);
    
    return sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
           sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
}

// Optimized 8-bit matrix multiplication using AVX2
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time with AVX2
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features];
            const int8_t* w_row = &w->data[j * in_features];
            
            // Main vectorized loop - process 32 elements at a time
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_row[k]);
                __m256i w_vec = _mm256_loadu_si256((__m256i*)&w_row[k]);
                
                // Multiply 8-bit integers and accumulate into 16-bit integers
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                // Extend 16-bit to 32-bit and accumulate
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce the vector to a single value
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                acc += x_row[k] * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 8-bit input x 4-bit weights
void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features];
            const int8_t* w_row = &w->data[j * in_features / 2]; // 4-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_row[k]);
                
                // Extract 4-bit values from w
                __m256i w_vec = extract_4bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                // Extend 16-bit to 32-bit and accumulate
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce the vector to a single value
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_w = w_row[k/2];
                temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += x_row[k] * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 8-bit input x 2-bit weights
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features];
            const int8_t* w_row = &w->data[j * in_features / 4]; // 2-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_row[k]);
                
                // Extract 2-bit values from w
                __m256i w_vec = extract_2bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate using the same pattern as previous functions
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                acc += x_row[k] * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 8-bit input x 1-bit weights
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features];
            const int8_t* w_row = &w->data[j * in_features / 8]; // 1-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_row[k]);
                
                // Extract 1-bit values from w and convert to -1/1
                __m256i w_vec = extract_1bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_w = w_row[k/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                acc += x_row[k] * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 4-bit input x 4-bit weights
void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 2]; // 4-bit packed
            const int8_t* w_row = &w->data[j * in_features / 2]; // 4-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract 4-bit values from x and w
                __m256i x_vec = extract_4bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_4bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                
                int8_t temp_w = w_row[k/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 4-bit input x 2-bit weights
void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 2]; // 4-bit packed
            const int8_t* w_row = &w->data[j * in_features / 4]; // 2-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract bits for vectorized computation
                __m256i x_vec = extract_4bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_2bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 4-bit input x 1-bit weights  
void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 2]; // 4-bit packed
            const int8_t* w_row = &w->data[j * in_features / 8]; // 1-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract bits
                __m256i x_vec = extract_4bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_1bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                
                int8_t temp_w = w_row[k/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit input x 2-bit weights
void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 4]; // 2-bit packed
            const int8_t* w_row = &w->data[j * in_features / 4]; // 2-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract 2-bit values
                __m256i x_vec = extract_2bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_2bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                
                int8_t temp_w = w_row[k/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit input x 1-bit weights
void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 4]; // 2-bit packed
            const int8_t* w_row = &w->data[j * in_features / 8]; // 1-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract bits
                __m256i x_vec = extract_2bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_1bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                
                int8_t temp_w = w_row[k/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                
                acc += temp_a * temp_w;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit input x 1-bit weights (Binary Neural Network)
void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 64 inputs at a time (since 1-bit operations are very efficient)
    const size_t vec_size = 64;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            const int8_t* x_row = &x->data[i * in_features / 8]; // 1-bit packed
            const int8_t* w_row = &w->data[j * in_features / 8]; // 1-bit packed
            
            // For binary operations, we can use XOR and population count
            // This is much faster than extracting each bit individually
            for (size_t k = 0; k < in_features/8; k++) {
                // XNOR of bits gives 1 when bits match, 0 when they differ
                uint8_t xnor_result = ~(x_row[k] ^ w_row[k]);
                
                // Count set bits (each set bit contributes -1, each unset bit +1)
                acc += (8 - __builtin_popcount(xnor_result) - __builtin_popcount(xnor_result));
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// Optimized implementations for 4-bit x 8-bit
void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 2]; // 4-bit packed
            const int8_t* w_row = &w->data[j * in_features]; // 8-bit
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract 4-bit values from x
                __m256i x_vec = extract_4bit_values(x_row, k, vec_size);
                __m256i w_vec = _mm256_loadu_si256((__m256i*)&w_row[k]);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit input x 2-bit weights
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 8]; // 1-bit packed
            const int8_t* w_row = &w->data[j * in_features / 4]; // 2-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract bits
                __m256i x_vec = extract_1bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_2bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
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

// 1-bit input x 4-bit weights
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 8]; // 1-bit packed
            const int8_t* w_row = &w->data[j * in_features / 2]; // 4-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract bits
                __m256i x_vec = extract_1bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_4bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
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

// 2-bit input x 4-bit weights
void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 4]; // 2-bit packed
            const int8_t* w_row = &w->data[j * in_features / 2]; // 4-bit packed
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract bits
                __m256i x_vec = extract_2bit_values(x_row, k, vec_size);
                __m256i w_vec = extract_4bit_values(w_row, k, vec_size);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
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

// 2-bit input x 8-bit weights
void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 4]; // 2-bit packed
            const int8_t* w_row = &w->data[j * in_features]; // 8-bit
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract 2-bit values from x
                __m256i x_vec = extract_2bit_values(x_row, k, vec_size);
                __m256i w_vec = _mm256_loadu_si256((__m256i*)&w_row[k]);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                
                acc += temp_a * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit input x 8-bit weights
void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Process 32 inputs at a time
    const size_t vec_size = 32;
    const size_t aligned_features = (in_features / vec_size) * vec_size;
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            __m256i acc_vec = _mm256_setzero_si256();
            const int8_t* x_row = &x->data[i * in_features / 8]; // 1-bit packed
            const int8_t* w_row = &w->data[j * in_features]; // 8-bit
            
            // Main vectorized loop
            for (size_t k = 0; k < aligned_features; k += vec_size) {
                // Extract 1-bit values from x
                __m256i x_vec = extract_1bit_values(x_row, k, vec_size);
                __m256i w_vec = _mm256_loadu_si256((__m256i*)&w_row[k]);
                
                // Multiply and accumulate
                __m256i prod_lo = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_vec)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_vec))
                );
                
                __m256i prod_hi = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_vec, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_vec, 1))
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1))
                    )
                );
                
                acc_vec = _mm256_add_epi32(acc_vec, 
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1))
                    )
                );
            }
            
            // Reduce vector to scalar
            int32_t acc = horizontal_sum_epi32(acc_vec);
            
            // Handle remaining elements
            for (size_t k = aligned_features; k < in_features; k++) {
                int8_t temp_a = x_row[k/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                
                acc += temp_a * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}
