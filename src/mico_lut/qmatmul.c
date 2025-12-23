#include "mico_qnn.h"

// LUT-based Mixed Precision MatMul Kernels
// Based on the T-MAC approach: https://github.com/microsoft/T-MAC
//
// Key Idea:
// For low-bitwidth weights (1, 2, 4 bits), the number of possible weight values
// is small (2, 4, 16 respectively). Instead of extracting each weight and 
// multiplying, we precompute a Look-Up Table (LUT) of partial sums for groups 
// of activations. This converts multiply-accumulate operations into table 
// lookups and additions.
//
// For example, with 2-bit weights having values {0, 1, -1, -2} (encoded as 0,1,2,3):
// - For each group of 4 activations [a0, a1, a2, a3], precompute:
//   LUT[0] = 0*a0 + 0*a1 + 0*a2 + 0*a3 = 0
//   LUT[1] = 1*a0 + 1*a1 + 1*a2 + 1*a3 = sum(a)
//   LUT[2] = -2*a0 + -2*a1 + ... = -2*sum(a)
//   LUT[3] = -1*a0 + -1*a1 + ... = -sum(a)
// - Then for each weight byte (containing 4 x 2-bit weights), just look up and accumulate
//
// This is especially beneficial when:
// - Weight bit-width is very low (1-4 bits)
// - The same activation is reused across many output features
// - LUT construction cost is amortized across output features

// Group size for LUT construction (number of activations processed together)
// Using 4 as default since 4 x 2-bit = 8 bits = 1 byte
#define LUT_GROUP_SIZE 4

// =============================================================================
// 8-bit activation x 2-bit weight LUT-based MatMul
// =============================================================================
// 
// For 2-bit weights encoded as: 0->0, 1->+1, 2->-2, 3->-1
// For a group of G activations, we build a LUT with 4^G entries
// But for G=1, we have 4 entries: val * {0, 1, -2, -1}
// We use a simplified approach: process weights byte-by-byte (4 weights per byte)
// and sum up contributions using per-weight LUTs

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // LUT for 2-bit weight values: maps 2-bit code to multiplier
    // 0 -> 0, 1 -> +1, 2 -> -2, 3 -> -1
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    
    // Process each batch
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        // Build LUT: For each activation, precompute val * {0, 1, -2, -1}
        // We store activation-scaled LUT values
        // To save memory, we compute on-the-fly but group activations
        
        // Process each output feature
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Process 4 weights (1 byte) at a time
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t wb = (uint8_t)w_row[k / 4];
                
                // Extract 4 x 2-bit weights and look up multipliers
                int8_t w0 = lut_2bit[(wb >> 0) & 0x03];
                int8_t w1 = lut_2bit[(wb >> 2) & 0x03];
                int8_t w2 = lut_2bit[(wb >> 4) & 0x03];
                int8_t w3 = lut_2bit[(wb >> 6) & 0x03];
                
                // Multiply and accumulate
                acc += x_row[k + 0] * w0;
                acc += x_row[k + 1] * w1;
                acc += x_row[k + 2] * w2;
                acc += x_row[k + 3] * w3;
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t w_val = lut_2bit[(wb >> (2 * (k & 0x03))) & 0x03];
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 8-bit activation x 4-bit weight LUT-based MatMul
// =============================================================================
//
// For 4-bit signed weights in range [-8, 7]:
// Instead of extracting and sign-extending each 4-bit value,
// we use a LUT that directly maps the 4-bit pattern to the signed value

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // LUT for 4-bit signed values: maps 4-bit code to signed int8
    // Values 0-7 stay as is, 8-15 become -8 to -1
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            // Process 2 weights (1 byte) at a time
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                uint8_t wb = (uint8_t)w_row[k / 2];
                
                // Extract 2 x 4-bit weights using LUT
                int8_t w0 = lut_4bit[wb & 0x0F];
                int8_t w1 = lut_4bit[(wb >> 4) & 0x0F];
                
                acc += x_row[k + 0] * w0;
                acc += x_row[k + 1] * w1;
            }
            
            // Handle remaining element
            if (k < in_features) {
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t w0 = lut_4bit[wb & 0x0F];
                acc += x_row[k] * w0;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 8-bit activation x 1-bit weight LUT-based MatMul
// =============================================================================
//
// For 1-bit weights: 0 -> +1, 1 -> -1
// This is mathematically equivalent to: result = total_sum - 2 * neg_sum
// where neg_sum is the sum of activations where weight bit is 1
// But we can also use a simple LUT approach for consistency

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // LUT for 1-bit weight: 0 -> +1, 1 -> -1
    static const int8_t lut_1bit[2] = {1, -1};
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Process 8 weights (1 byte) at a time
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t wb = (uint8_t)w_row[k / 8];
                
                // Extract and process 8 x 1-bit weights using LUT
                acc += x_row[k + 0] * lut_1bit[(wb >> 0) & 0x01];
                acc += x_row[k + 1] * lut_1bit[(wb >> 1) & 0x01];
                acc += x_row[k + 2] * lut_1bit[(wb >> 2) & 0x01];
                acc += x_row[k + 3] * lut_1bit[(wb >> 3) & 0x01];
                acc += x_row[k + 4] * lut_1bit[(wb >> 4) & 0x01];
                acc += x_row[k + 5] * lut_1bit[(wb >> 5) & 0x01];
                acc += x_row[k + 6] * lut_1bit[(wb >> 6) & 0x01];
                acc += x_row[k + 7] * lut_1bit[(wb >> 7) & 0x01];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t w_val = lut_1bit[(wb >> (k & 0x07)) & 0x01];
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 4-bit activation x 2-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Process 4 elements at a time (2 bytes of activations, 1 byte of weights)
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb0 = (uint8_t)x_row[k / 2];
                uint8_t xb1 = (uint8_t)x_row[k / 2 + 1];
                uint8_t wb = (uint8_t)w_row[k / 4];
                
                // Extract activations and weights using LUTs
                int8_t a0 = lut_4bit[xb0 & 0x0F];
                int8_t a1 = lut_4bit[(xb0 >> 4) & 0x0F];
                int8_t a2 = lut_4bit[xb1 & 0x0F];
                int8_t a3 = lut_4bit[(xb1 >> 4) & 0x0F];
                
                int8_t w0 = lut_2bit[(wb >> 0) & 0x03];
                int8_t w1 = lut_2bit[(wb >> 2) & 0x03];
                int8_t w2 = lut_2bit[(wb >> 4) & 0x03];
                int8_t w3 = lut_2bit[(wb >> 6) & 0x03];
                
                acc += a0 * w0;
                acc += a1 * w1;
                acc += a2 * w2;
                acc += a3 * w3;
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = lut_4bit[(xb >> (4 * (k & 0x01))) & 0x0F];
                int8_t w_val = lut_2bit[(wb >> (2 * (k & 0x03))) & 0x03];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 4-bit activation x 1-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_1bit[2] = {1, -1};
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Process 8 elements at a time
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb0 = (uint8_t)x_row[k / 2];
                uint8_t xb1 = (uint8_t)x_row[k / 2 + 1];
                uint8_t xb2 = (uint8_t)x_row[k / 2 + 2];
                uint8_t xb3 = (uint8_t)x_row[k / 2 + 3];
                uint8_t wb = (uint8_t)w_row[k / 8];
                
                int8_t a0 = lut_4bit[xb0 & 0x0F];
                int8_t a1 = lut_4bit[(xb0 >> 4) & 0x0F];
                int8_t a2 = lut_4bit[xb1 & 0x0F];
                int8_t a3 = lut_4bit[(xb1 >> 4) & 0x0F];
                int8_t a4 = lut_4bit[xb2 & 0x0F];
                int8_t a5 = lut_4bit[(xb2 >> 4) & 0x0F];
                int8_t a6 = lut_4bit[xb3 & 0x0F];
                int8_t a7 = lut_4bit[(xb3 >> 4) & 0x0F];
                
                acc += a0 * lut_1bit[(wb >> 0) & 0x01];
                acc += a1 * lut_1bit[(wb >> 1) & 0x01];
                acc += a2 * lut_1bit[(wb >> 2) & 0x01];
                acc += a3 * lut_1bit[(wb >> 3) & 0x01];
                acc += a4 * lut_1bit[(wb >> 4) & 0x01];
                acc += a5 * lut_1bit[(wb >> 5) & 0x01];
                acc += a6 * lut_1bit[(wb >> 6) & 0x01];
                acc += a7 * lut_1bit[(wb >> 7) & 0x01];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t a_val = lut_4bit[(xb >> (4 * (k & 0x01))) & 0x0F];
                int8_t w_val = lut_1bit[(wb >> (k & 0x07)) & 0x01];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 4-bit activation x 4-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            // Process 2 elements (1 byte each) at a time
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 2];
                
                int8_t a0 = lut_4bit[xb & 0x0F];
                int8_t a1 = lut_4bit[(xb >> 4) & 0x0F];
                int8_t w0 = lut_4bit[wb & 0x0F];
                int8_t w1 = lut_4bit[(wb >> 4) & 0x0F];
                
                acc += a0 * w0;
                acc += a1 * w1;
            }
            
            // Handle remaining element
            if (k < in_features) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a0 = lut_4bit[xb & 0x0F];
                int8_t w0 = lut_4bit[wb & 0x0F];
                acc += a0 * w0;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 2-bit activation x 2-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Process 4 elements (1 byte each) at a time
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 4];
                
                int8_t a0 = lut_2bit[(xb >> 0) & 0x03];
                int8_t a1 = lut_2bit[(xb >> 2) & 0x03];
                int8_t a2 = lut_2bit[(xb >> 4) & 0x03];
                int8_t a3 = lut_2bit[(xb >> 6) & 0x03];
                
                int8_t w0 = lut_2bit[(wb >> 0) & 0x03];
                int8_t w1 = lut_2bit[(wb >> 2) & 0x03];
                int8_t w2 = lut_2bit[(wb >> 4) & 0x03];
                int8_t w3 = lut_2bit[(wb >> 6) & 0x03];
                
                acc += a0 * w0;
                acc += a1 * w1;
                acc += a2 * w2;
                acc += a3 * w3;
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = lut_2bit[(xb >> (2 * (k & 0x03))) & 0x03];
                int8_t w_val = lut_2bit[(wb >> (2 * (k & 0x03))) & 0x03];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 2-bit activation x 1-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_1bit[2] = {1, -1};
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Process 8 elements at a time (2 bytes of activations, 1 byte of weights)
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb0 = (uint8_t)x_row[k / 4];
                uint8_t xb1 = (uint8_t)x_row[k / 4 + 1];
                uint8_t wb = (uint8_t)w_row[k / 8];
                
                int8_t a0 = lut_2bit[(xb0 >> 0) & 0x03];
                int8_t a1 = lut_2bit[(xb0 >> 2) & 0x03];
                int8_t a2 = lut_2bit[(xb0 >> 4) & 0x03];
                int8_t a3 = lut_2bit[(xb0 >> 6) & 0x03];
                int8_t a4 = lut_2bit[(xb1 >> 0) & 0x03];
                int8_t a5 = lut_2bit[(xb1 >> 2) & 0x03];
                int8_t a6 = lut_2bit[(xb1 >> 4) & 0x03];
                int8_t a7 = lut_2bit[(xb1 >> 6) & 0x03];
                
                acc += a0 * lut_1bit[(wb >> 0) & 0x01];
                acc += a1 * lut_1bit[(wb >> 1) & 0x01];
                acc += a2 * lut_1bit[(wb >> 2) & 0x01];
                acc += a3 * lut_1bit[(wb >> 3) & 0x01];
                acc += a4 * lut_1bit[(wb >> 4) & 0x01];
                acc += a5 * lut_1bit[(wb >> 5) & 0x01];
                acc += a6 * lut_1bit[(wb >> 6) & 0x01];
                acc += a7 * lut_1bit[(wb >> 7) & 0x01];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t a_val = lut_2bit[(xb >> (2 * (k & 0x03))) & 0x03];
                int8_t w_val = lut_1bit[(wb >> (k & 0x07)) & 0x01];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 1-bit activation x 1-bit weight LUT-based MatMul (Binary Neural Network)
// =============================================================================

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // For BNN, we use XNOR + popcount optimization
    // XNOR(a,w) gives 1 when a==w, 0 otherwise
    // result = 2*popcount(XNOR(a,w)) - n where n is number of bits
    
    for (size_t i = 0; i < batch_size; i++) {
        const uint8_t *x_row = (const uint8_t *)&x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const uint8_t *w_row = (const uint8_t *)&w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Process 8 bits (1 byte) at a time
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb = x_row[k / 8];
                uint8_t wb = w_row[k / 8];
                
                // XNOR and count matching bits
                uint8_t xnor_result = ~(xb ^ wb);
                
                // Popcount for 8-bit value
                int popcount = 0;
                uint8_t temp = xnor_result;
                while (temp) {
                    popcount++;
                    temp &= temp - 1;
                }
                
                // 2*matches - 8 (since 1-bit values are +1 or -1)
                acc += 2 * popcount - 8;
            }
            
            // Handle remaining bits
            for (; k < in_features; k++) {
                uint8_t xb = x_row[k / 8];
                uint8_t wb = w_row[k / 8];
                int bit_pos = k & 0x07;
                int x_bit = (xb >> bit_pos) & 0x01;
                int w_bit = (wb >> bit_pos) & 0x01;
                // +1 if same, -1 if different
                acc += (x_bit == w_bit) ? 1 : -1;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Standard 8-bit MatMul (baseline, not LUT-optimized)
// =============================================================================

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                acc += x->data[i * in_features + k] * w->data[j * in_features + k];
            }
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Reversed precision operations
// =============================================================================

// 4-bit activation x 8-bit weight
void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            int32_t acc = 0;
            
            // Process 2 elements at a time
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                
                int8_t a0 = lut_4bit[xb & 0x0F];
                int8_t a1 = lut_4bit[(xb >> 4) & 0x0F];
                
                acc += a0 * w_row[k];
                acc += a1 * w_row[k + 1];
            }
            
            // Handle remaining element
            if (k < in_features) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                int8_t a0 = lut_4bit[xb & 0x0F];
                acc += a0 * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit activation x 8-bit weight
void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            int32_t acc = 0;
            
            // Process 4 elements at a time
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                
                int8_t a0 = lut_2bit[(xb >> 0) & 0x03];
                int8_t a1 = lut_2bit[(xb >> 2) & 0x03];
                int8_t a2 = lut_2bit[(xb >> 4) & 0x03];
                int8_t a3 = lut_2bit[(xb >> 6) & 0x03];
                
                acc += a0 * w_row[k];
                acc += a1 * w_row[k + 1];
                acc += a2 * w_row[k + 2];
                acc += a3 * w_row[k + 3];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                int8_t a_val = lut_2bit[(xb >> (2 * (k & 0x03))) & 0x03];
                acc += a_val * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit activation x 8-bit weight
void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_1bit[2] = {1, -1};
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            int32_t acc = 0;
            
            // Process 8 elements at a time
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                
                acc += lut_1bit[(xb >> 0) & 0x01] * w_row[k];
                acc += lut_1bit[(xb >> 1) & 0x01] * w_row[k + 1];
                acc += lut_1bit[(xb >> 2) & 0x01] * w_row[k + 2];
                acc += lut_1bit[(xb >> 3) & 0x01] * w_row[k + 3];
                acc += lut_1bit[(xb >> 4) & 0x01] * w_row[k + 4];
                acc += lut_1bit[(xb >> 5) & 0x01] * w_row[k + 5];
                acc += lut_1bit[(xb >> 6) & 0x01] * w_row[k + 6];
                acc += lut_1bit[(xb >> 7) & 0x01] * w_row[k + 7];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                int8_t a_val = lut_1bit[(xb >> (k & 0x07)) & 0x01];
                acc += a_val * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 2-bit activation x 4-bit weight
void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            // Process 4 elements at a time
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb0 = (uint8_t)w_row[k / 2];
                uint8_t wb1 = (uint8_t)w_row[k / 2 + 1];
                
                int8_t a0 = lut_2bit[(xb >> 0) & 0x03];
                int8_t a1 = lut_2bit[(xb >> 2) & 0x03];
                int8_t a2 = lut_2bit[(xb >> 4) & 0x03];
                int8_t a3 = lut_2bit[(xb >> 6) & 0x03];
                
                int8_t w0 = lut_4bit[wb0 & 0x0F];
                int8_t w1 = lut_4bit[(wb0 >> 4) & 0x0F];
                int8_t w2 = lut_4bit[wb1 & 0x0F];
                int8_t w3 = lut_4bit[(wb1 >> 4) & 0x0F];
                
                acc += a0 * w0;
                acc += a1 * w1;
                acc += a2 * w2;
                acc += a3 * w3;
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a_val = lut_2bit[(xb >> (2 * (k & 0x03))) & 0x03];
                int8_t w_val = lut_4bit[(wb >> (4 * (k & 0x01))) & 0x0F];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit activation x 4-bit weight
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_1bit[2] = {1, -1};
    static const int8_t lut_4bit[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1
    };
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            // Process 8 elements at a time
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                uint8_t wb0 = (uint8_t)w_row[k / 2];
                uint8_t wb1 = (uint8_t)w_row[k / 2 + 1];
                uint8_t wb2 = (uint8_t)w_row[k / 2 + 2];
                uint8_t wb3 = (uint8_t)w_row[k / 2 + 3];
                
                acc += lut_1bit[(xb >> 0) & 0x01] * lut_4bit[wb0 & 0x0F];
                acc += lut_1bit[(xb >> 1) & 0x01] * lut_4bit[(wb0 >> 4) & 0x0F];
                acc += lut_1bit[(xb >> 2) & 0x01] * lut_4bit[wb1 & 0x0F];
                acc += lut_1bit[(xb >> 3) & 0x01] * lut_4bit[(wb1 >> 4) & 0x0F];
                acc += lut_1bit[(xb >> 4) & 0x01] * lut_4bit[wb2 & 0x0F];
                acc += lut_1bit[(xb >> 5) & 0x01] * lut_4bit[(wb2 >> 4) & 0x0F];
                acc += lut_1bit[(xb >> 6) & 0x01] * lut_4bit[wb3 & 0x0F];
                acc += lut_1bit[(xb >> 7) & 0x01] * lut_4bit[(wb3 >> 4) & 0x0F];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a_val = lut_1bit[(xb >> (k & 0x07)) & 0x01];
                int8_t w_val = lut_4bit[(wb >> (4 * (k & 0x01))) & 0x0F];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// 1-bit activation x 2-bit weight
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    static const int8_t lut_1bit[2] = {1, -1};
    static const int8_t lut_2bit[4] = {0, 1, -2, -1};
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Process 8 elements at a time
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                uint8_t wb0 = (uint8_t)w_row[k / 4];
                uint8_t wb1 = (uint8_t)w_row[k / 4 + 1];
                
                acc += lut_1bit[(xb >> 0) & 0x01] * lut_2bit[(wb0 >> 0) & 0x03];
                acc += lut_1bit[(xb >> 1) & 0x01] * lut_2bit[(wb0 >> 2) & 0x03];
                acc += lut_1bit[(xb >> 2) & 0x01] * lut_2bit[(wb0 >> 4) & 0x03];
                acc += lut_1bit[(xb >> 3) & 0x01] * lut_2bit[(wb0 >> 6) & 0x03];
                acc += lut_1bit[(xb >> 4) & 0x01] * lut_2bit[(wb1 >> 0) & 0x03];
                acc += lut_1bit[(xb >> 5) & 0x01] * lut_2bit[(wb1 >> 2) & 0x03];
                acc += lut_1bit[(xb >> 6) & 0x01] * lut_2bit[(wb1 >> 4) & 0x03];
                acc += lut_1bit[(xb >> 7) & 0x01] * lut_2bit[(wb1 >> 6) & 0x03];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = lut_1bit[(xb >> (k & 0x07)) & 0x01];
                int8_t w_val = lut_2bit[(wb >> (2 * (k & 0x03))) & 0x03];
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}
