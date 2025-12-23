#include "mico_qnn.h"

// LUT-based Mixed Precision MatMul Kernels
// Based on the T-MAC approach: https://github.com/microsoft/T-MAC
//
// T-MAC Key Insight:
// For low-bitwidth weights, use a small LUT (16 entries for 4-bit indices) indexed
// by weight nibbles. Each LUT entry stores the partial sum for 4 activations with
// all possible sign combinations. For multi-bit weights, decompose into bit planes
// and scale+accumulate the results.
//
// This implementation provides a portable scalar version of T-MAC's approach,
// optimized for general CPUs and RISC-V without requiring SIMD intrinsics.
//
// For 4-bit nibble-based LUT:
// - Group 4 activations [a0, a1, a2, a3]
// - Build 16-entry LUT where index bits [b0,b1,b2,b3] determine signs
// - LUT[idx] = a0*(b0?-1:1) + a1*(b1?-1:1) + a2*(b2?-1:1) + a3*(b3?-1:1)
// - Weight nibble directly indexes into this LUT

// =============================================================================
// Weight/activation decoding functions  
// =============================================================================

// 2-bit weight encoding: 0->0, 1->+1, 2->-2, 3->-1
static inline int8_t decode_2bit(uint8_t bits) {
    static const int8_t lut[4] = {0, 1, -2, -1};
    return lut[bits & 0x03];
}

// 1-bit weight encoding: 0->+1, 1->-1 (for BitNet-style models)
static inline int8_t decode_1bit(uint8_t bit) {
    return (bit & 0x01) ? -1 : 1;
}

// 4-bit signed weight: values 0-7 -> 0..7, values 8-15 -> -8..-1
static inline int8_t decode_4bit(uint8_t bits) {
    int8_t val = bits & 0x0F;
    return (val >= 8) ? (val - 16) : val;
}

// =============================================================================
// T-MAC style LUT building for 4 activations indexed by 4-bit sign pattern
// =============================================================================
// 
// For a group of 4 activations [a0, a1, a2, a3], build a 16-entry LUT where:
// LUT[idx] = sum of (ai * sign_i) where sign_i = (idx & (1<<i)) ? -1 : +1
//
// This is the core T-MAC building block for 1-bit weights (sign only).
// For multi-bit weights, we decompose into bit planes and scale+accumulate.

static inline void build_sign_lut_4(int32_t *lut, int8_t a0, int8_t a1, int8_t a2, int8_t a3) {
    // Precompute partial sums for faster LUT building
    int32_t sum_all = a0 + a1 + a2 + a3;
    
    // LUT[0] = +a0 + a1 + a2 + a3 (all signs positive)
    lut[0] = sum_all;
    
    // LUT[idx] = sum_all - 2*(sum of activations where bit is set)
    // This uses the identity: if sign flips from +1 to -1, contribution changes by -2*a
    lut[1]  = sum_all - 2*a0;                           // -a0 +a1 +a2 +a3
    lut[2]  = sum_all - 2*a1;                           // +a0 -a1 +a2 +a3
    lut[3]  = sum_all - 2*a0 - 2*a1;                    // -a0 -a1 +a2 +a3
    lut[4]  = sum_all - 2*a2;                           // +a0 +a1 -a2 +a3
    lut[5]  = sum_all - 2*a0 - 2*a2;                    // -a0 +a1 -a2 +a3
    lut[6]  = sum_all - 2*a1 - 2*a2;                    // +a0 -a1 -a2 +a3
    lut[7]  = sum_all - 2*a0 - 2*a1 - 2*a2;             // -a0 -a1 -a2 +a3
    lut[8]  = sum_all - 2*a3;                           // +a0 +a1 +a2 -a3
    lut[9]  = sum_all - 2*a0 - 2*a3;                    // -a0 +a1 +a2 -a3
    lut[10] = sum_all - 2*a1 - 2*a3;                    // +a0 -a1 +a2 -a3
    lut[11] = sum_all - 2*a0 - 2*a1 - 2*a3;             // -a0 -a1 +a2 -a3
    lut[12] = sum_all - 2*a2 - 2*a3;                    // +a0 +a1 -a2 -a3
    lut[13] = sum_all - 2*a0 - 2*a2 - 2*a3;             // -a0 +a1 -a2 -a3
    lut[14] = sum_all - 2*a1 - 2*a2 - 2*a3;             // +a0 -a1 -a2 -a3
    lut[15] = sum_all - 2*a0 - 2*a1 - 2*a2 - 2*a3;      // -a0 -a1 -a2 -a3
}

// Build LUT for 8 activations (used for 1-bit weights where 8 weights per byte)
static inline void build_sign_lut_8(int32_t *lut, const int8_t *a) {
    // This builds a 256-entry LUT for 8 activations
    int32_t sum_all = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    
    for (int idx = 0; idx < 256; idx++) {
        int32_t result = sum_all;
        for (int bit = 0; bit < 8; bit++) {
            if (idx & (1 << bit)) {
                result -= 2 * a[bit];  // Flip sign from +1 to -1
            }
        }
        lut[idx] = result;
    }
}

// =============================================================================
// 8-bit activation x 1-bit weight LUT-based MatMul (T-MAC style)
// =============================================================================
//
// For 1-bit weights: 8 weights packed per byte, each weight is +1 or -1
// Build a 256-entry LUT for 8 activations, indexed by weight byte

__attribute__((weak)) void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Process 8 activations at a time (1 weight byte)
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                // Build LUT for these 8 activations
                build_sign_lut_8(lut, &x_row[k]);
                
                // Look up result using weight byte
                uint8_t wb = (uint8_t)w_row[k / 8];
                acc += lut[wb];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t w_val = decode_1bit((wb >> (k & 0x07)) & 0x01);
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 8-bit activation x 2-bit weight LUT-based MatMul (T-MAC style)
// =============================================================================
//
// For 2-bit weights: 4 weights per byte, encoded as {0, +1, -2, -1}
// T-MAC decomposes 2-bit into 2 bit planes and scales by 2^bit_position
// But for scalar code, we use a 256-entry LUT indexed by the full weight byte

__attribute__((weak)) void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // 256-entry LUT for 4 activations x 2-bit weights
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Process 4 activations at a time (1 weight byte = 4 x 2-bit weights)
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                int8_t a0 = x_row[k + 0];
                int8_t a1 = x_row[k + 1];
                int8_t a2 = x_row[k + 2];
                int8_t a3 = x_row[k + 3];
                
                // Build LUT: LUT[wb] = sum(ai * decode_2bit(wi))
                for (int wb = 0; wb < 256; wb++) {
                    int8_t w0 = decode_2bit((wb >> 0) & 0x03);
                    int8_t w1 = decode_2bit((wb >> 2) & 0x03);
                    int8_t w2 = decode_2bit((wb >> 4) & 0x03);
                    int8_t w3 = decode_2bit((wb >> 6) & 0x03);
                    lut[wb] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 4];
                acc += lut[wb];
            }
            
            // Handle remaining elements
            for (; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Optimized: Precompute LUTs for all activation groups, reuse across outputs
// =============================================================================

__attribute__((weak)) void MiCo_Q8x2_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    const size_t num_groups = in_features / 4;
    
    // Allocate LUT storage: 256 entries per group
    int32_t lut_storage[256 * 64];  // Stack buffer for up to 64 groups
    int32_t *luts = (num_groups <= 64) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    
    if (luts == NULL && num_groups > 64) {
        // Fallback to non-optimized version
        MiCo_Q8x2_MatMul(O, x, w);
        return;
    }
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        // Precompute LUTs for all activation groups
        for (size_t g = 0; g < num_groups; g++) {
            int8_t a0 = x_row[g * 4 + 0];
            int8_t a1 = x_row[g * 4 + 1];
            int8_t a2 = x_row[g * 4 + 2];
            int8_t a3 = x_row[g * 4 + 3];
            int32_t *group_lut = &luts[g * 256];
            
            for (int wb = 0; wb < 256; wb++) {
                int8_t w0 = decode_2bit((wb >> 0) & 0x03);
                int8_t w1 = decode_2bit((wb >> 2) & 0x03);
                int8_t w2 = decode_2bit((wb >> 4) & 0x03);
                int8_t w3 = decode_2bit((wb >> 6) & 0x03);
                group_lut[wb] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
            }
        }
        
        // Process each output using precomputed LUTs
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            // Handle remaining elements
            size_t k = num_groups * 4;
            for (; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 64) {
        free(luts);
    }
}

// =============================================================================
// 8-bit activation x 4-bit weight LUT-based MatMul
// =============================================================================

__attribute__((weak)) void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // 256-entry LUT for 2 activations x 4-bit weights
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            // Process 2 activations at a time (1 weight byte = 2 x 4-bit weights)
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                int8_t a0 = x_row[k + 0];
                int8_t a1 = x_row[k + 1];
                
                // Build LUT
                for (int wb = 0; wb < 256; wb++) {
                    int8_t w0 = decode_4bit(wb & 0x0F);
                    int8_t w1 = decode_4bit((wb >> 4) & 0x0F);
                    lut[wb] = a0 * w0 + a1 * w1;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 2];
                acc += lut[wb];
            }
            
            // Handle remaining element
            if (k < in_features) {
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t w0 = decode_4bit(wb & 0x0F);
                acc += x_row[k] * w0;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Standard 8-bit MatMul (no LUT needed)
// =============================================================================

__attribute__((weak)) void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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
// 4-bit x 4-bit LUT-based MatMul
// =============================================================================

__attribute__((weak)) void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                int8_t a0 = decode_4bit(xb & 0x0F);
                int8_t a1 = decode_4bit((xb >> 4) & 0x0F);
                
                for (int wb = 0; wb < 256; wb++) {
                    int8_t w0 = decode_4bit(wb & 0x0F);
                    int8_t w1 = decode_4bit((wb >> 4) & 0x0F);
                    lut[wb] = a0 * w0 + a1 * w1;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 2];
                acc += lut[wb];
            }
            
            if (k < in_features) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a0 = decode_4bit(xb & 0x0F);
                int8_t w0 = decode_4bit(wb & 0x0F);
                acc += a0 * w0;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 2-bit x 2-bit LUT-based MatMul
// =============================================================================

__attribute__((weak)) void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                int8_t a0 = decode_2bit((xb >> 0) & 0x03);
                int8_t a1 = decode_2bit((xb >> 2) & 0x03);
                int8_t a2 = decode_2bit((xb >> 4) & 0x03);
                int8_t a3 = decode_2bit((xb >> 6) & 0x03);
                
                for (int wb = 0; wb < 256; wb++) {
                    int8_t w0 = decode_2bit((wb >> 0) & 0x03);
                    int8_t w1 = decode_2bit((wb >> 2) & 0x03);
                    int8_t w2 = decode_2bit((wb >> 4) & 0x03);
                    int8_t w3 = decode_2bit((wb >> 6) & 0x03);
                    lut[wb] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 4];
                acc += lut[wb];
            }
            
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = decode_2bit((xb >> (2 * (k & 0x03))) & 0x03);
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// 1-bit x 1-bit MatMul (BNN - uses XNOR + popcount)
// =============================================================================

__attribute__((weak)) void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const uint8_t *x_row = (const uint8_t *)&x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const uint8_t *w_row = (const uint8_t *)&w->data[j * in_features / 8];
            int32_t acc = 0;
            
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb = x_row[k / 8];
                uint8_t wb = w_row[k / 8];
                
                // XNOR + popcount for BNN
                uint8_t xnor = ~(xb ^ wb);
#ifdef __GNUC__
                int matches = __builtin_popcount((unsigned int)xnor);
#else
                int matches = 0;
                uint8_t temp = xnor;
                while (temp) { matches++; temp &= temp - 1; }
#endif
                acc += 2 * matches - 8;
            }
            
            for (; k < in_features; k++) {
                uint8_t xb = x_row[k / 8];
                uint8_t wb = w_row[k / 8];
                int bit_pos = k & 0x07;
                int x_bit = (xb >> bit_pos) & 0x01;
                int w_bit = (wb >> bit_pos) & 0x01;
                acc += (x_bit == w_bit) ? 1 : -1;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Mixed precision: 4-bit activation x 2-bit weight
// =============================================================================

__attribute__((weak)) void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb0 = (uint8_t)x_row[k / 2];
                uint8_t xb1 = (uint8_t)x_row[k / 2 + 1];
                int8_t a0 = decode_4bit(xb0 & 0x0F);
                int8_t a1 = decode_4bit((xb0 >> 4) & 0x0F);
                int8_t a2 = decode_4bit(xb1 & 0x0F);
                int8_t a3 = decode_4bit((xb1 >> 4) & 0x0F);
                
                for (int wb = 0; wb < 256; wb++) {
                    int8_t w0 = decode_2bit((wb >> 0) & 0x03);
                    int8_t w1 = decode_2bit((wb >> 2) & 0x03);
                    int8_t w2 = decode_2bit((wb >> 4) & 0x03);
                    int8_t w3 = decode_2bit((wb >> 6) & 0x03);
                    lut[wb] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 4];
                acc += lut[wb];
            }
            
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = decode_4bit((xb >> (4 * (k & 0x01))) & 0x0F);
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Mixed precision: 4-bit activation x 1-bit weight
// =============================================================================

__attribute__((weak)) void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                int8_t a[8];
                for (int n = 0; n < 4; n++) {
                    uint8_t xb = (uint8_t)x_row[k / 2 + n];
                    a[n * 2] = decode_4bit(xb & 0x0F);
                    a[n * 2 + 1] = decode_4bit((xb >> 4) & 0x0F);
                }
                
                build_sign_lut_8(lut, a);
                
                uint8_t wb = (uint8_t)w_row[k / 8];
                acc += lut[wb];
            }
            
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t a_val = decode_4bit((xb >> (4 * (k & 0x01))) & 0x0F);
                int8_t w_val = decode_1bit((wb >> (k & 0x07)) & 0x01);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Mixed precision: 2-bit activation x 1-bit weight
// =============================================================================

__attribute__((weak)) void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                int8_t a[8];
                uint8_t xb0 = (uint8_t)x_row[k / 4];
                uint8_t xb1 = (uint8_t)x_row[k / 4 + 1];
                a[0] = decode_2bit((xb0 >> 0) & 0x03);
                a[1] = decode_2bit((xb0 >> 2) & 0x03);
                a[2] = decode_2bit((xb0 >> 4) & 0x03);
                a[3] = decode_2bit((xb0 >> 6) & 0x03);
                a[4] = decode_2bit((xb1 >> 0) & 0x03);
                a[5] = decode_2bit((xb1 >> 2) & 0x03);
                a[6] = decode_2bit((xb1 >> 4) & 0x03);
                a[7] = decode_2bit((xb1 >> 6) & 0x03);
                
                build_sign_lut_8(lut, a);
                
                uint8_t wb = (uint8_t)w_row[k / 8];
                acc += lut[wb];
            }
            
            for (; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t a_val = decode_2bit((xb >> (2 * (k & 0x03))) & 0x03);
                int8_t w_val = decode_1bit((wb >> (k & 0x07)) & 0x01);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

// =============================================================================
// Reversed precision (activation bits < weight bits) - direct computation
// These don't benefit as much from LUT since weight space is larger
// =============================================================================

__attribute__((weak)) void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            int32_t acc = 0;
            
            for (size_t k = 0; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                int8_t a_val = decode_4bit((xb >> (4 * (k & 0x01))) & 0x0F);
                acc += a_val * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            int32_t acc = 0;
            
            for (size_t k = 0; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                int8_t a_val = decode_2bit((xb >> (2 * (k & 0x03))) & 0x03);
                acc += a_val * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features];
            int32_t acc = 0;
            
            for (size_t k = 0; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                int8_t a_val = decode_1bit((xb >> (k & 0x07)) & 0x01);
                acc += a_val * w_row[k];
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            for (size_t k = 0; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a_val = decode_2bit((xb >> (2 * (k & 0x03))) & 0x03);
                int8_t w_val = decode_4bit((wb >> (4 * (k & 0x01))) & 0x0F);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            for (size_t k = 0; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a_val = decode_1bit((xb >> (k & 0x07)) & 0x01);
                int8_t w_val = decode_4bit((wb >> (4 * (k & 0x01))) & 0x0F);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}

__attribute__((weak)) void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            for (size_t k = 0; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 8];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = decode_1bit((xb >> (k & 0x07)) & 0x01);
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
}
