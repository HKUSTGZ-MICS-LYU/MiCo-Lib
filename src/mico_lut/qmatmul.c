#include "mico_qnn.h"

// LUT-based Mixed Precision MatMul Kernels
// Based on the T-MAC approach: https://github.com/microsoft/T-MAC
//
// T-MAC Key Idea:
// For low-bitwidth weights (e.g., 2-bit), group weights together and precompute
// ALL possible partial sums for a group of activations. The packed weight bits
// become an INDEX into a precomputed LUT, eliminating multiply operations.
//
// Example for 2-bit weights (4 weights per byte):
// - For 4 activations [a0, a1, a2, a3], we have 4^4 = 256 possible weight combinations
// - Precompute LUT[0..255] where LUT[idx] = sum of (ai * decode(weight_i))
// - The weight byte directly indexes into this LUT
//
// This converts expensive multiply-accumulate into simple table lookup + add.
// The LUT construction is amortized across many output features.

// =============================================================================
// Weight decoding functions
// =============================================================================

// 2-bit weight: 0->0, 1->+1, 2->-2, 3->-1
static inline int8_t decode_2bit(uint8_t bits) {
    static const int8_t lut[4] = {0, 1, -2, -1};
    return lut[bits & 0x03];
}

// 1-bit weight: 0->+1, 1->-1
static inline int8_t decode_1bit(uint8_t bit) {
    return (bit & 0x01) ? -1 : 1;
}

// 4-bit signed weight
static inline int8_t decode_4bit(uint8_t bits) {
    int8_t val = bits & 0x0F;
    return (val >= 8) ? (val - 16) : val;
}

// =============================================================================
// 8-bit activation x 2-bit weight LUT-based MatMul (T-MAC style)
// =============================================================================
//
// For 2-bit weights: 4 weights packed per byte, each weight can be {0, +1, -2, -1}
// For each group of 4 activations, precompute a 256-entry LUT indexed by the weight byte
// LUT[wb] = a0*w0 + a1*w1 + a2*w2 + a3*w3 where wi = decode_2bit((wb >> 2i) & 3)

__attribute__((weak)) void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Temporary LUT for 4 activations, indexed by weight byte (256 entries)
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Process 4 activations at a time (1 weight byte)
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                // Get 4 activations
                int8_t a0 = x_row[k + 0];
                int8_t a1 = x_row[k + 1];
                int8_t a2 = x_row[k + 2];
                int8_t a3 = x_row[k + 3];
                
                // Build LUT for these 4 activations: LUT[wb] = sum(ai * decode(wi))
                // Each entry corresponds to one possible weight byte value
                for (int wb = 0; wb < 256; wb++) {
                    int8_t w0 = decode_2bit((wb >> 0) & 0x03);
                    int8_t w1 = decode_2bit((wb >> 2) & 0x03);
                    int8_t w2 = decode_2bit((wb >> 4) & 0x03);
                    int8_t w3 = decode_2bit((wb >> 6) & 0x03);
                    lut[wb] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
                
                // Look up the result using the actual weight byte as index
                uint8_t wb = (uint8_t)w_row[k / 4];
                acc += lut[wb];
            }
            
            // Handle remaining elements (less than 4)
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
// Optimized version: Precompute LUT once per activation group, reuse across outputs
// =============================================================================
//
// Key optimization: The LUT for a group of activations is the SAME across all
// output features. So we can precompute LUTs for all activation groups first,
// then reuse them when iterating over output features.

__attribute__((weak)) void MiCo_Q8x2_MatMul_Opt(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    // Number of complete 4-activation groups
    const size_t num_groups = in_features / 4;
    
    // Allocate LUT storage: 256 entries per group
    // Using stack allocation for small sizes, heap for large
    int32_t lut_storage[256 * 64];  // Stack buffer for up to 64 groups
    int32_t *luts = (num_groups <= 64) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    
    if (luts == NULL && num_groups > 64) {
        // Fallback to non-optimized version if allocation fails
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
        
        // Process each output feature using precomputed LUTs
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            // Use LUT lookup for each group
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
// 8-bit activation x 4-bit weight LUT-based MatMul (T-MAC style)
// =============================================================================
//
// For 4-bit weights: 2 weights packed per byte
// For each pair of activations, precompute a 256-entry LUT

__attribute__((weak)) void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    int32_t lut[256];
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            // Process 2 activations at a time (1 weight byte)
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                int8_t a0 = x_row[k + 0];
                int8_t a1 = x_row[k + 1];
                
                // Build LUT for these 2 activations
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
// 8-bit activation x 1-bit weight LUT-based MatMul (T-MAC style)
// =============================================================================
//
// For 1-bit weights: 8 weights packed per byte
// For each group of 8 activations, precompute a 256-entry LUT

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
                int8_t a[8];
                for (int n = 0; n < 8; n++) {
                    a[n] = x_row[k + n];
                }
                
                // Build LUT for these 8 activations
                for (int wb = 0; wb < 256; wb++) {
                    int32_t sum = 0;
                    for (int n = 0; n < 8; n++) {
                        int8_t w_val = decode_1bit((wb >> n) & 0x01);
                        sum += a[n] * w_val;
                    }
                    lut[wb] = sum;
                }
                
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
// Standard 8-bit MatMul (no LUT optimization needed for 8x8)
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
            
            // Process 2 activations at a time (1 byte each)
            size_t k = 0;
            for (; k + 2 <= in_features; k += 2) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                int8_t a0 = decode_4bit(xb & 0x0F);
                int8_t a1 = decode_4bit((xb >> 4) & 0x0F);
                
                // Build LUT for these 2 activations
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
            
            // Process 4 activations at a time (1 byte each)
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                int8_t a0 = decode_2bit((xb >> 0) & 0x03);
                int8_t a1 = decode_2bit((xb >> 2) & 0x03);
                int8_t a2 = decode_2bit((xb >> 4) & 0x03);
                int8_t a3 = decode_2bit((xb >> 6) & 0x03);
                
                // Build LUT
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
// 1-bit x 1-bit LUT-based MatMul (BNN)
// =============================================================================
//
// For BNN, XNOR + popcount is more efficient than LUT

__attribute__((weak)) void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        const uint8_t *x_row = (const uint8_t *)&x->data[i * in_features / 8];
        
        for (size_t j = 0; j < out_features; j++) {
            const uint8_t *w_row = (const uint8_t *)&w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Process 8 bits at a time using XNOR + popcount
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                uint8_t xb = x_row[k / 8];
                uint8_t wb = w_row[k / 8];
                
                // XNOR gives 1 when bits match (both represent same sign)
                // Result = 2 * popcount(XNOR) - 8
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
            
            // Handle remaining bits
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
// 4-bit activation x 2-bit weight LUT-based MatMul
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
            
            // Process 4 activations at a time (2 bytes of 4-bit acts, 1 byte of 2-bit weights)
            size_t k = 0;
            for (; k + 4 <= in_features; k += 4) {
                uint8_t xb0 = (uint8_t)x_row[k / 2];
                uint8_t xb1 = (uint8_t)x_row[k / 2 + 1];
                int8_t a0 = decode_4bit(xb0 & 0x0F);
                int8_t a1 = decode_4bit((xb0 >> 4) & 0x0F);
                int8_t a2 = decode_4bit(xb1 & 0x0F);
                int8_t a3 = decode_4bit((xb1 >> 4) & 0x0F);
                
                // Build LUT indexed by weight byte
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
// 4-bit activation x 1-bit weight LUT-based MatMul
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
            
            // Process 8 activations at a time (4 bytes of 4-bit acts, 1 byte of 1-bit weights)
            size_t k = 0;
            for (; k + 8 <= in_features; k += 8) {
                int8_t a[8];
                for (int n = 0; n < 4; n++) {
                    uint8_t xb = (uint8_t)x_row[k / 2 + n];
                    a[n * 2] = decode_4bit(xb & 0x0F);
                    a[n * 2 + 1] = decode_4bit((xb >> 4) & 0x0F);
                }
                
                // Build LUT
                for (int wb = 0; wb < 256; wb++) {
                    int32_t sum = 0;
                    for (int n = 0; n < 8; n++) {
                        int8_t w_val = decode_1bit((wb >> n) & 0x01);
                        sum += a[n] * w_val;
                    }
                    lut[wb] = sum;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 8];
                acc += lut[wb];
            }
            
            // Handle remaining elements
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
// 2-bit activation x 1-bit weight LUT-based MatMul
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
            
            // Process 8 activations at a time (2 bytes of 2-bit acts, 1 byte of 1-bit weights)
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
                
                // Build LUT
                for (int wb = 0; wb < 256; wb++) {
                    int32_t sum = 0;
                    for (int n = 0; n < 8; n++) {
                        int8_t w_val = decode_1bit((wb >> n) & 0x01);
                        sum += a[n] * w_val;
                    }
                    lut[wb] = sum;
                }
                
                uint8_t wb = (uint8_t)w_row[k / 8];
                acc += lut[wb];
            }
            
            // Handle remaining elements
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
// Reversed precision operations (activation bits < weight bits)
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
