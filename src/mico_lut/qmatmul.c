#include "mico_qnn.h"

// LUT-based Mixed Precision MatMul Kernels
// Based on the T-MAC approach: https://github.com/microsoft/T-MAC
//
// T-MAC Key Insight:
// For low-bitwidth weights, precompute partial sums for groups of activations.
// The weight bits become an INDEX into this precomputed LUT, eliminating multiplies.
//
// IMPORTANT: LUTs are built ONCE per activation group, OUTSIDE the output loop.
// The LUT construction cost is amortized across ALL output features.
//
// Loop order for optimal performance:
// 1. For each batch sample
// 2. Precompute ALL LUTs for activation groups (OUTSIDE output loop)
// 3. For each output feature, accumulate using precomputed LUTs

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
// LUT building helpers (called ONCE per activation group)
// =============================================================================

// Build 256-entry LUT for 8 activations with 1-bit weights (sign pattern)
static inline void build_lut_8x1(int32_t *lut, const int8_t *a) {
    // Precompute sum for fast LUT building
    int32_t sum_all = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    
    for (int idx = 0; idx < 256; idx++) {
        int32_t result = sum_all;
        // Each bit flips sign from +1 to -1 (subtract 2*a[bit])
        if (idx & 0x01) result -= 2 * a[0];
        if (idx & 0x02) result -= 2 * a[1];
        if (idx & 0x04) result -= 2 * a[2];
        if (idx & 0x08) result -= 2 * a[3];
        if (idx & 0x10) result -= 2 * a[4];
        if (idx & 0x20) result -= 2 * a[5];
        if (idx & 0x40) result -= 2 * a[6];
        if (idx & 0x80) result -= 2 * a[7];
        lut[idx] = result;
    }
}

// Build 256-entry LUT for 4 activations with 2-bit weights
static inline void build_lut_4x2(int32_t *lut, int8_t a0, int8_t a1, int8_t a2, int8_t a3) {
    for (int wb = 0; wb < 256; wb++) {
        int8_t w0 = decode_2bit((wb >> 0) & 0x03);
        int8_t w1 = decode_2bit((wb >> 2) & 0x03);
        int8_t w2 = decode_2bit((wb >> 4) & 0x03);
        int8_t w3 = decode_2bit((wb >> 6) & 0x03);
        lut[wb] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
    }
}

// Build 256-entry LUT for 2 activations with 4-bit weights
static inline void build_lut_2x4(int32_t *lut, int8_t a0, int8_t a1) {
    for (int wb = 0; wb < 256; wb++) {
        int8_t w0 = decode_4bit(wb & 0x0F);
        int8_t w1 = decode_4bit((wb >> 4) & 0x0F);
        lut[wb] = a0 * w0 + a1 * w1;
    }
}

// =============================================================================
// 8-bit activation x 1-bit weight LUT-based MatMul
// =============================================================================
// LUTs are built ONCE per activation group, then reused for ALL outputs

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 8;
    
    // Allocate LUT storage for all groups
    int32_t lut_storage[256 * 64];
    int32_t *luts = (num_groups <= 64) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 64) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        // STEP 1: Build ALL LUTs for this batch (OUTSIDE output loop)
        for (size_t g = 0; g < num_groups; g++) {
            build_lut_8x1(&luts[g * 256], &x_row[g * 8]);
        }
        
        // STEP 2: Process each output using precomputed LUTs
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            // Use precomputed LUTs - just lookup and add
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            // Handle remaining elements (< 8)
            for (size_t k = num_groups * 8; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t w_val = decode_1bit((wb >> (k & 0x07)) & 0x01);
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 64) free(luts);
}

// =============================================================================
// 8-bit activation x 2-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 4;
    
    int32_t lut_storage[256 * 64];
    int32_t *luts = (num_groups <= 64) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 64) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        // STEP 1: Build ALL LUTs (OUTSIDE output loop)
        for (size_t g = 0; g < num_groups; g++) {
            build_lut_4x2(&luts[g * 256], 
                         x_row[g * 4], x_row[g * 4 + 1], 
                         x_row[g * 4 + 2], x_row[g * 4 + 3]);
        }
        
        // STEP 2: Process outputs using precomputed LUTs
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            for (size_t k = num_groups * 4; k < in_features; k++) {
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += x_row[k] * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 64) free(luts);
}

// =============================================================================
// 8-bit activation x 4-bit weight LUT-based MatMul
// =============================================================================

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 2;
    
    int32_t lut_storage[256 * 128];
    int32_t *luts = (num_groups <= 128) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 128) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features];
        
        // Build ALL LUTs
        for (size_t g = 0; g < num_groups; g++) {
            build_lut_2x4(&luts[g * 256], x_row[g * 2], x_row[g * 2 + 1]);
        }
        
        // Process outputs
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            if (in_features & 1) {
                size_t k = num_groups * 2;
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t w0 = decode_4bit(wb & 0x0F);
                acc += x_row[k] * w0;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 128) free(luts);
}

// =============================================================================
// Standard 8-bit MatMul (no LUT needed)
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
// 4-bit x 4-bit LUT-based MatMul
// =============================================================================

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 2;
    
    int32_t lut_storage[256 * 128];
    int32_t *luts = (num_groups <= 128) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 128) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        // Build LUTs for 4-bit activations
        for (size_t g = 0; g < num_groups; g++) {
            uint8_t xb = (uint8_t)x_row[g];
            int8_t a0 = decode_4bit(xb & 0x0F);
            int8_t a1 = decode_4bit((xb >> 4) & 0x0F);
            build_lut_2x4(&luts[g * 256], a0, a1);
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 2];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            if (in_features & 1) {
                size_t k = num_groups * 2;
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 2];
                int8_t a0 = decode_4bit(xb & 0x0F);
                int8_t w0 = decode_4bit(wb & 0x0F);
                acc += a0 * w0;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 128) free(luts);
}

// =============================================================================
// 2-bit x 2-bit LUT-based MatMul
// =============================================================================

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 4;
    
    int32_t lut_storage[256 * 64];
    int32_t *luts = (num_groups <= 64) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 64) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t g = 0; g < num_groups; g++) {
            uint8_t xb = (uint8_t)x_row[g];
            int8_t a0 = decode_2bit((xb >> 0) & 0x03);
            int8_t a1 = decode_2bit((xb >> 2) & 0x03);
            int8_t a2 = decode_2bit((xb >> 4) & 0x03);
            int8_t a3 = decode_2bit((xb >> 6) & 0x03);
            build_lut_4x2(&luts[g * 256], a0, a1, a2, a3);
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            for (size_t k = num_groups * 4; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = decode_2bit((xb >> (2 * (k & 0x03))) & 0x03);
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 64) free(luts);
}

// =============================================================================
// 1-bit x 1-bit MatMul (BNN - uses XNOR + popcount)
// =============================================================================

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 4;
    
    int32_t lut_storage[256 * 64];
    int32_t *luts = (num_groups <= 64) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 64) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        // Build LUTs: 4 4-bit activations -> LUT indexed by 2-bit weight byte
        for (size_t g = 0; g < num_groups; g++) {
            uint8_t xb0 = (uint8_t)x_row[g * 2];
            uint8_t xb1 = (uint8_t)x_row[g * 2 + 1];
            int8_t a0 = decode_4bit(xb0 & 0x0F);
            int8_t a1 = decode_4bit((xb0 >> 4) & 0x0F);
            int8_t a2 = decode_4bit(xb1 & 0x0F);
            int8_t a3 = decode_4bit((xb1 >> 4) & 0x0F);
            build_lut_4x2(&luts[g * 256], a0, a1, a2, a3);
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 4];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            for (size_t k = num_groups * 4; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 4];
                int8_t a_val = decode_4bit((xb >> (4 * (k & 0x01))) & 0x0F);
                int8_t w_val = decode_2bit((wb >> (2 * (k & 0x03))) & 0x03);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 64) free(luts);
}

// =============================================================================
// Mixed precision: 4-bit activation x 1-bit weight
// =============================================================================

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 8;
    
    int32_t lut_storage[256 * 32];
    int32_t *luts = (num_groups <= 32) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 32) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 2];
        
        // Build LUTs: 8 4-bit activations -> LUT indexed by 1-bit weight byte
        for (size_t g = 0; g < num_groups; g++) {
            int8_t a[8];
            for (int n = 0; n < 4; n++) {
                uint8_t xb = (uint8_t)x_row[g * 4 + n];
                a[n * 2] = decode_4bit(xb & 0x0F);
                a[n * 2 + 1] = decode_4bit((xb >> 4) & 0x0F);
            }
            build_lut_8x1(&luts[g * 256], a);
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            for (size_t k = num_groups * 8; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 2];
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t a_val = decode_4bit((xb >> (4 * (k & 0x01))) & 0x0F);
                int8_t w_val = decode_1bit((wb >> (k & 0x07)) & 0x01);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 32) free(luts);
}

// =============================================================================
// Mixed precision: 2-bit activation x 1-bit weight
// =============================================================================

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t num_groups = in_features / 8;
    
    int32_t lut_storage[256 * 32];
    int32_t *luts = (num_groups <= 32) ? lut_storage : (int32_t*)malloc(num_groups * 256 * sizeof(int32_t));
    if (luts == NULL && num_groups > 32) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        const int8_t *x_row = &x->data[i * in_features / 4];
        
        for (size_t g = 0; g < num_groups; g++) {
            int8_t a[8];
            uint8_t xb0 = (uint8_t)x_row[g * 2];
            uint8_t xb1 = (uint8_t)x_row[g * 2 + 1];
            a[0] = decode_2bit((xb0 >> 0) & 0x03);
            a[1] = decode_2bit((xb0 >> 2) & 0x03);
            a[2] = decode_2bit((xb0 >> 4) & 0x03);
            a[3] = decode_2bit((xb0 >> 6) & 0x03);
            a[4] = decode_2bit((xb1 >> 0) & 0x03);
            a[5] = decode_2bit((xb1 >> 2) & 0x03);
            a[6] = decode_2bit((xb1 >> 4) & 0x03);
            a[7] = decode_2bit((xb1 >> 6) & 0x03);
            build_lut_8x1(&luts[g * 256], a);
        }
        
        for (size_t j = 0; j < out_features; j++) {
            const int8_t *w_row = &w->data[j * in_features / 8];
            int32_t acc = 0;
            
            for (size_t g = 0; g < num_groups; g++) {
                uint8_t wb = (uint8_t)w_row[g];
                acc += luts[g * 256 + wb];
            }
            
            for (size_t k = num_groups * 8; k < in_features; k++) {
                uint8_t xb = (uint8_t)x_row[k / 4];
                uint8_t wb = (uint8_t)w_row[k / 8];
                int8_t a_val = decode_2bit((xb >> (2 * (k & 0x03))) & 0x03);
                int8_t w_val = decode_1bit((wb >> (k & 0x07)) & 0x01);
                acc += a_val * w_val;
            }
            
            O[i * out_features + j] = acc;
        }
    }
    
    if (num_groups > 32) free(luts);
}

// =============================================================================
// Reversed precision (activation bits < weight bits) - direct computation
// LUT doesn't help much when weight space is large
// =============================================================================

void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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
