#include "mico_qnn.h"

// Tiled Output-Stationary Implementation of MatMuls
// This dataflow improves cache utilization by dividing the computation into tiles
//
// Standard MatMul: O[batch, out] = sum_k X[batch, k] * W[out, k]
//
// Output-stationary (baseline in src/mico/qmatmul.c):
//   Loop order: for i (batch), for j (out_features), for k (in_features)
//   - Accumulator O[i,j] stays in register
//   - Poor cache locality for large matrices
//
// Tiled output-stationary (this file):
//   Loop order: for i_tile, for j_tile, for k_tile, for i, for j, for k
//   - Divides computation into tiles for better cache utilization
//   - Accumulator for each output tile element stays in register
//   - Better data reuse within tiles
//   - Reduces cache misses for large matrices
//
// Benefits:
//   - Better cache utilization compared to non-tiled version
//   - Maintains output-stationary property (minimizes output writes)
//   - Configurable tile sizes for different cache hierarchies
//   - Good balance between input reuse and output accumulation

// Default tile sizes (can be overridden at compile time)
#ifndef TILE_I
#define TILE_I 4  // Batch tile size
#endif
#ifndef TILE_J
#define TILE_J 4  // Output tile size
#endif
#ifndef TILE_K
#define TILE_K 8  // Reduction tile size
#endif

// Helper macro for minimum
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// =============================================================================
// Q8 x Q8 Tiled Output-Stationary MatMul
// =============================================================================

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop: iterate over tiles
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                // Process tile
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            acc += x->data[i * in_features + k] * 
                                   w->data[j * in_features + k];
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q8 x Q4 Tiled Output-Stationary MatMul (8-bit activation, 4-bit weight)
// =============================================================================

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_w = w->data[(j * in_features + k) / 2];
                            temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                            temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                            acc += x->data[i * in_features + k] * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q8 x Q2 Tiled Output-Stationary MatMul (8-bit activation, 2-bit weight)
// =============================================================================

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_w = w->data[(j * in_features + k) / 4];
                            temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                            temp_w = TWO_BIT_TO_INT8(temp_w);
                            acc += x->data[i * in_features + k] * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q8 x Q1 Tiled Output-Stationary MatMul (8-bit activation, 1-bit weight)
// =============================================================================

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_w = w->data[(j * in_features + k) / 8];
                            temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                            temp_w = BIT_TO_INT8(temp_w);
                            acc += x->data[i * in_features + k] * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q4 x Q4 Tiled Output-Stationary MatMul (4-bit activation, 4-bit weight)
// =============================================================================

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 2];
                            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                            int8_t temp_w = w->data[(j * in_features + k) / 2];
                            temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                            temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q4 x Q2 Tiled Output-Stationary MatMul (4-bit activation, 2-bit weight)
// =============================================================================

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 2];
                            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                            int8_t temp_w = w->data[(j * in_features + k) / 4];
                            temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                            temp_w = TWO_BIT_TO_INT8(temp_w);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q4 x Q1 Tiled Output-Stationary MatMul (4-bit activation, 1-bit weight)
// =============================================================================

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 2];
                            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                            int8_t temp_w = w->data[(j * in_features + k) / 8];
                            temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                            temp_w = BIT_TO_INT8(temp_w);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q2 x Q2 Tiled Output-Stationary MatMul (2-bit activation, 2-bit weight)
// =============================================================================

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 4];
                            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                            temp_a = TWO_BIT_TO_INT8(temp_a);
                            int8_t temp_w = w->data[(j * in_features + k) / 4];
                            temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                            temp_w = TWO_BIT_TO_INT8(temp_w);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q2 x Q1 Tiled Output-Stationary MatMul (2-bit activation, 1-bit weight)
// =============================================================================

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 4];
                            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                            temp_a = TWO_BIT_TO_INT8(temp_a);
                            int8_t temp_w = w->data[(j * in_features + k) / 8];
                            temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                            temp_w = BIT_TO_INT8(temp_w);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Q1 x Q1 Tiled Output-Stationary MatMul (1-bit activation, 1-bit weight - BNN)
// =============================================================================

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 8];
                            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                            temp_a = BIT_TO_INT8(temp_a);
                            int8_t temp_w = w->data[(j * in_features + k) / 8];
                            temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                            temp_w = BIT_TO_INT8(temp_w);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Reversed precision operations (low activation bits x high weight bits)
// =============================================================================

// Q4 x Q8 Tiled Output-Stationary MatMul (4-bit activation, 8-bit weight)
void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 2];
                            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                            acc += temp_a * w->data[j * in_features + k];
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// Q2 x Q8 Tiled Output-Stationary MatMul (2-bit activation, 8-bit weight)
void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 4];
                            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                            temp_a = TWO_BIT_TO_INT8(temp_a);
                            acc += temp_a * w->data[j * in_features + k];
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// Q1 x Q8 Tiled Output-Stationary MatMul (1-bit activation, 8-bit weight)
void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 8];
                            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                            temp_a = BIT_TO_INT8(temp_a);
                            acc += temp_a * w->data[j * in_features + k];
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// Q2 x Q4 Tiled Output-Stationary MatMul (2-bit activation, 4-bit weight)
void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 4];
                            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                            temp_a = TWO_BIT_TO_INT8(temp_a);
                            int8_t temp_w = w->data[(j * in_features + k) / 2];
                            temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                            temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// Q1 x Q4 Tiled Output-Stationary MatMul (1-bit activation, 4-bit weight)
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 8];
                            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                            temp_a = BIT_TO_INT8(temp_a);
                            int8_t temp_w = w->data[(j * in_features + k) / 2];
                            temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                            temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}

// Q1 x Q2 Tiled Output-Stationary MatMul (1-bit activation, 2-bit weight)
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Tiled loop
    for (size_t i0 = 0; i0 < batch_size; i0 += TILE_I) {
        for (size_t j0 = 0; j0 < out_features; j0 += TILE_J) {
            for (size_t k0 = 0; k0 < in_features; k0 += TILE_K) {
                size_t i_end = MIN(i0 + TILE_I, batch_size);
                size_t j_end = MIN(j0 + TILE_J, out_features);
                size_t k_end = MIN(k0 + TILE_K, in_features);

                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        int32_t acc = O[i * out_features + j];
                        for (size_t k = k0; k < k_end; k++) {
                            int8_t temp_a = x->data[(i * in_features + k) / 8];
                            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                            temp_a = BIT_TO_INT8(temp_a);
                            int8_t temp_w = w->data[(j * in_features + k) / 4];
                            temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                            temp_w = TWO_BIT_TO_INT8(temp_w);
                            acc += temp_a * temp_w;
                        }
                        O[i * out_features + j] = acc;
                    }
                }
            }
        }
    }
}
