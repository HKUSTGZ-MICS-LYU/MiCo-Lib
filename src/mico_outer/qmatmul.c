#include "mico_qnn.h"

// Outer-Product (Input-Stationary) Implementation of MatMuls
// This dataflow reuses input elements across multiple output computations
//
// Standard MatMul: O[batch, out] = sum_k X[batch, k] * W[out, k]
//
// Output-stationary (baseline in src/mico/qmatmul.c):
//   Loop order: for i (batch), for j (out_features), for k (in_features)
//   - Accumulator O[i,j] stays in register
//   - X[i,k] and W[j,k] are loaded for each k
//
// Input-stationary (outer-product, this file):
//   Loop order: for k (in_features), for i (batch), for j (out_features)
//   - X[i,k] is loaded once, broadcast to all j
//   - W[j,k] is loaded once, broadcast to all i
//   - O[i,j] is updated incrementally (rank-1 outer product update)
//
// Benefits:
//   - Better input reuse when batch_size and out_features are large
//   - Each X element and W element is loaded only once per k iteration
//   - Useful for certain hardware architectures with broadcast capabilities

// =============================================================================
// Q8 x Q8 Outer-Product MatMul
// =============================================================================

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order: k is the outermost loop
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t x_ik = x->data[i * in_features + k];
            for (size_t j = 0; j < out_features; j++) {
                O[i * out_features + j] += x_ik * w->data[j * in_features + k];
            }
        }
    }
}

// =============================================================================
// Q8 x Q4 Outer-Product MatMul (8-bit activation, 4-bit weight)
// =============================================================================

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t x_ik = x->data[i * in_features + k];
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 2];
                temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                O[i * out_features + j] += x_ik * temp_w;
            }
        }
    }
}

// =============================================================================
// Q8 x Q2 Outer-Product MatMul (8-bit activation, 2-bit weight)
// =============================================================================

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t x_ik = x->data[i * in_features + k];
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                O[i * out_features + j] += x_ik * temp_w;
            }
        }
    }
}

// =============================================================================
// Q8 x Q1 Outer-Product MatMul (8-bit activation, 1-bit weight)
// =============================================================================

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t x_ik = x->data[i * in_features + k];
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                O[i * out_features + j] += x_ik * temp_w;
            }
        }
    }
}

// =============================================================================
// Q4 x Q4 Outer-Product MatMul (4-bit activation, 4-bit weight)
// =============================================================================

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 2];
            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// =============================================================================
// Q4 x Q2 Outer-Product MatMul (4-bit activation, 2-bit weight)
// =============================================================================

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 2];
            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// =============================================================================
// Q4 x Q1 Outer-Product MatMul (4-bit activation, 1-bit weight)
// =============================================================================

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 2];
            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// =============================================================================
// Q2 x Q2 Outer-Product MatMul (2-bit activation, 2-bit weight)
// =============================================================================

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 4];
            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
            temp_a = TWO_BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// =============================================================================
// Q2 x Q1 Outer-Product MatMul (2-bit activation, 1-bit weight)
// =============================================================================

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 4];
            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
            temp_a = TWO_BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// =============================================================================
// Q1 x Q1 Outer-Product MatMul (1-bit activation, 1-bit weight - BNN)
// =============================================================================

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 8];
            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
            temp_a = BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// =============================================================================
// Reversed precision operations (low activation bits x high weight bits)
// =============================================================================

// Q4 x Q8 Outer-Product MatMul (4-bit activation, 8-bit weight)
void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 2];
            temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
            temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
            for (size_t j = 0; j < out_features; j++) {
                O[i * out_features + j] += temp_a * w->data[j * in_features + k];
            }
        }
    }
}

// Q2 x Q8 Outer-Product MatMul (2-bit activation, 8-bit weight)
void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 4];
            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
            temp_a = TWO_BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                O[i * out_features + j] += temp_a * w->data[j * in_features + k];
            }
        }
    }
}

// Q1 x Q8 Outer-Product MatMul (1-bit activation, 8-bit weight)
void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 8];
            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
            temp_a = BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                O[i * out_features + j] += temp_a * w->data[j * in_features + k];
            }
        }
    }
}

// Q2 x Q4 Outer-Product MatMul (2-bit activation, 4-bit weight)
void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 4];
            temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
            temp_a = TWO_BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// Q1 x Q4 Outer-Product MatMul (1-bit activation, 4-bit weight)
void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 8];
            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
            temp_a = BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}

// Q1 x Q2 Outer-Product MatMul (1-bit activation, 2-bit weight)
void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {

    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Initialize output to zero
    for (size_t i = 0; i < batch_size * out_features; i++) {
        O[i] = 0;
    }

    // Outer-product loop order
    for (size_t k = 0; k < in_features; k++) {
        for (size_t i = 0; i < batch_size; i++) {
            int8_t temp_a = x->data[(i * in_features + k) / 8];
            temp_a = EXTRACT_BIT(temp_a, k & 0b111);
            temp_a = BIT_TO_INT8(temp_a);
            for (size_t j = 0; j < out_features; j++) {
                int8_t temp_w = w->data[(j * in_features + k) / 4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                O[i * out_features + j] += temp_a * temp_w;
            }
        }
    }
}
