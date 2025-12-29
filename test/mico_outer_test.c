// Test for Outer-Product MatMul implementations
// Compares outer-product results against locally-defined reference implementations
// that follow the same algorithmic logic as src/mico/qmatmul_ref.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mico_qnn.h"

// Test configuration
#define MAX_ERRORS_TO_PRINT 5
#define TEST_SEED 42

// Test dimensions
#ifndef N
#define N 4  // batch size
#endif
#ifndef M 
#define M 16  // output features
#endif
#ifndef K
#define K 32  // input features (must be multiple of 8 for 1-bit alignment)
#endif

// Reference implementations for testing
// These are simple, correct implementations to compare against

static void ref_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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

static void ref_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/2];
                temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += x->data[i * in_features + k] * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                acc += x->data[i * in_features + k] * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                acc += x->data[i * in_features + k] * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                int8_t temp_a = x->data[(i * in_features + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                int8_t temp_a = x->data[(i * in_features + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                int8_t temp_a = x->data[(i * in_features + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                int8_t temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                int8_t temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
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

// Reversed precision reference implementations

static void ref_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_a = x->data[(i * in_features + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc += temp_a * w->data[j * in_features + k];
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc += temp_a * w->data[j * in_features + k];
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_a = x->data[(i * in_features + k)/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc += temp_a * w->data[j * in_features + k];
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                int8_t temp_w = w->data[(j * in_features + k)/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_a = x->data[(i * in_features + k)/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                int8_t temp_w = w->data[(j * in_features + k)/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

static void ref_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k++) {
                int8_t temp_a = x->data[(i * in_features + k)/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                int8_t temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                acc += temp_a * temp_w;
            }
            O[i * out_features + j] = acc;
        }
    }
}

// Helper function to initialize random data
static void init_random_8bit(int8_t *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (int8_t)(rand() % 256 - 128);
    }
}

// Compare two output arrays
static int compare_outputs(const int32_t *expected, const int32_t *actual, size_t size, const char *test_name) {
    int errors = 0;
    for (size_t i = 0; i < size; i++) {
        if (expected[i] != actual[i]) {
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("[%s] Mismatch at index %zu: expected %d, got %d\n", 
                       test_name, i, expected[i], actual[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("[%s] PASSED\n", test_name);
    } else {
        printf("[%s] FAILED with %d errors\n", test_name, errors);
    }
    return errors;
}

int main() {
    printf("=== Outer-Product MatMul Regression Test ===\n");
    printf("Dimensions: N=%d, M=%d, K=%d\n\n", N, M, K);
    
    srand(TEST_SEED);  // Fixed seed for reproducibility
    
    int total_errors = 0;
    
    // Allocate buffers
    int8_t *x_data_8bit = (int8_t *)malloc(N * K * sizeof(int8_t));
    int8_t *w_data_8bit = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *x_data_4bit = (int8_t *)malloc(N * K / 2);
    int8_t *w_data_4bit = (int8_t *)malloc(M * K / 2);
    int8_t *x_data_2bit = (int8_t *)malloc(N * K / 4);
    int8_t *w_data_2bit = (int8_t *)malloc(M * K / 4);
    int8_t *x_data_1bit = (int8_t *)malloc(N * K / 8);
    int8_t *w_data_1bit = (int8_t *)malloc(M * K / 8);
    
    int32_t *output_ref = (int32_t *)malloc(N * M * sizeof(int32_t));
    int32_t *output_outer = (int32_t *)malloc(N * M * sizeof(int32_t));
    
    // Initialize random data
    init_random_8bit(x_data_8bit, N * K);
    init_random_8bit(w_data_8bit, M * K);
    init_random_8bit(x_data_4bit, N * K / 2);
    init_random_8bit(w_data_4bit, M * K / 2);
    init_random_8bit(x_data_2bit, N * K / 4);
    init_random_8bit(w_data_2bit, M * K / 4);
    init_random_8bit(x_data_1bit, N * K / 8);
    init_random_8bit(w_data_1bit, M * K / 8);
    
    // Create tensor structures
    Tensor2D_Q8 x_8bit = { .shape = {N, K}, .data = x_data_8bit, .scale = 1.0f, .wq = 8 };
    Tensor2D_Q8 w_8bit = { .shape = {M, K}, .data = w_data_8bit, .scale = 1.0f, .wq = 8 };
    Tensor2D_Q8 x_4bit = { .shape = {N, K}, .data = x_data_4bit, .scale = 1.0f, .wq = 4 };
    Tensor2D_Q8 w_4bit = { .shape = {M, K}, .data = w_data_4bit, .scale = 1.0f, .wq = 4 };
    Tensor2D_Q8 x_2bit = { .shape = {N, K}, .data = x_data_2bit, .scale = 1.0f, .wq = 2 };
    Tensor2D_Q8 w_2bit = { .shape = {M, K}, .data = w_data_2bit, .scale = 1.0f, .wq = 2 };
    Tensor2D_Q8 x_1bit = { .shape = {N, K}, .data = x_data_1bit, .scale = 1.0f, .wq = 1 };
    Tensor2D_Q8 w_1bit = { .shape = {M, K}, .data = w_data_1bit, .scale = 1.0f, .wq = 1 };
    
    // Test Q8 MatMul (8-bit activation, 8-bit weight)
    printf("Testing Q8_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q8_MatMul(output_ref, &x_8bit, &w_8bit);
    MiCo_Q8_MatMul(output_outer, &x_8bit, &w_8bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q8_MatMul");
    
    // Test Q8x4 MatMul (8-bit activation, 4-bit weight)
    printf("Testing Q8x4_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q8x4_MatMul(output_ref, &x_8bit, &w_4bit);
    MiCo_Q8x4_MatMul(output_outer, &x_8bit, &w_4bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q8x4_MatMul");
    
    // Test Q8x2 MatMul (8-bit activation, 2-bit weight)
    printf("Testing Q8x2_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q8x2_MatMul(output_ref, &x_8bit, &w_2bit);
    MiCo_Q8x2_MatMul(output_outer, &x_8bit, &w_2bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q8x2_MatMul");
    
    // Test Q8x1 MatMul (8-bit activation, 1-bit weight)
    printf("Testing Q8x1_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q8x1_MatMul(output_ref, &x_8bit, &w_1bit);
    MiCo_Q8x1_MatMul(output_outer, &x_8bit, &w_1bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q8x1_MatMul");
    
    // Test Q4 MatMul (4-bit activation, 4-bit weight)
    printf("Testing Q4_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q4_MatMul(output_ref, &x_4bit, &w_4bit);
    MiCo_Q4_MatMul(output_outer, &x_4bit, &w_4bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q4_MatMul");
    
    // Test Q4x2 MatMul (4-bit activation, 2-bit weight)
    printf("Testing Q4x2_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q4x2_MatMul(output_ref, &x_4bit, &w_2bit);
    MiCo_Q4x2_MatMul(output_outer, &x_4bit, &w_2bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q4x2_MatMul");
    
    // Test Q4x1 MatMul (4-bit activation, 1-bit weight)
    printf("Testing Q4x1_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q4x1_MatMul(output_ref, &x_4bit, &w_1bit);
    MiCo_Q4x1_MatMul(output_outer, &x_4bit, &w_1bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q4x1_MatMul");
    
    // Test Q2 MatMul (2-bit activation, 2-bit weight)
    printf("Testing Q2_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q2_MatMul(output_ref, &x_2bit, &w_2bit);
    MiCo_Q2_MatMul(output_outer, &x_2bit, &w_2bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q2_MatMul");
    
    // Test Q2x1 MatMul (2-bit activation, 1-bit weight)
    printf("Testing Q2x1_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q2x1_MatMul(output_ref, &x_2bit, &w_1bit);
    MiCo_Q2x1_MatMul(output_outer, &x_2bit, &w_1bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q2x1_MatMul");
    
    // Test Q1 MatMul (1-bit activation, 1-bit weight - BNN)
    printf("Testing Q1_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q1_MatMul(output_ref, &x_1bit, &w_1bit);
    MiCo_Q1_MatMul(output_outer, &x_1bit, &w_1bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q1_MatMul");
    
    // Test reversed precision operations
    
    // Test Q4x8 MatMul (4-bit activation, 8-bit weight)
    printf("Testing Q4x8_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q4x8_MatMul(output_ref, &x_4bit, &w_8bit);
    MiCo_Q4x8_MatMul(output_outer, &x_4bit, &w_8bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q4x8_MatMul");
    
    // Test Q2x8 MatMul (2-bit activation, 8-bit weight)
    printf("Testing Q2x8_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q2x8_MatMul(output_ref, &x_2bit, &w_8bit);
    MiCo_Q2x8_MatMul(output_outer, &x_2bit, &w_8bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q2x8_MatMul");
    
    // Test Q1x8 MatMul (1-bit activation, 8-bit weight)
    printf("Testing Q1x8_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q1x8_MatMul(output_ref, &x_1bit, &w_8bit);
    MiCo_Q1x8_MatMul(output_outer, &x_1bit, &w_8bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q1x8_MatMul");
    
    // Test Q2x4 MatMul (2-bit activation, 4-bit weight)
    printf("Testing Q2x4_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q2x4_MatMul(output_ref, &x_2bit, &w_4bit);
    MiCo_Q2x4_MatMul(output_outer, &x_2bit, &w_4bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q2x4_MatMul");
    
    // Test Q1x4 MatMul (1-bit activation, 4-bit weight)
    printf("Testing Q1x4_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q1x4_MatMul(output_ref, &x_1bit, &w_4bit);
    MiCo_Q1x4_MatMul(output_outer, &x_1bit, &w_4bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q1x4_MatMul");
    
    // Test Q1x2 MatMul (1-bit activation, 2-bit weight)
    printf("Testing Q1x2_MatMul...\n");
    memset(output_ref, 0, N * M * sizeof(int32_t));
    memset(output_outer, 0, N * M * sizeof(int32_t));
    ref_Q1x2_MatMul(output_ref, &x_1bit, &w_2bit);
    MiCo_Q1x2_MatMul(output_outer, &x_1bit, &w_2bit);
    total_errors += compare_outputs(output_ref, output_outer, N * M, "Q1x2_MatMul");
    
    // Clean up
    free(x_data_8bit);
    free(w_data_8bit);
    free(x_data_4bit);
    free(w_data_4bit);
    free(x_data_2bit);
    free(w_data_2bit);
    free(x_data_1bit);
    free(w_data_1bit);
    free(output_ref);
    free(output_outer);
    
    printf("\n=== Test Summary ===\n");
    if (total_errors == 0) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Tests FAILED with %d total errors\n", total_errors);
        return 1;
    }
}
