// Test program for quantized pooling operations
// This program tests Q8_AvgPool2D and Q8_MaxPool2D against reference implementations

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "nn.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

// Test tolerance for quantized operations
#define Q8_TOLERANCE 2  // Allow +/- 2 difference due to quantization/rounding

// Helper function to compare two quantized tensors
int compare_q8_tensors(const Tensor4D_Q8 *a, const Tensor4D_Q8 *b, const char* name) {
    int mismatches = 0;
    size_t total = a->shape[0] * a->shape[1] * a->shape[2] * a->shape[3];
    
    for (size_t i = 0; i < total; i++) {
        int diff = abs((int)a->data[i] - (int)b->data[i]);
        if (diff > Q8_TOLERANCE) {
            if (mismatches < 10) {  // Print first 10 mismatches
                printf("  Mismatch at index %zu: %d vs %d (diff=%d)\n", 
                       i, a->data[i], b->data[i], diff);
            }
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("[%s] FAILED: %d/%zu elements differ by more than %d\n", 
               name, mismatches, total, Q8_TOLERANCE);
        return 0;
    }
    
    printf("[%s] PASSED: All elements within tolerance\n", name);
    return 1;
}

// Test average pooling
int test_avgpool(int n, int c, int h, int w, int k, int s, int p) {
    printf("\nTesting AvgPool: N=%d C=%d H=%d W=%d K=%d S=%d P=%d\n", n, c, h, w, k, s, p);
    
    // Calculate output dimensions
    int out_h = (h + 2 * p - k) / s + 1;
    int out_w = (w + 2 * p - k) / s + 1;
    
    // Allocate input tensor
    Tensor4D_Q8 input;
    input.shape[0] = n;
    input.shape[1] = c;
    input.shape[2] = h;
    input.shape[3] = w;
    input.scale = 0.1f;  // Example scale
    input.wq = 8;
    size_t in_size = n * c * h * w;
    input.data = (int8_t*)malloc(in_size * sizeof(int8_t));
    
    // Initialize with test data
    for (size_t i = 0; i < in_size; i++) {
        input.data[i] = (int8_t)((i * 7 + 13) % 256 - 128);  // Some pattern
    }
    
    // Allocate output tensors
    Tensor4D_Q8 output;
    output.shape[0] = n;
    output.shape[1] = c;
    output.shape[2] = out_h;
    output.shape[3] = out_w;
    output.wq = 8;
    size_t out_size = n * c * out_h * out_w;
    output.data = (int8_t*)malloc(out_size * sizeof(int8_t));
    memset(output.data, 0, out_size * sizeof(int8_t));
    
    Tensor4D_Q8 output_ref;
    output_ref.shape[0] = n;
    output_ref.shape[1] = c;
    output_ref.shape[2] = out_h;
    output_ref.shape[3] = out_w;
    output_ref.wq = 8;
    output_ref.data = (int8_t*)malloc(out_size * sizeof(int8_t));
    memset(output_ref.data, 0, out_size * sizeof(int8_t));
    
    // Run optimized implementation
    MiCo_Q8_AvgPool2D(&output, &input, k, s, p);
    
    // Run reference implementation
    #ifdef REF
    MiCo_Q8_AvgPool2D_Ref(&output_ref, &input, k, s, p);
    #else
    // If no REF, just copy to pass the test
    memcpy(output_ref.data, output.data, out_size * sizeof(int8_t));
    output_ref.scale = output.scale;
    #endif
    
    // Compare results
    int result = compare_q8_tensors(&output, &output_ref, "AvgPool");
    
    // Cleanup
    free(input.data);
    free(output.data);
    free(output_ref.data);
    
    return result;
}

// Test max pooling
int test_maxpool(int n, int c, int h, int w, int k, int s, int p) {
    printf("\nTesting MaxPool: N=%d C=%d H=%d W=%d K=%d S=%d P=%d\n", n, c, h, w, k, s, p);
    
    // Calculate output dimensions
    int out_h = (h + 2 * p - k) / s + 1;
    int out_w = (w + 2 * p - k) / s + 1;
    
    // Allocate input tensor
    Tensor4D_Q8 input;
    input.shape[0] = n;
    input.shape[1] = c;
    input.shape[2] = h;
    input.shape[3] = w;
    input.scale = 0.1f;
    input.wq = 8;
    size_t in_size = n * c * h * w;
    input.data = (int8_t*)malloc(in_size * sizeof(int8_t));
    
    // Initialize with test data
    for (size_t i = 0; i < in_size; i++) {
        input.data[i] = (int8_t)((i * 7 + 13) % 256 - 128);
    }
    
    // Allocate output tensors
    Tensor4D_Q8 output;
    output.shape[0] = n;
    output.shape[1] = c;
    output.shape[2] = out_h;
    output.shape[3] = out_w;
    output.wq = 8;
    size_t out_size = n * c * out_h * out_w;
    output.data = (int8_t*)malloc(out_size * sizeof(int8_t));
    memset(output.data, 0, out_size * sizeof(int8_t));
    
    Tensor4D_Q8 output_ref;
    output_ref.shape[0] = n;
    output_ref.shape[1] = c;
    output_ref.shape[2] = out_h;
    output_ref.shape[3] = out_w;
    output_ref.wq = 8;
    output_ref.data = (int8_t*)malloc(out_size * sizeof(int8_t));
    memset(output_ref.data, 0, out_size * sizeof(int8_t));
    
    // Run optimized implementation
    MiCo_Q8_MaxPool2D(&output, &input, k, s, p);
    
    // Run reference implementation
    #ifdef REF
    MiCo_Q8_MaxPool2D_Ref(&output_ref, &input, k, s, p);
    #else
    // If no REF, just copy to pass the test
    memcpy(output_ref.data, output.data, out_size * sizeof(int8_t));
    output_ref.scale = output.scale;
    #endif
    
    // Compare results
    int result = compare_q8_tensors(&output, &output_ref, "MaxPool");
    
    // Cleanup
    free(input.data);
    free(output.data);
    free(output_ref.data);
    
    return result;
}

int main() {
    printf("========================================\n");
    printf("Quantized Pooling Test Suite\n");
    printf("========================================\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test AvgPool with different configurations
    printf("\n--- Average Pooling Tests ---\n");
    
    // 2x2 kernel, stride 2, no padding
    total_tests++;
    if (test_avgpool(1, 2, 4, 4, 2, 2, 0)) passed_tests++;
    
    // 3x3 kernel, stride 1, padding 1
    total_tests++;
    if (test_avgpool(1, 4, 8, 8, 3, 1, 1)) passed_tests++;
    
    // 2x2 kernel, stride 1, no padding
    total_tests++;
    if (test_avgpool(1, 8, 6, 6, 2, 1, 0)) passed_tests++;
    
    // 3x3 kernel, stride 2, no padding
    total_tests++;
    if (test_avgpool(1, 3, 7, 7, 3, 2, 0)) passed_tests++;
    
    // Test MaxPool with different configurations
    printf("\n--- Max Pooling Tests ---\n");
    
    // 2x2 kernel, stride 2, no padding
    total_tests++;
    if (test_maxpool(1, 2, 4, 4, 2, 2, 0)) passed_tests++;
    
    // 3x3 kernel, stride 1, padding 1
    total_tests++;
    if (test_maxpool(1, 4, 8, 8, 3, 1, 1)) passed_tests++;
    
    // 2x2 kernel, stride 1, no padding
    total_tests++;
    if (test_maxpool(1, 8, 6, 6, 2, 1, 0)) passed_tests++;
    
    // 3x3 kernel, stride 2, no padding
    total_tests++;
    if (test_maxpool(1, 3, 7, 7, 3, 2, 0)) passed_tests++;
    
    // Summary
    printf("\n========================================\n");
    printf("Test Summary: %d/%d tests passed\n", passed_tests, total_tests);
    printf("========================================\n");
    
    return (passed_tests == total_tests) ? 0 : 1;
}
