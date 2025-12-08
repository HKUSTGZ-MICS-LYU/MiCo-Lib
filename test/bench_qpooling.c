// Benchmark for quantized pooling operations
// This program measures the performance of Q8_AvgPool2D and Q8_MaxPool2D

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#include "nn.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 100

// Helper to get time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Benchmark a pooling operation
double benchmark_pooling(void (*pool_func)(Tensor4D_Q8*, const Tensor4D_Q8*, size_t, size_t, size_t),
                         int n, int c, int h, int w, int k, int s, int p, const char* name) {
    
    // Calculate output dimensions
    int out_h = (h + 2 * p - k) / s + 1;
    int out_w = (w + 2 * p - k) / s + 1;
    
    // Allocate and initialize input
    Tensor4D_Q8 input;
    input.shape[0] = n;
    input.shape[1] = c;
    input.shape[2] = h;
    input.shape[3] = w;
    input.scale = 0.05f;
    input.wq = 8;
    size_t in_size = n * c * h * w;
    input.data = (int8_t*)malloc(in_size * sizeof(int8_t));
    
    // Initialize with test data
    for (size_t i = 0; i < in_size; i++) {
        input.data[i] = (int8_t)((i * 7 + 13) % 256 - 128);
    }
    
    // Allocate output
    Tensor4D_Q8 output;
    output.shape[0] = n;
    output.shape[1] = c;
    output.shape[2] = out_h;
    output.shape[3] = out_w;
    output.wq = 8;
    size_t out_size = n * c * out_h * out_w;
    output.data = (int8_t*)malloc(out_size * sizeof(int8_t));
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        pool_func(&output, &input, k, s, p);
    }
    
    // Benchmark
    double start = get_time();
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        pool_func(&output, &input, k, s, p);
    }
    double end = get_time();
    double elapsed = (end - start) / BENCHMARK_ITERATIONS;
    
    // Calculate throughput
    size_t ops_per_pool = (size_t)out_h * out_w * c * k * k;  // Operations per output element * num outputs
    double gops = (ops_per_pool * 1e-9) / elapsed;
    
    printf("[%s] N=%d C=%d H=%dx%d K=%d S=%d P=%d\n", name, n, c, h, w, k, s, p);
    printf("  Output: %dx%dx%dx%d\n", n, c, out_h, out_w);
    printf("  Time: %.6f ms\n", elapsed * 1000);
    printf("  Throughput: %.3f GOps/s\n", gops);
    printf("\n");
    
    // Cleanup
    free(input.data);
    free(output.data);
    
    return elapsed;
}

int main() {
    printf("========================================\n");
    printf("Quantized Pooling Benchmark\n");
    printf("========================================\n");
    printf("Warmup iterations: %d\n", WARMUP_ITERATIONS);
    printf("Benchmark iterations: %d\n", BENCHMARK_ITERATIONS);
    printf("========================================\n\n");
    
    // Common MobileNet/ResNet pooling configurations
    printf("--- Typical CNN Pooling Scenarios ---\n\n");
    
    // Early layer pooling (larger feature maps)
    benchmark_pooling(MiCo_Q8_MaxPool2D, 1, 64, 112, 112, 3, 2, 1, "MaxPool 3x3/2 (Early)");
    benchmark_pooling(MiCo_Q8_AvgPool2D, 1, 64, 112, 112, 3, 2, 1, "AvgPool 3x3/2 (Early)");
    
    // Mid layer pooling
    benchmark_pooling(MiCo_Q8_MaxPool2D, 1, 128, 56, 56, 2, 2, 0, "MaxPool 2x2/2 (Mid)");
    benchmark_pooling(MiCo_Q8_AvgPool2D, 1, 128, 56, 56, 2, 2, 0, "AvgPool 2x2/2 (Mid)");
    
    // Later layer pooling (smaller feature maps)
    benchmark_pooling(MiCo_Q8_MaxPool2D, 1, 256, 28, 28, 2, 2, 0, "MaxPool 2x2/2 (Late)");
    benchmark_pooling(MiCo_Q8_AvgPool2D, 1, 256, 28, 28, 2, 2, 0, "AvgPool 2x2/2 (Late)");
    
    // Global pooling (very small feature maps)
    benchmark_pooling(MiCo_Q8_MaxPool2D, 1, 512, 7, 7, 2, 1, 0, "MaxPool 2x2/1 (Global)");
    benchmark_pooling(MiCo_Q8_AvgPool2D, 1, 512, 7, 7, 2, 1, 0, "AvgPool 2x2/1 (Global)");
    
    printf("========================================\n");
    printf("Benchmark Complete\n");
    printf("========================================\n");
    
    return 0;
}
