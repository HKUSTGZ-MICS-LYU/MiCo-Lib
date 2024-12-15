#include "nn.h"

// FP32 MatMul Kernel
// Unrolled Implementation of 8-bit MatMul
// The most simple optimization to apply on MatMul
void MiCo_MatMul_f32(
    float* y, const float* x, const float* w, 
    const size_t m, const size_t n, const size_t p){
        
    size_t unrolled_end = (n / 4) * 4;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            float sum = 0;
            for (size_t k = 0; k < unrolled_end; k+=4) {
                sum += x[i * n + k] * w[j * n + k];
                sum += x[i * n + k+1] * w[j * n + k+1];
                sum += x[i * n + k+2] * w[j * n + k+2];
                sum += x[i * n + k+3] * w[j * n + k+3];
            }
            for (size_t k = unrolled_end; k < n; k++) {
                sum += x[i * n + k] * w[j * n + k];
            }
            y[i * p + j] += sum;
        }
    }
}