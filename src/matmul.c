#include "nn.h"

// FP32 MatMul Kernel
__attribute__((weak)) void MiCo_MatMul_f32(
    float* y, const float* x, const float* w, 
    const size_t m, const size_t n, const size_t p){
        
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            for (size_t k = 0; k < n; k++) {
                #ifdef USE_ALT_LAYOUT
                y[i * p + j] += x[i * n + k] * w[k * p + j];
                #else
                y[i * p + j] += x[i * n + k] * w[j * n + k];
                #endif
            }
        }
    }
}