#include "nn.h"

__attribute__((weak)) void MiCo_MatMul_f32(
    float* y, const float* x, const float* w, 
    const size_t n, const size_t k, const size_t m){
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            for (size_t l = 0; l < k; l++) {
                y[i * m + j] += x[i * k + l] * w[l * m + j];
            }
        }
    }
}