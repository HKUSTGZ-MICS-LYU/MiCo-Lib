#include "mico_qnn.h"
#include "profile.h"

#include <cuda_runtime.h>

__global__ void q8_matmul_kernel(int32_t *O, const qbyte *x, const qbyte *w, 
                                 size_t batch_size, size_t in_features, size_t out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row < batch_size && col < out_features) {
        int32_t acc = 0;
        for (size_t k = 0; k < in_features; k++) {
            acc += x[row * in_features + k] * w[col * in_features + k];
        }
        O[row * out_features + col] = acc;
    }
}

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    dim3 blockSize(16, 16);
    dim3 gridSize((out_features + blockSize.x - 1) / blockSize.x, 
                  (batch_size + blockSize.y - 1) / blockSize.y);
    qbyte *cx, *cw;
    cudaMalloc((void**)&cx, batch_size * in_features * sizeof(qbyte));
    cudaMalloc((void**)&cw, out_features * in_features * sizeof(qbyte));
    int32_t *cO;
    cudaMalloc((void**)&cO, batch_size * out_features * sizeof(int32_t));

    cudaMemcpy(cx, x->data, batch_size * in_features * sizeof(qbyte), cudaMemcpyHostToDevice);
    cudaMemcpy(cw, w->data, out_features * in_features * sizeof(qbyte), cudaMemcpyHostToDevice);    
    q8_matmul_kernel<<<gridSize, blockSize>>>(
        cO, cx, cw, batch_size, in_features, out_features);
    cudaDeviceSynchronize();
    cudaFree(cx);
    cudaFree(cw);
    cudaMemcpy(O, cO, batch_size * out_features * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(cO);
    return; 
}