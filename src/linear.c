#include "nn.h"

__attribute__((weak)) void MiCo_linear_f32(
    Tensor2D_F32 *y, 
    const Tensor2D_F32 *x, 
    const Tensor2D_F32 *weight, 
    const Tensor1D_F32 *bias) { 
    // nn_assert(x->shape[1] == weight->shape[1], "Cannot perform Linear on tensors of different shapes");
    // nn_assert(bias->shape[0] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
    // nn_assert(y->shape[0] == x->shape[0] && y->shape[1] == weight->shape[0], "Cannot perform Linear on tensors of different shapes");
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = weight->shape[0];

    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_features; i++) {
            y->data[i] = 0.f;
        }
    } else {
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                y->data[i * out_features + j] = bias->data[j];
            }
        }
    }

    MiCo_MatMul_f32(y->data, x->data, weight->data, 
        batch_size, in_features, out_features);
}