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

    // Detecting Weight Layout
    #ifdef USE_ALT_LAYOUT
    const size_t out_features = weight->shape[1];
    #else
    const size_t out_features = weight->shape[0];
    #endif
    
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

void MiCo_linear3d_f32(
    Tensor3D_F32 *y,
    const Tensor3D_F32 *x,
    const Tensor2D_F32 *weight,
    const Tensor1D_F32 *bias) {
    Tensor2D_F32 x2d;
    Tensor2D_F32 y2d;
    x2d.shape[0] = x->shape[0] * x->shape[1];
    x2d.shape[1] = x->shape[2];
    x2d.data = x->data;

    y2d.shape[0] = y->shape[0] * y->shape[1];
    y2d.shape[1] = y->shape[2];
    y2d.data = y->data;

    MiCo_linear_f32(&y2d, &x2d, weight, bias);
}
