#include "nn.h"

// Convolution Functions with Layout NCHW
void MiCo_conv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_F32* weight, const Tensor1D_F32* bias, 
    const size_t stride, const size_t padding, const size_t dilation, const size_t groups){
    // groups and dilation are not implemented yet
    size_t batch_size = x->shape[0];

    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    size_t k_h = weight->shape[2];
    size_t k_w = weight->shape[3];

    size_t out_c = y->shape[1];
    size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;
    
    size_t feature_size = in_h * in_w;
    size_t kernel_size = k_h * k_w;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");

    size_t in_c_per_group = in_c / groups;
    size_t out_c_per_group = out_c / groups;
    
    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_c * out_h * out_w; i++) {
            y->data[i] = 0.f;
        }
    } else {
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_c; j++) {
                for (size_t k = 0; k < out_h; k++) {
                    for (size_t l = 0; l < out_w; l++) {
                        y->data[i * out_c * out_h * out_w + j * out_h * out_w + k * out_w + l] = bias->data[j];
                    }
                }
            }
        }
    }
    
    float* col = malloc(in_c_per_group * kernel_size * out_h * out_w * sizeof(float));
    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            // Get the input data for the current group
            float* img_group = x->data + (b * in_c * in_h * in_w) + (g * in_c_per_group * in_h * in_w);

            // Perform im2col on the current group
            im2col_T(img_group, in_c_per_group, in_h, in_w, k_h, stride, padding, col);

            // Get the weights for the current group
            float* w_group = weight->data + (g * out_c_per_group * in_c_per_group * kernel_size);

            // Get the output data for the current group
            float* out_group = y->data + (b * out_c * out_h * out_w) + (g * out_c_per_group * out_h * out_w);

            // Perform matrix multiplication for the current group
            MiCo_MatMul_f32(out_group, w_group, col, out_c_per_group, in_c_per_group * kernel_size, out_h * out_w);
        }
    }
    free(col);
}