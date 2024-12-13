#include "nn.h"

// Convolution Functions with Layout NCHW
void MiCo_im2col_conv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
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
    float* col = malloc(in_c*kernel_size*out_h*out_w*sizeof(float));
    for (size_t b = 0; b < batch_size; b++){
        float* img = x->data + (b * in_c * in_h * in_w);
        im2col(img, in_c, in_h, in_w, k_h, stride, padding, col);
        float* w = weight->data;
        float* out = y->data + (b * out_c * out_h * out_w);
        // MatMul-Based Convolution
        MiCo_MatMul_f32(out, w, col, out_c, in_c*kernel_size, out_h*out_w);
    }
    free(col);
}