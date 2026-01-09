#include "nn.h"

#ifdef USE_ALT_LAYOUT
// NHWC Layout: N, H, W, C
// Access: (n, c, h, w) -> n*H*W*C + h*W*C + w*C + c
#define OFFSET_4D(n, c, h, w, N, C, H, W) ((n) * (H) * (W) * (C) + (h) * (W) * (C) + (w) * (C) + (c))
#else
// NCHW Layout: N, C, H, W
// Access: (n, c, h, w) -> n*C*H*W + c*H*W + h*W + w
#define OFFSET_4D(n, c, h, w, N, C, H, W) ((n) * (C) * (H) * (W) + (c) * (H) * (W) + (h) * (W) + (w))
#endif

// Convolution Functions with Layout NCHW
__attribute__((weak)) void MiCo_conv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_F32* weight, const Tensor1D_F32* bias, 
    const size_t stride, const size_t padding, const size_t dilation, const size_t groups){
    // dilation is not implemented yet
    size_t batch_size = x->shape[0];

    #ifdef USE_ALT_LAYOUT
    // Layout NHWC
    size_t in_h = x->shape[1];
    size_t in_w = x->shape[2];
    size_t in_c = x->shape[3];
    
    // Layout HWIO
    size_t k_h = weight->shape[0];
    size_t k_w = weight->shape[1];

    size_t out_c = y->shape[3];
    #else
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    size_t k_h = weight->shape[2];
    size_t k_w = weight->shape[3];

    size_t out_c = y->shape[1];
    #endif

    size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;
    // size_t feature_size = in_h * in_w;
    size_t kernel_size = k_h * k_w;

    #ifdef USE_ALT_LAYOUT
    MiCo_assert(out_h == y->shape[1] && out_w == y->shape[2], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");
    #else
    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");
    #endif
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
                        y->data[OFFSET_4D(i, j, k, l, batch_size, out_c, out_h, out_w)] = bias->data[j];
                    }
                }
            }
        }
    }

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < out_c_per_group; oc++){
                for (size_t oh = 0; oh < out_h; oh++){
                    for (size_t ow = 0; ow < out_w; ow++){
                        float sum = 0;
                        for (size_t ic = 0; ic < in_c_per_group; ic++){
                            for (size_t kh = 0; kh < k_h; kh++){
                                for (size_t kw = 0; kw < k_w; kw++){
                                    size_t ih = oh * stride + kh - padding;
                                    size_t iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w){
                                        size_t in_index = OFFSET_4D(b, (g * in_c_per_group + ic), ih, iw, batch_size, in_c, in_h, in_w);
                                        
                                        #ifdef USE_ALT_LAYOUT
                                        // Weight Layout: HWIO (h, w, i, o)
                                        // Access: h*W*I*O + w*I*O + i*O + o
                                        size_t weight_index = kh * k_w * in_c_per_group * out_c_per_group 
                                                            + kw * in_c_per_group * out_c_per_group 
                                                            + ic * out_c_per_group 
                                                            + (g * out_c_per_group + oc); 
                                        
                                        size_t global_oc = g * out_c_per_group + oc;
                                        weight_index = kh * k_w * in_c_per_group * out_c +
                                                       kw * in_c_per_group * out_c +
                                                       ic * out_c +
                                                       global_oc;
                                        #else
                                        size_t weight_index = (g * out_c_per_group + oc) * in_c_per_group * kernel_size + ic * kernel_size + kh * k_w + kw;
                                        #endif

                                        sum += x->data[in_index] * weight->data[weight_index];
                                    }
                                }
                            }
                        }
                        size_t out_index = OFFSET_4D(b, (g * out_c_per_group + oc), oh, ow, batch_size, out_c, out_h, out_w);
                        y->data[out_index] += sum;
                    }
                }
            }
        }
    }
}