#include "nn.h"

// Convolution Functions with Layout NCL (Batch, Channels, Length)
__attribute__((weak)) void MiCo_conv1d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, 
    const Tensor3D_F32* weight, const Tensor1D_F32* bias, 
    const size_t stride, const size_t padding, const size_t dilation, const size_t groups){
    // dilation is not implemented yet
    size_t batch_size = x->shape[0];

    size_t in_c = x->shape[1];
    size_t in_l = x->shape[2];

    size_t k_l = weight->shape[2];

    size_t out_c = y->shape[1];
    size_t out_l = (in_l + 2 * padding - k_l) / stride + 1;

    MiCo_assert(out_l == y->shape[2], 
        "[Conv1D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv1D] Group Mismatched!");

    size_t in_c_per_group = in_c / groups;
    size_t out_c_per_group = out_c / groups;

    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_c * out_l; i++) {
            y->data[i] = 0.f;
        }
    } else {
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_c; j++) {
                for (size_t k = 0; k < out_l; k++) {
                    y->data[i * out_c * out_l + j * out_l + k] = bias->data[j];
                }
            }
        }
    }

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < out_c_per_group; oc++){
                for (size_t ol = 0; ol < out_l; ol++){
                    float sum = 0;
                    for (size_t ic = 0; ic < in_c_per_group; ic++){
                        for (size_t kl = 0; kl < k_l; kl++){
                            size_t il = ol * stride + kl - padding;
                            if (il >= 0 && il < in_l){
                                size_t in_index = b * in_c * in_l + (g * in_c_per_group + ic) * in_l + il;
                                size_t weight_index = (g * out_c_per_group + oc) * in_c_per_group * k_l + ic * k_l + kl;
                                sum += x->data[in_index] * weight->data[weight_index];
                            }
                        }
                    }
                    size_t out_index = b * out_c * out_l + (g * out_c_per_group + oc) * out_l + ol;
                    y->data[out_index] += sum;
                }
            }
        }
    }
}
