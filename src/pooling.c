#include "nn.h"

void MiCo_avgpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t k_size, const size_t stride){
    
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    size_t k_h = k_size;
    size_t k_w = k_size;

    size_t out_c = y->shape[1];
    size_t out_h = (in_h - k_h) / stride + 1;
    size_t out_w = (in_w - k_w) / stride + 1;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[AvgPool2D] Output Shape Mismatched!");

    MiCo_assert(out_c == in_c, 
        "[AvgPool2D] Output Channel Mismatched!");

    for (size_t b = 0; b < batch_size; b++){
        for (size_t oc = 0; oc < out_c; oc++){
            for (size_t oh = 0; oh < out_h; oh++){
                for (size_t ow = 0; ow < out_w; ow++){
                    float sum = 0;
                    for (size_t kh = 0; kh < k_h; kh++){
                        for (size_t kw = 0; kw < k_w; kw++){
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w){
                                sum += x->data[b * in_c * in_h * in_w + oc * in_h * in_w + ih * in_w + iw];
                            }
                        }
                    }
                    y->data[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum / k_h / k_w;
                }
            }
        }
    }
}