#include "nn.h"

void MiCo_avgpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t k_size, const size_t stride, const size_t padding){
    
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    size_t k_h = k_size;
    size_t k_w = k_size;

    size_t out_c = y->shape[1];
    size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[AvgPool2D] Output Shape Mismatched!");

    MiCo_assert(out_c == in_c, 
        "[AvgPool2D] Output Channel Mismatched!");

    for (size_t b = 0; b < batch_size; b++){
        for (size_t oc = 0; oc < out_c; oc++){
            for (size_t oh = 0; oh < out_h; oh++){
                for (size_t ow = 0; ow < out_w; ow++){
                    float sum = 0;
                    size_t valid_count = 0;
                    for (size_t kh = 0; kh < k_h; kh++){
                        for (size_t kw = 0; kw < k_w; kw++){
                            // Calculate input position considering padding
                            int ih = (int)(oh * stride + kh) - (int)padding;
                            int iw = (int)(ow * stride + kw) - (int)padding;
                            
                            // Check bounds (padding areas are treated as 0)
                            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w){
                                sum += x->data[b * in_c * in_h * in_w + oc * in_h * in_w + ih * in_w + iw];
                                valid_count++;
                            }
                        }
                    }
                    // Average over valid pixels only (excludes padding)
                    y->data[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = 
                        valid_count > 0 ? sum / valid_count : 0;
                }
            }
        }
    }
}

void MiCo_maxpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t k_size, const size_t stride, const size_t padding){
    
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    size_t k_h = k_size;
    size_t k_w = k_size;

    size_t out_c = y->shape[1];
    size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[MaxPool2D] Output Shape Mismatched!");

    MiCo_assert(out_c == in_c, 
        "[MaxPool2D] Output Channel Mismatched!");

    for (size_t b = 0; b < batch_size; b++){
        for (size_t oc = 0; oc < out_c; oc++){
            for (size_t oh = 0; oh < out_h; oh++){
                for (size_t ow = 0; ow < out_w; ow++){
                    float max = -FLOAT_MAX;
                    int has_valid = 0;
                    for (size_t kh = 0; kh < k_h; kh++){
                        for (size_t kw = 0; kw < k_w; kw++){
                            // Calculate input position considering padding
                            int ih = (int)(oh * stride + kh) - (int)padding;
                            int iw = (int)(ow * stride + kw) - (int)padding;
                            
                            // Check bounds (padding areas are treated as -infinity)
                            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w){
                                float data = x->data[b * in_c * in_h * in_w + oc * in_h * in_w + ih * in_w + iw];
                                max = max > data ? max : data;
                                has_valid = 1;
                            }
                        }
                    }
                    y->data[b * out_c * out_h * out_w + 
                        oc * out_h * out_w + oh * out_w + ow] = has_valid ? max : 0;
                }
            }
        }
    }
}

void MiCo_adaptive_avgpool4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t s){

    // Infer Pooling Kernel Size
    MiCo_assert(x->shape[2] == x->shape[3], 
        "[AdaptiveAvgPool2D] Invalid Input, H =/= W!");
    size_t input_size = x->shape[2];
    size_t k_size = input_size - s + 1;
    MiCo_avgpool4d_f32(y, x, k_size, 1, 0);
}

// 1D Pooling Functions with Layout NCL (Batch, Channels, Length)
void MiCo_avgpool3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, 
    const size_t k_size, const size_t stride, const size_t padding){
    
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_l = x->shape[2];

    size_t out_c = y->shape[1];
    size_t out_l = (in_l + 2 * padding - k_size) / stride + 1;

    MiCo_assert(out_l == y->shape[2], 
        "[AvgPool1D] Output Shape Mismatched!");

    MiCo_assert(out_c == in_c, 
        "[AvgPool1D] Output Channel Mismatched!");

    for (size_t b = 0; b < batch_size; b++){
        for (size_t oc = 0; oc < out_c; oc++){
            for (size_t ol = 0; ol < out_l; ol++){
                float sum = 0;
                size_t valid_count = 0;
                for (size_t kl = 0; kl < k_size; kl++){
                    // Calculate input position considering padding
                    int il = (int)(ol * stride + kl) - (int)padding;
                    
                    // Check bounds (padding areas are treated as 0)
                    if (il >= 0 && il < (int)in_l){
                        sum += x->data[b * in_c * in_l + oc * in_l + il];
                        valid_count++;
                    }
                }
                // Average over valid elements only (excludes padding)
                y->data[b * out_c * out_l + oc * out_l + ol] = 
                    valid_count > 0 ? sum / valid_count : 0;
            }
        }
    }
}

void MiCo_maxpool3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, 
    const size_t k_size, const size_t stride, const size_t padding){
    
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_l = x->shape[2];

    size_t out_c = y->shape[1];
    size_t out_l = (in_l + 2 * padding - k_size) / stride + 1;

    MiCo_assert(out_l == y->shape[2], 
        "[MaxPool1D] Output Shape Mismatched!");

    MiCo_assert(out_c == in_c, 
        "[MaxPool1D] Output Channel Mismatched!");

    for (size_t b = 0; b < batch_size; b++){
        for (size_t oc = 0; oc < out_c; oc++){
            for (size_t ol = 0; ol < out_l; ol++){
                float max = -FLOAT_MAX;
                int has_valid = 0;
                for (size_t kl = 0; kl < k_size; kl++){
                    // Calculate input position considering padding
                    int il = (int)(ol * stride + kl) - (int)padding;
                    
                    // Check bounds (padding areas are treated as -infinity)
                    if (il >= 0 && il < (int)in_l){
                        float data = x->data[b * in_c * in_l + oc * in_l + il];
                        max = max > data ? max : data;
                        has_valid = 1;
                    }
                }
                y->data[b * out_c * out_l + oc * out_l + ol] = has_valid ? max : 0;
            }
        }
    }
}

void MiCo_adaptive_avgpool3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, 
    const size_t s){

    // Infer Pooling Kernel Size
    size_t input_size = x->shape[2];
    size_t k_size = input_size - s + 1;
    MiCo_avgpool3d_f32(y, x, k_size, 1, 0);
}