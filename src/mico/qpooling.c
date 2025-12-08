#include "mico_qnn.h"
#include "mico_quant.h"
#include <stdlib.h>
#include <string.h>

// Im2Col transformation for quantized pooling
// This function transforms the input tensor into a column matrix where each column
// contains the elements of one pooling window. The layout is optimized for pooling operations.
// 
// Parameters:
//   data_im: Input image data in NCHW format (flattened)
//   channels: Number of input channels
//   height: Input height
//   width: Input width
//   kernel_size: Size of pooling kernel (assumes square kernel)
//   stride: Stride for pooling
//   pad: Padding size
//   data_col: Output column matrix [out_h*out_w, channels*kernel_size*kernel_size]
//
// Note: For pooling, each output position corresponds to one column in data_col.
// The column contains all input values within the pooling window for that position.
void im2col_pool_q8(const int8_t* data_im, const int channels,
                    const int height, const int width, const int kernel_size,
                    const int stride, const int pad, int8_t* data_col) {
    
    const int height_col = (height + 2 * pad - kernel_size) / stride + 1;
    const int width_col = (width + 2 * pad - kernel_size) / stride + 1;
    const int channels_col = channels * kernel_size * kernel_size;
    
    // For each output position
    for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
            // For each element in the pooling window
            int col_idx = 0;
            for (int c = 0; c < channels; ++c) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        const int h_pad = h * stride - pad + kh;
                        const int w_pad = w * stride - pad + kw;
                        
                        // Output index: (h*width_col + w)*channels_col + col_idx
                        const int out_idx = (h * width_col + w) * channels_col + col_idx;
                        
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                            // Valid input position
                            data_col[out_idx] = data_im[(c * height + h_pad) * width + w_pad];
                        } else {
                            // Padding: use minimum int8 value for max pooling compatibility
                            // This ensures padded values don't affect max pooling
                            data_col[out_idx] = -128;
                        }
                        col_idx++;
                    }
                }
            }
        }
    }
}

// Quantized Average Pooling using im2col + averaging
// 
// This implementation uses im2col to transform pooling windows into columns,
// then computes the average over each column. The result is requantized to int8.
//
// Input/Output quantization:
//   - Input scale is preserved from x->scale
//   - Output uses the same scale as input (no rescaling needed for average)
//   - Accumulation uses int32 to prevent overflow
//
// Supported configurations:
//   - kernel_size: 2 or 3
//   - stride: 1 or 2
//   - padding: 0 or 1
//   - Layout: NCHW
void MiCo_Q8_AvgPool2D(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
                       const size_t kernel_size, const size_t stride, const size_t padding) {
    
    // Validate supported configurations
    MiCo_assert(kernel_size == 2 || kernel_size == 3, 
        "[Q8_AvgPool2D] Unsupported kernel size! Only 2 and 3 are supported.");
    MiCo_assert(stride == 1 || stride == 2, 
        "[Q8_AvgPool2D] Unsupported stride! Only 1 and 2 are supported.");
    MiCo_assert(padding == 0 || padding == 1, 
        "[Q8_AvgPool2D] Unsupported padding! Only 0 and 1 are supported.");
    
    const size_t batch_size = x->shape[0];
    const size_t in_c = x->shape[1];
    const size_t in_h = x->shape[2];
    const size_t in_w = x->shape[3];
    
    const size_t out_c = y->shape[1];
    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Q8_AvgPool2D] Output shape mismatch!");
    MiCo_assert(out_c == in_c, 
        "[Q8_AvgPool2D] Channel count mismatch!");
    
    // Preserve input scale for output
    y->scale = x->scale;
    
    // Allocate buffer for im2col output
    const size_t window_size = kernel_size * kernel_size;
    const size_t col_size = out_h * out_w * window_size;
    int8_t* col = (int8_t*)malloc(col_size * sizeof(int8_t));
    
    // Process each batch and channel
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < in_c; c++) {
            // Get pointer to current channel
            const int8_t* img_channel = x->data + (b * in_c * in_h * in_w) + (c * in_h * in_w);
            
            // Apply im2col for this channel (treating as single-channel image)
            im2col_pool_q8(img_channel, 1, in_h, in_w, kernel_size, stride, padding, col);
            
            // Compute average for each output position
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    int valid_count = 0;
                    
                    // Get the window data from col
                    const int col_offset = (oh * out_w + ow) * window_size;
                    
                    for (size_t k = 0; k < window_size; k++) {
                        int8_t val = col[col_offset + k];
                        // Count valid values (not padding)
                        // We use -128 for padding in im2col_pool_q8
                        if (val != -128 || (padding == 0)) {
                            // Check if this position is actually within valid input
                            int kh = k / kernel_size;
                            int kw = k % kernel_size;
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                sum += val;
                                valid_count++;
                            }
                        }
                    }
                    
                    // Compute average (division in quantized space)
                    int8_t avg = (valid_count > 0) ? (int8_t)(sum / valid_count) : 0;
                    
                    // Store result
                    y->data[b * out_c * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = avg;
                }
            }
        }
    }
    
    free(col);
}

// Quantized Max Pooling using im2col
//
// This implementation uses im2col to transform pooling windows into columns,
// then computes the maximum over each column.
//
// Input/Output quantization:
//   - Input scale is preserved from x->scale
//   - Output uses the same scale as input
//   - Max operation is performed in quantized space (valid for monotonic quantization)
//
// Supported configurations:
//   - kernel_size: 2 or 3
//   - stride: 1 or 2
//   - padding: 0 or 1
//   - Layout: NCHW
void MiCo_Q8_MaxPool2D(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
                       const size_t kernel_size, const size_t stride, const size_t padding) {
    
    // Validate supported configurations
    MiCo_assert(kernel_size == 2 || kernel_size == 3, 
        "[Q8_MaxPool2D] Unsupported kernel size! Only 2 and 3 are supported.");
    MiCo_assert(stride == 1 || stride == 2, 
        "[Q8_MaxPool2D] Unsupported stride! Only 1 and 2 are supported.");
    MiCo_assert(padding == 0 || padding == 1, 
        "[Q8_MaxPool2D] Unsupported padding! Only 0 and 1 are supported.");
    
    const size_t batch_size = x->shape[0];
    const size_t in_c = x->shape[1];
    const size_t in_h = x->shape[2];
    const size_t in_w = x->shape[3];
    
    const size_t out_c = y->shape[1];
    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Q8_MaxPool2D] Output shape mismatch!");
    MiCo_assert(out_c == in_c, 
        "[Q8_MaxPool2D] Channel count mismatch!");
    
    // Preserve input scale for output
    y->scale = x->scale;
    
    // Allocate buffer for im2col output
    const size_t window_size = kernel_size * kernel_size;
    const size_t col_size = out_h * out_w * window_size;
    int8_t* col = (int8_t*)malloc(col_size * sizeof(int8_t));
    
    // Process each batch and channel
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < in_c; c++) {
            // Get pointer to current channel
            const int8_t* img_channel = x->data + (b * in_c * in_h * in_w) + (c * in_h * in_w);
            
            // Apply im2col for this channel
            im2col_pool_q8(img_channel, 1, in_h, in_w, kernel_size, stride, padding, col);
            
            // Compute max for each output position
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    int8_t max_val = -128; // Minimum int8 value
                    int has_valid = 0;
                    
                    // Get the window data from col
                    const int col_offset = (oh * out_w + ow) * window_size;
                    
                    for (size_t k = 0; k < window_size; k++) {
                        // Check if this position is within valid input (not padding)
                        int kh = k / kernel_size;
                        int kw = k % kernel_size;
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        
                        if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                            int8_t val = col[col_offset + k];
                            if (!has_valid || val > max_val) {
                                max_val = val;
                                has_valid = 1;
                            }
                        }
                    }
                    
                    // Store result (use 0 if no valid values, though this shouldn't happen)
                    y->data[b * out_c * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = 
                        has_valid ? max_val : 0;
                }
            }
        }
    }
    
    free(col);
}

// Reference implementation: Naive quantized average pooling
// This is used for testing correctness of the optimized implementation
#ifdef REF
void MiCo_Q8_AvgPool2D_Ref(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
                           const size_t kernel_size, const size_t stride, const size_t padding) {
    
    const size_t batch_size = x->shape[0];
    const size_t in_c = x->shape[1];
    const size_t in_h = x->shape[2];
    const size_t in_w = x->shape[3];
    
    const size_t out_c = y->shape[1];
    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Q8_AvgPool2D_Ref] Output shape mismatch!");
    MiCo_assert(out_c == in_c, 
        "[Q8_AvgPool2D_Ref] Channel count mismatch!");
    
    // Preserve input scale
    y->scale = x->scale;
    
    // Direct implementation without im2col
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oc = 0; oc < out_c; oc++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    int32_t sum = 0;
                    int valid_count = 0;
                    
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            int ih = (int)(oh * stride + kh) - (int)padding;
                            int iw = (int)(ow * stride + kw) - (int)padding;
                            
                            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                int8_t val = x->data[b * in_c * in_h * in_w + 
                                                     oc * in_h * in_w + 
                                                     ih * in_w + iw];
                                sum += val;
                                valid_count++;
                            }
                        }
                    }
                    
                    int8_t avg = (valid_count > 0) ? (int8_t)(sum / valid_count) : 0;
                    y->data[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = avg;
                }
            }
        }
    }
}

// Reference implementation: Naive quantized max pooling
void MiCo_Q8_MaxPool2D_Ref(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
                           const size_t kernel_size, const size_t stride, const size_t padding) {
    
    const size_t batch_size = x->shape[0];
    const size_t in_c = x->shape[1];
    const size_t in_h = x->shape[2];
    const size_t in_w = x->shape[3];
    
    const size_t out_c = y->shape[1];
    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Q8_MaxPool2D_Ref] Output shape mismatch!");
    MiCo_assert(out_c == in_c, 
        "[Q8_MaxPool2D_Ref] Channel count mismatch!");
    
    // Preserve input scale
    y->scale = x->scale;
    
    // Direct implementation without im2col
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t oc = 0; oc < out_c; oc++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    int8_t max_val = -128;
                    int has_valid = 0;
                    
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            int ih = (int)(oh * stride + kh) - (int)padding;
                            int iw = (int)(ow * stride + kw) - (int)padding;
                            
                            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                int8_t val = x->data[b * in_c * in_h * in_w + 
                                                     oc * in_h * in_w + 
                                                     ih * in_w + iw];
                                if (!has_valid || val > max_val) {
                                    max_val = val;
                                    has_valid = 1;
                                }
                            }
                        }
                    }
                    
                    y->data[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = 
                        has_valid ? max_val : 0;
                }
            }
        }
    }
}
#endif
