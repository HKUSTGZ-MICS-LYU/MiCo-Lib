#include "nn.h"
#include <string.h>

// Reference: github.com/pjreddie/darknet
float im2col_get_pixel(float *im, int height, int width,
                        int row, int col, int channel, int pad){
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void MiCo_im2col(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col){
    
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, 
                height, width, im_row, im_col, c_im, pad);
            }
        }
    }
}

void im2col_T(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col){
    
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                // Transposed Index
                int col_index = (h * width_col + w) * channels_col + c;
                data_col[col_index] = im2col_get_pixel(data_im, 
                height, width, im_row, im_col, c_im, pad);
            }
        }
    }
}

void im2col_block_T(const float* data_im, const int channels,
                   const int height, const int width, const int kernel_size,
                   const int stride, const int pad, float* data_col,
                   const int row_offset, const int num_rows, const int out_width) {
    
    // const int height_col = (height + 2 * pad - kernel_size) / stride + 1;
    const int width_col = (width + 2 * pad - kernel_size) / stride + 1;
    const int channels_col = channels * kernel_size * kernel_size;
    
    // Calculate parameters for the partial block
    const int start_h = row_offset;
    const int end_h = row_offset + num_rows;
    
    for (int c = 0; c < channels_col; ++c) {
        const int w_offset = c % kernel_size;
        const int h_offset = (c / kernel_size) % kernel_size;
        const int c_im = c / (kernel_size * kernel_size);
        
        for (int h = start_h; h < end_h; ++h) {
            for (int w = 0; w < width_col; ++w) {
                const int h_pad = h * stride - pad + h_offset;
                const int w_pad = w * stride - pad + w_offset;
                
                // Calculate output index in our partial col matrix
                int out_idx = ((h - start_h) * out_width + w) * channels_col + c;
                
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                    data_col[out_idx] = data_im[(c_im * height + h_pad) * width + w_pad];
                } else {
                    data_col[out_idx] = 0;
                }
            }
        }
    }
}

// TODO: too many args here, not good for registers
void im2col_block_T_aligned(const float* data_im, const int channels,
                   const int height, const int width, const int kernel_size,
                   const int stride, const int pad, float* data_col,
                   const int row_offset, const int num_rows, const int out_width) {
    
    // const int height_col = (height + 2 * pad - kernel_size) / stride + 1;
    const int width_col = (width + 2 * pad - kernel_size) / stride + 1;
    const int channels_col = channels * kernel_size * kernel_size;
    
    // Calculate alignment padding if needed
    int aligned_channels_col = channels_col;
    if (channels_col % 32 != 0) {
        aligned_channels_col = (channels_col / 32 + 1) * 32;
    }
    
    // Calculate parameters for the partial block
    const int start_h = row_offset;
    const int end_h = row_offset + num_rows;
    
    // First, initialize all values to zero (including padding)
    for (int h = start_h; h < end_h; ++h) {
        for (int w = 0; w < width_col; ++w) {
            // Get the starting position for this pixel in the output
            int base_idx = ((h - start_h) * out_width + w) * aligned_channels_col;
            
            // Zero out the entire aligned row
            memset(&data_col[base_idx], 0, aligned_channels_col * sizeof(float));
        }
    }
    
    // Then fill in the actual data
    for (int c = 0; c < channels_col; ++c) {
        const int w_offset = c % kernel_size;
        const int h_offset = (c / kernel_size) % kernel_size;
        const int c_im = c / (kernel_size * kernel_size);
        
        for (int h = start_h; h < end_h; ++h) {
            for (int w = 0; w < width_col; ++w) {
                const int h_pad = h * stride - pad + h_offset;
                const int w_pad = w * stride - pad + w_offset;
                
                // Calculate output index in our aligned col matrix
                int out_idx = ((h - start_h) * out_width + w) * aligned_channels_col + c;
                
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                    data_col[out_idx] = data_im[(c_im * height + h_pad) * width + w_pad];
                }
                // No else needed since we've already zeroed everything
            }
        }
    }
}

// Im2Col for NHWC input layout
// Input: data_im in NHWC format (height, width, channels)
// Output: data_col in transposed format for matmul
// The output matrix has shape (out_h * out_w, kernel_h * kernel_w * channels)
// where each row corresponds to one output position
// Column ordering matches HWIO weight layout: ic varies fastest, then kw, then kh
// Note: kernel_size parameter assumes square kernels (kernel_h == kernel_w)
void im2col_block_T_NHWC(const float* data_im, const int channels,
                   const int height, const int width, const int kernel_size,
                   const int stride, const int pad, float* data_col,
                   const int row_offset, const int num_rows, const int out_width) {
    
    const int width_col = (width + 2 * pad - kernel_size) / stride + 1;
    const int channels_col = channels * kernel_size * kernel_size;
    
    // Calculate parameters for the partial block
    const int start_h = row_offset;
    const int end_h = row_offset + num_rows;
    
    // Column ordering to match HWIO: iterate kh (slowest), then kw, then ic (fastest)
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            for (int ic = 0; ic < channels; ++ic) {
                // Column index in the output matrix
                int c = (kh * kernel_size + kw) * channels + ic;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = 0; w < width_col; ++w) {
                        const int h_pad = h * stride - pad + kh;
                        const int w_pad = w * stride - pad + kw;
                        
                        // Calculate output index in our partial col matrix
                        int out_idx = ((h - start_h) * out_width + w) * channels_col + c;
                        
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                            // NHWC: (h_pad * width + w_pad) * channels + ic
                            data_col[out_idx] = data_im[(h_pad * width + w_pad) * channels + ic];
                        } else {
                            data_col[out_idx] = 0;
                        }
                    }
                }
            }
        }
    }
}

// Im2Col for NHWC input layout with grouped convolution support
// Input: data_im points to the beginning of the batch's spatial data with a channel offset.
//        The total_channels parameter specifies the memory stride between pixels.
//        For group g, data_im should be: base + g * channels_per_group
//        This allows extracting a subset of channels from NHWC data.
// Output: data_col in transposed format for matmul
// The output matrix has shape (out_h * out_w, kernel_h * kernel_w * channels_per_group)
// Column ordering matches HWIO weight layout: ic varies fastest, then kw, then kh
// Note: kernel_size parameter assumes square kernels (kernel_h == kernel_w)
void im2col_block_T_NHWC_grouped(const float* data_im, const int channels_per_group,
                   const int total_channels, const int height, const int width, 
                   const int kernel_size, const int stride, const int pad, float* data_col,
                   const int row_offset, const int num_rows, const int out_width) {
    
    const int width_col = (width + 2 * pad - kernel_size) / stride + 1;
    const int channels_col = channels_per_group * kernel_size * kernel_size;
    
    // Calculate parameters for the partial block
    const int start_h = row_offset;
    const int end_h = row_offset + num_rows;
    
    // Column ordering to match HWIO: iterate kh (slowest), then kw, then ic (fastest)
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            for (int ic = 0; ic < channels_per_group; ++ic) {
                // Column index in the output matrix
                int c = (kh * kernel_size + kw) * channels_per_group + ic;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = 0; w < width_col; ++w) {
                        const int h_pad = h * stride - pad + kh;
                        const int w_pad = w * stride - pad + kw;
                        
                        // Calculate output index in our partial col matrix
                        int out_idx = ((h - start_h) * out_width + w) * channels_col + c;
                        
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                            // NHWC with groups: use total_channels as stride, ic as local channel offset
                            // data_im already points to the start of the channel group
                            data_col[out_idx] = data_im[(h_pad * width + w_pad) * total_channels + ic];
                        } else {
                            data_col[out_idx] = 0;
                        }
                    }
                }
            }
        }
    }
}