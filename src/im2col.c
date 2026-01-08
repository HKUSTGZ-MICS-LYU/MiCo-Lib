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