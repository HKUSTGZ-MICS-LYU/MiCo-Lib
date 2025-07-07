#include "nn.h"

#include "nn.h"

void MiCo_channel_shuffle(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const size_t channels, const size_t groups) {

    MiCo_assert(channels % groups == 0, 
        "[Channel Shuffle] Channels must be divisible by groups!");
    MiCo_assert(x->shape[1] == channels, 
        "[Channel Shuffle] Input channels do not match the specified channels!");

    size_t group_size = channels / groups;
    size_t batch_size = x->shape[0];
    size_t height = x->shape[2];
    size_t width = x->shape[3];

    // Set output dimensions (same as input)
    y->shape[0] = batch_size;
    y->shape[1] = channels;
    y->shape[2] = height;
    y->shape[3] = width;
    
    // Calculate strides for efficient indexing
    size_t stride_width = 1;
    size_t stride_height = width;
    size_t stride_channel = height * width;
    size_t stride_batch = channels * height * width;
    
    // Perform channel shuffle
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t g = 0; g < groups; g++) {
            for (size_t c = 0; c < group_size; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        // Original layout: [g0c0, g0c1, ..., g1c0, g1c1, ...]
                        // Shuffled layout: [g0c0, g1c0, ..., g0c1, g1c1, ...]
                        
                        // Calculate channel indices before and after shuffle
                        size_t input_channel = g * group_size + c;
                        size_t output_channel = c * groups + g;
                        
                        // Calculate flat array indices
                        size_t input_idx = b * stride_batch + 
                                          input_channel * stride_channel + 
                                          h * stride_height + 
                                          w * stride_width;
                        
                        size_t output_idx = b * stride_batch + 
                                           output_channel * stride_channel + 
                                           h * stride_height + 
                                           w * stride_width;
                        
                        // Copy the value
                        y->data[output_idx] = x->data[input_idx];
                    }
                }
            }
        }
    }
}