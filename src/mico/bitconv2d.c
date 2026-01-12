#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "mico_runtime.h"

extern long QMATMUL_TIMER;
extern long QUANT_TIMER;
extern long IM2COL_TIMER;

extern MiCoRuntime MiCo_runtime;

// TODO: Maybe we have too many arguments here
void MiCo_bitconv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups, const size_t align){

    const size_t batch_size = x->shape[0];

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

    const size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;
    
    const size_t kernel_size = k_h * k_w;
    #ifndef USE_ALT_LAYOUT
    const size_t out_size = out_h * out_w;
    #endif


    #ifdef USE_ALT_LAYOUT
    MiCo_assert(out_h == y->shape[1] && out_w == y->shape[2], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");
    MiCo_assert(wq==8 && aq==8, 
        "[BitConv2D] NHWC currently only support 8-bit quantization!");
    #else
    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");
    #endif

    const size_t in_c_per_group = in_c / groups;
    const size_t out_c_per_group = out_c / groups;

    size_t b_addr, h_addr;
    #ifndef USE_ALT_LAYOUT
    size_t c_addr;
    #endif

    long start; // Profiler

    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_c * out_h * out_w; i++) {
            y->data[i] = 0.f;
        }
    } else {
        #ifdef USE_ALT_LAYOUT
        // NHWC output layout: (batch, height, width, channels)
        for (size_t i = 0; i < batch_size; i++) {
            b_addr = i * out_h * out_w * out_c;
            for (size_t k = 0; k < out_h; k++) {
                h_addr = b_addr + k * out_w * out_c;
                for (size_t l = 0; l < out_w; l++) {
                    size_t w_addr = h_addr + l * out_c;
                    for (size_t j = 0; j < out_c; j++) {
                        y->data[w_addr + j] = bias->data[j];
                    }
                }
            }
        }
        #else
        // NCHW output layout: (batch, channels, height, width)
        for (size_t i = 0; i < batch_size; i++) {
            b_addr = i * out_c * out_h * out_w;
            for (size_t j = 0; j < out_c; j++) {
                c_addr = b_addr + j * out_h * out_w;
                for (size_t k = 0; k < out_h; k++) {
                    h_addr = c_addr + k * out_w;
                    for (size_t l = 0; l < out_w; l++) {
                        y->data[h_addr + l] = bias->data[j];
                    }
                }
            }
        }
        #endif
    }
    
    // Check if Need Alignment Padding
    // TODO: Further adjustment on both Activation and Weight
    // Currently we pad the data during the code generation

    const size_t align_factor = align;

    size_t aligned_size = in_c_per_group * kernel_size;
    if (in_c_per_group * kernel_size % align_factor != 0){
        aligned_size = (in_c_per_group * kernel_size / align_factor + 1) * align_factor;
    }
    
    // Define block size for partial im2col (process this many output rows at a time)
    const size_t block_rows = 2;  // Can be tuned based on cache size and input dimensions
    
    // Calculate memory requirements for one block
    size_t block_out_size = block_rows * out_w;


    float* col = malloc(in_c_per_group * kernel_size * block_out_size * sizeof(float));
    int32_t *qO = malloc(out_c_per_group * block_out_size * sizeof(int32_t));
    
    #ifdef USE_ALT_LAYOUT
    // Allocate temp buffer for weight reordering in grouped convolution
    qbyte* temp_weight = NULL;
    if (groups > 1) {
        temp_weight = malloc(aligned_size * out_c_per_group * sizeof(qbyte));
    }
    #endif

    size_t qx_size = aligned_size * block_out_size * sizeof(qbyte);
    qx_size /= (8 / aq); // Num of Act per Byte
    MiCo_assert(qx_size < QUANTIZE_BUFFER_SIZE, "Quantization Buffer Overflow");
    qbyte* qx_data = MiCo_QBuffer;

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            // Get the input data for the current group
            #ifdef USE_ALT_LAYOUT
            // NHWC layout: data is (batch, height, width, channels)
            // For groups, we need to offset by g * in_c_per_group in the channel dimension
            // The base pointer is at batch b, and we pass the group channel offset
            float* img_group = x->data + (b * in_h * in_w * in_c) + (g * in_c_per_group);
            #else
            // NCHW layout: data is (batch, channels, height, width)
            float* img_group = x->data + (b * in_c * in_h * in_w) + (g * in_c_per_group * in_h * in_w);
            #endif
            
            // Process output rows in blocks
            for (size_t row_offset = 0; row_offset < out_h; row_offset += block_rows) {
                // Calculate actual block size (handling edge case at the end)
                size_t current_block_rows = (row_offset + block_rows <= out_h) ? block_rows : out_h - row_offset;
                size_t current_block_out_size = current_block_rows * out_w;
                
                start = MiCo_time();
                // Partial im2col on the current group - only process the needed rows
                #ifdef USE_ALT_LAYOUT
                // Use NHWC im2col for NHWC input layout
                // Note: For grouped convolution with NHWC, we need a special im2col
                // that can handle non-contiguous channel groups.
                // For simplicity, we use a wrapper approach here.
                im2col_block_T_NHWC_grouped(img_group, in_c_per_group, in_c, in_h, in_w, k_h, stride, padding, 
                              col, row_offset, current_block_rows, out_w);
                #else
                im2col_block_T(img_group, in_c_per_group, in_h, in_w, k_h, stride, padding, 
                              col, row_offset, current_block_rows, out_w);
                #endif
                
                Tensor2D_F32 x_col;
                x_col.data = col;
                x_col.shape[0] = current_block_out_size;
                x_col.shape[1] = in_c_per_group * kernel_size;

                Tensor2D_Q8 qx;
                qx.data = qx_data;
                qx.shape[0] = current_block_out_size;
                qx.shape[1] = aligned_size;
                qx.scale = 0.0f; // To be calculated later

                IM2COL_TIMER += MiCo_time() - start;
                
                start = MiCo_time();
                // Activation Quantization for the current block
                MiCo_2D_quant(&qx, &x_col, aq);
                QUANT_TIMER += MiCo_time() - start;
                // printf("Quant Speed: %ld\n", MiCo_time() - start);

                // Get the weights for the current group
                Tensor2D_Q8 qw;
                #ifdef USE_ALT_LAYOUT
                // HWIO layout: weights are stored as (k_h, k_w, in_c_per_group, out_c)
                // where out_c is the total output channels across all groups.
                // For matmul, we need a (K, M) matrix where:
                //   K = kernel_size * in_c_per_group (aligned_size)
                //   M = out_c_per_group
                // 
                // The weight data stride is out_c (total), but we only need out_c_per_group columns
                // starting at column g * out_c_per_group.
                // 
                // For the matmul: w->data[k * out_features + j]
                // When groups == 1, we can use the weights directly since out_c == out_c_per_group.
                // When groups > 1, we need to copy weights to a temp buffer with correct stride.
                
                size_t weight_k = aligned_size;  // K dimension
                size_t weight_m = out_c_per_group; // M dimension
                
                if (groups == 1) {
                    // No grouping - use weights directly
                    qw.data = weight->data;
                    qw.shape[0] = weight_k;
                    qw.shape[1] = weight_m;
                } else {
                    // Grouped convolution - need to copy weights with correct stride
                    // Source stride: out_c (total output channels)
                    // Destination stride: out_c_per_group
                    size_t group_start_oc = g * out_c_per_group;
                    for (size_t k = 0; k < weight_k; k++) {
                        for (size_t m = 0; m < weight_m; m++) {
                            // Source index: k * out_c + (group_start_oc + m)
                            // Destination index: k * weight_m + m
                            temp_weight[k * weight_m + m] = weight->data[k * out_c + group_start_oc + m];
                        }
                    }
                    qw.data = temp_weight;
                    qw.shape[0] = weight_k;
                    qw.shape[1] = weight_m;
                }
                #else
                size_t offset = (g * out_c_per_group * aligned_size) / (8 / wq);
                qw.data = weight->data + offset;
                qw.shape[0] = out_c_per_group;
                qw.shape[1] = aligned_size;
                #endif
                qw.scale = weight->scale;

                // Initialize qO for the current block
                for(size_t i = 0; i < out_c_per_group * current_block_out_size; i++){
                    qO[i] = 0;
                }
                
                // Debug Information
                // if (row_offset == 0) {
                //     printf("Im2Col MatMul Shape (block): %ldx%ldx%ld\n", 
                //           qw.shape[0], qw.shape[1], qx.shape[0]);
                // }
                // TODO: Handle VLEN ?
                // MatMul-Based Convolution for the current block
                start = MiCo_time();
                #ifdef USE_ALT_LAYOUT
                // For NHWC: qx (activation) is first arg, qw (weight) is second
                // Index order: [first_tensor_bits][second_tensor_bits]
                MiCo_runtime.matmul_matrix[qlog(aq)][qlog(wq)](qO, &qx, &qw);
                #else
                // For NCHW: qw (weight) is first arg, qx (activation) is second
                MiCo_runtime.matmul_matrix[qlog(wq)][qlog(aq)](qO, &qw, &qx);
                #endif
                QMATMUL_TIMER += MiCo_time() - start;

                #ifdef USE_ALT_LAYOUT
                // NHWC output layout: (batch, out_h, out_w, out_c)
                // qO is in (current_block_out_size, out_c_per_group) layout
                // We need to write to y at positions (b, row_offset+h, w, g*out_c_per_group + oc)
                float scale = weight->scale * qx.scale;
                start = MiCo_time();
                for (size_t j = 0; j < current_block_out_size; j++) {
                    size_t h_out = row_offset + (j / out_w);
                    size_t w_out = j % out_w;
                    size_t y_base = (b * out_h * out_w * out_c) + (h_out * out_w * out_c) + (w_out * out_c) + (g * out_c_per_group);
                    for (size_t oc = 0; oc < out_c_per_group; oc++) {
                        size_t qo_idx = j * out_c_per_group + oc;
                        y->data[y_base + oc] += (float)qO[qo_idx] * scale;
                    }
                }
                QUANT_TIMER += MiCo_time() - start;
                #else
                // NCHW output layout
                // Calculate output position for this block
                size_t block_output_addr = b * out_c * out_size + 
                                          (g * out_c_per_group * out_size) + 
                                          row_offset * out_w;
                
                float scale = weight->scale * qx.scale;
                start = MiCo_time();
                // De-Quantization for the current block
                for (size_t oc = 0; oc < out_c_per_group; oc++) {
                    for (size_t j = 0; j < current_block_out_size; j++) {
                        size_t qo_idx = oc * current_block_out_size + j;
                        size_t y_idx = block_output_addr + oc * out_size + j;
                        y->data[y_idx] += (float)qO[qo_idx] * scale;
                    }
                }
                QUANT_TIMER += MiCo_time() - start;
                #endif
                // printf("DeQuant Speed: %ld\n", MiCo_time() - start);
            }
        }
    }
    #ifdef USE_ALT_LAYOUT
    if (temp_weight != NULL) {
        free(temp_weight);
    }
    #endif
    free(qO);
    free(col);
}