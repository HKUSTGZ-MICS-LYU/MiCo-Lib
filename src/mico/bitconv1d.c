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

// Im2Col helper function for 1D convolution
static void im2col_1d(const float* data_im, const int channels, const int length,
    const int kernel_size, const int stride, const int pad, float* data_col) {
    
    int out_l = (length + 2 * pad - kernel_size) / stride + 1;
    
    for (int ol = 0; ol < out_l; ol++) {
        for (int c = 0; c < channels; c++) {
            for (int kl = 0; kl < kernel_size; kl++) {
                int il = ol * stride + kl - pad;
                int col_idx = ol * channels * kernel_size + c * kernel_size + kl;
                if (il >= 0 && il < length) {
                    data_col[col_idx] = data_im[c * length + il];
                } else {
                    data_col[col_idx] = 0;
                }
            }
        }
    }
}

// Block-based im2col for 1D convolution (transposed for matmul)
static void im2col_block_1d_T(const float* data_im, const int channels, const int length,
    const int kernel_size, const int stride, const int pad, float* data_col,
    const int offset, const int num_elements, const int out_length) {
    
    for (int i = 0; i < num_elements; i++) {
        int ol = offset + i;
        if (ol >= out_length) break;
        
        for (int c = 0; c < channels; c++) {
            for (int kl = 0; kl < kernel_size; kl++) {
                int il = ol * stride + kl - pad;
                int col_idx = i * channels * kernel_size + c * kernel_size + kl;
                if (il >= 0 && il < length) {
                    data_col[col_idx] = data_im[c * length + il];
                } else {
                    data_col[col_idx] = 0;
                }
            }
        }
    }
}

// Quantized 1D Convolution with Layout NCL (Batch, Channels, Length)
void MiCo_bitconv1d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, 
    const Tensor3D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups, const size_t align){

    const size_t batch_size = x->shape[0];

    const size_t in_c = x->shape[1];
    const size_t in_l = x->shape[2];

    const size_t k_l = weight->shape[2];

    const size_t out_c = y->shape[1];
    const size_t out_l = (in_l + 2 * padding - k_l) / stride + 1;

    MiCo_assert(out_l == y->shape[2], 
        "[Conv1D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv1D] Group Mismatched!");

    const size_t in_c_per_group = in_c / groups;
    const size_t out_c_per_group = out_c / groups;

    size_t b_addr, c_addr;

    long start; // Profiler

    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_c * out_l; i++) {
            y->data[i] = 0.f;
        }
    } else {
        for (size_t i = 0; i < batch_size; i++) {
            b_addr = i * out_c * out_l;
            for (size_t j = 0; j < out_c; j++) {
                c_addr = b_addr + j * out_l;
                for (size_t l = 0; l < out_l; l++) {
                    y->data[c_addr + l] = bias->data[j];
                }
            }
        }
    }
    
    // Check if Need Alignment Padding
    const size_t align_factor = align;

    size_t aligned_size = in_c_per_group * k_l;
    if (in_c_per_group * k_l % align_factor != 0){
        aligned_size = (in_c_per_group * k_l / align_factor + 1) * align_factor;
    }
    
    // Define block size for partial im2col
    const size_t block_elements = 4;  // Can be tuned based on cache size
    
    // Calculate memory requirements for one block
    size_t block_out_size = block_elements;

    float* col = malloc(in_c_per_group * k_l * block_out_size * sizeof(float));
    int32_t *qO = malloc(out_c_per_group * block_out_size * sizeof(int32_t));

    size_t qx_size = aligned_size * block_out_size * sizeof(qbyte);
    qx_size /= (8 / aq); // Num of Act per Byte
    MiCo_assert(qx_size < QUANTIZE_BUFFER_SIZE, "Quantization Buffer Overflow");
    qbyte* qx_data = MiCo_QBuffer;

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            // Get the input data for the current group
            float* img_group = x->data + (b * in_c * in_l) + (g * in_c_per_group * in_l);
            
            // Process output elements in blocks
            for (size_t elem_offset = 0; elem_offset < out_l; elem_offset += block_elements) {
                // Calculate actual block size (handling edge case at the end)
                size_t current_block_elements = (elem_offset + block_elements <= out_l) ? block_elements : out_l - elem_offset;
                
                start = MiCo_time();
                // Partial im2col on the current group
                im2col_block_1d_T(img_group, in_c_per_group, in_l, k_l, stride, padding, 
                              col, elem_offset, current_block_elements, out_l);
                
                Tensor2D_F32 x_col;
                x_col.data = col;
                x_col.shape[0] = current_block_elements;
                x_col.shape[1] = in_c_per_group * k_l;

                Tensor2D_Q8 qx;
                qx.data = qx_data;
                qx.shape[0] = current_block_elements;
                qx.shape[1] = aligned_size;
                qx.scale = 0.0f; // To be calculated later

                IM2COL_TIMER += MiCo_time() - start;
                
                start = MiCo_time();
                // Activation Quantization for the current block
                MiCo_2D_quant(&qx, &x_col, aq);
                QUANT_TIMER += MiCo_time() - start;

                // Get the weights for the current group
                Tensor2D_Q8 qw;
                size_t offset = (g * out_c_per_group * aligned_size) / (8 / wq);
                qw.data = weight->data + offset;
                qw.shape[0] = out_c_per_group;
                qw.shape[1] = aligned_size;
                qw.scale = weight->scale;

                // Initialize qO for the current block
                for(size_t i = 0; i < out_c_per_group * current_block_elements; i++){
                    qO[i] = 0;
                }
                
                // MatMul-Based Convolution for the current block
                start = MiCo_time();
                MiCo_runtime.matmul_matrix[qlog(wq)][qlog(aq)](qO, &qw, &qx);
                QMATMUL_TIMER += MiCo_time() - start;

                // Calculate output position for this block
                size_t block_output_addr = b * out_c * out_l + 
                                          (g * out_c_per_group * out_l) + 
                                          elem_offset;
                
                float scale = weight->scale * qx.scale;
                start = MiCo_time();
                // De-Quantization for the current block
                for (size_t oc = 0; oc < out_c_per_group; oc++) {
                    for (size_t j = 0; j < current_block_elements; j++) {
                        size_t qo_idx = oc * current_block_elements + j;
                        size_t y_idx = block_output_addr + oc * out_l + j;
                        y->data[y_idx] += (float)qO[qo_idx] * scale;
                    }
                }
                QUANT_TIMER += MiCo_time() - start;
            }
        }
    }
    free(qO);
    free(col);
}
