#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

extern long QMATMUL_TIMER;
extern long QUANT_TIMER;
extern long IM2COL_TIMER;

typedef void (*MatMulFunc)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*);

static MatMulFunc MiCo_QMatMul[4][4] = {
  {MiCo_Q1_MatMul,   MiCo_Q1x2_MatMul, MiCo_Q1x4_MatMul, MiCo_Q1x8_MatMul},
  {MiCo_Q2x1_MatMul, MiCo_Q2_MatMul,   MiCo_Q2x4_MatMul, MiCo_Q2x8_MatMul},
  {MiCo_Q4x1_MatMul, MiCo_Q4x2_MatMul, MiCo_Q4_MatMul,   MiCo_Q4x8_MatMul},
  {MiCo_Q8x1_MatMul, MiCo_Q8x2_MatMul, MiCo_Q8x4_MatMul, MiCo_Q8_MatMul},
};

static int qlog(qtype x){
    int result = 0;
    while (x >>= 1) result++;
    return result;
}

// TODO: Maybe we have too many arguments here
void MiCo_bitconv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups, const size_t align){

    const size_t batch_size = x->shape[0];

    const size_t in_c = x->shape[1];
    const size_t in_h = x->shape[2];
    const size_t in_w = x->shape[3];

    const size_t k_h = weight->shape[2];
    const size_t k_w = weight->shape[3];

    const size_t out_c = y->shape[1];
    const size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;
    
    const size_t kernel_size = k_h * k_w;
    const size_t out_size = out_h * out_w;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");

    const size_t in_c_per_group = in_c / groups;
    const size_t out_c_per_group = out_c / groups;

    size_t b_addr, c_addr, h_addr;

    long start; // Profiler

    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_c * out_h * out_w; i++) {
            y->data[i] = 0.f;
        }
    } else {
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

    size_t qx_size = aligned_size * block_out_size * sizeof(qbyte);
    qx_size /= (8 / aq); // Num of Act per Byte
    MiCo_assert(qx_size < QUANTIZE_BUFFER_SIZE, "Quantization Buffer Overflow");
    qbyte* qx_data = MiCo_QBuffer;

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            // Get the input data for the current group
            float* img_group = x->data + (b * in_c * in_h * in_w) + (g * in_c_per_group * in_h * in_w);
            
            // Process output rows in blocks
            for (size_t row_offset = 0; row_offset < out_h; row_offset += block_rows) {
                // Calculate actual block size (handling edge case at the end)
                size_t current_block_rows = (row_offset + block_rows <= out_h) ? block_rows : out_h - row_offset;
                size_t current_block_out_size = current_block_rows * out_w;
                
                start = MiCo_time();
                // Partial im2col on the current group - only process the needed rows
                im2col_block_T(img_group, in_c_per_group, in_h, in_w, k_h, stride, padding, 
                              col, row_offset, current_block_rows, out_w);
                
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
                switch (aq)
                {
                    case 8:
                    MiCo_2D_FP32toQ8(&qx, &x_col);
                    break;
                    case 4:
                    MiCo_2D_FP32toQ4(&qx, &x_col);
                    break;
                    case 2:
                    MiCo_2D_FP32toQ2(&qx, &x_col);
                    break;
                    case 1:
                    MiCo_2D_FP32toQ1(&qx, &x_col);
                    break;
                    default:
                        printf("[Warning] Unsupported Weight Quantization - %d\n", aq);
                    break;
                }
                QUANT_TIMER += MiCo_time() - start;

                // Get the weights for the current group
                Tensor2D_Q8 qw;
                size_t offset = (g * out_c_per_group * aligned_size) / (8 / wq);
                qw.data = weight->data + offset;
                qw.shape[0] = out_c_per_group;
                qw.shape[1] = aligned_size;
                qw.scale = weight->scale;

                // Initialize qO for the current block
                for(int i = 0; i < out_c_per_group * current_block_out_size; i++){
                    qO[i] = 0;
                }
                
                // Debug Information
                // if (row_offset == 0) {
                //     printf("Im2Col MatMul Shape (block): %ldx%ldx%ld\n", 
                //           qw.shape[0], qw.shape[1], qx.shape[0]);
                // }
                // #ifdef VLEN
                const int VLEN = 64;
                // if ((uintptr_t)(qx.data) % (VLEN/8) != 0){
                //     printf("[Warning] Activation Not Aligned to VLEN(%d) - %p\n", VLEN, qx.data);
                // }
                // if ((uintptr_t)(qw.data) % (VLEN/8) != 0){
                //     printf("[Warning] Weight Not Aligned to VLEN(%d) - %p\n", VLEN, weight->data);
                // }
                // #endif
                // MatMul-Based Convolution for the current block
                start = MiCo_time();
                MiCo_QMatMul[qlog(wq)][qlog(aq)](qO, &qw, &qx);
                QMATMUL_TIMER += MiCo_time() - start;

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
            }
        }
    }
    free(qO);
    free(col);
}