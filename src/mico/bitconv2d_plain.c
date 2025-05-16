#include "nn.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "profile.h"

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

// Legacy code: Full im2col Conv2D implementation
void MiCo_bitconv2d_f32_plain(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
    const Tensor4D_Q8 *weight, const Tensor1D_F32 *bias, 
    const qtype wq, const qtype aq,
    const size_t stride, const size_t padding, 
    const size_t dilation, const size_t groups){

    size_t batch_size = x->shape[0];

    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    size_t k_h = weight->shape[2];
    size_t k_w = weight->shape[3];

    size_t out_c = y->shape[1];
    size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;
    
    // size_t feature_size = in_h * in_w;
    size_t kernel_size = k_h * k_w;
    size_t out_size = out_h * out_w;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");

    size_t in_c_per_group = in_c / groups;
    size_t out_c_per_group = out_c / groups;

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
    // size_t aligned_size = in_c_per_group * kernel_size;
    // if (in_c_per_group * kernel_size % 32 != 0){
    //     aligned_size = (in_c_per_group * kernel_size / 32 + 1) * 32;
    // }

    float* col = malloc(in_c_per_group * kernel_size * out_h * out_w * sizeof(float));
    int32_t *qO = malloc(out_c_per_group * out_size * sizeof(int32_t));

    size_t qx_size = in_c_per_group * kernel_size * out_h * out_w * sizeof(qbyte);
    qx_size /= (8 / aq); // Num of Act per Byte
    qbyte* qx_data = malloc(qx_size);

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            // Get the input data for the current group
            float* img_group = x->data + (b * in_c * in_h * in_w) + (g * in_c_per_group * in_h * in_w);
            start = MiCo_time();
            // Perform im2col on the current group
            im2col_T(img_group, in_c_per_group, in_h, in_w, k_h, stride, padding, col);
            IM2COL_TIMER += MiCo_time() - start;
            start = MiCo_time();
            float qs = 0.0;
            switch (aq)
            {
                case 8:
                qs = __FP32toQ8(qx_data, col, in_c_per_group * kernel_size * out_size);
                break;
                case 4:
                qs = __FP32toQ4(qx_data, col, in_c_per_group * kernel_size * out_size);
                break;
                case 2:
                qs = __FP32toQ2(qx_data, col, in_c_per_group * kernel_size * out_size);
                break;
                case 1:
                qs = __FP32toQ1(qx_data, col, in_c_per_group * kernel_size * out_size);
                break;
                default:
                    printf("[Warning] Unsupported Weight Quantization - %d\n", aq);
                break;
            }
            QUANT_TIMER += MiCo_time() - start;
            
            Tensor2D_Q8 qx;
            qx.data = qx_data;
            qx.shape[0] = out_size;
            qx.shape[1] = in_c_per_group * kernel_size;
            qx.scale = qs;

            // Get the weights for the current group
            Tensor2D_Q8 qw;
            qw.data = weight->data + (g * out_c_per_group * in_c_per_group * kernel_size);
            qw.shape[0] = out_c_per_group;
            qw.shape[1] = in_c_per_group * kernel_size;
            qw.scale = weight->scale;

            // Initialize qO for the current group
            for(int i = 0; i < out_c_per_group * out_size; i++){
                qO[i] = 0;
            }
            printf("Im2Col MatMul Shape: %ldx%ldx%ld\n",\
                qw.shape[0], qw.shape[1], qx.shape[0]);
            // MatMul-Based Convolution for the current group
            // TODO: Need Alignment!
            start = MiCo_time();
            MiCo_QMatMul[qlog(wq)][qlog(aq)](qO, &qw, &qx);
            QMATMUL_TIMER += MiCo_time() - start;

            size_t group_addr = b * out_c * out_size + (g * out_c_per_group * out_size);
            float scale = weight->scale * qx.scale;
            start = MiCo_time();
            // Re-Quantization for the current group
            for (size_t j = 0; j < out_c_per_group * out_size; j++) {
                y->data[group_addr + j] += (float)qO[j] * scale;
            }
            QUANT_TIMER += MiCo_time() - start;
        }
    }
    free(qx_data);
    free(qO);
    free(col);
}