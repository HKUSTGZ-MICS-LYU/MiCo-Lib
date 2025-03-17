#include "nn.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

typedef void (*MatMulFunc)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*);

static MatMulFunc MiCo_QMatMul[4][4] = {
  {MiCo_Q1_MatMul, NULL, NULL, NULL},
  {MiCo_Q2x1_MatMul, MiCo_Q2_MatMul, NULL, NULL},
  {MiCo_Q4x1_MatMul, MiCo_Q4x2_MatMul, MiCo_Q4_MatMul, NULL},
  {MiCo_Q8x1_MatMul, MiCo_Q8x2_MatMul, MiCo_Q8x4_MatMul, MiCo_Q8_MatMul},
};

void MiCo_bitconv2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, 
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
    
    size_t feature_size = in_h * in_w;
    size_t kernel_size = k_h * k_w;
    size_t out_size = out_h * out_w;

    MiCo_assert(out_h == y->shape[2] && out_w == y->shape[3], 
        "[Conv2D] Output Shape Mismatched!");

    MiCo_assert(in_c % groups == 0 && out_c % groups == 0, 
        "[Conv2D] Group Mismatched!");

    size_t in_c_per_group = in_c / groups;
    size_t out_c_per_group = out_c / groups;

    // Initialize Output Tensor
    if (bias->shape[0] == 0){
        for (size_t i = 0; i < batch_size * out_c * out_h * out_w; i++) {
            y->data[i] = 0.f;
        }
    } else {
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_c; j++) {
                for (size_t k = 0; k < out_h; k++) {
                    for (size_t l = 0; l < out_w; l++) {
                        y->data[i * out_c * out_h * out_w + j * out_h * out_w + k * out_w + l] = bias->data[j];
                    }
                }
            }
        }
    }
    
    float* col = malloc(in_c_per_group * kernel_size * out_h * out_w * sizeof(float));
    int32_t *qO = malloc(out_c_per_group * out_size * sizeof(int32_t));

    size_t qx_size = in_c_per_group * kernel_size * out_h * out_w * sizeof(int8_t);
    qx_size /= (8 / aq); // Num of Act per Byte
    int8_t* qx_data = malloc(qx_size);

    for (size_t b = 0; b < batch_size; b++){
        for (size_t g = 0; g < groups; g++) {
            // Get the input data for the current group
            float* img_group = x->data + (b * in_c * in_h * in_w) + (g * in_c_per_group * in_h * in_w);

            // Perform im2col on the current group
            im2col_T(img_group, in_c_per_group, in_h, in_w, k_h, stride, padding, col);

            // TODO: Adapt to Different Presicion
            float qs;
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

            // MatMul-Based Convolution for the current group
            MiCo_Q8_MatMul(qO, &qw, &qx);

            // Re-Quantization for the current group
            for (size_t j = 0; j < out_c_per_group * out_size; j++) {
                y->data[b * out_c * out_size + (g * out_c_per_group * out_size) + j] += (float)qO[j] * weight->scale * qx.scale;
            }
        }
    }
    free(qx_data);
    free(qO);
    free(col);
}