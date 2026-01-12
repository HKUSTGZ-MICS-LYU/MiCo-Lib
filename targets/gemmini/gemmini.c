#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "mico_runtime.h"

#include "gemmini_nn.h"


extern long QMATMUL_TIMER;
extern long QUANT_TIMER;
extern long IM2COL_TIMER;

extern MiCoRuntime MiCo_runtime;


void MiCo_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8 *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq, const size_t align){

    // Check qtype legality
    if (wq != 8 || aq != 8){
        printf("[Error] Unsupported Quantization Type\n");
        return;
    }

    const size_t b = x->shape[0];
    const size_t n = x->shape[1];
    #ifdef USE_ALT_LAYOUT
    const size_t m = weight->shape[1];
    #else
    const size_t m = weight->shape[0];
    #endif
    
    
    const size_t align_factor = align;
    const size_t aligned_size = (n + align_factor - 1) / align_factor * align_factor;
    
    // Activation Quantization
    Tensor2D_Q8 qx;
    qx.shape[0] = b;
    qx.shape[1] = aligned_size;
    long start;

    start = MiCo_time();
    const size_t qx_size = b*aligned_size*sizeof(int8_t) / (8/aq);
    MiCo_assert(qx_size < QUANTIZE_BUFFER_SIZE, "Quantization Buffer Overflow");
    qx.data = MiCo_QX_Buffer_Global.buffer;
    MiCo_2D_quant(&qx, x, aq);
    // printf("Quant Speed: %ld\n", MiCo_time() - start);
    float scale = weight->scale * qx.scale;

    // Address
    size_t baddr;
    // Initialization
    bool use_bias = (bias->shape[0] != 0); 

    int32_t* qB = malloc(m*sizeof(int32_t));
    int8_t C[n][m];
    for (size_t i = 0; i < b; i++) {
        for (size_t j = 0; j < m; j++){
            C[i][j] = 0;
        }
    }
    for (size_t j = 0; j < m; j++){
        if (use_bias){
            qB[j] = bias->data[j] / scale;
        } else {
            qB[j] = 0;
        }
    }
    QUANT_TIMER += MiCo_time() - start;

    // Pointer Casting
    const int8_t (*A)[b] = (const int8_t (*)[b]) qx.data;
    const int8_t (*B)[m] = (const int8_t (*)[m]) weight->data;

    start = MiCo_time();
    // gemmini_flush(0);
    enum tiled_matmul_type_t tiled_matmul_type;
    tiled_matmul_type = WS;

    tiled_matmul_nn_auto(b, m, n,
        A, B, qB, C,
        NO_ACTIVATION, scale, true,
        tiled_matmul_type, false, "bitlinear");
    
    QMATMUL_TIMER += MiCo_time() - start;

    // printf("MatMul Speed: %ld\n", MiCo_time() - start);

    // De-Quantization (TODO: Heavy in FP32 operations)
    start = MiCo_time();
    for (size_t i = 0; i < b; i++) {
      baddr = i * m;
      for (size_t j = 0; j < m; j++) {
        y->data[baddr + j] = (float)C[i][j];
      }
    }
    QUANT_TIMER += MiCo_time() - start;
    // printf("DeQuant Speed: %ld\n", MiCo_time() - start);
    // printf("DeQuant Scale: %.4f\n", scale);
    
    // Free Quantized Memory
    free(qB);
}



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

    size_t qx_size = in_c_per_group * kernel_size * out_h * out_w * sizeof(qbyte);
    MiCo_assert(qx_size < QUANTIZE_BUFFER_SIZE, "Quantization Buffer Overflow");

    // Initialization
    bool use_bias = (bias->shape[0] != 0); 

    int32_t *qb = NULL;

    Tensor4D_Q8 qx = {
        .data = MiCo_QBuffer,
        .shape = {batch_size, in_c_per_group * kernel_size, out_h, out_w},
        .scale = 1.0f
    };

    start = MiCo_time();
    MiCo_4D_quant(&qx, x, aq);
    float scale = weight->scale * qx.scale;

    if (use_bias){
        qb = malloc(out_c_per_group * sizeof(int32_t));
        for (size_t i = 0; i < out_c_per_group; i++){
            qb[i] = bias->data[i] / scale;
        }
    }
    QUANT_TIMER += MiCo_time() - start;

    elem_t* conv_x = (elem_t*)qx.data;
    elem_t* conv_w = (elem_t*)weight->data;

    elem_t conv_o[batch_size][out_h][out_w][out_c];

    // gemmini_flush(0);
    enum tiled_matmul_type_t tiled_matmul_type = WS;

    start = MiCo_time();
    tiled_conv_auto(
        batch_size, 
        in_h, in_w, in_c,
        out_c, out_h, out_w,
        stride, 1, 1, padding, k_h,
        false, false, false, false, false,
        (elem_t*)conv_x, (elem_t*)conv_w, (acc_t*)qb, (elem_t*)conv_o,
        NO_ACTIVATION, scale,
        0, 0, 0, // No Pooling
        tiled_matmul_type
    );
    QMATMUL_TIMER += MiCo_time() - start;

    // Dequantization and Copy to output
    for (size_t i = 0; i < batch_size; i++) {
        b_addr = i * out_h * out_w * out_c;
        for (size_t k = 0; k < out_h; k++) {
            h_addr = b_addr + k * out_w * out_c;
            for (size_t l = 0; l < out_w; l++) {
                size_t w_addr = h_addr + l * out_c;
                for (size_t j = 0; j < out_c; j++) {
                    y->data[w_addr + j] = (float)conv_o[i][k][l][j];
                }
            }
        }
    }
}