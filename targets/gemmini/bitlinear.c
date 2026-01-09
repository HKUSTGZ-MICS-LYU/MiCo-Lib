#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "mico_runtime.h"

#include "gemmini_nn.h"

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
    QUANT_TIMER += MiCo_time() - start;
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

    // Pointer Casting
    const int8_t (*A)[b] = (const int8_t (*)[b]) qx.data;
    const int8_t (*B)[m] = (const int8_t (*)[m]) weight->data;

    start = MiCo_time();
    gemmini_flush(0);
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