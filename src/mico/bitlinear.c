#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"
#include "mico_runtime.h"

extern long QMATMUL_TIMER;
extern long QUANT_TIMER;

extern MiCoRuntime MiCo_runtime;

__attribute__((weak)) void MiCo_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8 *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq, const size_t align){

    // Check qtype legality
    if (wq > 8 || aq > 8){
        printf("[Error] Unsupported Quantization Type\n");
        return;
    }

    const size_t b = x->shape[0];
    const size_t n = x->shape[1];
    #ifdef USE_ALT_LAYOUT
    const size_t m = weight->shape[1];
    MiCo_assert(wq == 8 && aq == 8, "N,K x K,M layout only supports INT8 weights and activations");
    #else
    const size_t m = weight->shape[0];
    #endif
    
    // Address
    size_t baddr;
    long start;
    // Initialization
    if (bias->shape[0] == 0){
      for (size_t i = 0; i < b * m; i++) {
          y->data[i] = 0.f;
      }
    } else {
      for (size_t i = 0; i < b; i++) {
        baddr = i * m;
        for (size_t j = 0; j < m; j++) {
          y->data[baddr + j] = bias->data[j];
        }
      }
    }

    int32_t* qO = malloc(b*m*sizeof(int32_t));
    for (size_t i = 0; i < b * m; i++) {
        qO[i] = 0;
    }

    const size_t align_factor = align;

    const size_t aligned_size = (n + align_factor - 1) / align_factor * align_factor;
    // Activation Quantization
    Tensor2D_Q8 qx;
    qx.shape[0] = b;
    qx.shape[1] = aligned_size;

    start = MiCo_time();
    const size_t qx_size = b*aligned_size*sizeof(int8_t) / (8/aq);
    MiCo_assert(qx_size < QUANTIZE_BUFFER_SIZE, "Quantization Buffer Overflow");
    qx.data = MiCo_QX_Buffer_Global.buffer;
    MiCo_2D_quant(&qx, x, aq);
    QUANT_TIMER += MiCo_time() - start;
    // printf("Quant Speed: %ld\n", MiCo_time() - start);

    // TODO: Maybe we should use Enum for aq and wq, so that we can skip qlog
    start = MiCo_time();
    MiCo_runtime.matmul_matrix[qlog(aq)][qlog(wq)](qO, &qx, weight);
    QMATMUL_TIMER += MiCo_time() - start;
    // printf("MatMul Speed: %ld\n", MiCo_time() - start);

    float scale = weight->scale * qx.scale;
    // De-Quantization (TODO: Heavy in FP32 operations)
    start = MiCo_time();
    for (size_t i = 0; i < b; i++) {
      baddr = i * m;
      for (size_t j = 0; j < m; j++) {
          y->data[baddr + j] += (float)qO[baddr + j] * scale;
      }
    }
    QUANT_TIMER += MiCo_time() - start;
    // printf("DeQuant Speed: %ld\n", MiCo_time() - start);
    // printf("DeQuant Scale: %.4f\n", scale);
    
    // Free Quantized Memory
    free(qO);
}