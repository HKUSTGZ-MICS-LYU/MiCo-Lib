#include "nn.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

typedef void (*MatMulFunc)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*);

static MatMulFunc MiCo_QMatMul[4][4] = {
  {NULL, NULL, NULL, NULL},
  {NULL, NULL, NULL, NULL},
  {NULL, NULL, NULL, NULL},
  {MiCo_Q8x1_MatMul, MiCo_Q8x2_MatMul, MiCo_Q8x4_MatMul, MiCo_Q8_MatMul},
};

static int qlog(qtype x){
    int result = 0;
    while (x >>= 1) result++;
    return result;
}

void MiCo_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8 *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq){

    // Check qtype legality
    if (wq > 8 || aq > 8){
        printf("[Error] Unsupported Quantization Type\n");
        return;
    }

    const size_t b = x->shape[0];
    const size_t n = x->shape[1];
    const size_t m = weight->shape[0];

    // Initialization
    if (bias->shape[0] == 0){
      for (size_t i = 0; i < b * m; i++) {
          y->data[i] = 0.f;
      }
    } else {
      for (size_t i = 0; i < b; i++) {
        for (size_t j = 0; j < m; j++) {
          y->data[i * m + j] = bias->data[j];
        }
      }
    }

    int32_t* qO = malloc(b*m*sizeof(int32_t));
    for (size_t i = 0; i < b * m; i++) {
        qO[i] = 0;
    }

    // Activation Quantization
    Tensor2D_Q8 qx;
    qx.shape[0] = b;
    qx.shape[1] = n;
    switch (aq)
    {
      case 8:
        qx.data = malloc(b*n*sizeof(int8_t));
        MiCo_2D_FP32toQ8(&qx, x);
        break;
      default:
        printf("[Warning] Unsupported Weight Quantization - %d\n", aq);
        break;
    }

    // TODO: Maybe we should use Enum for aq and wq, so that we can skip qlog
    MiCo_QMatMul[qlog(aq)][qlog(wq)](qO, &qx, weight);
  
    // Re-Quantization
    for (size_t i = 0; i < b; i++) {
      for (size_t j = 0; j < m; j++) {
          y->data[i * m + j] += (float)qO[i * m + j] * weight->scale * qx.scale;
      }
    }

    // Free Quantized Memory
    free(qx.data);
    free(qO);
}