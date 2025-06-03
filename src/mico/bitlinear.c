#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

extern long QMATMUL_TIMER;
extern long QUANT_TIMER;

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

    const size_t align_factor = 32;

    const size_t aligned_size = (n + align_factor - 1) / align_factor * align_factor;
    // Activation Quantization
    Tensor2D_Q8 qx;
    qx.shape[0] = b;
    qx.shape[1] = aligned_size;

    start = MiCo_time();
    switch (aq)
    {
      case 8:
        qx.data = malloc(b*aligned_size*sizeof(int8_t));
        MiCo_2D_FP32toQ8(&qx, x);
        break;
      case 4:
        qx.data = malloc(b*aligned_size*sizeof(int8_t)/2);
        MiCo_2D_FP32toQ4(&qx, x);
        break;
      case 2:
        qx.data = malloc(b*aligned_size*sizeof(int8_t)/4);
        MiCo_2D_FP32toQ2(&qx, x);
        break;
      case 1:
        qx.data = malloc(b*aligned_size*sizeof(int8_t)/8);
        MiCo_2D_FP32toQ1(&qx, x);
        break;
      default:
        printf("[Warning] Unsupported Weight Quantization - %d\n", aq);
        break;
    }
    QUANT_TIMER += MiCo_time() - start;
    // printf("Quant Speed: %ld\n", MiCo_time() - start);

    // TODO: Maybe we should use Enum for aq and wq, so that we can skip qlog
    start = MiCo_time();
    MiCo_QMatMul[qlog(aq)][qlog(wq)](qO, &qx, weight);
    QMATMUL_TIMER += MiCo_time() - start;
    // printf("MatMul Speed: %ld\n", MiCo_time() - start);

    float scale = weight->scale * qx.scale;
    // De-Quantization
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
    free(qx.data);
    free(qO);
}