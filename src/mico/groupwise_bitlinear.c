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

void MiCo_groupwise_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8_Groupwise *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq, const size_t align){

    // Check qtype legality
    if (wq > 8 || aq > 8){
        printf("[Error] Unsupported Quantization Type\n");
        return;
    }

    const size_t b = x->shape[0];
    const size_t n = x->shape[1];
    const size_t m = weight->shape[0];
    const size_t group_size = weight->group_size;

    // Check if group_size divides m
    if (m % group_size != 0) {
        printf("[Error] Number of output features (%zu) must be divisible by group_size (%zu)\n", m, group_size);
        return;
    }

    const size_t num_groups = m / group_size;

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
    qx.data = MiCo_QBuffer;

    switch (aq)
    {
      case 8:
        MiCo_2D_FP32toQ8(&qx, x);
        break;
      case 4:
        MiCo_2D_FP32toQ4(&qx, x);
        break;
      case 2:
        MiCo_2D_FP32toQ2(&qx, x);
        break;
      case 1:
        MiCo_2D_FP32toQ1(&qx, x);
        break;
      default:
        printf("[Warning] Unsupported Weight Quantization - %d\n", aq);
        break;
    }
    QUANT_TIMER += MiCo_time() - start;

    // Prepare weight tensor for MatMul (reuse existing data structure)
    Tensor2D_Q8 qw;
    qw.shape[0] = weight->shape[0];
    qw.shape[1] = weight->shape[1];
    qw.data = weight->data;
    qw.wq = weight->wq;
    qw.scale = 1.0f; // Will be applied per-group later

    // Perform MatMul using standard INT MatMul functions
    start = MiCo_time();
    MiCo_QMatMul[qlog(aq)][qlog(wq)](qO, &qx, &qw);
    QMATMUL_TIMER += MiCo_time() - start;

    // Group-wise De-Quantization
    start = MiCo_time();
    for (size_t i = 0; i < b; i++) {
      baddr = i * m;
      for (size_t g = 0; g < num_groups; g++) {
        float scale = weight->scales[g] * qx.scale;
        size_t group_start = g * group_size;
        for (size_t j = 0; j < group_size; j++) {
          size_t idx = baddr + group_start + j;
          y->data[idx] += (float)qO[idx] * scale;
        }
      }
    }
    QUANT_TIMER += MiCo_time() - start;
    
    // Free Quantized Memory
    free(qO);
}
