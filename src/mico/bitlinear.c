#include "nn.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

void MiCo_bitlinear_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x,
    const Tensor2D_Q8 *weight, const Tensor1D_F32 *bias,
    const qtype wq, const qtype aq){

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

    // TODO: should have a better way to handle Qa/Qw combinations
    if (aq == 8){        
        switch (wq)
        {
          case 8:
            // MatMul Computation
            MiCo_Q8_MatMul(qO, &qx, weight);
            break;
          case 4:
            MiCo_Q8x4_MatMul(qO, &qx, weight);
            break;
          case 2:
            MiCo_Q8x2_MatMul(qO, &qx, weight);
            break;
          case 1:
            MiCo_Q8x1_MatMul(qO, &qx, weight);
            break;
          default:
            printf("[Warning] Unsupported Weight Quantization - %d\n", wq);
            break;
        }

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
}