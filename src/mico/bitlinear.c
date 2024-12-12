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

    if (aq == 8 && wq == 8){
        Tensor2D_Q8 qx;
        qx.shape[0] = b;
        qx.shape[1] = n;
        
        // Activation Quantization
        qx.data = malloc(b*n*sizeof(int8_t));
        MiCo_2D_FP32toQ8(&qx, x);
        
        int32_t* qO = malloc(b*m*sizeof(int32_t));
        
        // MatMul Computation
        MiCo_Q8_MatMul(qO, &qx, weight);

        // Re-Quantization
        for (size_t i = 0; i < b; i++) {
            for (size_t j = 0; j < m; j++) {
                y->data[i * m + j] += (float)qO[i * m + j] \
                    * weight->scale * qx.scale;
            }
        }

        // Free Quantized Memory
        free(qx.data);
        free(qO);
    }
}