#include "nn.h"

void MiCo_relu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x){
  const size_t n = y->shape[0] * y->shape[1];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = x_val > 0 ? x_val : 0;
  }
}
void MiCo_relu3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x){
  const size_t n = y->shape[0] * y->shape[1] * y->shape[2];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = x_val > 0 ? x_val : 0;
  }
}
void MiCo_relu4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x){
  const size_t n = y->shape[0] * y->shape[1] * y->shape[2] * y->shape[3];
  for (size_t i = 0; i < n; i += 1) {
    float x_val = x->data[i];
    y->data[i] = x_val > 0 ? x_val : 0;
  }
}