#include "nn.h"

void MiCo_concat4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x1, const Tensor4D_F32 *x2){

    MiCo_assert(x1->shape[0] == x2->shape[0], "[Concat4D] Batch Size Mismatched!");
    MiCo_assert(y->shape[1] == x1->shape[1] + x2->shape[1], "[Concat4D] Channel Size Mismatched!");
    MiCo_assert(y->shape[2] == x1->shape[2], "[Concat4D] Height Size Mismatched!");
    MiCo_assert(y->shape[3] == x1->shape[3], "[Concat4D] Width Size Mismatched!");

    size_t batch_size = x1->shape[0];
    size_t in_c1 = x1->shape[1];
    size_t in_c2 = x2->shape[1];
    size_t out_c = in_c1 + in_c2;

    size_t in_h = x1->shape[2];
    size_t in_w = x1->shape[3];

    for (int b = 0; b < batch_size; b++){
        for(int c = 0; c < out_c; c++){
            for(int h = 0; h < in_h; h++){
                for(int w = 0; w < in_w; w++){
                    if(c < in_c1){
                        y->data[b * out_c * in_h * in_w + c * in_h * in_w + h * in_w + w] = \
                        x1->data[b * in_c1 * in_h * in_w + c * in_h * in_w + h * in_w + w];
                    } else {
                        y->data[b * out_c * in_h * in_w + c * in_h * in_w + h * in_w + w] = \
                        x2->data[b * in_c2 * in_h * in_w + (c - in_c1) * in_h * in_w + h * in_w + w];
                    }
                }
            }
        }
    }
}

void MiCo_concat2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x1, const Tensor2D_F32 *x2){

    MiCo_assert(x1->shape[0] == x2->shape[0], "[Concat2D] Batch Size Mismatched!");
    MiCo_assert(y->shape[1] == x1->shape[1] + x2->shape[1], "[Concat2D] Out Size Mismatched!");

    size_t batch_size = x1->shape[0];
    size_t in_c1 = x1->shape[1];
    size_t in_c2 = x2->shape[1];
    size_t out_c = in_c1 + in_c2;

    for (int b = 0; b < batch_size; b++){
        for(int c = 0; c < out_c; c++){
            if(c < in_c1){
                y->data[b * out_c + c] = x1->data[b * in_c1 + c];
            } else {
                y->data[b * out_c + c] = x2->data[b * in_c2 + (c - in_c1)];
            }
        }
    
    }
}