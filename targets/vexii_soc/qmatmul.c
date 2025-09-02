#include "mico_qnn.h"

extern void cfu_dotp(qword* a, qword* w, int32_t* o, int n, int m);
extern void cfu_enable();
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    cfu_enable();
    // Check if it is possible to unroll
    if(in_features % 4 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            cfu_dotp(
                (qword*)(x->data+i*in_features), 
                (qword*)(w->data), 
                (int32_t*)(O+i*out_features), 
                in_features, 
                out_features);
        }
    }else{
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                int32_t sum = 0;
                for (size_t k = 0; k < in_features; k++) {
                    sum += x->data[i * in_features + k] * \
                        w->data[j * in_features + k];
                }
                O[i * out_features + j] = sum;
            }
        }
    }
}