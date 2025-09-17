#include "mico_qnn.h"

extern void cfu_enable();
extern void cfu_dotp_int8(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    
    cfu_enable();
    // Check if it is possible to unroll
    if(in_features % 4 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            cfu_dotp_int8(
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

extern void cfu_dotp_int4(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 8 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            cfu_dotp_int4(
                (qword*)(x->data+i*in_features/2), 
                (qword*)(w->data), 
                (int32_t*)(O+i*out_features), 
                in_features, 
                out_features);
        }
    }
    else{
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                acc = 0;
                for (size_t k = 0; k < in_features; k++) {
                    // TODO: What will happen if k is odd?
                    temp_w = w->data[(j * in_features + k)/2];
                    temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                    temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                    temp_a = x->data[(i * in_features + k)/2];
                    temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                    temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                    acc += temp_a * temp_w;
                }
                O[i * out_features + j] = acc;
            }
        }
    }
}

extern void cfu_dotp_int2(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 16 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            cfu_dotp_int2(
                (qword*)(x->data+i*in_features/4), 
                (qword*)(w->data), 
                (int32_t*)(O+i*out_features), 
                in_features, 
                out_features);
        }
    }
    else{
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                acc = 0;
                for (size_t k = 0; k < in_features; k++) {
                    // TODO: What will happen if k is odd?
                    temp_w = w->data[(j * in_features + k)/4];
                    temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                    temp_w = TWO_BIT_TO_INT8(temp_w);
                    temp_a = x->data[(i * in_features + k)/4];
                    temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                    temp_a = TWO_BIT_TO_INT8(temp_a);
                    acc += temp_a * temp_w;
                }
                O[i * out_features + j] = acc;
            }
        }
    }
}

extern void cfu_dotp_int1(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            cfu_dotp_int1(
                (qword*)(x->data+i*in_features/8), 
                (qword*)(w->data), 
                (int32_t*)(O+i*out_features), 
                in_features, 
                out_features);
        }
    }
    else{
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < out_features; j++) {
                acc = 0;
                for (size_t k = 0; k < in_features; k++) {
                    // TODO: What will happen if k is odd?
                    temp_w = w->data[(j * in_features + k)/8];
                    temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                    temp_w = BIT_TO_INT8(temp_w);
                    temp_a = x->data[(i * in_features + k)/8];
                    temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                    temp_a = BIT_TO_INT8(temp_a);
                    // TODO: BNN could be optimized if we use some dithering here.
                    acc += temp_a * temp_w;
                }
                O[i * out_features + j] = acc;
            }
        }
    }
}