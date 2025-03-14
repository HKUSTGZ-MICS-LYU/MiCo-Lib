#include "mico_qnn.h"

// SIMD accelerated Implementation of 8-bit MatMul
// Requires ISA support
extern void __mico_v4s8_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Check if it is possible to unroll
    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_v4s8_vecXmat(
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

extern void __mico_mixed8x4_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_mixed8x4_vecXmat(
                (qword*)(x->data+i*in_features), 
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
                    temp_w = (k & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                    temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                    acc += x->data[i * in_features + k] * temp_w;
                }
                O[i * out_features + j] = acc;
            }
        }
    }
}

extern void __mico_mixed8x2_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;
    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_mixed8x2_vecXmat(
                (qword*)(x->data+i*in_features), 
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
                    temp_w = temp_w >> (((j * in_features + k) & 0b11) << 1) & 0x03;
                    acc += AMUX_2BIT(temp_w, x->data[i * in_features + k]);
                }
                O[i * out_features + j] = acc;
            }
        }
    }
}

extern void __mico_mixed8x1_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;
    
    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_mixed8x1_vecXmat(
                (qword*)(x->data+i*in_features), 
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
                    temp_w = EXTRACT_BIT(temp_w, (j * in_features + k) & 0b111);
                    acc += AMUX_1BIT(temp_w, x->data[i * in_features + k]);
                }
                O[i * out_features + j] = acc;
            }
        }
    }
}

extern void __mico_v8s4_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_v8s4_vecXmat(
                (qword*)(x->data+i*in_features), 
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

extern void __mico_mixed4x2_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_mixed4x2_vecXmat(
                (qword*)(x->data+i*in_features), 
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

extern void __mico_mixed4x1_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_mixed4x1_vecXmat(
                (qword*)(x->data+i*in_features), 
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

extern void __mico_v16s2_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_v16s2_vecXmat(
                (qword*)(x->data+i*in_features), 
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

extern void __mico_mixed2x1_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_mixed2x1_vecXmat(
                (qword*)(x->data+i*in_features), 
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

extern void __mico_v32s1_vecXmat(qword* a, qword* w, int32_t* o, int n, int m);
void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    if(in_features % 32 == 0){
        for (size_t i = 0; i < batch_size; i++) {
            __mico_v32s1_vecXmat(
                (qword*)(x->data+i*in_features), 
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
