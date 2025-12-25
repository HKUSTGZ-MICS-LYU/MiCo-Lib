#include "mico_qnn.h"

// SIMD accelerated Implementation of 8-bit MatMul
// Requires ISA support

#define mico_v8s8_mac(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x4, 0x01, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})

#define mico_v16s4_mac(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x5, 0x02, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})

#define mico_v32s2_mac(a, b) ({                \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x6, 0x04, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})

#define mico_v64s1_mac(a, b) ({                \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x7, 0x08, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})

#define mico_mac_8x4(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x5, 0x01, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})
#define mico_mac_8x2(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x6, 0x01, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})
#define mico_mac_8x1(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x7, 0x01, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})
#define mico_mac_4x2(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x6, 0x02, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})
#define mico_mac_4x1(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x7, 0x02, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})
#define mico_mac_2x1(a, b) ({                 \
    int32_t _r;                                \
    __asm__ volatile(".insn r 0x0B, 0x7, 0x04, %0, %1, %2" \
                     : "=r"(_r)                \
                     : "r"(a), "r"(b));        \
    _r;                                        \
})


void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    // Check if it is possible to unroll
    const size_t qdwords = in_features / 8;
    const size_t remain = in_features % 8;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features);
            for (size_t k = 0; k < qdwords; k++) {
                sum += mico_v8s8_mac(a_ptr[k], w_ptr[k]);
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                sum += x->data[i * in_features + k] *
                       w->data[j * in_features + k];
            }
            O[i * out_features + j] = sum;
        }
    }
}

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;
    const size_t a_qdwords = in_features / 8;
    const size_t remain = in_features % 8;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 2);
            size_t w_idx = 0;
            size_t k = 0;
            while (k + 1 < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_8x4(a_ptr[k], w_word);
                acc_sum += mico_mac_8x4(a_ptr[k + 1], w_word);
                k += 2;
            }
            if (k < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_8x4(a_ptr[k], w_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/2];
                temp_w = (t & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc_sum += x->data[i * in_features + t] * temp_w;
                }
                O[i * out_features + j] = acc_sum;
            }
    }
}

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;
    const size_t a_qdwords = in_features / 8;
    const size_t remain = in_features % 8;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 4);
            size_t w_idx = 0;
            size_t k = 0;
            while (k + 3 < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_8x2(a_ptr[k + 0], w_word);
                acc_sum += mico_mac_8x2(a_ptr[k + 1], w_word);
                acc_sum += mico_mac_8x2(a_ptr[k + 2], w_word);
                acc_sum += mico_mac_8x2(a_ptr[k + 3], w_word);
                k += 4;
            }
            if (k < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_8x2(a_ptr[k], w_word);
                ++k;
                if (k < a_qdwords) acc_sum += mico_mac_8x2(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x2(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x2(a_ptr[k++], w_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/4];
                temp_w = (temp_w >> (((j * in_features + t) & 0b11) << 1)) & 0x03;
                acc_sum += AMUX_2BIT(temp_w, x->data[i * in_features + t]);
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_w;
    int32_t acc;
    
    const size_t a_qdwords = in_features / 8;
    const size_t remain = in_features % 8;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 8);
            size_t w_idx = 0;
            size_t k = 0;
            while (k + 7 < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_8x1(a_ptr[k + 0], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 1], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 2], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 3], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 4], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 5], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 6], w_word);
                acc_sum += mico_mac_8x1(a_ptr[k + 7], w_word);
                k += 8;
            }
            if (k < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_8x1(a_ptr[k], w_word);
                ++k;
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_8x1(a_ptr[k++], w_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/8];
                temp_w = EXTRACT_BIT(temp_w, (j * in_features + t) & 0b111);
                acc_sum += AMUX_1BIT(temp_w, x->data[i * in_features + t]);
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t qdwords = in_features / 16;
    const size_t remain = in_features % 16;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 2);
            for (size_t k = 0; k < qdwords; k++) {
                acc_sum += mico_v16s4_mac(a_ptr[k], w_ptr[k]);
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                temp_w = w->data[(j * in_features + k)/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                temp_a = x->data[(i * in_features + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t a_qdwords = in_features / 16;
    const size_t remain = in_features % 16;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 4);
            size_t w_idx = 0;
            size_t k = 0;
            while (k + 1 < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_4x2(a_ptr[k], w_word);
                acc_sum += mico_mac_4x2(a_ptr[k + 1], w_word);
                k += 2;
            }
            if (k < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_4x2(a_ptr[k], w_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/4];
                temp_w = EXTRACT_2BIT(temp_w, t & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + t)/2];
                temp_a = EXTRACT_4BIT(temp_a, t & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t a_qdwords = in_features / 16;
    const size_t remain = in_features % 16;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 8);
            size_t w_idx = 0;
            size_t k = 0;
            while (k + 3 < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_4x1(a_ptr[k + 0], w_word);
                acc_sum += mico_mac_4x1(a_ptr[k + 1], w_word);
                acc_sum += mico_mac_4x1(a_ptr[k + 2], w_word);
                acc_sum += mico_mac_4x1(a_ptr[k + 3], w_word);
                k += 4;
            }
            if (k < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_4x1(a_ptr[k], w_word);
                ++k;
                if (k < a_qdwords) acc_sum += mico_mac_4x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_4x1(a_ptr[k++], w_word);
                if (k < a_qdwords) acc_sum += mico_mac_4x1(a_ptr[k++], w_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/8];
                temp_w = EXTRACT_BIT(temp_w, t & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + t)/2];
                temp_a = EXTRACT_4BIT(temp_a, t & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t qdwords = in_features / 32;
    const size_t remain = in_features % 32;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 4);
            for (size_t k = 0; k < qdwords; k++) {
                acc_sum += mico_v32s2_mac(a_ptr[k], w_ptr[k]);
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t a_qdwords = in_features / 32;
    const size_t remain = in_features % 32;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 8);
            size_t w_idx = 0;
            size_t k = 0;
            while (k + 1 < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_2x1(a_ptr[k], w_word);
                acc_sum += mico_mac_2x1(a_ptr[k + 1], w_word);
                k += 2;
            }
            if (k < a_qdwords) {
                const qdword w_word = w_ptr[w_idx++];
                acc_sum += mico_mac_2x1(a_ptr[k], w_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/8];
                temp_w = EXTRACT_BIT(temp_w, t & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + t)/4];
                temp_a = EXTRACT_2BIT(temp_a, t & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t qdwords = in_features / 64;
    const size_t remain = in_features % 64;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 8);
            for (size_t k = 0; k < qdwords; k++) {
                acc_sum += mico_v64s1_mac(a_ptr[k], w_ptr[k]);
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + k)/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}


// -----------------------------------------------------------------------------
void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int32_t acc;
    const size_t a_qdwords = in_features / 16;
    const size_t remain = in_features % 16;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features);
            for (size_t a_idx = 0; a_idx < a_qdwords; a_idx++) {
                const qdword a_word = a_ptr[a_idx];
                const size_t w_base = a_idx * 2;
                acc_sum += mico_mac_8x4(w_ptr[w_base], a_word);
                acc_sum += mico_mac_8x4(w_ptr[w_base + 1], a_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/2];
                temp_a = EXTRACT_4BIT(temp_a, t & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * w->data[j * in_features + t];
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int32_t acc;
    const size_t a_qdwords = in_features / 32;
    const size_t remain = in_features % 32;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features);
            for (size_t a_idx = 0; a_idx < a_qdwords; a_idx++) {
                const qdword a_word = a_ptr[a_idx];
                const size_t w_base = a_idx * 4;
                acc_sum += mico_mac_8x2(w_ptr[w_base + 0], a_word);
                acc_sum += mico_mac_8x2(w_ptr[w_base + 1], a_word);
                acc_sum += mico_mac_8x2(w_ptr[w_base + 2], a_word);
                acc_sum += mico_mac_8x2(w_ptr[w_base + 3], a_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/4];
                temp_a = EXTRACT_2BIT(temp_a, t & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc_sum += temp_a * w->data[j * in_features + t];
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int32_t acc;

    const size_t a_qdwords = in_features / 64;
    const size_t remain = in_features % 64;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features);
            for (size_t a_idx = 0; a_idx < a_qdwords; a_idx++) {
                const qdword a_word = a_ptr[a_idx];
                const size_t w_base = a_idx * 8;
                acc_sum += mico_mac_8x1(w_ptr[w_base + 0], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 1], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 2], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 3], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 4], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 5], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 6], a_word);
                acc_sum += mico_mac_8x1(w_ptr[w_base + 7], a_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/8];
                temp_a = EXTRACT_BIT(temp_a, t & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc_sum += temp_a * w->data[j * in_features + t];
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t a_qdwords = in_features / 32;
    const size_t remain = in_features % 32;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 2);
            for (size_t a_idx = 0; a_idx < a_qdwords; a_idx++) {
                const qdword a_word = a_ptr[a_idx];
                const size_t w_base = a_idx * 2;
                acc_sum += mico_mac_4x2(w_ptr[w_base], a_word);
                acc_sum += mico_mac_4x2(w_ptr[w_base + 1], a_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/4];
                temp_a = EXTRACT_2BIT(temp_a, t & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                temp_w = w->data[(j * in_features + t)/2];
                temp_w = EXTRACT_4BIT(temp_w, t & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;

    const size_t a_qdwords = in_features / 64;
    const size_t remain = in_features % 64;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 2);
            for (size_t a_idx = 0; a_idx < a_qdwords; a_idx++) {
                const qdword a_word = a_ptr[a_idx];
                const size_t w_base = a_idx * 4;
                acc_sum += mico_mac_4x1(w_ptr[w_base + 0], a_word);
                acc_sum += mico_mac_4x1(w_ptr[w_base + 1], a_word);
                acc_sum += mico_mac_4x1(w_ptr[w_base + 2], a_word);
                acc_sum += mico_mac_4x1(w_ptr[w_base + 3], a_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/8];
                temp_a = EXTRACT_BIT(temp_a, t & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                temp_w = w->data[(j * in_features + t)/2];
                temp_w = EXTRACT_4BIT(temp_w, t & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    int8_t temp_a;
    int8_t temp_w;
    int32_t acc;
    const size_t a_qdwords = in_features / 64;
    const size_t remain = in_features % 64;
    for (size_t i = 0; i < batch_size; i++) {
        const qdword *a_ptr = (const qdword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qdword *w_ptr = (const qdword*)(w->data + j * in_features / 4);
            for (size_t a_idx = 0; a_idx < a_qdwords; a_idx++) {
                const qdword a_word = a_ptr[a_idx];
                const size_t w_base = a_idx * 2;
                acc_sum += mico_mac_2x1(w_ptr[w_base], a_word);
                acc_sum += mico_mac_2x1(w_ptr[w_base + 1], a_word);
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/8];
                temp_a = EXTRACT_BIT(temp_a, t & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                temp_w = w->data[(j * in_features + t)/4];
                temp_w = EXTRACT_2BIT(temp_w, t & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}
