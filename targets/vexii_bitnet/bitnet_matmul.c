#include "mico_qnn.h"

#ifndef USE_SIMD
#define USE_SIMD 32
#endif

#ifndef BITNET_QUANT
#define BITNET_QUANT 3
#endif

// #define USE_DOTP8X2

// Activation
typedef uint32_t int8x4_t;

// Quantized Weight
typedef uint8_t int2x4_t;
typedef uint16_t int2x8_t;
typedef uint32_t int2x16_t;
typedef uint8_t int1x8_t;
typedef uint16_t int1x16_t;
typedef uint32_t int1x32_t;

#define bn_sum8(a, b) ({                                      \
    int32_t _r;                                               \
    __asm__ volatile(".insn r 0x0B, 0x1, 0x00, %0, %1, %2"   \
                     : "=r"(_r)                               \
                     : "r"(a), "r"(b));                      \
    _r;                                                       \
})

#define bn_sum4(a, b) ({                                      \
    int32_t _r;                                               \
    __asm__ volatile(".insn r 0x0B, 0x1, 0x00, %0, %1, %2"   \
                     : "=r"(_r)                               \
                     : "r"(a), "r"(b));                      \
    _r;                                                       \
})

#define bn_store(w0, w1)                                      \
    __asm__ volatile(".insn r 0x0B, 0x2, 0x00, x0, %0, %1"   \
                     :                                        \
                     : "r"(w0), "r"(w1))

#define bn_dotp8(a, b) ({                                     \
    int32_t _r;                                               \
    __asm__ volatile(".insn r 0x0B, 0x4, 0x01, %0, %1, %2"   \
                     : "=r"(_r)                               \
                     : "r"(a), "r"(b));                      \
    _r;                                                       \
})

#define bn_dotp8x2(a, b) ({                                   \
    int32_t _r;                                               \
    __asm__ volatile(".insn r 0x0B, 0x5, 0x01, %0, %1, %2"   \
                     : "=r"(_r)                               \
                     : "r"(a), "r"(b));                      \
    _r;                                                       \
})

static inline int32_t bitnet_row_acc_q8xq(const int8_t *input, const int8_t *weight, int n) {
    int32_t acc = 0;

    // BitNetBufferPlugin maps BNSUM as:
    // RESULT = bitnetadd4(rs1, buffer_low) + bitnetadd4(rs2, buffer_high)
    // so software keeps one consistent operand order across quant modes.
#if BITNET_QUANT == 2
    int j = 0;
#if USE_SIMD == 64
    const int aligned = n & ~63;
    for (; j < aligned; j += 64) {
        const uint32_t *w_ptr = (const uint32_t *)(weight + (j >> 3));
        bn_store(w_ptr[1], w_ptr[0]);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 8), *(const int8x4_t *)(input + j + 12));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 16), *(const int8x4_t *)(input + j + 20));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 24), *(const int8x4_t *)(input + j + 28));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 32), *(const int8x4_t *)(input + j + 36));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 40), *(const int8x4_t *)(input + j + 44));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 48), *(const int8x4_t *)(input + j + 52));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 56), *(const int8x4_t *)(input + j + 60));
    }
#elif USE_SIMD == 32
    const int aligned = n & ~31;
    for (; j < aligned; j += 32) {
        const uint32_t w_word = *(const uint32_t *)(weight + (j >> 3));
        bn_store(0, w_word);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 8), *(const int8x4_t *)(input + j + 12));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 16), *(const int8x4_t *)(input + j + 20));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 24), *(const int8x4_t *)(input + j + 28));
    }
#elif USE_SIMD == 16
    const int aligned = n & ~15;
    for (; j < aligned; j += 16) {
        const uint16_t w_word = *(const uint16_t *)(weight + (j >> 3));
        bn_store(0, w_word);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 8), *(const int8x4_t *)(input + j + 12));
    }
#elif USE_SIMD == 8
    const int aligned = n & ~7;
    for (; j < aligned; j += 8) {
        const uint8_t w_word = *(const uint8_t *)(weight + (j >> 3));
        bn_store(0, w_word);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
    }
#elif USE_SIMD == 4
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const uint8_t w_byte = *(const uint8_t *)(weight + (j >> 3));
        const int32_t w_pack = (int32_t)((w_byte >> (j & 0x7)) & 0x0F);
        acc += bn_sum4(*(const int8x4_t *)(input + j), w_pack);
    }
#else
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const uint8_t w_byte = *(const uint8_t *)(weight + (j >> 3));
        acc += AMUX_1BIT(EXTRACT_BIT(w_byte, (j + 0) & 0x7), input[j + 0]);
        acc += AMUX_1BIT(EXTRACT_BIT(w_byte, (j + 1) & 0x7), input[j + 1]);
        acc += AMUX_1BIT(EXTRACT_BIT(w_byte, (j + 2) & 0x7), input[j + 2]);
        acc += AMUX_1BIT(EXTRACT_BIT(w_byte, (j + 3) & 0x7), input[j + 3]);
    }
#endif
    for (; j < n; ++j) {
        const uint8_t w_byte = *(const uint8_t *)(weight + (j >> 3));
        const uint8_t bit = EXTRACT_BIT(w_byte, j & 0x7);
        acc += AMUX_1BIT(bit, input[j]);
    }
#else
    int j = 0;
#if USE_SIMD == 32
    const int aligned = n & ~31;
    for (; j < aligned; j += 32) {
        const uint32_t *w_ptr = (const uint32_t *)(weight + (j >> 2));
        bn_store(w_ptr[1], w_ptr[0]);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 8), *(const int8x4_t *)(input + j + 12));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 16), *(const int8x4_t *)(input + j + 20));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 24), *(const int8x4_t *)(input + j + 28));
    }
#elif USE_SIMD == 16
    const int aligned = n & ~15;
    for (; j < aligned; j += 16) {
        const uint32_t w_word = *(const uint32_t *)(weight + (j >> 2));
        bn_store(0, w_word);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(input + j + 8), *(const int8x4_t *)(input + j + 12));
    }
#elif USE_SIMD == 8
    const int aligned = n & ~7;
    for (; j < aligned; j += 8) {
        const uint16_t w_word = *(const uint16_t *)(weight + (j >> 2));
        bn_store(0, w_word);
        acc += bn_sum8(*(const int8x4_t *)(input + j + 0), *(const int8x4_t *)(input + j + 4));
    }
#elif USE_SIMD == 4
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const int32_t w_pack = *(const uint8_t *)(weight + (j >> 2));
        acc += bn_sum4(*(const int8x4_t *)(input + j), w_pack);
    }
#else
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const uint8_t w_word = *(const uint8_t *)(weight + (j >> 2));
        acc += AMUX_2BIT(EXTRACT_2BIT(w_word, 0), input[j + 0]);
        acc += AMUX_2BIT(EXTRACT_2BIT(w_word, 1), input[j + 1]);
        acc += AMUX_2BIT(EXTRACT_2BIT(w_word, 2), input[j + 2]);
        acc += AMUX_2BIT(EXTRACT_2BIT(w_word, 3), input[j + 3]);
    }
#endif
    for (; j < n; ++j) {
        const uint8_t w_byte = *(const uint8_t *)(weight + (j >> 2));
        acc += AMUX_2BIT(EXTRACT_2BIT(w_byte, j & 0x3), input[j]);
    }
#endif

    return acc;
}

static inline int32_t bitnet_row_acc_qx8(const int8_t *packed_input, const int8_t *weight, int n) {
    int32_t acc = 0;

    // Keep the same BNSUM operand order as bitnet_row_acc_q8xq():
    // rs1 pairs with buffer_low, rs2 pairs with buffer_high.
#if BITNET_QUANT == 2
    int j = 0;
#if USE_SIMD == 64
    const int aligned = n & ~63;
    for (; j < aligned; j += 64) {
        const uint32_t *a_ptr = (const uint32_t *)(packed_input + (j >> 3));
        bn_store(a_ptr[1], a_ptr[0]);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 8), *(const int8x4_t *)(weight + j + 12));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 16), *(const int8x4_t *)(weight + j + 20));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 24), *(const int8x4_t *)(weight + j + 28));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 32), *(const int8x4_t *)(weight + j + 36));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 40), *(const int8x4_t *)(weight + j + 44));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 48), *(const int8x4_t *)(weight + j + 52));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 56), *(const int8x4_t *)(weight + j + 60));
    }
#elif USE_SIMD == 32
    const int aligned = n & ~31;
    for (; j < aligned; j += 32) {
        const uint32_t a_word = *(const uint32_t *)(packed_input + (j >> 3));
        bn_store(0, a_word);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 8), *(const int8x4_t *)(weight + j + 12));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 16), *(const int8x4_t *)(weight + j + 20));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 24), *(const int8x4_t *)(weight + j + 28));
    }
#elif USE_SIMD == 16
    const int aligned = n & ~15;
    for (; j < aligned; j += 16) {
        const uint16_t a_word = *(const uint16_t *)(packed_input + (j >> 3));
        bn_store(0, a_word);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 8), *(const int8x4_t *)(weight + j + 12));
    }
#elif USE_SIMD == 8
    const int aligned = n & ~7;
    for (; j < aligned; j += 8) {
        const uint8_t a_word = *(const uint8_t *)(packed_input + (j >> 3));
        bn_store(0, a_word);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
    }
#elif USE_SIMD == 4
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const uint8_t a_byte = *(const uint8_t *)(packed_input + (j >> 3));
        const int32_t a_pack = (int32_t)((a_byte >> (j & 0x7)) & 0x0F);
        acc += bn_sum4(*(const int8x4_t *)(weight + j), a_pack);
    }
#else
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const uint8_t a_byte = *(const uint8_t *)(packed_input + (j >> 3));
        acc += AMUX_1BIT(EXTRACT_BIT(a_byte, (j + 0) & 0x7), weight[j + 0]);
        acc += AMUX_1BIT(EXTRACT_BIT(a_byte, (j + 1) & 0x7), weight[j + 1]);
        acc += AMUX_1BIT(EXTRACT_BIT(a_byte, (j + 2) & 0x7), weight[j + 2]);
        acc += AMUX_1BIT(EXTRACT_BIT(a_byte, (j + 3) & 0x7), weight[j + 3]);
    }
#endif
    for (; j < n; ++j) {
        const uint8_t a_byte = *(const uint8_t *)(packed_input + (j >> 3));
        const uint8_t bit = EXTRACT_BIT(a_byte, j & 0x7);
        acc += AMUX_1BIT(bit, weight[j]);
    }
#else
    int j = 0;
#if USE_SIMD == 32
    const int aligned = n & ~31;
    for (; j < aligned; j += 32) {
        const uint32_t *a_ptr = (const uint32_t *)(packed_input + (j >> 2));
        bn_store(a_ptr[1], a_ptr[0]);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 8), *(const int8x4_t *)(weight + j + 12));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 16), *(const int8x4_t *)(weight + j + 20));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 24), *(const int8x4_t *)(weight + j + 28));
    }
#elif USE_SIMD == 16
    const int aligned = n & ~15;
    for (; j < aligned; j += 16) {
        const uint32_t a_word = *(const uint32_t *)(packed_input + (j >> 2));
        bn_store(0, a_word);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 8), *(const int8x4_t *)(weight + j + 12));
    }
#elif USE_SIMD == 8
    const int aligned = n & ~7;
    for (; j < aligned; j += 8) {
        const uint16_t a_word = *(const uint16_t *)(packed_input + (j >> 2));
        bn_store(0, a_word);
        acc += bn_sum8(*(const int8x4_t *)(weight + j + 0), *(const int8x4_t *)(weight + j + 4));
    }
#elif USE_SIMD == 4
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const int32_t a_pack = *(const uint8_t *)(packed_input + (j >> 2));
        acc += bn_sum4(*(const int8x4_t *)(weight + j), a_pack);
    }
#else
    const int aligned = n & ~3;
    for (; j < aligned; j += 4) {
        const uint8_t a_word = *(const uint8_t *)(packed_input + (j >> 2));
        acc += AMUX_2BIT(EXTRACT_2BIT(a_word, 0), weight[j + 0]);
        acc += AMUX_2BIT(EXTRACT_2BIT(a_word, 1), weight[j + 1]);
        acc += AMUX_2BIT(EXTRACT_2BIT(a_word, 2), weight[j + 2]);
        acc += AMUX_2BIT(EXTRACT_2BIT(a_word, 3), weight[j + 3]);
    }
#endif
    for (; j < n; ++j) {
        const uint8_t a_byte = *(const uint8_t *)(packed_input + (j >> 2));
        acc += AMUX_2BIT(EXTRACT_2BIT(a_byte, j & 0x3), weight[j]);
    }
#endif

    return acc;
}

void bitnet_qmatmul(int8_t *input, int32_t *output, int8_t *weight, int n, int d) {
    for (int i = 0; i < d; ++i) {
#if BITNET_QUANT == 2
        const int8_t *w_row = weight + ((i * n) >> 3);
#else
        const int8_t *w_row = weight + ((i * n) >> 2);
#endif
        output[i] = bitnet_row_acc_q8xq(input, w_row, n);
    }
}

void bitnet_qmatmul_rev(
        int8_t *input,
        int32_t *output,
        int8_t *weight,
        int n, int d) {

    for (int i = 0; i < d; ++i) {
        const int8_t *w_row = weight + i * n;
        output[i] = bitnet_row_acc_qx8(input, w_row, n);
    }
}

#if BITNET_QUANT == 2
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (size_t i = 0; i < batch_size; ++i){
        bitnet_qmatmul(
            x->data + i * in_features,
            O + i * out_features,
            w->data,
            (int)in_features,
            (int)out_features
        );
    }
}

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (size_t i = 0; i < batch_size; ++i){
        bitnet_qmatmul_rev(
            x->data + (i * in_features) / 8,
            O + i * out_features,
            w->data,
            (int)in_features,
            (int)out_features
        );
    }
}

#elif BITNET_QUANT != 0
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

#ifdef USE_DOTP8
    int8_t w_buf[4];
    int8_t temp_w;

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k += 4) {
                temp_w = w->data[(j * in_features + k) / 4];
                for (size_t l = 0; l < 4; ++l) {
                    w_buf[l] = EXTRACT_2BIT(temp_w, l);
                    w_buf[l] = TWO_BIT_TO_INT8(w_buf[l]);
                }
                acc += bn_dotp8(*(int32_t *)(x->data + i * in_features + k), *(int32_t *)w_buf);
            }
            O[i * out_features + j] = acc;
        }
    }
#else
#ifdef USE_DOTP8X2
    int8_t temp_w;

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            int32_t acc = 0;
            for (size_t k = 0; k < in_features; k += 4) {
                temp_w = w->data[(j * in_features + k) / 4];
                acc += bn_dotp8x2(*(int32_t *)(x->data + i * in_features + k), temp_w);
            }
            O[i * out_features + j] = acc;
        }
    }
#else
    for (size_t i = 0; i < batch_size; ++i){
        bitnet_qmatmul(
            x->data + i * in_features,
            O + i * out_features,
            w->data,
            (int)in_features,
            (int)out_features
        );
    }
#endif
#endif
}

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    for (size_t i = 0; i < batch_size; ++i){
        bitnet_qmatmul_rev(
            x->data + (i * in_features) / 4,
            O + i * out_features,
            w->data,
            (int)in_features,
            (int)out_features
        );
    }
}
#endif
