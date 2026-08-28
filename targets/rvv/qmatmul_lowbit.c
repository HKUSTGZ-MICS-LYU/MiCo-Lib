#include "mico_qnn.h"

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

#ifndef __riscv_vector
#error "The MiCo RVV target requires the RISC-V vector extension"
#endif

/*
 * Low-bit weights are packed along the innermost logical dimension. The
 * current quantizers use low-bit-first ordering within each byte.
 */
static inline vint8m1_t mico_rvv_decode_packed(
    vuint8m1_t packed, unsigned bits, unsigned lane, size_t vl) {
    if (bits == 4) {
        vuint8m1_t code = __riscv_vsrl_vx_u8m1(packed, lane * 4u, vl);
        code = __riscv_vand_vx_u8m1(code, 0x0f, vl);
        vint8m1_t signed_code = __riscv_vreinterpret_v_u8m1_i8m1(code);
        signed_code = __riscv_vsll_vx_i8m1(signed_code, 4, vl);
        return __riscv_vsra_vx_i8m1(signed_code, 4, vl);
    }

    if (bits == 2) {
        vuint8m1_t code = __riscv_vsrl_vx_u8m1(packed, lane * 2u, vl);
        code = __riscv_vand_vx_u8m1(code, 0x03, vl);
        vint8m1_t signed_code = __riscv_vreinterpret_v_u8m1_i8m1(code);
        signed_code = __riscv_vsll_vx_i8m1(signed_code, 6, vl);
        return __riscv_vsra_vx_i8m1(signed_code, 6, vl);
    }

    /* Q1 convention: bit 0 is +1 and bit 1 is -1. */
    vuint8m1_t bit = __riscv_vsrl_vx_u8m1(packed, lane, vl);
    bit = __riscv_vand_vx_u8m1(bit, 1, vl);
    vint8m1_t twice = __riscv_vreinterpret_v_u8m1_i8m1(bit);
    twice = __riscv_vsll_vx_i8m1(twice, 1, vl);
    vint8m1_t one = __riscv_vmv_v_x_i8m1(1, vl);
    return __riscv_vsub_vv_i8m1(one, twice, vl);
}

/*
 * Dot product between a packed K-vector and a raw INT8 K-vector. The raw
 * vector may be contiguous or have a fixed byte stride. Keeping each packed
 * lane separate avoids materializing a full dequantized weight matrix.
 */
static int32_t mico_rvv_dot_packed_k(
    const uint8_t *packed, const int8_t *raw, size_t raw_stride,
    size_t length, unsigned bits) {
    const size_t values_per_byte = 8u / bits;
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    int32_t result = 0;

    for (unsigned lane = 0; lane < values_per_byte; ++lane) {
        const size_t lane_length = length > lane
            ? (length - lane + values_per_byte - 1u) / values_per_byte
            : 0;
        vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax32);

        for (size_t offset = 0; offset < lane_length;) {
            const size_t vl = __riscv_vsetvl_e8m1(lane_length - offset);
            const vuint8m1_t packed_v =
                __riscv_vle8_v_u8m1(packed + offset, vl);
            const size_t raw_index = offset * values_per_byte + lane;
            const vint8m1_t raw_v = __riscv_vlse8_v_i8m1(
                raw + raw_index * raw_stride,
                (ptrdiff_t)(values_per_byte * raw_stride), vl);
            const vint8m1_t weight_v =
                mico_rvv_decode_packed(packed_v, bits, lane, vl);
            const vint16m2_t raw16 =
                __riscv_vwcvt_x_x_v_i16m2(raw_v, vl);
            const vint16m2_t weight16 =
                __riscv_vwcvt_x_x_v_i16m2(weight_v, vl);

            vacc = __riscv_vwmacc_vv_i32m4_tu(
                vacc, raw16, weight16, vl);
            offset += vl;
        }

        const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
        const vint32m1_t sum =
            __riscv_vredsum_vs_i32m4_i32m1(vacc, zero, vlmax32);
        result += __riscv_vmv_x_s_i32m1_i32(sum);
    }

    return result;
}

#ifdef USE_ALT_LAYOUT
/* Q8 input times W[K, M] packed over M, used by alternate layout. */
static int32_t mico_rvv_dot_q8_packed_m(
    const int8_t *raw, const uint8_t *packed, size_t length,
    size_t output, size_t outputs, unsigned bits) {
    const size_t values_per_byte = 8u / bits;
    const size_t row_bytes =
        (outputs + values_per_byte - 1u) / values_per_byte;
    const size_t byte_index = output / values_per_byte;
    const unsigned lane = (unsigned)(output % values_per_byte);
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax32);

    for (size_t offset = 0; offset < length;) {
        const size_t vl = __riscv_vsetvl_e8m1(length - offset);
        const vint8m1_t raw_v = __riscv_vle8_v_i8m1(raw + offset, vl);
        const vuint8m1_t packed_v = __riscv_vlse8_v_u8m1(
            packed + offset * row_bytes + byte_index,
            (ptrdiff_t)row_bytes, vl);
        const vint8m1_t weight_v =
            mico_rvv_decode_packed(packed_v, bits, lane, vl);
        const vint16m2_t raw16 =
            __riscv_vwcvt_x_x_v_i16m2(raw_v, vl);
        const vint16m2_t weight16 =
            __riscv_vwcvt_x_x_v_i16m2(weight_v, vl);

        vacc = __riscv_vwmacc_vv_i32m4_tu(vacc, raw16, weight16, vl);
        offset += vl;
    }

    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const vint32m1_t sum =
        __riscv_vredsum_vs_i32m4_i32m1(vacc, zero, vlmax32);
    return __riscv_vmv_x_s_i32m1_i32(sum);
}
#endif

static inline vint16m2_t mico_rvv_extract_bit16(
    vuint8m1_t packed, unsigned shift, size_t vl) {
    vuint8m1_t bit = __riscv_vsrl_vx_u8m1(packed, shift, vl);
    bit = __riscv_vand_vx_u8m1(bit, 1, vl);
    const vint8m1_t bit8 = __riscv_vreinterpret_v_u8m1_i8m1(bit);
    return __riscv_vwcvt_x_x_v_i16m2(bit8, vl);
}

static inline int16_t mico_rvv_decode_scalar(
    uint8_t packed, unsigned bits, unsigned lane) {
    if (bits == 4) {
        const unsigned code = (packed >> (lane * 4u)) & 0x0fu;
        return code < 8u ? (int16_t)code : (int16_t)code - 16;
    }
    if (bits == 2) {
        const unsigned code = (packed >> (lane * 2u)) & 0x03u;
        if (code == 1) return 1;
        if (code == 2) return -2;
        if (code == 3) return -1;
        return 0;
    }
    return ((packed >> lane) & 1u) ? -1 : 1;
}

#ifdef USE_ALT_LAYOUT
/*
 * Direct Q8 x Q2/Q1 dot product. The packed operand stays packed: Q1/Q2
 * are accumulated as bit planes and reduced only after all K groups.
 */
static __attribute__((noinline)) int32_t mico_rvv_dot_packed_direct_k(
    const int8_t *raw, size_t raw_stride, const uint8_t *packed,
    size_t packed_stride, size_t length, unsigned bits, unsigned lane) {
    const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, vlmax32);
    vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, vlmax32);

    for (size_t offset = 0; offset < length;) {
        const size_t remaining = length - offset;
        const size_t vl = remaining < vlmax8 ? remaining : vlmax8;
        (void)__riscv_vsetvl_e8m1(vl);
        const vint8m1_t raw_v = __riscv_vlse8_v_i8m1(
            raw + offset * raw_stride, (ptrdiff_t)raw_stride, vl);
        const vuint8m1_t packed_v = __riscv_vlse8_v_u8m1(
            packed + offset * packed_stride, (ptrdiff_t)packed_stride, vl);
        const vint16m2_t raw16 =
            __riscv_vwcvt_x_x_v_i16m2(raw_v, vl);

        if (bits == 1) {
            const vint16m2_t bit =
                mico_rvv_extract_bit16(packed_v, lane, vl);
            acc0 = __riscv_vwmacc_vx_i32m4_tu(
                acc0, 1, raw16, vl);
            acc1 = __riscv_vwmacc_vv_i32m4_tu(
                acc1, raw16, bit, vl);
        } else {
            const unsigned shift = lane * 2u;
            const vint16m2_t bit0 =
                mico_rvv_extract_bit16(packed_v, shift, vl);
            const vint16m2_t bit1 =
                mico_rvv_extract_bit16(packed_v, shift + 1u, vl);
            acc0 = __riscv_vwmacc_vv_i32m4_tu(
                acc0, raw16, bit0, vl);
            acc1 = __riscv_vwmacc_vv_i32m4_tu(
                acc1, raw16, bit1, vl);
        }
        offset += vl;
    }

    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const vint32m1_t sum0 =
        __riscv_vredsum_vs_i32m4_i32m1(acc0, zero, vlmax32);
    const vint32m1_t sum1 =
        __riscv_vredsum_vs_i32m4_i32m1(acc1, zero, vlmax32);
    return __riscv_vmv_x_s_i32m1_i32(sum0) -
           2 * __riscv_vmv_x_s_i32m1_i32(sum1);
}
#endif

#ifndef USE_ALT_LAYOUT
/* Default W[M,K]: vectorize output channels and accumulate bit planes. */
static void mico_rvv_q8xpacked_direct(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
    const size_t batch_size = x->shape[0];
    const size_t inputs = x->shape[1];
    const size_t outputs = w->shape[0];
    const size_t values_per_byte = 8u / bits;
    const size_t row_bytes =
        (inputs + values_per_byte - 1u) / values_per_byte;
    const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const int8_t *x_row = x->data + batch * inputs;
        for (size_t output_offset = 0; output_offset < outputs;) {
            const size_t remaining = outputs - output_offset;
            const size_t vl = remaining < vlmax8 ? remaining : vlmax8;
            (void)__riscv_vsetvl_e8m1(vl);
            const uint8_t *packed = (const uint8_t *)w->data +
                output_offset * row_bytes;
            vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, vlmax32);
            vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, vlmax32);

            for (size_t k = 0; k < inputs; ++k) {
                const unsigned shift =
                    (unsigned)bits * (unsigned)(k % values_per_byte);
                const vuint8m1_t packed_v = __riscv_vlse8_v_u8m1(
                    packed + k / values_per_byte,
                    (ptrdiff_t)row_bytes, vl);
                const int16_t scalar = (int16_t)x_row[k];
                if (bits == 1) {
                    const vint16m2_t bit =
                        mico_rvv_extract_bit16(packed_v, shift, vl);
                    acc0 = __riscv_vadd_vx_i32m4(
                        acc0, (int32_t)scalar, vl);
                    acc1 = __riscv_vwmacc_vx_i32m4_tu(
                        acc1, scalar, bit, vl);
                } else {
                    const vint16m2_t bit0 =
                        mico_rvv_extract_bit16(packed_v, shift, vl);
                    const vint16m2_t bit1 =
                        mico_rvv_extract_bit16(packed_v, shift + 1u, vl);
                    acc0 = __riscv_vwmacc_vx_i32m4_tu(
                        acc0, scalar, bit0, vl);
                    acc1 = __riscv_vwmacc_vx_i32m4_tu(
                        acc1, scalar, bit1, vl);
                }
            }

            const vint32m4_t twice =
                __riscv_vsll_vx_i32m4(acc1, 1, vl);
            acc0 = __riscv_vsub_vv_i32m4(acc0, twice, vl);
            (void)__riscv_vsetvl_e32m4(vl);
            __riscv_vse32_v_i32m4(
                output + batch * outputs + output_offset, acc0, vl);
            output_offset += vl;
        }
    }
}
#else
/* Alternate W[K,M]: keep each output's packed lane in a K-stream. */
static void mico_rvv_q8xpacked_direct(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
    const size_t batch_size = x->shape[0];
    const size_t inputs = x->shape[1];
    const size_t outputs = w->shape[1];
    const size_t values_per_byte = 8u / bits;
    const size_t row_bytes =
        (outputs + values_per_byte - 1u) / values_per_byte;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const int8_t *x_row = x->data + batch * inputs;
        for (size_t output_index = 0; output_index < outputs;
             ++output_index) {
            const size_t lane = output_index % values_per_byte;
            const uint8_t *packed = (const uint8_t *)w->data +
                output_index / values_per_byte;
            output[batch * outputs + output_index] =
                mico_rvv_dot_packed_direct_k(
                    x_row, 1, packed, row_bytes, inputs, bits,
                    (unsigned)lane);
        }
    }
}
#endif

/* Direct packed Q2/Q1 input x raw INT8 weight path. */
static void mico_rvv_qpackedx8_direct(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
    const size_t batch_size = x->shape[0];
    const size_t inputs = x->shape[1];
#ifdef USE_ALT_LAYOUT
    const size_t outputs = w->shape[1];
#else
    const size_t outputs = w->shape[0];
#endif
    const size_t values_per_byte = 8u / bits;
    const size_t row_bytes =
        (inputs + values_per_byte - 1u) / values_per_byte;
    const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const uint8_t *packed =
            (const uint8_t *)x->data + batch * row_bytes;
        for (size_t output_offset = 0; output_offset < outputs;) {
            const size_t remaining = outputs - output_offset;
            const size_t vl = remaining < vlmax8 ? remaining : vlmax8;
            (void)__riscv_vsetvl_e8m1(vl);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax32);

            for (size_t k = 0; k < inputs; ++k) {
                const int16_t scalar = mico_rvv_decode_scalar(
                    packed[k / values_per_byte], bits,
                    (unsigned)(k % values_per_byte));
                const vint8m1_t raw =
#ifdef USE_ALT_LAYOUT
                    __riscv_vle8_v_i8m1(
                        w->data + k * outputs + output_offset, vl);
#else
                    __riscv_vlse8_v_i8m1(
                        w->data + output_offset * inputs + k,
                        (ptrdiff_t)inputs, vl);
#endif
                const vint16m2_t raw16 =
                    __riscv_vwcvt_x_x_v_i16m2(raw, vl);
                acc = __riscv_vwmacc_vx_i32m4_tu(
                    acc, scalar, raw16, vl);
            }

            (void)__riscv_vsetvl_e32m4(vl);
            __riscv_vse32_v_i32m4(
                output + batch * outputs + output_offset, acc, vl);
            output_offset += vl;
        }
    }
}

/* Q8 input times packed low-bit weight. */
static void mico_rvv_q8xpacked_matmul(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
    const size_t batch_size = x->shape[0];
    const size_t inputs = x->shape[1];

#ifdef USE_ALT_LAYOUT
    const size_t outputs = w->shape[1];
#else
    const size_t outputs = w->shape[0];
    const size_t values_per_byte = 8u / bits;
    const size_t row_bytes =
        (inputs + values_per_byte - 1u) / values_per_byte;
#endif

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const int8_t *x_row = x->data + batch * inputs;
        for (size_t output_index = 0; output_index < outputs;
             ++output_index) {
#ifdef USE_ALT_LAYOUT
            const uint8_t *packed = (const uint8_t *)w->data;
            output[batch * outputs + output_index] =
                mico_rvv_dot_q8_packed_m(
                    x_row, packed, inputs, output_index, outputs, bits);
#else
            const uint8_t *packed =
                (const uint8_t *)w->data + output_index * row_bytes;
            output[batch * outputs + output_index] =
                mico_rvv_dot_packed_k(packed, x_row, 1, inputs, bits);
#endif
        }
    }
}

/* Packed low-bit input times raw INT8 weight. */
static void mico_rvv_qpackedx8_matmul(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
    const size_t batch_size = x->shape[0];
    const size_t inputs = x->shape[1];
    const size_t values_per_byte = 8u / bits;

#ifdef USE_ALT_LAYOUT
    const size_t outputs = w->shape[1];
#else
    const size_t outputs = w->shape[0];
#endif
    const size_t row_bytes =
        (inputs + values_per_byte - 1u) / values_per_byte;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const uint8_t *packed =
            (const uint8_t *)x->data + batch * row_bytes;
        for (size_t output_index = 0; output_index < outputs;
             ++output_index) {
#ifdef USE_ALT_LAYOUT
            const int8_t *raw = w->data + output_index;
            const size_t raw_stride = outputs;
#else
            const int8_t *raw = w->data + output_index * inputs;
            const size_t raw_stride = 1;
#endif
            output[batch * outputs + output_index] =
                mico_rvv_dot_packed_k(
                    packed, raw, raw_stride, inputs, bits);
        }
    }
}

/* Packed low-bit activation times packed low-bit weight. */
static void mico_rvv_packedxpacked_matmul(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned activation_bits, unsigned weight_bits) {
    const size_t batch_size = x->shape[0];
    const size_t inputs = x->shape[1];
    const size_t activation_values_per_byte = 8u / activation_bits;
    const size_t weight_values_per_byte = 8u / weight_bits;
#ifdef USE_ALT_LAYOUT
    const size_t outputs = w->shape[1];
    const size_t weight_row_bytes =
        (outputs + weight_values_per_byte - 1u) /
        weight_values_per_byte;
    const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const uint8_t *activation = (const uint8_t *)x->data +
            batch * ((inputs + activation_values_per_byte - 1u) /
                     activation_values_per_byte);
        for (unsigned lane = 0; lane < weight_values_per_byte; ++lane) {
            const size_t lane_groups = outputs > lane
                ? (outputs - lane + weight_values_per_byte - 1u) /
                    weight_values_per_byte
                : 0;
            for (size_t group = 0; group < lane_groups;) {
                const size_t remaining = lane_groups - group;
                const size_t vl = remaining < vlmax8 ? remaining : vlmax8;
                (void)__riscv_vsetvl_e8m1(vl);
                vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax32);

                for (size_t k = 0; k < inputs; ++k) {
                    const int16_t activation_value =
                        mico_rvv_decode_scalar(
                            activation[k / activation_values_per_byte],
                            activation_bits,
                            (unsigned)(k % activation_values_per_byte));
                    const size_t weight_byte = k * weight_row_bytes + group;
                    const vuint8m1_t packed_weight = __riscv_vle8_v_u8m1(
                        (const uint8_t *)w->data + weight_byte, vl);
                    const vint8m1_t weight_value = mico_rvv_decode_packed(
                        packed_weight, weight_bits, lane, vl);
                    const vint16m2_t weight16 =
                        __riscv_vwcvt_x_x_v_i16m2(weight_value, vl);
                    acc = __riscv_vwmacc_vx_i32m4_tu(
                        acc, activation_value, weight16, vl);
                }

                (void)__riscv_vsetvl_e32m4(vl);
                __riscv_vsse32_v_i32m4(
                    output + batch * outputs + group * weight_values_per_byte + lane,
                    (ptrdiff_t)(weight_values_per_byte * sizeof(int32_t)),
                    acc, vl);
                group += vl;
            }
        }
    }
#else
    const size_t outputs = w->shape[0];
    const size_t activation_row_bytes =
        (inputs + activation_values_per_byte - 1u) /
        activation_values_per_byte;
    const size_t weight_row_bytes =
        (inputs + weight_values_per_byte - 1u) /
        weight_values_per_byte;
    const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const uint8_t *activation = (const uint8_t *)x->data +
            batch * activation_row_bytes;
        for (size_t output_offset = 0; output_offset < outputs;) {
            const size_t remaining = outputs - output_offset;
            const size_t vl = remaining < vlmax8 ? remaining : vlmax8;
            (void)__riscv_vsetvl_e8m1(vl);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax32);

            for (size_t k = 0; k < inputs; ++k) {
                const int16_t activation_value =
                    mico_rvv_decode_scalar(
                        activation[k / activation_values_per_byte],
                        activation_bits,
                        (unsigned)(k % activation_values_per_byte));
                const vuint8m1_t packed_weight = __riscv_vlse8_v_u8m1(
                    (const uint8_t *)w->data +
                        output_offset * weight_row_bytes +
                        k / weight_values_per_byte,
                    (ptrdiff_t)weight_row_bytes, vl);
                const vint8m1_t weight_value = mico_rvv_decode_packed(
                    packed_weight, weight_bits,
                    (unsigned)(k % weight_values_per_byte), vl);
                const vint16m2_t weight16 =
                    __riscv_vwcvt_x_x_v_i16m2(weight_value, vl);
                acc = __riscv_vwmacc_vx_i32m4_tu(
                    acc, activation_value, weight16, vl);
            }

            (void)__riscv_vsetvl_e32m4(vl);
            __riscv_vse32_v_i32m4(
                output + batch * outputs + output_offset, acc, vl);
            output_offset += vl;
        }
    }
#endif
}

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                    const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 4, 4);
}

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 4, 2);
}

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 4, 1);
}

void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 2, 4);
}

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                    const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 2, 2);
}

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 2, 1);
}

void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 1, 4);
}

void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 1, 2);
}

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                    const Tensor2D_Q8 *w) {
    mico_rvv_packedxpacked_matmul(O, x, w, 1, 1);
}

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_q8xpacked_matmul(O, x, w, 4);
}

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_q8xpacked_direct(O, x, w, 2);
}

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_q8xpacked_direct(O, x, w, 1);
}

void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_qpackedx8_matmul(O, x, w, 4);
}

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_qpackedx8_direct(O, x, w, 2);
}

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
    mico_rvv_qpackedx8_direct(O, x, w, 1);
}
