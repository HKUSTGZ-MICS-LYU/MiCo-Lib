#include "mico_qnn.h"
#include "mico_rvv_common.h"
#include "mico_rvv_vicuna_cache.h"

/* The same EEW == SEW restriction applies to packed byte loads. In a
 * widening K-loop GCC may keep e16/m2 across an EEW8 load; both have identical
 * element counts. Pin e8/m1 to the actual load, and expose its vector result.
 */
static inline __attribute__((always_inline)) vuint8m1_t mico_rvv_load_packed_u8m1(
    const uint8_t *src, size_t vl) {
#if defined(RISCV_VEXII) && defined(MICO_RVV)
    vuint8m1_t value;
#if defined(MICO_RVV_BACKEND_VICUNA2)
    /* Same hazard as the Q8 load helper: pinning e8/m1 leaves the hardware
     * VTYPE changed while the compiler still assumes its own setting, so the
     * following widening MAC can silently run at the wrong SEW. Restore the
     * entry VL/VTYPE instead of relying on the compiler to re-issue vsetvl. */
    size_t saved_vl, saved_vtype;
    __asm__ volatile("csrr %[oldvl], vl\n\t"
                     "csrr %[oldtype], vtype\n\t"
                     "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                     "vle8.v %[value], (%[src])\n\t"
                     "vsetvl zero, %[oldvl], %[oldtype]"
                     : [value] "=vr"(value), [oldvl] "=&r"(saved_vl),
                       [oldtype] "=&r"(saved_vtype)
                     : [vl] "r"(vl), [src] "r"(src)
                     : "memory");
#else
    __asm__ volatile("vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                     "vle8.v %[value], (%[src])"
                     : [value] "=vr"(value)
                     : [vl] "r"(vl), [src] "r"(src)
                     : "memory");
#endif
    return value;
#else
    return __riscv_vle8_v_u8m1(src, vl);
#endif
}

static inline __attribute__((always_inline)) vuint8m1_t mico_rvv_load_packed_strided_u8m1(
    const uint8_t *src, ptrdiff_t stride, size_t vl) {
#if defined(RISCV_VEXII) && defined(MICO_RVV)
    vuint8m1_t value;
#if defined(MICO_RVV_BACKEND_VICUNA2)
    /* See mico_rvv_load_packed_u8m1: restore VL/VTYPE after pinning e8/m1. */
    size_t saved_vl, saved_vtype;
    __asm__ volatile("csrr %[oldvl], vl\n\t"
                     "csrr %[oldtype], vtype\n\t"
                     "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                     "vlse8.v %[value], (%[src]), %[stride]\n\t"
                     "vsetvl zero, %[oldvl], %[oldtype]"
                     : [value] "=vr"(value), [oldvl] "=&r"(saved_vl),
                       [oldtype] "=&r"(saved_vtype)
                     : [vl] "r"(vl), [src] "r"(src), [stride] "r"(stride)
                     : "memory");
#else
    __asm__ volatile("vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                     "vlse8.v %[value], (%[src]), %[stride]"
                     : [value] "=vr"(value)
                     : [vl] "r"(vl), [src] "r"(src), [stride] "r"(stride)
                     : "memory");
#endif
    return value;
#else
    return __riscv_vlse8_v_u8m1(src, stride, vl);
#endif
}

/* Coral on Vexii currently accepts output stores only when EEW == SEW.
 * GCC may remove a standalone vsetvl when e16/m2 and e32/m4 have equal VLMAX.
 * Keep configuration and store inseparable, with an explicit accumulator input
 * and memory side effect. Other RVV targets retain the intrinsic implementation.
 */
static inline __attribute__((always_inline)) void mico_rvv_store_i32m4(
    int32_t *dst, vint32m4_t value, size_t vl) {
#if defined(RISCV_VEXII) && defined(MICO_RVV)
    __asm__ volatile("vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
                     "vse32.v %[value], (%[dst])"
                     :: [vl] "r"(vl), [value] "vr"(value), [dst] "r"(dst)
                     : "memory");
#else
    (void)__riscv_vsetvl_e32m4(vl);
    __riscv_vse32_v_i32m4(dst, value, vl);
#endif
}

#ifdef USE_ALT_LAYOUT
static inline __attribute__((always_inline)) void mico_rvv_store_strided_i32m4(
    int32_t *dst, ptrdiff_t stride, vint32m4_t value, size_t vl) {
#if defined(RISCV_VEXII) && defined(MICO_RVV)
    __asm__ volatile("vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
                     "vsse32.v %[value], (%[dst]), %[stride]"
                     :: [vl] "r"(vl), [value] "vr"(value), [dst] "r"(dst),
                        [stride] "r"(stride)
                     : "memory");
#else
    (void)__riscv_vsetvl_e32m4(vl);
    __riscv_vsse32_v_i32m4(dst, stride, value, vl);
#endif
}
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
                mico_rvv_load_packed_u8m1(packed + offset, vl);
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

        result += mico_rvv_reduce_i32m4(vacc, vlmax32);
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
        const vuint8m1_t packed_v = mico_rvv_load_packed_strided_u8m1(
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

    return mico_rvv_reduce_i32m4(vacc, vlmax32);
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
        const vuint8m1_t packed_v = mico_rvv_load_packed_strided_u8m1(
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

    return mico_rvv_reduce_i32m4(acc0, vlmax32) -
           2 * mico_rvv_reduce_i32m4(acc1, vlmax32);
}
#endif

#ifndef USE_ALT_LAYOUT
/* Default W[M,K]: vectorize output channels and accumulate bit planes. */
static void mico_rvv_q8xpacked_direct(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
#ifdef MICO_RVV_EXPLICIT_CACHE
    const mico_rvv_cache_output cache_output = mico_rvv_cache_output_begin(
        output, x->shape[0], w->shape[0]);
    output = cache_output.data;
#endif
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

            /* One packed byte holds values_per_byte consecutive K positions, so
             * load each byte group once and consume all of its sub-lanes. The
             * per-k form reloaded the same byte group four times. */
            const size_t groups =
                (inputs + values_per_byte - 1u) / values_per_byte;
#if defined(MICO_RVV_BACKEND_VICUNA2)
            if (bits == 2) {
                for (size_t g = 0; g < groups; ++g) {
                    const uint8_t *g_ptr =
                        (const uint8_t *)w->data + g * outputs + output_offset;
                    const size_t k_base = g * 4u;
                    vuint8m1_t packed_v;
                    __asm__ volatile(
                        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                        "vle8.v %[v], (%[p])"
                        : [v] "=vr"(packed_v)
                        : [vl] "r"(vl), [p] "r"(g_ptr)
                        : "memory");
                    for (unsigned s = 0; s < 4u; ++s) {
                        const size_t k = k_base + s;
                        if (k >= inputs) break;
                        const int16_t scalar = (int16_t)x_row[k];
                        const unsigned lshift = 6u - 2u * s;
                        vint8m1_t s8 = __riscv_vreinterpret_v_u8m1_i8m1(packed_v);
                        if (lshift != 0)
                            s8 = __riscv_vsll_vx_i8m1(s8, lshift, vl);
                        s8 = __riscv_vsra_vx_i8m1(s8, 6u, vl);
                        const vint16m2_t w16 = __riscv_vwcvt_x_x_v_i16m2(s8, vl);
                        acc0 = __riscv_vwmacc_vx_i32m4_tu(acc0, scalar, w16, vl);
                    }
                }
                mico_rvv_store_i32m4(
                    output + batch * outputs + output_offset, acc0, vl);
                output_offset += vl;
                continue;
            }
#endif
            for (size_t g = 0; g < groups; ++g) {
              const vuint8m1_t saved_packed = mico_rvv_load_packed_u8m1(
                  (const uint8_t *)w->data + g * outputs + output_offset, vl);
              for (unsigned s = 0; s < values_per_byte; ++s) {
                const size_t k = g * values_per_byte + s;
                if (k >= inputs) break;
                const unsigned shift = (unsigned)bits * s;
#if defined(MICO_RVV_BACKEND_VICUNA2)
                /* Vicuna2's vlse8.v is unusable. The packed weights are stored
                 * K-major: byte (k/4, output o) lives at (k/4)*outputs+o, so the
                 * vl consecutive outputs of this K group are contiguous. */
                const vuint8m1_t packed_v = saved_packed;
#else
                const vuint8m1_t packed_v = mico_rvv_load_packed_strided_u8m1(
                    packed + k / values_per_byte,
                    (ptrdiff_t)row_bytes, vl);
#endif
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
            }

            const vint32m4_t twice =
                __riscv_vsll_vx_i32m4(acc1, 1, vl);
            acc0 = __riscv_vsub_vv_i32m4(acc0, twice, vl);
            mico_rvv_store_i32m4(
                output + batch * outputs + output_offset, acc0, vl);
            output_offset += vl;
        }
    }
#ifdef MICO_RVV_EXPLICIT_CACHE
    mico_rvv_cache_output_end(&cache_output);
#endif
}
#else
/* Alternate W[K,M]: keep each output's packed lane in a K-stream. */
static void mico_rvv_q8xpacked_direct(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
#ifdef MICO_RVV_EXPLICIT_CACHE
    /* Alternate direct dots return scalar reductions and CPU-store O. */
    mico_rvv_cache_inputs_clean();
#endif
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
#ifdef MICO_RVV_EXPLICIT_CACHE
    const mico_rvv_cache_output cache_output = mico_rvv_cache_output_begin(
        output, x->shape[0],
#ifdef USE_ALT_LAYOUT
        w->shape[1]);
#else
        w->shape[0]);
#endif
    output = cache_output.data;
#endif
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

            mico_rvv_store_i32m4(
                output + batch * outputs + output_offset, acc, vl);
            output_offset += vl;
        }
    }
#ifdef MICO_RVV_EXPLICIT_CACHE
    mico_rvv_cache_output_end(&cache_output);
#endif
}

/* Q8 input times packed low-bit weight. */
static void mico_rvv_q8xpacked_matmul(
    int32_t *output, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w,
    unsigned bits) {
#ifdef MICO_RVV_EXPLICIT_CACHE
    mico_rvv_cache_inputs_clean(); /* Scalar output: never invalidate O. */
#endif
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
#ifdef MICO_RVV_EXPLICIT_CACHE
    mico_rvv_cache_inputs_clean(); /* Scalar output: never invalidate O. */
#endif
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
#ifdef MICO_RVV_EXPLICIT_CACHE
    const mico_rvv_cache_output cache_output = mico_rvv_cache_output_begin(
        output, x->shape[0],
#ifdef USE_ALT_LAYOUT
        w->shape[1]);
#else
        w->shape[0]);
#endif
    output = cache_output.data;
#endif
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
                    const vuint8m1_t packed_weight = mico_rvv_load_packed_u8m1(
                        (const uint8_t *)w->data + weight_byte, vl);
                    const vint8m1_t weight_value = mico_rvv_decode_packed(
                        packed_weight, weight_bits, lane, vl);
                    const vint16m2_t weight16 =
                        __riscv_vwcvt_x_x_v_i16m2(weight_value, vl);
                    acc = __riscv_vwmacc_vx_i32m4_tu(
                        acc, activation_value, weight16, vl);
                }

                mico_rvv_store_strided_i32m4(
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
                const vuint8m1_t packed_weight = mico_rvv_load_packed_strided_u8m1(
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

            mico_rvv_store_i32m4(
                output + batch * outputs + output_offset, acc, vl);
            output_offset += vl;
        }
    }
#endif
#ifdef MICO_RVV_EXPLICIT_CACHE
    mico_rvv_cache_output_end(&cache_output);
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

#ifdef MICO_RVV_PROFILE
uint32_t mico_rvv_q8x2_calls;
#endif


#if defined(MICO_RVV_BACKEND_VICUNA2)
/* W2A8 on Vicuna2: expand the current layer's packed weights to INT8 and run
 * the ordinary INT8 kernel. Zve32x has no instruction that consumes 2-bit
 * packed weights, and in-kernel unpacking costs more than the MACs; expanding
 * once per call was measured at 3.38x over the scalar baseline even when the
 * expansion is charged every time. The layer is expanded per call rather than
 * the whole model, matching a loader that materialises one layer at a time. */
#define MICO_RVV_W2_EXPAND_MAX (4u * 1024u * 1024u)
static int8_t mico_rvv_w2_expand_buf[MICO_RVV_W2_EXPAND_MAX] __attribute__((aligned(64)));

#define MICO_W2_DEC1(c) ((uint32_t)(uint8_t)(int8_t)((int)((c) & 1u) - 2 * (int)(((c) >> 1) & 1u)))
#define MICO_W2_LUT_ENTRY(b) ( \
    (MICO_W2_DEC1((b) >> 0) <<  0) | \
    (MICO_W2_DEC1((b) >> 2) <<  8) | \
    (MICO_W2_DEC1((b) >> 4) << 16) | \
    (MICO_W2_DEC1((b) >> 6) << 24) )
#define MICO_W2_LUT_4(base) \
    MICO_W2_LUT_ENTRY((base) + 0), MICO_W2_LUT_ENTRY((base) + 1), \
    MICO_W2_LUT_ENTRY((base) + 2), MICO_W2_LUT_ENTRY((base) + 3)
#define MICO_W2_LUT_16(base) \
    MICO_W2_LUT_4((base) +  0), MICO_W2_LUT_4((base) +  4), \
    MICO_W2_LUT_4((base) +  8), MICO_W2_LUT_4((base) + 12)
#define MICO_W2_LUT_64(base) \
    MICO_W2_LUT_16((base) +  0), MICO_W2_LUT_16((base) + 16), \
    MICO_W2_LUT_16((base) + 32), MICO_W2_LUT_16((base) + 48)

static const uint32_t mico_rvv_w2_expand_lut[256] = {
    MICO_W2_LUT_64(0), MICO_W2_LUT_64(64),
    MICO_W2_LUT_64(128), MICO_W2_LUT_64(192)
};

static bool mico_rvv_w2_expand(int32_t *O, const Tensor2D_Q8 *x,
                               const Tensor2D_Q8 *w) {
    const size_t n = w->shape[0];
    const size_t k = w->shape[1];
    if (n == 0 || k == 0) return false;
    if (n > MICO_RVV_W2_EXPAND_MAX / k) return false;
    const uint8_t *src = (const uint8_t *)w->data;
    const size_t row_bytes = (k + 3u) / 4u;
    const size_t full_groups = k >> 2;
    const size_t tail = k & 3u;
    if (tail == 0) {
        const size_t total_bytes = n * full_groups;
        uint32_t *dst32 = (uint32_t *)mico_rvv_w2_expand_buf;
        size_t b = 0;
        for (; b + 4 <= total_bytes; b += 4) {
            dst32[b + 0] = mico_rvv_w2_expand_lut[src[b + 0]];
            dst32[b + 1] = mico_rvv_w2_expand_lut[src[b + 1]];
            dst32[b + 2] = mico_rvv_w2_expand_lut[src[b + 2]];
            dst32[b + 3] = mico_rvv_w2_expand_lut[src[b + 3]];
        }
        for (; b < total_bytes; ++b) {
            dst32[b] = mico_rvv_w2_expand_lut[src[b]];
        }
    } else {
        for (size_t o = 0; o < n; ++o) {
            const uint8_t *row = src + o * row_bytes;
            int8_t *dst = mico_rvv_w2_expand_buf + o * k;
            size_t g = 0;
            for (; g < full_groups; ++g) {
                const uint32_t word = mico_rvv_w2_expand_lut[row[g]];
                __builtin_memcpy(dst + (g << 2), &word, 4);
            }
            const uint8_t last = row[full_groups];
            for (size_t s = 0; s < tail; ++s) {
                const unsigned code = (unsigned)((last >> (s * 2u)) & 3u);
                dst[(full_groups << 2) + s] =
                    (int8_t)((int)(code & 1u) - 2 * (int)(code >> 1));
            }
        }
    }
#if defined(__riscv_zicbom) && !defined(MICO_RVV_VICUNA_L1_CBO_HANDOFF)
    /* Only flush here when mico_rvv_q8_matmul_impl does not already perform
     * the L1 CBO handoff on w->data. */
    {
        const size_t need = n * k;
        const uintptr_t first = (uintptr_t)mico_rvv_w2_expand_buf & ~(uintptr_t)63u;
        const uintptr_t end = ((uintptr_t)mico_rvv_w2_expand_buf + need + 63u)
                              & ~(uintptr_t)63u;
        __asm__ volatile("fence rw, rw" ::: "memory");
        for (uintptr_t line = first; line < end; line += 64u)
            __asm__ volatile("cbo.flush (%0)" :: "r"(line) : "memory");
        __asm__ volatile("fence iorw, iorw" ::: "memory");
    }
#endif
    Tensor2D_Q8 w8 = {.shape = {n, k}, .data = (qbyte *)mico_rvv_w2_expand_buf,
                      .scale = w->scale, .wq = 8};
    /* Uncounted entry: the W2A8 call must not inflate the Q8 call counter. */
    extern void mico_rvv_q8_matmul_impl(int32_t *, const Tensor2D_Q8 *,
                                        const Tensor2D_Q8 *);
    mico_rvv_q8_matmul_impl(O, x, &w8);
    return true;
}
#endif

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                      const Tensor2D_Q8 *w) {
#ifdef MICO_RVV_PROFILE
    ++mico_rvv_q8x2_calls;
#endif
#if defined(MICO_RVV_BACKEND_VICUNA2)
    if (mico_rvv_w2_expand(O, x, w)) return;
#endif
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
