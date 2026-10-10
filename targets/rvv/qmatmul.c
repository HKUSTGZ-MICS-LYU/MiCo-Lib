#include "mico_qnn.h"
#include "mico_rvv_common.h"

/* Number of output dots computed in one pass over the shared activation row.
   ISA-generic tuning, identical on every RVV implementation: batching amortises
   the activation-row load, its widening convert, and the chunk-width setup
   (vsetvl) across several outputs instead of paying them per output element. */
#ifndef MICO_RVV_Q8_BATCH
#define MICO_RVV_Q8_BATCH 4
#endif
#if MICO_RVV_Q8_BATCH < 1 || MICO_RVV_Q8_BATCH > 4
#error "MICO_RVV_Q8_BATCH must be between 1 and 4"
#endif

/* Always inlined for the load helper: keeping it inline avoids outlined-call
   shuffles around the batched accumulators. */
static inline __attribute__((always_inline)) vint8m1_t mico_rvv_q8_load_b(
    const int8_t *p, ptrdiff_t stride, size_t vl) {
    return stride == 1 ? __riscv_vle8_v_i8m1(p, vl)
                       : __riscv_vlse8_v_i8m1(p, stride, vl);
}

/* Batching amortises the activation-row load and the chunk-width setup across
   outputs, but pays a fixed per-batch cost (extra accumulators and the batched
   result scatter). Below this row length the plain per-output dot is the
   faster form on any implementation; the threshold is ISA-generic. */
#ifndef MICO_RVV_Q8_BATCH_MIN_LENGTH
#define MICO_RVV_Q8_BATCH_MIN_LENGTH 32
#endif

/* Single-output INT8 dot product: the small-row form, same lane-wise
   accumulation and reduction as the batched form. Always inlined so the
   per-output loop carries no calling-convention shuffles. */
static inline __attribute__((always_inline)) int32_t mico_rvv_q8_dot(
    const int8_t *a, const int8_t *b, size_t length, ptrdiff_t b_stride,
    size_t vlmax32) {
    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax32);
    size_t offset = 0;
    size_t vl = __riscv_vsetvl_e8m1(length);
    while (vl != 0 && offset + vl <= length) {
        const vint8m1_t va = __riscv_vle8_v_i8m1(a + offset, vl);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, vl);
        const vint8m1_t vb = mico_rvv_q8_load_b(b + offset * b_stride, b_stride, vl);
        const vint16m2_t vb16 = __riscv_vwcvt_x_x_v_i16m2(vb, vl);
        vacc = __riscv_vwmacc_vv_i32m4_tu(vacc, va16, vb16, vl);
        offset += vl;
    }
    if (offset < length) {
        const size_t tail = __riscv_vsetvl_e8m1(length - offset);
        const vint8m1_t va = __riscv_vle8_v_i8m1(a + offset, tail);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, tail);
        const vint8m1_t vb = mico_rvv_q8_load_b(b + offset * b_stride, b_stride, tail);
        const vint16m2_t vb16 = __riscv_vwcvt_x_x_v_i16m2(vb, tail);
        vacc = __riscv_vwmacc_vv_i32m4_tu(vacc, va16, vb16, tail);
    }
    return mico_rvv_reduce_i32m4(vacc, vlmax32);
}

/*
 * Batched INT8 dot products over one shared activation row: computes `count`
 * (1..MICO_RVV_Q8_BATCH) dots of `a` against `b_rows[i]` in a single pass,
 * each into its own INT32 lane accumulator. The activation row and its
 * widening convert execute once per chunk for the whole batch, and the chunk
 * width is set once per pass (plus once for the tail) instead of per chunk.
 * The widening multiply operates on sign-extended INT16 vectors and produces
 * an INT32 vector; per-lane accumulation and the final reduction are
 * unchanged, so results are bit-identical to the per-output form.
 */
static inline void mico_rvv_q8_dot_batch(
    int32_t *results, size_t count, const int8_t *a,
    const int8_t *const *b_rows, size_t length, ptrdiff_t b_stride,
    size_t vlmax32) {
    vint32m4_t vacc0 = __riscv_vmv_v_x_i32m4(0, vlmax32);
    vint32m4_t vacc1 = vacc0;
    vint32m4_t vacc2 = vacc0;
    vint32m4_t vacc3 = vacc0;

    size_t offset = 0;
    size_t vl = __riscv_vsetvl_e8m1(length);
    while (vl != 0 && offset + vl <= length) {
        const vint8m1_t va = __riscv_vle8_v_i8m1(a + offset, vl);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, vl);

        const vint8m1_t vb0 = mico_rvv_q8_load_b(b_rows[0] + offset * b_stride, b_stride, vl);
        const vint16m2_t vb0_16 = __riscv_vwcvt_x_x_v_i16m2(vb0, vl);
        vacc0 = __riscv_vwmacc_vv_i32m4_tu(vacc0, va16, vb0_16, vl);
        if (count > 1) {
            const vint8m1_t vb1 = mico_rvv_q8_load_b(b_rows[1] + offset * b_stride, b_stride, vl);
            const vint16m2_t vb1_16 = __riscv_vwcvt_x_x_v_i16m2(vb1, vl);
            vacc1 = __riscv_vwmacc_vv_i32m4_tu(vacc1, va16, vb1_16, vl);
        }
        if (count > 2) {
            const vint8m1_t vb2 = mico_rvv_q8_load_b(b_rows[2] + offset * b_stride, b_stride, vl);
            const vint16m2_t vb2_16 = __riscv_vwcvt_x_x_v_i16m2(vb2, vl);
            vacc2 = __riscv_vwmacc_vv_i32m4_tu(vacc2, va16, vb2_16, vl);
        }
        if (count > 3) {
            const vint8m1_t vb3 = mico_rvv_q8_load_b(b_rows[3] + offset * b_stride, b_stride, vl);
            const vint16m2_t vb3_16 = __riscv_vwcvt_x_x_v_i16m2(vb3, vl);
            vacc3 = __riscv_vwmacc_vv_i32m4_tu(vacc3, va16, vb3_16, vl);
        }
        offset += vl;
    }
    if (offset < length) {
        const size_t tail = __riscv_vsetvl_e8m1(length - offset);
        const vint8m1_t va = __riscv_vle8_v_i8m1(a + offset, tail);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, tail);

        const vint8m1_t vb0 = mico_rvv_q8_load_b(b_rows[0] + offset * b_stride, b_stride, tail);
        const vint16m2_t vb0_16 = __riscv_vwcvt_x_x_v_i16m2(vb0, tail);
        vacc0 = __riscv_vwmacc_vv_i32m4_tu(vacc0, va16, vb0_16, tail);
        if (count > 1) {
            const vint8m1_t vb1 = mico_rvv_q8_load_b(b_rows[1] + offset * b_stride, b_stride, tail);
            const vint16m2_t vb1_16 = __riscv_vwcvt_x_x_v_i16m2(vb1, tail);
            vacc1 = __riscv_vwmacc_vv_i32m4_tu(vacc1, va16, vb1_16, tail);
        }
        if (count > 2) {
            const vint8m1_t vb2 = mico_rvv_q8_load_b(b_rows[2] + offset * b_stride, b_stride, tail);
            const vint16m2_t vb2_16 = __riscv_vwcvt_x_x_v_i16m2(vb2, tail);
            vacc2 = __riscv_vwmacc_vv_i32m4_tu(vacc2, va16, vb2_16, tail);
        }
        if (count > 3) {
            const vint8m1_t vb3 = mico_rvv_q8_load_b(b_rows[3] + offset * b_stride, b_stride, tail);
            const vint16m2_t vb3_16 = __riscv_vwcvt_x_x_v_i16m2(vb3, tail);
            vacc3 = __riscv_vwmacc_vv_i32m4_tu(vacc3, va16, vb3_16, tail);
        }
    }

    results[0] = mico_rvv_reduce_i32m4(vacc0, vlmax32);
    if (count > 1) results[1] = mico_rvv_reduce_i32m4(vacc1, vlmax32);
    if (count > 2) results[2] = mico_rvv_reduce_i32m4(vacc2, vlmax32);
    if (count > 3) results[3] = mico_rvv_reduce_i32m4(vacc3, vlmax32);
}

#ifdef MICO_RVV_PROFILE
uint32_t mico_rvv_q8_calls;
#endif

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
#ifdef MICO_RVV_PROFILE
    ++mico_rvv_q8_calls;
#endif
#ifdef MICO_RVV_EXPLICIT_CACHE
    /* All O writes are scalar; invalidating O would discard dirty CPU data. */
    mico_rvv_cache_inputs_clean();
#endif
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
#ifdef USE_ALT_LAYOUT
    const size_t out_features = w->shape[1];
    const ptrdiff_t b_stride = (ptrdiff_t)out_features;
#else
    const size_t out_features = w->shape[0];
    const ptrdiff_t b_stride = 1;
#endif
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const int8_t *x_row = x->data + batch * in_features;
        if (in_features < (size_t)MICO_RVV_Q8_BATCH_MIN_LENGTH) {
            // Small rows: plain per-output dots, no batching overhead.
            for (size_t output = 0; output < out_features; ++output) {
#ifdef USE_ALT_LAYOUT
                const int8_t *w_column = w->data + output;
                O[batch * out_features + output] =
                    mico_rvv_q8_dot(x_row, w_column, in_features, b_stride,
                                    vlmax32);
#else
                const int8_t *w_row = w->data + output * in_features;
                O[batch * out_features + output] =
                    mico_rvv_q8_dot(x_row, w_row, in_features, b_stride,
                                    vlmax32);
#endif
            }
            continue;
        }
        for (size_t output = 0; output < out_features;
             output += MICO_RVV_Q8_BATCH) {
            const size_t count = (out_features - output < MICO_RVV_Q8_BATCH)
                ? (out_features - output)
                : (size_t)MICO_RVV_Q8_BATCH;
            const int8_t *b_rows[MICO_RVV_Q8_BATCH];
            int32_t results[MICO_RVV_Q8_BATCH];
            for (size_t i = 0; i < count; ++i) {
#ifdef USE_ALT_LAYOUT
                b_rows[i] = w->data + (output + i);
#else
                b_rows[i] = w->data + (output + i) * in_features;
#endif
            }
            mico_rvv_q8_dot_batch(results, count, x_row, b_rows, in_features,
                                  b_stride, vlmax32);
            for (size_t i = 0; i < count; ++i) {
                O[batch * out_features + output + i] = results[i];
            }
        }
    }
}
