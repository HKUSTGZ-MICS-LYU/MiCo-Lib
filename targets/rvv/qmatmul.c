#include "mico_qnn.h"

#include "mico_rvv_common.h"

/*
 * Accumulate an INT8 dot product in INT32 lanes. The widening multiply
 * operates on sign-extended INT16 vectors and produces an INT32 vector.
 */
static inline int32_t mico_rvv_q8_dot_contiguous(
    const int8_t *a, const int8_t *b, size_t length) {
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax32);

    for (size_t offset = 0; offset < length;) {
        const size_t vl = __riscv_vsetvl_e8m1(length - offset);
        const vint8m1_t va = __riscv_vle8_v_i8m1(a + offset, vl);
        const vint8m1_t vb = __riscv_vle8_v_i8m1(b + offset, vl);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, vl);
        const vint16m2_t vb16 = __riscv_vwcvt_x_x_v_i16m2(vb, vl);

        vacc = __riscv_vwmacc_vv_i32m4_tu(vacc, va16, vb16, vl);
        offset += vl;
    }

    return mico_rvv_reduce_i32m4(vacc, vlmax32);
}

/* Same dot product for the existing K-by-M alternate weight layout. */
static inline int32_t mico_rvv_q8_dot_strided(
    const int8_t *a, const int8_t *b, size_t length, ptrdiff_t stride) {
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax32);

    for (size_t offset = 0; offset < length;) {
        const size_t vl = __riscv_vsetvl_e8m1(length - offset);
        const vint8m1_t va = __riscv_vle8_v_i8m1(a + offset, vl);
        const vint8m1_t vb =
            __riscv_vlse8_v_i8m1(b + offset * stride, stride, vl);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, vl);
        const vint16m2_t vb16 = __riscv_vwcvt_x_x_v_i16m2(vb, vl);

        vacc = __riscv_vwmacc_vv_i32m4_tu(vacc, va16, vb16, vl);
        offset += vl;
    }

    return mico_rvv_reduce_i32m4(vacc, vlmax32);
}

#ifdef MICO_RVV_PROFILE
uint32_t mico_rvv_q8_calls;
#endif

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x,
                    const Tensor2D_Q8 *w) {
#ifdef MICO_RVV_PROFILE
    ++mico_rvv_q8_calls;
#endif
#ifdef MICO_RVV_EXPLICIT_CACHE
    /* Scalar output stays CPU-owned; do not invalidate it. */
    mico_rvv_cache_inputs_clean();
#endif
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
#ifdef USE_ALT_LAYOUT
    const size_t out_features = w->shape[1];
#else
    const size_t out_features = w->shape[0];
#endif

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const int8_t *x_row = x->data + batch * in_features;
        for (size_t output = 0; output < out_features; ++output) {
#ifdef USE_ALT_LAYOUT
            const int8_t *w_column = w->data + output;
            O[batch * out_features + output] =
                mico_rvv_q8_dot_strided(x_row, w_column, in_features,
                                        (ptrdiff_t)out_features);
#else
            const int8_t *w_row = w->data + output * in_features;
            O[batch * out_features + output] =
                mico_rvv_q8_dot_contiguous(x_row, w_row, in_features);
#endif
        }
    }
}
