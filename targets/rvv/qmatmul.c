#include "mico_qnn.h"
#include "mico_rvv_common.h"
#include "mico_rvv_vicuna_cache.h"

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

/* Keep legacy Vexii callers conservative even outside vexii_soc.mk. Generic
 * Spike/native targets retain their previous intrinsic-only load behavior.
 * The Vicuna2 runner intentionally keeps this capability enabled until a
 * mixed-EEW correctness gate exists; backend identity alone cannot disable it. */
#ifndef MICO_RVV_Q8_EEW_EQ_SEW
#if defined(RISCV_VEXII) && defined(MICO_RVV)
#define MICO_RVV_Q8_EEW_EQ_SEW 1
#else
#define MICO_RVV_Q8_EEW_EQ_SEW 0
#endif
#endif
#if MICO_RVV_Q8_EEW_EQ_SEW != 0 && MICO_RVV_Q8_EEW_EQ_SEW != 1
#error "MICO_RVV_Q8_EEW_EQ_SEW must be 0 or 1"
#endif

/* Always inlined for the load helper: keeping it inline avoids outlined-call
   shuffles around the batched accumulators. */
static inline __attribute__((always_inline)) vint8m1_t mico_rvv_q8_load_b(
    const int8_t *p, ptrdiff_t stride, size_t vl) {
#if MICO_RVV_Q8_EEW_EQ_SEW
    /* Conservatively require EEW == SEW. Restore compiler entry state:
     * pinning e8 alone can leave the following widening MAC in the wrong mode. */
    vint8m1_t value;
    size_t saved_vl, saved_vtype;
    if (stride == 1) {
        __asm__ volatile("csrr %[oldvl], vl\n\t"
                         "csrr %[oldtype], vtype\n\t"
                         "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                         "vle8.v %[value], (%[p])\n\t"
                         "vsetvl zero, %[oldvl], %[oldtype]"
                         : [value] "=vr"(value), [oldvl] "=&r"(saved_vl),
                           [oldtype] "=&r"(saved_vtype)
                         : [vl] "r"(vl), [p] "r"(p)
                         : "memory");
    } else {
        __asm__ volatile("csrr %[oldvl], vl\n\t"
                         "csrr %[oldtype], vtype\n\t"
                         "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
                         "vlse8.v %[value], (%[p]), %[stride]\n\t"
                         "vsetvl zero, %[oldvl], %[oldtype]"
                         : [value] "=vr"(value), [oldvl] "=&r"(saved_vl),
                           [oldtype] "=&r"(saved_vtype)
                         : [vl] "r"(vl), [p] "r"(p), [stride] "r"(stride)
                         : "memory");
    }
    return value;
#else
    return stride == 1 ? __riscv_vle8_v_i8m1(p, vl)
                       : __riscv_vlse8_v_i8m1(p, stride, vl);
#endif
}


/* Vicuna conservative fast path: five load/widen pairs under one VL/VTYPE save. */
#ifndef MICO_RVV_Q8_GROUP_LOAD
/* Enabled only for the Vicuna2 RVV backend on the conservative EEW==SEW,
 * standard-layout path. Coral builds with MICO_RVV_BACKEND_CORAL, so its
 * kernel stays byte-for-byte identical to the original. */
#if defined(MICO_RVV_BACKEND_VICUNA2) && MICO_RVV_Q8_EEW_EQ_SEW && !defined(USE_ALT_LAYOUT)
#define MICO_RVV_Q8_GROUP_LOAD 1
#else
#define MICO_RVV_Q8_GROUP_LOAD 0
#endif
#endif
#if MICO_RVV_Q8_GROUP_LOAD != 0 && MICO_RVV_Q8_GROUP_LOAD != 1
#error "MICO_RVV_Q8_GROUP_LOAD must be 0 or 1"
#endif
#if MICO_RVV_Q8_GROUP_LOAD && MICO_RVV_Q8_EEW_EQ_SEW && !defined(USE_ALT_LAYOUT)
/* All five outputs are real i16/m2 values. The i8/m1 temporary is explicitly
 * clobbered; early-clobber outputs cannot alias any other live register input.
 * Save/restore opaque entry VL/VTYPE exactly; do not assume compiler state.
 * A copied, non-x0 AVL register also makes requested AVL=0 unambiguous. */
#define Q8_LOAD5_GROUPED(va16,vb0_16,vb1_16,vb2_16,vb3_16,p_a,p_b0,p_b1,p_b2,p_b3,element_count) do { \
    size_t private_avl; \
    __asm__ volatile( \
        "# private_q8_group_load_begin\n\t" \
        "mv %[avl], %[n]\n\t" \
        "vsetvli zero, %[avl], e8, m1, ta, ma\n\t" \
        "vle8.v v29, (%[pa])\n\t" \
        "vle8.v v30, (%[pb0])\n\t" \
        "vle8.v v31, (%[pb1])\n\t" \
        "vwcvt.x.x.v %[va], v29\n\t" \
        "vwcvt.x.x.v %[vb0], v30\n\t" \
        "vwcvt.x.x.v %[vb1], v31\n\t" \
        "vle8.v v29, (%[pb2])\n\t" \
        "vle8.v v30, (%[pb3])\n\t" \
        "vwcvt.x.x.v %[vb2], v29\n\t" \
        "vwcvt.x.x.v %[vb3], v30\n\t" \
        "# private_q8_group_load_end" \
        : [va] "=&vr"(va16), [vb0] "=&vr"(vb0_16), \
          [vb1] "=&vr"(vb1_16), [vb2] "=&vr"(vb2_16), \
          [vb3] "=&vr"(vb3_16), [avl] "=&r"(private_avl) \
        : [pa] "r"(p_a), [pb0] "r"(p_b0), [pb1] "r"(p_b1), \
          [pb2] "r"(p_b2), [pb3] "r"(p_b3), [n] "r"(element_count) \
        : "v29", "v30", "v31", "memory"); \
} while (0)

/* 2x3 2D register-tiled kernel for batch_size >= 2 and count >= 3:
 * Computes 2 activation rows (a0, a1) against 3 weight rows (b0, b1, b2) = 6
 * outputs in a single K pass. All 32 vector registers are utilized with zero
 * spills:
 *   v0..v3, v4..v7, v8..v11     : row 0 accumulators (acc00, acc01, acc02)
 *   v12..v15, v16..v19, v20..v23: row 1 accumulators (acc10, acc11, acc12)
 *   v24..v25, v26..v27          : widened activations va0_16, va1_16 (i16m2)
 *   v28..v29, v30..v31          : widened weights & i8m1 load temporaries
 */
static __attribute__((noinline)) void mico_rvv_q8_dot_2x4(
    int32_t *out0, int32_t *out1,
    const int8_t *a0, const int8_t *a1,
    const int8_t *b0, const int8_t *b1, const int8_t *b2, const int8_t *b3,
    size_t length) {
    size_t rem = length;
    size_t vl;
    int32_t r00, r01, r02, r03, r10, r11, r12, r13;
    __asm__ volatile(
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.v.i v0, 0\n\t"
        "vmv.v.i v4, 0\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v12, 0\n\t"
        "vmv.v.i v16, 0\n\t"
        "vmv.v.i v20, 0\n\t"
        "1:\n\t"
        "vsetvli %[vl], %[rem], e8, m1, ta, ma\n\t"
        "vle8.v v28, (%[pa0])\n\t"
        "vle8.v v29, (%[pa1])\n\t"
        "vle8.v v30, (%[pb0])\n\t"
        "vle8.v v31, (%[pb1])\n\t"
        "vwcvt.x.x.v v24, v28\n\t"
        "vwcvt.x.x.v v26, v29\n\t"
        "vwcvt.x.x.v v28, v30\n\t"
        "vwcvt.x.x.v v30, v31\n\t"
        "vsetvli zero, %[vl], e16, m2, tu, ma\n\t"
        "vwmacc.vv v0, v24, v28\n\t"
        "vwmacc.vv v12, v26, v28\n\t"
        "vwmacc.vv v4, v24, v30\n\t"
        "vwmacc.vv v16, v26, v30\n\t"
        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
        "vle8.v v30, (%[pb2])\n\t"
        "add %[pa0], %[pa0], %[vl]\n\t"
        "add %[pa1], %[pa1], %[vl]\n\t"
        "add %[pb0], %[pb0], %[vl]\n\t"
        "add %[pb1], %[pb1], %[vl]\n\t"
        "add %[pb2], %[pb2], %[vl]\n\t"
        "sub %[rem], %[rem], %[vl]\n\t"
        "vwcvt.x.x.v v28, v30\n\t"
        "vsetvli zero, %[vl], e16, m2, tu, ma\n\t"
        "vwmacc.vv v8, v24, v28\n\t"
        "vwmacc.vv v20, v26, v28\n\t"
        "bnez %[rem], 1b\n\t"
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.s.x v28, zero\n\t"
        "vredsum.vs v0, v0, v28\n\t"
        "vredsum.vs v4, v4, v28\n\t"
        "vredsum.vs v8, v8, v28\n\t"
        "vredsum.vs v12, v12, v28\n\t"
        "vredsum.vs v16, v16, v28\n\t"
        "vredsum.vs v20, v20, v28\n\t"
        "vmv.x.s %[r00], v0\n\t"
        "vmv.x.s %[r01], v4\n\t"
        "vmv.x.s %[r02], v8\n\t"
        "vmv.x.s %[r10], v12\n\t"
        "vmv.x.s %[r11], v16\n\t"
        "vmv.x.s %[r12], v20\n\t"
        : [pa0] "+r"(a0), [pa1] "+r"(a1),
          [pb0] "+r"(b0), [pb1] "+r"(b1), [pb2] "+r"(b2),
          [rem] "+r"(rem), [vl] "=&r"(vl),
          [r00] "=r"(r00), [r01] "=r"(r01), [r02] "=r"(r02),
          [r10] "=r"(r10), [r11] "=r"(r11), [r12] "=r"(r12)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "cc", "memory");
    out0[0] = r00;
    out0[1] = r01;
    out0[2] = r02;
    out1[0] = r10;
    out1[1] = r11;
    out1[2] = r12;
    (void)b3;
    (void)r03;
    (void)r13;
}

/* 1x4 register-tiled kernel in one self-contained asm loop (no dynamic stack
 * alignment or per-iteration csrr vlenb / stack pointer reloads):
 *   v0..v3, v4..v7, v8..v11, v12..v15: 4 output accumulators (i32m4)
 *   v16..v17                         : widened activation va16 (i16m2)
 *   v20..v21, v22..v23, v24..v25, v26..v27: widened weights vb0..vb3 (i16m2)
 *   v28..v31                         : i8m1 load temporaries
 */
static __attribute__((noinline)) void mico_rvv_q8_dot_1x4(
    int32_t *out, const int8_t *a,
    const int8_t *b0, const int8_t *b1, const int8_t *b2, const int8_t *b3,
    size_t length) {
    size_t rem = length;
    size_t vl;
    int32_t r0, r1, r2, r3;
    __asm__ volatile(
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.v.i v0, 0\n\t"
        "vmv.v.i v4, 0\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v12, 0\n\t"
        "1:\n\t"
        "vsetvli %[vl], %[rem], e8, m1, ta, ma\n\t"
        "vle8.v v29, (%[pa])\n\t"
        "vle8.v v30, (%[pb0])\n\t"
        "vle8.v v31, (%[pb1])\n\t"
        "vwcvt.x.x.v v16, v29\n\t"
        "vwcvt.x.x.v v20, v30\n\t"
        "vwcvt.x.x.v v22, v31\n\t"
        "vle8.v v29, (%[pb2])\n\t"
        "vle8.v v30, (%[pb3])\n\t"
        "add %[pa], %[pa], %[vl]\n\t"
        "add %[pb0], %[pb0], %[vl]\n\t"
        "add %[pb1], %[pb1], %[vl]\n\t"
        "add %[pb2], %[pb2], %[vl]\n\t"
        "add %[pb3], %[pb3], %[vl]\n\t"
        "sub %[rem], %[rem], %[vl]\n\t"
        "vwcvt.x.x.v v24, v29\n\t"
        "vwcvt.x.x.v v26, v30\n\t"
        "vsetvli zero, %[vl], e16, m2, tu, ma\n\t"
        "vwmacc.vv v0, v16, v20\n\t"
        "vwmacc.vv v4, v16, v22\n\t"
        "vwmacc.vv v8, v16, v24\n\t"
        "vwmacc.vv v12, v16, v26\n\t"
        "bnez %[rem], 1b\n\t"
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.s.x v28, zero\n\t"
        "vredsum.vs v0, v0, v28\n\t"
        "vredsum.vs v4, v4, v28\n\t"
        "vredsum.vs v8, v8, v28\n\t"
        "vredsum.vs v12, v12, v28\n\t"
        "vmv.x.s %[r0], v0\n\t"
        "vmv.x.s %[r1], v4\n\t"
        "vmv.x.s %[r2], v8\n\t"
        "vmv.x.s %[r3], v12\n\t"
        : [pa] "+r"(a), [pb0] "+r"(b0), [pb1] "+r"(b1),
          [pb2] "+r"(b2), [pb3] "+r"(b3),
          [rem] "+r"(rem), [vl] "=&r"(vl),
          [r0] "=r"(r0), [r1] "=r"(r1), [r2] "=r"(r2), [r3] "=r"(r3)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "cc", "memory");
    out[0] = r0;
    out[1] = r1;
    out[2] = r2;
    out[3] = r3;
}

/* 4x2 2D register-tiled kernel for batch_size >= 4 and count >= 2:
 * Computes 4 activation rows (a0..a3) against 2 weight rows (b0, b1) = 8
 * outputs in a single K pass!
 *   8 accumulators of i32m4? No, 8 * 4 = 32 registers, no room for operands.
 *   What about 3x2 (3 activation rows x 2 weight rows = 6 outputs)?
 *   Or 2x3 (2 activation rows x 3 weight rows = 6 outputs, above) + 2x2 (2
 *   activation rows x 2 weight rows = 4 outputs):
 *   With 2x2: 2 activation loads + 2 weight loads = 4 loads for 4 outputs (1.0
 *   load/output, vs 1.25 load/output in 1x4)! And 2x3 gives 5 loads for 6
 *   outputs (0.833 load/output)!
 *   Even better: 3x2 (3 activation rows a0,a1,a2 x 2 weight rows b0,b1 = 6
 *   outputs) also does 5 loads for 6 outputs, and since N is a power of two
 *   (e.g. N=8, 16, 256), stepping N by 2 divides N=8/16/256 evenly!
 *   And for M=8 (or M=128), M=8 = 3 + 3 + 2!
 */
static __attribute__((noinline)) void mico_rvv_q8_dot_3x2(
    int32_t *out0, int32_t *out1, int32_t *out2,
    const int8_t *a0, const int8_t *a1, const int8_t *a2,
    const int8_t *b0, const int8_t *b1,
    size_t length) {
    size_t rem = length;
    size_t vl;
    int32_t r00, r01, r10, r11, r20, r21;
    __asm__ volatile(
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.v.i v0, 0\n\t"
        "vmv.v.i v4, 0\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v12, 0\n\t"
        "vmv.v.i v16, 0\n\t"
        "vmv.v.i v20, 0\n\t"
        "1:\n\t"
        "vsetvli %[vl], %[rem], e8, m1, ta, ma\n\t"
        "vle8.v v28, (%[pb0])\n\t"
        "vle8.v v29, (%[pb1])\n\t"
        "vle8.v v30, (%[pa0])\n\t"
        "vle8.v v31, (%[pa1])\n\t"
        "vwcvt.x.x.v v24, v28\n\t"
        "vwcvt.x.x.v v26, v29\n\t"
        "vwcvt.x.x.v v28, v30\n\t"
        "vwcvt.x.x.v v30, v31\n\t"
        "vsetvli zero, %[vl], e16, m2, tu, ma\n\t"
        "vwmacc.vv v0, v28, v24\n\t"
        "vwmacc.vv v4, v28, v26\n\t"
        "vwmacc.vv v8, v30, v24\n\t"
        "vwmacc.vv v12, v30, v26\n\t"
        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
        "vle8.v v30, (%[pa2])\n\t"
        "add %[pb0], %[pb0], %[vl]\n\t"
        "add %[pb1], %[pb1], %[vl]\n\t"
        "add %[pa0], %[pa0], %[vl]\n\t"
        "add %[pa1], %[pa1], %[vl]\n\t"
        "add %[pa2], %[pa2], %[vl]\n\t"
        "sub %[rem], %[rem], %[vl]\n\t"
        "vwcvt.x.x.v v28, v30\n\t"
        "vsetvli zero, %[vl], e16, m2, tu, ma\n\t"
        "vwmacc.vv v16, v28, v24\n\t"
        "vwmacc.vv v20, v28, v26\n\t"
        "bnez %[rem], 1b\n\t"
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.s.x v28, zero\n\t"
        "vredsum.vs v0, v0, v28\n\t"
        "vredsum.vs v4, v4, v28\n\t"
        "vredsum.vs v8, v8, v28\n\t"
        "vredsum.vs v12, v12, v28\n\t"
        "vredsum.vs v16, v16, v28\n\t"
        "vredsum.vs v20, v20, v28\n\t"
        "vmv.x.s %[r00], v0\n\t"
        "vmv.x.s %[r01], v4\n\t"
        "vmv.x.s %[r10], v8\n\t"
        "vmv.x.s %[r11], v12\n\t"
        "vmv.x.s %[r20], v16\n\t"
        "vmv.x.s %[r21], v20\n\t"
        : [pa0] "+r"(a0), [pa1] "+r"(a1), [pa2] "+r"(a2),
          [pb0] "+r"(b0), [pb1] "+r"(b1),
          [rem] "+r"(rem), [vl] "=&r"(vl),
          [r00] "=r"(r00), [r01] "=r"(r01),
          [r10] "=r"(r10), [r11] "=r"(r11),
          [r20] "=r"(r20), [r21] "=r"(r21)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "cc", "memory");
    out0[0] = r00;
    out0[1] = r01;
    out1[0] = r10;
    out1[1] = r11;
    out2[0] = r20;
    out2[1] = r21;
}

static __attribute__((noinline)) void mico_rvv_q8_dot_2x2(
    int32_t *out0, int32_t *out1,
    const int8_t *a0, const int8_t *a1,
    const int8_t *b0, const int8_t *b1,
    size_t length) {
    size_t rem = length;
    size_t vl;
    int32_t r00, r01, r10, r11;
    __asm__ volatile(
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.v.i v0, 0\n\t"
        "vmv.v.i v4, 0\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v12, 0\n\t"
        "1:\n\t"
        "vsetvli %[vl], %[rem], e8, m1, ta, ma\n\t"
        "vle8.v v28, (%[pa0])\n\t"
        "vle8.v v29, (%[pa1])\n\t"
        "vle8.v v30, (%[pb0])\n\t"
        "vle8.v v31, (%[pb1])\n\t"
        "add %[pa0], %[pa0], %[vl]\n\t"
        "add %[pa1], %[pa1], %[vl]\n\t"
        "add %[pb0], %[pb0], %[vl]\n\t"
        "add %[pb1], %[pb1], %[vl]\n\t"
        "sub %[rem], %[rem], %[vl]\n\t"
        "vwcvt.x.x.v v16, v28\n\t"
        "vwcvt.x.x.v v18, v29\n\t"
        "vwcvt.x.x.v v20, v30\n\t"
        "vwcvt.x.x.v v22, v31\n\t"
        "vsetvli zero, %[vl], e16, m2, tu, ma\n\t"
        "vwmacc.vv v0, v16, v20\n\t"
        "vwmacc.vv v4, v16, v22\n\t"
        "vwmacc.vv v8, v18, v20\n\t"
        "vwmacc.vv v12, v18, v22\n\t"
        "bnez %[rem], 1b\n\t"
        "vsetvli %[vl], zero, e32, m4, ta, ma\n\t"
        "vmv.s.x v28, zero\n\t"
        "vredsum.vs v0, v0, v28\n\t"
        "vredsum.vs v4, v4, v28\n\t"
        "vredsum.vs v8, v8, v28\n\t"
        "vredsum.vs v12, v12, v28\n\t"
        "vmv.x.s %[r00], v0\n\t"
        "vmv.x.s %[r01], v4\n\t"
        "vmv.x.s %[r10], v8\n\t"
        "vmv.x.s %[r11], v12\n\t"
        : [pa0] "+r"(a0), [pa1] "+r"(a1),
          [pb0] "+r"(b0), [pb1] "+r"(b1),
          [rem] "+r"(rem), [vl] "=&r"(vl),
          [r00] "=r"(r00), [r01] "=r"(r01),
          [r10] "=r"(r10), [r11] "=r"(r11)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v28", "v29", "v30", "v31",
          "cc", "memory");
    out0[0] = r00;
    out0[1] = r01;
    out1[0] = r10;
    out1[1] = r11;
}
#endif

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
        const vint8m1_t va = mico_rvv_q8_load_b(a + offset, 1, vl);
        const vint16m2_t va16 = __riscv_vwcvt_x_x_v_i16m2(va, vl);
        const vint8m1_t vb = mico_rvv_q8_load_b(b + offset * b_stride, b_stride, vl);
        const vint16m2_t vb16 = __riscv_vwcvt_x_x_v_i16m2(vb, vl);
        vacc = __riscv_vwmacc_vv_i32m4_tu(vacc, va16, vb16, vl);
        offset += vl;
    }
    if (offset < length) {
        const size_t tail = __riscv_vsetvl_e8m1(length - offset);
        const vint8m1_t va = mico_rvv_q8_load_b(a + offset, 1, tail);
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
#if MICO_RVV_Q8_GROUP_LOAD && MICO_RVV_Q8_EEW_EQ_SEW && !defined(USE_ALT_LAYOUT)
    if (count == 4 && b_stride == 1) {
        while (vl != 0 && offset + vl <= length) {
        vint16m2_t va16, vb0_16, vb1_16, vb2_16, vb3_16;
        Q8_LOAD5_GROUPED(va16, vb0_16, vb1_16, vb2_16, vb3_16,
                         a + offset, b_rows[0] + offset, b_rows[1] + offset,
                         b_rows[2] + offset, b_rows[3] + offset, vl);
        vacc0 = __riscv_vwmacc_vv_i32m4_tu(vacc0, va16, vb0_16, vl);
        vacc1 = __riscv_vwmacc_vv_i32m4_tu(vacc1, va16, vb1_16, vl);
        vacc2 = __riscv_vwmacc_vv_i32m4_tu(vacc2, va16, vb2_16, vl);
        vacc3 = __riscv_vwmacc_vv_i32m4_tu(vacc3, va16, vb3_16, vl);
            offset += vl;
        }
        if (offset < length) {
            const size_t tail = __riscv_vsetvl_e8m1(length - offset);
        vint16m2_t va16, vb0_16, vb1_16, vb2_16, vb3_16;
        Q8_LOAD5_GROUPED(va16, vb0_16, vb1_16, vb2_16, vb3_16,
                         a + offset, b_rows[0] + offset, b_rows[1] + offset,
                         b_rows[2] + offset, b_rows[3] + offset, tail);
        vacc0 = __riscv_vwmacc_vv_i32m4_tu(vacc0, va16, vb0_16, tail);
        vacc1 = __riscv_vwmacc_vv_i32m4_tu(vacc1, va16, vb1_16, tail);
        vacc2 = __riscv_vwmacc_vv_i32m4_tu(vacc2, va16, vb2_16, tail);
        vacc3 = __riscv_vwmacc_vv_i32m4_tu(vacc3, va16, vb3_16, tail);
        }
    } else
#endif
    {
    while (vl != 0 && offset + vl <= length) {
        const vint8m1_t va = mico_rvv_q8_load_b(a + offset, 1, vl);
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
        const vint8m1_t va = mico_rvv_q8_load_b(a + offset, 1, tail);
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
    }

    results[0] = mico_rvv_reduce_i32m4(vacc0, vlmax32);
    if (count > 1) results[1] = mico_rvv_reduce_i32m4(vacc1, vlmax32);
    if (count > 2) results[2] = mico_rvv_reduce_i32m4(vacc2, vlmax32);
    if (count > 3) results[3] = mico_rvv_reduce_i32m4(vacc3, vlmax32);
}

#ifdef MICO_RVV_PROFILE
uint32_t mico_rvv_q8_calls;
#endif

void mico_rvv_q8_matmul_impl(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
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
#ifdef MICO_RVV_VICUNA_L1_CBO_HANDOFF
    mico_rvv_vicuna_cbo_flush_range(x->data, batch_size * in_features);
    mico_rvv_vicuna_cbo_flush_range(w->data, w->shape[0] * w->shape[1]);
    mico_rvv_vicuna_fence();
#endif
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();

#if MICO_RVV_Q8_GROUP_LOAD && MICO_RVV_Q8_EEW_EQ_SEW && !defined(USE_ALT_LAYOUT)
    if (in_features >= (size_t)MICO_RVV_Q8_BATCH_MIN_LENGTH && out_features >= 2) {
        size_t batch = 0;
        while (batch + 3 <= batch_size) {
            const int8_t *a0 = x->data + (batch + 0) * in_features;
            const int8_t *a1 = x->data + (batch + 1) * in_features;
            const int8_t *a2 = x->data + (batch + 2) * in_features;
            int32_t *o0 = O + (batch + 0) * out_features;
            int32_t *o1 = O + (batch + 1) * out_features;
            int32_t *o2 = O + (batch + 2) * out_features;
            size_t output = 0;
            for (; output + 2 <= out_features; output += 2) {
                const int8_t *b0 = w->data + (output + 0) * in_features;
                const int8_t *b1 = w->data + (output + 1) * in_features;
                mico_rvv_q8_dot_3x2(o0 + output, o1 + output, o2 + output,
                                    a0, a1, a2, b0, b1, in_features);
            }
            for (; output < out_features; ++output) {
                const int8_t *w_row = w->data + output * in_features;
                o0[output] = mico_rvv_q8_dot(a0, w_row, in_features, 1, vlmax32);
                o1[output] = mico_rvv_q8_dot(a1, w_row, in_features, 1, vlmax32);
                o2[output] = mico_rvv_q8_dot(a2, w_row, in_features, 1, vlmax32);
            }
            batch += 3;
        }
        while (batch + 2 <= batch_size) {
            const int8_t *a0 = x->data + (batch + 0) * in_features;
            const int8_t *a1 = x->data + (batch + 1) * in_features;
            int32_t *o0 = O + (batch + 0) * out_features;
            int32_t *o1 = O + (batch + 1) * out_features;
            size_t output = 0;
            for (; output + 3 <= out_features; output += 3) {
                const int8_t *b0 = w->data + (output + 0) * in_features;
                const int8_t *b1 = w->data + (output + 1) * in_features;
                const int8_t *b2 = w->data + (output + 2) * in_features;
                mico_rvv_q8_dot_2x4(o0 + output, o1 + output,
                                    a0, a1, b0, b1, b2, NULL, in_features);
            }
            for (; output + 2 <= out_features; output += 2) {
                const int8_t *b0 = w->data + (output + 0) * in_features;
                const int8_t *b1 = w->data + (output + 1) * in_features;
                mico_rvv_q8_dot_2x2(o0 + output, o1 + output,
                                    a0, a1, b0, b1, in_features);
            }
            for (; output < out_features; ++output) {
                const int8_t *w_row = w->data + output * in_features;
                o0[output] = mico_rvv_q8_dot(a0, w_row, in_features, 1, vlmax32);
                o1[output] = mico_rvv_q8_dot(a1, w_row, in_features, 1, vlmax32);
            }
            batch += 2;
        }
        for (; batch < batch_size; ++batch) {
            const int8_t *x_row = x->data + batch * in_features;
            int32_t *o_row = O + batch * out_features;
            size_t output = 0;
            for (; output + 4 <= out_features; output += 4) {
                mico_rvv_q8_dot_1x4(o_row + output, x_row,
                                    w->data + (output + 0) * in_features,
                                    w->data + (output + 1) * in_features,
                                    w->data + (output + 2) * in_features,
                                    w->data + (output + 3) * in_features,
                                    in_features);
            }
            if (output < out_features) {
                const size_t count = out_features - output;
                const int8_t *b_rows[MICO_RVV_Q8_BATCH];
                int32_t results[MICO_RVV_Q8_BATCH];
                for (size_t i = 0; i < count; ++i) {
                    b_rows[i] = w->data + (output + i) * in_features;
                }
                mico_rvv_q8_dot_batch(results, count, x_row, b_rows, in_features,
                                      b_stride, vlmax32);
                for (size_t i = 0; i < count; ++i) {
                    o_row[output + i] = results[i];
                }
            }
        }
#ifdef MICO_RVV_VICUNA_L1_CBO_HANDOFF
        mico_rvv_vicuna_fence();
#endif
        return;
    }
#endif

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
#ifdef MICO_RVV_VICUNA_L1_CBO_HANDOFF
    mico_rvv_vicuna_fence();
#endif
}

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
#ifdef MICO_RVV_PROFILE
    ++mico_rvv_q8_calls;
#endif
    mico_rvv_q8_matmul_impl(O, x, w);
}
