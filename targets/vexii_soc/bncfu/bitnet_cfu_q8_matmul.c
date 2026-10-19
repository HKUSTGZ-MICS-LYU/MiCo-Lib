#include "mico_qnn.h"
#if defined(BNCFU_WITH_Q2) && BNCFU_WITH_Q2
#include "bitnet_cfu_radix4.h"
#define BNCFU_Q8_PACKED_SIZE bncfu_radix4_packed_size
#define BNCFU_Q8_GEMM bncfu_radix4_gemm
#else
#include "bitnet_cfu_ternary3.h"
#define BNCFU_Q8_PACKED_SIZE bncfu_ternary3_packed_size
#define BNCFU_Q8_GEMM bncfu_ternary3_gemm
#endif

/* BNCFU-only PREPACKED Q8 contract (generic MiCo targets use raw INT8 W):
 * x->data is ordinary contiguous INT8 X[M,K]; w->shape is logical [N,K].
 * w->data MUST be Python weight_recode.py output, bits=8, matching VLEN:
 *   BNCFU_WITH_Q2=1: --qtype 2b, bncfu-radix4-v1, four base-4 planes.
 *   BNCFU_WITH_Q2=0: --qtype 1.5b, bncfu-ternary3-v1, six balanced base-3
 *                   planes, requiring only weights {-1,0,+1}, no -2 path.
 * B=VLEN/8; the loader must provide B-byte alignment and the full padded
 * byte length reported by the corresponding Python metadata/size query.
 * Tensor shape/scale/wq do NOT identify layout, VLEN or actual byte capacity.
 * The two packed formats are NOT interchangeable. No runtime format probing.
 *
 * We assume offline Python preparation: NO C weight packing, heap allocation,
 * weight copying, pointer cache or raw-weight scalar fallback. Reuse read-only
 * packed weights across calls; repack externally when weights change. Never
 * pass ordinary raw INT8 W to this override. Scale/bias/requantization remain
 * the caller's responsibility. Activation tails may use aligned stack scratch.
 */
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    if (!x || !w) {
        MiCo_assert(0, "[BNCFU Q8 MatMul] null tensor");
        return;
    }
    const size_t m = x->shape[0], k = x->shape[1], n = w->shape[0];
    if (w->shape[1] != k) {
        MiCo_assert(0, "[BNCFU Q8 MatMul] expected logical packed weight shape [N,K]");
        return;
    }
    if (!m || !n) return;
#ifdef USE_ALT_LAYOUT
    if (k) {
        MiCo_assert(0, "[BNCFU Q8 MatMul] USE_ALT_LAYOUT is incompatible with Python prepacked [N,K] weights");
        return;
    }
#endif
    const size_t bytes = BNCFU_Q8_PACKED_SIZE(n, k, 8);
    if (k && !bytes) {
        MiCo_assert(0, "[BNCFU Q8 MatMul] overflowing packed weight dimensions");
        return;
    }
    const BncfuRadix4Status status = BNCFU_Q8_GEMM(
        O, x->data, (const uint8_t *)w->data, bytes, m, n, k, 8);
    MiCo_assert(status == BNCFU_RADIX4_OK,
        "[BNCFU Q8 MatMul] invalid dimensions/alignment/data or output exceeds INT32 range");
}
