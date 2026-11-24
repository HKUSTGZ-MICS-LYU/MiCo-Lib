#ifndef BNCFU_BITNET_CFU_RADIX4_H
#define BNCFU_BITNET_CFU_RADIX4_H

#include <stddef.h>
#include <stdint.h>

#ifndef VLEN
#define VLEN 256
#endif
#if VLEN < 32 || VLEN > 512 || (VLEN & (VLEN - 1)) != 0
#error "BNCFU radix4 requires a power-of-two VLEN in [32, 512]"
#endif

#define BNCFU_RADIX4_LANES (VLEN / 8)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BNCFU_RADIX4_OK = 0,
    BNCFU_RADIX4_EINVAL = -1,
    BNCFU_RADIX4_ENOSPC = -2,
    BNCFU_RADIX4_ERANGE = -3
} BncfuRadix4Status;

/* bncfu-radix4-v1: W[N,K], bits=4 or 8, P=bits/2, B=VLEN/8.
 * Layout [N][ceil(K/(B*(4/P)))][4/P tiles][P planes][B/4 bytes].
 * Each byte contains four signed INT2 codes, lowest lane first.
 * Padding represents original weight zero, NOT all-zero digit planes.
 * Returns 0 for empty dimensions, invalid bits, or size overflow.
 */
size_t bncfu_radix4_packed_size(size_t n, size_t k, unsigned bits);

/* No C weight-packing API: use weight_recode.py offline. Both explicit
 * kernels below and the BNCFU MiCo_Q8_MatMul override consume packed data.
 * The override is opt-in: BNCFU_INT8_MATMUL=1 (default 0 uses raw W).
 * With BNCFU_WITH_Q2=1, MiCo_Q8_MatMul keeps logical w->shape=[N,K], but
 * w->data is bits=8 radix4 packed bytes, NOT generic MiCo's raw INT8 layout.
 * BNCFU_WITH_Q2=0 instead selects the distinct ternary3 packed format.
 */

/* Y[M,N] = X[M,K] * W[N,K]^T, with prepacked W and contiguous signed X.
 * Requires hardware --bitnet-cfu-with-q2, matching VLEN and at least the
 * compiled BNCFU_REG_DEPTH physical banks (software supports 2..8).
 * Uses all configured banks, fixed A bank0/W banks1+, and leaves qtype=2b.
 * X may be unaligned: unsafe/full-vector tails use aligned zeroed scratch.
 * CPU plane accumulation, activation sums and reconstruction use INT32
 * in <=32768-lane chunks. A 32-bit low word plus signed wrap count preserves
 * exact long-K cancellation/range checks without int64 arithmetic. The
 * activation correction is still scanned once per row, shared across N.
 * ERANGE means a final result does not fit INT32; earlier outputs may be written.
 * Empty M/N is a no-op; K=0 writes zeros. K must be <= INT32_MAX.
 * Buffers must not overlap; calls require exclusive access to the CFU.
 */
BncfuRadix4Status bncfu_radix4_gemm(
    int32_t *output, const int8_t *activation,
    const uint8_t *packed, size_t packed_bytes,
    size_t m, size_t n, size_t k, unsigned bits);

BncfuRadix4Status bncfu_radix4_gemv(
    int32_t *output, const int8_t *activation,
    const uint8_t *packed, size_t packed_bytes,
    size_t n, size_t k, unsigned bits);

#ifdef __cplusplus
}
#endif
#endif
