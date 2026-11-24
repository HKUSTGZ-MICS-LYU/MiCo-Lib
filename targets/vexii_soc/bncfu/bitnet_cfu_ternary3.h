#ifndef BNCFU_BITNET_CFU_TERNARY3_H
#define BNCFU_BITNET_CFU_TERNARY3_H

/* Share status codes and the common power-of-two VLEN validation. */
#include "bitnet_cfu_radix4.h"

#define BNCFU_TERNARY3_LANES BNCFU_RADIX4_LANES

#ifdef __cplusplus
extern "C" {
#endif

/* bncfu-ternary3-v1: logical W[N,K], bits=8 or 4, P=6 or 3, B=VLEN/8.
 * Each weight is exactly sum(d[p] * 3^p), with d[p] in {-1,0,+1}.
 * Digits use two-bit codes 3/0/1 respectively; code 2 is unused.
 * tiles=ceil(K/B), slots=tiles*P, row_vectors=ceil(slots/4).
 * Layout: [N][row_vectors][4 plane slots][B/4 bytes]. For tile t and
 * plane p, s=t*P+p selects vector s/4 and slot s%4. Within each slot,
 * bytes contain four digit codes, lowest lane first. Final partial-K
 * lanes and unused slots in the final vector must all be zero.
 * Returns N*row_vectors*B, or 0 for empty dimensions, invalid bits, or
 * overflow (including intermediate tile/slot/vector size products).
 * There is no C weight packer; callers must supply offline-packed data.
 */
size_t bncfu_ternary3_packed_size(size_t n, size_t k, unsigned bits);

/* Y[M,N] = X[M,K] * W[N,K]^T, with contiguous signed INT8 X and exact
 * prepacked signed W8/W4 weights. Requires matching hardware VLEN and at
 * least the compiled BNCFU_REG_DEPTH physical banks (2..8). Q1.58/1.5b
 * hardware suffices: true Q2 support is not required. Uses A bank0 and
 * all W banks1+, and leaves qtype=BNCFU_QTYPE_15B after nonempty compute.
 * packed must be B-byte aligned and its complete packed_size bytes must
 * be readable, including zero padding. packed_bytes below that size
 * returns ENOSPC; invalid arguments or size overflow return EINVAL.
 * Unaligned X and partial-K tiles use aligned, zero-filled stack scratch.
 * CPU plane accumulation and reconstruction use INT32 in <=32768-lane
 * chunks. Long K uses a 32-bit low word plus signed wrap count between
 * chunks (no int64 arithmetic), preserving cancellation before the final
 * INT32 range check. ERANGE means a final result does not fit INT32, and
 * earlier outputs may have been written. No saturation is performed.
 * Empty M/N is a no-op; K=0 writes zeros, both without touching hardware.
 * bits must be 4 or 8 even for empty inputs. K must be <= INT32_MAX for
 * nonempty M/N. Buffers must not overlap; exclusive CFU access is required.
 * No allocation, activation correction scan, or runtime weight recoding.
 */
BncfuRadix4Status bncfu_ternary3_gemm(
    int32_t *output, const int8_t *activation,
    const uint8_t *packed, size_t packed_bytes,
    size_t m, size_t n, size_t k, unsigned bits);

BncfuRadix4Status bncfu_ternary3_gemv(
    int32_t *output, const int8_t *activation,
    const uint8_t *packed, size_t packed_bytes,
    size_t n, size_t k, unsigned bits);

#ifdef __cplusplus
}
#endif
#endif
