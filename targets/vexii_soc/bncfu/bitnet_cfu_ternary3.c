#include "bitnet_cfu_ternary3.h"
#ifdef BNCFU_RADIX4_TEST_BACKEND
#include "bncfu_radix4_test_backend.h"
#else
#include "bitnet_cfu.h"
#endif
#include <limits.h>
#include <string.h>

static int valid_bits(unsigned bits) {
    return bits == 4 || bits == 8;
}

static int product_fits(size_t a, size_t b) {
    return b == 0 || a <= SIZE_MAX / b;
}

size_t bncfu_ternary3_packed_size(size_t n, size_t k, unsigned bits) {
    if (!valid_bits(bits) || n == 0 || k == 0) return 0;
    const size_t b = BNCFU_TERNARY3_LANES;
    const unsigned planes = bits == 8 ? 6 : 3;
    const size_t tiles = k / b + (k % b != 0);
    if (!product_fits(tiles, planes)) return 0;
    const size_t slots = tiles * planes;
    const size_t row_vectors = slots / 4 + (slots % 4 != 0);
    if (!product_fits(row_vectors, b)) return 0;
    const size_t row_bytes = row_vectors * b;
    return product_fits(n, row_bytes) ? n * row_bytes : 0;
}

BncfuRadix4Status bncfu_ternary3_gemm(
        int32_t *output, const int8_t *activation,
        const uint8_t *packed, size_t packed_bytes,
        size_t m, size_t n, size_t k, unsigned bits) {
    if (!valid_bits(bits)) return BNCFU_RADIX4_EINVAL;
    if (m == 0 || n == 0) return BNCFU_RADIX4_OK;
    if (k > INT32_MAX || !product_fits(m, k)
            || !product_fits(m, n) || !product_fits(m * n, sizeof(*output)))
        return BNCFU_RADIX4_EINVAL;
    if (!output) return BNCFU_RADIX4_EINVAL;
    if (k == 0) {
        memset(output, 0, m * n * sizeof(*output));
        return BNCFU_RADIX4_OK;
    }
    const size_t bytes = bncfu_ternary3_packed_size(n, k, bits);
    const size_t b = BNCFU_TERNARY3_LANES;
    if (!bytes || !activation || !packed || (uintptr_t)packed % b != 0)
        return BNCFU_RADIX4_EINVAL;
    if (packed_bytes < bytes) return BNCFU_RADIX4_ENOSPC;
    const unsigned planes = bits == 8 ? 6 : 3;
    const size_t tiles = k / b + (k % b != 0);
    const size_t row_bytes = bytes / n;
    static const int64_t coefficients[6] = {1, 3, 9, 27, 81, 243};
    int8_t scratch[BNCFU_TERNARY3_LANES]
        __attribute__((aligned(BNCFU_TERNARY3_LANES)));

    bncfu_enable();
    bncfu_dma_fence();
    bncfu_config(BNCFU_QTYPE_15B);
    for (size_t row = 0; row < m; ++row) {
        const int8_t *x = activation + row * k;
        for (size_t col = 0; col < n; col += BNCFU_REUSE_REGS) {
            const size_t cols = n - col < BNCFU_REUSE_REGS
                ? n - col : BNCFU_REUSE_REGS;
            int64_t acc[BNCFU_REUSE_REGS][6] = {{0}};
            for (size_t tile = 0; tile < tiles; ++tile) {
                const size_t start = tile * b;
                const size_t count = k - start < b ? k - start : b;
                const int8_t *a = x + start;
                if (count < b || (uintptr_t)a % b != 0) {
                    memset(scratch, 0, sizeof(scratch));
                    memcpy(scratch, a, count);
                    /* CFU DMA does not snoop dirty CPU cache lines. */
                    bncfu_dma_fence();
                    a = scratch;
                }
                /* Loading A must not reset any W bank's low-bit cursor. */
                bncfu_load(0, a);
                for (unsigned p = 0; p < planes; ++p) {
                    const size_t s = tile * planes + p;
                    if (s % 4 == 0) {
                        for (size_t c = 0; c < cols; ++c)
                            bncfu_load((unsigned)c + 1,
                                packed + (col + c) * row_bytes + (s / 4) * b);
                    }
                    /* Advance each W cursor once per plane, including
                     * planes that continue in a vector across K tiles. */
                    for (size_t c = 0; c < cols; ++c)
                        acc[c][p] += bncfu_bdot2(0, (unsigned)c + 1);
                }
            }
            for (size_t c = 0; c < cols; ++c) {
                int64_t result = 0;
                for (unsigned p = 0; p < planes; ++p)
                    result += acc[c][p] * coefficients[p];
                if (result < INT32_MIN || result > INT32_MAX)
                    return BNCFU_RADIX4_ERANGE;
                output[row * n + col + c] = (int32_t)result;
            }
        }
    }
    return BNCFU_RADIX4_OK;
}

BncfuRadix4Status bncfu_ternary3_gemv(
        int32_t *output, const int8_t *activation,
        const uint8_t *packed, size_t packed_bytes,
        size_t n, size_t k, unsigned bits) {
    return bncfu_ternary3_gemm(output, activation, packed, packed_bytes,
                             1, n, k, bits);
}
