#include "bitnet_cfu_radix4.h"
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

size_t bncfu_radix4_packed_size(size_t n, size_t k, unsigned bits) {
    if (!valid_bits(bits)) return 0;
    const size_t group_k = BNCFU_RADIX4_LANES * (8u / bits);
    const size_t groups = k / group_k + (k % group_k != 0);
    if (!product_fits(groups, BNCFU_RADIX4_LANES)) return 0;
    const size_t row_bytes = groups * BNCFU_RADIX4_LANES;
    return product_fits(n, row_bytes) ? n * row_bytes : 0;
}

/* Weights are prepared offline by weight_recode.py. Keep only layout-size
 * validation and compute kernels in the C library, not a runtime packer. */

BncfuRadix4Status bncfu_radix4_gemm(
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
    const size_t bytes = bncfu_radix4_packed_size(n, k, bits);
    const size_t b = BNCFU_RADIX4_LANES;
    if (!bytes || !activation || !packed || (uintptr_t)packed % b != 0)
        return BNCFU_RADIX4_EINVAL;
    if (packed_bytes < bytes) return BNCFU_RADIX4_ENOSPC;
    const unsigned planes = bits / 2;
    const size_t tiles = 4 / planes;
    const size_t groups = (bytes / n) / b;
    const int correction_factor = bits == 8 ? 42 : 2;
    int8_t scratch[BNCFU_RADIX4_LANES] __attribute__((aligned(BNCFU_RADIX4_LANES)));

    bncfu_enable();
    bncfu_dma_fence();
    bncfu_config(BNCFU_QTYPE_2B);
    for (size_t row = 0; row < m; ++row) {
        const int8_t *x = activation + row * k;
        int64_t sum = 0;
        for (size_t i = 0; i < k; ++i) sum += x[i];
        const int64_t correction = sum * correction_factor;
        for (size_t col = 0; col < n; col += BNCFU_REUSE_REGS) {
            const size_t cols = n - col < BNCFU_REUSE_REGS ? n - col : BNCFU_REUSE_REGS;
            int64_t acc[BNCFU_REUSE_REGS][4] = {{0}};
            for (size_t g = 0; g < groups; ++g) {
                for (size_t c = 0; c < cols; ++c)
                    bncfu_load((unsigned)c + 1, packed + ((col + c) * groups + g) * b);
                for (size_t t = 0; t < tiles; ++t) {
                    const size_t start = (g * tiles + t) * b;
                    if (start >= k) break;
                    const size_t count = k - start < b ? k - start : b;
                    const int8_t *a = x + start;
                    if (count < b || (uintptr_t)a % b != 0) {
                        memset(scratch, 0, sizeof(scratch));
                        memcpy(scratch, a, count);
                        /* CFU DMA does not snoop dirty CPU cache lines. */
                        bncfu_dma_fence();
                        a = scratch;
                    }
                    bncfu_load(0, a);
                    for (unsigned p = 0; p < planes; ++p) {
                        for (size_t c = 0; c < cols; ++c)
                            acc[c][p] += bncfu_bdot2(0, (unsigned)c + 1);
                    }
                }
            }
            for (size_t c = 0; c < cols; ++c) {
                int64_t result = correction;
                for (unsigned p = 0; p < planes; ++p)
                    result += acc[c][p] * (INT64_C(1) << (2 * p));
                if (result < INT32_MIN || result > INT32_MAX) return BNCFU_RADIX4_ERANGE;
                output[row * n + col + c] = (int32_t)result;
            }
        }
    }
    return BNCFU_RADIX4_OK;
}

BncfuRadix4Status bncfu_radix4_gemv(
        int32_t *output, const int8_t *activation,
        const uint8_t *packed, size_t packed_bytes,
        size_t n, size_t k, unsigned bits) {
    return bncfu_radix4_gemm(output, activation, packed, packed_bytes, 1, n, k, bits);
}
