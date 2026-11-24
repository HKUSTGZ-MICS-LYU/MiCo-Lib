#ifndef BNCFU_BITNET_CFU_RECODE_ACCUM_H
#define BNCFU_BITNET_CFU_RECODE_ACCUM_H

#include <limits.h>
#include <stdint.h>

/* Bound every plane and reconstruction intermediate, not just the final dot.
 * 128 * 32768 * (1+3+9+27+81+243) = 1526726656 < INT32_MAX.
 * Radix4's plane part plus correction is bounded by
 * 128 * 32768 * (2*(1+4+16+64)+42) = 889192448.
 * The chunk is divisible by every supported VLEN/8 and plane-slot group.
 */
#define BNCFU_RECODE_CHUNK_K 32768u

/* Only long-K calls need this range extension. All CPU arithmetic stays
 * 32-bit: the mathematical value is wraps * 2^32 + low. Do not reject an
 * intermediate wrap, because later chunks may cancel it. K <= INT32_MAX
 * bounds |wraps| below 32768 even for the full digit-alphabet triangle bound.
 * This is not saturation or modulo-only output arithmetic.
 */
typedef struct {
    uint32_t low;
    int32_t wraps;
} BncfuRecodeSum;

static inline __attribute__((always_inline)) void bncfu_recode_sum_add(
        BncfuRecodeSum *sum, int32_t value) {
    const uint32_t old = sum->low;
    sum->low += (uint32_t)value;
    sum->wraps += (int32_t)(sum->low < old) - (int32_t)(value < 0);
}

/* Leave *output untouched on overflow. Avoid an out-of-range unsigned-to-
 * signed conversion, and never left-shift a negative signed integer.
 */
static inline __attribute__((always_inline)) int bncfu_recode_sum_result(
        const BncfuRecodeSum *sum, int32_t *output) {
    if (sum->wraps == 0 && sum->low <= (uint32_t)INT32_MAX) {
        *output = (int32_t)sum->low;
        return 1;
    }
    if (sum->wraps == -1 && sum->low > (uint32_t)INT32_MAX) {
        *output = -1 - (int32_t)(UINT32_MAX - sum->low);
        return 1;
    }
    return 0;
}

#endif
