#ifndef MICO_RVV_COMMON_H
#define MICO_RVV_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include "mico_rvv_cache.h"

#ifndef __riscv_vector
#error "The MiCo RVV target requires the RISC-V vector extension"
#endif

/* Hardware capability is separate from the compiler ISA feature macro.
 * Explicit-cache Vexii must not vector-store reduction scratch on the stack. */
static inline int32_t mico_rvv_reduce_i32m4(vint32m4_t value, size_t vlmax32) {
#if !defined(MICO_RVV_EXPLICIT_CACHE) && defined(__riscv_zve32x) && \
    (!defined(MICO_RVV_HAS_REDUCTION) || !MICO_RVV_HAS_REDUCTION)
    int32_t lanes[vlmax32];
    __riscv_vse32_v_i32m4(lanes, value, vlmax32);
    int32_t result = 0;
    for (size_t lane = 0; lane < vlmax32; ++lane) {
        result += lanes[lane];
    }
    return result;
#else
    const vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
    const vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(value, vzero, vlmax32);
    return __riscv_vmv_x_s_i32m1_i32(vsum);
#endif
}

#endif /* MICO_RVV_COMMON_H */
