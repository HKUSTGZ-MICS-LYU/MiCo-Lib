#ifndef MICO_RVV_VICUNA_CACHE_H
#define MICO_RVV_VICUNA_CACHE_H

#include <stddef.h>
#include <stdint.h>

#if defined(MICO_RVV_VICUNA_L1_CBO_HANDOFF)
#if !defined(MICO_RVV_BACKEND_VICUNA2)
#error "Vicuna L1 CBO handoff requires RVV_BACKEND=vicuna2"
#endif
#if !defined(__riscv_zicbom)
#error "Vicuna L1 CBO handoff requires Zicbom in MARCH"
#endif

#define MICO_RVV_VICUNA_CBO_LINE_BYTES 64u

/* Vicuna2 OBI bypasses the CPU LSU D-cache.  Flush CPU-dirty lines before
 * vector reads; flush vector-written lines after the API returns so CPU loads
 * cannot observe a stale cached copy.  The Large SoC uses 64-byte CBO blocks. */
static inline void mico_rvv_vicuna_cbo_flush_range(const void *ptr, size_t bytes) {
    if (bytes == 0) return;
    uintptr_t first = (uintptr_t)ptr & ~(uintptr_t)(MICO_RVV_VICUNA_CBO_LINE_BYTES - 1u);
    uintptr_t raw_end = (uintptr_t)ptr + bytes;
    if (raw_end > UINTPTR_MAX - (MICO_RVV_VICUNA_CBO_LINE_BYTES - 1u))
        __builtin_trap();
    uintptr_t end = (raw_end + MICO_RVV_VICUNA_CBO_LINE_BYTES - 1u) &
                    ~(uintptr_t)(MICO_RVV_VICUNA_CBO_LINE_BYTES - 1u);
    __asm__ volatile("fence rw, rw" ::: "memory");
    for (uintptr_t line = first; line < end; line += MICO_RVV_VICUNA_CBO_LINE_BYTES)
        __asm__ volatile("cbo.flush (%0)" :: "r"(line) : "memory");
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline void mico_rvv_vicuna_fence(void) {
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}
#endif /* MICO_RVV_VICUNA_L1_CBO_HANDOFF */

#endif /* MICO_RVV_VICUNA_CACHE_H */
