#ifndef MICO_RVV_CACHE_H
#define MICO_RVV_CACHE_H

/* Opt-in ONLY: Vexii's noncoherent L1/Coral configuration, 64-byte lines,
 * no L2. This is not a portable interpretation of fence.i or CBO completion.
 * Ordinary fence drains CPU stores but does not write back dirty L1 data.
 * The terminal vector store must retire only after engine done/native idle;
 * a CPU fence alone cannot wait for an early-acknowledged accelerator store.
 */
#ifdef MICO_RVV_EXPLICIT_CACHE
#if !defined(RISCV_VEXII) || !defined(MICO_RVV) || !defined(__riscv_zicbom)
#error "MICO_RVV_EXPLICIT_CACHE requires RISCV_VEXII, MICO_RVV and Zicbom"
#endif

#include "nn.h"
#include <stddef.h>
#include <stdint.h>

#define MICO_RVV_CACHE_LINE_BYTES 64u
#define MICO_RVV_CACHE_INLINE static inline __attribute__((always_inline))

typedef struct {
    int32_t *user;
    int32_t *data;
    size_t count;
    size_t padded_bytes;
} mico_rvv_cache_output;

/* Fail closed: never vector-store into an unisolated user buffer on failure. */
static inline __attribute__((noreturn)) void mico_rvv_cache_fail(void) {
    MiCo_assert(0, "RVV cache bounce allocation/size/alignment failure");
    __builtin_trap();
}

MICO_RVV_CACHE_INLINE void mico_rvv_cache_barrier(void) {
    __asm__ volatile("fence rw, rw" ::: "memory");
}

/* Once per API/kernel invocation, not per K chunk. The Vexii fence.i cleans
 * all dirty L1 lines while retaining clean valid lines, matching the BNCFU
 * helper. Avoid scanning large weights that are no longer resident in L1.
 * Literal fence.i encoding also works with toolchains whose march omits
 * Zifencei; the enabled hardware configuration must support this instruction.
 */
MICO_RVV_CACHE_INLINE void mico_rvv_cache_inputs_clean(void) {
    mico_rvv_cache_barrier();
    __asm__ volatile(".word 0x0000100f" ::: "memory");
    mico_rvv_cache_barrier();
}

/* Only use these CBO ranges on aligned, whole-line, exclusively owned bounce
 * storage. Never invalidate user O: it may share a line with dirty CPU data.
 * Offset loops avoid forming a wrapping end-address after the last line.
 */
MICO_RVV_CACHE_INLINE void mico_rvv_cache_output_flush(
    const mico_rvv_cache_output *buffer) {
    for (size_t offset = 0; offset < buffer->padded_bytes;
         offset += MICO_RVV_CACHE_LINE_BYTES) {
        const void *line = (const uint8_t *)buffer->data + offset;
        __asm__ volatile("cbo.flush (%0)" :: "r"(line) : "memory");
    }
    mico_rvv_cache_barrier(); /* Wait for CBO writebacks before RVV. */
}

MICO_RVV_CACHE_INLINE void mico_rvv_cache_output_invalidate(
    const mico_rvv_cache_output *buffer) {
    mico_rvv_cache_barrier();
    for (size_t offset = 0; offset < buffer->padded_bytes;
         offset += MICO_RVV_CACHE_LINE_BYTES) {
        const void *line = (const uint8_t *)buffer->data + offset;
        __asm__ volatile("cbo.inval (%0)" :: "r"(line) : "memory");
    }
    mico_rvv_cache_barrier();
}

/* Allocator bookkeeping lies before the first output line, not inside it.
 * Round the PAYLOAD up as well as aligning its start so the last M-tail line
 * cannot contain the next allocation or unrelated dirty CPU data.
 */
MICO_RVV_CACHE_INLINE mico_rvv_cache_output mico_rvv_cache_output_begin(
    int32_t *user, size_t batches, size_t outputs) {
    mico_rvv_cache_output buffer = {user, NULL, 0, 0};
    if (outputs != 0 && batches > SIZE_MAX / outputs)
        mico_rvv_cache_fail();
    buffer.count = batches * outputs;
    if (buffer.count > SIZE_MAX / sizeof(int32_t))
        mico_rvv_cache_fail();
    const size_t bytes = buffer.count * sizeof(int32_t);
    const size_t mask = MICO_RVV_CACHE_LINE_BYTES - 1u;
    if (bytes > SIZE_MAX - mask)
        mico_rvv_cache_fail();
    buffer.padded_bytes = (bytes + mask) & ~mask;
    /* MiCo_alloc itself adds alignment padding and a saved raw pointer. */
    if (buffer.padded_bytes > SIZE_MAX - mask - sizeof(void *))
        mico_rvv_cache_fail();
    if (buffer.padded_bytes != 0) {
        buffer.data = (int32_t *)MiCo_alloc(buffer.padded_bytes,
                                           MICO_RVV_CACHE_LINE_BYTES);
        if (buffer.data == NULL || ((uintptr_t)buffer.data & mask) != 0 ||
            (uintptr_t)buffer.data > UINTPTR_MAX - buffer.padded_bytes)
            mico_rvv_cache_fail();
    }
    /* Includes allocator CPU writes as well as all CPU-produced inputs. */
    mico_rvv_cache_inputs_clean();
    mico_rvv_cache_output_flush(&buffer);
    return buffer;
}

/* Call only AFTER the terminal real vector store has completed. Maintenance,
 * scalar copy (exact count, no padding), and free remain inside MatMul timing.
 * Volatile accesses prevent even an independently vectorizing build from
 * introducing another noncoherent vector store to the user destination.
 */
MICO_RVV_CACHE_INLINE void mico_rvv_cache_output_end(
    const mico_rvv_cache_output *buffer) {
    mico_rvv_cache_output_invalidate(buffer);
    volatile int32_t *dst = buffer->user;
    const volatile int32_t *src = buffer->data;
    for (size_t i = 0; i < buffer->count; ++i)
        dst[i] = src[i];
    MiCo_free(buffer->data);
}

#undef MICO_RVV_CACHE_INLINE
#endif /* MICO_RVV_EXPLICIT_CACHE */
#endif /* MICO_RVV_CACHE_H */
