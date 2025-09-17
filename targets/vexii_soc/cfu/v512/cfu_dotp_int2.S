
#include "custom_asm.h"

.global cfu_dotp_int2
// Process one output of a Vec x Mat (Linear) operation (512-bit)
// a0: pointer to A Vec (2-bit)
// a1: pointer to W Mat (2-bit)
// a2: pointer to O Vec (32-bit)
// a3: Number of elements in A Vec (Inner loop 1)
// a4: Number of elements in W Mat (Outer loop 2)
cfu_dotp_int2:
    fence.i
    vpu_CONFIG(2, 2)
    li t0, 0  // Loop counter 1
loop1:
    li t1, 0  // Loop counter 2
    mv a5, a0 // Save pointer to W Mat
    li t4, 0  // Accumulator
loop2:

    vpu_LOAD(a5, v0)
    vpu_LOAD(a1, v1)
    vpu_VDOT(t2, v0, v1)    // SIMD Dot Product
    add t4, t4, t2
    addi a5, a5, 64
    addi a1, a1, 64

    addi t1, t1, 256
    blt t1, a3, loop2

    sw t4, 0(a2)
    addi t0, t0, 1
    addi a2, a2, 4
    blt t0, a4, loop1
    ret
