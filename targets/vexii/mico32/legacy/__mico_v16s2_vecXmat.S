#include "custom_asm.h"

.global __mico_v16s2_vecXmat
// Process one output of a Vec x Mat (Linear) operation
// a0: pointer to A Vec (2-bit)
// a1: pointer to W Mat (2-bit)
// a2: pointer to O Vec (32-bit)
// a3: Number of elements in A Vec (Inner loop 1)
// a4: Number of elements in W Mat (Outer loop 2)
__mico_v16s2_vecXmat:
    li t0, 0  // Loop counter 1
loop1:
    li t1, 0  // Loop counter 2
    mv a5, a0 // Save pointer to W Mat
    li t4, 0  // Accumulator
loop2:
    lw t2, 0(a5) // Load Vec A
    lw t3, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x6, 0x04, t2, t2, t3)
    add t4, t4, t2
    addi a5, a5, 4
    addi a1, a1, 4

    lw t2, 0(a5) // Load Vec A
    lw t3, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x6, 0x04, t2, t2, t3)
    add t4, t4, t2
    addi a5, a5, 4
    addi a1, a1, 4

    addi t1, t1, 32
    blt t1, a3, loop2

    sw t4, 0(a2)
    addi t0, t0, 1
    addi a2, a2, 4
    blt t0, a4, loop1
    ret