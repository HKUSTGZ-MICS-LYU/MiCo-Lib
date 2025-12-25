#include "custom_asm.h"

.global __mico_mixed4x8_vecXmat
// Process one output of a Vec x Mat (Linear) operation
// a0: pointer to A Vec (4-bit)
// a1: pointer to W Mat (8-bit)
// a2: pointer to O Vec (32-bit)
// a3: Number of elements in A Vec (Inner loop 1)
// a4: Number of elements in W Mat (Outer loop 2)
__mico_mixed4x8_vecXmat:
    li t0, 0  // Loop counter 1
loop1:
    li a6, 8  // Activation Per Word
    li t1, 0  // Loop counter 2
    mv a5, a0 // Save the starting pointer of A Vec
    li t4, 0  // Accumulator
loop2:
    ld t3, 0(a5) // Load Vec A
    addi a5, a5, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t3, 0(a5) // Load Vec A
    addi a5, a5, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t3, 0(a5) // Load Vec A
    addi a5, a5, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t3, 0(a5) // Load Vec A
    addi a5, a5, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    ld t2, 0(a1) // Load Mat W
    opcode_R(CUSTOM0, 0x5, 0x01, t2, t2, t3)
    add t4, t4, t2
    addi a1, a1, 8

    addi t1, t1, 64
    blt t1, a3, loop2
    sw t4, 0(a2)
    addi t0, t0, 1
    addi a2, a2, 4
    blt t0, a4, loop1
    ret