#ifndef BNCFU_BITNET_CFU_H
#define BNCFU_BITNET_CFU_H

#include <stdint.h>

#ifndef VLEN
#define VLEN 256
#endif

#ifndef BITNET_QUANT
#define BITNET_QUANT 3
#endif

#ifndef BNCFU_REG_DEPTH
#define BNCFU_REG_DEPTH 2
#endif

#if BNCFU_REG_DEPTH < 2
#error "BNCFU_REG_DEPTH must be >= 2"
#endif

#if BNCFU_REG_DEPTH > 8
#error "BNCFU raw rs-field software intrinsics currently support BNCFU_REG_DEPTH <= 8"
#endif

#if VLEN > 512
#error "BNCFU Q2T packs one VLEN result into rd, so VLEN must be <= 512"
#endif

#define BNCFU_REUSE_REGS (BNCFU_REG_DEPTH - 1)
#define BNCFU_BYTES (VLEN / 8)
#define BNCFU_Q8_ELEMS (VLEN / 8)
#define BNCFU_Q2T_ELEMS (VLEN / 32)
#define BNCFU_Q2T_BYTES (BNCFU_Q2T_ELEMS / 4)
#define BNCFU_Q1_DOTS_PER_LOAD 8
#define BNCFU_Q2_DOTS_PER_LOAD 4
#define BNCFU_Q2_SLICE_BYTES (BNCFU_BYTES / BNCFU_Q2_DOTS_PER_LOAD)
#define BNCFU_Q1_FULL_ELEMS (BNCFU_Q8_ELEMS * BNCFU_Q1_DOTS_PER_LOAD)
#define BNCFU_Q2_FULL_ELEMS (BNCFU_Q8_ELEMS * BNCFU_Q2_DOTS_PER_LOAD)

#define BNCFU_ALWAYS_INLINE static inline __attribute__((always_inline))
#define BNCFU_A5 15
#define BNCFU_A0 10
#define BNCFU_ENCODE_R(func3, rd, rs1, rs2) \
    (0x0B | ((rd) << 7) | ((func3) << 12) | ((rs1) << 15) | ((rs2) << 20))
#define BNCFU_CASES_0_7(macro) \
    macro(0) macro(1) macro(2) macro(3) macro(4) macro(5) macro(6) macro(7)

BNCFU_ALWAYS_INLINE void bncfu_enable(void){
    __asm__ volatile(
        "li t1, 0x80000000\n\t"
        "csrs 0xBC0, t1"
        ::: "t1"
    );
}

BNCFU_ALWAYS_INLINE void bncfu_fence(void){
    __asm__ volatile("fence rw, rw" ::: "memory");
}

BNCFU_ALWAYS_INLINE void bncfu_dma_fence(void){
    __asm__ volatile("fence rw, rw" ::: "memory");
    __asm__ volatile(".word 0x0000100f" ::: "memory");
}

BNCFU_ALWAYS_INLINE void bncfu_load(unsigned int bank, const void *addr){
    uintptr_t addr_reg = (uintptr_t)addr;
    register uintptr_t addr_a5 asm("a5") = addr_reg;

    switch(bank) {
#define BNCFU_LOAD_CASE(id) \
    case id: \
        __asm__ volatile( \
            ".word %[insn]" \
            : "+r"(addr_a5) \
            : [insn] "i"(BNCFU_ENCODE_R(4, 0, BNCFU_A5, id)) \
            : "memory"); \
        break;
        BNCFU_CASES_0_7(BNCFU_LOAD_CASE)
#undef BNCFU_LOAD_CASE
    default:
        break;
    }
}

BNCFU_ALWAYS_INLINE int32_t bncfu_bdot2(unsigned int int8_bank, unsigned int lowbit_bank){
    register int32_t result asm("a0");
    const unsigned int insn = BNCFU_ENCODE_R(1, BNCFU_A0, int8_bank, lowbit_bank);

    switch(insn) {
#define BNCFU_BDOT_CASE(rs1, rs2) \
    case BNCFU_ENCODE_R(1, BNCFU_A0, rs1, rs2): \
        __asm__ volatile( \
            ".word %[enc]" \
            : "=&r"(result) \
            : [enc] "i"(BNCFU_ENCODE_R(1, BNCFU_A0, rs1, rs2)) \
            : "memory"); \
        break;
#define BNCFU_BDOT_CASES_FOR_RS1(rs1) \
    BNCFU_BDOT_CASE(rs1, 0) BNCFU_BDOT_CASE(rs1, 1) BNCFU_BDOT_CASE(rs1, 2) BNCFU_BDOT_CASE(rs1, 3) \
    BNCFU_BDOT_CASE(rs1, 4) BNCFU_BDOT_CASE(rs1, 5) BNCFU_BDOT_CASE(rs1, 6) BNCFU_BDOT_CASE(rs1, 7)
        BNCFU_CASES_0_7(BNCFU_BDOT_CASES_FOR_RS1)
#undef BNCFU_BDOT_CASES_FOR_RS1
#undef BNCFU_BDOT_CASE
    default:
        result = 0;
        break;
    }
    return result;
}

BNCFU_ALWAYS_INLINE int32_t bncfu_bdot2_hold(unsigned int int8_reg, unsigned int lowbit_reg){
    register int32_t result asm("a0");
    const unsigned int insn = BNCFU_ENCODE_R(0, BNCFU_A0, int8_reg, lowbit_reg);

    switch(insn) {
#define BNCFU_BDOT_HOLD_CASE(rs1, rs2) \
    case BNCFU_ENCODE_R(0, BNCFU_A0, rs1, rs2): \
        __asm__ volatile( \
            ".word %[enc]" \
            : "=&r"(result) \
            : [enc] "i"(BNCFU_ENCODE_R(0, BNCFU_A0, rs1, rs2)) \
            : "memory"); \
        break;
#define BNCFU_BDOT_HOLD_CASES_FOR_RS1(rs1) \
    BNCFU_BDOT_HOLD_CASE(rs1, 0) BNCFU_BDOT_HOLD_CASE(rs1, 1) BNCFU_BDOT_HOLD_CASE(rs1, 2) BNCFU_BDOT_HOLD_CASE(rs1, 3) \
    BNCFU_BDOT_HOLD_CASE(rs1, 4) BNCFU_BDOT_HOLD_CASE(rs1, 5) BNCFU_BDOT_HOLD_CASE(rs1, 6) BNCFU_BDOT_HOLD_CASE(rs1, 7)
        BNCFU_CASES_0_7(BNCFU_BDOT_HOLD_CASES_FOR_RS1)
#undef BNCFU_BDOT_HOLD_CASES_FOR_RS1
#undef BNCFU_BDOT_HOLD_CASE
    default:
        result = 0;
        break;
    }
    return result;
}

BNCFU_ALWAYS_INLINE int32_t bncfu_bdot(unsigned int lowbit_bank){
    return bncfu_bdot2(0, lowbit_bank);
}

BNCFU_ALWAYS_INLINE uint32_t bncfu_q2t(uint32_t absmax_bits, unsigned int bank){
    register uintptr_t absmax_a5 asm("a5") = (uintptr_t)absmax_bits;
    register uint32_t result asm("a0");

    switch(bank) {
#define BNCFU_Q2T_CASE(id) \
    case id: \
        __asm__ volatile( \
            ".word %[insn]" \
            : "=&r"(result) \
            : [insn] "i"(BNCFU_ENCODE_R(3, BNCFU_A0, BNCFU_A5, id)), "r"(absmax_a5) \
            : "memory"); \
        break;
        BNCFU_CASES_0_7(BNCFU_Q2T_CASE)
#undef BNCFU_Q2T_CASE
    default:
        result = 0;
        break;
    }
    return result;
}

BNCFU_ALWAYS_INLINE uint32_t bncfu_float_bits(float value){
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

#endif
