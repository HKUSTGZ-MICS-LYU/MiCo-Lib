#include "mico_qnn.h"

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

#define BNCFU_REUSE_REGS (BNCFU_REG_DEPTH - 1)
#define BNCFU_BYTES (VLEN / 8)
#define BNCFU_Q8_ELEMS (VLEN / 8)
#define BNCFU_Q1_DOTS_PER_LOAD 8
#define BNCFU_Q2_DOTS_PER_LOAD 4
#define BNCFU_Q1_FULL_ELEMS (BNCFU_Q8_ELEMS * BNCFU_Q1_DOTS_PER_LOAD)
#define BNCFU_Q2_FULL_ELEMS (BNCFU_Q8_ELEMS * BNCFU_Q2_DOTS_PER_LOAD)

#define bncfu_enable() do { \
    __asm__ volatile( \
        "li t1, 0x80000000\n\t" \
        "csrs 0xBC0, t1" \
        ::: "t1" \
    ); \
} while(0)

#define bncfu_fence() __asm__ volatile(".word 0x0000100f" ::: "memory")

#define bncfu_load(bank, addr) do { \
    uintptr_t _addr_reg = (uintptr_t)(addr); \
    uintptr_t _bank_reg = (uintptr_t)(bank); \
    __asm__ volatile( \
        ".insn r 0x0B, 0x4, 0, x0, %0, %1" \
        :: "r"(_addr_reg), "r"(_bank_reg) : "memory" \
    ); \
} while(0)

#define bncfu_bdot2(int8_reg, lowbit_reg) ({ \
    uintptr_t _int8_reg = (uintptr_t)(int8_reg); \
    uintptr_t _lowbit_reg = (uintptr_t)(lowbit_reg); \
    int32_t _result; \
    __asm__ volatile( \
        ".insn r 0x0B, 0x1, 0, %0, %1, %2" \
        : "=r"(_result) : "r"(_int8_reg), "r"(_lowbit_reg) \
    ); \
    _result; \
})

#define bncfu_bdot2_hold(int8_reg, lowbit_reg) ({ \
    uintptr_t _int8_reg = (uintptr_t)(int8_reg); \
    uintptr_t _lowbit_reg = (uintptr_t)(lowbit_reg); \
    int32_t _result; \
    __asm__ volatile( \
        ".insn r 0x0B, 0x0, 0, %0, %1, %2" \
        : "=r"(_result) : "r"(_int8_reg), "r"(_lowbit_reg) \
    ); \
    _result; \
})

#define bncfu_bdot(bank) bncfu_bdot2(0, bank)

#if BITNET_QUANT == 2
void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q1_FULL_ELEMS;

    bncfu_enable();
    bncfu_fence();

    for(size_t i = 0; i < batch_size; ++i) {
        const int8_t *x_base = x->data + i * in_features;
        for(size_t j = 0; j < out_features; j += BNCFU_REUSE_REGS) {
            const size_t cols = ((out_features - j) < BNCFU_REUSE_REGS) ? (out_features - j) : BNCFU_REUSE_REGS;
            const int8_t *w_base[BNCFU_REUSE_REGS];
            int32_t acc[BNCFU_REUSE_REGS] = {0};
            for(size_t b = 0; b < cols; ++b) {
                w_base[b] = w->data + (((j + b) * in_features) >> 3);
            }
            for(size_t k = 0; k < full_iters; ++k) {
                for(size_t b = 0; b < cols; ++b) {
                    bncfu_load(b + 1, w_base[b] + k * BNCFU_BYTES);
                }
                for(size_t d = 0; d < BNCFU_Q1_DOTS_PER_LOAD; ++d) {
                    const int8_t *x_ptr = x_base + (k * BNCFU_Q1_DOTS_PER_LOAD + d) * BNCFU_BYTES;
                    bncfu_load(0, x_ptr);
                    for(size_t b = 0; b < cols; ++b) {
                        acc[b] += bncfu_bdot2(0, b + 1);
                    }
                }
            }
            for(size_t b = 0; b < cols; ++b) {
                O[i * out_features + j + b] = acc[b];
            }
        }
    }
}

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q1_FULL_ELEMS;

    bncfu_enable();
    bncfu_fence();

    for(size_t i = 0; i < batch_size; ++i) {
        const int8_t *x_base = x->data + ((i * in_features) >> 3);
        for(size_t j = 0; j < out_features; j += BNCFU_REUSE_REGS) {
            const size_t cols = ((out_features - j) < BNCFU_REUSE_REGS) ? (out_features - j) : BNCFU_REUSE_REGS;
            const int8_t *w_base[BNCFU_REUSE_REGS];
            const int8_t *w_ptr[BNCFU_REUSE_REGS];
            int32_t acc[BNCFU_REUSE_REGS] = {0};
            for(size_t b = 0; b < cols; ++b) {
                w_base[b] = w->data + (j + b) * in_features;
                w_ptr[b] = w_base[b];
            }
            const int8_t *x_ptr = x_base;
            for(size_t k = 0; k < full_iters; ++k) {
                bncfu_load(0, x_ptr);
                x_ptr += BNCFU_BYTES;
                for(size_t d = 0; d < BNCFU_Q1_DOTS_PER_LOAD; ++d) {
                    for(size_t b = 0; b < cols; ++b) {
                        bncfu_load(b + 1, w_ptr[b]);
                    }
                    for(size_t b = 0; b < cols; ++b) {
                        if(b + 1 == cols) {
                            acc[b] += bncfu_bdot2(b + 1, 0);
                        } else {
                            acc[b] += bncfu_bdot2_hold(b + 1, 0);
                        }
                        w_ptr[b] += BNCFU_BYTES;
                    }
                }
            }
            for(size_t b = 0; b < cols; ++b) {
                O[i * out_features + j + b] = acc[b];
            }
        }
    }
}
#elif BITNET_QUANT != 0
void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q2_FULL_ELEMS;

    bncfu_enable();
    bncfu_fence();

    for(size_t i = 0; i < batch_size; ++i) {
        const int8_t *x_base = x->data + i * in_features;
        for(size_t j = 0; j < out_features; j += BNCFU_REUSE_REGS) {
            const size_t cols = ((out_features - j) < BNCFU_REUSE_REGS) ? (out_features - j) : BNCFU_REUSE_REGS;
            const int8_t *w_base[BNCFU_REUSE_REGS];
            int32_t acc[BNCFU_REUSE_REGS] = {0};
            for(size_t b = 0; b < cols; ++b) {
                w_base[b] = w->data + (((j + b) * in_features) >> 2);
            }
            for(size_t k = 0; k < full_iters; ++k) {
                for(size_t b = 0; b < cols; ++b) {
                    bncfu_load(b + 1, w_base[b] + k * BNCFU_BYTES);
                }
                for(size_t d = 0; d < BNCFU_Q2_DOTS_PER_LOAD; ++d) {
                    const int8_t *x_ptr = x_base + (k * BNCFU_Q2_DOTS_PER_LOAD + d) * BNCFU_BYTES;
                    bncfu_load(0, x_ptr);
                    for(size_t b = 0; b < cols; ++b) {
                        acc[b] += bncfu_bdot2(0, b + 1);
                    }
                }
            }
            for(size_t b = 0; b < cols; ++b) {
                O[i * out_features + j + b] = acc[b];
            }
        }
    }
}

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q2_FULL_ELEMS;

    bncfu_enable();
    bncfu_fence();

    for(size_t i = 0; i < batch_size; ++i) {
        const int8_t *x_base = x->data + ((i * in_features) >> 2);
        for(size_t j = 0; j < out_features; j += BNCFU_REUSE_REGS) {
            const size_t cols = ((out_features - j) < BNCFU_REUSE_REGS) ? (out_features - j) : BNCFU_REUSE_REGS;
            const int8_t *w_base[BNCFU_REUSE_REGS];
            const int8_t *w_ptr[BNCFU_REUSE_REGS];
            int32_t acc[BNCFU_REUSE_REGS] = {0};
            for(size_t b = 0; b < cols; ++b) {
                w_base[b] = w->data + (j + b) * in_features;
                w_ptr[b] = w_base[b];
            }
            const int8_t *x_ptr = x_base;
            for(size_t k = 0; k < full_iters; ++k) {
                bncfu_load(0, x_ptr);
                x_ptr += BNCFU_BYTES;
                for(size_t d = 0; d < BNCFU_Q2_DOTS_PER_LOAD; ++d) {
                    for(size_t b = 0; b < cols; ++b) {
                        bncfu_load(b + 1, w_ptr[b]);
                    }
                    for(size_t b = 0; b < cols; ++b) {
                        if(b + 1 == cols) {
                            acc[b] += bncfu_bdot2(b + 1, 0);
                        } else {
                            acc[b] += bncfu_bdot2_hold(b + 1, 0);
                        }
                        w_ptr[b] += BNCFU_BYTES;
                    }
                }
            }
            for(size_t b = 0; b < cols; ++b) {
                O[i * out_features + j + b] = acc[b];
            }
        }
    }
}
#endif
