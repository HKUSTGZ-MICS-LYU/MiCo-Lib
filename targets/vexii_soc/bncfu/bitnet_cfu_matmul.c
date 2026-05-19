#include "mico_qnn.h"
#include "mico_quant.h"
#include "profile.h"

extern float MiCo_absmax(float* x, size_t n);

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

#define bncfu_q2t(absmax_bits, bank) ({ \
    uintptr_t _absmax_bits = (uintptr_t)(absmax_bits); \
    uintptr_t _bank_reg = (uintptr_t)(bank); \
    uint32_t _result; \
    __asm__ volatile( \
        ".insn r 0x0B, 0x3, 0, %0, %1, %2" \
        : "=r"(_result) : "r"(_absmax_bits), "r"(_bank_reg) \
    ); \
    _result; \
})

static inline uint32_t bncfu_float_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

#ifdef BNCFU_Q2T
float __FP32toQ2T(qbyte* qx, float* x, size_t n) {
    long start = MiCo_time();
    float scale = 1.0 / MiCo_absmax(x, n);
    const float absmax = 1.0 / scale;
    const uint32_t absmax_bits = bncfu_float_bits(absmax);
    int i = 0;

    bncfu_enable();
    bncfu_fence();

    for(; i + BNCFU_Q2T_ELEMS <= (int)n; i += BNCFU_Q2T_ELEMS) {
        bncfu_load(0, x + i);
        const uint32_t packed = bncfu_q2t(absmax_bits, 0);
        for(int j = 0; j < BNCFU_Q2T_BYTES; ++j) {
            qx[i / 4 + j] = (qbyte)(packed >> (j * 8));
        }
    }

    for(; i + 4 <= (int)n; i += 4) {
        qx[i/4] = (CLAMP_INT2T((int8_t)(roundf2i(x[i] * scale))) & 0x3) |
            ((CLAMP_INT2T((int8_t)(roundf2i(x[i+1] * scale))) & 0x3) << 2) |
            ((CLAMP_INT2T((int8_t)(roundf2i(x[i+2] * scale))) & 0x3) << 4) |
            ((CLAMP_INT2T((int8_t)(roundf2i(x[i+3] * scale))) & 0x3) << 6);
    }
    if(i < (int)n) {
        qx[i/4] = 0;
        for(int j = 0; i + j < (int)n; ++j) {
            qx[i/4] |= (CLAMP_INT2T((int8_t)(roundf2i(x[i+j] * scale))) & 0x3) << (2 * j);
        }
    }

    QUANT_TIMER += MiCo_time() - start;
    return 1.0 / scale;
}
#endif

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
