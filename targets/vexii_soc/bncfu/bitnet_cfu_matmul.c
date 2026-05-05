#include "mico_qnn.h"

#ifndef VLEN
#define VLEN 256
#endif

#ifndef BITNET_QUANT
#define BITNET_QUANT 3
#endif

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

#define bncfu_load_v0(addr) do { \
    register uintptr_t _addr_reg asm("a5") = (uintptr_t)(addr); \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | (15 << 15) | (0 << 20) | (0x4 << 12) | (0 << 25))" \
        : "+r"(_addr_reg) :: "memory" \
    ); \
} while(0)

#define bncfu_load_v1(addr) do { \
    register uintptr_t _addr_reg asm("a6") = (uintptr_t)(addr); \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | (16 << 15) | (1 << 20) | (0x4 << 12) | (0 << 25))" \
        : "+r"(_addr_reg) :: "memory" \
    ); \
} while(0)

#define bncfu_bdot_v0_v1() ({ \
    register int32_t _result asm("t0"); \
    __asm__ volatile( \
        ".word (0x0B | (5 << 7) | (0 << 15) | (1 << 20) | (0x1 << 12) | (0 << 25))" \
        : "=r"(_result) \
    ); \
    _result; \
})

#define bncfu_bdot_v1_v0() ({ \
    register int32_t _result asm("t0"); \
    __asm__ volatile( \
        ".word (0x0B | (5 << 7) | (1 << 15) | (0 << 20) | (0x1 << 12) | (0 << 25))" \
        : "=r"(_result) \
    ); \
    _result; \
})

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
        for(size_t j = 0; j < out_features; ++j) {
            const int8_t *w_base = w->data + ((j * in_features) >> 3);
            const int8_t *x_ptr = x_base;
            const int8_t *w_ptr = w_base;
            int32_t acc = 0;
            for(size_t k = 0; k < full_iters; ++k) {
                bncfu_load_v1(w_ptr);
                w_ptr += BNCFU_BYTES;
                for(size_t d = 0; d < BNCFU_Q1_DOTS_PER_LOAD; ++d) {
                    bncfu_load_v0(x_ptr);
                    acc += bncfu_bdot_v0_v1();
                    x_ptr += BNCFU_BYTES;
                }
            }
            O[i * out_features + j] = acc;
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
        for(size_t j = 0; j < out_features; ++j) {
            const int8_t *w_base = w->data + j * in_features;
            const int8_t *x_ptr = x_base;
            const int8_t *w_ptr = w_base;
            int32_t acc = 0;
            for(size_t k = 0; k < full_iters; ++k) {
                bncfu_load_v0(x_ptr);
                x_ptr += BNCFU_BYTES;
                for(size_t d = 0; d < BNCFU_Q1_DOTS_PER_LOAD; ++d) {
                    bncfu_load_v1(w_ptr);
                    acc += bncfu_bdot_v1_v0();
                    w_ptr += BNCFU_BYTES;
                }
            }
            O[i * out_features + j] = acc;
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
        for(size_t j = 0; j < out_features; ++j) {
            const int8_t *w_base = w->data + ((j * in_features) >> 2);
            const int8_t *x_ptr = x_base;
            const int8_t *w_ptr = w_base;
            int32_t acc = 0;
            for(size_t k = 0; k < full_iters; ++k) {
                bncfu_load_v1(w_ptr);
                w_ptr += BNCFU_BYTES;
                for(size_t d = 0; d < BNCFU_Q2_DOTS_PER_LOAD; ++d) {
                    bncfu_load_v0(x_ptr);
                    acc += bncfu_bdot_v0_v1();
                    x_ptr += BNCFU_BYTES;
                }
            }
            O[i * out_features + j] = acc;
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
        for(size_t j = 0; j < out_features; ++j) {
            const int8_t *w_base = w->data + j * in_features;
            const int8_t *x_ptr = x_base;
            const int8_t *w_ptr = w_base;
            int32_t acc = 0;
            for(size_t k = 0; k < full_iters; ++k) {
                bncfu_load_v0(x_ptr);
                x_ptr += BNCFU_BYTES;
                for(size_t d = 0; d < BNCFU_Q2_DOTS_PER_LOAD; ++d) {
                    bncfu_load_v1(w_ptr);
                    acc += bncfu_bdot_v1_v0();
                    w_ptr += BNCFU_BYTES;
                }
            }
            O[i * out_features + j] = acc;
        }
    }
}
#endif
