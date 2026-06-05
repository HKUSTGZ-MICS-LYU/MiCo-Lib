#include "mico_qnn.h"
#include "mico_quant.h"
#include "profile.h"
#include "bitnet_cfu.h"
#include <string.h>

typedef void (*BncfuMatMulKernel)(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

static size_t bncfu_round_up(size_t value, size_t align) {
    return ((value + align - 1u) / align) * align;
}

static size_t bncfu_row_bytes(size_t elements, unsigned int bits) {
    return (elements * bits + 7u) >> 3;
}

static void bncfu_pad_rows(
        qbyte *dst,
        const qbyte *src,
        size_t rows,
        size_t old_cols,
        size_t new_cols,
        unsigned int bits) {
    const size_t old_row_bytes = bncfu_row_bytes(old_cols, bits);
    const size_t new_row_bytes = bncfu_row_bytes(new_cols, bits);

    memset(dst, 0, rows * new_row_bytes);
    for(size_t row = 0; row < rows; ++row) {
        memcpy(dst + row * new_row_bytes, src + row * old_row_bytes, old_row_bytes);
    }
}

static int bncfu_pad_and_run(
        int32_t *O,
        const Tensor2D_Q8 *x,
        const Tensor2D_Q8 *w,
        size_t full_elems,
        unsigned int x_bits,
        unsigned int w_bits,
        BncfuMatMulKernel kernel) {
    const size_t in_features = x->shape[1];
    if((in_features % full_elems) == 0) {
        return 0;
    }

    const size_t padded_k = bncfu_round_up(in_features, full_elems);
    Tensor2D_Q8 px = *x;
    Tensor2D_Q8 pw = *w;
    px.shape[1] = padded_k;
    pw.shape[1] = padded_k;
    px.data = (qbyte *)MiCo_alloc(x->shape[0] * bncfu_row_bytes(padded_k, x_bits), MICO_ALIGN);
    pw.data = (qbyte *)MiCo_alloc(w->shape[0] * bncfu_row_bytes(padded_k, w_bits), MICO_ALIGN);
    MiCo_assert(px.data != NULL && pw.data != NULL, "[BNCFU MatMul] failed to allocate padded buffers");

    bncfu_pad_rows(px.data, x->data, x->shape[0], in_features, padded_k, x_bits);
    bncfu_pad_rows(pw.data, w->data, w->shape[0], in_features, padded_k, w_bits);
    kernel(O, &px, &pw);

    MiCo_free(px.data);
    MiCo_free(pw.data);
    return 1;
}

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q1_FULL_ELEMS;

    if(bncfu_pad_and_run(O, x, w, BNCFU_Q1_FULL_ELEMS, 8, 1, MiCo_Q8x1_MatMul)) {
        return;
    }

    bncfu_enable();
    bncfu_fence();
    bncfu_config(BNCFU_QTYPE_1B);

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

    if(bncfu_pad_and_run(O, x, w, BNCFU_Q1_FULL_ELEMS, 1, 8, MiCo_Q1x8_MatMul)) {
        return;
    }

    bncfu_enable();
    bncfu_fence();
    bncfu_config(BNCFU_QTYPE_1B);

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

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q2_FULL_ELEMS;

    if(bncfu_pad_and_run(O, x, w, BNCFU_Q2_FULL_ELEMS, 8, 2, MiCo_Q8x2_MatMul)) {
        return;
    }

    bncfu_enable();
    bncfu_fence();
    bncfu_config(BNCFU_QTYPE_15B);

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

    if(bncfu_pad_and_run(O, x, w, BNCFU_Q2_FULL_ELEMS, 2, 8, MiCo_Q2x8_MatMul)) {
        return;
    }

    bncfu_enable();
    bncfu_fence();
    bncfu_config(BNCFU_QTYPE_15B);

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
