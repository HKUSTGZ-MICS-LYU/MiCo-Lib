#include "mico_qnn.h"
#include "mico_quant.h"
#include "profile.h"
#include "bitnet_cfu.h"

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w) {
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];
    const size_t full_iters = in_features / BNCFU_Q1_FULL_ELEMS;

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
