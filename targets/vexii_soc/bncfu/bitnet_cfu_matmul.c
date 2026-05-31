#include "mico_qnn.h"
#include "mico_quant.h"
#include "profile.h"
#include "bitnet_cfu.h"

#ifndef USE_RVF
#include <math.h>
#define roundf2i(x) roundf(x)
#endif

extern float MiCo_absmax(float* x, size_t n);

#ifdef BNCFU_Q8
float __FP32toQ8(qbyte* qx, float* x, size_t n) {
    long start = MiCo_time();
    float scale = 127.0 / MiCo_absmax(x, n);
    const float absmax = 127.0 / scale;
    const uint32_t absmax_bits = bncfu_float_bits(absmax);
    int i = 0;

    bncfu_enable();
    bncfu_dma_fence();

    for(; i + BNCFU_FP32_ELEMS <= (int)n; i += BNCFU_FP32_ELEMS) {
        bncfu_load(0, x + i);
        for(unsigned int chunk = 0; chunk < BNCFU_Q8_QUANT_CHUNKS; ++chunk) {
            const uint32_t packed = bncfu_q8(absmax_bits, 0, chunk);
            const int base = i + (int)(chunk * BNCFU_Q8_QUANT_ELEMS);
            for(int j = 0; j < BNCFU_Q8_QUANT_BYTES; ++j) {
                qx[base + j] = (qbyte)(packed >> (j * 8));
            }
        }
    }

    for(; i < (int)n; ++i) {
        qx[i] = (int8_t)(roundf2i(x[i] * scale));
    }

    QUANT_TIMER += MiCo_time() - start;
    return 1.0 / scale;
}
#endif

#ifdef BNCFU_Q2T
float __FP32toQ2T(qbyte* qx, float* x, size_t n) {
    long start = MiCo_time();
    float scale = 1.0 / MiCo_absmax(x, n);
    const float absmax = 1.0 / scale;
    const uint32_t absmax_bits = bncfu_float_bits(absmax);
    int i = 0;

    bncfu_enable();
    bncfu_dma_fence();

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
