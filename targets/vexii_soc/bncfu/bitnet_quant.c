#include "mico_quant.h"
#include "profile.h"
#include "bitnet_cfu.h"

#include <string.h>

#ifndef USE_RVF
#include <math.h>
#define roundf2i(x) roundf(x)
#endif

extern float MiCo_absmax(float* x, size_t n);

#ifdef BNCFU_Q8
static float bncfu_scalar_fp32_to_q8_padded(Tensor2D_Q8 *qx, const Tensor2D_F32 *x) {
    const size_t batch_size = x->shape[0];
    const size_t n = x->shape[1];
    const size_t qx_n = qx->shape[1];
    const float scale = 127.0 / MiCo_absmax(x->data, batch_size * n);

    for(size_t b = 0; b < batch_size; ++b) {
        for(size_t i = 0; i < qx_n; ++i) {
            if(i >= n) {
                qx->data[b * qx_n + i] = 0;
            } else {
                qx->data[b * qx_n + i] = (int8_t)(roundf2i(x->data[b * n + i] * scale));
            }
        }
    }
    return 1.0 / scale;
}

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

void MiCo_2D_FP32toQ8(Tensor2D_Q8 *qx, const Tensor2D_F32 *x) {
    const size_t batch_size = x->shape[0];
    const size_t n = x->shape[1];
    const size_t qx_b = qx->shape[0];
    const size_t qx_n = qx->shape[1];

    MiCo_assert(batch_size == qx_b,
        "[Quantization] Batch Size Mismatched!");

    if(qx_n == n) {
        qx->scale = __FP32toQ8(qx->data, x->data, batch_size * n);
        return;
    }

    const size_t packed_count = batch_size * n;
    qbyte *packed = MiCo_alloc(packed_count, MICO_ALIGN);
    if(!packed) {
        qx->scale = bncfu_scalar_fp32_to_q8_padded(qx, x);
        return;
    }

    qx->scale = __FP32toQ8(packed, x->data, packed_count);
    const size_t row_copy = qx_n < n ? qx_n : n;
    for(size_t b = 0; b < qx_b; ++b) {
        memcpy(qx->data + b * qx_n, packed + b * n, row_copy);
        if(qx_n > n) {
            memset(qx->data + b * qx_n + n, 0, qx_n - n);
        }
    }
    MiCo_free(packed);
}

void MiCo_4D_FP32toQ8(Tensor4D_Q8 *qx, const Tensor4D_F32 *x) {
    const size_t batch_size = x->shape[0];
    const size_t n = x->shape[1] * x->shape[2] * x->shape[3];

    qx->scale = __FP32toQ8(qx->data, x->data, batch_size * n);
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
