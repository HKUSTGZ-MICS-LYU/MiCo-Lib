#include "nn.h"
#include "profile.h"

#include <math.h>
#include <string.h>

extern long ATTN_TIMER;

static inline size_t idx2(size_t i0, size_t i1, size_t d1){
    return i0 * d1 + i1;
}

static inline size_t idx3(size_t i0, size_t i1, size_t i2, size_t d1, size_t d2){
    return (i0 * d1 + i1) * d2 + i2;
}

static inline size_t idx4(size_t i0, size_t i1, size_t i2, size_t i3, size_t d1, size_t d2, size_t d3){
    return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

static void MiCo_softmax_vec(float *dst, const float *src, size_t n){
    float max_val = src[0];
    for (size_t i = 1; i < n; i++){
        if (src[i] > max_val){
            max_val = src[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++){
        dst[i] = expf(src[i] - max_val);
        sum += dst[i];
    }
    for (size_t i = 0; i < n; i++){
        dst[i] /= sum;
    }
}

void MiCo_view3d4d_f32(Tensor4D_F32 *y, const Tensor3D_F32 *x){
    y->data = x->data;
}

void MiCo_flatten3d_f32(Tensor3D_F32 *y, const Tensor4D_F32 *x){
    y->shape[0] = x->shape[0];
    y->shape[1] = x->shape[1];
    y->shape[2] = x->shape[2] * x->shape[3];
    y->data = x->data;
}

void MiCo_repeat3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x,
    const size_t rep0, const size_t rep1, const size_t rep2){
    MiCo_assert(y->shape[0] == x->shape[0] * rep0, "[Repeat3D] dim0 mismatch");
    MiCo_assert(y->shape[1] == x->shape[1] * rep1, "[Repeat3D] dim1 mismatch");
    MiCo_assert(y->shape[2] == x->shape[2] * rep2, "[Repeat3D] dim2 mismatch");

    for (size_t i0 = 0; i0 < y->shape[0]; i0++){
        size_t s0 = i0 % x->shape[0];
        for (size_t i1 = 0; i1 < y->shape[1]; i1++){
            size_t s1 = i1 % x->shape[1];
            for (size_t i2 = 0; i2 < y->shape[2]; i2++){
                size_t s2 = i2 % x->shape[2];
                y->data[idx3(i0, i1, i2, y->shape[1], y->shape[2])] =
                    x->data[idx3(s0, s1, s2, x->shape[1], x->shape[2])];
            }
        }
    }
}

void MiCo_transpose4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, const size_t dim0, const size_t dim1){
    MiCo_assert(dim0 < 4 && dim1 < 4, "[Transpose4D] invalid dims");

    size_t xshape[4] = {x->shape[0], x->shape[1], x->shape[2], x->shape[3]};
    size_t yshape[4] = {y->shape[0], y->shape[1], y->shape[2], y->shape[3]};

    size_t expected[4] = {xshape[0], xshape[1], xshape[2], xshape[3]};
    expected[dim0] = xshape[dim1];
    expected[dim1] = xshape[dim0];
    MiCo_assert(yshape[0] == expected[0] && yshape[1] == expected[1] &&
                yshape[2] == expected[2] && yshape[3] == expected[3],
                "[Transpose4D] output shape mismatch");

    for (size_t i0 = 0; i0 < yshape[0]; i0++){
        for (size_t i1 = 0; i1 < yshape[1]; i1++){
            for (size_t i2 = 0; i2 < yshape[2]; i2++){
                for (size_t i3 = 0; i3 < yshape[3]; i3++){
                    size_t out_coord[4] = {i0, i1, i2, i3};
                    size_t in_coord[4] = {i0, i1, i2, i3};
                    in_coord[dim0] = out_coord[dim1];
                    in_coord[dim1] = out_coord[dim0];
                    size_t out_idx = idx4(i0, i1, i2, i3, yshape[1], yshape[2], yshape[3]);
                    size_t in_idx = idx4(in_coord[0], in_coord[1], in_coord[2], in_coord[3],
                                         xshape[1], xshape[2], xshape[3]);
                    y->data[out_idx] = x->data[in_idx];
                }
            }
        }
    }
}

void MiCo_getitem3d_to2d_f32(Tensor2D_F32 *y, const Tensor3D_F32 *x, const size_t index1){
    MiCo_assert(index1 < x->shape[1], "[GetItem3D] index out of range");
    MiCo_assert(y->shape[0] == x->shape[0] && y->shape[1] == x->shape[2], "[GetItem3D] output shape mismatch");

    for (size_t b = 0; b < x->shape[0]; b++){
        for (size_t f = 0; f < x->shape[2]; f++){
            y->data[idx2(b, f, y->shape[1])] = x->data[idx3(b, index1, f, x->shape[1], x->shape[2])];
        }
    }
}

void MiCo_im2word(Tensor3D_F32 *y, const Tensor4D_F32 *x, const size_t patch){
    const size_t batch = x->shape[0];
    #ifdef USE_ALT_LAYOUT
    const size_t in_h = x->shape[1];
    const size_t in_w = x->shape[2];
    const size_t in_c = x->shape[3];
    #else
    const size_t in_c = x->shape[1];
    const size_t in_h = x->shape[2];
    const size_t in_w = x->shape[3];
    #endif

    MiCo_assert(in_h % patch == 0 && in_w % patch == 0, "[Im2Word] image size must be divisible by patch count");
    const size_t patch_size_h = in_h / patch;
    const size_t patch_size_w = in_w / patch;
    const size_t n_tokens = patch * patch;
    const size_t feat = patch_size_h * patch_size_w * in_c;

    MiCo_assert(y->shape[0] == batch && y->shape[1] == n_tokens && y->shape[2] == feat,
                "[Im2Word] output shape mismatch");

    for (size_t b = 0; b < batch; b++){
        for (size_t ph = 0; ph < patch; ph++){
            for (size_t pw = 0; pw < patch; pw++){
                const size_t token = ph * patch + pw;
                for (size_t ih = 0; ih < patch_size_h; ih++){
                    for (size_t iw = 0; iw < patch_size_w; iw++){
                        for (size_t c = 0; c < in_c; c++){
                            size_t src_h = ph * patch_size_h + ih;
                            size_t src_w = pw * patch_size_w + iw;
                            size_t src_idx = OFFSET_4D(b, c, src_h, src_w, batch, in_c, in_h, in_w);
                            size_t f = (ih * patch_size_w + iw) * in_c + c;
                            size_t dst_idx = idx3(b, token, f, n_tokens, feat);
                            y->data[dst_idx] = x->data[src_idx];
                        }
                    }
                }
            }
        }
    }
}

void MiCo_softmax2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const int dim){
    int real_dim = dim < 0 ? dim + 2 : dim;
    MiCo_assert(real_dim == 1, "[Softmax2D] only last-dim softmax is supported");
    for (size_t i = 0; i < x->shape[0]; i++){
        size_t base = i * x->shape[1];
        MiCo_softmax_vec(y->data + base, x->data + base, x->shape[1]);
    }
}

void MiCo_softmax3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, const int dim){
    int real_dim = dim < 0 ? dim + 3 : dim;
    MiCo_assert(real_dim == 2, "[Softmax3D] only last-dim softmax is supported");
    for (size_t b = 0; b < x->shape[0]; b++){
        for (size_t s = 0; s < x->shape[1]; s++){
            size_t base = idx3(b, s, 0, x->shape[1], x->shape[2]);
            MiCo_softmax_vec(y->data + base, x->data + base, x->shape[2]);
        }
    }
}

void MiCo_softmax4d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, const int dim){
    int real_dim = dim < 0 ? dim + 4 : dim;
    MiCo_assert(real_dim == 3, "[Softmax4D] only last-dim softmax is supported");
    for (size_t b = 0; b < x->shape[0]; b++){
        for (size_t h = 0; h < x->shape[1]; h++){
            for (size_t i = 0; i < x->shape[2]; i++){
                size_t base = idx4(b, h, i, 0, x->shape[1], x->shape[2], x->shape[3]);
                MiCo_softmax_vec(y->data + base, x->data + base, x->shape[3]);
            }
        }
    }
}

void MiCo_div2d_scalar_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const float scalar){
    const size_t n = x->shape[0] * x->shape[1];
    for (size_t i = 0; i < n; i++){
        y->data[i] = x->data[i] / scalar;
    }
}

void MiCo_div3d_scalar_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, const float scalar){
    const size_t n = x->shape[0] * x->shape[1] * x->shape[2];
    for (size_t i = 0; i < n; i++){
        y->data[i] = x->data[i] / scalar;
    }
}

void MiCo_div4d_scalar_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, const float scalar){
    const size_t n = x->shape[0] * x->shape[1] * x->shape[2] * x->shape[3];
    for (size_t i = 0; i < n; i++){
        y->data[i] = x->data[i] / scalar;
    }
}

void MiCo_gelu2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x){
    const size_t n = x->shape[0] * x->shape[1];
    const float inv_sqrt2 = 0.7071067811865475f;
    for (size_t i = 0; i < n; i++){
        float xv = x->data[i];
        y->data[i] = 0.5f * xv * (1.0f + erff(xv * inv_sqrt2));
    }
}

void MiCo_gelu3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x){
    const size_t n = x->shape[0] * x->shape[1] * x->shape[2];
    const float inv_sqrt2 = 0.7071067811865475f;
    for (size_t i = 0; i < n; i++){
        float xv = x->data[i];
        y->data[i] = 0.5f * xv * (1.0f + erff(xv * inv_sqrt2));
    }
}

void MiCo_einsum_bhif_bhjf_bhij_f32(Tensor4D_F32 *y, const Tensor4D_F32 *q, const Tensor4D_F32 *k){
    const size_t B = q->shape[0];
    const size_t H = q->shape[1];
    const size_t I = q->shape[2];
    const size_t F = q->shape[3];
    const size_t J = k->shape[2];

    MiCo_assert(k->shape[0] == B && k->shape[1] == H && k->shape[3] == F, "[Einsum1] input shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == H && y->shape[2] == I && y->shape[3] == J, "[Einsum1] output shape mismatch");

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            for (size_t i = 0; i < I; i++){
                for (size_t j = 0; j < J; j++){
                    float sum = 0.0f;
                    for (size_t f = 0; f < F; f++){
                        size_t q_idx = idx4(b, h, i, f, H, I, F);
                        size_t k_idx = idx4(b, h, j, f, H, J, F);
                        sum += q->data[q_idx] * k->data[k_idx];
                    }
                    y->data[idx4(b, h, i, j, H, I, J)] = sum;
                }
            }
        }
    }
}

void MiCo_einsum_bhij_bhjf_bihf_f32(Tensor4D_F32 *y, const Tensor4D_F32 *score, const Tensor4D_F32 *v){
    const size_t B = score->shape[0];
    const size_t H = score->shape[1];
    const size_t I = score->shape[2];
    const size_t J = score->shape[3];
    const size_t F = v->shape[3];

    MiCo_assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == J, "[Einsum2] input shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == I && y->shape[2] == H && y->shape[3] == F, "[Einsum2] output shape mismatch");

    for (size_t b = 0; b < B; b++){
        for (size_t i = 0; i < I; i++){
            for (size_t h = 0; h < H; h++){
                for (size_t f = 0; f < F; f++){
                    float sum = 0.0f;
                    for (size_t j = 0; j < J; j++){
                        size_t s_idx = idx4(b, h, i, j, H, I, J);
                        size_t v_idx = idx4(b, h, j, f, H, J, F);
                        sum += score->data[s_idx] * v->data[v_idx];
                    }
                    y->data[idx4(b, i, h, f, I, H, F)] = sum;
                }
            }
        }
    }
}

void MiCo_ViT_attention_f32(
    Tensor4D_F32 *y,
    const Tensor4D_F32 *q,
    const Tensor4D_F32 *k,
    const Tensor4D_F32 *v,
    const float scale
){
    const size_t B = q->shape[0];
    const size_t H = q->shape[1];
    const size_t I = q->shape[2];
    const size_t F = q->shape[3];
    const size_t J = k->shape[2];

    MiCo_assert(k->shape[0] == B && k->shape[1] == H && k->shape[3] == F, "[Attention] k shape mismatch");
    MiCo_assert(v->shape[0] == B && v->shape[1] == H && v->shape[2] == J && v->shape[3] == F, "[Attention] v shape mismatch");
    MiCo_assert(y->shape[0] == B && y->shape[1] == I && y->shape[2] == H && y->shape[3] == F, "[Attention] y shape mismatch");
    MiCo_assert(scale != 0.0f, "[Attention] scale must be non-zero");

    float *scores = (float *)malloc(J * sizeof(float));
    MiCo_assert(scores != NULL, "[Attention] failed to allocate scores buffer");

    long start_time = MiCo_time();

    for (size_t b = 0; b < B; b++){
        for (size_t h = 0; h < H; h++){
            for (size_t i = 0; i < I; i++){
                for (size_t j = 0; j < J; j++){
                    float sum = 0.0f;
                    for (size_t f = 0; f < F; f++){
                        size_t q_idx = idx4(b, h, i, f, H, I, F);
                        size_t k_idx = idx4(b, h, j, f, H, J, F);
                        sum += q->data[q_idx] * k->data[k_idx];
                    }
                    scores[j] = sum / scale;
                }

                MiCo_softmax_vec(scores, scores, J);

                for (size_t f = 0; f < F; f++){
                    float out_sum = 0.0f;
                    for (size_t j = 0; j < J; j++){
                        size_t v_idx = idx4(b, h, j, f, H, J, F);
                        out_sum += scores[j] * v->data[v_idx];
                    }
                    y->data[idx4(b, i, h, f, I, H, F)] = out_sum;
                }
            }
        }
    }
    ATTN_TIMER += MiCo_time() - start_time;

    free(scores);
}
