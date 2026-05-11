#include "nn.h"

// Mean reduction (keepdim=False): 2D input -> 1D output
void MiCo_mean1d_f32(Tensor1D_F32 *y, const Tensor2D_F32 *x, const size_t dim) {
    size_t size0 = x->shape[0];
    size_t size1 = x->shape[1];

    if (dim == 0 || dim == -2) {
        y->shape[0] = size1;
        for (size_t j = 0; j < size1; j++) {
            float sum = 0;
            for (size_t i = 0; i < size0; i++) {
                sum += x->data[i * size1 + j];
            }
            y->data[j] = sum / (float)size0;
        }
    } else {
        // dim == 1 or dim == -1
        y->shape[0] = size0;
        for (size_t i = 0; i < size0; i++) {
            float sum = 0;
            for (size_t j = 0; j < size1; j++) {
                sum += x->data[i * size1 + j];
            }
            y->data[i] = sum / (float)size1;
        }
    }
}

// Mean reduction (keepdim=False): 3D input -> 2D output
void MiCo_mean2d_f32(Tensor2D_F32 *y, const Tensor3D_F32 *x, const size_t dim) {
    size_t d0 = x->shape[0];
    size_t d1 = x->shape[1];
    size_t d2 = x->shape[2];

    if (dim == 0 || dim == -3) {
        y->shape[0] = d1;
        y->shape[1] = d2;
        for (size_t i = 0; i < d1; i++) {
            for (size_t j = 0; j < d2; j++) {
                float sum = 0;
                for (size_t k = 0; k < d0; k++) {
                    sum += x->data[k * d1 * d2 + i * d2 + j];
                }
                y->data[i * d2 + j] = sum / (float)d0;
            }
        }
    } else if (dim == 1 || dim == -2) {
        y->shape[0] = d0;
        y->shape[1] = d2;
        for (size_t k = 0; k < d0; k++) {
            for (size_t j = 0; j < d2; j++) {
                float sum = 0;
                for (size_t i = 0; i < d1; i++) {
                    sum += x->data[k * d1 * d2 + i * d2 + j];
                }
                y->data[k * d2 + j] = sum / (float)d1;
            }
        }
    } else {
        // dim == 2 or dim == -1
        y->shape[0] = d0;
        y->shape[1] = d1;
        for (size_t k = 0; k < d0; k++) {
            for (size_t i = 0; i < d1; i++) {
                float sum = 0;
                for (size_t j = 0; j < d2; j++) {
                    sum += x->data[k * d1 * d2 + i * d2 + j];
                }
                y->data[k * d1 + i] = sum / (float)d2;
            }
        }
    }
}

// Mean reduction (keepdim=False): 4D input -> 3D output
void MiCo_mean3d_f32(Tensor3D_F32 *y, const Tensor4D_F32 *x, const size_t dim) {
    size_t d0 = x->shape[0];
    size_t d1 = x->shape[1];
    size_t d2 = x->shape[2];
    size_t d3 = x->shape[3];
    size_t plane = d2 * d3;
    size_t block = d1 * plane;

    if (dim == 0 || dim == -4) {
        y->shape[0] = d1; y->shape[1] = d2; y->shape[2] = d3;
        for (size_t i = 0; i < d1; i++) {
            for (size_t j = 0; j < d2; j++) {
                for (size_t k = 0; k < d3; k++) {
                    float sum = 0;
                    for (size_t b = 0; b < d0; b++) {
                        sum += x->data[b * block + i * plane + j * d3 + k];
                    }
                    y->data[i * plane + j * d3 + k] = sum / (float)d0;
                }
            }
        }
    } else if (dim == 1 || dim == -3) {
        y->shape[0] = d0; y->shape[1] = d2; y->shape[2] = d3;
        for (size_t b = 0; b < d0; b++) {
            for (size_t j = 0; j < d2; j++) {
                for (size_t k = 0; k < d3; k++) {
                    float sum = 0;
                    for (size_t i = 0; i < d1; i++) {
                        sum += x->data[b * block + i * plane + j * d3 + k];
                    }
                    y->data[b * plane + j * d3 + k] = sum / (float)d1;
                }
            }
        }
    } else if (dim == 2 || dim == -2) {
        y->shape[0] = d0; y->shape[1] = d1; y->shape[2] = d3;
        size_t out_plane = d1 * d3;
        for (size_t b = 0; b < d0; b++) {
            for (size_t i = 0; i < d1; i++) {
                for (size_t k = 0; k < d3; k++) {
                    float sum = 0;
                    for (size_t j = 0; j < d2; j++) {
                        sum += x->data[b * block + i * plane + j * d3 + k];
                    }
                    y->data[b * out_plane + i * d3 + k] = sum / (float)d2;
                }
            }
        }
    } else {
        // dim == 3 or dim == -1
        y->shape[0] = d0; y->shape[1] = d1; y->shape[2] = d2;
        for (size_t b = 0; b < d0; b++) {
            for (size_t i = 0; i < d1; i++) {
                for (size_t j = 0; j < d2; j++) {
                    float sum = 0;
                    for (size_t k = 0; k < d3; k++) {
                        sum += x->data[b * block + i * plane + j * d3 + k];
                    }
                    y->data[b * d1 * d2 + i * d2 + j] = sum / (float)d3;
                }
            }
        }
    }
}

// Mean with keepdim=True: 2D input -> 2D output (one dimension becomes 1)
void MiCo_meankp2d_f32(Tensor2D_F32 *y, const Tensor2D_F32 *x, const size_t dim) {
    size_t size0 = x->shape[0];
    size_t size1 = x->shape[1];

    if (dim == 0 || dim == -2) {
        y->shape[0] = 1;
        y->shape[1] = size1;
        for (size_t j = 0; j < size1; j++) {
            float sum = 0;
            for (size_t i = 0; i < size0; i++) {
                sum += x->data[i * size1 + j];
            }
            y->data[j] = sum / (float)size0;
        }
    } else {
        // dim == 1 or dim == -1
        y->shape[0] = size0;
        y->shape[1] = 1;
        for (size_t i = 0; i < size0; i++) {
            float sum = 0;
            for (size_t j = 0; j < size1; j++) {
                sum += x->data[i * size1 + j];
            }
            y->data[i] = sum / (float)size1;
        }
    }
}

// Mean with keepdim=True: 3D input -> 3D output (one dimension becomes 1)
void MiCo_meankp3d_f32(Tensor3D_F32 *y, const Tensor3D_F32 *x, const size_t dim) {
    size_t d0 = x->shape[0];
    size_t d1 = x->shape[1];
    size_t d2 = x->shape[2];
    size_t stride1 = d1 * d2;

    if (dim == 0 || dim == -3) {
        y->shape[0] = 1; y->shape[1] = d1; y->shape[2] = d2;
        for (size_t i = 0; i < d1; i++) {
            for (size_t j = 0; j < d2; j++) {
                float sum = 0;
                for (size_t k = 0; k < d0; k++) {
                    sum += x->data[k * stride1 + i * d2 + j];
                }
                y->data[i * d2 + j] = sum / (float)d0;
            }
        }
    } else if (dim == 1 || dim == -2) {
        y->shape[0] = d0; y->shape[1] = 1; y->shape[2] = d2;
        for (size_t k = 0; k < d0; k++) {
            for (size_t j = 0; j < d2; j++) {
                float sum = 0;
                for (size_t i = 0; i < d1; i++) {
                    sum += x->data[k * stride1 + i * d2 + j];
                }
                y->data[k * d2 + j] = sum / (float)d1;
            }
        }
    } else {
        // dim == 2 or dim == -1
        y->shape[0] = d0; y->shape[1] = d1; y->shape[2] = 1;
        for (size_t k = 0; k < d0; k++) {
            for (size_t i = 0; i < d1; i++) {
                float sum = 0;
                for (size_t j = 0; j < d2; j++) {
                    sum += x->data[k * stride1 + i * d2 + j];
                }
                y->data[k * d1 + i] = sum / (float)d2;
            }
        }
    }
}
