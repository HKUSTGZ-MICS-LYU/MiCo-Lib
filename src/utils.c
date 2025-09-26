#include "nn.h"

void MiCo_print_tensor2d_f32(const Tensor2D_F32 *x){
    for (int i=0; i < x->shape[0]; i++){
        printf("Batch %d: \n", i);
        for (int j=0; j < x->shape[1]; j++){
            printf("%.4f ", x->data[i*x->shape[1] + j]);
        }
        printf("\n");
    }
}

void MiCo_print_tensor3d_f32(const Tensor3D_F32 *x){
    for (int i=0; i < x->shape[0]; i++){
        printf("Batch %d: \n", i);
        for (int j=0; j < x->shape[1]; j++){
            for (int k=0; k < x->shape[2]; k++){
                printf("%.4f ", x->data[i*x->shape[1]*x->shape[2] + j*x->shape[2] + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void MiCo_print_tensor4d_f32(const Tensor4D_F32 *x){
    for (int i=0; i < x->shape[0]; i++){
        printf("Batch %d: \n", i);
        for (int j=0; j < x->shape[1]; j++){
            for (int k=0; k < x->shape[2]; k++){
                for (int l=0; l < x->shape[3]; l++){
                    printf("%.4f ", x->data[i*x->shape[1]*x->shape[2]*x->shape[3] + j*x->shape[2]*x->shape[3] + k*x->shape[3] + l]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void MiCo_assert(const int condition, const char *message){
    if (!condition){
        printf("%s\n", message);
        exit(1);
    }
}

void* MiCo_alloc(const size_t size, const int align){
    if (size == 0) return NULL;
    size_t a = (align > 0) ? (size_t)align : sizeof(void*);
    // require power-of-two alignment
    if ((a & (a - 1)) != 0) return NULL;

    // extra space for alignment padding + to store the raw pointer
    void *raw = malloc(size + a - 1 + sizeof(void*));
    if (!raw) return NULL;

    uintptr_t addr = (uintptr_t)raw + sizeof(void*);
    uintptr_t aligned_addr = (addr + (a - 1)) & ~(uintptr_t)(a - 1);
    void **aligned = (void**)aligned_addr;

    // store the original allocation just before the aligned block
    aligned[-1] = raw;
    return aligned;
}

void MiCo_free(void *ptr){
    if (!ptr) return;
    void *raw = ((void**)ptr)[-1];
    free(raw);
}