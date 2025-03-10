#include "nn.h"

void MiCo_print_tensor2d_f32(const Tensor2D_F32 *x){
    for (int i=0; i < x->shape[0]; i++){
        printf("Batch %d: \n", i);
        for (int j=0; j < x->shape[1]; j++){
            printf("%f ", x->data[i*x->shape[1] + j]);
        }
        printf("\n");
    }
}

void MiCo_print_tensor3d_f32(const Tensor3D_F32 *x){
    for (int i=0; i < x->shape[0]; i++){
        printf("Batch %d: \n", i);
        for (int j=0; j < x->shape[1]; j++){
            for (int k=0; k < x->shape[2]; k++){
                printf("%f ", x->data[i*x->shape[1]*x->shape[2] + j*x->shape[2] + k]);
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
                    printf("%f ", x->data[i*x->shape[1]*x->shape[2]*x->shape[3] + j*x->shape[2]*x->shape[3] + k*x->shape[3] + l]);
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

// TODO: Remove Dummy Implementation
void MiCo_batchnorm2d_f32(Tensor4D_F32 *y, const Tensor4D_F32 *x, const float eps){
    MiCo_CONNECT(y,x);
}