#ifndef __MICO_RUNTIME_H
#define __MICO_RUNTIME_H

#include "mico_qnn.h"

#define MAX_QTYPE_LOG2 (3)  // log2(8) = 3

typedef void (*MatMulFunc)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*);

typedef enum {
    MiCo_MatMul_Opt_Default = 0,
    MiCo_MatMul_Opt_Unroll = 1,
    MiCo_MatMul_Opt_LUT = 2
} MiCo_MatMul_Opt;

typedef struct {
    // A Function Pointer to the Current MatMul Implementation
    MatMulFunc (*matmul_matrix)[MAX_QTYPE_LOG2+1];

} MiCoRuntime;

void MiCo_set_runtime(MiCo_MatMul_Opt opt);

// Helper function to compute log2 of qtype
int qlog(qtype x);

#endif