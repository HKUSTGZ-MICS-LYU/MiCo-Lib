#include "mico_runtime.h"

// Default MatMul Function Pointer Matrix
MatMulFunc MiCo_QMatMul[4][4] = {
    {MiCo_Q1_MatMul,   MiCo_Q1x2_MatMul, MiCo_Q1x4_MatMul, MiCo_Q1x8_MatMul},
    {MiCo_Q2x1_MatMul, MiCo_Q2_MatMul,   MiCo_Q2x4_MatMul, MiCo_Q2x8_MatMul},
    {MiCo_Q4x1_MatMul, MiCo_Q4x2_MatMul, MiCo_Q4_MatMul,   MiCo_Q4x8_MatMul},
    {MiCo_Q8x1_MatMul, MiCo_Q8x2_MatMul, MiCo_Q8x4_MatMul, MiCo_Q8_MatMul},
};

MiCoRuntime MiCo_runtime = {
    .matmul_matrix = MiCo_QMatMul
};

void MiCo_set_runtime(MiCo_MatMul_Opt opt) {
    switch (opt) {
        case MiCo_MatMul_Opt_Default:
            MiCo_runtime.matmul_matrix = MiCo_QMatMul;
            break;
        // case MiCo_MatMul_Opt_Unroll:
        //     MiCo_runtime.matmul_matrix = MiCo_QMatMul_Unroll;
        //     break;
        // case MiCo_MatMul_Opt_LUT:
        //     MiCo_runtime.matmul_matrix = MiCo_QMatMul_LUT;
        //     break;
        default:
            // Invalid option, fallback to default
            MiCo_runtime.matmul_matrix = MiCo_QMatMul;
            break;
    }
}

int qlog(qtype x){
    int result = 0;
    while (x >>= 1) result++;
    return result;
}