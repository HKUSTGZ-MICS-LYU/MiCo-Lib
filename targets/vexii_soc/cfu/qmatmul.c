#include "mico_qnn.h"

// SIMD accelerated Implementation of MatMul using VPU
// Requires ISA support for CFU (Custom Function Unit)

// VLEN: Vector register length in bits (default 128)
// This should be passed via compiler flags (e.g., -DVLEN=64)
// Supported values: 64, 128, 256, 512
#ifndef VLEN
#define VLEN 128
#endif

// Compile-time validation: VLEN must be at least 64 bits
// (minimum requirement for the VPU implementation)
#if VLEN < 64
#error "VLEN must be at least 64 bits for VPU operations"
#endif

// Calculate step size and elements per vector operation based on VLEN and precision
#define VPU_STEP_BYTES (VLEN / 8)  // Number of bytes per vector load
#define VPU_ELEMS_Q8 (VLEN / 8)    // Elements per vector op for 8-bit
#define VPU_ELEMS_Q4 (VLEN / 4)    // Elements per vector op for 4-bit
#define VPU_ELEMS_Q2 (VLEN / 2)    // Elements per vector op for 2-bit
#define VPU_ELEMS_Q1 (VLEN / 1)    // Elements per vector op for 1-bit

// Inline macro to enable CFU
#define cfu_enable() do { \
    __asm__ volatile( \
        "li t1, 0x80000000\n\t" \
        "csrs 0xBC0, t1" \
        ::: "t1" \
    ); \
} while(0)

// Inline macro for fence.i instruction
#define fence_i() __asm__ volatile("fence.i")

// VPU Configuration macro - sets precision for the VPU operation
// vpu_config(qa, qb) - qa: larger precision, qb: smaller precision
// Note: For asymmetric operations, always pass max(prec1, prec2) first, then min(prec1, prec2)
// Encoding: opcode=0x0B, func3=0x2, func7=0x00
// qa goes in bits 15-19 (rs1 position), qb goes in bits 20-24 (rs2 position)
#define vpu_config(qa, qb) do { \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | (%0 << 15) | (%1 << 20) | (0x2 << 12) | (0 << 25))" \
        :: "i"(qa), "i"(qb) \
    ); \
} while(0)

// VPU Load macro - loads vector from memory into vector register
// Encoding: opcode=0x0B, func3=0x4, func7=0x00
// bits 20-24 (rs2 position): destination vector register id (0 or 1)
// bits 15-19 (rs1 position): source address register
// We need to use inline assembly with proper register constraints
#define vpu_load_v0(addr) do { \
    register uintptr_t _addr_reg asm("a5") = (uintptr_t)(addr); \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | (15 << 15) | (0 << 20) | (0x4 << 12) | (0 << 25))" \
        : "+r"(_addr_reg) :: "memory" \
    ); \
} while(0)

#define vpu_load_v1(addr) do { \
    register uintptr_t _addr_reg asm("a6") = (uintptr_t)(addr); \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | (16 << 15) | (1 << 20) | (0x4 << 12) | (0 << 25))" \
        : "+r"(_addr_reg) :: "memory" \
    ); \
} while(0)

// VPU Dot Product macro - computes dot product of two vectors
// Encoding: opcode=0x0B, func3=0x1, func7=0x00
// bits 7-11 (rd position): scalar result register
// bits 15-19 (rs1 position): vector register id for first operand (0 or 1)
// bits 20-24 (rs2 position): vector register id for second operand (0 or 1)
// The result register must be specified in the instruction encoding
#define vpu_vdot_v0_v1() ({ \
    register int32_t _result asm("t0"); \
    __asm__ volatile( \
        ".word (0x0B | (5 << 7) | (0 << 15) | (1 << 20) | (0x1 << 12) | (0 << 25))" \
        : "=r"(_result) \
    ); \
    _result; \
})

// Alternative dot product with reversed operands (v1, v0)
#define vpu_vdot_v1_v0() ({ \
    register int32_t _result asm("t0"); \
    __asm__ volatile( \
        ".word (0x0B | (5 << 7) | (1 << 15) | (0 << 20) | (0x1 << 12) | (0 << 25))" \
        : "=r"(_result) \
    ); \
    _result; \
})


void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 8);

    // VPU_ELEMS_Q8 = VLEN/8 (number of 8-bit elements per vector op)
    // For 8-bit data in qword (32-bit), each qword has 4 elements
    // So we step by VPU_ELEMS_Q8/4 qwords per iteration
    const size_t elems_per_iter = VPU_ELEMS_Q8;
    const size_t qword_step = elems_per_iter / 4;  // qwords per vector load
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * qword_step]);
                vpu_load_v1(&w_ptr[k * qword_step]);
                sum += vpu_vdot_v0_v1();
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                sum += x->data[i * in_features + k] *
                       w->data[j * in_features + k];
            }
            O[i * out_features + j] = sum;
        }
    }
}

void MiCo_Q4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(4, 4);

    int8_t temp_a;
    int8_t temp_w;

    // VPU_ELEMS_Q4 = VLEN/4 (number of 4-bit elements per vector op)
    // For 4-bit data in qword (32-bit), each qword has 8 elements
    // So we step by VPU_ELEMS_Q4/8 qwords per iteration
    const size_t elems_per_iter = VPU_ELEMS_Q4;
    const size_t qword_step = elems_per_iter / 8;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 2);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * qword_step]);
                vpu_load_v1(&w_ptr[k * qword_step]);
                acc_sum += vpu_vdot_v0_v1();
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                temp_w = w->data[(j * in_features + k)/2];
                temp_w = EXTRACT_4BIT(temp_w, k & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                temp_a = x->data[(i * in_features + k)/2];
                temp_a = EXTRACT_4BIT(temp_a, k & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(2, 2);

    int8_t temp_a;
    int8_t temp_w;

    // VPU_ELEMS_Q2 = VLEN/2 (number of 2-bit elements per vector op)
    // For 2-bit data in qword (32-bit), each qword has 16 elements
    // So we step by VPU_ELEMS_Q2/16 qwords per iteration
    const size_t elems_per_iter = VPU_ELEMS_Q2;
    const size_t qword_step = elems_per_iter / 16;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 4);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * qword_step]);
                vpu_load_v1(&w_ptr[k * qword_step]);
                acc_sum += vpu_vdot_v0_v1();
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                temp_w = w->data[(j * in_features + k)/4];
                temp_w = EXTRACT_2BIT(temp_w, k & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + k)/4];
                temp_a = EXTRACT_2BIT(temp_a, k & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(1, 1);

    int8_t temp_a;
    int8_t temp_w;

    // VPU_ELEMS_Q1 = VLEN/1 (number of 1-bit elements per vector op)
    // For 1-bit data in qword (32-bit), each qword has 32 elements
    // So we step by VPU_ELEMS_Q1/32 qwords per iteration
    const size_t elems_per_iter = VPU_ELEMS_Q1;
    const size_t qword_step = elems_per_iter / 32;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 8);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * qword_step]);
                vpu_load_v1(&w_ptr[k * qword_step]);
                acc_sum += vpu_vdot_v0_v1();
            }
            // Handle remaining elements
            for (size_t k = in_features - remain; k < in_features; k++) {
                temp_w = w->data[(j * in_features + k)/8];
                temp_w = EXTRACT_BIT(temp_w, k & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + k)/8];
                temp_a = EXTRACT_BIT(temp_a, k & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q8x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 4);

    int8_t temp_w;
    // For Q8x4: activation is 8-bit, weight is 4-bit
    // VPU load processes VPU_STEP_BYTES bytes at a time
    // Activation elements per VPU load: VPU_ELEMS_Q8
    // Weight elements per VPU load: VPU_STEP_BYTES * 2 (each byte has 2 x 4-bit)
    // Ratio: 8/4 = 2 activation loads per weight load
    const size_t act_elems_per_load = VPU_ELEMS_Q8;
    const size_t ratio = 8 / 4;  // activation_prec / weight_prec
    const size_t act_qword_step = act_elems_per_load / 4;
    const size_t wt_qword_step = (act_elems_per_load * ratio) / 8;  // weight has 8 elements per qword
    const size_t elems_per_iter = act_elems_per_load * ratio;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 2);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v1(&w_ptr[k * wt_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v0(&a_ptr[k * ratio * act_qword_step + r * act_qword_step]);
                    acc_sum += vpu_vdot_v0_v1();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/2];
                temp_w = (t & 1) ? (temp_w >> 4) : (temp_w & 0x0F);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc_sum += x->data[i * in_features + t] * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q8x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 2);

    int8_t temp_w;
    // For Q8x2: activation is 8-bit, weight is 2-bit
    // Ratio: 8/2 = 4 activation loads per weight load
    const size_t act_elems_per_load = VPU_ELEMS_Q8;
    const size_t ratio = 8 / 2;
    const size_t act_qword_step = act_elems_per_load / 4;
    const size_t wt_qword_step = (act_elems_per_load * ratio) / 16;  // weight has 16 elements per qword
    const size_t elems_per_iter = act_elems_per_load * ratio;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 4);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v1(&w_ptr[k * wt_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v0(&a_ptr[k * ratio * act_qword_step + r * act_qword_step]);
                    acc_sum += vpu_vdot_v0_v1();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/4];
                temp_w = (temp_w >> (((j * in_features + t) & 0b11) << 1)) & 0x03;
                acc_sum += AMUX_2BIT(temp_w, x->data[i * in_features + t]);
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q8x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 1);

    int8_t temp_w;
    // For Q8x1: activation is 8-bit, weight is 1-bit
    // Ratio: 8/1 = 8 activation loads per weight load
    const size_t act_elems_per_load = VPU_ELEMS_Q8;
    const size_t ratio = 8 / 1;
    const size_t act_qword_step = act_elems_per_load / 4;
    const size_t wt_qword_step = (act_elems_per_load * ratio) / 32;  // weight has 32 elements per qword
    const size_t elems_per_iter = act_elems_per_load * ratio;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 8);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v1(&w_ptr[k * wt_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v0(&a_ptr[k * ratio * act_qword_step + r * act_qword_step]);
                    acc_sum += vpu_vdot_v0_v1();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/8];
                temp_w = EXTRACT_BIT(temp_w, (j * in_features + t) & 0b111);
                acc_sum += AMUX_1BIT(temp_w, x->data[i * in_features + t]);
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q4x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(4, 2);

    int8_t temp_a;
    int8_t temp_w;

    // For Q4x2: activation is 4-bit, weight is 2-bit
    // Ratio: 4/2 = 2 activation loads per weight load
    const size_t act_elems_per_load = VPU_ELEMS_Q4;
    const size_t ratio = 4 / 2;
    const size_t act_qword_step = act_elems_per_load / 8;  // 4-bit has 8 elements per qword
    const size_t wt_qword_step = (act_elems_per_load * ratio) / 16;  // 2-bit has 16 elements per qword
    const size_t elems_per_iter = act_elems_per_load * ratio;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 4);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v1(&w_ptr[k * wt_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v0(&a_ptr[k * ratio * act_qword_step + r * act_qword_step]);
                    acc_sum += vpu_vdot_v0_v1();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/4];
                temp_w = EXTRACT_2BIT(temp_w, t & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + t)/2];
                temp_a = EXTRACT_4BIT(temp_a, t & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q4x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(4, 1);

    int8_t temp_a;
    int8_t temp_w;

    // For Q4x1: activation is 4-bit, weight is 1-bit
    // Ratio: 4/1 = 4 activation loads per weight load
    const size_t act_elems_per_load = VPU_ELEMS_Q4;
    const size_t ratio = 4 / 1;
    const size_t act_qword_step = act_elems_per_load / 8;  // 4-bit has 8 elements per qword
    const size_t wt_qword_step = (act_elems_per_load * ratio) / 32;  // 1-bit has 32 elements per qword
    const size_t elems_per_iter = act_elems_per_load * ratio;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 8);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v1(&w_ptr[k * wt_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v0(&a_ptr[k * ratio * act_qword_step + r * act_qword_step]);
                    acc_sum += vpu_vdot_v0_v1();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/8];
                temp_w = EXTRACT_BIT(temp_w, t & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + t)/2];
                temp_a = EXTRACT_4BIT(temp_a, t & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2x1_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(2, 1);

    int8_t temp_a;
    int8_t temp_w;

    // For Q2x1: activation is 2-bit, weight is 1-bit
    // Ratio: 2/1 = 2 activation loads per weight load
    const size_t act_elems_per_load = VPU_ELEMS_Q2;
    const size_t ratio = 2 / 1;
    const size_t act_qword_step = act_elems_per_load / 16;  // 2-bit has 16 elements per qword
    const size_t wt_qword_step = (act_elems_per_load * ratio) / 32;  // 1-bit has 32 elements per qword
    const size_t elems_per_iter = act_elems_per_load * ratio;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 8);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v1(&w_ptr[k * wt_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v0(&a_ptr[k * ratio * act_qword_step + r * act_qword_step]);
                    acc_sum += vpu_vdot_v0_v1();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_w = w->data[(j * in_features + t)/8];
                temp_w = EXTRACT_BIT(temp_w, t & 0b111);
                temp_w = BIT_TO_INT8(temp_w);
                temp_a = x->data[(i * in_features + t)/4];
                temp_a = EXTRACT_2BIT(temp_a, t & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}


// -----------------------------------------------------------------------------
// Reversed precision functions (weight precision > activation precision)
// In these cases, for vpu_VDOT, the operands are swapped (v1, v0) instead of (v0, v1)
// Note: vpu_config still uses (larger_prec, smaller_prec) ordering per hardware requirement
// -----------------------------------------------------------------------------

void MiCo_Q4x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 4);

    int8_t temp_a;
    // For Q4x8: activation is 4-bit, weight is 8-bit
    // Ratio: 8/4 = 2 weight loads per activation load
    const size_t act_elems_per_load = VPU_ELEMS_Q4;
    const size_t ratio = 8 / 4;
    const size_t act_qword_step = act_elems_per_load / 8;  // 4-bit has 8 elements per qword
    const size_t wt_qword_step = act_elems_per_load / 4;   // 8-bit has 4 elements per qword
    const size_t elems_per_iter = act_elems_per_load;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 2);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * act_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v1(&w_ptr[k * ratio * wt_qword_step + r * wt_qword_step]);
                    acc_sum += vpu_vdot_v1_v0();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/2];
                temp_a = EXTRACT_4BIT(temp_a, t & 0b1);
                temp_a = SIGN_EXTEND_TO_INT8(temp_a, 4);
                acc_sum += temp_a * w->data[j * in_features + t];
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 2);

    int8_t temp_a;
    // For Q2x8: activation is 2-bit, weight is 8-bit
    // Ratio: 8/2 = 4 weight loads per activation load
    const size_t act_elems_per_load = VPU_ELEMS_Q2;
    const size_t ratio = 8 / 2;
    const size_t act_qword_step = act_elems_per_load / 16;  // 2-bit has 16 elements per qword
    const size_t wt_qword_step = act_elems_per_load / 4;    // 8-bit has 4 elements per qword
    const size_t elems_per_iter = act_elems_per_load;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * act_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v1(&w_ptr[k * ratio * wt_qword_step + r * wt_qword_step]);
                    acc_sum += vpu_vdot_v1_v0();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/4];
                temp_a = EXTRACT_2BIT(temp_a, t & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                acc_sum += temp_a * w->data[j * in_features + t];
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1x8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(8, 1);

    int8_t temp_a;

    // For Q1x8: activation is 1-bit, weight is 8-bit
    // Ratio: 8/1 = 8 weight loads per activation load
    const size_t act_elems_per_load = VPU_ELEMS_Q1;
    const size_t ratio = 8 / 1;
    const size_t act_qword_step = act_elems_per_load / 32;  // 1-bit has 32 elements per qword
    const size_t wt_qword_step = act_elems_per_load / 4;    // 8-bit has 4 elements per qword
    const size_t elems_per_iter = act_elems_per_load;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * act_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v1(&w_ptr[k * ratio * wt_qword_step + r * wt_qword_step]);
                    acc_sum += vpu_vdot_v1_v0();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/8];
                temp_a = EXTRACT_BIT(temp_a, t & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                acc_sum += temp_a * w->data[j * in_features + t];
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q2x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(4, 2);

    int8_t temp_a;
    int8_t temp_w;

    // For Q2x4: activation is 2-bit, weight is 4-bit
    // Ratio: 4/2 = 2 weight loads per activation load
    const size_t act_elems_per_load = VPU_ELEMS_Q2;
    const size_t ratio = 4 / 2;
    const size_t act_qword_step = act_elems_per_load / 16;  // 2-bit has 16 elements per qword
    const size_t wt_qword_step = act_elems_per_load / 8;    // 4-bit has 8 elements per qword
    const size_t elems_per_iter = act_elems_per_load;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 4);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 2);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * act_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v1(&w_ptr[k * ratio * wt_qword_step + r * wt_qword_step]);
                    acc_sum += vpu_vdot_v1_v0();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/4];
                temp_a = EXTRACT_2BIT(temp_a, t & 0b11);
                temp_a = TWO_BIT_TO_INT8(temp_a);
                temp_w = w->data[(j * in_features + t)/2];
                temp_w = EXTRACT_4BIT(temp_w, t & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1x4_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(4, 1);

    int8_t temp_a;
    int8_t temp_w;

    // For Q1x4: activation is 1-bit, weight is 4-bit
    // Ratio: 4/1 = 4 weight loads per activation load
    const size_t act_elems_per_load = VPU_ELEMS_Q1;
    const size_t ratio = 4 / 1;
    const size_t act_qword_step = act_elems_per_load / 32;  // 1-bit has 32 elements per qword
    const size_t wt_qword_step = act_elems_per_load / 8;    // 4-bit has 8 elements per qword
    const size_t elems_per_iter = act_elems_per_load;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 2);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * act_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v1(&w_ptr[k * ratio * wt_qword_step + r * wt_qword_step]);
                    acc_sum += vpu_vdot_v1_v0();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/8];
                temp_a = EXTRACT_BIT(temp_a, t & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                temp_w = w->data[(j * in_features + t)/2];
                temp_w = EXTRACT_4BIT(temp_w, t & 0b1);
                temp_w = SIGN_EXTEND_TO_INT8(temp_w, 4);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}

void MiCo_Q1x2_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w){
    const size_t batch_size = x->shape[0];
    const size_t in_features = x->shape[1];
    const size_t out_features = w->shape[0];

    cfu_enable();
    fence_i();
    vpu_config(2, 1);

    int8_t temp_a;
    int8_t temp_w;
    // For Q1x2: activation is 1-bit, weight is 2-bit
    // Ratio: 2/1 = 2 weight loads per activation load
    const size_t act_elems_per_load = VPU_ELEMS_Q1;
    const size_t ratio = 2 / 1;
    const size_t act_qword_step = act_elems_per_load / 32;  // 1-bit has 32 elements per qword
    const size_t wt_qword_step = act_elems_per_load / 16;   // 2-bit has 16 elements per qword
    const size_t elems_per_iter = act_elems_per_load;
    const size_t num_iters = in_features / elems_per_iter;
    const size_t remain = in_features % elems_per_iter;
    for (size_t i = 0; i < batch_size; i++) {
        const qword *a_ptr = (const qword*)(x->data + i * in_features / 8);
        for (size_t j = 0; j < out_features; j++) {
            int32_t acc_sum = 0;
            const qword *w_ptr = (const qword*)(w->data + j * in_features / 4);
            for (size_t k = 0; k < num_iters; k++) {
                vpu_load_v0(&a_ptr[k * act_qword_step]);
                for (size_t r = 0; r < ratio; r++) {
                    vpu_load_v1(&w_ptr[k * ratio * wt_qword_step + r * wt_qword_step]);
                    acc_sum += vpu_vdot_v1_v0();
                }
            }
            for (size_t t = in_features - remain; t < in_features; t++) {
                temp_a = x->data[(i * in_features + t)/8];
                temp_a = EXTRACT_BIT(temp_a, t & 0b111);
                temp_a = BIT_TO_INT8(temp_a);
                temp_w = w->data[(j * in_features + t)/4];
                temp_w = EXTRACT_2BIT(temp_w, t & 0b11);
                temp_w = TWO_BIT_TO_INT8(temp_w);
                acc_sum += temp_a * temp_w;
            }
            O[i * out_features + j] = acc_sum;
        }
    }
}
