# A Simple Script to Generate Assembly Code

prec = [8,4,2,1]
vlen = [64,128,256,512]

min_inner_loop = 32

template = """
#include "custom_asm.h"

.global {func_name}
// Process one output of a Vec x Mat (Linear) operation ({vlen}-bit)
// a0: pointer to A Vec ({prec1}-bit)
// a1: pointer to W Mat ({prec2}-bit)
// a2: pointer to O Vec (32-bit)
// a3: Number of elements in A Vec (Inner loop 1)
// a4: Number of elements in W Mat (Outer loop 2)
{func_name}:
    fence.i
    vpu_CONFIG({prec1}, {prec2})
    li t0, 0  // Loop counter 1
loop1:
    li t1, 0  // Loop counter 2
    mv a5, a0 // Save pointer to W Mat
    li t4, 0  // Accumulator
loop2:
{compute_unroll}
    addi t1, t1, {elem_per_compute}
    blt t1, a3, loop2

    sw t4, 0(a2)
    addi t0, t0, 1
    addi a2, a2, 4
    blt t0, a4, loop1
    ret
"""
compute_template = """
    vpu_LOAD(a5, v0)
    vpu_LOAD(a1, v1)
    vpu_VDOT(t2, v0, v1)    // SIMD Dot Product
    add t4, t4, t2
    addi a5, a5, {step}
    addi a1, a1, {step}
"""

def gen_NxN_asm(prec, vlen):
    func_name = f"cfu_dotp_int{prec}"
    file_path = f"v{vlen}/{func_name}.S"

    step = (vlen // 8)
    elem_per_compute = vlen // prec

    if (elem_per_compute < min_inner_loop):
        repeat = min_inner_loop // elem_per_compute
        compute_asm = compute_template.format(
            step = step
        )
        compute_asm = compute_asm * repeat
        elem_per_compute = min_inner_loop
    else:
        compute_asm = compute_template.format(
            step = step
        )

    asm = template.format(
        func_name = func_name,
        prec1 = prec,
        prec2 = prec,
        vlen = vlen,
        compute_unroll = compute_asm,
        elem_per_compute = elem_per_compute
    )

    with open(file_path, "w") as f:
        f.write(asm)
    print(f"Generated {file_path}")

for p in prec:
    for v in vlen:
        gen_NxN_asm(p, v)