# MiCo C Library

## Code Structure

### Main Sources
+ `include`: Basic Headers of MiCo NN Kernels.

+ `src`: Hardware-agnostic Basic Sources of MiCo NN Kernels.

+ `src/mico`: Sources for Mixed-precision Integer Kernels.

### Optimized Sources
Without hardware requirement:
+ `src/mico_unrolled`: Unrolled INT8 MatMul Kernels.
+ `src/im2col_conv2d`: Im2Col-based Conv2D Kernels.
+ `src/riscv`: Optimized mixed precision operators for regular RISC-V CPUs (RV32IM/RV64IM).

With hardware requirement:
+ `src/mico_simd`: SIMD-based MatMul Kernels.

### Platform-related Files

`vexii`: Drivers, libraries and link scripts for MiCo to run on the VexiiRiscv hardware or simulator.

## Quantized Kernels

We **assumes an alignment of 32/64 elements** for quantized data. Although it cost overhead during the quantization, it improves the overall performance:

+ It maximizes the utilization of SIMD or unrolling on quantized kernels (`bitlinear` and `bitconv2d`).
+ It ensures aligned memory accesses, which will result in exceptions for some CPUs (like VexiiRiscv).

The padding and aligning for activations are applied at the runtime, but weights need to be padded when exporting the model. The codegen of `MiCo-python` will detect the alignment and apply padding during the exporting of models. If you want to develop or use other way to export your models, please ensure the alignment or work around the asm kernels.