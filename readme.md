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

With hardware requirement:
+ `src/mico_simd`: SIMD-based MatMul Kernels.

### Platform-related Files

`vexii`: Drivers, libraries and link scripts for MiCo to run on the VexiiRiscv hardware or simulator.