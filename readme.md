# MiCo C Library

MiCo-Lib is a C library for quantized neural network inference, specifically optimized for mixed-precision integer operations on embedded systems and RISC-V architectures.

## Documentation

*   [**User Guide**](doc/User_Guide.md): Detailed instructions on building, configuring, and using the library.
*   [**API Reference**](doc/API_Reference.md): Documentation for the core C API and data structures.
*   [**Quantized Computing Flow**](doc/Quantized_Computing_Flow.md): Theory behind the quantization scheme.

## Code Structure

### Main Sources
+ `include`: Public header files (`mico_nn.h`, `qtypes.h`).
+ `src`: Core source code for neural network kernels.
+ `src/mico`: Reference implementations of mixed-precision integer kernels.

### Optimized Sources
**Software-only (No custom hardware required):**
+ `src/mico_unrolled`: Loop-unrolled INT8 MatMul kernels.
+ `src/im2col_conv2d`: Im2Col-based Conv2D kernels.
+ `src/optimized`: Optimized mixed-precision operators for standard RISC-V CPUs (RV32IM/RV64IM).
+ `src/mico_lut`: LUT-based kernels (T-MAC style) for efficient low-bit weight inference.

**Hardware-accelerated:**
+ `targets/vexii`: Drivers and assembly kernels for VexiiRiscv with custom SIMD extensions.

### Platform Support
Platform-specific build files and drivers are located in `targets/`:
+ `targets/x86`: For host simulation on Linux/Windows.
+ `targets/vexii`: For VexiiRiscv hardware and simulator.
+ `targets/cuda`: For CUDA-enabled GPUs.

## Quantized Kernels & Alignment

We **assume an alignment of 32/64 elements** for quantized data. While this adds overhead during quantization, it significantly improves performance:

+ It maximizes the utilization of SIMD instructions or loop unrolling.
+ It ensures aligned memory accesses, preventing exceptions on strict architectures (like VexiiRiscv).

**Note:** Padding and alignment for activations are applied at runtime. However, **weights must be padded during model export**. The `MiCo-python` codegen handles this automatically. If you export models manually, ensure correct alignment to avoid runtime errors.
