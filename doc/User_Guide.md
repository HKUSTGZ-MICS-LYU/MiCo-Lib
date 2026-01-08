# MiCo-Lib User Guide

## Introduction

MiCo-Lib is a C library for quantized neural network inference, designed for embedded systems and specifically optimized for RISC-V architectures (including VexiiRiscv). It provides kernels for mixed-precision integer matrix multiplication and convolution.

## Integration

To use MiCo-Lib in your project, you need to include the source files and headers in your build system. The library uses a Makefile-based system that can be easily integrated.

### Prerequisites

*   **Compiler**: GCC or Clang (supporting C99 or later).
*   **Make**: GNU Make.
*   **RISC-V Toolchain**: (Optional) Required for targeting RISC-V platforms.

### Makefile Integration

1.  Define the `MICO_DIR` variable pointing to the root of the MiCo-Lib repository.
2.  Set the `OPT` variable to enable specific optimizations (optional).
3.  Include `targets/common.mk`.
4.  Include a platform-specific target makefile (e.g., `targets/x86.mk`, `targets/vexii.mk`).

**Example `Makefile`:**

```makefile
# Path to MiCo-Lib
MICO_DIR = ./path/to/MiCo-Lib

# Enable optimizations (optional)
# Options: unroll, im2col, opt, lut
OPT += lut opt

# Include common definitions
include $(MICO_DIR)/targets/common.mk

# Include target platform (choose one)
# For Host (x86/Linux):
include $(MICO_DIR)/targets/x86.mk
# For VexiiRiscv:
# include $(MICO_DIR)/targets/vexii.mk

# Add your own sources
SRCS += main.c

# Compile
main: $(SRCS) $(MICO_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
```

## Build Configuration

### Optimization Flags (`OPT`)

The `OPT` variable controls which kernel implementations are selected. You can combine multiple flags.

| Flag | Description |
| :--- | :--- |
| `lut` | Enables Look-Up Table (LUT) based mixed-precision kernels (T-MAC style). Efficient for low-bit weights. |
| `opt` | Enables generic optimized mixed-precision kernels for RISC-V (RV32IM/RV64IM). Uses unrolling and bit-hacks. |
| `unroll` | Enables loop-unrolled implementations for INT8 MatMul. |
| `im2col`| Enables Im2Col-based convolution kernels. |
| `ref` | Enables reference implementations (useful for debugging). |

### Target Platforms

Platform-specific makefiles in `targets/` set up the necessary compiler flags and source files.

*   `targets/x86.mk`: For running on x86 host (Linux/Windows). Uses `-mavx -mavx2` if available.
*   `targets/vexii.mk`: For VexiiRiscv hardware/simulator.
*   `targets/cuda.mk`: For NVIDIA GPUs (experimental).
*   `targets/openmp.mk`: OpenMP accelerated kernels (experimental).

## Directory Structure

*   `include/`: Header files (`mico_nn.h`, `qtypes.h`).
*   `src/`: Core source code.
    *   `mico/`: Reference mixed-precision kernels.
    *   `mico_lut/`: LUT-based kernels.
    *   `mico_unrolled/`: Unrolled kernels.
    *   `optimized/`: RISC-V optimized kernels.
*   `targets/`: Platform-specific build files and drivers.
*   `doc/`: Documentation.
*   `test/`: Unit tests.

## Running Tests

Tests are located in the `test/` directory. To run tests, you typically need to build a test runner that links against MiCo-Lib.
(Refer to specific target documentation for detailed test instructions).
