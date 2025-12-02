# Optimized Mixed Precision Operators for Regular RISC-V CPUs

This directory contains optimized implementations of mixed precision matrix multiplication kernels for regular RISC-V CPUs without custom ISA extensions.

## Overview

Unlike the VexiiRiscv target (`targets/vexii/mico32` and `targets/vexii/mico64`) which requires custom SIMD instructions, these implementations use only standard RISC-V instructions and are compatible with any RISC-V CPU that implements the base integer instruction set with multiplication extension (RV32IM/RV64IM).

## Optimizations

The kernels use several software optimization techniques:

1. **Loop Unrolling**: Inner loops are unrolled to reduce loop overhead and improve instruction-level parallelism.

2. **Batch Memory Loads**: Multiple packed values are loaded at once to reduce memory access overhead.

3. **XOR + Popcount for Binary Networks**: 1-bit x 1-bit operations use XNOR and population count for efficient binary neural network computations.

4. **Register Reuse**: Variables are kept in registers where possible to minimize memory traffic.

## Supported Operations

All standard MiCo mixed precision matmul operations are supported:

- `MiCo_Q8_MatMul`: 8-bit x 8-bit
- `MiCo_Q8x4_MatMul`: 8-bit activation x 4-bit weight
- `MiCo_Q8x2_MatMul`: 8-bit activation x 2-bit weight
- `MiCo_Q8x1_MatMul`: 8-bit activation x 1-bit weight
- `MiCo_Q4_MatMul`: 4-bit x 4-bit
- `MiCo_Q4x2_MatMul`: 4-bit activation x 2-bit weight
- `MiCo_Q4x1_MatMul`: 4-bit activation x 1-bit weight
- `MiCo_Q2_MatMul`: 2-bit x 2-bit
- `MiCo_Q2x1_MatMul`: 2-bit activation x 1-bit weight
- `MiCo_Q1_MatMul`: 1-bit x 1-bit (Binary Neural Network)
- `MiCo_Q4x8_MatMul`: 4-bit activation x 8-bit weight
- `MiCo_Q2x8_MatMul`: 2-bit activation x 8-bit weight
- `MiCo_Q2x4_MatMul`: 2-bit activation x 4-bit weight
- `MiCo_Q1x8_MatMul`: 1-bit activation x 8-bit weight
- `MiCo_Q1x4_MatMul`: 1-bit activation x 4-bit weight
- `MiCo_Q1x2_MatMul`: 1-bit activation x 2-bit weight

## Usage

Include the RISC-V target in your makefile:

```makefile
include $(MICO_DIR)/targets/riscv.mk
```

Or set the `TARGET` variable:

```bash
make TARGET=riscv ...
```

## Comparison with Other Targets

| Target | ISA Requirements | Optimization Level |
|--------|------------------|-------------------|
| `src/mico` | None (baseline) | Reference implementation |
| `src/mico_unrolled` | None | Loop unrolling for Q8 only |
| `targets/riscv` | RV32IM/RV64IM | **Full mixed precision optimization** |
| `targets/vexii/mico32` | Custom SIMD | Hardware SIMD acceleration (32-bit) |
| `targets/vexii/mico64` | Custom SIMD | Hardware SIMD acceleration (64-bit) |
