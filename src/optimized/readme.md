# Optimized Mixed Precision Operators for Regular RISC-V CPUs

This directory contains optimized implementations of mixed precision matrix multiplication kernels for regular RISC-V CPUs without custom ISA extensions.

## Overview

Unlike the VexiiRiscv target (`targets/vexii/mico32` and `targets/vexii/mico64`) which requires custom SIMD instructions, these implementations use only standard RISC-V instructions and are compatible with any RISC-V CPU that implements the base integer instruction set with multiplication extension (RV32IM/RV64IM).

## Optimizations

The kernels use several software optimization techniques, inspired by best practices from projects like [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn):

1. **5x Row Unrolling**: Processing 5 output rows simultaneously to expose more instruction-level parallelism (ILP), similar to the approach used in muriscv-nn's scalar fallback path.

2. **4x Column Unrolling**: Inner loops are unrolled by 4 to reduce loop overhead and improve pipeline utilization.

3. **Batch Memory Loads**: Multiple packed values are loaded at once to reduce memory access overhead.

4. **XOR + Population Count for Binary Networks**: 1-bit x 1-bit operations use XNOR and population count for efficient binary neural network computations.

5. **Register Reuse**: Variables are kept in registers where possible to minimize memory traffic.

6. **1-bit Weight/Activation Optimization**: For operations involving 1-bit values, we use the mathematical identity:
   - For 1-bit weights: `sum(x_i * w_i) = total_sum - 2 * sum(x_i where w_i = -1)`
   - This eliminates per-bit multiplication and uses efficient bit scanning (CTZ + clear lowest bit)
   - Pre-computed activation sums are reused across output features

7. **Sparse Bit Processing**: Instead of iterating through all bits, we only process set bits using `while(wb) { pos = ctz(wb); wb &= wb - 1; }` pattern, which is faster when weights are sparse.

## Performance Characteristics

Low bitwidth operators are optimized to be faster than INT8 baseline when:
- Weight sparsity is moderate (for 1-bit ops, ~50% zeros is optimal)
- The reduction in memory bandwidth (due to packed weights) compensates for bit extraction overhead
- Batch sizes allow activation sum reuse

## Supported Operations

All standard MiCo mixed precision matmul operations are supported:

- `MiCo_Q8_MatMul`: 8-bit x 8-bit *(5x row unrolling)*
- `MiCo_Q8x4_MatMul`: 8-bit activation x 4-bit weight
- `MiCo_Q8x2_MatMul`: 8-bit activation x 2-bit weight
- `MiCo_Q8x1_MatMul`: 8-bit activation x 1-bit weight *(optimized)*
- `MiCo_Q4_MatMul`: 4-bit x 4-bit
- `MiCo_Q4x2_MatMul`: 4-bit activation x 2-bit weight
- `MiCo_Q4x1_MatMul`: 4-bit activation x 1-bit weight *(optimized)*
- `MiCo_Q2_MatMul`: 2-bit x 2-bit
- `MiCo_Q2x1_MatMul`: 2-bit activation x 1-bit weight *(optimized)*
- `MiCo_Q1_MatMul`: 1-bit x 1-bit (Binary Neural Network) *(optimized)*
- `MiCo_Q4x8_MatMul`: 4-bit activation x 8-bit weight
- `MiCo_Q2x8_MatMul`: 2-bit activation x 8-bit weight
- `MiCo_Q2x4_MatMul`: 2-bit activation x 4-bit weight
- `MiCo_Q1x8_MatMul`: 1-bit activation x 8-bit weight *(optimized)*
- `MiCo_Q1x4_MatMul`: 1-bit activation x 4-bit weight *(optimized)*
- `MiCo_Q1x2_MatMul`: 1-bit activation x 2-bit weight *(optimized)*

## References

- [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn) - RISC-V optimized neural network kernels from TUM

## Usage

Enable the RISC-V optimizations by adding `opt` to the `OPT` variable in your makefile:

```makefile
OPT += opt
include $(MICO_DIR)/targets/common.mk
```

Or when invoking make:

```bash
make OPT=opt ...
```

## Comparison with Other Sources

| Source | ISA Requirements | Optimization Level |
|--------|------------------|-------------------|
| `src/mico` | None (baseline) | Reference implementation |
| `src/mico_unrolled` | None | Loop unrolling for Q8 only |
| `src/opt` | RV32IM/RV64IM | **Full mixed precision optimization** |
| `targets/vexii/mico32` | Custom SIMD | Hardware SIMD acceleration (32-bit) |
| `targets/vexii/mico64` | Custom SIMD | Hardware SIMD acceleration (64-bit) |
