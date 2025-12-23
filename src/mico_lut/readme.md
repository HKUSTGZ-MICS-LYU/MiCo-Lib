# LUT-based Mixed Precision MatMul Kernels

This directory contains LUT (Look-Up Table) based implementations of mixed precision matrix multiplication kernels, inspired by [T-MAC](https://github.com/microsoft/T-MAC).

## Overview

For low-bitwidth weights (1, 2, 4 bits), the number of possible weight values is very small:
- 1-bit: 2 values (+1, -1)
- 2-bit: 4 values (0, +1, -1, -2 in the ternary encoding)
- 4-bit: 16 values (-8 to +7 for signed)

The LUT-based approach replaces bit extraction and sign extension operations with simple table lookups, which can be more efficient on some architectures.

## Key Optimizations

1. **Static LUTs**: Pre-defined lookup tables map packed bit patterns directly to their signed integer values, eliminating runtime sign extension.

2. **Batch Processing**: Operations are batched to process multiple elements per byte (e.g., 4 x 2-bit weights per byte, 2 x 4-bit weights per byte).

3. **Reduced Branching**: LUT lookups are branchless operations, improving instruction pipeline efficiency.

## Weight Encoding

The LUT tables use the following encoding schemes:

### 1-bit Weights
```
0 -> +1
1 -> -1
```

### 2-bit Weights (Ternary)
```
0b00 (0) -> 0
0b01 (1) -> +1
0b10 (2) -> -2
0b11 (3) -> -1
```

### 4-bit Weights (Signed)
```
0-7   -> 0 to +7
8-15  -> -8 to -1
```

## Supported Operations

| Kernel | Activation Bits | Weight Bits | Notes |
|--------|-----------------|-------------|-------|
| `MiCo_Q8_MatMul` | 8 | 8 | Standard 8-bit (not LUT-optimized) |
| `MiCo_Q8x4_MatMul` | 8 | 4 | LUT for 4-bit weight extraction |
| `MiCo_Q8x2_MatMul` | 8 | 2 | LUT for 2-bit weight extraction |
| `MiCo_Q8x1_MatMul` | 8 | 1 | LUT for 1-bit weight extraction |
| `MiCo_Q4_MatMul` | 4 | 4 | LUT for both operands |
| `MiCo_Q4x2_MatMul` | 4 | 2 | LUT for both operands |
| `MiCo_Q4x1_MatMul` | 4 | 1 | LUT for both operands |
| `MiCo_Q2_MatMul` | 2 | 2 | LUT for both operands |
| `MiCo_Q2x1_MatMul` | 2 | 1 | LUT for both operands |
| `MiCo_Q1_MatMul` | 1 | 1 | XNOR + popcount for BNN |
| `MiCo_Q4x8_MatMul` | 4 | 8 | LUT for activation extraction |
| `MiCo_Q2x8_MatMul` | 2 | 8 | LUT for activation extraction |
| `MiCo_Q1x8_MatMul` | 1 | 8 | LUT for activation extraction |
| `MiCo_Q2x4_MatMul` | 2 | 4 | LUT for both operands |
| `MiCo_Q1x4_MatMul` | 1 | 4 | LUT for both operands |
| `MiCo_Q1x2_MatMul` | 1 | 2 | LUT for both operands |

## Usage

Enable LUT-based optimizations by adding `lut` to the `OPT` variable in your makefile:

```makefile
OPT += lut
include $(MICO_DIR)/targets/common.mk
```

Or when invoking make:

```bash
make OPT=lut ...
```

## Performance Considerations

The LUT approach is beneficial when:
- Target architecture has efficient byte-level memory access
- Sign extension is expensive (some RISC-V implementations)
- Cache is large enough to hold the small LUT tables

The LUT approach may be less beneficial when:
- Bit manipulation instructions are very fast
- Memory access latency is high compared to ALU operations

## References

- **T-MAC Paper**: Wei et al., "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge", arXiv:2407.00088
- **T-MAC GitHub**: https://github.com/microsoft/T-MAC

## Comparison with Other Sources

| Source | Optimization Strategy |
|--------|----------------------|
| `src/mico` | Reference implementation (bit extraction + sign extend) |
| `src/mico_unrolled` | Loop unrolling |
| `src/optimized` | Software optimizations (CTZ, popcount, etc.) |
| `src/mico_lut` | **LUT-based value extraction** |
