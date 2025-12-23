# LUT-based Mixed Precision MatMul Kernels (T-MAC Style)

This directory contains LUT (Look-Up Table) based implementations of mixed precision matrix multiplication kernels, following the T-MAC approach from Microsoft.

## References

- **T-MAC Paper**: Wei et al., "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge", arXiv:2407.00088
- **T-MAC GitHub**: https://github.com/microsoft/T-MAC

## Overview: The T-MAC Approach

Traditional low-bit matrix multiplication requires extracting packed weights, sign-extending them, and performing multiply-accumulate operations. T-MAC introduces a fundamentally different approach:

### Key Insight

For N-bit weights packed in a byte, there are only 256 possible byte values. For each group of activations corresponding to one weight byte, we can **precompute all 256 possible partial sums** and store them in a Look-Up Table (LUT). The weight byte then directly **indexes** into this LUT, eliminating multiply operations entirely.

### Example: 8-bit Activation × 2-bit Weight

```
Given:
- 4 activations: [a0, a1, a2, a3]
- 1 weight byte containing 4×2-bit weights

T-MAC Approach:
1. Build LUT[0..255] where each entry is a precomputed partial sum:
   LUT[wb] = a0 * decode(wb[1:0]) + a1 * decode(wb[3:2]) + 
             a2 * decode(wb[5:4]) + a3 * decode(wb[7:6])

2. For each output feature, just look up: acc += LUT[weight_byte]
```

### Benefits

1. **Eliminates multiply operations**: Converts multiply-accumulate into table lookup + add
2. **Linear scaling**: Performance scales linearly with bit-width reduction (unlike dequantization methods)
3. **Energy efficient**: Table lookup uses less power than multiply-add
4. **LUT reuse**: Same activation LUT is reused across all output features

## Weight Encoding

| Bits | Values | Mapping |
|------|--------|---------|
| 1-bit | 2 | `0→+1, 1→-1` |
| 2-bit | 4 | `0→0, 1→+1, 2→-2, 3→-1` |
| 4-bit | 16 | `0-7→0..+7, 8-15→-8..-1` |

## Supported Operations

### High Activation Bits × Low Weight Bits (LUT-optimized)
- `MiCo_Q8x2_MatMul`: 8-bit activation × 2-bit weight (256-entry LUT per 4 activations)
- `MiCo_Q8x4_MatMul`: 8-bit activation × 4-bit weight (256-entry LUT per 2 activations)
- `MiCo_Q8x1_MatMul`: 8-bit activation × 1-bit weight (256-entry LUT per 8 activations)
- `MiCo_Q4x2_MatMul`: 4-bit activation × 2-bit weight
- `MiCo_Q4x1_MatMul`: 4-bit activation × 1-bit weight
- `MiCo_Q2x1_MatMul`: 2-bit activation × 1-bit weight

### Same Precision
- `MiCo_Q8_MatMul`: 8-bit × 8-bit (no LUT benefit)
- `MiCo_Q4_MatMul`: 4-bit × 4-bit
- `MiCo_Q2_MatMul`: 2-bit × 2-bit
- `MiCo_Q1_MatMul`: 1-bit × 1-bit (XNOR + popcount)

### Low Activation Bits × High Weight Bits
- `MiCo_Q4x8_MatMul`, `MiCo_Q2x8_MatMul`, `MiCo_Q1x8_MatMul`
- `MiCo_Q2x4_MatMul`, `MiCo_Q1x4_MatMul`, `MiCo_Q1x2_MatMul`

Note: Reversed precision (low act × high weight) doesn't benefit as much from LUT since the weight space is larger.

## Usage

Enable LUT-based optimizations:

```makefile
OPT += lut
include $(MICO_DIR)/targets/common.mk
```

Or:

```bash
make OPT=lut ...
```

## Implementation Notes

- Each function uses `__attribute__((weak))` to allow platform-specific overrides
- LUTs are built dynamically per activation group to avoid memory constraints
- An optimized variant `MiCo_Q8x2_MatMul_Opt` precomputes all LUTs upfront for better reuse
- 1-bit × 1-bit uses XNOR + popcount instead of LUT for efficiency

## Comparison with Other Sources

| Source | Approach |
|--------|----------|
| `src/mico` | Reference (bit extraction + multiply) |
| `src/mico_unrolled` | Loop unrolling |
| `src/optimized` | Software optimizations (CTZ, popcount) |
| `src/mico_lut` | **T-MAC style LUT (precomputed partial sums)** |
