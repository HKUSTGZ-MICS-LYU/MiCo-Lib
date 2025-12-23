# LUT-based Mixed Precision MatMul Kernels (T-MAC Style)

This directory contains LUT (Look-Up Table) based implementations of mixed precision matrix multiplication kernels, following the T-MAC approach from Microsoft.

## References

- **T-MAC Paper**: Wei et al., "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge", arXiv:2407.00088, EuroSys 2025
- **T-MAC GitHub**: https://github.com/microsoft/T-MAC
- **T-MAC Kernel Code**: https://github.com/microsoft/T-MAC/blob/main/deploy/tuned/kernels.cc

## Overview: The T-MAC Approach

T-MAC introduces a novel approach to low-bit matrix multiplication that replaces expensive multiply-accumulate operations with simple table lookups.

### Key Insight

For N-bit weights, instead of extracting weights and multiplying, T-MAC:
1. Groups activations together (e.g., 4 activations for 4-bit nibble indices)
2. Precomputes ALL possible partial sums for that group
3. Uses the weight bits as a direct INDEX into the precomputed LUT
4. For multi-bit weights, decomposes into bit planes and scales+accumulates

### Example: Sign-based LUT (1-bit weights)

For 4 activations `[a0, a1, a2, a3]` with 1-bit weights (sign only), build a 16-entry LUT:
```
LUT[0b0000] = +a0 + a1 + a2 + a3  (all positive)
LUT[0b0001] = -a0 + a1 + a2 + a3  (a0 negative)
LUT[0b0010] = +a0 - a1 + a2 + a3  (a1 negative)
...
LUT[0b1111] = -a0 - a1 - a2 - a3  (all negative)
```

The 4-bit weight nibble directly indexes into this LUT.

### Scalar vs SIMD

T-MAC's original implementation uses SIMD intrinsics (ARM NEON `vqtbl1q`, x86 `pshufb`) for fast parallel table lookups. This implementation provides a portable **scalar version** for:
- General CPUs without SIMD support
- RISC-V processors
- Embedded systems

While the scalar version doesn't achieve the same speedup as SIMD, it still:
- Eliminates multiply operations for low-bit weights
- Converts multiplication into simple table lookup + addition
- Provides correct reference implementations for all precision combinations

## Implementation Details

### LUT Building Functions

- `build_sign_lut_4()`: Builds 16-entry LUT for 4 activations with sign indices
- `build_sign_lut_8()`: Builds 256-entry LUT for 8 activations with sign indices

### LUT Size by Weight Precision

| Weight Bits | Activations per Group | LUT Entries | Index Bits |
|-------------|----------------------|-------------|------------|
| 1-bit       | 8                    | 256         | 8          |
| 2-bit       | 4                    | 256         | 8 (full byte) |
| 4-bit       | 2                    | 256         | 8 (full byte) |

## Supported Operations

### High Activation Bits × Low Weight Bits (LUT-optimized)
- `MiCo_Q8x1_MatMul`: 8-bit activation × 1-bit weight
- `MiCo_Q8x2_MatMul`: 8-bit activation × 2-bit weight  
- `MiCo_Q8x2_MatMul_Opt`: Optimized version with precomputed LUTs
- `MiCo_Q8x4_MatMul`: 8-bit activation × 4-bit weight
- `MiCo_Q4x1_MatMul`, `MiCo_Q4x2_MatMul`: 4-bit activation variants
- `MiCo_Q2x1_MatMul`: 2-bit activation × 1-bit weight

### Same Precision
- `MiCo_Q8_MatMul`: 8-bit × 8-bit (standard, no LUT needed)
- `MiCo_Q4_MatMul`: 4-bit × 4-bit
- `MiCo_Q2_MatMul`: 2-bit × 2-bit
- `MiCo_Q1_MatMul`: 1-bit × 1-bit (uses XNOR + popcount)

### Low Activation Bits × High Weight Bits
- `MiCo_Q4x8_MatMul`, `MiCo_Q2x8_MatMul`, `MiCo_Q1x8_MatMul`
- `MiCo_Q2x4_MatMul`, `MiCo_Q1x4_MatMul`, `MiCo_Q1x2_MatMul`

Note: Reversed precision (low act × high weight) uses direct computation since the weight space is larger.

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

## Weight Encoding

| Bits | Values | Mapping |
|------|--------|---------|
| 1-bit | 2 | `0→+1, 1→-1` |
| 2-bit | 4 | `0→0, 1→+1, 2→-2, 3→-1` |
| 4-bit | 16 | `0-7→0..+7, 8-15→-8..-1` |

## Comparison with Other Implementations

| Source | Approach |
|--------|----------|
| `src/mico` | Reference (bit extraction + multiply) |
| `src/mico_unrolled` | Loop unrolling |
| `src/optimized` | Software optimizations (CTZ, popcount) |
| `src/mico_lut` | **T-MAC style LUT (precomputed partial sums)** |

## Notes on Portability

This implementation:
- Does NOT use SIMD intrinsics (no ARM NEON, no AVX2)
- Works on any C compiler (GCC, Clang, etc.)
- Uses `__builtin_popcount` where available (with fallback)
- Uses `__attribute__((weak))` for override support
