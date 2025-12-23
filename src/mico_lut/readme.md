# LUT-based Mixed Precision MatMul Kernels (T-MAC Style)

This directory contains LUT (Look-Up Table) based implementations of mixed precision matrix multiplication kernels, following the T-MAC approach from Microsoft.

## References

- **T-MAC Paper**: Wei et al., "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge", arXiv:2407.00088, EuroSys 2025
- **T-MAC GitHub**: https://github.com/microsoft/T-MAC
- **T-MAC Kernel Code**: https://github.com/microsoft/T-MAC/blob/main/deploy/tuned/kernels.cc

## Key Optimization: Loop Ordering

**CRITICAL**: LUTs must be built **OUTSIDE** the output feature loop to achieve performance benefits.

### Correct Loop Order (this implementation)
```c
for (batch) {
    // STEP 1: Build ALL LUTs for activation groups ONCE
    for (g = 0; g < num_groups; g++) {
        build_lut(&luts[g], activations[g]);  // O(256) per group
    }
    
    // STEP 2: Process ALL outputs using precomputed LUTs
    for (output) {
        for (g = 0; g < num_groups; g++) {
            acc += luts[g][weight_byte[g]];   // O(1) lookup
        }
    }
}
```
**Cost**: O(batch * groups * 256) for LUT building + O(batch * outputs * groups) for lookups

### Wrong Loop Order (causes huge performance problem)
```c
for (batch) {
    for (output) {                      // <-- Output loop
        for (g = 0; g < num_groups; g++) {
            build_lut(&lut, activations[g]);  // BUG: Rebuilds LUT for EVERY output!
            acc += lut[weight_byte[g]];
        }
    }
}
```
**Cost**: O(batch * outputs * groups * 256) - rebuilds same LUT repeatedly!

## T-MAC Approach

For N-bit weights, we precompute **ALL possible partial sums** for groups of activations:
- Group activations together (e.g., 4 activations for 2-bit weights = 1 byte)
- Build a 256-entry LUT where index = weight byte
- Each LUT entry = sum of (activation * decoded_weight) for all elements in group
- The LUT is built ONCE and reused across ALL output features

## LUT Building Functions

| Function | Activations | Weight Bits | LUT Size |
|----------|-------------|-------------|----------|
| `build_lut_8x1()` | 8 | 1-bit | 256 |
| `build_lut_4x2()` | 4 | 2-bit | 256 |
| `build_lut_2x4()` | 2 | 4-bit | 256 |

## Supported Operations

### High Activation Bits × Low Weight Bits (LUT-optimized)
- `MiCo_Q8x1_MatMul`: 8-bit activation × 1-bit weight
- `MiCo_Q8x2_MatMul`: 8-bit activation × 2-bit weight  
- `MiCo_Q8x4_MatMul`: 8-bit activation × 4-bit weight
- `MiCo_Q4x1_MatMul`, `MiCo_Q4x2_MatMul`: 4-bit activation variants
- `MiCo_Q2x1_MatMul`: 2-bit activation × 1-bit weight

### Same Precision
- `MiCo_Q8_MatMul`: 8-bit × 8-bit (standard, no LUT)
- `MiCo_Q4_MatMul`: 4-bit × 4-bit
- `MiCo_Q2_MatMul`: 2-bit × 2-bit
- `MiCo_Q1_MatMul`: 1-bit × 1-bit (XNOR + popcount)

### Low Activation Bits × High Weight Bits
- Direct computation (LUT doesn't help when weight space is large)

## Usage

Enable LUT-based optimizations:

```makefile
OPT += lut
include $(MICO_DIR)/targets/common.mk
```

## Weight Encoding

| Bits | Values | Mapping |
|------|--------|---------|
| 1-bit | 2 | `0→+1, 1→-1` |
| 2-bit | 4 | `0→0, 1→+1, 2→-2, 3→-1` |
| 4-bit | 16 | `0-7→0..+7, 8-15→-8..-1` |

## Memory Usage

LUTs are allocated on stack for small sizes (up to 64 groups × 256 entries = 64KB) with heap fallback for larger matrices.

## Portability

This implementation:
- Does NOT use SIMD intrinsics (no ARM NEON, no AVX2)
- Works on any C compiler (GCC, Clang, etc.)
- Targets general CPUs and RISC-V
- Uses `__attribute__((weak))` for override support
