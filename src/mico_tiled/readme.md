# Tiled Output-Stationary MatMul Kernels

This directory contains tiled output-stationary implementations of mixed precision matrix multiplication kernels. This approach improves cache utilization by dividing the computation into tiles while maintaining the output-stationary property.

## Dataflow Comparison

### Output-Stationary (Baseline in `src/mico/qmatmul.c`)

Loop order: `for i (batch), for j (out_features), for k (in_features)`

```c
for (i = 0; i < batch_size; i++) {
    for (j = 0; j < out_features; j++) {
        acc = 0;
        for (k = 0; k < in_features; k++) {
            acc += X[i,k] * W[j,k];
        }
        O[i,j] = acc;
    }
}
```

**Characteristics:**
- Simple loop structure
- Accumulator in register minimizes output writes
- Poor cache locality for large matrices

### Tiled Output-Stationary (This Directory)

Loop order: `for i_tile, for j_tile, for k_tile, for i, for j, for k`

```c
for (i0 = 0; i0 < batch_size; i0 += TILE_I) {
    for (j0 = 0; j0 < out_features; j0 += TILE_J) {
        for (k0 = 0; k0 < in_features; k0 += TILE_K) {
            // Process tile
            for (i = i0; i < min(i0+TILE_I, batch_size); i++) {
                for (j = j0; j < min(j0+TILE_J, out_features); j++) {
                    acc = O[i,j];
                    for (k = k0; k < min(k0+TILE_K, in_features); k++) {
                        acc += X[i,k] * W[j,k];
                    }
                    O[i,j] = acc;
                }
            }
        }
    }
}
```

**Characteristics:**
- Divides computation into tiles for better cache utilization
- Each tile fits in cache, reducing cache misses
- Maintains output-stationary property within tiles
- Configurable tile sizes for different cache hierarchies

### Input-Stationary / Outer-Product (in `src/mico_outer/`)

Loop order: `for k (in_features), for i (batch), for j (out_features)`

**Characteristics:**
- Better input reuse (broadcasts input across outputs)
- More memory writes to output array
- Good for SIMD broadcast operations

## When to Use Each Dataflow

| Scenario | Preferred Dataflow |
|----------|-------------------|
| Small matrices | Baseline (non-tiled) |
| Large matrices | Tiled output-stationary |
| Limited cache | Tiled with smaller tiles |
| Large cache (L2/L3) | Tiled with larger tiles |
| SIMD broadcast support | Input-stationary (outer) |
| Minimize memory writes | Output-stationary |

## Tile Size Configuration

Default tile sizes can be overridden at compile time:

```makefile
CFLAGS += -DTILE_I=8 -DTILE_J=8 -DTILE_K=16
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TILE_I` | 4 | Batch tile size |
| `TILE_J` | 4 | Output features tile size |
| `TILE_K` | 8 | Reduction (input features) tile size |

**Tuning Guidelines:**
- Larger tiles = more data reuse, but may exceed cache
- `TILE_I * TILE_J * sizeof(int32_t)` should fit in L1 cache for output tile
- `TILE_I * TILE_K` elements of X should fit in cache
- `TILE_J * TILE_K` elements of W should fit in cache

## Supported Operations

All 16 mixed-precision MatMul operations are implemented:

### Same Precision
- `MiCo_Q8_MatMul`: 8-bit × 8-bit
- `MiCo_Q4_MatMul`: 4-bit × 4-bit
- `MiCo_Q2_MatMul`: 2-bit × 2-bit
- `MiCo_Q1_MatMul`: 1-bit × 1-bit (BNN)

### High Activation × Low Weight
- `MiCo_Q8x4_MatMul`: 8-bit activation × 4-bit weight
- `MiCo_Q8x2_MatMul`: 8-bit activation × 2-bit weight
- `MiCo_Q8x1_MatMul`: 8-bit activation × 1-bit weight
- `MiCo_Q4x2_MatMul`: 4-bit activation × 2-bit weight
- `MiCo_Q4x1_MatMul`: 4-bit activation × 1-bit weight
- `MiCo_Q2x1_MatMul`: 2-bit activation × 1-bit weight

### Low Activation × High Weight
- `MiCo_Q4x8_MatMul`: 4-bit activation × 8-bit weight
- `MiCo_Q2x8_MatMul`: 2-bit activation × 8-bit weight
- `MiCo_Q1x8_MatMul`: 1-bit activation × 8-bit weight
- `MiCo_Q2x4_MatMul`: 2-bit activation × 4-bit weight
- `MiCo_Q1x4_MatMul`: 1-bit activation × 4-bit weight
- `MiCo_Q1x2_MatMul`: 1-bit activation × 2-bit weight

## Usage

Enable tiled output-stationary optimizations:

```makefile
OPT += tiled
include $(MICO_DIR)/targets/common.mk
```

With custom tile sizes:

```makefile
OPT += tiled
CFLAGS += -DTILE_I=8 -DTILE_J=8 -DTILE_K=16
include $(MICO_DIR)/targets/common.mk
```

## Verification

Results should match exactly with the reference implementation in `src/mico/qmatmul_ref.c`. Use the test file in `test/mico_tiled_test.c` to verify correctness.

## Performance Notes

- Tiling improves performance most significantly for large matrices
- For very small matrices, overhead of tile management may not be worthwhile
- Optimal tile sizes depend on:
  - Cache size and hierarchy (L1/L2/L3)
  - Memory bandwidth
  - Target architecture

## References

- "Anatomy of High-Performance Matrix Multiplication" - Goto & Van De Geijn, 2008
- "Efficient Processing of Deep Neural Networks: A Tutorial and Survey" - Sze et al., 2017
- "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" - Van Zee & van de Geijn, 2015
