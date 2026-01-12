# Outer-Product (Input-Stationary) MatMul Kernels

This directory contains outer-product based implementations of mixed precision matrix multiplication kernels. This is an alternative dataflow to the output-stationary approach used in the baseline implementation (`src/mico/qmatmul.c`).

## Dataflow Comparison

### Output-Stationary (Baseline in `src/mico/qmatmul.c`)

Loop order: `for i (batch), for j (out_features), for k (in_features)`

```c
for (i = 0; i < batch_size; i++) {
    for (j = 0; j < out_features; j++) {
        acc = 0;
        for (k = 0; k < in_features; k++) {
            acc += X[i,k] * W[j,k];  // Accumulator stays in register
        }
        O[i,j] = acc;
    }
}
```

**Characteristics:**
- Accumulator (`acc`) is kept in a register, minimizing output writes
- Each X[i,k] and W[j,k] element is loaded once per (i,j) pair
- Good for minimizing accumulation memory traffic

### Input-Stationary / Outer-Product (This Directory)

Loop order: `for k (in_features), for i (batch), for j (out_features)`

```c
// Initialize O to zero
for (k = 0; k < in_features; k++) {
    for (i = 0; i < batch_size; i++) {
        x_ik = X[i,k];  // Load once, use for all j
        for (j = 0; j < out_features; j++) {
            O[i,j] += x_ik * W[j,k];  // Rank-1 update
        }
    }
}
```

**Characteristics:**
- Input element X[i,k] is loaded once and broadcast to all j iterations
- Computes rank-1 outer product update: X[:,k] ⊗ W[:,k]ᵀ
- Better input reuse when batch_size and out_features are large
- Useful for hardware with broadcast/multicast capabilities
- Output array is updated incrementally (more memory writes)

## When to Use Each Dataflow

| Scenario | Preferred Dataflow |
|----------|-------------------|
| Small batch, large in_features | Output-stationary |
| Large batch, large out_features | Input-stationary |
| Limited registers | Output-stationary |
| SIMD broadcast support | Input-stationary |
| Minimize memory writes | Output-stationary |
| Maximize input reuse | Input-stationary |

## Supported Operations

All operations from the baseline are implemented with outer-product dataflow:

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

Enable outer-product optimizations:

```makefile
OPT += outer
include $(MICO_DIR)/targets/common.mk
```

## Verification

Results should match exactly with the reference implementation in `src/mico/qmatmul_ref.c`. Use the test file in `test/mico_outer_test.c` to verify correctness.

## References

- "Efficient Processing of Deep Neural Networks: A Tutorial and Survey" - Sze et al., 2017
- "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs" - Chen et al., 2016
