# MiCo-Lib API Reference

The main header file is `mico_nn.h`. This library provides functions for quantized matrix multiplication and convolution.

## Data Types

### Tensor Structures

MiCo-Lib uses `TensorND_Q8` structures to represent quantized tensors. Despite the name `Q8`, they support variable bit-widths for weights.

```c
typedef struct {
    size_t shape[N]; // Dimensions
    qbyte *data;     // Quantized data pointer
    float scale;     // Quantization scale factor
    qtype wq;        // Weight quantization bits (e.g., 1, 2, 4, 8)
} TensorND_Q8;
```
*   `Tensor1D_Q8`, `Tensor2D_Q8`, `Tensor3D_Q8`, `Tensor4D_Q8` are available.
*   `qbyte` is typically `int8_t` or `uint8_t`.
*   `qtype` is an integer type representing bit width.

### Global Buffer

```c
extern qbyte MiCo_QBuffer[QUANTIZE_BUFFER_SIZE];
```
A global buffer used for intermediate quantization operations. `QUANTIZE_BUFFER_SIZE` defaults to 32KB but can be overridden.

## Core Functions

### Linear (Fully Connected)

```c
void MiCo_bitlinear_f32(
    Tensor2D_F32 *y,          // Output tensor (FP32)
    const Tensor2D_F32 *x,    // Input tensor (FP32)
    const Tensor2D_Q8 *weight,// Quantized weight tensor
    const Tensor1D_F32 *bias, // Bias tensor (FP32)
    const qtype wq,           // Weight bit-width
    const qtype aq,           // Activation bit-width
    const size_t align        // Alignment requirement
);
```
Performs: $Y = X \times W + b$.
*   Inputs $X$ are dynamically quantized to `aq` bits.
*   Weights $W$ are pre-quantized to `wq` bits.

### Convolution (2D)

```c
void MiCo_bitconv2d_f32(
    Tensor4D_F32 *y,          // Output
    const Tensor4D_F32 *x,    // Input
    const Tensor4D_Q8 *weight,// Weights
    const Tensor1D_F32 *bias, // Bias
    const qtype wq,           // Weight bits
    const qtype aq,           // Activation bits
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t groups,
    const size_t align
);
```
Performs 2D convolution with quantized kernels.

### Convolution (1D)

```c
void MiCo_bitconv1d_f32(
    Tensor3D_F32 *y,
    const Tensor3D_F32 *x,
    const Tensor3D_Q8 *weight,
    const Tensor1D_F32 *bias,
    const qtype wq,
    const qtype aq,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t groups,
    const size_t align
);
```

## Quantization details

*   **Weights**: Must be pre-quantized offline (e.g., during model export).
*   **Activations**: Quantized dynamically at runtime using the specified `aq` bit-width.
*   **Alignment**: The `align` parameter specifies alignment constraints (usually 32 or 64 elements) for SIMD optimizations.
