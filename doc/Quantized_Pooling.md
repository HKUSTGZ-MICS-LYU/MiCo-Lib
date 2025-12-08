# Quantized Pooling Implementation

## Overview

This document describes the quantized pooling implementation in MiCo-Lib, which implements 2D pooling operations (average and max) for int8 quantized tensors using an im2col-based approach.

## API

### Functions

```c
void MiCo_Q8_AvgPool2D(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);

void MiCo_Q8_MaxPool2D(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);
```

### Reference Implementations (for testing)

```c
#ifdef REF
void MiCo_Q8_AvgPool2D_Ref(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);
void MiCo_Q8_MaxPool2D_Ref(Tensor4D_Q8 *y, const Tensor4D_Q8 *x, 
    const size_t kernel_size, const size_t stride, const size_t padding);
#endif
```

## Data Layout

- **Input/Output Format**: NCHW (batch, channels, height, width)
- **Quantization**: INT8 with per-tensor scale
- **Accumulation**: INT32 accumulators to prevent overflow

## Supported Configurations

### Kernel Sizes
- 2x2
- 3x3

### Strides
- 1 (overlapping windows)
- 2 (non-overlapping windows)

### Padding
- 0 (no padding, valid convolution)
- 1 (pad with zeros/appropriate values)

## Quantization Handling

### Scale Preservation
Both pooling operations preserve the input scale for the output:
```c
y->scale = x->scale;
```

This is valid because:
- **Max pooling**: The maximum value maintains the same quantization relationship
- **Average pooling**: Averaging in quantized space produces results in the same scale

### Zero-Point Assumptions
- Symmetric quantization with zero-point at 0 is assumed
- Padding values for average pooling are handled correctly (excluded from average computation)
- Padding values for max pooling use INT8_MIN (-128) to ensure they don't affect the result

## Implementation Details

### Im2Col Transformation

The `im2col_pool_q8` function transforms the input tensor into a column matrix where each column contains the elements of one pooling window:

```
Input (NCHW):          Im2Col Output:
[C, H, W]       ->     [out_h*out_w, C*kernel_size*kernel_size]
```

This transformation allows pooling to be expressed as operations on columns:
- **Average pooling**: Sum over column elements and divide by count
- **Max pooling**: Find maximum over column elements

### Memory Layout

For a 3x3 pooling kernel on a single-channel 5x5 input:
- Each output position corresponds to one row in the im2col output
- Each row contains 9 values (3x3 kernel window)
- Total im2col output size: (out_h * out_w) x 9

### Padding Handling

**Average Pooling**:
- Padding areas are excluded from the sum
- Division is by the count of valid (non-padded) elements only
- This matches the behavior of typical deep learning frameworks

**Max Pooling**:
- Padding values are set to INT8_MIN (-128)
- These values never affect the maximum due to being smaller than any valid int8 value
- Only valid input positions are considered

## Performance Characteristics

### Average Pooling
- Time complexity: O(N * C * out_h * out_w * kernel_size²)
- Space complexity: O(out_h * out_w * kernel_size²) for im2col buffer
- The im2col overhead is amortized when processing multiple channels

### Max Pooling  
- Time complexity: O(N * C * out_h * out_w * kernel_size²)
- Space complexity: O(out_h * out_w * kernel_size²) for im2col buffer
- Direct max operation is more efficient than matmul-based approach

## Testing

Test files are provided in `test/`:
- `q8_avgpool_test.h`: Test configurations for average pooling
- `q8_maxpool_test.h`: Test configurations for max pooling
- `test_qpooling.c`: Complete test suite

To build and run tests:
```bash
cd test
make -f Makefile.qpooling
./test_qpooling
```

### Test Coverage

Tests verify:
1. Correctness vs reference implementations
2. Various input shapes (4x4, 6x6, 7x7, 8x8)
3. Different channel counts (2, 3, 4, 8)
4. All supported kernel sizes (2x2, 3x3)
5. All supported strides (1, 2)
6. All supported padding values (0, 1)

### Numerical Accuracy

Quantized operations may introduce rounding errors. The test suite allows a tolerance of ±2 in quantized space, which is typical for int8 quantization with averaging operations.

## Integration with Matmul Backend

While the current implementation uses im2col transformation followed by direct averaging or max operations, the im2col step provides a foundation for potential future integration with the optimized matmul backend:

1. **For Average Pooling**: Could be expressed as im2col + matmul with a matrix of averaging weights
2. **For Max Pooling**: Direct implementation is more efficient, but im2col provides shared infrastructure

The current implementation prioritizes correctness and clarity while maintaining the structure needed for future optimization.

## Example Usage

```c
#include "mico_qnn.h"
#include "mico_quant.h"

// Setup input tensor (already quantized)
Tensor4D_Q8 input;
input.shape[0] = 1;     // batch size
input.shape[1] = 16;    // channels
input.shape[2] = 32;    // height
input.shape[3] = 32;    // width
input.scale = 0.05f;
input.wq = 8;
input.data = /* quantized data */;

// Setup output tensor
int out_h = (32 - 2) / 2 + 1;  // 16
int out_w = (32 - 2) / 2 + 1;  // 16
Tensor4D_Q8 output;
output.shape[0] = 1;
output.shape[1] = 16;
output.shape[2] = out_h;
output.shape[3] = out_w;
output.wq = 8;
output.data = malloc(1 * 16 * out_h * out_w * sizeof(int8_t));

// Perform 2x2 max pooling with stride 2, no padding
MiCo_Q8_MaxPool2D(&output, &input, 2, 2, 0);

// output now contains pooled results with same scale as input
```

## Future Enhancements

Potential future improvements:
1. Support for additional kernel sizes (e.g., 4x4, 5x5)
2. Support for non-square kernels
3. Support for per-channel quantization
4. SIMD optimizations for im2col transformation
5. Integration with optimized matmul kernels for average pooling
6. Support for NHWC layout in addition to NCHW
