# Quantized Computing Kernel Specification

## Linear Layer

```
# bX, bW - bitwidths of X and W

float   W[N,M];             // FP32 Weight
int{bW} qW[N,M];            // INT Quantized Weight
float   sW;                 // Quantized Weight Scale

float B[M];

# Export Time
qW, sW = Quantize(W, bW);   // Weight Quantization

# Inference Time
float   X[N];               // FP32 Activation
int{bX} qX[N];              // INT Quantized Activation
float   sX;                 // Quantized Activation Scale

float   O[M];               // FP32 Output
int32   qO[M];              // INT Quantized Output

O = B;                      // Initialize Output with Bias
qX, sX = Quantize(X, bX);   // Activation Quantization
qO = qX * qW;               // Integer Vec X Mat Operations
O += qO * sW * sX;          // Re-Quantization
```

## Conv2D Layer (Im2Col)
