# MiCo Quantized Pooling Tests

This directory contains unit tests and benchmarks for the quantized pooling operations implemented in MiCo-Lib.

## Files

- `q8_avgpool_test.h`: Test configuration parameters for average pooling
- `q8_maxpool_test.h`: Test configuration parameters for max pooling
- `test_qpooling.c`: Complete test suite verifying correctness vs reference implementations
- `bench_qpooling.c`: Performance benchmark for typical CNN pooling scenarios
- `Makefile.qpooling`: Build system for tests and benchmarks

## Building

To build both the test and benchmark executables:

```bash
make -f Makefile.qpooling
```

To build only the test executable:

```bash
make -f Makefile.qpooling test_qpooling
```

To build only the benchmark executable:

```bash
make -f Makefile.qpooling bench_qpooling
```

## Running Tests

To run the correctness tests:

```bash
make -f Makefile.qpooling test
```

or directly:

```bash
./test_qpooling
```

### Expected Output

All 8 tests should pass:
- 4 average pooling tests with various kernel/stride/padding combinations
- 4 max pooling tests with various kernel/stride/padding combinations

Example output:
```
========================================
Quantized Pooling Test Suite
========================================
...
========================================
Test Summary: 8/8 tests passed
========================================
```

## Running Benchmarks

To run performance benchmarks:

```bash
make -f Makefile.qpooling bench
```

or directly:

```bash
./bench_qpooling
```

### Benchmark Scenarios

The benchmark tests typical CNN pooling configurations:
- Early layer pooling (112x112 -> 56x56)
- Mid layer pooling (56x56 -> 28x28)
- Late layer pooling (28x28 -> 14x14)
- Global pooling (7x7 -> 6x6)

Performance is reported in GOps/s (billion operations per second).

## Test Coverage

The test suite covers:
- ✅ Kernel sizes: 2x2, 3x3
- ✅ Strides: 1, 2
- ✅ Padding: 0, 1
- ✅ Various input sizes and channel counts
- ✅ Comparison against reference implementations
- ✅ Quantization error tolerance validation

## Clean Up

To remove build artifacts:

```bash
make -f Makefile.qpooling clean
```
