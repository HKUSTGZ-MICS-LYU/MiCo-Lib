Unrolled Version of MiCo Kernels.

All inner loops are unrolled to handle 32 MACs per loops.
However, all MatMuls in the networks should be larger than 32 elements.