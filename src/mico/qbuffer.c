#include "mico_nn.h"

qbyte MiCo_QBuffer[QUANTIZE_BUFFER_SIZE] __attribute__((aligned(MICO_ALIGN)));
MiCo_QX_Buffer MiCo_QX_Buffer_Global = {
    MiCo_QBuffer, 
    NULL, 
    QUANTIZE_BUFFER_SIZE, 
    0,
    0};
