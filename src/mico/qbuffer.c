#include "mico_nn.h"

#ifdef SPRAM
// If on Vexii MiCo SoC with Scratch Pad, the Quantized Buffer will be on Scratch Pad
#define ONCHIP_SECTION __attribute__((section(".onchip"), aligned(MICO_ALIGN)))
#else
#define ONCHIP_SECTION __attribute__((aligned(MICO_ALIGN)))
#endif

ONCHIP_SECTION qbyte MiCo_QBuffer[QUANTIZE_BUFFER_SIZE];

MiCo_QX_Buffer MiCo_QX_Buffer_Global = {
    MiCo_QBuffer,
    NULL,
    QUANTIZE_BUFFER_SIZE,
    0,
    0};