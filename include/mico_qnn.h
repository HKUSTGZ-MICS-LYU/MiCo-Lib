#ifndef __MICO_QNN_H
#define __MICO_QNN_H

#include "mico_nn.h"
#include "qtypes.h"

void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);

#endif // __MICO_QNN_H