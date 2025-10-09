#ifndef PROFILE_H
#define PROFILE_H

long int MiCo_time();
void MiCo_print_profilers();
extern long QUANT_TIMER;
extern long QMATMUL_TIMER;
extern long IM2COL_TIMER;
extern long SOFTMAX_TIMER;

#endif // PROFILE_H