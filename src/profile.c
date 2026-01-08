#include "profile.h"

#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <time.h>
#include <stdio.h>
#endif

__attribute__((weak)) long int MiCo_time(){
    #ifdef USE_HOST
    return clock() / (CLOCKS_PER_SEC / 1000000); // Convert to ms
    #else
    #ifdef USE_CHIPYARD
    unsigned long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
    #endif
    return 0;
    #endif
}

void MiCo_print_profilers(){
    printf("QUANT_TIMER: %ld\n", QUANT_TIMER);
    printf("QMATMUL_TIMER: %ld\n", QMATMUL_TIMER);
    printf("IM2COL_TIMER: %ld\n", IM2COL_TIMER);
}
