#include "profile.h"

#ifdef USE_HOST
#include <time.h>
#endif

__attribute__((weak)) long int MiCo_time(){
    #ifdef USE_HOST
    return clock();
    #else
    return 0;
    #endif
}