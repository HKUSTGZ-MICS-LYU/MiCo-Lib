#include "profile.h"

#ifdef USE_HOST
#include <time.h>
#endif

__attribute__((weak)) long int MiCo_time(){
    #ifdef USE_HOST
    return clock() / (CLOCKS_PER_SEC / 1000000); // Convert to ms
    #else
    return 0;
    #endif
}