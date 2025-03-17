#include "profile.h"

__attribute__((weak)) long int MiCo_time(){
    // return clock();
    return 0;
}