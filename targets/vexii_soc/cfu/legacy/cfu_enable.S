#include "custom_asm.h"

.global cfu_enable
cfu_enable:
    li t1, 0x80000000
    csrs 0xBC0, t1
    ret