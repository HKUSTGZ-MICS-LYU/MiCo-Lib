#ifndef _SCRATCHPAD_H_
#define _SCRATCHPAD_H_

#include "soc_stdlib.h"

#ifdef SPRAM
#ifndef SCRATCH_DEFAULT_ALIGNMENT
#define SCRATCH_DEFAULT_ALIGNMENT 64U
#endif

extern uint8_t __scratchpad_start[], __scratchpad_end[];
extern uint8_t __onchip_data_start[], __onchip_data_end[];

typedef struct {
    uintptr_t head;
    uintptr_t tail;
} scratch_checkpoint_t;

void* scratch_malloc(size_t size, uint32_t alignment);
void* scratch_calloc(size_t count, size_t size, uint32_t alignment);
void* scratch_temp_malloc(size_t size, uint32_t alignment);
void* scratch_temp_calloc(size_t count, size_t size, uint32_t alignment);
void scratch_free(void* ptr);
void scratch_checkpoint(scratch_checkpoint_t *checkpoint);
void scratch_restore(const scratch_checkpoint_t *checkpoint);
void scratch_reset(void);
size_t scratch_bytes_total(void);
size_t scratch_bytes_used(void);
size_t scratch_bytes_peak(void);
size_t scratch_bytes_free(void);
#endif

#endif // _SCRATCHPAD_H_