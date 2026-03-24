// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

#ifndef _SOC_STDLIB_H_
#define _SOC_STDLIB_H_

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>

#include "driver/io.h"
#include "driver/peripherals.h"
#include "driver/type.h"
#include "driver/plic.h"
#include "driver/clint.h"
#include "driver/uart.h"
#include "driver/dma.h"


#define SOC_FREQUENCY_HZ 100000000 // 100 MHz

void setStats(int enable);

void exit(int error);

long time();
void time_stamp();

void delay(int ms);

int puts(const char *s);
int putchar(int c);

// stdio.h
int printf(const char *format, ...);
int sprintf(char *str, const char *format, ... );
int sscanf(const char *str, const char *format, ...);

#ifdef SPRAM
extern uint8_t __onchip_data_start[], __onchip_data_end[];
void* scratch_malloc(size_t size, uint32_t alignment);
void scratch_free(void* ptr);
void scratch_reset(void);
#endif

#endif // _SOC_STDLIB_H_
