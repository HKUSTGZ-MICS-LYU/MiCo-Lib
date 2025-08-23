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

// typedef int size_t;

void setStats(int enable);

void exit(int error);

long time();
void time_stamp();

int puts(const char *s);
int putchar(int c);

// stdio.h
int printf(const char *format, ...);
int sprintf(char *str, const char *format, ... );
int sscanf(const char *str, const char *format, ...);

#endif // _SOC_STDLIB_H_