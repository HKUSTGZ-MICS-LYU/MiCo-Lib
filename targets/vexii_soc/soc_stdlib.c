// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

#include "soc_stdlib.h"

void setStats(int enable)
{

}

void exit(int error){
    extern void pass();
    extern void fail();
    if(error) fail(); else pass();
    while(1);
}

void delay(int ms){
    clint_uDelay(ms*1000, SOC_FREQUENCY_HZ ,CLINT);
}

long time(){
    return clint_getTime(CLINT);
}
void time_stamp(){
    static long last_time = 0;
    long curr_time = time();
    printf("Time: %ld\n", curr_time - last_time);
    last_time = curr_time;
}

int puts(const char *s){
  while (*s) {
    putchar(*s);
    s++;
  }
  putchar('\n');
  return 0;
}

int putchar(int c){
    uart_write(UART_A, (char)c);
    return c;
}

// llama2 tokenizer

static int hex2int(char c){
    if(c >= '0' && c <= '9'){
        return c - '0';
    }else if(c >= 'a' && c <= 'f'){
        return c - 'a' + 10;
    }else if(c >= 'A' && c <= 'F'){
        return c - 'A' + 10;
    }
    return 0;
}


int strlen(const char *s){
    int i = 0;
    while(s[i] != '\0'){
        i++;
    }
    return i;
}

void *_sbrk (incr)
     int incr;
{
    extern char   end; /* Set by linker.  */
    extern const char   _sp; /* Set by linker.  */

    static char * heap_end;
    char *        prev_heap_end;
    
    if (heap_end == 0)
      heap_end = &end;

    prev_heap_end = heap_end;

    /* Check if allocation would exceed the maximum heap size */
    if ((heap_end + incr > &_sp)) {
        printf("WARNING: _sbrk failed to allocate %d bytes - heap usage %ld/%d\n", 
            incr, heap_end - &end, (int)&_sp - (int)&end);
        return (void *)-1;  /* Standard sbrk error return value */
    }

    heap_end += incr;
    #ifdef DEBUG
    printf("sbrk: %d, curr_heap_end: %x, prog_end: %x\n", incr, heap_end, &end);
    #endif
    return (void *) prev_heap_end;
}

void* bsearch( const void *key, const void *ptr, size_t count, size_t size,
               int (*comp)(const void*, const void*) ){
  
    const char *base = ptr;
    size_t lim;
    int cmp;
    const void *p;

    for(lim = count; lim !=0; lim >>=1){
        p = base + (lim >> 1) * size;
        cmp = comp(key, p);
        if (cmp == 0){
            return (void *)p;
        }
        if (cmp > 0){
            base = (char*)p + size;
            lim --;
        }
    }
    return (void*)NULL;
}

// Apporximate Math
float approx_sinf(float x){
    return 0;
}

float approx_cosf(float x){
    return 0;
}