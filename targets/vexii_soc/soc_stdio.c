#include "soc_stdlib.h"

// --- minimal printf core ---

typedef void (*putc_fn)(void* ctx, char c);

static void sink_uart(void* ctx, char c) {
    (void)ctx;
    putchar((unsigned char)c);
}

struct buf_sink {
    char* dst;
    unsigned long cap;  // includes space for '\0'
    unsigned long len;  // number of chars produced (may exceed cap-1)
};

static void sink_buf(void* ctx, char c) {
    struct buf_sink* s = (struct buf_sink*)ctx;
    if (s->len + 1 < s->cap) {
        s->dst[s->len] = c;
    }
    s->len++;
}

static int is_digit(char c) { return c >= '0' && c <= '9'; }

static unsigned long long uabs_ll(long long v) {
    return (v < 0) ? (unsigned long long)(-v) : (unsigned long long)v;
}

static int utoa_base(unsigned long long val, unsigned base, int upper, char* tmp) {
    // returns length, digits placed in tmp reversed
    const char* digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0;
    if (val == 0) {
        tmp[n++] = '0';
        return n;
    }
    while (val) {
        tmp[n++] = digits[val % base];
        val /= base;
    }
    return n;
}

static int kvprintf(putc_fn out, void* ctx, const char* fmt, va_list ap) {
    int count = 0;

    for (; *fmt; ++fmt) {
        if (*fmt != '%') {
            out(ctx, *fmt);
            count++;
            continue;
        }
        // parse flags and width
        ++fmt;
        char pad = ' ';
        int width = 0;
        int long_count = 0; // 0: int, 1: long, 2: long long

        if (*fmt == '0') { pad = '0'; ++fmt; }
        while (is_digit(*fmt)) { width = width * 10 + (*fmt - '0'); ++fmt; }
        while (*fmt == 'l') { long_count++; ++fmt; }

        char spec = *fmt ? *fmt : '\0';
        if (!spec) break;

        if (spec == '%') {
            out(ctx, '%'); count++;
            continue;
        }

        if (spec == 'c') {
            int ch = va_arg(ap, int);
            out(ctx, (char)ch); count++;
            continue;
        }

        if (spec == 's') {
            const char* s = va_arg(ap, const char*);
            if (!s) s = "(null)";
            // width padding (right-align)
            int len = 0; for (const char* p = s; *p; ++p) len++;
            for (int i = len; i < width; ++i) { out(ctx, pad); count++; }
            for (; *s; ++s) { out(ctx, *s); count++; }
            continue;
        }

        // numeric formats
        int base = 10;
        int upper = 0;
        int is_signed = 0;

        switch (spec) {
            case 'd': case 'i': is_signed = 1; base = 10; break;
            case 'u': base = 10; break;
            case 'x': base = 16; upper = 0; break;
            case 'X': base = 16; upper = 1; break;
            case 'p': base = 16; upper = 0; long_count = sizeof(void*) == 8 ? 2 : 1; break;
            default:
                // unsupported specifier: print it literally
                out(ctx, '%'); out(ctx, spec); count += 2;
                continue;
        }

        long long sval = 0;
        unsigned long long uval = 0;
        int negative = 0;

        if (spec == 'p') {
            // print 0x + pointer
            unsigned long long v = (unsigned long long)(unsigned long)va_arg(ap, void*);
            out(ctx, '0'); out(ctx, 'x'); count += 2;
            char tmp[32];
            int n = utoa_base(v, 16, 0, tmp);
            // no width on %p by default
            while (n--) { out(ctx, tmp[n]); count++; }
            continue;
        }

        if (is_signed) {
            if (long_count >= 2)      sval = va_arg(ap, long long);
            else if (long_count == 1) sval = va_arg(ap, long);
            else                      sval = va_arg(ap, int);
            if (sval < 0) { negative = 1; uval = (unsigned long long)(-sval); }
            else          { uval = (unsigned long long)sval; }
        } else {
            if (long_count >= 2)      uval = va_arg(ap, unsigned long long);
            else if (long_count == 1) uval = va_arg(ap, unsigned long);
            else                      uval = va_arg(ap, unsigned int);
        }

        char tmp[32];
        int n = utoa_base(uval, (unsigned)base, upper, tmp);

        int sign_chars = negative ? 1 : 0;
        int total_width = (n + sign_chars);
        // padding (only left pad for simplicity)
        for (int i = total_width; i < width; ++i) { out(ctx, pad); count++; }
        if (negative) { out(ctx, '-'); count++; }
        while (n--) { out(ctx, tmp[n]); count++; }
    }
    return count;
}

// printf family

int vprintf(const char* format, va_list ap) {
    va_list aq;
    va_copy(aq, ap);
    int n = kvprintf(sink_uart, 0, format, aq);
    va_end(aq);
    return n;
}

int printf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int n = vprintf(format, ap);
    va_end(ap);
    return n;
}

int sprintf(char *str, const char *format, ... ){
    va_list ap;
    va_start(ap, format);
    int n = vsprintf(str, format, ap);
    va_end(ap);
    return n;
}
