#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

extern char __heap_start__;
extern char __heap_end__;

static char *heap_ptr;

void * _sbrk(ptrdiff_t increment) {
    if (heap_ptr == NULL) {
        heap_ptr = &__heap_start__;
    }
    if (increment < 0 || heap_ptr > &__heap_end__ - increment) {
        errno = ENOMEM;
        return (void *)-1;
    }

    char *previous = heap_ptr;
    heap_ptr += increment;
    return previous;
}

int _close(int file) {
    (void)file;
    errno = EBADF;
    return -1;
}

int _fstat(int file, struct stat *status) {
    if (file != STDOUT_FILENO && file != STDERR_FILENO) {
        errno = EBADF;
        return -1;
    }
    if (status == NULL) {
        errno = EFAULT;
        return -1;
    }
    status->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file) {
    if (file != STDOUT_FILENO && file != STDERR_FILENO) {
        errno = EBADF;
        return 0;
    }
    return 1;
}

int _lseek(int file, int offset, int direction) {
    (void)offset;
    (void)direction;
    if (file != STDOUT_FILENO && file != STDERR_FILENO) {
        errno = EBADF;
        return -1;
    }
    return 0;
}

int _read(int file, char *buffer, int length) {
    (void)buffer;
    (void)length;
    if (file != STDIN_FILENO) {
        errno = EBADF;
    } else {
        errno = EIO;
    }
    return -1;
}

int _write(int file, char *buffer, int length) {
    (void)buffer;
    if (file != STDOUT_FILENO && file != STDERR_FILENO) {
        errno = EBADF;
        return -1;
    }
    return length;
}

int _getpid(void) {
    return 1;
}

int _kill(int pid, int signal) {
    (void)pid;
    (void)signal;
    errno = EINVAL;
    return -1;
}

void _exit(int status) {
    (void)status;
    __asm__ volatile("ebreak");
    for (;;) {
    }
}

void coralnpu_exception_handler(void) {
    __asm__ volatile("ebreak");
    for (;;) {
    }
}
