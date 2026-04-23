#include "scratchpad.h"

#ifdef SPRAM
typedef struct {
    uint8_t *previous_cursor;
    size_t reserved_size;
    size_t requested_size;
    uint32_t magic;
} scratch_block_header_t;

typedef struct {
    uint8_t *base;
    uint8_t *limit;
    uint8_t *head;
    uint8_t *tail;
    size_t peak_usage;
    int initialized;
} scratch_arena_t;

enum {
    SCRATCH_HEAD_MAGIC = 0x53484348u,
    SCRATCH_TAIL_MAGIC = 0x53484354u,
};

static scratch_arena_t scratch_arena = {0};

static uintptr_t scratch_align_up(uintptr_t value, uint32_t alignment){
    return (value + alignment - 1u) & ~((uintptr_t)alignment - 1u);
}

static uintptr_t scratch_align_down(uintptr_t value, uint32_t alignment){
    return value & ~((uintptr_t)alignment - 1u);
}

static uint32_t scratch_next_pow2(uint32_t value){
    uint32_t power = (uint32_t)sizeof(uintptr_t);
    while (power < value) {
        power <<= 1u;
    }
    return power;
}

static uint32_t scratch_normalize_alignment(uint32_t alignment){
    uint32_t normalized = alignment == 0 ? SCRATCH_DEFAULT_ALIGNMENT : alignment;
    if (normalized < (uint32_t)sizeof(uintptr_t)) {
        normalized = (uint32_t)sizeof(uintptr_t);
    }
    if ((normalized & (normalized - 1u)) != 0u) {
        normalized = scratch_next_pow2(normalized);
    }
    return normalized;
}

static void scratch_memzero(void *ptr, size_t size){
    uint8_t *cursor = (uint8_t *)ptr;
    while (size--) {
        *cursor++ = 0;
    }
}

static void scratch_init_if_needed(void){
    if (scratch_arena.initialized) {
        return;
    }

    scratch_arena.base = __scratchpad_start;
    scratch_arena.limit = __scratchpad_end;
    if (scratch_arena.limit < scratch_arena.base) {
        scratch_arena.limit = scratch_arena.base;
    }

    scratch_arena.head = scratch_arena.base;
    scratch_arena.tail = scratch_arena.limit;
    scratch_arena.peak_usage = 0;
    scratch_arena.initialized = 1;
}

static size_t scratch_current_usage(void){
    return (size_t)(scratch_arena.head - scratch_arena.base) +
           (size_t)(scratch_arena.limit - scratch_arena.tail);
}

static void scratch_update_peak(void){
    size_t current = scratch_current_usage();
    if (current > scratch_arena.peak_usage) {
        scratch_arena.peak_usage = current;
    }
}

static int scratch_ptr_in_arena(const uint8_t *ptr){
    return ptr >= scratch_arena.base && ptr < scratch_arena.limit;
}

static void scratch_report_failure(const char *kind, size_t size, uint32_t alignment){
    printf("WARNING: %s failed for %u bytes (align %u), used %u / %u, free %u\n",
        kind,
        (unsigned int)size,
        (unsigned int)alignment,
        (unsigned int)scratch_bytes_used(),
        (unsigned int)scratch_bytes_total(),
        (unsigned int)scratch_bytes_free());
}

static void* scratch_alloc_head(size_t size, uint32_t alignment){
    uint8_t *start;
    uintptr_t payload_addr;
    scratch_block_header_t *header;
    uint8_t *new_head;

    scratch_init_if_needed();
    if (size == 0) {
        return NULL;
    }

    alignment = scratch_normalize_alignment(alignment);
    start = scratch_arena.head;
    payload_addr = scratch_align_up((uintptr_t)start + sizeof(*header), alignment);
    if (payload_addr > UINTPTR_MAX - size) {
        scratch_report_failure("scratch_malloc", size, alignment);
        return NULL;
    }

    header = (scratch_block_header_t *)(payload_addr - sizeof(*header));
    new_head = (uint8_t *)(payload_addr + size);
    if (new_head > scratch_arena.tail) {
        scratch_report_failure("scratch_malloc", size, alignment);
        return NULL;
    }

    header->previous_cursor = start;
    header->reserved_size = (size_t)(new_head - start);
    header->requested_size = size;
    header->magic = SCRATCH_HEAD_MAGIC;

    scratch_arena.head = new_head;
    scratch_update_peak();
    return (void *)payload_addr;
}

static void* scratch_alloc_tail(size_t size, uint32_t alignment){
    uintptr_t payload_addr;
    scratch_block_header_t *header;
    uint8_t *new_tail;

    scratch_init_if_needed();
    if (size == 0) {
        return NULL;
    }

    alignment = scratch_normalize_alignment(alignment);
    if (size > (size_t)(scratch_arena.tail - scratch_arena.head)) {
        scratch_report_failure("scratch_temp_malloc", size, alignment);
        return NULL;
    }

    payload_addr = scratch_align_down((uintptr_t)scratch_arena.tail - size, alignment);
    if (payload_addr < (uintptr_t)scratch_arena.head + sizeof(*header)) {
        scratch_report_failure("scratch_temp_malloc", size, alignment);
        return NULL;
    }

    header = (scratch_block_header_t *)(payload_addr - sizeof(*header));
    new_tail = (uint8_t *)header;
    if (new_tail < scratch_arena.head) {
        scratch_report_failure("scratch_temp_malloc", size, alignment);
        return NULL;
    }

    header->previous_cursor = scratch_arena.tail;
    header->reserved_size = (size_t)(scratch_arena.tail - new_tail);
    header->requested_size = size;
    header->magic = SCRATCH_TAIL_MAGIC;

    scratch_arena.tail = new_tail;
    scratch_update_peak();
    return (void *)payload_addr;
}

void* scratch_malloc(size_t size, uint32_t alignment){
    return scratch_alloc_head(size, alignment);
}

void* scratch_calloc(size_t count, size_t size, uint32_t alignment){
    size_t total_size;
    void *ptr;

    if (count == 0 || size == 0) {
        return NULL;
    }
    if (count > SIZE_MAX / size) {
        scratch_report_failure("scratch_calloc", SIZE_MAX, alignment);
        return NULL;
    }

    total_size = count * size;
    ptr = scratch_alloc_head(total_size, alignment);
    if (ptr != NULL) {
        scratch_memzero(ptr, total_size);
    }
    return ptr;
}

void* scratch_temp_malloc(size_t size, uint32_t alignment){
    return scratch_alloc_tail(size, alignment);
}

void* scratch_temp_calloc(size_t count, size_t size, uint32_t alignment){
    size_t total_size;
    void *ptr;

    if (count == 0 || size == 0) {
        return NULL;
    }
    if (count > SIZE_MAX / size) {
        scratch_report_failure("scratch_temp_calloc", SIZE_MAX, alignment);
        return NULL;
    }

    total_size = count * size;
    ptr = scratch_alloc_tail(total_size, alignment);
    if (ptr != NULL) {
        scratch_memzero(ptr, total_size);
    }
    return ptr;
}

void scratch_free(void* ptr){
    uint8_t *payload = (uint8_t *)ptr;
    scratch_block_header_t *header;

    scratch_init_if_needed();
    if (payload == NULL) {
        return;
    }
    if (payload < scratch_arena.base + sizeof(*header) || payload > scratch_arena.limit) {
        printf("WARNING: scratch_free ignored invalid pointer %x\n", (unsigned int)(uintptr_t)payload);
        return;
    }

    header = (scratch_block_header_t *)(payload - sizeof(*header));
    if (!scratch_ptr_in_arena((uint8_t *)header)) {
        printf("WARNING: scratch_free ignored invalid pointer %x\n", (unsigned int)(uintptr_t)payload);
        return;
    }

    if (header->magic == SCRATCH_HEAD_MAGIC) {
        if (scratch_arena.head == header->previous_cursor + header->reserved_size) {
            scratch_arena.head = header->previous_cursor;
            return;
        }
    } else if (header->magic == SCRATCH_TAIL_MAGIC) {
        if ((uint8_t *)header == scratch_arena.tail) {
            scratch_arena.tail = header->previous_cursor;
            return;
        }
    } else {
        printf("WARNING: scratch_free ignored unknown pointer %x\n", (unsigned int)(uintptr_t)payload);
        return;
    }

    printf("WARNING: scratch_free only supports the most recent scratch allocation\n");
}

void scratch_checkpoint(scratch_checkpoint_t *checkpoint){
    scratch_init_if_needed();
    if (checkpoint == NULL) {
        return;
    }

    checkpoint->head = (uintptr_t)scratch_arena.head;
    checkpoint->tail = (uintptr_t)scratch_arena.tail;
}

void scratch_restore(const scratch_checkpoint_t *checkpoint){
    scratch_init_if_needed();
    if (checkpoint == NULL) {
        return;
    }

    if ((uint8_t *)checkpoint->head < scratch_arena.base ||
        (uint8_t *)checkpoint->head > scratch_arena.limit ||
        (uint8_t *)checkpoint->tail < scratch_arena.base ||
        (uint8_t *)checkpoint->tail > scratch_arena.limit ||
        checkpoint->head > checkpoint->tail) {
        printf("WARNING: scratch_restore ignored invalid checkpoint\n");
        return;
    }

    scratch_arena.head = (uint8_t *)checkpoint->head;
    scratch_arena.tail = (uint8_t *)checkpoint->tail;
}

void scratch_reset(void){
    scratch_init_if_needed();
    scratch_arena.head = scratch_arena.base;
    scratch_arena.tail = scratch_arena.limit;
    scratch_arena.peak_usage = 0;
}

size_t scratch_bytes_total(void){
    scratch_init_if_needed();
    return (size_t)(scratch_arena.limit - scratch_arena.base);
}

size_t scratch_bytes_used(void){
    scratch_init_if_needed();
    return scratch_current_usage();
}

size_t scratch_bytes_peak(void){
    scratch_init_if_needed();
    return scratch_arena.peak_usage;
}

size_t scratch_bytes_free(void){
    scratch_init_if_needed();
    return (size_t)(scratch_arena.tail - scratch_arena.head);
}
#endif