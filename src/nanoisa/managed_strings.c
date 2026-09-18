/* I retain byte strings with context-local handles. No opcode/profile calls
 * this core yet; my later lowering must supply frame/global cleanup. */
#include "managed_strings.h"
#include <limits.h>
#ifndef __wasm32__
#include <stdlib.h>
_Static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8,
               "I require the declared native 64-bit allocator ABI");
#endif

#ifdef NMS_TESTING
static uint64_t live_allocations;
#endif
static void copy_bytes(unsigned char *to, const unsigned char *from, uint64_t n) {
    /* Volatile byte accesses keep the freestanding Wasm core independent of
     * compiler-created memcpy/memmove imports, including optimized builds. */
    volatile unsigned char *dst = to;
    const volatile unsigned char *src = from;
    for (uint64_t i = 0; i < n; i++) dst[i] = src[i];
}

#ifdef __wasm32__
/* Free-block headers live inside memory owned by this allocator. The sole
 * external boundary is wasm-ld's heap base, after data and reserved stack. */
extern unsigned char __heap_base;
typedef struct FreeBlock { uint64_t size; struct FreeBlock *next; } FreeBlock;
_Static_assert(sizeof(FreeBlock) <= 16, "I reserve a 16-byte block header");
static FreeBlock *free_blocks;
static unsigned pool_initialized;

static void pool_insert(FreeBlock *block) {
    FreeBlock *previous = NULL, *next = free_blocks;
    while (next && (uintptr_t)next < (uintptr_t)block) {
        previous = next; next = next->next;
    }
    block->next = next;
    if (next && (uint64_t)(uintptr_t)block + block->size == (uintptr_t)next) {
        block->size += next->size;
        block->next = next->next;
    }
    if (previous) {
        if ((uint64_t)(uintptr_t)previous + previous->size == (uintptr_t)block) {
            previous->size += block->size;
            previous->next = block->next;
        } else previous->next = block;
    } else free_blocks = block;
}
static void pool_init(void) {
    if (pool_initialized) return;
    pool_initialized = 1;
    uint64_t start = ((uint64_t)(uintptr_t)&__heap_base + 15) & ~UINT64_C(15);
    uint64_t end = (uint64_t)__builtin_wasm_memory_size(0) * 65536;
    if (start < end && end - start >= 32) {
        FreeBlock *block = (FreeBlock *)(uintptr_t)start;
        block->size = end - start;
        block->next = NULL;
        free_blocks = block;
    }
}
static void *backend_allocate(uint64_t bytes) {
    /* Widen before header/alignment arithmetic; no wrapped request reaches
     * the free list or memory.grow. All published block sizes are aligned. */
    if (bytes > UINT32_MAX - UINT64_C(31)) return NULL;
    uint64_t need = (bytes + 16 + 15) & ~UINT64_C(15);
    pool_init();
    for (unsigned attempt = 0; attempt < 2; attempt++) {
        FreeBlock **link = &free_blocks;
        for (FreeBlock *block = *link; block; link = &block->next, block = *link) {
            if (block->size < need) continue;
            uint64_t remaining = block->size - need;
            if (remaining >= 32) {
                FreeBlock *tail = (FreeBlock *)((unsigned char *)block + (size_t)need);
                tail->size = remaining; tail->next = block->next;
                *link = tail; block->size = need;
            } else *link = block->next;
            block->next = NULL;
            return (unsigned char *)block + 16;
        }
        if (attempt) break;
        uint64_t current = __builtin_wasm_memory_size(0);
        uint64_t missing = need;
        FreeBlock *tail = free_blocks;
        while (tail && tail->next) tail = tail->next;
        if (tail && (uint64_t)(uintptr_t)tail + tail->size == current * 65536)
            missing -= tail->size; /* No free block fitted, so this is positive. */
        uint64_t pages = (missing + 65535) / 65536;
        if (current > 65536 || pages > 65536 - current) return NULL;
        size_t old = __builtin_wasm_memory_grow(0, (size_t)pages);
        if (old == (size_t)-1) return NULL;
        FreeBlock *added = (FreeBlock *)(uintptr_t)((uint64_t)old * 65536);
        added->size = pages * 65536; added->next = NULL;
        pool_insert(added);
    }
    return NULL;
}
static void backend_free(void *memory) {
    if (memory) pool_insert((FreeBlock *)((unsigned char *)memory - 16));
}
#else
static void *backend_allocate(uint64_t bytes) {
    if (bytes > SIZE_MAX) return NULL;
    return malloc((size_t)bytes);
}
static void backend_free(void *memory) { free(memory); }
#endif

static void *allocate(NmsRuntime *runtime, uint64_t bytes) {
#ifdef NMS_TESTING
    if (!runtime->fail_after) return NULL;
    if (runtime->fail_after != UINT64_MAX) runtime->fail_after--;
#else
    (void)runtime;
#endif
    void *memory = backend_allocate(bytes);
#ifdef NMS_TESTING
    if (memory) live_allocations++;
#endif
    return memory;
}
static void deallocate(void *memory) {
    if (!memory) return;
#ifdef NMS_TESTING
    live_allocations--;
#endif
    backend_free(memory);
}
void nms_init(NmsRuntime *runtime, const NmsView *literals, uint32_t count) {
    runtime->literals = literals;
    runtime->literal_count = count;
    runtime->slots = NULL;
    runtime->capacity = runtime->free_head = 0;
    runtime->live_bytes = runtime->live_objects = 0;
    runtime->active = runtime->disposed = 0;
#ifdef NMS_TESTING
    runtime->fail_after = UINT64_MAX;
#endif
}
static NmsStatus slot_for(const NmsRuntime *runtime, NmsHandle handle, uint32_t *index) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    uint64_t raw = handle & ~NMS_DYNAMIC;
    if (!(handle & NMS_DYNAMIC) || !raw || raw > runtime->capacity ||
        !runtime->slots || !runtime->slots[raw].references) return NMS_STATE;
    *index = (uint32_t)raw;
    return NMS_OK;
}
NmsStatus nms_view(const NmsRuntime *runtime, NmsHandle handle, NmsView *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (handle & NMS_DYNAMIC) {
        uint32_t index;
        NmsStatus status = slot_for(runtime, handle, &index);
        if (status != NMS_OK) return status;
        out->data = runtime->slots[index].data;
        out->length = runtime->slots[index].length;
    } else {
        if (!handle || handle > runtime->literal_count || !runtime->literals)
            return NMS_STATE;
        const NmsView *literal = &runtime->literals[handle - 1];
        if (!literal->data) return NMS_STATE;
        out->data = literal->data; out->length = literal->length;
    }
    return NMS_OK;
}
NmsStatus nms_create(NmsRuntime *runtime, const unsigned char *data, uint64_t length,
                     NmsHandle *out) {
    if (!runtime || !out || (!data && length)) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (length > UINT32_MAX || length == SIZE_MAX ||
        length > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    uint32_t new_capacity = runtime->capacity;
    if (!runtime->free_head) {
        if (runtime->capacity >= UINT32_MAX - 1) return NMS_MEMORY;
        new_capacity = runtime->capacity ?
            (runtime->capacity > (UINT32_MAX - 1) / 2 ? UINT32_MAX - 1 : runtime->capacity * 2) : 8;
        if ((uint64_t)new_capacity + 1 > SIZE_MAX / sizeof(NmsSlot)) return NMS_MEMORY;
    }
    unsigned char *bytes = allocate(runtime, length + 1);
    if (!bytes) return NMS_MEMORY;
    copy_bytes(bytes, data, length);
    bytes[length] = 0;
    NmsSlot *slots = runtime->slots;
    if (new_capacity != runtime->capacity) {
        slots = allocate(runtime, ((uint64_t)new_capacity + 1) * sizeof(NmsSlot));
        if (!slots) { deallocate(bytes); return NMS_MEMORY; }
        for (uint64_t i = 0; i <= new_capacity; i++) {
            if (runtime->slots && i <= runtime->capacity) {
                slots[i].data = runtime->slots[i].data;
                slots[i].references = runtime->slots[i].references;
                slots[i].length = runtime->slots[i].length;
                slots[i].next_free = runtime->slots[i].next_free;
            } else {
                slots[i].data = NULL; slots[i].references = 0; slots[i].length = 0;
                slots[i].next_free = i < new_capacity ? (uint32_t)i + 1 : 0;
            }
        }
        NmsSlot *old = runtime->slots;
        runtime->slots = slots;
        runtime->free_head = runtime->capacity + 1;
        runtime->capacity = new_capacity;
        deallocate(old);
    }
    uint32_t index = runtime->free_head;
    runtime->free_head = slots[index].next_free;
    slots[index].data = bytes; slots[index].length = (uint32_t)length;
    slots[index].references = 1; slots[index].next_free = 0;
    runtime->live_bytes += length; runtime->live_objects++;
    *out = NMS_DYNAMIC | index;
    return NMS_OK;
}
NmsStatus nms_retain(NmsRuntime *runtime, NmsHandle handle) {
    if (!(handle & NMS_DYNAMIC)) { NmsView view; return nms_view(runtime, handle, &view); }
    uint32_t index;
    NmsStatus status = slot_for(runtime, handle, &index);
    if (status != NMS_OK) return status;
    if (runtime->slots[index].references == UINT64_MAX) return NMS_MEMORY;
    runtime->slots[index].references++;
    return NMS_OK;
}
NmsStatus nms_release(NmsRuntime *runtime, NmsHandle handle) {
    if (!(handle & NMS_DYNAMIC)) { NmsView view; return nms_view(runtime, handle, &view); }
    uint32_t index;
    NmsStatus status = slot_for(runtime, handle, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (--slot->references) return NMS_OK;
    runtime->live_bytes -= slot->length; runtime->live_objects--;
    deallocate(slot->data);
    slot->data = NULL; slot->length = 0;
    slot->next_free = runtime->free_head; runtime->free_head = index;
    return NMS_OK;
}
NmsStatus nms_begin(NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (runtime->active) return NMS_BUSY;
    runtime->active = 1;
    return NMS_OK;
}
uint64_t nms_finish(NmsRuntime *runtime, NmsStatus status, int32_t result) {
    if (!runtime || !runtime->active) return (uint64_t)NMS_STATE << 32;
    runtime->active = 0;
    if ((unsigned)status > NMS_STATE) status = NMS_STATE;
    return ((uint64_t)status << 32) | (status == NMS_OK ? (uint32_t)result : 0);
}
NmsStatus nms_dispose(NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->active) return NMS_BUSY;
    if (runtime->disposed) return NMS_OK;
    /* Terminal instance disposal invalidates all context-local handles.
     * Later lowering must first release frame/global roots normally. */
    for (uint64_t i = 1; i <= runtime->capacity; i++)
        if (runtime->slots[i].references) deallocate(runtime->slots[i].data);
    deallocate(runtime->slots);
    runtime->slots = NULL; runtime->capacity = runtime->free_head = 0;
    runtime->live_bytes = runtime->live_objects = 0;
    runtime->disposed = 1;
    return NMS_OK;
}
int nms_reserved_entry(const char *name) {
    if (!name) return 0;
    const char *reserved[] = {"nano_try_entry", "nano_dispose", "nano_runtime_", "nms_"};
    for (unsigned i = 0; i < sizeof reserved / sizeof reserved[0]; i++) {
        unsigned n = 0;
        while (reserved[i][n] && name[n] == reserved[i][n]) n++;
        if (!reserved[i][n] && (i >= 2 || !name[n])) return 1;
    }
    return 0;
}
#ifdef NMS_TESTING
void nms_test_fail_after(NmsRuntime *runtime, uint64_t count) { runtime->fail_after = count; }
uint64_t nms_test_live_allocations(void) { return live_allocations; }
uint64_t nms_test_memory_pages(void) {
#ifdef __wasm32__
    return __builtin_wasm_memory_size(0);
#else
    return 0;
#endif
}
#endif
