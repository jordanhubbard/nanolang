/* I retain byte strings with context-local handles. My managed lowering
 * supplies frame/global ownership and cleanup around this allocator core. */
#include "managed_strings.h"
#include "binary64_parse.h"
#include <limits.h>
#ifndef __wasm32__
#include <stdlib.h>
_Static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8,
               "I require the declared native 64-bit allocator ABI");
#endif

#ifdef NMS_TESTING
static uint64_t live_allocations;
#endif
#ifdef NMS_TEST_ALLOC_HOOKS
#ifndef NMS_TESTING
#error "I require NMS_TESTING for allocator observation hooks"
#endif
/* My fixture owns these nonallocating callbacks; production has no hooks. */
extern int nms_test_allocation_permitted(uint64_t bytes);
extern void nms_test_allocation_created(void *memory, uint64_t bytes);
extern void nms_test_allocation_destroyed(void *memory);
#endif
static void copy_bytes(unsigned char *to, const unsigned char *from, uint64_t n) {
    /* Volatile byte accesses keep the freestanding Wasm core independent of
     * compiler-created memcpy/memmove imports, including optimized builds. */
    volatile unsigned char *dst = to;
    const volatile unsigned char *src = from;
    for (uint64_t i = 0; i < n; i++) dst[i] = src[i];
}

#ifdef __wasm32__
/* I supply the memory operations my freestanding C compiler can introduce for
 * aggregate initialization and copying. Volatile bytes prevent recursive
 * lowering back to these same routines at any selected optimization level. */
void *memset(void *memory,int value,size_t bytes) {
    volatile unsigned char *out=(volatile unsigned char *)memory;
    for(size_t i=0;i<bytes;i++)out[i]=(unsigned char)value;
    return memory;
}
void *memcpy(void *destination,const void *source,size_t bytes) {
    volatile unsigned char *out=(volatile unsigned char *)destination;
    const volatile unsigned char *in=(const volatile unsigned char *)source;
    for(size_t i=0;i<bytes;i++)out[i]=in[i];
    return destination;
}
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
#ifdef NMS_TEST_ALLOC_HOOKS
    if (!nms_test_allocation_permitted(bytes)) return NULL;
#endif
#ifdef NMS_TESTING
    if (!runtime->fail_after) return NULL;
    if (runtime->fail_after != UINT64_MAX) runtime->fail_after--;
#else
    (void)runtime;
#endif
    void *memory = backend_allocate(bytes);
#ifdef NMS_TEST_ALLOC_HOOKS
    if (memory) nms_test_allocation_created(memory, bytes);
#endif
#ifdef NMS_TESTING
    if (memory) live_allocations++;
#endif
    return memory;
}
static void deallocate(void *memory) {
    if (!memory) return;
#ifdef NMS_TEST_ALLOC_HOOKS
    nms_test_allocation_destroyed(memory);
#endif
#ifdef NMS_TESTING
    live_allocations--;
#endif
    backend_free(memory);
}
void nms_init(NmsRuntime *runtime, const NmsView *literals, uint32_t count) {
    runtime->record_descriptors = NULL;
    runtime->record_count = runtime->records_bound = 0;
    runtime->literals = literals;
    runtime->literal_count = count;
    runtime->slots = NULL;
    runtime->collection_workspace = NULL;
    runtime->collection_capacity = runtime->collection_prepared = 0;
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
        if (runtime->slots[index].kind != NMS_SLOT_STRING) return NMS_TYPE;
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
/* uint64 trial counts, byte marks, then aligned uint32 queue. All arithmetic
 * is widened before checking the target size_t limit. */
static int collection_layout(uint32_t capacity, uint64_t *mark_offset,
                             uint64_t *queue_offset, uint64_t *bytes) {
    uint64_t count = (uint64_t)capacity + 1;
    uint64_t marks = count * sizeof(uint64_t);
    uint64_t queue = (marks + count + 3) & ~UINT64_C(3);
    uint64_t total = queue + count * sizeof(uint32_t);
    if (total > SIZE_MAX) return 0;
    *mark_offset = marks; *queue_offset = queue; *bytes = total;
    return 1;
}
static void *collection_allocate(NmsRuntime *runtime, uint32_t capacity) {
    uint64_t marks, queue, bytes;
    if (!collection_layout(capacity, &marks, &queue, &bytes)) return NULL;
    return allocate(runtime, bytes);
}
NmsStatus nms_prepare_collection(NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (runtime->collection_prepared) return NMS_OK;
    void *workspace = collection_allocate(runtime, runtime->capacity);
    if (!workspace) return NMS_MEMORY;
    runtime->collection_workspace = workspace;
    runtime->collection_capacity = runtime->capacity;
    runtime->collection_prepared = 1;
    return NMS_OK;
}
/* Publication borrows prepared storage until success; I never publish a
 * partial table or consume that storage on allocation failure. */
static NmsStatus publish_slot(NmsRuntime *runtime, unsigned char *bytes,
                              uint32_t length, uint32_t capacity, uint32_t kind,
                              uint64_t storage_bytes, NmsHandle *out) {
    if (storage_bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    uint32_t new_capacity = runtime->capacity;
    if (!runtime->free_head) {
        if (runtime->capacity >= UINT32_MAX - 1) return NMS_MEMORY;
        new_capacity = runtime->capacity ?
            (runtime->capacity > (UINT32_MAX - 1) / 2 ? UINT32_MAX - 1 : runtime->capacity * 2) : 8;
        if ((uint64_t)new_capacity + 1 > SIZE_MAX / sizeof(NmsSlot)) return NMS_MEMORY;
    }
    NmsSlot *slots = runtime->slots;
    if (new_capacity != runtime->capacity) {
        slots = allocate(runtime, ((uint64_t)new_capacity + 1) * sizeof(NmsSlot));
        if (!slots) return NMS_MEMORY;
        void *workspace = NULL;
        if (runtime->collection_prepared) {
            workspace = collection_allocate(runtime, new_capacity);
            if (!workspace) { deallocate(slots); return NMS_MEMORY; }
        }
        for (uint64_t i = 0; i <= new_capacity; i++) {
            if (runtime->slots && i <= runtime->capacity) {
                slots[i].data = runtime->slots[i].data;
                slots[i].references = runtime->slots[i].references;
                slots[i].length = runtime->slots[i].length;
                slots[i].next_free = runtime->slots[i].next_free;
                slots[i].element_tag = runtime->slots[i].element_tag;
                slots[i].vm_array_policy = runtime->slots[i].vm_array_policy;
                slots[i].record_ordinal = runtime->slots[i].record_ordinal;
                slots[i].kind = runtime->slots[i].kind;
                slots[i].capacity = runtime->slots[i].capacity;
            } else {
                slots[i].data = NULL; slots[i].references = 0; slots[i].length = 0;
                slots[i].kind = NMS_SLOT_FREE; slots[i].capacity = 0; slots[i].element_tag = 0; slots[i].vm_array_policy = 0; slots[i].record_ordinal = 0;
                slots[i].next_free = i < new_capacity ? (uint32_t)i + 1 : 0;
            }
        }
        NmsSlot *old = runtime->slots;
        if (runtime->collection_prepared) {
            void *old_workspace = runtime->collection_workspace;
            runtime->collection_workspace = workspace;
            runtime->collection_capacity = new_capacity;
            deallocate(old_workspace);
        }
        runtime->slots = slots;
        runtime->free_head = runtime->capacity + 1;
        runtime->capacity = new_capacity;
        deallocate(old);
    }
    uint32_t index = runtime->free_head;
    runtime->free_head = slots[index].next_free;
    slots[index].data = bytes; slots[index].length = (uint32_t)length;
    slots[index].element_tag = 0; slots[index].vm_array_policy = 0; slots[index].record_ordinal = 0;
    slots[index].kind = kind; slots[index].capacity = capacity;
    slots[index].references = 1; slots[index].next_free = 0;
    runtime->live_bytes += storage_bytes; runtime->live_objects++;
    *out = NMS_DYNAMIC | index;
    return NMS_OK;
}
static NmsStatus create_parts(NmsRuntime *runtime,
                              const unsigned char *data, uint64_t first_length,
                              const unsigned char *second, uint64_t second_length,
                              NmsHandle *out) {
    if (!runtime || !out || (!data && first_length) || (!second && second_length))
        return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (first_length > UINT32_MAX || second_length > UINT32_MAX) return NMS_MEMORY;
    uint64_t length = first_length + second_length;
    if (length > UINT32_MAX || length == SIZE_MAX ||
        length > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    unsigned char *bytes = allocate(runtime, length + 1);
    if (!bytes) return NMS_MEMORY;
    copy_bytes(bytes, data, first_length);
    copy_bytes(bytes + first_length, second, second_length);
    bytes[length] = 0;
    NmsStatus status = publish_slot(runtime, bytes, (uint32_t)length, 0,
                                   NMS_SLOT_STRING, length, out);
    if (status != NMS_OK) deallocate(bytes);
    return status;
}
static NmsStatus string_array_slot(const NmsRuntime *runtime, NmsHandle handle,
                                   uint32_t *index) {
    if (!(handle & NMS_DYNAMIC)) {
        NmsView view;
        NmsStatus status = nms_view(runtime, handle, &view);
        return status == NMS_OK ? NMS_TYPE : status;
    }
    NmsStatus status = slot_for(runtime, handle, index);
    if (status != NMS_OK) return status;
    return runtime->slots[*index].kind == NMS_SLOT_STRING_ARRAY ? NMS_OK : NMS_TYPE;
}
NmsStatus nms_string_array_create(NmsRuntime *runtime, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    return publish_slot(runtime, NULL, 0, 0, NMS_SLOT_STRING_ARRAY, 0, out);
}
NmsStatus nms_string_array_append(NmsRuntime *runtime, NmsHandle array, NmsHandle child) {
    uint32_t index;
    NmsStatus status = string_array_slot(runtime, array, &index);
    if (status != NMS_OK) return status;
    NmsView view;
    status = nms_view(runtime, child, &view);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (slot->length == UINT32_MAX) return NMS_MEMORY;
    uint32_t capacity = slot->capacity;
    NmsHandle *buffer = (NmsHandle *)slot->data;
    if (slot->length == capacity) {
        capacity = capacity ? (capacity > UINT32_MAX / 2 ? UINT32_MAX : capacity * 2) : 4;
        uint64_t bytes = (uint64_t)capacity * sizeof(NmsHandle);
        uint64_t added = (uint64_t)(capacity - slot->capacity) * sizeof(NmsHandle);
        if (bytes > SIZE_MAX || added > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
        buffer = allocate(runtime, bytes);
        if (!buffer) return NMS_MEMORY;
        copy_bytes((unsigned char *)buffer, slot->data, (uint64_t)slot->length * sizeof(NmsHandle));
    }
    status = nms_retain(runtime, child);
    if (status != NMS_OK) {
        if ((unsigned char *)buffer != slot->data) deallocate(buffer);
        return status;
    }
    if ((unsigned char *)buffer != slot->data) {
        runtime->live_bytes += (uint64_t)(capacity - slot->capacity) * sizeof(NmsHandle);
        deallocate(slot->data);
        slot->data = (unsigned char *)buffer;
        slot->capacity = capacity;
    }
    buffer[slot->length++] = child;
    return NMS_OK;
}
NmsStatus nms_string_array_get(NmsRuntime *runtime, NmsHandle array, uint64_t index,
                              NmsHandle *out) {
    if (!out) return NMS_STATE;
    uint32_t slot_index;
    NmsStatus status = string_array_slot(runtime, array, &slot_index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[slot_index];
    if (index >= slot->length) { *out = 0; return NMS_OK; }
    NmsHandle child = ((NmsHandle *)slot->data)[index];
    status = nms_retain(runtime, child);
    if (status == NMS_OK) *out = child;
    return status;
}
NmsStatus nms_string_array_length(const NmsRuntime *runtime, NmsHandle array, uint32_t *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = string_array_slot(runtime, array, &index);
    if (status == NMS_OK) *out = runtime->slots[index].length;
    return status;
}
NmsStatus nms_create(NmsRuntime *runtime, const unsigned char *data, uint64_t length,
                     NmsHandle *out) {
    return create_parts(runtime, data, length, NULL, 0, out);
}
_Static_assert(offsetof(NmsValue, payload) == 0 && offsetof(NmsValue, tag) == 8 &&
               sizeof(NmsValue) == 16, "I require the private boxed leaf-value ABI");
static NmsStatus value_array_slot(const NmsRuntime *, NmsHandle, uint32_t *);
static NmsStatus record_slot(const NmsRuntime *runtime, NmsHandle handle, uint32_t *index) {
    NmsStatus status = slot_for(runtime, handle, index);
    if (status != NMS_OK) return status;
    const NmsSlot *slot = &runtime->slots[*index];
    if (slot->kind != NMS_SLOT_RECORD) return NMS_TYPE;
    if (!runtime->records_bound || !runtime->record_descriptors ||
        slot->record_ordinal >= runtime->record_count) return NMS_STATE;
    uint32_t fields = runtime->record_descriptors[slot->record_ordinal].field_count;
    return slot->length == fields && slot->capacity == fields &&
        (!fields || slot->data) ? NMS_OK : NMS_STATE;
}
static int value_is_reference(NmsValue value) {
    return value.tag == 5 || value.tag == NMS_ARRAY_TAG || value.tag == NMS_RECORD_TAG;
}
static int slot_has_children(const NmsSlot *slot) {
    return slot->kind == NMS_SLOT_STRING_ARRAY || slot->kind == NMS_SLOT_BOXED_ARRAY ||
        slot->kind == NMS_SLOT_RECORD;
}

static NmsStatus value_valid(const NmsRuntime *runtime, NmsValue value) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (value.tag == 5) {
        NmsView view;
        return nms_view(runtime, value.payload, &view);
    }
    if (value.tag == NMS_ARRAY_TAG) {
        uint32_t index;
        return value_array_slot(runtime, value.payload, &index);
    }
    if (value.tag == NMS_RECORD_TAG) {
        uint32_t index;
        return record_slot(runtime, value.payload, &index);
    }
    return value.tag < 5 || value.tag == 9 ? NMS_OK : NMS_TYPE;
}
NmsStatus nms_value_retain(NmsRuntime *runtime, NmsValue value) {
    NmsStatus status = value_valid(runtime, value);
    if (status != NMS_OK) return status;
    return value_is_reference(value) ? nms_retain(runtime, value.payload) : NMS_OK;
}
NmsStatus nms_value_release(NmsRuntime *runtime, NmsValue value) {
    NmsStatus status = value_valid(runtime, value);
    if (status != NMS_OK) return status;
    return value_is_reference(value) ? nms_release(runtime, value.payload) : NMS_OK;
}
static NmsStatus value_array_slot(const NmsRuntime *runtime, NmsHandle array,
                                  uint32_t *index) {
    if (!(array & NMS_DYNAMIC)) {
        NmsView view;
        NmsStatus status = nms_view(runtime, array, &view);
        return status == NMS_OK ? NMS_TYPE : status;
    }
    NmsStatus status = slot_for(runtime, array, index);
    if (status != NMS_OK) return status;
    uint32_t kind = runtime->slots[*index].kind;
    return kind == NMS_SLOT_STRING_ARRAY || kind == NMS_SLOT_BOXED_ARRAY || kind == NMS_SLOT_PACKED_SCALAR_ARRAY ? NMS_OK : NMS_TYPE;
}
static uint32_t packed_width(uint32_t tag) {
    return tag == 1 || tag == 3 ? 8 : tag == 2 || tag == 4 ? 1 : 0;
}
/* I distinguish VM logical growth from older private empty-buffer factories. */
static NmsStatus array_next_capacity(uint32_t old, uint32_t width, uint32_t vm_policy,
                                     uint32_t *out) {
    uint64_t next = old ? (uint64_t)old * 2 : 4;
    if (vm_policy) {
        if (!width || next <= old || next > UINT32_MAX / width) return NMS_MEMORY;
    } else if (next > UINT32_MAX) next = UINT32_MAX;
    *out = (uint32_t)next;
    return NMS_OK;
}
static NmsStatus vm_array_create_capacity(NmsRuntime *runtime, uint32_t tag,
                                           uint32_t capacity, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!(tag <= 5 || tag == NMS_ARRAY_TAG || tag == 9)) return NMS_TYPE;
    uint32_t width = packed_width(tag);
    uint32_t kind = width ? NMS_SLOT_PACKED_SCALAR_ARRAY : NMS_SLOT_BOXED_ARRAY;
    if (!width) width = sizeof(NmsValue);
    if (capacity < 8) capacity = 8;
    uint64_t bytes = (uint64_t)capacity * width;
    if (bytes > SIZE_MAX || bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    unsigned char *buffer = allocate(runtime, bytes);
    if (!buffer) return NMS_MEMORY;
    NmsHandle result = 0;
    NmsStatus status = publish_slot(runtime, buffer, 0, capacity, kind, bytes, &result);
    if (status != NMS_OK) { deallocate(buffer); return status; }
    NmsSlot *slot = &runtime->slots[(uint32_t)result];
    slot->element_tag = tag; slot->vm_array_policy = 1;
    *out = result;
    return NMS_OK;
}
NmsStatus nms_vm_array_create(NmsRuntime *runtime, uint32_t tag, NmsHandle *out) {
    return vm_array_create_capacity(runtime, tag, 8, out);
}
static NmsValue packed_value(const NmsSlot *slot, uint32_t index) {
    uint32_t width = packed_width(slot->element_tag);
    uint64_t offset = (uint64_t)index * width;
    NmsValue value = {0, slot->element_tag};
    for (uint32_t i = 0; i < width; i++) value.payload |= (uint64_t)slot->data[offset + i] << (8 * i);
    return value;
}
static void packed_publish(NmsSlot *slot, uint32_t index, uint64_t bits) {
    uint32_t width = packed_width(slot->element_tag);
    uint64_t offset = (uint64_t)index * width;
    for (uint32_t i = 0; i < width; i++) slot->data[offset + i] = (unsigned char)(bits >> (8 * i));
}
static NmsStatus packed_prepare(uint32_t tag, NmsValue value, uint64_t *bits) {
    if ((value.tag == 2 && value.payload > 255) || (value.tag == 4 && value.payload > 1)) return NMS_TYPE;
    if (tag == value.tag && packed_width(tag)) { *bits = value.payload; return NMS_OK; }
    if (tag == 1 && value.tag == 2) { *bits = value.payload; return NMS_OK; }
    if (tag == 2 && value.tag == 1) { *bits = value.payload & 255; return NMS_OK; }
    if (tag == 3 && value.tag == 1) {
        /* I avoid an implementation-defined unsigned-to-signed conversion. */
        int64_t integer = value.payload <= INT64_MAX ? (int64_t)value.payload :
                          -1 - (int64_t)(UINT64_MAX - value.payload);
        double converted = (double)integer;
        _Static_assert(sizeof(double) == sizeof(uint64_t), "I require binary64 scalar storage");
        copy_bytes((unsigned char *)bits, (const unsigned char *)&converted, sizeof converted);
        return NMS_OK;
    }
    return NMS_TYPE;
}
NmsStatus nms_packed_array_create(NmsRuntime *runtime, uint32_t tag, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!packed_width(tag)) return NMS_TYPE;
    NmsHandle result = 0;
    NmsStatus status = publish_slot(runtime, NULL, 0, 0, NMS_SLOT_PACKED_SCALAR_ARRAY, 0, &result);
    if (status != NMS_OK) return status;
    runtime->slots[(uint32_t)result].element_tag = tag;
    *out = result;
    return NMS_OK;
}
static NmsStatus packed_write(NmsRuntime *runtime, NmsSlot *slot, uint64_t index,
                               NmsValue value, int append) {
    uint64_t bits = 0;
    NmsStatus status = packed_prepare(slot->element_tag, value, &bits);
    if (status != NMS_OK) return status;
    if (!append && index >= slot->length) return NMS_STATE;
    if (append && slot->length == UINT32_MAX) return NMS_MEMORY;
    uint32_t width = packed_width(slot->element_tag);
    if (!width) return NMS_STATE;
    uint32_t target = append ? slot->length : (uint32_t)index;
    if (append && slot->length == slot->capacity) {
        uint32_t capacity = 0;
        status = array_next_capacity(slot->capacity, width, slot->vm_array_policy, &capacity);
        if (status != NMS_OK) return status;
        uint64_t bytes = (uint64_t)capacity * width;
        uint64_t delta = (uint64_t)(capacity - slot->capacity) * width;
        if (bytes > SIZE_MAX || delta > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
        unsigned char *buffer = allocate(runtime, bytes);
        if (!buffer) return NMS_MEMORY;
        copy_bytes(buffer, slot->data, (uint64_t)slot->length * width);
        unsigned char *old = slot->data;
        slot->data = buffer; slot->capacity = capacity;
        runtime->live_bytes += delta;
        deallocate(old);
    }
    packed_publish(slot, target, bits);
    if (append) slot->length++;
    return NMS_OK;
}
static NmsValue slot_value(const NmsSlot *slot, uint32_t index) {
    if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY) return packed_value(slot, index);
    if (slot->kind == NMS_SLOT_STRING_ARRAY) {
        NmsValue value = {((const NmsHandle *)slot->data)[index], 5};
        return value;
    }
    return ((const NmsValue *)slot->data)[index];
}
static uint64_t slot_storage_bytes(const NmsSlot *slot) {
    return (uint64_t)slot->capacity *
        (slot->kind == NMS_SLOT_STRING_ARRAY ? sizeof(NmsHandle) :
         slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY ? packed_width(slot->element_tag) : sizeof(NmsValue));
}
NmsStatus nms_bind_records(NmsRuntime *runtime, const NmsRecordDescriptor *descriptors,
                            uint32_t count) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (runtime->active) return NMS_BUSY;
    if (runtime->records_bound || runtime->capacity || runtime->live_objects ||
        runtime->live_bytes || (count && !descriptors)) return NMS_STATE;
    if (count > 256) return NMS_TYPE;
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (descriptors[i].global_layout_index >= 256 ||
            (i && descriptors[i].global_layout_index <= descriptors[i-1].global_layout_index) ||
            descriptors[i].field_count > UINT16_MAX ||
            descriptors[i].field_count > 65536u - total) return NMS_TYPE;
        total += descriptors[i].field_count;
    }
    runtime->record_descriptors = descriptors;
    runtime->record_count = count;
    runtime->records_bound = 1;
    return NMS_OK;
}
NmsStatus nms_record_create(NmsRuntime *runtime, uint32_t ordinal, const NmsValue *values,
                             uint32_t count, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!runtime->records_bound || !runtime->record_descriptors ||
        ordinal >= runtime->record_count) return NMS_TYPE;
    if (count != runtime->record_descriptors[ordinal].field_count) return NMS_TYPE;
    if (count && !values) return NMS_STATE;
    uint64_t bytes = (uint64_t)count * sizeof(NmsValue);
    if (bytes > SIZE_MAX || bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    for (uint32_t i = 0; i < count; i++) {
        NmsStatus status = value_valid(runtime, values[i]);
        if (status != NMS_OK) return status;
    }
    NmsValue *fields = bytes ? allocate(runtime, bytes) : NULL;
    if (bytes && !fields) return NMS_MEMORY;
    uint32_t retained = 0;
    NmsStatus status = NMS_OK;
    for (; retained < count; retained++) {
        fields[retained] = values[retained];
        status = nms_value_retain(runtime, fields[retained]);
        if (status != NMS_OK) break;
    }
    NmsHandle result;
    if (status == NMS_OK)
        status = publish_slot(runtime, (unsigned char *)fields, count, count,
                              NMS_SLOT_RECORD, bytes, &result);
    if (status != NMS_OK) {
        while (retained) nms_value_release(runtime, fields[--retained]);
        deallocate(fields);
        return status;
    }
    runtime->slots[(uint32_t)result].record_ordinal = ordinal;
    *out = result;
    return NMS_OK;
}
NmsStatus nms_record_identity(const NmsRuntime *runtime, NmsHandle record,
                               uint32_t *ordinal, uint32_t *global_layout) {
    if (!ordinal || !global_layout) return NMS_STATE;
    uint32_t index;
    NmsStatus status = record_slot(runtime, record, &index);
    if (status != NMS_OK) return status;
    uint32_t definition = runtime->slots[index].record_ordinal;
    *ordinal = definition;
    *global_layout = runtime->record_descriptors[definition].global_layout_index;
    return NMS_OK;
}
NmsStatus nms_record_get(NmsRuntime *runtime, NmsHandle record, uint64_t field, NmsValue *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = record_slot(runtime, record, &index);
    if (status != NMS_OK) return status;
    if (field >= runtime->slots[index].length) return NMS_BOUNDS;
    NmsValue value = slot_value(&runtime->slots[index], (uint32_t)field);
    status = nms_value_retain(runtime, value);
    if (status == NMS_OK) *out = value;
    return status;
}
NmsStatus nms_record_set(NmsRuntime *runtime, NmsHandle record, uint64_t field, NmsValue value) {
    uint32_t index;
    NmsStatus status = record_slot(runtime, record, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (field >= slot->length) return NMS_BOUNDS;
    status = nms_value_retain(runtime, value);
    if (status != NMS_OK) return status;
    NmsValue *fields = (NmsValue *)slot->data;
    NmsValue previous = fields[field];
    fields[field] = value;
    return nms_value_release(runtime, previous);
}
NmsStatus nms_value_array_create(NmsRuntime *runtime, NmsHandle *out) {
    if (!runtime || !out) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    return publish_slot(runtime, NULL, 0, 0, NMS_SLOT_BOXED_ARRAY, 0, out);
}
static NmsStatus value_array_write(NmsRuntime *runtime, NmsHandle array,
                                    uint64_t requested_index, NmsValue value, int append) {
    uint32_t index;
    NmsStatus status = value_array_slot(runtime, array, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY)
        return packed_write(runtime, slot, requested_index, value, append);
    status = value_valid(runtime, value);
    if (status != NMS_OK) return status;
    if (!append && requested_index >= slot->length) return NMS_STATE;
    if (append && slot->length == UINT32_MAX) return NMS_MEMORY;
    uint32_t target = append ? slot->length : (uint32_t)requested_index;
    uint32_t capacity = slot->capacity;
    if (append && slot->length == capacity) {
        status = array_next_capacity(slot->capacity, sizeof(NmsValue), slot->vm_array_policy, &capacity);
        if (status != NMS_OK) return status;
    }
    uint64_t bytes = (uint64_t)capacity * sizeof(NmsValue);
    uint64_t old_bytes = slot_storage_bytes(slot);
    if (bytes > SIZE_MAX || bytes - old_bytes > UINT64_MAX - runtime->live_bytes) return NMS_MEMORY;
    NmsValue *buffer = (NmsValue *)slot->data;
    int replacement = slot->kind == NMS_SLOT_STRING_ARRAY || capacity != slot->capacity;
    if (replacement) {
        buffer = allocate(runtime, bytes);
        if (!buffer) return NMS_MEMORY;
        /* I move existing edge ownership without changing reference counts. */
        for (uint32_t i = 0; i < slot->length; i++) buffer[i] = slot_value(slot, i);
    }
    status = nms_value_retain(runtime, value);
    if (status != NMS_OK) {
        if (replacement) deallocate(buffer);
        return status;
    }
    NmsValue previous = {0, 0};
    if (!append) previous = slot_value(slot, target);
    if (replacement) {
        unsigned char *old = slot->data;
        slot->data = (unsigned char *)buffer;
        slot->capacity = capacity;
        slot->kind = NMS_SLOT_BOXED_ARRAY;
        runtime->live_bytes += bytes - old_bytes;
        deallocate(old);
    }
    buffer[target] = value;
    if (append) slot->length++;
    /* My new edge is visible before I drop the old child's owner. */
    return append ? NMS_OK : nms_value_release(runtime, previous);
}
NmsStatus nms_value_array_append(NmsRuntime *runtime, NmsHandle array, NmsValue value) {
    return value_array_write(runtime, array, 0, value, 1);
}
NmsStatus nms_value_array_set(NmsRuntime *runtime, NmsHandle array, uint64_t index, NmsValue value) {
    return value_array_write(runtime, array, index, value, 0);
}
NmsStatus nms_value_array_get(NmsRuntime *runtime, NmsHandle array, uint64_t index, NmsValue *out) {
    if (!out) return NMS_STATE;
    uint32_t slot_index;
    NmsStatus status = value_array_slot(runtime, array, &slot_index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[slot_index];
    if (index >= slot->length) { *out = (NmsValue){0, 0}; return NMS_OK; }
    NmsValue value = slot_value(slot, (uint32_t)index);
    status = nms_value_retain(runtime, value);
    if (status == NMS_OK) *out = value;
    return status;
}
NmsStatus nms_value_array_pop(NmsRuntime *runtime, NmsHandle array, NmsValue *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = value_array_slot(runtime, array, &index);
    if (status != NMS_OK) return status;
    NmsSlot *slot = &runtime->slots[index];
    if (!slot->length) { *out = (NmsValue){0, 0}; return NMS_OK; }
    NmsValue value = slot_value(slot, slot->length - 1);
    slot->length--;
    if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY) packed_publish(slot, slot->length, 0);
    else if (slot->kind == NMS_SLOT_STRING_ARRAY) ((NmsHandle *)slot->data)[slot->length] = 0;
    else ((NmsValue *)slot->data)[slot->length] = (NmsValue){0, 0};
    *out = value;
    return NMS_OK;
}
NmsStatus nms_value_array_length(const NmsRuntime *runtime, NmsHandle array, uint32_t *out) {
    if (!out) return NMS_STATE;
    uint32_t index;
    NmsStatus status = value_array_slot(runtime, array, &index);
    if (status == NMS_OK) *out = runtime->slots[index].length;
    return status;
}
/* I borrow complete input arrays; my result remains private until all edges exist. */
NmsStatus nms_vm_array_literal(NmsRuntime *runtime, uint32_t tag,
                                const uint64_t *payloads, const uint32_t *tags,
                                uint32_t count, NmsHandle *out) {
    if (!runtime || !out || count > UINT16_MAX || (count && (!payloads || !tags))) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!(tag <= 5 || tag == NMS_ARRAY_TAG || tag == 9)) return NMS_TYPE;
    for (uint32_t i = 0; i < count; i++) {
        NmsValue value = {payloads[i], tags[i]};
        uint64_t ignored;
        NmsStatus status = packed_width(tag) ? packed_prepare(tag, value, &ignored) : value_valid(runtime, value);
        if (status != NMS_OK) return status;
    }
    NmsHandle result = 0;
    NmsStatus status = vm_array_create_capacity(runtime, tag, count, &result);
    if (status != NMS_OK) return status;
    for (uint32_t i = 0; i < count; i++) {
        status = nms_value_array_append(runtime, result, (NmsValue){payloads[i], tags[i]});
        if (status != NMS_OK) { nms_release(runtime, result); return status; }
    }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_vm_array_slice(NmsRuntime *runtime, NmsHandle source,
                              uint32_t start, uint32_t end, NmsHandle *out) {
    if (!out) return NMS_STATE;
    uint32_t source_index;
    NmsStatus status = value_array_slot(runtime, source, &source_index);
    if (status != NMS_OK) return status;
    const NmsSlot *input = &runtime->slots[source_index];
    uint32_t length = input->length;
    uint32_t tag = input->kind == NMS_SLOT_STRING_ARRAY ? 5 : input->element_tag;
    if (start > length) start = length;
    if (end > length) end = length;
    uint32_t count = end > start ? end - start : 0;
    NmsHandle result = 0;
    status = vm_array_create_capacity(runtime, tag, count, &result);
    if (status != NMS_OK) return status;
    /* Descriptor growth may relocate the source slot; its immutable buffer
     * remains owned by the borrowed source handle. Reacquire both slots. */
    input = &runtime->slots[source_index];
    NmsSlot *output = &runtime->slots[(uint32_t)result];
    if (input->kind == NMS_SLOT_PACKED_SCALAR_ARRAY) {
        uint32_t width = packed_width(tag);
        if (count) copy_bytes(output->data, input->data + (uint64_t)start * width,
                              (uint64_t)count * width);
        output->length = count;
    } else {
        for (uint32_t i = 0; i < count; i++) {
            NmsValue value = slot_value(input, start + i);
            status = nms_value_array_append(runtime, result, value);
            if (status != NMS_OK) { nms_release(runtime, result); return status; }
        }
    }
    *out = result;
    return NMS_OK;
}

/* I format the exact binary64 rational with decimal integer arithmetic. The
 * largest coefficient needs fewer than 800 digits; 1100 is an explicit cap. */
static NmsStatus nms_format_binary64(NmsRuntime *runtime, uint64_t bits, NmsHandle *out) {
    unsigned char output[32];
    unsigned used = 0;
    int negative = (int)(bits >> 63);
    if (negative) output[used++] = '-';
    uint32_t exponent_bits = (uint32_t)((bits >> 52) & 2047);
    uint64_t significand = bits & ((UINT64_C(1) << 52) - 1);
    if (exponent_bits == 2047) {
        const char *word = significand ? "nan" : "inf";
        for (unsigned i = 0; i < 3; i++) output[used++] = (unsigned char)word[i];
        return nms_create(runtime, output, used, out);
    }
    if (!exponent_bits && !significand) {
        output[used++] = '0';
        return nms_create(runtime, output, used, out);
    }
    int binary_exponent = exponent_bits ? (int)exponent_bits - 1023 - 52 : -1074;
    if (exponent_bits) significand |= UINT64_C(1) << 52;
    unsigned char digits[1100]; /* Little endian, one decimal digit per byte. */
    unsigned count = 0;
    do { digits[count++] = (unsigned char)(significand % 10); significand /= 10; }
    while (significand);
    unsigned steps = (unsigned)(binary_exponent < 0 ? -binary_exponent : binary_exponent);
    unsigned multiplier = binary_exponent < 0 ? 5 : 2;
    for (unsigned step = 0; step < steps; step++) {
        unsigned carry = 0;
        for (unsigned i = 0; i < count; i++) {
            unsigned value = digits[i] * multiplier + carry;
            digits[i] = (unsigned char)(value % 10);
            carry = value / 10;
        }
        if (carry) {
            if (count == sizeof digits) return NMS_STATE;
            digits[count++] = (unsigned char)carry;
        }
    }
    int exponent = (int)count - 1 + (binary_exponent < 0 ? binary_exponent : 0);
    uint32_t rounded = 0;
    for (unsigned i = 0; i < 6; i++)
        rounded = rounded * 10 + (i < count ? digits[count - 1 - i] : 0);
    if (count > 6) {
        unsigned next = digits[count - 7];
        int lower_nonzero = 0;
        for (unsigned i = 0; i + 7 < count; i++) lower_nonzero |= digits[i] != 0;
        if (next > 5 || (next == 5 && (lower_nonzero || (rounded & 1)))) rounded++;
        if (rounded == 1000000) { rounded = 100000; exponent++; }
    }
    unsigned char leading[6];
    for (unsigned i = 6; i > 0; i--) { leading[i-1] = (unsigned char)('0' + rounded % 10); rounded /= 10; }
    unsigned length = 6;
    while (length > 1 && leading[length - 1] == '0') length--;
    /* At most 14 bytes: sign + six digits + point + e + sign + three exponent
     * digits. Fixed notation is smaller because its exponent lies in [-4,5]. */
    if (exponent < -4 || exponent >= 6) {
        output[used++] = leading[0];
        if (length > 1) {
            output[used++] = '.';
            for (unsigned i = 1; i < length; i++) output[used++] = leading[i];
        }
        output[used++] = 'e';
        output[used++] = exponent < 0 ? '-' : '+';
        unsigned magnitude = (unsigned)(exponent < 0 ? -exponent : exponent);
        if (magnitude >= 100) output[used++] = (unsigned char)('0' + magnitude / 100);
        output[used++] = (unsigned char)('0' + (magnitude / 10) % 10);
        output[used++] = (unsigned char)('0' + magnitude % 10);
    } else if (exponent < 0) {
        output[used++] = '0'; output[used++] = '.';
        for (int i = -1; i > exponent; i--) output[used++] = '0';
        for (unsigned i = 0; i < length; i++) output[used++] = leading[i];
    } else {
        unsigned integer_length = (unsigned)exponent + 1;
        for (unsigned i = 0; i < integer_length; i++) output[used++] = leading[i];
        if (length > integer_length) {
            output[used++] = '.';
            for (unsigned i = integer_length; i < length; i++) output[used++] = leading[i];
        }
    }
    return nms_create(runtime, output, used, out);
}
NmsStatus nms_format_scalar(NmsRuntime *runtime, uint64_t bits, uint32_t tag, NmsHandle *out) {
    if (tag == 3) return nms_format_binary64(runtime, bits, out);
    if (tag == 0 || tag == NMS_ARRAY_TAG || tag == 9) return nms_create(runtime, NULL, 0, out);
    if (tag == 4) return nms_create(runtime,
        (const unsigned char *)(bits ? "true" : "false"), bits ? 4 : 5, out);
    if (tag != 1 && tag != 2) return NMS_TYPE;
    int negative = tag == 1 && (bits >> 63);
    uint64_t magnitude = tag == 2 ? bits & 255 : negative ? UINT64_C(0) - bits : bits;
    unsigned char digits[20];
    unsigned position = sizeof digits;
    do { digits[--position] = (unsigned char)('0' + magnitude % 10); magnitude /= 10; }
    while (magnitude);
    if (negative) digits[--position] = '-';
    return nms_create(runtime, digits + position, sizeof digits - position, out);
}
NmsStatus nms_parse_f64(const NmsRuntime *runtime, NmsHandle source, uint64_t *out) {
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) return status;
    return nbp_parse(view.data, view.length, out) ? NMS_OK : NMS_STATE;
}
NmsStatus nms_parse_i64(const NmsRuntime *runtime, NmsHandle source, int64_t *out) {
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) return status;
    if (!out) return NMS_STATE;
    uint32_t i = 0;
    while (i < view.length) {
        unsigned char c = view.data[i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\v' && c != '\f') break;
        i++;
    }
    int negative = 0;
    if (i < view.length && (view.data[i] == '-' || view.data[i] == '+')) {
        negative = view.data[i] == '-';
        i++;
    }
    uint64_t limit = negative ? (UINT64_C(1) << 63) : INT64_MAX;
    uint64_t value = 0;
    while (i < view.length) {
        unsigned char c = view.data[i++];
        if (c < '0' || c > '9') break;
        uint64_t digit = c - '0';
        if (value > (limit - digit) / 10) { value = limit; break; }
        value = value * 10 + digit;
    }
    /* I never cast 2^63 to signed or negate INT64_MIN. */
    *out = negative ? (value == (UINT64_C(1) << 63) ? INT64_MIN : -(int64_t)value)
                    : (int64_t)value;
    return NMS_OK;
}
NmsStatus nms_case_owned(NmsRuntime *runtime, NmsHandle source, uint32_t upper,
                         NmsHandle *out) {
    NmsView view;
    NmsHandle result = 0;
    NmsStatus status = (!out || upper > 1) ? NMS_STATE : nms_view(runtime, source, &view);
    if (status == NMS_OK) status = nms_create(runtime, view.data, view.length, &result);
    if (status == NMS_OK) {
        /* Creation can move the descriptor table. This new owner is private;
         * I reacquire its slot after allocation and transform before publish. */
        NmsSlot *slot = &runtime->slots[(uint32_t)(result & ~NMS_DYNAMIC)];
        for (uint32_t i = 0; i < slot->length; i++) {
            unsigned char byte = slot->data[i];
            slot->data[i] = upper ? (byte >= 'a' && byte <= 'z' ? byte - 32 : byte)
                                  : (byte >= 'A' && byte <= 'Z' ? byte + 32 : byte);
        }
    }
    NmsStatus released = nms_release(runtime, source);
    if (status != NMS_OK) return status;
    if (released != NMS_OK) { nms_release(runtime, result); return released; }
    *out = result;
    return NMS_OK;
}
static int trim_space(unsigned char byte) {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
}
NmsStatus nms_trim_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle *out) {
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) { nms_release(runtime, source); return status; }
    uint32_t start = 0, end = view.length;
    while (start < end && trim_space(view.data[start])) start++;
    while (end > start && trim_space(view.data[end - 1])) end--;
    return nms_substr_owned(runtime, source, start, end - start, out);
}
NmsStatus nms_substr_owned(NmsRuntime *runtime, NmsHandle source,
                           uint32_t start, uint32_t length, NmsHandle *out) {
    NmsView view;
    NmsHandle result = 0;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status == NMS_OK) {
        if (start >= view.length) { start = 0; length = 0; }
        else if (length > view.length - start) length = view.length - start;
        status = out ? nms_create(runtime, length ? view.data + start : NULL, length, &result) : NMS_STATE;
    }
    NmsStatus released = nms_release(runtime, source);
    if (status != NMS_OK) return status;
    if (released != NMS_OK) { nms_release(runtime, result); return released; }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_concat_owned(NmsRuntime *runtime, NmsHandle left, NmsHandle right,
                           NmsHandle *out) {
    NmsView a, b;
    NmsHandle result = 0;
    NmsStatus status = nms_view(runtime, left, &a);
    if (status == NMS_OK) status = nms_view(runtime, right, &b);
    if (status == NMS_OK)
        status = out ? create_parts(runtime, a.data, a.length, b.data, b.length, &result) : NMS_STATE;
    /* Each input represents a transferred owner, including equal handles.
     * Copying finishes before release or descriptor-table replacement. */
    NmsStatus left_status = nms_release(runtime, left);
    NmsStatus right_status = nms_release(runtime, right);
    if (status != NMS_OK) return status;
    /* Ownership-correct callers cannot fail these releases. */
    if (left_status != NMS_OK || right_status != NMS_OK) {
        nms_release(runtime, result);
        return left_status != NMS_OK ? left_status : right_status;
    }
    *out = result;
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
    if (--runtime->slots[index].references) return NMS_OK;
    /* Zero-count slots are unavailable to live-handle lookup. I use their
     * next_free field as a private worklist until all outgoing edges are gone.
     * No allocation or table relocation occurs in this loop. */
    uint32_t pending = index;
    runtime->slots[index].next_free = 0;
    NmsStatus first_error = NMS_OK;
    while (pending) {
        index = pending;
        NmsSlot *slot = &runtime->slots[index];
        pending = slot->next_free;
        if (slot_has_children(slot)) {
            for (uint32_t i = 0; i < slot->length; i++) {
                NmsValue child = slot_value(slot, i);
                if (!value_is_reference(child) || !(child.payload & NMS_DYNAMIC)) continue;
                uint32_t child_index;
                NmsStatus child_status = slot_for(runtime, child.payload, &child_index);
                if (child_status != NMS_OK) {
                    if (first_error == NMS_OK) first_error = child_status;
                    continue;
                }
                NmsSlot *child_slot = &runtime->slots[child_index];
                if (!--child_slot->references) {
                    child_slot->next_free = pending;
                    pending = child_index;
                }
            }
        }
        runtime->live_bytes -= slot->kind == NMS_SLOT_STRING ? slot->length : slot_storage_bytes(slot);
        runtime->live_objects--;
        deallocate(slot->data);
        slot->data = NULL; slot->length = 0; slot->capacity = 0;
        slot->kind = NMS_SLOT_FREE; slot->element_tag = 0; slot->vm_array_policy = 0; slot->record_ordinal = 0;
        slot->next_free = runtime->free_head; runtime->free_head = index;
    }
    return first_error;
}
/* I complete validation and scratch allocation before changing any owner.
 * Trial counts identify external roots; graph edges do not become roots. */
static NmsStatus collect_with_workspace(NmsRuntime *runtime, uint64_t *trial,
                                        unsigned char *marked, uint32_t *queue) {
    uint64_t count = (uint64_t)runtime->capacity + 1;
    uint64_t bytes = 0, objects = 0;
    NmsStatus status = NMS_STATE;
    for (uint64_t i = 0; i < count; i++) {
        const NmsSlot *slot = &runtime->slots[i];
        trial[i] = slot->references; marked[i] = 0;
        if (!slot->references) {
            if (slot->kind != NMS_SLOT_FREE || slot->data || slot->length || slot->capacity) goto done;
            continue;
        }
        if (!i || slot->kind < NMS_SLOT_STRING || slot->kind > NMS_SLOT_RECORD) goto done;
        uint64_t storage;
        if (slot->kind == NMS_SLOT_STRING) {
            if (!slot->data) goto done;
            storage = slot->length;
        } else {
            if (slot->length > slot->capacity || (slot->capacity && !slot->data)) goto done;
            if (slot->kind == NMS_SLOT_PACKED_SCALAR_ARRAY && !packed_width(slot->element_tag)) goto done;
            if (slot->kind == NMS_SLOT_RECORD) {
                uint32_t index;
                if (record_slot(runtime, NMS_DYNAMIC | i, &index) != NMS_OK) goto done;
            }
            storage = slot_storage_bytes(slot);
            if (storage > SIZE_MAX) goto done;
        }
        if (storage > UINT64_MAX - bytes) goto done;
        bytes += storage; objects++;
    }
    if (objects != runtime->live_objects || bytes != runtime->live_bytes) goto done;
    for (uint64_t i = 1; i < count; i++) {
        const NmsSlot *slot = &runtime->slots[i];
        if (!slot_has_children(slot)) continue;
        for (uint32_t j = 0; j < slot->length; j++) {
            NmsValue child = slot_value(slot, j);
            if (value_valid(runtime, child) != NMS_OK) goto done;
            if (value_is_reference(child) && (child.payload & NMS_DYNAMIC)) {
                uint32_t index = (uint32_t)child.payload;
                if (!trial[index]) goto done;
                trial[index]--;
            }
        }
    }
    uint64_t head = 0, tail = 0;
    for (uint64_t i = 1; i < count; i++) if (trial[i]) {
        marked[i] = 1; queue[tail++] = (uint32_t)i;
    }
    while (head < tail) {
        const NmsSlot *slot = &runtime->slots[queue[head++]];
        if (!slot_has_children(slot)) continue;
        for (uint32_t j = 0; j < slot->length; j++) {
            NmsValue child = slot_value(slot, j);
            if (!value_is_reference(child) || !(child.payload & NMS_DYNAMIC)) continue;
            uint32_t index = (uint32_t)child.payload;
            if (!marked[index]) { marked[index] = 1; queue[tail++] = index; }
        }
    }
    /* Every marked slot is reached from an external owner. Removing only
     * dead-to-live edges therefore cannot remove its final live owner. */
    for (uint64_t i = 1; i < count; i++) {
        const NmsSlot *slot = &runtime->slots[i];
        if (marked[i] || !slot->references ||
            !slot_has_children(slot)) continue;
        for (uint32_t j = 0; j < slot->length; j++) {
            NmsValue child = slot_value(slot, j);
            if (value_is_reference(child) && (child.payload & NMS_DYNAMIC)) {
                uint32_t index = (uint32_t)child.payload;
                if (marked[index]) runtime->slots[index].references--;
            }
        }
    }
    for (uint64_t i = 1; i < count; i++) {
        NmsSlot *slot = &runtime->slots[i];
        if (marked[i] || !slot->references) continue;
        runtime->live_bytes -= slot->kind == NMS_SLOT_STRING ? slot->length : slot_storage_bytes(slot);
        runtime->live_objects--; deallocate(slot->data);
        slot->data = NULL; slot->references = 0; slot->length = 0; slot->capacity = 0;
        slot->kind = NMS_SLOT_FREE; slot->element_tag = 0; slot->vm_array_policy = 0; slot->record_ordinal = 0;
        slot->next_free = runtime->free_head; runtime->free_head = (uint32_t)i;
    }
    status = NMS_OK;
 done:
    return status;
}
static NmsStatus collection_ready(const NmsRuntime *runtime) {
    if (!runtime) return NMS_STATE;
    if (runtime->disposed) return NMS_DISPOSED;
    if (!runtime->capacity)
        return !runtime->slots && !runtime->live_objects && !runtime->live_bytes ? NMS_OK : NMS_STATE;
    return runtime->slots ? NMS_OK : NMS_STATE;
}
NmsStatus nms_collect(NmsRuntime *runtime) {
    NmsStatus status = collection_ready(runtime);
    if (status != NMS_OK || !runtime->capacity) return status;
    uint64_t count = (uint64_t)runtime->capacity + 1;
    if (count > SIZE_MAX / sizeof(uint64_t) || count > SIZE_MAX / sizeof(uint32_t)) return NMS_MEMORY;
    uint64_t *trial = allocate(runtime, count * sizeof(uint64_t));
    unsigned char *marked = allocate(runtime, count);
    uint32_t *queue = allocate(runtime, count * sizeof(uint32_t));
    status = trial && marked && queue ? collect_with_workspace(runtime, trial, marked, queue) : NMS_MEMORY;
    deallocate(queue); deallocate(marked); deallocate(trial);
    return status;
}
NmsStatus nms_collect_prepared(NmsRuntime *runtime) {
    NmsStatus status = collection_ready(runtime);
    if (status != NMS_OK) return status;
    if (!runtime->collection_prepared || !runtime->collection_workspace ||
        runtime->collection_capacity < runtime->capacity) return NMS_STATE;
    if (!runtime->capacity) return NMS_OK;
    uint64_t marks, queue, bytes;
    if (!collection_layout(runtime->collection_capacity, &marks, &queue, &bytes)) return NMS_STATE;
    unsigned char *workspace = runtime->collection_workspace;
    return collect_with_workspace(runtime, (uint64_t *)workspace, workspace + marks,
                                  (uint32_t *)(workspace + queue));
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
    if ((unsigned)status > NMS_BOUNDS) status = NMS_STATE;
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
    deallocate(runtime->collection_workspace);
    runtime->collection_workspace = NULL;
    runtime->collection_capacity = runtime->collection_prepared = 0;
    runtime->slots = NULL; runtime->capacity = runtime->free_head = 0;
    runtime->live_bytes = runtime->live_objects = 0;
    runtime->record_descriptors = NULL;
    runtime->record_count = runtime->records_bound = 0;
    runtime->disposed = 1;
    return NMS_OK;
}
int nms_reserved_entry(const char *name) {
    if (!name) return 0;
    const char *reserved[] = {"nano_try_entry", "nano_dispose", "nano_runtime_", "nms_", "memcpy", "memset"};
    for (unsigned i = 0; i < sizeof reserved / sizeof reserved[0]; i++) {
        unsigned n = 0;
        while (reserved[i][n] && name[n] == reserved[i][n]) n++;
        if (!reserved[i][n] && (i == 2 || i == 3 || !name[n])) return 1;
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

/* I compare complete byte views, including embedded zero bytes. */
static int equal_bytes(const unsigned char *left, const unsigned char *right, uint32_t length) {
    for (uint32_t i = 0; i < length; i++)
        if (left[i] != right[i]) return 0;
    return 1;
}
static int find_bytes(NmsView source, NmsView needle, uint32_t start, uint32_t *found) {
    if (!needle.length || start > source.length || needle.length > source.length - start)
        return 0;
    uint32_t last = source.length - needle.length;
    for (uint32_t position = start; position <= last; position++) {
        if (equal_bytes(source.data + position, needle.data, needle.length)) {
            *found = position;
            return 1;
        }
    }
    return 0;
}
static NmsStatus append_segment(NmsRuntime *runtime, NmsHandle array,
                                 const unsigned char *bytes, uint32_t length, int values) {
    NmsHandle child = 0;
    NmsStatus status = nms_create(runtime, bytes, length, &child);
    if (status != NMS_OK) return status;
    status = values ? nms_value_array_append(runtime, array, (NmsValue){child, 5}) :
                      nms_string_array_append(runtime, array, child);
    NmsStatus released = nms_release(runtime, child);
    return status != NMS_OK ? status : released;
}
static NmsStatus split_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                               NmsHandle *out, int values) {
    NmsView a, b;
    NmsHandle array = 0;
    NmsStatus status = out ? nms_view(runtime, source, &a) : NMS_STATE;
    if (status == NMS_OK) status = nms_view(runtime, delimiter, &b);
    if (status == NMS_OK) status = values ? nms_vm_array_create(runtime, 5, &array) :
                                          nms_string_array_create(runtime, &array);
    if (status == NMS_OK && !b.length) {
        for (uint32_t i = 0; i < a.length && status == NMS_OK; i++)
            status = append_segment(runtime, array, a.data + i, 1, values);
    } else if (status == NMS_OK) {
        uint32_t position = 0, found = 0;
        while (status == NMS_OK && find_bytes(a, b, position, &found)) {
            status = append_segment(runtime, array, a.data + position, found - position, values);
            position = found + b.length;
        }
        if (status == NMS_OK)
            status = append_segment(runtime, array, a.data + position, a.length - position, values);
    }
    /* Views point at retained byte buffers, never relocated slot-table cells.
     * I finish every read before consuming either input owner. */
    NmsStatus left = nms_release(runtime, source);
    NmsStatus right = nms_release(runtime, delimiter);
    if (status == NMS_OK) status = left != NMS_OK ? left : right;
    if (status != NMS_OK) {
        if (array) nms_release(runtime, array);
        return status;
    }
    *out = array;
    return NMS_OK;
}
NmsStatus nms_split_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                          NmsHandle *out) {
    return split_owned(runtime, source, delimiter, out, 0);
}
NmsStatus nms_split_values_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                                 NmsHandle *out) {
    return split_owned(runtime, source, delimiter, out, 1);
}
NmsStatus nms_replace_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle needle,
                            NmsHandle replacement, NmsHandle *out) {
    NmsView a, b, c;
    NmsHandle result = 0;
    unsigned char *scratch = NULL;
    NmsStatus status = out ? nms_view(runtime, source, &a) : NMS_STATE;
    if (status == NMS_OK) status = nms_view(runtime, needle, &b);
    if (status == NMS_OK) status = nms_view(runtime, replacement, &c);
    if (status == NMS_OK && !b.length) {
        status = nms_create(runtime, a.data, a.length, &result);
    } else if (status == NMS_OK) {
        uint64_t count = 0;
        uint32_t position = 0, found;
        while (find_bytes(a, b, position, &found)) {
            count++;
            position = found + b.length;
        }
        uint32_t length = 0;
        if (count > a.length / b.length) status = NMS_STATE;
        else {
            uint32_t remaining = a.length - (uint32_t)count * b.length;
            if (c.length && count > (UINT32_MAX - remaining) / c.length) status = NMS_MEMORY;
            else length = remaining + (uint32_t)count * c.length;
        }
        if (status == NMS_OK && (uint64_t)length + 1 > SIZE_MAX) status = NMS_MEMORY;
        if (status == NMS_OK) {
            scratch = allocate(runtime, (uint64_t)length + 1);
            if (!scratch) status = NMS_MEMORY;
        }
        if (status == NMS_OK) {
            uint32_t written = 0;
            position = 0;
            while (find_bytes(a, b, position, &found)) {
                uint32_t segment = found - position;
                copy_bytes(scratch + written, a.data + position, segment);
                written += segment;
                copy_bytes(scratch + written, c.data, c.length);
                written += c.length;
                position = found + b.length;
            }
            copy_bytes(scratch + written, a.data + position, a.length - position);
            scratch[length] = 0;
            /* All three immutable byte views remain owned through this copy;
             * descriptor-table relocation cannot invalidate their storage. */
            status = nms_create(runtime, scratch, length, &result);
        }
    }
    deallocate(scratch);
    NmsStatus sa = nms_release(runtime, source);
    NmsStatus sb = nms_release(runtime, needle);
    NmsStatus sc = nms_release(runtime, replacement);
    if (status != NMS_OK) return status;
    if (sa != NMS_OK || sb != NMS_OK || sc != NMS_OK) {
        nms_release(runtime, result);
        return sa != NMS_OK ? sa : sb != NMS_OK ? sb : sc;
    }
    *out = result;
    return NMS_OK;
}
NmsStatus nms_char_at(const NmsRuntime *runtime, NmsHandle source,
                      uint64_t index_bits, uint32_t is_integer, int64_t *out) {
    if (!out || is_integer > 1) return NMS_STATE;
    NmsView view;
    NmsStatus status = nms_view(runtime, source, &view);
    if (status != NMS_OK) return status;
    uint64_t index = is_integer ? index_bits : 0;
    *out = (index & (UINT64_C(1) << 63)) || index >= view.length
         ? -1 : (int64_t)view.data[(uint32_t)index];
    return NMS_OK;
}
NmsStatus nms_predicate(const NmsRuntime *runtime, NmsHandle source, NmsHandle affix,
                        uint32_t operation, uint32_t *out) {
    if (!out || operation > NMS_ENDS_WITH) return NMS_STATE;
    NmsView haystack, needle;
    NmsStatus status = nms_view(runtime, source, &haystack);
    if (status != NMS_OK) return status;
    status = nms_view(runtime, affix, &needle);
    if (status != NMS_OK) return status;
    uint32_t answer = 0;
    if (needle.length == 0) answer = 1;
    else if (needle.length <= haystack.length) {
        uint32_t last = haystack.length - needle.length;
        if (operation == NMS_STARTS_WITH)
            answer = equal_bytes(haystack.data, needle.data, needle.length);
        else if (operation == NMS_ENDS_WITH)
            answer = equal_bytes(haystack.data + last, needle.data, needle.length);
        else {
            /* A nonempty needle makes last < UINT32_MAX; increment cannot wrap. */
            for (uint32_t position = 0; position <= last; position++) {
                if (equal_bytes(haystack.data + position, needle.data, needle.length)) {
                    answer = 1;
                    break;
                }
            }
        }
    }
    *out = answer;
    return NMS_OK;
}
