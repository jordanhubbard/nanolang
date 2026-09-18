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
        for (uint64_t i = 0; i <= new_capacity; i++) {
            if (runtime->slots && i <= runtime->capacity) {
                slots[i].data = runtime->slots[i].data;
                slots[i].references = runtime->slots[i].references;
                slots[i].length = runtime->slots[i].length;
                slots[i].next_free = runtime->slots[i].next_free;
                slots[i].kind = runtime->slots[i].kind;
                slots[i].capacity = runtime->slots[i].capacity;
            } else {
                slots[i].data = NULL; slots[i].references = 0; slots[i].length = 0;
                slots[i].kind = NMS_SLOT_FREE; slots[i].capacity = 0;
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
    NmsSlot *slot = &runtime->slots[index];
    if (--slot->references) return NMS_OK;
    if (slot->kind == NMS_SLOT_STRING_ARRAY) {
        NmsHandle *children = (NmsHandle *)slot->data;
        for (uint32_t i = 0; i < slot->length; i++) nms_release(runtime, children[i]);
        runtime->live_bytes -= (uint64_t)slot->capacity * sizeof(NmsHandle);
    } else runtime->live_bytes -= slot->length;
    runtime->live_objects--;
    deallocate(slot->data);
    slot->data = NULL; slot->length = 0; slot->capacity = 0; slot->kind = NMS_SLOT_FREE;
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
                                 const unsigned char *bytes, uint32_t length) {
    NmsHandle child = 0;
    NmsStatus status = nms_create(runtime, bytes, length, &child);
    if (status != NMS_OK) return status;
    status = nms_string_array_append(runtime, array, child);
    NmsStatus released = nms_release(runtime, child);
    return status != NMS_OK ? status : released;
}
NmsStatus nms_split_owned(NmsRuntime *runtime, NmsHandle source, NmsHandle delimiter,
                          NmsHandle *out) {
    NmsView a, b;
    NmsHandle array = 0;
    NmsStatus status = out ? nms_view(runtime, source, &a) : NMS_STATE;
    if (status == NMS_OK) status = nms_view(runtime, delimiter, &b);
    if (status == NMS_OK) status = nms_string_array_create(runtime, &array);
    if (status == NMS_OK && !b.length) {
        for (uint32_t i = 0; i < a.length && status == NMS_OK; i++)
            status = append_segment(runtime, array, a.data + i, 1);
    } else if (status == NMS_OK) {
        uint32_t position = 0, found = 0;
        while (status == NMS_OK && find_bytes(a, b, position, &found)) {
            status = append_segment(runtime, array, a.data + position, found - position);
            position = found + b.length;
        }
        if (status == NMS_OK)
            status = append_segment(runtime, array, a.data + position, a.length - position);
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
