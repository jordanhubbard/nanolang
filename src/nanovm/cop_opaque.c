#include "cop_opaque.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

_Static_assert(__atomic_always_lock_free(sizeof(uint32_t), 0),
               "I require lock-free COP generation allocation");
static uint32_t last_generation;

static bool grow(void **storage, size_t *capacity, size_t count,
                 size_t additional, size_t width) {
    if (additional > SIZE_MAX - count) return false;
    size_t needed = count + additional;
    if (needed <= *capacity) return true;
    if (needed > SIZE_MAX / width) return false;
    size_t next = *capacity ? *capacity : 8;
    while (next < needed) {
        if (next > SIZE_MAX / 2) { next = needed; break; }
        next *= 2;
    }
    if (next > SIZE_MAX / width) next = needed;
    void *copy = malloc(next * width);
    if (!copy) return false;
    if (count) memcpy(copy, *storage, count * width);
    free(*storage);
    *storage = copy;
    *capacity = next;
    return true;
}

void cop_opaque_owner_clear(CopOpaqueOwner *owner) {
    free(owner->issued);
    free(owner->reply);
    memset(owner, 0, sizeof *owner);
}

bool cop_opaque_owner_start(CopOpaqueOwner *owner) {
    if (!owner || owner->generation || owner->issued) return false;
    uint32_t previous = __atomic_load_n(&last_generation, __ATOMIC_RELAXED);
    do {
        if (previous == UINT32_MAX) return false;
    } while (!__atomic_compare_exchange_n(&last_generation, &previous, previous + 1,
                                          false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    owner->generation = previous + 1;
    owner->process = getpid();
    return true;
}

bool cop_opaque_owner_live(const CopOpaqueOwner *owner) {
    return owner && owner->generation && owner->process == getpid();
}

bool cop_opaque_owner_reserve(CopOpaqueOwner *owner, size_t additional) {
    if (!cop_opaque_owner_live(owner)) return false;
    void *storage = owner->issued;
    if (!grow(&storage, &owner->capacity, owner->count, additional, sizeof *owner->issued)) return false;
    owner->issued = storage;
    return true;
}

bool cop_opaque_owner_reply(CopOpaqueOwner *owner, size_t capacity) {
    if (!cop_opaque_owner_live(owner)) return false;
    if (capacity <= owner->reply_capacity) return true;
    uint8_t *reply = malloc(capacity);
    if (!reply) return false;
    free(owner->reply);
    owner->reply = reply;
    owner->reply_capacity = capacity;
    return true;
}

static bool issued(const CopOpaqueOwner *owner, uint64_t slot) {
    for (size_t i = 0; i < owner->count; ++i)
        if (owner->issued[i] == slot) return true;
    return false;
}

bool cop_opaque_owner_argument(const CopOpaqueOwner *owner, NanoValue value) {
    if (value.tag != TAG_OPAQUE) return true;
    if (!value.as.i64) return value.opaque_owner == 0;
    return cop_opaque_owner_live(owner) && value.opaque_owner == owner->generation &&
        issued(owner, (uint64_t)value.as.i64);
}

bool cop_opaque_owner_preview(const CopOpaqueOwner *owner, NanoValue wire,
                              NanoValue *value) {
    if (wire.tag != TAG_OPAQUE || wire.opaque_owner || !cop_opaque_owner_live(owner)) return false;
    uint64_t slot = (uint64_t)wire.as.i64;
    if (slot > SIZE_MAX / sizeof(void *)) return false;
    if (slot && !issued(owner, slot) && owner->count >= owner->capacity) return false;
    *value = wire;
    value->opaque_owner = slot ? owner->generation : 0;
    return true;
}

bool cop_opaque_owner_publish(CopOpaqueOwner *owner, NanoValue value) {
    if (!cop_opaque_owner_live(owner) || value.tag != TAG_OPAQUE) return false;
    uint64_t slot = (uint64_t)value.as.i64;
    if (!slot) return value.opaque_owner == 0;
    if (value.opaque_owner != owner->generation) return false;
    if (issued(owner, slot)) return true;
    if (owner->count >= owner->capacity) return false;
    owner->issued[owner->count++] = slot;
    return true;
}

bool cop_opaque_worker_reserve(CopOpaqueWorker *worker, size_t additional,
                               size_t reply_capacity) {
    if (!worker || additional > UINT64_MAX - worker->count) return false;
    void *storage = worker->pointers;
    if (!grow(&storage, &worker->capacity, worker->count, additional, sizeof *worker->pointers)) return false;
    worker->pointers = storage;
    if (reply_capacity > worker->reply_capacity) {
        uint8_t *reply = malloc(reply_capacity);
        if (!reply) return false;
        free(worker->reply);
        worker->reply = reply;
        worker->reply_capacity = reply_capacity;
    }
    return true;
}

void cop_opaque_worker_clear(CopOpaqueWorker *worker) {
    free(worker->pointers);
    free(worker->reply);
    memset(worker, 0, sizeof *worker);
}

bool cop_opaque_worker_argument(const CopOpaqueWorker *worker, NanoValue wire,
                                NanoValue *local) {
    if (wire.tag == TAG_INT && !wire.as.i64) { *local = val_opaque(NULL); return true; }
    if (wire.tag != TAG_OPAQUE || wire.opaque_owner) return false;
    uint64_t slot = (uint64_t)wire.as.i64;
    if (slot > worker->count) return false;
    *local = val_opaque(slot ? worker->pointers[slot - 1] : NULL);
    return true;
}

bool cop_opaque_worker_capture(void *context, void *pointer, uint64_t *slot) {
    CopOpaqueWorker *worker = context;
    if (!pointer) { *slot = 0; return true; }
    for (size_t i = 0; i < worker->count; ++i)
        if (worker->pointers[i] == pointer) { *slot = i + 1; return true; }
    if (worker->count == worker->capacity) return false;
    worker->pointers[worker->count++] = pointer;
    *slot = worker->count;
    return true;
}
