#ifndef NANO_COP_OPAQUE_H
#define NANO_COP_OPAQUE_H

#include "value.h"
#include <sys/types.h>

/* I own only token metadata. Provider objects keep their declared lifetime. */
typedef struct {
    uint32_t generation;
    pid_t process;
    uint64_t *issued;
    size_t count, capacity;
    uint8_t *reply;
    size_t reply_capacity;
} CopOpaqueOwner;

typedef struct {
    void **pointers;
    size_t count, capacity;
    uint8_t *reply;
    size_t reply_capacity;
} CopOpaqueWorker;

bool cop_opaque_owner_start(CopOpaqueOwner *owner);
void cop_opaque_owner_clear(CopOpaqueOwner *owner);
bool cop_opaque_owner_live(const CopOpaqueOwner *owner);
bool cop_opaque_owner_reserve(CopOpaqueOwner *owner, size_t additional);
bool cop_opaque_owner_reply(CopOpaqueOwner *owner, size_t capacity);
bool cop_opaque_owner_argument(const CopOpaqueOwner *owner, NanoValue value);
bool cop_opaque_owner_preview(const CopOpaqueOwner *owner, NanoValue wire,
                              NanoValue *value);
bool cop_opaque_owner_publish(CopOpaqueOwner *owner, NanoValue value);

bool cop_opaque_worker_reserve(CopOpaqueWorker *worker, size_t additional,
                               size_t reply_capacity);
void cop_opaque_worker_clear(CopOpaqueWorker *worker);
bool cop_opaque_worker_argument(const CopOpaqueWorker *worker, NanoValue wire,
                                NanoValue *local);
/* I perform no allocation here: the caller reserves before foreign entry. */
bool cop_opaque_worker_capture(void *context, void *pointer, uint64_t *slot);

#endif
