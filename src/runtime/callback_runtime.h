#ifndef NANO_CALLBACK_RUNTIME_H
#define NANO_CALLBACK_RUNTIME_H

#include "nano_callback.h"
#include <stdbool.h>

typedef struct NanoCallbackRuntime NanoCallbackRuntime;
typedef NanoCallbackStatus (*NanoCallbackExecute)(void *payload,
    const NanoCallbackValue *arguments, uint32_t count,
    NanoCallbackValue *result);
typedef void (*NanoCallbackDrop)(void *payload);

/* I bind the runtime to its creating thread. Only that thread may publish,
 * pump, collect, close, or destroy. Native handles may be used on any thread.
 * A failed publication leaves payload ownership with its caller. */
NanoCallbackRuntime *nano_callback_runtime_create(void);
NanoCallbackV1 *nano_callback_create(NanoCallbackRuntime *runtime,
    const NanoCallbackSignature *signature, NanoCallbackExecute execute,
    NanoCallbackDrop drop, void *payload);

/* I execute at most one request. wait permits a condition-variable wait;
 * spurious or explicit wakeups can return 0. Results: 1 = executed,
 * 0 = no request, -1 = wrong thread. I never execute under my queue lock. */
int nano_callback_pump(NanoCallbackRuntime *runtime, bool wait);
/* A foreign-call worker may wake my owner after completing a native call.
 * The host must keep the runtime alive until that worker has joined. */
void nano_callback_wake(NanoCallbackRuntime *runtime);
void nano_callback_collect(NanoCallbackRuntime *runtime);
NanoCallbackStatus nano_callback_close(NanoCallbackRuntime *runtime);
NanoCallbackStatus nano_callback_runtime_destroy(NanoCallbackRuntime *runtime);

#endif
