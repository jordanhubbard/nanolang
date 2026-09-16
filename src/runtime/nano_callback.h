#ifndef NANO_CALLBACK_H
#define NANO_CALLBACK_H

#include <stdint.h>

/* I expose a retained handle, not a plain C function pointer. */
#define NANO_CALLBACK_ABI_V1 1u
#define NANO_CALLBACK_MAX_ARGS 16u

typedef enum {
    NANO_CALLBACK_VOID = 0,
    NANO_CALLBACK_INT = 1,
    NANO_CALLBACK_FLOAT = 2,
    NANO_CALLBACK_BOOL = 3,
    NANO_CALLBACK_BYTE = 4,
    NANO_CALLBACK_POINTER = 5
} NanoCallbackTag;

typedef struct {
    uint32_t tag;
    union { int64_t integer; double number; uint8_t byte; void *pointer; } as;
} NanoCallbackValue;

typedef struct {
    uint32_t argument_count;
    uint32_t result_tag;
    uint32_t argument_tags[NANO_CALLBACK_MAX_ARGS];
} NanoCallbackSignature;

typedef enum {
    NANO_CALLBACK_OK = 0,
    NANO_CALLBACK_CANCELLED,
    NANO_CALLBACK_TYPE_ERROR,
    NANO_CALLBACK_EXECUTION_ERROR,
    NANO_CALLBACK_WRONG_THREAD,
    NANO_CALLBACK_BUSY
} NanoCallbackStatus;

typedef struct NanoCallbackV1 NanoCallbackV1;
struct NanoCallbackV1 {
    uint32_t abi_version;
    NanoCallbackSignature signature;
    void (*retain)(NanoCallbackV1 *callback);
    void (*release)(NanoCallbackV1 *callback);
    NanoCallbackStatus (*invoke)(NanoCallbackV1 *callback,
        const NanoCallbackValue *arguments, uint32_t count,
        NanoCallbackValue *result);
};

/* I keep this header independent of VM layouts and runtime linkage. Native
 * adapters borrow the handle, retain before publication, and release after
 * their last possible invocation. All fields are immutable to adapters.
 * invoke borrows a live reference and argument storage until it returns.
 * Pointer values are borrowed, in-process addresses; they never travel over
 * COP. On any failure, result is void. A NULL result discards a valid result.
 * These operations are not async-signal-safe. */

#endif
