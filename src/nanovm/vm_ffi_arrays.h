#ifndef NANOVM_FFI_ARRAYS_H
#define NANOVM_FFI_ARRAYS_H
#include "heap.h"
#include "../nanoisa/nvm_format.h"
#include "../runtime/dyn_array.h"

typedef struct {
    VmArray *original;
    DynArray *native;
    VmArray *pending;
    char **strings;
    uint32_t string_count;
    uint8_t element_tag;
} VmFfiArrayEntry;

typedef struct {
    VmHeap *heap;
    int count;
    VmFfiArrayEntry entries[NANO_MAX_FFI_ARGS];
} VmFfiArrayFrame;

bool vm_ffi_array_argument(VmFfiArrayFrame *frame, NanoValue value, void **out,
                           char *error, size_t size);
VmArray *vm_ffi_array_import(VmHeap *heap, const DynArray *array, char *error, size_t size);
bool vm_ffi_arrays_commit(VmFfiArrayFrame *frame, char *error, size_t size);
bool vm_ffi_array_alias_result(VmFfiArrayFrame *frame, const void *pointer, NanoValue *out);
void vm_ffi_arrays_dispose(VmFfiArrayFrame *frame);
#endif
