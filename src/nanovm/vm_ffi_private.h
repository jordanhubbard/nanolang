/* I share a fresh service classification only until native dispatch begins. */
#ifndef NANOVM_FFI_PRIVATE_H
#define NANOVM_FFI_PRIVATE_H
#include "vm_ffi.h"
#include "../nanoisa/service_classification_private.h"
/* The caller owns immutable code facts within one invocation and refreshes
 * mutable bindings immediately before this call. NULL or mismatched facts
 * take the full public service check. I never retain facts across host code. */
bool vm_ffi_call_vm_classified(VmState *vm, const NvmModule *module,
    uint32_t import_idx, NanoValue *args, int arg_count, NanoValue *result,
    char *error_msg, size_t error_msg_size, const NvmServiceClassification *facts);
#endif
