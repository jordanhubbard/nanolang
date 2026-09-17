#include "vm.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    VmState *vm;
    NanoValue callable;
} VmCallbackRoot;

static uint32_t native_tag(uint8_t tag) {
    switch (tag) {
    case TAG_VOID: return NANO_CALLBACK_VOID;
    case TAG_INT: return NANO_CALLBACK_INT;
    case TAG_FLOAT: return NANO_CALLBACK_FLOAT;
    case TAG_BOOL: return NANO_CALLBACK_BOOL;
    case TAG_U8: return NANO_CALLBACK_BYTE;
    case TAG_OPAQUE: return NANO_CALLBACK_POINTER;
    default: return UINT32_MAX;
    }
}

static void callback_drop(void *payload) {
    VmCallbackRoot *root = payload;
    vm_release(&root->vm->heap, root->callable);
    free(root);
}

static NanoCallbackStatus callback_execute(void *payload, const NanoCallbackValue *args,
                                            uint32_t count, NanoCallbackValue *result) {
    VmCallbackRoot *root = payload;
    NanoValue values[NANO_CALLBACK_MAX_ARGS];
    for (uint32_t i = 0; i < count; i++) {
        switch (args[i].tag) {
        case NANO_CALLBACK_INT: values[i] = val_int(args[i].as.integer); break;
        case NANO_CALLBACK_FLOAT: values[i] = val_float(args[i].as.number); break;
        case NANO_CALLBACK_BOOL: values[i] = val_bool(args[i].as.byte != 0); break;
        case NANO_CALLBACK_BYTE: values[i] = val_u8(args[i].as.byte); break;
        case NANO_CALLBACK_POINTER:
            values[i] = val_void(); values[i].tag = TAG_OPAQUE;
            values[i].as.obj = args[i].as.pointer; break;
        default: return NANO_CALLBACK_TYPE_ERROR;
        }
    }
    NanoValue returned = val_void();
    VmResult status = vm_invoke_callable(root->vm, root->callable, values, (uint16_t)count, &returned);
    if (status != VM_OK) {
        if (root->vm->callback_error == VM_OK) {
            root->vm->callback_error = status;
            memcpy(root->vm->callback_error_msg, root->vm->error_msg, sizeof(root->vm->callback_error_msg));
        }
        return NANO_CALLBACK_EXECUTION_ERROR;
    }
    memset(result, 0, sizeof(*result));
    result->tag = native_tag(returned.tag);
    switch (returned.tag) {
    case TAG_VOID: break;
    case TAG_INT: result->as.integer = returned.as.i64; break;
    case TAG_FLOAT: result->as.number = returned.as.f64; break;
    case TAG_BOOL: result->as.byte = returned.as.boolean; break;
    case TAG_U8: result->as.byte = returned.as.u8; break;
    case TAG_OPAQUE: result->as.pointer = returned.as.obj; break;
    default:
        vm_release(&root->vm->heap, returned);
        return NANO_CALLBACK_TYPE_ERROR;
    }
    return NANO_CALLBACK_OK;
}

NanoCallbackV1 *vm_callback_create(VmState *vm, NanoValue callable,
                                  const NvmCallbackContract *contract) {
    /* I do not write diagnostics or touch heap state from a foreign thread. */
    if (!vm || !pthread_equal(vm->owner_thread, pthread_self())) return NULL;
    if (vm->callbacks_closed || vm->isolate_ffi || !contract ||
        contract->abi_version != NVM_CALLBACK_ABI_RETAINED_V1 ||
        contract->execution > NVM_FOREIGN_WORKER_THREAD ||
        contract->parameter_idx == NVM_CALLBACK_NO_PARAMETER ||
        !nvm_callback_shape_valid(contract->param_tags, contract->param_count, contract->return_tag))
        return NULL;
    const NvmModule *module;
    uint32_t index;
    if (!vm_callable_target(vm, callable, &module, &index)) return NULL;
    const NvmFunctionEntry *fn = &module->functions[index];
    if (fn->arity != contract->param_count || fn->result_count != (contract->return_tag != TAG_VOID) ||
        fn->result_tag != contract->return_tag || fn->local_count < fn->arity ||
        (callable.tag == TAG_CLOSURE ? callable.as.closure->capture_count != fn->upvalue_count :
                                      fn->upvalue_count != 0)) return NULL;
    NanoCallbackSignature signature = {.argument_count = fn->arity,
                                      .result_tag = native_tag(fn->result_tag)};
    for (uint16_t i = 0; i < fn->arity; i++) {
        if (!module->function_param_types || !module->function_param_types[index] ||
            module->function_param_types[index][i] != contract->param_tags[i]) return NULL;
        signature.argument_tags[i] = native_tag(contract->param_tags[i]);
    }
    if (!vm->callbacks) {
        vm->callbacks = nano_callback_runtime_create();
        if (!vm->callbacks) return NULL;
    }
    VmCallbackRoot *root = malloc(sizeof(*root));
    if (!root) return NULL;
    root->vm = vm;
    root->callable = callable;
    vm_retain(&vm->heap, callable);
    NanoCallbackV1 *handle = nano_callback_create(vm->callbacks, &signature,
                                                 callback_execute, callback_drop, root);
    if (!handle) callback_drop(root);
    return handle;
}

int vm_callback_pump(VmState *vm, bool wait) {
    if (!vm || !pthread_equal(vm->owner_thread, pthread_self())) return -1;
    if (!vm->callbacks || vm->callbacks_closed) return 0;
    nano_callback_collect(vm->callbacks);
    int result = nano_callback_pump(vm->callbacks, wait);
    nano_callback_collect(vm->callbacks);
    return result;
}

NanoCallbackStatus vm_callback_shutdown(VmState *vm) {
    if (!vm || !pthread_equal(vm->owner_thread, pthread_self())) return NANO_CALLBACK_WRONG_THREAD;
    if (vm->callbacks) {
        NanoCallbackStatus status = nano_callback_runtime_destroy(vm->callbacks);
        if (status != NANO_CALLBACK_OK) return status;
        vm->callbacks = NULL;
    }
    vm->callbacks_closed = true;
    return NANO_CALLBACK_OK;
}
