/*
 * NanoVM FFI Bridge - Call native C functions from the VM
 *
 * Module loading and symbol resolution are delegated to the shared
 * ffi_loader. This file handles VM-specific marshaling between
 * NanoValue and C function signatures, plus module introspection.
 */

/* usleep(), kill(), fork(), pipe(), exec*() need _GNU_SOURCE */

#include "../nanoisa/service_bindings_module.h"
#include "vm_ffi.h"
#include "vm_ffi_arrays.h"
#include "module_builder.h"
#include "runtime/dyn_array.h"
#include "runtime/ffi_loader.h"
#include "runtime/path_normalize.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <ffi.h>

/* ========================================================================
 * Module Registry (delegates to ffi_loader)
 * ======================================================================== */

static Environment *ffi_env = NULL;  /* For module introspection */

void vm_ffi_set_env(Environment *env) {
    ffi_env = env;
}

void vm_ffi_init(void) {
    ffi_loader_init(false);
}

void vm_ffi_shutdown(void) {
    ffi_loader_shutdown();
}

bool vm_ffi_load_import(const NvmModule *module, uint32_t import_idx) {
    if (nvm_service_execution_pending(module)) return false;
    if (!module || import_idx >= module->import_count) return false;
    const NvmImportEntry *imp = &module->imports[import_idx];
    const char *name = nvm_get_string(module, imp->module_name_idx);
    if (imp->kind == NVM_IMPORT_ARTIFACT) {
        if (!name || name[0] != '/' ||
            strlen(name) != nvm_get_string_len(module, imp->module_name_idx)) return false;
        if (!ffi_loader_is_initialized()) ffi_loader_init(false);
        return ffi_loader_open(name, name);
    }
    if (imp->kind > NVM_IMPORT_ARTIFACT) return false;
    return vm_ffi_load_module(name);
}

bool vm_ffi_load_module(const char *module_name) {
    if (!module_name || !module_name[0]) return false;
    if (!ffi_loader_is_initialized()) ffi_loader_init(false);
    if (ffi_loader_find(module_name)) return true;

    /* I retain source-path context, as the interpreter does. Logical module
     * names still use the shared loader's standard-module fallbacks. */
    char *module_dir = NULL;
    size_t name_len = strlen(module_name);
    if (name_len >= 5 && strcmp(module_name + name_len - 5, ".nano") == 0) {
        module_dir = strdup(module_name);
        if (!module_dir) return false;
        char *slash = strrchr(module_dir, '/');
        if (slash == module_dir) slash[1] = '\0';
        else if (slash) *slash = '\0';
        else strcpy(module_dir, ".");
    }
    char path[1024];
    /* I read build metadata to resolve its library name; I never build or
     * install dependencies from the runtime loader. */
    ModuleBuildMetadata *meta = module_dir ? module_load_metadata(module_dir) : NULL;
    bool found = ffi_loader_find_library(meta ? meta->name : module_name,
                                         module_dir, path, sizeof(path));
    module_metadata_free(meta);
    free(module_dir);
    if (!found) {
        /* Not fatal - function might be in main executable or already-loaded lib */
        return false;
    }

    return ffi_loader_open(module_name, path);
}

/* ========================================================================
 * NanoValue ↔ C Marshaling
 * ======================================================================== */

/* Marshal NanoValue args to C void* array for polymorphic dispatch */
static bool marshal_args(NanoValue *args, int arg_count,
                        const NvmImportEntry *imp, const uint8_t *param_types,
                        void **arg_ptrs, VmFfiArrayFrame *arrays, char *error, size_t size) {
    for (int i = 0; i < arg_count; i++) {
        uint8_t expected_tag = (i < imp->param_count && param_types)
                               ? param_types[i] : args[i].tag;

        switch (expected_tag) {
            case TAG_INT:
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_FLOAT:
                /* Typed dispatch reads floating arguments directly below. */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_BOOL:
                arg_ptrs[i] = (void *)(intptr_t)(args[i].as.boolean ? 1 : 0);
                break;
            case TAG_STRING:
                if (args[i].tag == TAG_STRING && args[i].as.string) {
                    arg_ptrs[i] = (void *)vmstring_cstr(args[i].as.string);
                } else {
                    arg_ptrs[i] = (void *)"";
                }
                break;
            case TAG_OPAQUE:
                /* Opaque values stored as raw pointer in i64 */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_ARRAY:
                if (!vm_ffi_array_argument(arrays, args[i], &arg_ptrs[i], error, size)) return false;
                break;
            default:
                /* Pass as raw int64 (best effort) */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
        }
    }
    return true;
}

/* Convert C int64_t result to NanoValue based on return type tag */
static NanoValue marshal_result(int64_t raw_result, uint8_t return_tag,
                                VmHeap *heap, bool *success) {
    *success = true;
    switch (return_tag) {
        case TAG_INT:
            return val_int(raw_result);
        case TAG_U8:
            return val_u8((uint8_t)raw_result);
        case TAG_FLOAT: {
            /* Result is actually a double bit-pattern in int64_t */
            double d;
            memcpy(&d, &raw_result, sizeof(double));
            return val_float(d);
        }
        case TAG_BOOL:
            return val_bool(raw_result != 0);
        case TAG_STRING: {
            const char *str = (const char *)(intptr_t)raw_result;
            if (str) {
                size_t length = strlen(str);
                if (length > UINT32_MAX) { *success = false; return val_void(); }
                VmString *vs = vm_string_new(heap, str, (uint32_t)length);
                if (!vs) { *success = false; return val_void(); }
                return val_string(vs);
            }
            return val_void();
        }
        case TAG_VOID:
            return val_void();
        case TAG_OPAQUE: {
            /* Opaque pointer stored as int64 */
            NanoValue v = {0};
            v.tag = TAG_OPAQUE;
            v.as.i64 = raw_result;
            return v;
        }
        case TAG_ARRAY: {
            DynArray *array = (DynArray *)(intptr_t)raw_result;
            if (!array) return val_void();
            VmArray *copy = vm_ffi_array_import(heap, array, NULL, 0);
            if (!copy) { *success = false; return val_void(); }
            return val_array(copy);
        }
        default:
            return val_int(raw_result);
    }
}

/* ========================================================================
 * FFI Call Dispatch
 * ======================================================================== */

/* ========================================================================
 * Resolve-once typed call descriptors
 *
 * The first FFI call for a given import resolves the module, looks up the
 * native symbol through the shared loader, and precomputes the typed
 * signature (declared param types, return type, all-float classification).
 * The result is cached on the module keyed by import index so subsequent
 * calls skip module loading, string-pool lookups, and symbol resolution.
 * ======================================================================== */

/* Handle the ___module_* introspection pseudo-imports. Returns true when the
 * call was an introspection request (and *result was filled), false otherwise.
 * These are resolved on every call because their result depends on runtime
 * arguments and environment state, so they are intentionally never cached. */
static bool vm_ffi_try_module_introspection(const char *func_name,
                                            NanoValue *args, int arg_count,
                                            NanoValue *result, VmHeap *heap) {
    if (!func_name || strncmp(func_name, "___module_", 10) != 0 || !ffi_env) {
        return false;
    }
    const char *rest = func_name + 10;

    if (strncmp(rest, "is_unsafe_", 10) == 0) {
        const char *mname = rest + 10;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        *result = val_bool(mi ? mi->is_unsafe : false);
        return true;
    }
    if (strncmp(rest, "has_ffi_", 8) == 0) {
        const char *mname = rest + 8;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        *result = val_bool(mi ? mi->has_ffi : false);
        return true;
    }
    if (strncmp(rest, "name_", 5) == 0) {
        const char *mname = rest + 5;
        VmString *vs = vm_string_new(heap, mname, (uint32_t)strlen(mname));
        *result = val_string(vs);
        return true;
    }
    if (strncmp(rest, "path_", 5) == 0) {
        const char *mname = rest + 5;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        const char *path = (mi && mi->path) ? mi->path : "";
        VmString *vs = vm_string_new(heap, path, (uint32_t)strlen(path));
        *result = val_string(vs);
        return true;
    }
    if (strncmp(rest, "function_count_", 15) == 0) {
        const char *mname = rest + 15;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        *result = val_int(mi ? mi->function_count : 0);
        return true;
    }
    if (strncmp(rest, "function_name_", 14) == 0) {
        const char *mname = rest + 14;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        int64_t idx = (arg_count >= 1) ? args[0].as.i64 : 0;
        const char *fn = "";
        if (mi && mi->exported_functions && idx >= 0 && idx < mi->function_count) {
            fn = mi->exported_functions[idx] ? mi->exported_functions[idx] : "";
        }
        VmString *vs = vm_string_new(heap, fn, (uint32_t)strlen(fn));
        *result = val_string(vs);
        return true;
    }
    if (strncmp(rest, "struct_count_", 13) == 0) {
        const char *mname = rest + 13;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        *result = val_int(mi ? mi->struct_count : 0);
        return true;
    }
    if (strncmp(rest, "struct_name_", 12) == 0) {
        const char *mname = rest + 12;
        ModuleInfo *mi = env_get_module(ffi_env, mname);
        int64_t idx = (arg_count >= 1) ? args[0].as.i64 : 0;
        const char *sn = "";
        if (mi && mi->exported_structs && idx >= 0 && idx < mi->struct_count) {
            sn = mi->exported_structs[idx] ? mi->exported_structs[idx] : "";
        }
        VmString *vs = vm_string_new(heap, sn, (uint32_t)strlen(sn));
        *result = val_string(vs);
        return true;
    }
    return false;
}

/* Compute the all-float classification for a fully declared signature. */
static bool descriptor_all_float(const NvmCallDescriptor *desc) {
    if (desc->return_type != TAG_FLOAT) return false;
    for (uint16_t i = 0; i < desc->param_count; i++) {
        uint8_t tag = desc->param_types ? desc->param_types[i] : TAG_FLOAT;
        if (tag != TAG_FLOAT) return false;
    }
    return true;
}

/* Resolve (once) and return the typed call descriptor for import_idx. On the
 * first call the module is loaded and the symbol looked up; the result is
 * cached and reused thereafter. Returns NULL and fills error_msg on failure. */
static const NvmCallDescriptor *vm_ffi_resolve_descriptor(
        const NvmModule *module, uint32_t import_idx,
        char *error_msg, size_t error_msg_size) {
    /* The descriptor cache is a runtime-only memoization side table, so we
     * mutate it through a non-const view even for a const module. */
    NvmModule *m = (NvmModule *)module;

    if (import_idx >= m->import_count) {
        snprintf(error_msg, error_msg_size, "Import index %u out of range", import_idx);
        return NULL;
    }

    if (!m->call_descriptors || m->call_descriptor_count != m->import_count) {
        NvmCallDescriptor *table = calloc(m->import_count,
                                          sizeof(NvmCallDescriptor));
        if (!table) {
            snprintf(error_msg, error_msg_size,
                     "Out of memory allocating call descriptors");
            return NULL;
        }
        free(m->call_descriptors);
        m->call_descriptors = table;
        m->call_descriptor_count = m->import_count;
    }

    NvmCallDescriptor *desc = &m->call_descriptors[import_idx];

    if (desc->state == NVM_CALL_RESOLVED) {
        return desc;
    }
    if (desc->state == NVM_CALL_FAILED) {
        const char *fn = desc->func_name ? desc->func_name : "?";
        const char *mn = desc->module_name ? desc->module_name : "";
        snprintf(error_msg, error_msg_size,
                 "FFI: function '%s' not found (module '%s')", fn, mn);
        return NULL;
    }

    const NvmImportEntry *imp = &m->imports[import_idx];
    const char *func_name = nvm_get_string(m, imp->function_name_idx);
    const char *mod_name = nvm_get_string(m, imp->module_name_idx);

    desc->func_name = func_name;
    desc->module_name = mod_name;
    desc->param_count = imp->param_count;
    desc->return_type = imp->return_type;
    desc->param_types = m->import_param_types
                        ? m->import_param_types[import_idx] : NULL;
    desc->all_float = descriptor_all_float(desc);

    if (!func_name) {
        desc->state = NVM_CALL_FAILED;
        snprintf(error_msg, error_msg_size,
                 "NULL function name for import %u", import_idx);
        return NULL;
    }

    /* I require exact loading for artifacts. Only logical imports retain
     * best-effort loading and the legacy global symbol search. */
    bool loaded = vm_ffi_load_import(module, import_idx);
    void *func_ptr = NULL;
    if (imp->kind == NVM_IMPORT_ARTIFACT) {
        if (loaded) func_ptr = ffi_loader_resolve_module(func_name, mod_name);
    } else if (imp->kind <= NVM_IMPORT_COPROCESS) {
        func_ptr = ffi_loader_resolve(func_name);
    }
    if (!func_ptr) {
        desc->state = NVM_CALL_FAILED;
        snprintf(error_msg, error_msg_size,
                 "FFI: function '%s' not found (module '%s')",
                 func_name, mod_name ? mod_name : "");
        return NULL;
    }

    bool has_array = imp->return_type == TAG_ARRAY;
    for (uint16_t i = 0; i < imp->param_count; ++i)
        if (desc->param_types && desc->param_types[i] == TAG_ARRAY) has_array = true;
    if (has_array &&
        !ffi_loader_check_array_abi(imp->kind == NVM_IMPORT_ARTIFACT ? mod_name : NULL, func_name, func_ptr,
                                    NANO_DYN_ARRAY_ABI_VERSION, error_msg, error_msg_size)) {
        desc->state = NVM_CALL_FAILED;
        return NULL;
    }

    if (imp->kind == NVM_IMPORT_ARTIFACT && imp->return_type == TAG_STRING) {
        if (!ffi_loader_string_release(mod_name, func_name, func_ptr,
                                       &desc->string_release, error_msg, error_msg_size)) {
            desc->state = NVM_CALL_FAILED;
            return NULL;
        }
    }
    desc->func_ptr = func_ptr;
    desc->state = NVM_CALL_RESOLVED;
    return desc;
}

static bool callback_contract_pending(const NvmModule *module, uint32_t import_idx,
                                      char *error, size_t size) {
    for (uint32_t i = 0; i < module->callback_contract_count; i++) {
        if (module->callback_contracts[i].import_idx == import_idx) {
            snprintf(error, size, "I require the retained callback scheduler for this import");
            return true;
        }
    }
    return false;
}

typedef union {
    int64_t integer;
    double number;
    uint8_t byte;
    void *pointer;
    ffi_arg word;
} CallbackNativeSlot;

typedef struct {
    ffi_cif *cif;
    void *function;
    void **arguments;
    CallbackNativeSlot returned;
    pthread_mutex_t mutex;
    bool done;
    NanoCallbackRuntime *runtime;
    bool string_result, string_copy_failed;
    char *returned_string;
    uint32_t returned_length;
} CallbackNativeCall;

/* I snapshot borrowed results before worker TLS destructors can reclaim them.
 * Only the owner creates VM heap objects after the worker has joined. */
static void callback_native_invoke(CallbackNativeCall *call) {
    ffi_call(call->cif, FFI_FN(call->function), &call->returned, call->arguments);
    if (call->string_result && call->returned.pointer) {
        size_t length = strlen(call->returned.pointer);
        if (length > UINT32_MAX || length == SIZE_MAX) {
            call->string_copy_failed = true;
            return;
        }
        call->returned_string = malloc(length + 1);
        if (!call->returned_string) {
            call->string_copy_failed = true;
            return;
        }
        memcpy(call->returned_string, call->returned.pointer, length + 1);
        call->returned_length = (uint32_t)length;
    }
}

static void *callback_native_worker(void *opaque) {
    CallbackNativeCall *call = opaque;
    callback_native_invoke(call);
    pthread_mutex_lock(&call->mutex);
    call->done = true;
    pthread_mutex_unlock(&call->mutex);
    nano_callback_wake(call->runtime);
    return NULL;
}

static ffi_type *callback_native_type(uint8_t tag) {
    switch (tag) {
    case TAG_VOID: return &ffi_type_void;
    case TAG_INT: return &ffi_type_sint64;
    case TAG_FLOAT: return &ffi_type_double;
    case TAG_BOOL: case TAG_U8: return &ffi_type_uint8;
    case TAG_STRING: case TAG_OPAQUE: case TAG_FUNCTION: case TAG_CLOSURE: return &ffi_type_pointer;
    default: return NULL;
    }
}

bool vm_ffi_call_vm(VmState *vm, const NvmModule *module, uint32_t import_idx,
                    NanoValue *args, int arg_count, NanoValue *result,
                    char *error_msg, size_t error_msg_size) {
    if (nvm_service_execution_pending(module)) return false;
    if (!vm || !pthread_equal(vm->owner_thread, pthread_self())) return false;
    if (!module || !result) {
        snprintf(error_msg, error_msg_size, "I require a module and result storage for native dispatch");
        return false;
    }
    *result = val_void();
    if (!nvm_callback_contracts_valid(module)) {
        snprintf(error_msg, error_msg_size, "I require valid callback contracts for native dispatch");
        return false;
    }
    const NvmCallbackContract *policy = NULL;
    for (uint32_t i = 0; i < module->callback_contract_count; i++)
        if (module->callback_contracts[i].import_idx == import_idx) { policy = &module->callback_contracts[i]; break; }
    if (!policy) {
        return vm->isolate_ffi
            ? vm_ffi_call_cop(vm, module, import_idx, args, arg_count, result, &vm->heap, error_msg, error_msg_size)
            : vm_ffi_call(module, import_idx, args, arg_count, result, &vm->heap, error_msg, error_msg_size);
    }
    if (import_idx >= module->import_count || vm->callbacks_closed) {
        snprintf(error_msg, error_msg_size, "I require a valid callback contract and live VM host");
        return false;
    }
    const NvmImportEntry *import = &module->imports[import_idx];
    if (vm->isolate_ffi || import->kind == NVM_IMPORT_COPROCESS) {
        snprintf(error_msg, error_msg_size, "I cannot transport retained callback handles through isolated FFI");
        return false;
    }
    if (arg_count != import->param_count || arg_count > NANO_MAX_FFI_ARGS || arg_count < 0 || (arg_count && !args)) {
        snprintf(error_msg, error_msg_size, "I require the declared callback-aware foreign argument count");
        return false;
    }
    ffi_type *return_type = callback_native_type(import->return_type);
    if (!return_type || import->return_type == TAG_FUNCTION || import->return_type == TAG_CLOSURE) {
        snprintf(error_msg, error_msg_size, "I require a scalar or string result for a callback-aware native import");
        return false;
    }
    CallbackNativeSlot storage[NANO_MAX_FFI_ARGS] = {{0}};
    void *values[NANO_MAX_FFI_ARGS];
    ffi_type *types[NANO_MAX_FFI_ARGS];
    NanoCallbackV1 *handles[NANO_MAX_FFI_ARGS] = {0};
    char *strings[NANO_MAX_FFI_ARGS] = {0};
    CallbackNativeCall call = {.string_result = import->return_type == TAG_STRING};
    bool ok = false;
    for (int p = 0; p < arg_count; p++) {
        uint8_t tag = module->import_param_types[import_idx][p];
        types[p] = callback_native_type(tag);
        values[p] = &storage[p];
        bool opaque_null = tag == TAG_OPAQUE && args[p].tag == TAG_INT && args[p].as.i64 == 0;
        if (!types[p] || tag == TAG_VOID ||
            ((tag == TAG_FUNCTION || tag == TAG_CLOSURE) ? !val_is_function(args[p]) :
             (args[p].tag != tag && !opaque_null))) {
            snprintf(error_msg, error_msg_size, "I require matching scalar, string or callable parameters for this native adapter");
            goto cleanup;
        }
        switch (tag) {
        case TAG_INT: storage[p].integer = args[p].as.i64; break;
        case TAG_FLOAT: storage[p].number = args[p].as.f64; break;
        case TAG_BOOL: storage[p].byte = args[p].as.boolean; break;
        case TAG_U8: storage[p].byte = args[p].as.u8; break;
        case TAG_OPAQUE: storage[p].pointer = opaque_null ? NULL : args[p].as.obj; break;
        case TAG_STRING: {
            VmString *string = args[p].as.string;
            if (!string || memchr(string->data, '\0', string->length)) {
                snprintf(error_msg, error_msg_size, "I require a non-null string without embedded NUL bytes for this native adapter");
                goto cleanup;
            }
            size_t length = string->length;
            if (length == SIZE_MAX || !(strings[p] = malloc(length + 1))) {
                snprintf(error_msg, error_msg_size, "I could not copy a native string argument");
                goto cleanup;
            }
            memcpy(strings[p], string->data, length);
            strings[p][length] = '\0';
            storage[p].pointer = strings[p];
            break;
        }
        default: {
            const NvmCallbackContract *contract = policy;
            while (contract->parameter_idx != (uint16_t)p) contract++;
            handles[p] = vm_callback_create(vm, args[p], contract);
            if (!handles[p]) {
                snprintf(error_msg, error_msg_size, "I could not publish a retained callback with the declared target signature");
                goto cleanup;
            }
            storage[p].pointer = handles[p];
        }
        }
    }
    const char *symbol = nvm_get_string(module, policy->adapter_name_idx);
    const char *library = nvm_get_string(module, import->module_name_idx);
    if (!ffi_loader_is_initialized()) vm_ffi_init();
    void *function = vm_ffi_load_import(module, import_idx) && library && library[0]
        ? ffi_loader_resolve_retained(symbol, library) : NULL;
    if (!function) {
        snprintf(error_msg, error_msg_size, "I could not resolve retained adapter %s in its selected module", symbol);
        goto cleanup;
    }
    ffi_cif cif;
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, (unsigned)arg_count, return_type, types) != FFI_OK) {
        snprintf(error_msg, error_msg_size, "I could not prepare the retained adapter's native signature");
        goto cleanup;
    }
    call.cif = &cif;
    call.function = function;
    call.arguments = values;
    if (policy->execution == NVM_FOREIGN_WORKER_THREAD) {
        if (!vm->callbacks) vm->callbacks = nano_callback_runtime_create();
        if (!vm->callbacks || pthread_mutex_init(&call.mutex, NULL)) {
            snprintf(error_msg, error_msg_size, "I could not prepare the native-call worker");
            goto cleanup;
        }
        call.runtime = vm->callbacks;
        pthread_t thread;
        if (pthread_create(&thread, NULL, callback_native_worker, &call)) {
            pthread_mutex_destroy(&call.mutex);
            snprintf(error_msg, error_msg_size, "I could not start the native-call worker");
            goto cleanup;
        }
        for (;;) {
            pthread_mutex_lock(&call.mutex);
            bool done = call.done;
            pthread_mutex_unlock(&call.mutex);
            if (done) break;
            vm_callback_pump(vm, true);
        }
        pthread_join(thread, NULL);
        pthread_mutex_destroy(&call.mutex);
    } else callback_native_invoke(&call);
    if (call.string_copy_failed) {
        snprintf(error_msg, error_msg_size, "I could not copy the native string result");
        goto cleanup;
    }
    if (vm->callback_error != VM_OK) {
        snprintf(error_msg, error_msg_size, "I stopped after a callback failed: %s", vm->callback_error_msg);
        goto cleanup;
    }
    switch (import->return_type) {
    case TAG_VOID: break;
    case TAG_INT: *result = val_int(call.returned.integer); break;
    case TAG_FLOAT: *result = val_float(call.returned.number); break;
    case TAG_BOOL: *result = val_bool(call.returned.word != 0); break;
    case TAG_U8: *result = val_u8((uint8_t)call.returned.word); break;
    case TAG_OPAQUE: result->tag = TAG_OPAQUE; result->as.obj = call.returned.pointer; break;
    case TAG_STRING:
        if (call.returned_string) {
            VmString *string = vm_string_new(&vm->heap, call.returned_string, call.returned_length);
            if (!string) {
                snprintf(error_msg, error_msg_size, "I could not allocate the VM string result");
                goto cleanup;
            }
            *result = val_string(string);
        }
        break;
    }
    ok = true;
cleanup:
    free(call.returned_string);
    for (int p = 0; p < arg_count; p++) {
        free(strings[p]);
        if (handles[p]) handles[p]->release(handles[p]);
    }
    if (vm->callbacks) nano_callback_collect(vm->callbacks);
    return ok;
}

static bool finish_foreign_arrays(VmFfiArrayFrame *arrays, int64_t raw, uint8_t tag,
                                  NanoValue *result, char *error, size_t size) {
    bool ok = true;
    NanoValue converted = val_void();
    if (!(tag == TAG_ARRAY &&
          vm_ffi_array_alias_result(arrays, (void *)(intptr_t)raw, &converted))) {
        converted = marshal_result(raw, tag, arrays->heap, &ok);
        if (!ok) snprintf(error, size, "I could not validate or allocate the foreign result");
    }
    if (ok) ok = vm_ffi_arrays_commit(arrays, error, size);
    if (ok) *result = converted;
    else vm_release(arrays->heap, converted);
    vm_ffi_arrays_dispose(arrays);
    return ok;
}

bool vm_ffi_call(const NvmModule *module, uint32_t import_idx,
                 NanoValue *args, int arg_count,
                 NanoValue *result, VmHeap *heap,
                 char *error_msg, size_t error_msg_size) {
    if (nvm_service_execution_pending(module)) return false;
    if (callback_contract_pending(module, import_idx, error_msg, error_msg_size)) return false;
    if (!ffi_loader_is_initialized()) vm_ffi_init();

    if (import_idx >= module->import_count) {
        snprintf(error_msg, error_msg_size, "Import index %u out of range", import_idx);
        return false;
    }

    const NvmImportEntry *imp = &module->imports[import_idx];
    const char *func_name = nvm_get_string(module, imp->function_name_idx);

    /* I own only these exact builtin results; selected artifacts retain their
     * existing symbol identity and result ownership contract. */
    const char *namespace = nvm_get_string(module, imp->module_name_idx);
    if (imp->kind == NVM_IMPORT_FFI && namespace && !namespace[0] && func_name &&
        (!strcmp(func_name, "path_normalize") || !strcmp(func_name, "nl_os_path_normalize"))) {
        const uint8_t *types = module->import_param_types
            ? module->import_param_types[import_idx] : NULL;
        if (imp->param_count != 1 || imp->return_type != TAG_STRING || !types ||
            types[0] != TAG_STRING || arg_count != 1 || !args ||
            args[0].tag != TAG_STRING || !args[0].as.string) {
            snprintf(error_msg, error_msg_size, "I require path normalization to take and return a string");
            return false;
        }
        char *normalized = nl_normalize_path(vmstring_cstr(args[0].as.string));
        if (!normalized) {
            snprintf(error_msg, error_msg_size, "I could not allocate my normalized path");
            return false;
        }
        size_t length = strlen(normalized);
        VmString *value = length <= UINT32_MAX
            ? vm_string_new(heap, normalized, (uint32_t)length) : NULL;
        free(normalized);
        if (!value) {
            snprintf(error_msg, error_msg_size, "I could not retain my normalized path");
            return false;
        }
        *result = val_string(value);
        return true;
    }

    /* Module introspection functions (___module_*) are environment- and
     * argument-dependent, so they are dispatched directly and never cached. */
    if (imp->kind == NVM_IMPORT_FFI &&
        vm_ffi_try_module_introspection(func_name, args, arg_count, result, heap)) {
        return true;
    }

    /* Resolve the import once into a typed call descriptor; every later call
     * for this import reuses the cached symbol and signature. */
    const NvmCallDescriptor *desc =
        vm_ffi_resolve_descriptor(module, import_idx, error_msg, error_msg_size);
    if (!desc) {
        return false;
    }

    void *func_ptr = desc->func_ptr;
    const uint8_t *param_types = desc->param_types;

    /* Marshal arguments */
    void *arg_ptrs[NANO_MAX_FFI_ARGS] = {0};
    VmFfiArrayFrame arrays = {.heap = heap};
    if (arg_count < 0 || arg_count > NANO_MAX_FFI_ARGS) {
        snprintf(error_msg, error_msg_size,
                 "Too many FFI arguments (%d > %d)", arg_count, NANO_MAX_FFI_ARGS);
        return false;
    }
    if (arg_count != desc->param_count || (arg_count && !args)) {
        snprintf(error_msg, error_msg_size, "I require the declared foreign argument count and values");
        return false;
    }

    if (desc->string_release) {
        if (arg_count && !param_types) {
            snprintf(error_msg, error_msg_size, "I require declared parameters for provider string cleanup");
            return false;
        }
        for (int i = 0; i < arg_count; ++i) {
            if (param_types[i] == TAG_STRING &&
                (args[i].tag != TAG_STRING || !args[i].as.string)) {
                snprintf(error_msg, error_msg_size, "I require string values for provider string cleanup");
                return false;
            }
        }
    }

    /* A bytecode function index is not an executable C address. I reject it
     * until a callback bridge owns its ABI, lifetime, and execution context. */
    for (int i = 0; i < arg_count; i++) {
        if (args[i].tag == TAG_FUNCTION || args[i].tag == TAG_CLOSURE ||
            (param_types && i < desc->param_count &&
             (param_types[i] == TAG_FUNCTION || param_types[i] == TAG_CLOSURE))) {
            snprintf(error_msg, error_msg_size,
                     "I cannot pass a bytecode function as a native callback (%s argument %d)",
                     func_name, i + 1);
            return false;
        }
    }

    /* I use the declared ABI for every arity. Register-class compatibility
     * does not make mismatched C function pointer types interchangeable. */
    {
        ffi_type *types[NANO_MAX_FFI_ARGS];
        void *values[NANO_MAX_FFI_ARGS];
        union { int64_t integer; double floating; void *pointer; uint8_t byte; }
            storage[NANO_MAX_FFI_ARGS], returned = {0};
        if (!marshal_args(args, arg_count, imp, param_types, arg_ptrs, &arrays, error_msg, error_msg_size))
            goto ffi_array_failure;
        for (int i = 0; i < arg_count; i++) {
            uint8_t tag = param_types ? param_types[i] : args[i].tag;
            values[i] = &storage[i];
            if (tag == TAG_FLOAT) {
                types[i] = &ffi_type_double;
                storage[i].floating = args[i].tag == TAG_FLOAT ? args[i].as.f64 : (double)args[i].as.i64;
            } else if (tag == TAG_INT || tag == TAG_ENUM) {
                types[i] = &ffi_type_sint64;
                storage[i].integer = (int64_t)(intptr_t)arg_ptrs[i];
            } else if (tag == TAG_BOOL || tag == TAG_U8) {
                types[i] = &ffi_type_uint8;
                storage[i].byte = (uint8_t)(uintptr_t)arg_ptrs[i];
            } else if (tag == TAG_STRING || tag == TAG_BSTRING || tag == TAG_ARRAY || tag == TAG_OPAQUE) {
                types[i] = &ffi_type_pointer;
                storage[i].pointer = arg_ptrs[i];
            } else {
                snprintf(error_msg, error_msg_size, "I cannot marshal typed foreign argument tag %u", tag);
                goto ffi_array_failure;
            }
        }
        ffi_type *return_type = NULL;
        switch (imp->return_type) {
            case TAG_VOID: return_type = &ffi_type_void; break;
            case TAG_FLOAT: return_type = &ffi_type_double; break;
            case TAG_INT: case TAG_ENUM: return_type = &ffi_type_sint64; break;
            case TAG_BOOL: case TAG_U8: return_type = &ffi_type_uint8; break;
            case TAG_STRING: case TAG_BSTRING: case TAG_ARRAY: case TAG_OPAQUE:
                return_type = &ffi_type_pointer; break;
            default:
                snprintf(error_msg, error_msg_size, "I cannot marshal typed foreign result tag %u", imp->return_type);
                goto ffi_array_failure;
        }
        ffi_cif cif;
        if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, (unsigned)arg_count, return_type, types) != FFI_OK) {
            snprintf(error_msg, error_msg_size, "I could not prepare a typed foreign signature");
            goto ffi_array_failure;
        }
        ffi_call(&cif, FFI_FN(func_ptr), &returned, values);
        int64_t raw = returned.integer;
        if (imp->return_type == TAG_FLOAT) memcpy(&raw, &returned.floating, sizeof raw);
        else if (imp->return_type == TAG_BOOL || imp->return_type == TAG_U8) raw = returned.byte;
        else if (imp->return_type == TAG_ARRAY || imp->return_type == TAG_STRING ||
                 imp->return_type == TAG_BSTRING || imp->return_type == TAG_OPAQUE)
            raw = (int64_t)(intptr_t)returned.pointer;
        if (desc->string_release) {
            bool copied = false;
            NanoValue snapshot = marshal_result(raw, TAG_STRING, heap, &copied);
            bool has_text = returned.pointer != NULL;
            desc->string_release(returned.pointer);
            if (!copied || !has_text) {
                snprintf(error_msg, error_msg_size, "I could not retain the provider string result");
                vm_ffi_arrays_dispose(&arrays);
                return false;
            }
            bool committed = vm_ffi_arrays_commit(&arrays, error_msg, error_msg_size);
            if (committed) *result = snapshot;
            else vm_release(heap, snapshot);
            vm_ffi_arrays_dispose(&arrays);
            return committed;
        }
        return finish_foreign_arrays(&arrays, raw, imp->return_type, result, error_msg, error_msg_size);
    }

ffi_array_failure:
    vm_ffi_arrays_dispose(&arrays);
    return false;
}

/* ========================================================================
 * Co-Process FFI Isolation
 * ======================================================================== */

#include "cop_protocol.h"
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <poll.h>

/* MAP_ANON is BSD; MAP_ANONYMOUS is the POSIX/Linux name */
#ifndef MAP_ANON
#  ifdef MAP_ANONYMOUS
#    define MAP_ANON MAP_ANONYMOUS
#  else
#    define MAP_ANON 0x1000  /* macOS value */
#  endif
#endif

/* ========================================================================
 * Co-process lifecycle (shared-memory mailbox fast path)
 * ======================================================================== */

bool vm_ffi_cop_start(VmState *vm, const NvmModule *module) {
    if (nvm_service_execution_pending(module)) return false;
    if (vm->cop_pid > 0) return true;

    /* Create the shared-memory mailbox — inherited across fork() */
    size_t mbox_size = sizeof(CopMailbox);
    CopMailbox *mailbox = (CopMailbox *)mmap(NULL, mbox_size,
                                             PROT_READ | PROT_WRITE,
                                             MAP_SHARED | MAP_ANON, -1, 0);
    if (mailbox == MAP_FAILED) return false;
    memset(mailbox, 0, mbox_size);

    /* Signal pipes serve small mailbox calls; data pipes carry large calls
     * to the same worker and therefore the same native module state. */
    int sig_to_child[2] = {-1, -1}, sig_from_child[2] = {-1, -1};
    int data_to_child[2] = {-1, -1}, data_from_child[2] = {-1, -1};
    if (pipe(sig_to_child) != 0 || pipe(sig_from_child) != 0 ||
        pipe(data_to_child) != 0 || pipe(data_from_child) != 0) {
        for (int i = 0; i < 2; ++i) {
            if (sig_to_child[i] >= 0) close(sig_to_child[i]);
            if (sig_from_child[i] >= 0) close(sig_from_child[i]);
            if (data_to_child[i] >= 0) close(data_to_child[i]);
            if (data_from_child[i] >= 0) close(data_from_child[i]);
        }
        munmap(mailbox, mbox_size);
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        munmap(mailbox, mbox_size);
        close(sig_to_child[0]);  close(sig_to_child[1]);
        close(sig_from_child[0]); close(sig_from_child[1]);
        close(data_to_child[0]); close(data_to_child[1]);
        close(data_from_child[0]); close(data_from_child[1]);
        return false;
    }

    if (pid == 0) {
        /* Child: close parent-side pipe ends and run the cop logic inline
         * (no exec — mailbox pointer is valid because we forked, not exec'd) */
        close(sig_to_child[1]);
        close(sig_from_child[0]);
        close(data_to_child[1]);
        close(data_from_child[0]);
        cop_child_main(mailbox, mbox_size,
                       sig_to_child[0], sig_from_child[1],
                       data_to_child[0], data_from_child[1], module);
        _exit(0);  /* cop_child_main never returns normally */
    }

    /* Parent: close child-side pipe ends */
    close(sig_to_child[0]);
    close(sig_from_child[1]);
    close(data_to_child[0]);
    close(data_from_child[1]);

    vm->cop_pid = pid;
    vm->cop_mailbox = mailbox;
    vm->cop_mailbox_size = mbox_size;
    vm->cop_sig_send_fd = sig_to_child[1];
    vm->cop_sig_recv_fd = sig_from_child[0];
    vm->cop_in_fd = data_to_child[1];
    vm->cop_out_fd = data_from_child[0];

    /* Wait for ready signal from child (up to 5 s) */
    struct pollfd pfd = { .fd = vm->cop_sig_recv_fd, .events = POLLIN };
    if (poll(&pfd, 1, 5000) <= 0) {
        vm_ffi_cop_stop(vm);
        return false;
    }
    uint8_t byte;
    if (read(vm->cop_sig_recv_fd, &byte, 1) != 1) {
        vm_ffi_cop_stop(vm);
        return false;
    }

    return true;
}

void vm_ffi_cop_stop(VmState *vm) {
    if (vm->cop_pid <= 0) return;

    /* Close the send pipe — child sees EOF and exits cleanly */
    if (vm->cop_sig_send_fd >= 0) {
        close(vm->cop_sig_send_fd);
        vm->cop_sig_send_fd = -1;
    }
    if (vm->cop_sig_recv_fd >= 0) {
        close(vm->cop_sig_recv_fd);
        vm->cop_sig_recv_fd = -1;
    }
    /* Legacy pipe fds (may be -1 in mailbox mode) */
    if (vm->cop_in_fd >= 0) {
        /* Closing cannot block on a full request pipe after a timed-out send. */
        close(vm->cop_in_fd);
        vm->cop_in_fd = -1;
    }
    if (vm->cop_out_fd >= 0) {
        close(vm->cop_out_fd);
        vm->cop_out_fd = -1;
    }

    /* Wait up to 50 ms for EOF-driven exit, then kill the owned worker.
     * Foreign code can ignore SIGTERM; it must not defeat the call deadline. */
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w == 0) {
        usleep(50000);
        w = waitpid(vm->cop_pid, &status, WNOHANG);
        if (w == 0) {
            kill(vm->cop_pid, SIGKILL);
            waitpid(vm->cop_pid, &status, 0);
        }
    }
    vm->cop_pid = -1;

    if (vm->cop_mailbox) {
        munmap(vm->cop_mailbox, vm->cop_mailbox_size);
        vm->cop_mailbox = NULL;
        vm->cop_mailbox_size = 0;
    }
}

/* Check if the co-process is still alive, reaping zombie if dead. */
static bool cop_is_alive(VmState *vm) {
    if (vm->cop_pid <= 0) return false;
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w > 0) {
        vm->cop_pid = -1;
        if (vm->cop_sig_send_fd >= 0) { close(vm->cop_sig_send_fd); vm->cop_sig_send_fd = -1; }
        if (vm->cop_sig_recv_fd >= 0) { close(vm->cop_sig_recv_fd); vm->cop_sig_recv_fd = -1; }
        if (vm->cop_in_fd  >= 0) { close(vm->cop_in_fd);  vm->cop_in_fd  = -1; }
        if (vm->cop_out_fd >= 0) { close(vm->cop_out_fd); vm->cop_out_fd = -1; }
        if (vm->cop_mailbox) {
            munmap(vm->cop_mailbox, vm->cop_mailbox_size);
            vm->cop_mailbox = NULL;
            vm->cop_mailbox_size = 0;
        }
        return false;
    }
    return true;
}

static bool cop_ensure(VmState *vm, const NvmModule *module,
                       char *error_msg, size_t error_msg_size) {
    if (cop_is_alive(vm)) return true;
    if (!vm_ffi_cop_start(vm, module)) {
        snprintf(error_msg, error_msg_size, "COP: failed to launch co-process");
        return false;
    }
    return true;
}

/* ========================================================================
 * vm_ffi_call_cop — fast mailbox path + pipe fallback
 * ======================================================================== */

bool vm_ffi_call_cop(VmState *vm, const NvmModule *module, uint32_t import_idx,
                     NanoValue *args, int arg_count,
                     NanoValue *result, VmHeap *heap,
                     char *error_msg, size_t error_msg_size) {
    if (nvm_service_execution_pending(module)) return false;
    if (arg_count < 0 || arg_count > NANO_MAX_FFI_ARGS || (!args && arg_count) || !result) {
        snprintf(error_msg, error_msg_size, "I require a valid isolated argument count and result");
        return false;
    }
    if (callback_contract_pending(module, import_idx, error_msg, error_msg_size)) return false;
    if (!cop_ensure(vm, module, error_msg, error_msg_size)) {
        /* Isolation was explicitly requested (this function only runs under
         * vm->isolate_ffi). If the co-process can't be launched we must NOT
         * silently run the dangerous call in-process — that defeats the whole
         * boundary and is attacker-forceable (e.g. exhaust process slots).
         * Fail closed instead. */
        snprintf(error_msg, error_msg_size,
                 "COP: FFI isolation requested but co-process unavailable; "
                 "refusing to run extern call in-process");
        return false;
    }

    CopMailbox *mbox = vm->cop_mailbox;

    /* ── Fast path: mailbox ──────────────────────────────────────────── */
    if (mbox) {
        /* Serialize args directly into the mailbox request slot */
        uint32_t pos = cop_encode_call_values(args, (uint8_t)arg_count,
                                               mbox->req_data, COP_MAILBOX_SLOT_SIZE);
        if (pos) {
            int64_t started = cop_now_ms();
            if (started < 0 || vm->cop_timeout_ms <= 0) {
                snprintf(error_msg, error_msg_size, "I require a positive isolated call deadline");
                return false;
            }
            int64_t deadline = started + vm->cop_timeout_ms;
            cop_put_u32(mbox->req_batch_count, 0);
            cop_put_u32(mbox->req_import_idx, import_idx);
            cop_put_u16(mbox->req_argc, (uint16_t)arg_count);
            cop_put_u16(mbox->req_data_size, (uint16_t)pos);

            /* Wake the child — pipe write is a full memory barrier on POSIX */
            uint8_t sig = 1;
            if (write(vm->cop_sig_send_fd, &sig, 1) != 1) {
                vm_ffi_cop_stop(vm);
                snprintf(error_msg, error_msg_size,
                         "COP: signal pipe broken (child crash?)");
                return false;
            }

            /* Wait for response with per-call timeout */
            struct pollfd pfd = { .fd = vm->cop_sig_recv_fd, .events = POLLIN };
            int n = poll(&pfd, 1, vm->cop_timeout_ms);
            if (n == 0) {
                /* Timeout: kill cop, return error */
                vm_ffi_cop_stop(vm);
                snprintf(error_msg, error_msg_size,
                         "COP: timeout after %d ms", vm->cop_timeout_ms);
                return false;
            }
            if (n < 0 || read(vm->cop_sig_recv_fd, &sig, 1) != 1) {
                vm_ffi_cop_stop(vm);
                snprintf(error_msg, error_msg_size, "COP: ack pipe broken");
                return false;
            }

            /* Read result from mailbox */
            if (mbox->resp_is_error == 2) {
                int64_t now = cop_now_ms();
                int64_t remaining = now < 0 ? 0 : deadline - now;
                CopMsgType type;
                uint8_t *reply = NULL;
                uint32_t reply_size = 0;
                bool ok = remaining > 0 &&
                    cop_exchange(-1, vm->cop_out_fd, NULL, 0, (int)remaining,
                                  &type, &reply, &reply_size) &&
                    type == COP_MSG_FFI_RESULT &&
                    cop_apply_call_reply(reply, reply_size, args, (uint8_t)arg_count, result, heap);
                free(reply);
                if (!ok) {
                    vm_ffi_cop_stop(vm);
                    snprintf(error_msg, error_msg_size, "I could not receive the isolated spill reply before its deadline");
                }
                return ok;
            }
            if (mbox->resp_is_error) {
                snprintf(error_msg, error_msg_size, "%.*s",
                         (int)sizeof(mbox->resp_error), mbox->resp_error);
                return false;
            }
            uint32_t resp_wire_size = cop_get_u32(mbox->resp_data_size);
            if (resp_wire_size > COP_MAILBOX_SLOT_SIZE ||
                !cop_apply_call_reply(mbox->resp_data, resp_wire_size, args,
                                      (uint8_t)arg_count, result, heap)) {
                snprintf(error_msg, error_msg_size, "I rejected an invalid isolated reply");
                return false;
            }
            return true;
        }
        /* Args too large for mailbox slot — fall through to pipe path */
    }

    /* ── Pipe fallback: for large payloads or when mailbox unavailable ─ */
    if (vm->cop_in_fd < 0) {
        /* Mailbox-only mode has no legacy pipes and the args didn't fit the
         * mailbox slot. There is no isolated channel for this call, so fail
         * closed rather than silently running the dangerous op in-process
         * (an attacker who controls argument size could otherwise force the
         * bypass by exceeding COP_MAILBOX_SLOT_SIZE). */
        snprintf(error_msg, error_msg_size,
                 "COP: extern-call payload exceeds mailbox slot and no isolated "
                 "pipe channel is available; refusing in-process fallback");
        return false;
    }

    uint8_t payload[8192];
    uint8_t *request = payload;
    uint32_t capacity = sizeof payload;
    uint32_t encoded;
    while (!(encoded = cop_encode_call_values(args, (uint8_t)arg_count,
                                               request + 6, capacity - 6))) {
        if (request != payload) free(request);
        if (capacity >= COP_MAX_PAYLOAD) {
            snprintf(error_msg, error_msg_size, "I could not encode a bounded pipe request");
            return false;
        }
        capacity *= 2;
        request = malloc(capacity);
        if (!request) {
            snprintf(error_msg, error_msg_size, "I could not allocate the pipe request");
            return false;
        }
    }
    cop_put_u32(request, import_idx);
    cop_put_u16(request + 4, (uint16_t)arg_count);
    CopMsgType response_type;
    uint8_t *reply = NULL;
    uint32_t reply_size = 0;
    bool exchanged = cop_exchange(vm->cop_in_fd, vm->cop_out_fd, request, encoded + 6,
                                    vm->cop_timeout_ms, &response_type, &reply, &reply_size);
    if (request != payload) free(request);
    if (!exchanged) {
        vm_ffi_cop_stop(vm);
        snprintf(error_msg, error_msg_size, "I could not complete the isolated pipe exchange within %d ms",
                 vm->cop_timeout_ms);
        return false;
    }
    bool ok = false;
    if (response_type == COP_MSG_FFI_RESULT) {
        ok = cop_apply_call_reply(reply, reply_size, args, (uint8_t)arg_count, result, heap);
        if (!ok) snprintf(error_msg, error_msg_size, "I rejected an invalid isolated pipe reply");
    } else {
        snprintf(error_msg, error_msg_size, "%.*s", (int)reply_size, (const char *)reply);
    }
    free(reply);
    if (!ok) vm_ffi_cop_stop(vm);
    return ok;
}

/* ========================================================================
 * vm_ffi_call_cop_batch — coalesce many host calls into one crossing
 *
 * High-frequency host work (e.g. per-pixel or per-element transforms) would
 * otherwise pay a signal+ack round-trip through the co-process boundary for
 * every element.  This packs as many calls as fit in the mailbox request
 * slot into a single COP batch, dispatched with one signal and collected
 * with one ack, then repeats until the whole batch is drained.
 *
 * Results are written into results[0..count-1] (caller-owned array).  On the
 * first failing call, returns false with error_msg set; results before the
 * failure are valid and owned by the caller, later slots are set to void.
 * ======================================================================== */
bool vm_ffi_call_cop_batch(VmState *vm, const NvmModule *module,
                           const CopBatchCall *calls, int count,
                           NanoValue *results, VmHeap *heap,
                           char *error_msg, size_t error_msg_size) {
    if (nvm_service_execution_pending(module)) return false;
    if (count < 0) {
        snprintf(error_msg, error_msg_size, "COP: negative batch count");
        return false;
    }
    for (int i = 0; i < count; i++) results[i] = val_void();
    if (count == 0) return true;

    for (int i = 0; i < count; i++)
        if (callback_contract_pending(module, calls[i].import_idx, error_msg, error_msg_size)) return false;

    if (!cop_ensure(vm, module, error_msg, error_msg_size)) {
        snprintf(error_msg, error_msg_size,
                 "COP: FFI isolation requested but co-process unavailable; "
                 "refusing to run extern batch in-process");
        return false;
    }

    CopMailbox *mbox = vm->cop_mailbox;
    bool array_arguments = false;
    for (int i = 0; i < count; ++i) {
        if (calls[i].import_idx < module->import_count) {
            uint8_t returned = module->imports[calls[i].import_idx].return_type;
            if (returned == TAG_ARRAY || returned == TAG_STRING || returned == TAG_BSTRING)
                array_arguments = true;
        }
        if (calls[i].arg_count < 0 || calls[i].arg_count > NANO_MAX_FFI_ARGS ||
            (!calls[i].args && calls[i].arg_count)) {
            snprintf(error_msg, error_msg_size, "I require valid isolated batch arguments");
            return false;
        }
        for (int j = 0; j < calls[i].arg_count; ++j)
            if (calls[i].args[j].tag == TAG_ARRAY) array_arguments = true;
    }
    if (!mbox || array_arguments) {
        /* Variable-size replies need single-call spillover. Array calls must
         * observe the previous call's published mutations.
         * I retain scalar batching; packing array snapshots ahead of execution
         * would silently erase dependencies between calls sharing an array. */
        /* No shared-memory mailbox: fall back to per-call dispatch so batching
         * still works functionally over the pipe channel (just without the
         * single-crossing win). */
        for (int i = 0; i < count; i++) {
            if (!vm_ffi_call_cop(vm, module, calls[i].import_idx,
                                 (NanoValue *)calls[i].args, calls[i].arg_count,
                                 &results[i], heap, error_msg, error_msg_size)) {
                for (int j = i; j < count; j++) results[j] = val_void();
                return false;
            }
        }
        return true;
    }

    int next = 0;              /* index of next call to pack */
    while (next < count) {
        /* Pack as many calls as fit into the request slot for one crossing. */
        uint32_t wpos = 0;
        int batch_start = next;
        int batched = 0;
        while (next < count && batched < COP_MAX_BATCH) {
            const CopBatchCall *call = &calls[next];
            int argc = call->arg_count;
            if (argc > 16) argc = 16;

            /* Header (8 bytes) + serialized args must fit remaining slot. */
            if (wpos + 8 > COP_MAILBOX_SLOT_SIZE) break;
            uint32_t hdr = wpos;
            uint32_t apos = wpos + 8;
            bool fits = true;
            for (int a = 0; a < argc; a++) {
                uint32_t n = cop_serialize_value(&call->args[a],
                                                 mbox->req_data + apos,
                                                 COP_MAILBOX_SLOT_SIZE - apos);
                if (n == 0) { fits = false; break; }
                apos += n;
            }
            if (!fits) {
                if (batched == 0) {
                    snprintf(error_msg, error_msg_size,
                             "COP: batch call %d args exceed mailbox slot", next);
                    return false;
                }
                break;  /* flush what we have, retry this call next crossing */
            }
            uint32_t arg_bytes = apos - (hdr + 8);
            cop_put_u32(mbox->req_data + hdr,     call->import_idx);
            cop_put_u16(mbox->req_data + hdr + 4, (uint16_t)argc);
            cop_put_u16(mbox->req_data + hdr + 6, (uint16_t)arg_bytes);
            wpos = apos;
            next++;
            batched++;
        }

        cop_put_u32(mbox->req_batch_count, (uint32_t)batched);
        cop_put_u16(mbox->req_data_size, (uint16_t)wpos);
        cop_put_u16(mbox->req_argc, 0);
        cop_put_u32(mbox->req_import_idx, 0);

        uint8_t sig = 1;
        if (write(vm->cop_sig_send_fd, &sig, 1) != 1) {
            vm_ffi_cop_stop(vm);
            snprintf(error_msg, error_msg_size, "COP: signal pipe broken (child crash?)");
            return false;
        }

        struct pollfd pfd = { .fd = vm->cop_sig_recv_fd, .events = POLLIN };
        int pn = poll(&pfd, 1, vm->cop_timeout_ms);
        if (pn == 0) {
            vm_ffi_cop_stop(vm);
            snprintf(error_msg, error_msg_size, "COP: timeout after %d ms", vm->cop_timeout_ms);
            return false;
        }
        if (pn < 0 || read(vm->cop_sig_recv_fd, &sig, 1) != 1) {
            vm_ffi_cop_stop(vm);
            snprintf(error_msg, error_msg_size, "COP: ack pipe broken");
            return false;
        }

        uint32_t produced = cop_get_u32(mbox->resp_batch_count);
        uint32_t resp_wire = cop_get_u32(mbox->resp_data_size);
        if (resp_wire > COP_MAILBOX_SLOT_SIZE) resp_wire = COP_MAILBOX_SLOT_SIZE;

        /* Unpack the results that did complete, whether or not the batch errored. */
        uint32_t rpos = 0;
        uint32_t unpacked = 0;
        for (uint32_t k = 0; k < produced && k < (uint32_t)batched; k++) {
            NanoValue out;
            uint32_t consumed = cop_deserialize_value(mbox->resp_data + rpos,
                                                      resp_wire - rpos, &out, heap);
            if (consumed == 0) break;
            results[batch_start + (int)k] = out;
            rpos += consumed;
            unpacked++;
        }

        if (mbox->resp_is_error) {
            snprintf(error_msg, error_msg_size, "%s",
                     mbox->resp_error[0] ? mbox->resp_error : "COP: batch call failed");
            return false;
        }
        if (unpacked != (uint32_t)batched) {
            snprintf(error_msg, error_msg_size,
                     "COP: batch produced %u/%d results", unpacked, batched);
            return false;
        }
    }

    return true;
}
