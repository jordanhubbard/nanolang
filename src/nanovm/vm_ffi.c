#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
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
#include "ffi_dispatch_generated.h"
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
    if (nvm_capture_bindings_present(module) || nvm_service_execution_pending(module)) return false;
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
                /* Reached only for signatures the generated mixed dispatcher
                 * does not cover (e.g. a float alongside a string/array arg,
                 * or arity above FFI_DISPATCH_MAX_ARITY). Correctly-classified
                 * float/int mixes are handled earlier by ffi_call_mixed(); this
                 * fallback keeps the bit pattern for such rare aggregate mixes. */
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
                                VmHeap *heap, bool *success, const VmFfiOpaqueCapture *capture) {
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
            NanoValue v = val_opaque((void *)(intptr_t)raw_result);
            if (capture) {
                uint64_t slot;
                if (!capture->record(capture->context, v.as.obj, &slot)) {
                    *success = false;
                    return val_void();
                }
                v.as.i64 = (int64_t)slot;
            }
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

typedef int64_t (*FFI_Fn0)(void);
typedef int64_t (*FFI_Fn1)(void *);
typedef int64_t (*FFI_Fn2)(void *, void *);
typedef int64_t (*FFI_Fn3)(void *, void *, void *);
typedef int64_t (*FFI_Fn4)(void *, void *, void *, void *);
typedef int64_t (*FFI_Fn5)(void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn6)(void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn7)(void *, void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn8)(void *, void *, void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn9)(void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn10)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

/* Float-specific dispatch: on arm64, doubles use FP registers, not GP.
 * Using void-ptr and int64_t types puts args in wrong registers for C math fns. */
typedef double (*FFI_DFn0)(void);
typedef double (*FFI_DFn1)(double);
typedef double (*FFI_DFn2)(double, double);
typedef double (*FFI_DFn3)(double, double, double);
typedef double (*FFI_DFn4)(double, double, double, double);
typedef double (*FFI_DFn5)(double, double, double, double, double);
typedef double (*FFI_DFn6)(double, double, double, double, double, double);
typedef double (*FFI_DFn7)(double, double, double, double, double, double, double);
typedef double (*FFI_DFn8)(double, double, double, double, double, double, double, double);
typedef double (*FFI_DFn9)(double, double, double, double, double, double, double, double, double);
typedef double (*FFI_DFn10)(double, double, double, double, double, double, double, double, double, double);

/* Check if all params and return are float */
static bool is_all_float_signature(const NvmImportEntry *imp,
                                   const uint8_t *param_types, int arg_count) {
    if (imp->return_type != TAG_FLOAT) return false;
    for (int i = 0; i < arg_count; i++) {
        uint8_t tag = (i < imp->param_count && param_types) ? param_types[i] : TAG_FLOAT;
        if (tag != TAG_FLOAT) return false;
    }
    return true;
}

/* ABI class of a single argument tag. Floating-point values travel in the
 * FP/SIMD register bank; everything else (int, bool, char*, pointers, array
 * handles) travels in general-purpose registers. Classifying correctly is the
 * whole point of the generated mixed-signature dispatcher. */
static bool ffi_tag_is_float(uint8_t tag) {
    return tag == TAG_FLOAT;
}

/* True when a signature mixes GP and FP argument classes (or has an FP
 * argument alongside a GP return, etc.). Homogeneous all-GP and all-FP
 * signatures are handled by the existing fast paths; this predicate selects
 * the generated typed-stub dispatcher for everything in between that the
 * generic void*-cast path would marshal incorrectly. */
static bool ffi_uses_generated_dispatch(const NvmImportEntry *imp,
                                        const uint8_t *param_types,
                                        int arg_count) {
    if (arg_count > FFI_DISPATCH_MAX_ARITY) return false;
    /* Only int64/pointer-class and double-class scalars can be routed through
     * the generated stubs; aggregate/string marshaling stays on the classic
     * path so its VmString/VmArray conversions still run. */
    bool has_float = ffi_tag_is_float(imp->return_type);
    for (int i = 0; i < arg_count; i++) {
        uint8_t tag = (i < imp->param_count && param_types) ? param_types[i]
                                                            : TAG_INT;
        switch (tag) {
            case TAG_INT:
            case TAG_BOOL:
            case TAG_OPAQUE:
                break;
            case TAG_FLOAT:
                has_float = true;
                break;
            default:
                /* string/array/struct/etc. — needs classic marshaling */
                return false;
        }
    }
    if (imp->return_type != TAG_INT && imp->return_type != TAG_BOOL &&
        imp->return_type != TAG_FLOAT && imp->return_type != TAG_VOID &&
        imp->return_type != TAG_OPAQUE) {
        return false;
    }
    /* Use the generated dispatcher whenever any float is involved. Pure
     * int/bool/opaque signatures also dispatch correctly here, but we let the
     * legacy void* path own them to minimise behavioural change. */
    return has_float;
}

/* Fill one classified argument slot from a NanoValue given its declared tag. */
static void ffi_fill_slot(FfiArgSlot *slot, const NanoValue *v, uint8_t tag) {
    if (ffi_tag_is_float(tag)) {
        slot->is_float = 1;
        slot->i = 0;
        slot->f = (v->tag == TAG_FLOAT) ? v->as.f64
                : (v->tag == TAG_INT)   ? (double)v->as.i64
                : 0.0;
    } else {
        slot->is_float = 0;
        slot->f = 0.0;
        switch (tag) {
            case TAG_BOOL:
                slot->i = (long)(v->as.boolean ? 1 : 0);
                break;
            default: /* TAG_INT, TAG_OPAQUE */
                slot->i = (long)(intptr_t)v->as.i64;
                break;
        }
    }
}

/* Dispatch a mixed int/float signature through the generated typed stubs so
 * each argument lands in the correct ABI register class. Returns true on a
 * completed call (result populated); false only if the (arity,pattern) is
 * outside the generated table. */
static bool ffi_call_mixed(void *func_ptr, NanoValue *args, int arg_count,
                           const NvmImportEntry *imp, const uint8_t *param_types,
                           NanoValue *result, VmHeap *heap,
                           const VmFfiOpaqueCapture *capture, bool *entered) {
    *entered = false;
    FfiArgSlot slots[FFI_DISPATCH_MAX_ARITY];
    for (int i = 0; i < arg_count; i++) {
        uint8_t tag = (i < imp->param_count && param_types) ? param_types[i]
                                                            : args[i].tag;
        ffi_fill_slot(&slots[i], &args[i], tag);
    }

    if (ffi_tag_is_float(imp->return_type)) {
        double dr = 0.0;
        if (!ffi_dispatch_fp(func_ptr, slots, arg_count, &dr)) return false;
        *entered = true;
        *result = val_float(dr);
        return true;
    }

    int64_t r = 0;
    if (!ffi_dispatch_gp(func_ptr, slots, arg_count, &r)) return false;
    *entered = true;
    bool converted;
    NanoValue value = marshal_result(r, imp->return_type, heap, &converted, capture);
    if (converted) *result = value;
    return converted;
}

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
        if (desc->string_release) {
            bool supported = desc->param_count <= 2;
            for (uint16_t i = 0; i < desc->param_count; ++i)
                supported = supported && desc->param_types && desc->param_types[i] == TAG_STRING;
            if (!supported) {
                desc->state = NVM_CALL_FAILED;
                snprintf(error_msg, error_msg_size, "I require up to two string parameters for provider string cleanup");
                return NULL;
            }
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

static bool local_opaque_arguments(NanoValue *args, int count,
                                   char *error, size_t size);

bool vm_ffi_call_vm(VmState *vm, const NvmModule *module, uint32_t import_idx,
                    NanoValue *args, int arg_count, NanoValue *result,
                    char *error_msg, size_t error_msg_size) {
    if (!local_opaque_arguments(args, arg_count, error_msg, error_msg_size)) return false;
    if (nvm_capture_bindings_present(module) || nvm_service_execution_pending(module)) return false;
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
    case TAG_OPAQUE: *result = val_opaque(call.returned.pointer); break;
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
                                  NanoValue *result, char *error, size_t size,
                                  const VmFfiOpaqueCapture *capture) {
    bool ok = true;
    NanoValue converted = val_void();
    if (!(tag == TAG_ARRAY &&
          vm_ffi_array_alias_result(arrays, (void *)(intptr_t)raw, &converted))) {
        converted = marshal_result(raw, tag, arrays->heap, &ok, capture);
        if (!ok) snprintf(error, size, "I could not validate or allocate the foreign result");
    }
    if (ok) ok = vm_ffi_arrays_commit(arrays, error, size);
    if (ok) *result = converted;
    else vm_release(arrays->heap, converted);
    vm_ffi_arrays_dispose(arrays);
    return ok;
}

static bool vm_ffi_call_impl(const NvmModule *module, uint32_t import_idx,
                 NanoValue *args, int arg_count, NanoValue *result, VmHeap *heap,
                 char *error_msg, size_t error_msg_size,
                 const VmFfiOpaqueCapture *capture) {
    if (nvm_capture_bindings_present(module) || nvm_service_execution_pending(module)) return false;
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

    /* I keep exact zero/one/two-string artifact results pointer-typed whether
     * the provider lends storage or supplies cleanup. Arity was checked above. */
    bool artifact_string_call = imp->kind == NVM_IMPORT_ARTIFACT &&
        imp->return_type == TAG_STRING && arg_count <= 2;
    for (int i = 0; artifact_string_call && i < arg_count; ++i)
        if (!param_types || param_types[i] != TAG_STRING) artifact_string_call = false;
    if (desc->string_release || artifact_string_call) {
        const char *arguments[2] = {NULL, NULL};
        for (int i = 0; i < arg_count; ++i) {
            if (args[i].tag != TAG_STRING || !args[i].as.string) {
                snprintf(error_msg, error_msg_size, "I require string values for provider string cleanup");
                return false;
            }
            arguments[i] = vmstring_cstr(args[i].as.string);
        }
        const char *text = arg_count == 0 ? ((const char *(*)(void))func_ptr)() :
            arg_count == 1 ? ((const char *(*)(const char *))func_ptr)(arguments[0]) :
            ((const char *(*)(const char *, const char *))func_ptr)(arguments[0], arguments[1]);
        bool copied = false;
        bool has_text = text != NULL;
        NanoValue snapshot = marshal_result((int64_t)(intptr_t)text, TAG_STRING, heap, &copied, capture);
        if (desc->string_release) desc->string_release(text);
        if (!copied || !has_text) {
            snprintf(error_msg, error_msg_size, "I could not retain the provider string result");
            return false;
        }
        *result = snapshot;
        return true;
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

    /* I use typed ABI dispatch for wider signatures, including mixed
     * integer/pointer and floating-point register classes. */
    bool typed_abi = imp->return_type == TAG_FLOAT || imp->return_type == TAG_ARRAY ||
                     imp->return_type == TAG_OPAQUE;
    for (int i = 0; param_types && i < arg_count && i < desc->param_count; i++)
        if (param_types[i] == TAG_FLOAT || param_types[i] == TAG_ARRAY ||
            param_types[i] == TAG_OPAQUE) typed_abi = true;
    if (arg_count > 10 || typed_abi) {
        if (arg_count != desc->param_count) {
            snprintf(error_msg, error_msg_size, "I require the declared foreign argument count");
            return false;
        }
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
        return finish_foreign_arrays(&arrays, raw, imp->return_type, result, error_msg, error_msg_size, capture);
    }

    /* Fast path: all-float signatures use properly typed dispatch so
     * doubles go through FP registers (critical on arm64). The classification
     * is precomputed in the descriptor for the declared arity; fall back to a
     * runtime check when the call arity differs from the declaration. */
    bool all_float = (arg_count == (int)desc->param_count)
                     ? desc->all_float
                     : is_all_float_signature(imp, param_types, arg_count);
    if (all_float) {
        double dargs[NANO_MAX_FFI_ARGS];
        for (int i = 0; i < arg_count; i++) {
            dargs[i] = (args[i].tag == TAG_FLOAT) ? args[i].as.f64
                      : (args[i].tag == TAG_INT)   ? (double)args[i].as.i64
                      : 0.0;
        }
        double dresult = 0.0;
        switch (arg_count) {
            case 0: dresult = ((FFI_DFn0)func_ptr)(); break;
            case 1: dresult = ((FFI_DFn1)func_ptr)(dargs[0]); break;
            case 2: dresult = ((FFI_DFn2)func_ptr)(dargs[0], dargs[1]); break;
            case 3: dresult = ((FFI_DFn3)func_ptr)(dargs[0], dargs[1], dargs[2]); break;
            case 4: dresult = ((FFI_DFn4)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3]); break;
            case 5: dresult = ((FFI_DFn5)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4]); break;
            case 6: dresult = ((FFI_DFn6)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5]); break;
            case 7: dresult = ((FFI_DFn7)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6]); break;
            case 8: dresult = ((FFI_DFn8)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[7]); break;
            case 9: dresult = ((FFI_DFn9)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[7], dargs[8]); break;
            case 10: dresult = ((FFI_DFn10)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[7], dargs[8], dargs[9]); break;
            default:
                snprintf(error_msg, error_msg_size,
                         "FFI: unsupported float arg count %d (max %d)",
                         arg_count, NANO_MAX_FFI_ARGS);
                return false;
        }
        *result = val_float(dresult);
        return true;
    }

    /* Mixed integer/floating signatures: route through the generated typed
     * stubs so int/pointer args use GP registers and float args use FP
     * registers per the platform ABI. The generic void*-cast path below would
     * otherwise place doubles in the wrong register class. */
    if (ffi_uses_generated_dispatch(imp, param_types, arg_count)) {
        bool entered;
        if (ffi_call_mixed(func_ptr, args, arg_count, imp, param_types,
                           result, heap, capture, &entered)) {
            return true;
        }
        if (entered) {
            snprintf(error_msg, error_msg_size, "I could not retain the foreign result after native entry");
            return false;
        }
        /* Fall through to the generic path if the pattern was unsupported. */
    }

    if (!marshal_args(args, arg_count, imp, param_types, arg_ptrs, &arrays, error_msg, error_msg_size))
        goto ffi_array_failure;

    /* Call the function */
    int64_t raw_result = 0;
    switch (arg_count) {
        case 0: raw_result = ((FFI_Fn0)func_ptr)(); break;
        case 1: raw_result = ((FFI_Fn1)func_ptr)(arg_ptrs[0]); break;
        case 2: raw_result = ((FFI_Fn2)func_ptr)(arg_ptrs[0], arg_ptrs[1]); break;
        case 3: raw_result = ((FFI_Fn3)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2]); break;
        case 4: raw_result = ((FFI_Fn4)func_ptr)(arg_ptrs[0], arg_ptrs[1],
                                                  arg_ptrs[2], arg_ptrs[3]); break;
        case 5: raw_result = ((FFI_Fn5)func_ptr)(arg_ptrs[0], arg_ptrs[1],
                                                  arg_ptrs[2], arg_ptrs[3], arg_ptrs[4]); break;
        case 6: raw_result = ((FFI_Fn6)func_ptr)(arg_ptrs[0], arg_ptrs[1],
                                                  arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5]); break;
        case 7: raw_result = ((FFI_Fn7)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                  arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6]); break;
        case 8: raw_result = ((FFI_Fn8)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                  arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7]); break;
        case 9: raw_result = ((FFI_Fn9)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                  arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8]); break;
        case 10: raw_result = ((FFI_Fn10)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                   arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9]); break;
        default:
            snprintf(error_msg, error_msg_size,
                     "FFI: unsupported arg count %d (max %d)",
                     arg_count, NANO_MAX_FFI_ARGS);
            return false;
    }

    return finish_foreign_arrays(&arrays, raw_result, imp->return_type, result, error_msg, error_msg_size, capture);

ffi_array_failure:
    vm_ffi_arrays_dispose(&arrays);
    return false;
}

static bool local_opaque_arguments(NanoValue *args, int count,
                                   char *error, size_t size) {
    if (count < 0 || count > NANO_MAX_FFI_ARGS || (!args && count)) {
        snprintf(error, size, "I require a valid foreign argument count and argument storage");
        return false;
    }
    for (int i = 0; i < count; ++i) {
        if (args[i].tag == TAG_OPAQUE && args[i].opaque_owner) {
            snprintf(error, size, "I cannot pass an isolated opaque token as a local native pointer");
            return false;
        }
    }
    return true;
}

bool vm_ffi_call(const NvmModule *module, uint32_t import_idx,
                 NanoValue *args, int arg_count, NanoValue *result, VmHeap *heap,
                 char *error_msg, size_t error_msg_size) {
    if (!local_opaque_arguments(args, arg_count, error_msg, error_msg_size)) return false;
    return vm_ffi_call_impl(module, import_idx, args, arg_count, result, heap,
                            error_msg, error_msg_size, NULL);
}

bool vm_ffi_call_captured(const NvmModule *module, uint32_t import_idx,
                          NanoValue *args, int arg_count, NanoValue *result,
                          VmHeap *heap, const VmFfiOpaqueCapture *capture,
                          char *error_msg, size_t error_msg_size) {
    if (!capture || !capture->record) {
        snprintf(error_msg, error_msg_size, "I require an opaque result capture before native entry");
        return false;
    }
    if (!local_opaque_arguments(args, arg_count, error_msg, error_msg_size)) return false;
    return vm_ffi_call_impl(module, import_idx, args, arg_count, result, heap,
                            error_msg, error_msg_size, capture);
}

/* ========================================================================
 * Co-Process FFI Isolation
 * ======================================================================== */

#include "cop_protocol.h"
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <poll.h>
#include <errno.h>

#include "runtime/module_build_dir.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <spawn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <limits.h>

extern char **environ;

/* I keep every owned spawn source outside the fixed child descriptor range. */
static int cop_spawn_fd(int fd) {
    if (fd < 0) return -1;
    int copy = fcntl(fd, F_DUPFD_CLOEXEC, 16);
    close(fd);
    return copy;
}

static int cop_private_file(void) {
    char path[] = "/tmp/nanolang-cop-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    if (unlink(path) != 0) { close(fd); return -1; }
    return cop_spawn_fd(fd);
}

static bool cop_spawn_pipe(int fds[2]) {
    int pair[2];
#ifdef __linux__
    if (pipe2(pair, O_CLOEXEC) != 0) return false;
#else
    if (pipe(pair) != 0) return false;
#endif
    fds[0] = cop_spawn_fd(pair[0]);
    fds[1] = cop_spawn_fd(pair[1]);
    return fds[0] >= 0 && fds[1] >= 0;
}

static bool cop_snapshot_write(int fd, const uint8_t *bytes, size_t size) {
    while (size) {
        ssize_t count = write(fd, bytes, size);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        bytes += count;
        size -= (size_t)count;
    }
    return lseek(fd, 0, SEEK_SET) == 0;
}

static bool cop_spawn_ready(int fd, int64_t deadline) {
    uint8_t bytes[COP_EXEC_READY_SIZE];
    size_t offset = 0;
    while (offset < sizeof bytes) {
        int64_t now = cop_now_ms();
        if (now < 0 || now >= deadline) return false;
        struct pollfd wait = {.fd = fd, .events = POLLIN};
        int ready = poll(&wait, 1, (int)(deadline - now));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) return false;
        ssize_t count = read(fd, bytes + offset, sizeof bytes - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        offset += (size_t)count;
    }
    return memcmp(bytes, COP_EXEC_READY, sizeof bytes) == 0;
}

bool vm_ffi_cop_start(VmState *vm, const NvmModule *module) {
    if (!vm || !module || nvm_capture_bindings_present(module) ||
        nvm_service_execution_pending(module)) return false;
    if (vm->cop_pid > 0) return !vm->cop_opaque.generation || cop_opaque_owner_live(&vm->cop_opaque);

    char root[4096], path[4096];
    bool installed;
    struct stat st;
    if (nano_native_sdk_root(root, sizeof root, &installed) != NANO_SDK_OK) return false;
    int length = snprintf(path, sizeof path, "%s/bin/nano_cop", root);
    if (length < 0 || (size_t)length >= sizeof path || stat(path, &st) != 0 ||
        !S_ISREG(st.st_mode) || access(path, X_OK) != 0) return false;

    /* I hold cancellation until I publish the worker or reclaim every resource. */
    int prior_cancel_state;
    if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &prior_cancel_state) != 0) return false;
    bool ok = false, actions_live = false, attr_live = false;
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    int fds[10];
    for (size_t i = 0; i < sizeof fds / sizeof fds[0]; ++i) fds[i] = -1;
    CopMailbox *mailbox = MAP_FAILED;
    pid_t pid = -1;
    uint8_t *snapshot = NULL;
    uint32_t snapshot_size = 0;
    NanoisaErr error;
    int64_t now = cop_now_ms();
    if (now < 0) goto cleanup;
    int64_t deadline = now + 5000;
    snapshot = nanoisa_save_bytes(module, &snapshot_size, &error);
    if (!snapshot || !snapshot_size || snapshot_size > COP_MAX_PAYLOAD) goto cleanup;
    fds[0] = cop_private_file();
    fds[1] = cop_private_file();
    if (fds[0] < 0 || fds[1] < 0 || ftruncate(fds[0], sizeof(CopMailbox)) != 0 ||
        !cop_snapshot_write(fds[1], snapshot, snapshot_size)) goto cleanup;
    free(snapshot); snapshot = NULL;
    mailbox = mmap(NULL, sizeof(CopMailbox), PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], 0);
    if (mailbox == MAP_FAILED) goto cleanup;
    memset(mailbox, 0, sizeof(CopMailbox));
    for (int i = 2; i < 10; i += 2) if (!cop_spawn_pipe(fds + i)) goto cleanup;
    cop_opaque_owner_clear(&vm->cop_opaque);
    if (!cop_opaque_owner_start(&vm->cop_opaque)) goto cleanup;

    if (posix_spawn_file_actions_init(&actions) != 0) goto cleanup;
    actions_live = true;
    /* mailbox, signal read/write, data read/write, module snapshot */
    const int sources[] = {fds[0], fds[2], fds[5], fds[6], fds[9], fds[1]};
    for (int i = 0; i < 6; ++i)
        if (posix_spawn_file_actions_adddup2(&actions, sources[i], i + 3) != 0) goto cleanup;
    if (posix_spawnattr_init(&attributes) != 0) goto cleanup;
    attr_live = true;
#ifdef __APPLE__
    if (posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT) != 0) goto cleanup;
    for (int i = 0; i < 3; ++i) {
        if (fcntl(i, F_GETFD) != -1 &&
            posix_spawn_file_actions_addinherit_np(&actions, i) != 0) goto cleanup;
    }
#else
    if (posix_spawn_file_actions_addclosefrom_np(&actions, 9) != 0) goto cleanup;
#endif
    char *argv[] = {path, "--mailbox-v1", NULL};
    if (posix_spawn(&pid, path, &actions, &attributes, argv, environ) != 0) { pid = -1; goto cleanup; }
    /* Only the parent's four protocol ends survive here. */
    const int close_indices[] = {0, 1, 2, 5, 6, 9};
    for (size_t i = 0; i < sizeof close_indices / sizeof close_indices[0]; ++i) {
        int at = close_indices[i]; close(fds[at]); fds[at] = -1;
    }
    if (!cop_spawn_ready(fds[4], deadline)) goto cleanup;
    vm->cop_pid = pid;
    vm->cop_mailbox = mailbox;
    vm->cop_mailbox_size = sizeof(CopMailbox);
    vm->cop_sig_send_fd = fds[3]; vm->cop_sig_recv_fd = fds[4];
    vm->cop_in_fd = fds[7]; vm->cop_out_fd = fds[8];
    fds[3] = fds[4] = fds[7] = fds[8] = -1;
    pid = -1; mailbox = MAP_FAILED;
    ok = true;
cleanup:
    free(snapshot);
    if (actions_live) posix_spawn_file_actions_destroy(&actions);
    if (attr_live) posix_spawnattr_destroy(&attributes);
    for (size_t i = 0; i < sizeof fds / sizeof fds[0]; ++i) if (fds[i] >= 0) close(fds[i]);
    if (pid > 0) {
        kill(pid, SIGKILL);
        while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {}
    }
    if (mailbox != MAP_FAILED) munmap(mailbox, sizeof(CopMailbox));
    if (!ok) cop_opaque_owner_clear(&vm->cop_opaque);
    pthread_setcancelstate(prior_cancel_state, NULL);
    return ok;
}

void vm_ffi_cop_stop(VmState *vm) {
    bool foreign_process = vm->cop_opaque.generation && !cop_opaque_owner_live(&vm->cop_opaque);
    cop_opaque_owner_clear(&vm->cop_opaque);
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
    pid_t w = foreign_process ? -1 : waitpid(vm->cop_pid, &status, WNOHANG);
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
    if (vm->cop_opaque.generation && !cop_opaque_owner_live(&vm->cop_opaque)) return false;
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w > 0 || (w < 0 && errno == ECHILD)) {
        cop_opaque_owner_clear(&vm->cop_opaque);
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

static bool cop_prepare_opaque_call(VmState *vm, const NvmModule *module,
                                    uint32_t index, const NanoValue *args, int count,
                                    char *error, size_t size) {
    if (!module || index >= module->import_count || count < 0 || count > NANO_MAX_FFI_ARGS ||
        (!args && count) || count != module->imports[index].param_count) {
        snprintf(error, size, "I require a valid isolated import and its declared argument count");
        return false;
    }
    const uint8_t *types = module->import_param_types ? module->import_param_types[index] : NULL;
    for (int i = 0; i < count; ++i) {
        if (!cop_opaque_owner_argument(&vm->cop_opaque, args[i]) ||
            (args[i].tag == TAG_OPAQUE && (!types || types[i] != TAG_OPAQUE)) ||
            (types && types[i] == TAG_OPAQUE && args[i].tag != TAG_OPAQUE &&
             !(args[i].tag == TAG_INT && !args[i].as.i64))) {
            snprintf(error, size, "I require an opaque value issued by this live isolated worker");
            return false;
        }
    }
    if (module->imports[index].return_type == TAG_OPAQUE &&
        (!cop_opaque_owner_reserve(&vm->cop_opaque, 1) ||
         !cop_opaque_owner_reply(&vm->cop_opaque, COP_MAX_PAYLOAD))) {
        snprintf(error, size, "I could not reserve isolated opaque publication before native entry");
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
    if (nvm_capture_bindings_present(module) || nvm_service_execution_pending(module)) return false;
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

    if (!cop_prepare_opaque_call(vm, module, import_idx, args, arg_count,
                                 error_msg, error_msg_size)) return false;
    CopOpaqueOwner *owner = cop_opaque_owner_live(&vm->cop_opaque) ? &vm->cop_opaque : NULL;
    uint8_t returned_tag = module->imports[import_idx].return_type;
    uint8_t *reserved_reply = returned_tag == TAG_OPAQUE ? vm->cop_opaque.reply : NULL;
    uint32_t reserved_capacity = reserved_reply ? COP_MAX_PAYLOAD : 0;
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
                    cop_exchange_reserved(-1, vm->cop_out_fd, NULL, 0, (int)remaining,
                                  &type, &reply, &reply_size, reserved_reply, reserved_capacity) &&
                    type == COP_MSG_FFI_RESULT &&
                    cop_apply_call_reply_owned(reply, reply_size, args, (uint8_t)arg_count, result, heap, owner, returned_tag);
                if (reply != reserved_reply) free(reply);
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
                !cop_apply_call_reply_owned(mbox->resp_data, resp_wire_size, args,
                                      (uint8_t)arg_count, result, heap, owner, returned_tag)) {
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
    bool exchanged = cop_exchange_reserved(vm->cop_in_fd, vm->cop_out_fd, request, encoded + 6,
                                    vm->cop_timeout_ms, &response_type, &reply, &reply_size,
                                    reserved_reply, reserved_capacity);
    if (request != payload) free(request);
    if (!exchanged) {
        vm_ffi_cop_stop(vm);
        snprintf(error_msg, error_msg_size, "I could not complete the isolated pipe exchange within %d ms",
                 vm->cop_timeout_ms);
        return false;
    }
    bool ok = false;
    if (response_type == COP_MSG_FFI_RESULT) {
        ok = cop_apply_call_reply_owned(reply, reply_size, args, (uint8_t)arg_count, result, heap, owner, returned_tag);
        if (!ok) snprintf(error_msg, error_msg_size, "I rejected an invalid isolated pipe reply");
    } else {
        snprintf(error_msg, error_msg_size, "%.*s", (int)reply_size, (const char *)reply);
    }
    if (reply != reserved_reply) free(reply);
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
    if (nvm_capture_bindings_present(module) || nvm_service_execution_pending(module)) return false;
    if (count < 0 || (count && (!calls || !results))) {
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

    size_t opaque_results = 0;
    /* I reserve the worst-case metadata before native entry, but a later bad
     * declaration must not erase the existing completed-prefix contract. */
    for (int i = 0; i < count; ++i)
        if (calls[i].import_idx < module->import_count &&
            module->imports[calls[i].import_idx].return_type == TAG_OPAQUE) ++opaque_results;
    if (opaque_results &&
        (!cop_opaque_owner_reserve(&vm->cop_opaque, opaque_results) ||
         !cop_opaque_owner_reply(&vm->cop_opaque, COP_MAX_PAYLOAD))) {
        snprintf(error_msg, error_msg_size, "I could not reserve isolated batch publication before native entry");
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
        uint32_t wpos = 0, reply_reserved = 0;
        int batch_start = next;
        int batched = 0;
        while (next < count && batched < COP_MAX_BATCH) {
            const CopBatchCall *call = &calls[next];
            if (!cop_prepare_opaque_call(vm, module, call->import_idx, call->args,
                                         call->arg_count, error_msg, error_msg_size)) {
                if (!batched) return false;
                break; /* I publish this valid prefix before reporting the next call. */
            }
            uint32_t reply_width = cop_scalar_reply_size(module->imports[call->import_idx].return_type);
            if (!reply_width) {
                if (batched) break;
                snprintf(error_msg, error_msg_size, "I require a supported isolated scalar batch result");
                return false;
            }
            if (reply_width > COP_MAILBOX_SLOT_SIZE - reply_reserved) break;
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
            reply_reserved += reply_width;
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
        if (resp_wire > COP_MAILBOX_SLOT_SIZE || produced > (uint32_t)batched) {
            snprintf(error_msg, error_msg_size, "I rejected an invalid isolated batch reply bound");
            return false;
        }

        /* Unpack the results that did complete, whether or not the batch errored. */
        uint32_t rpos = 0;
        uint32_t unpacked = 0;
        for (uint32_t k = 0; k < produced && k < (uint32_t)batched; k++) {
            NanoValue out;
            uint32_t consumed = cop_deserialize_value(mbox->resp_data + rpos,
                                                      resp_wire - rpos, &out, heap);
            if (consumed == 0) break;
            uint8_t declared = module->imports[calls[batch_start + (int)k].import_idx].return_type;
            if ((out.tag == TAG_OPAQUE) != (declared == TAG_OPAQUE)) {
                vm_release(heap, out);
                break;
            }
            if (out.tag == TAG_OPAQUE) {
                NanoValue token;
                if (!cop_opaque_owner_preview(&vm->cop_opaque, out, &token)) break;
                if (!cop_opaque_owner_publish(&vm->cop_opaque, token)) break;
                out = token;
            }
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
