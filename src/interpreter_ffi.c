/**
 * interpreter_ffi.c - Foreign Function Interface for Interpreter
 *
 * Enables the interpreter to call extern functions from compiled modules.
 * Module loading and symbol resolution are delegated to the shared
 * ffi_loader; this file handles interpreter-specific marshaling,
 * metadata parsing, and module introspection.
 */

#include "interpreter_ffi.h"
#include "nanoisa/nvm_format.h"
#include "module_builder.h"
#include "runtime/gc.h"
#include "runtime/ffi_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <ffi.h>

static bool ffi_verbose = false;

/* Initialize FFI system */
bool ffi_init(bool verbose) {
    ffi_verbose = verbose;
    return ffi_loader_init(verbose);
}

/* Cleanup FFI system */
void ffi_cleanup(void) {
    /* Free interpreter-specific user_data (ModuleBuildMetadata) before shutdown */
    int count = 0;
    FfiModule *mods = ffi_loader_get_modules(&count);
    for (int i = 0; i < count; i++) {
        if (mods[i].user_data) {
            module_metadata_free((ModuleBuildMetadata *)mods[i].user_data);
            mods[i].user_data = NULL;
        }
    }
    ffi_loader_shutdown();
}

/* Check if FFI is available */
bool ffi_is_available(void) {
    return ffi_loader_is_initialized();
}

static bool module_owns_string_return(const ModuleBuildMetadata *meta, const char *function_name) {
    if (!meta || !function_name) return false;
    for (size_t i = 0; i < meta->owned_string_returns_count; i++) {
        if (meta->owned_string_returns[i] && strcmp(meta->owned_string_returns[i], function_name) == 0) {
            return true;
        }
    }
    return false;
}

static bool ffi_try_module_introspection(const char *function_name,
                                        Value *args,
                                        int arg_count,
                                        Function *func_info,
                                        Environment *env,
                                        Value *out) {
    if (!function_name || !func_info || !env || !out) return false;

    const char *module_name = NULL;

    /* ___module_is_unsafe_<mod>() -> bool */
    const char *pfx_is_unsafe = "___module_is_unsafe_";
    if (strncmp(function_name, pfx_is_unsafe, strlen(pfx_is_unsafe)) == 0) {
        module_name = function_name + strlen(pfx_is_unsafe);
        ModuleInfo *mod = env_get_module(env, module_name);
        *out = create_bool(mod ? mod->is_unsafe : false);
        return true;
    }

    /* ___module_has_ffi_<mod>() -> bool */
    const char *pfx_has_ffi = "___module_has_ffi_";
    if (strncmp(function_name, pfx_has_ffi, strlen(pfx_has_ffi)) == 0) {
        module_name = function_name + strlen(pfx_has_ffi);
        ModuleInfo *mod = env_get_module(env, module_name);
        *out = create_bool(mod ? mod->has_ffi : false);
        return true;
    }

    /* ___module_name_<mod>() -> string */
    const char *pfx_name = "___module_name_";
    if (strncmp(function_name, pfx_name, strlen(pfx_name)) == 0) {
        module_name = function_name + strlen(pfx_name);
        *out = create_string(module_name);
        return true;
    }

    /* ___module_path_<mod>() -> string */
    const char *pfx_path = "___module_path_";
    if (strncmp(function_name, pfx_path, strlen(pfx_path)) == 0) {
        module_name = function_name + strlen(pfx_path);
        ModuleInfo *mod = env_get_module(env, module_name);
        *out = create_string((mod && mod->path) ? mod->path : "");
        return true;
    }

    /* ___module_function_count_<mod>() -> int */
    const char *pfx_fn_count = "___module_function_count_";
    if (strncmp(function_name, pfx_fn_count, strlen(pfx_fn_count)) == 0) {
        module_name = function_name + strlen(pfx_fn_count);
        ModuleInfo *mod = env_get_module(env, module_name);
        *out = create_int(mod ? mod->function_count : 0);
        return true;
    }

    /* ___module_function_name_<mod>(idx: int) -> string */
    const char *pfx_fn_name = "___module_function_name_";
    if (strncmp(function_name, pfx_fn_name, strlen(pfx_fn_name)) == 0) {
        module_name = function_name + strlen(pfx_fn_name);
        ModuleInfo *mod = env_get_module(env, module_name);
        int64_t idx = 0;
        if (arg_count >= 1 && args) {
            idx = args[0].as.int_val;
        }
        if (mod && mod->exported_functions && idx >= 0 && idx < mod->function_count) {
            *out = create_string(mod->exported_functions[idx] ? mod->exported_functions[idx] : "");
        } else {
            *out = create_string("");
        }
        return true;
    }

    /* ___module_struct_count_<mod>() -> int */
    const char *pfx_struct_count = "___module_struct_count_";
    if (strncmp(function_name, pfx_struct_count, strlen(pfx_struct_count)) == 0) {
        module_name = function_name + strlen(pfx_struct_count);
        ModuleInfo *mod = env_get_module(env, module_name);
        *out = create_int(mod ? mod->struct_count : 0);
        return true;
    }

    /* ___module_struct_name_<mod>(idx: int) -> string */
    const char *pfx_struct_name = "___module_struct_name_";
    if (strncmp(function_name, pfx_struct_name, strlen(pfx_struct_name)) == 0) {
        module_name = function_name + strlen(pfx_struct_name);
        ModuleInfo *mod = env_get_module(env, module_name);
        int64_t idx = 0;
        if (arg_count >= 1 && args) {
            idx = args[0].as.int_val;
        }
        if (mod && mod->exported_structs && idx >= 0 && idx < mod->struct_count) {
            *out = create_string(mod->exported_structs[idx] ? mod->exported_structs[idx] : "");
        } else {
            *out = create_string("");
        }
        return true;
    }

    (void)func_info;
    return false;
}

static char* derive_module_dir_from_path(const char *module_path, const char *lib_path) {
    if (module_path && module_path[0] != '\0') {
        char *dir = strdup(module_path);
        if (!dir) return NULL;
        if (strstr(dir, ".nano") != NULL) {
            char *last_slash = strrchr(dir, '/');
            if (last_slash) *last_slash = '\0';
        }
        return dir;
    }

    if (lib_path && lib_path[0] != '\0') {
        char *dir = strdup(lib_path);
        if (!dir) return NULL;
        char *build = strstr(dir, "/.build/");
        if (build) {
            *build = '\0';
        } else {
            char *last_slash = strrchr(dir, '/');
            if (last_slash) *last_slash = '\0';
        }
        return dir;
    }

    return NULL;
}

/* Derive the module_dir that ffi_loader_find_library wants from
 * the interpreter's module_path (which may be a .nano file path). */
static char *interp_module_dir(const char *module_path) {
    if (!module_path || module_path[0] == '\0') return NULL;
    char *dir = strdup(module_path);
    if (!dir) return NULL;
    if (strstr(dir, ".nano") != NULL) {
        char *last_slash = strrchr(dir, '/');
        if (last_slash) *last_slash = '\0';
    }
    return dir;
}

/* Load a module's shared library */
bool ffi_load_module(const char *module_name, const char *module_path,
                     Environment *env, bool verbose) {
    (void)env;

    if (!ffi_loader_is_initialized()) {
        if (!ffi_loader_init(verbose)) {
            fprintf(stderr, "Error: FFI not initialized\n");
            return false;
        }
    }

    /* Already loaded? */
    if (ffi_loader_find(module_name)) {
        if (verbose) {
            printf("[FFI] Module '%s' already loaded\n", module_name);
        }
        return true;
    }

    /* Find library on disk */
    char lib_path[512];
    char *mod_dir = interp_module_dir(module_path);
    bool found = ffi_loader_find_library(module_name, mod_dir,
                                         lib_path, sizeof(lib_path));
    free(mod_dir);

    if (!found) {
        if (verbose) {
            printf("[FFI] No shared library found for module '%s'\n", module_name);
        }
        return false;
    }

    /* Open via shared loader */
    if (!ffi_loader_open(module_name, lib_path)) {
        return false;
    }

    /* Parse module.json for FFI ownership metadata (optional) */
    FfiModule *m = ffi_loader_find(module_name);
    if (m) {
        char *metadata_dir = derive_module_dir_from_path(module_path, lib_path);
        if (metadata_dir) {
            m->user_data = module_load_metadata(metadata_dir);
            free(metadata_dir);
        }
    }

    if (verbose) {
        printf("[FFI] Loaded module '%s' from %s\n", module_name, lib_path);
    }
    return true;
}

/* I give libffi aligned storage matching each declared scalar type. */
typedef union {
    ffi_arg word;
    int64_t integer;
    double floating;
    uint8_t boolean;
    void *pointer;
} InterpreterFFISlot;

static ffi_type *interpreter_ffi_type(Type type) {
    switch (type) {
        case TYPE_INT: return &ffi_type_sint64;
        case TYPE_FLOAT: return &ffi_type_double;
        case TYPE_BOOL: return &ffi_type_uint8;
        case TYPE_STRING:
        case TYPE_OPAQUE:
        case TYPE_ARRAY: return &ffi_type_pointer;
        case TYPE_VOID: return &ffi_type_void;
        default: return NULL;
    }
}

static bool interpreter_ffi_argument(Value value, Type type, InterpreterFFISlot *slot) {
    switch (type) {
        case TYPE_INT:
            if (value.type != VAL_INT) return false;
            slot->integer = value.as.int_val;
            return true;
        case TYPE_FLOAT:
            if (value.type != VAL_FLOAT) return false;
            slot->floating = value.as.float_val;
            return true;
        case TYPE_BOOL:
            if (value.type != VAL_BOOL) return false;
            slot->boolean = value.as.bool_val ? 1 : 0;
            return true;
        case TYPE_STRING:
            if (value.type != VAL_STRING) return false;
            slot->pointer = value.as.string_val;
            return true;
        case TYPE_OPAQUE:
            if (value.type != VAL_INT) return false;
            slot->pointer = gc_unwrap((void *)(intptr_t)value.as.int_val);
            return true;
        default:
            return false;
    }
}

/* Call an extern function via FFI */
Value ffi_call_extern(const char *function_name, Value *args, int arg_count,
                      Function *func_info, Environment *env) {
    bool success;
    return ffi_call_extern_checked(function_name, args, arg_count, func_info, env, &success);
}

Value ffi_call_extern_checked(const char *function_name, Value *args, int arg_count,
                             Function *func_info, Environment *env, bool *success) {
    if (!success) return create_void();
    *success = false;
    if (!function_name || !func_info || !env || arg_count < 0 ||
        arg_count > NANO_MAX_FFI_ARGS || arg_count != func_info->param_count ||
        (arg_count && (!args || !func_info->params))) {
        fprintf(stderr, "I cannot call FFI with invalid signature metadata or arguments.\n");
        return create_void();
    }
    if (!ffi_loader_is_initialized()) {
        fprintf(stderr, "Error: FFI not initialized\n");
        return create_void();
    }

    /* Resolve the function through the shared loader */
    FfiModule *module = NULL;
    void *func_ptr = ffi_loader_resolve_in(function_name, &module);

    if (!func_ptr) {
        if (ffi_verbose) {
            fprintf(stderr, "[FFI] Function '%s' not found in loaded modules\n",
                    function_name);
        }
        Value v;
        if (ffi_try_module_introspection(function_name, args, arg_count, func_info, env, &v)) {
            *success = true;
            return v;
        }
        fprintf(stderr, "I cannot resolve foreign function '%s'.\n", function_name);
        return create_void();
    }

    Type ret_type = func_info->return_type;
    if (ret_type == TYPE_STRUCT && func_info->return_struct_type_name &&
        env_get_opaque_type(env, func_info->return_struct_type_name)) {
        ret_type = TYPE_OPAQUE;
    }
    ffi_type *result_type = interpreter_ffi_type(ret_type);
    if (!result_type) {
        fprintf(stderr, "I cannot dispatch foreign result type %d for '%s'.\n", ret_type, function_name);
        return create_void();
    }

    InterpreterFFISlot slots[NANO_MAX_FFI_ARGS] = {0};
    void *values[NANO_MAX_FFI_ARGS] = {0};
    ffi_type *types[NANO_MAX_FFI_ARGS] = {0};
    for (int i = 0; i < arg_count; i++) {
        Type type = func_info->params[i].type;
        if (type == TYPE_STRUCT && func_info->params[i].struct_type_name &&
            env_get_opaque_type(env, func_info->params[i].struct_type_name)) {
            type = TYPE_OPAQUE;
        }
        types[i] = interpreter_ffi_type(type);
        if (!types[i] || !interpreter_ffi_argument(args[i], type, &slots[i])) {
            fprintf(stderr, "I cannot marshal foreign argument %d for '%s'.\n", i, function_name);
            return create_void();
        }
        values[i] = &slots[i];
    }

    ffi_cif cif;
    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, (unsigned int)arg_count,
                     result_type, types) != FFI_OK) {
        fprintf(stderr, "I cannot prepare the native signature for '%s'.\n", function_name);
        return create_void();
    }
    InterpreterFFISlot native_result = {0};
    ffi_call(&cif, FFI_FN(func_ptr), &native_result, values);
    *success = true;

    if (ret_type == TYPE_STRING) {
        const char *str = native_result.pointer;
        Value v = str ? create_string(str) : create_void();
        ModuleBuildMetadata *meta = module ? (ModuleBuildMetadata *)module->user_data : NULL;
        if (str && module_owns_string_return(meta, function_name)) {
            free((void*)str);
        }
        return v;
    }

    /* ARC: Wrap opaque return values if they require manual free */
    if (ret_type == TYPE_OPAQUE && func_info->requires_manual_free && !func_info->returns_borrowed) {
        void* external_ptr = native_result.pointer;

        if (external_ptr && func_info->cleanup_function) {
            /* Look up the cleanup function through the shared resolver */
            void (*cleanup_func)(void*) = NULL;
            cleanup_func = (void (*)(void*))ffi_loader_resolve(func_info->cleanup_function);

            if (cleanup_func) {
                /* Wrap external pointer in GC-managed object */
                void* wrapped = gc_wrap_external(external_ptr, cleanup_func);

                /* Opaque pointers stored as int64_t in interpreter */
                return create_int((int64_t)(intptr_t)wrapped);
            } else {
                fprintf(stderr, "[ARC] Warning: cleanup function '%s' not found for %s\n",
                        func_info->cleanup_function, function_name);
                /* Return unwrapped - will leak, but prevents crash */
            }
        }
    }

    switch (ret_type) {
        case TYPE_INT: return create_int(native_result.integer);
        case TYPE_FLOAT: return create_float(native_result.floating);
        /* I read libffi's widened result for sub-register integer types. */
        case TYPE_BOOL: return create_bool(native_result.word != 0);
        case TYPE_OPAQUE: return create_int((int64_t)(intptr_t)native_result.pointer);
        case TYPE_ARRAY: {
            Value value = create_void();
            value.type = VAL_DYN_ARRAY;
            value.as.dyn_array_val = native_result.pointer;
            return value;
        }
        case TYPE_VOID: return create_void();
        default:
            *success = false;
            return create_void();
    }
}
