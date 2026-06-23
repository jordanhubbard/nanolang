/**
 * interpreter_ffi.c - Foreign Function Interface for Interpreter
 *
 * Enables the interpreter to call extern functions from compiled modules.
 * Module loading and symbol resolution are delegated to the shared
 * ffi_loader; this file handles interpreter-specific marshaling,
 * metadata parsing, and module introspection.
 */

#include "interpreter_ffi.h"
#include "module_builder.h"
#include "runtime/gc.h"
#include "runtime/ffi_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

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

/* Marshal nanolang Value to C type and store in buffer 
 * Returns number of bytes used */
static size_t marshal_value_to_c(Value val, Type expected_type, 
                                  unsigned char *buffer, size_t buffer_size) {
    switch (expected_type) {
        case TYPE_INT:
            if (buffer_size < sizeof(int64_t)) return 0;
            *((int64_t*)buffer) = val.as.int_val;
            return sizeof(int64_t);
            
        case TYPE_FLOAT:
            if (buffer_size < sizeof(double)) return 0;
            *((double*)buffer) = val.as.float_val;
            return sizeof(double);
            
        case TYPE_BOOL:
            if (buffer_size < sizeof(bool)) return 0;
            *((bool*)buffer) = val.as.bool_val;
            return sizeof(bool);
            
        case TYPE_STRING:
            if (buffer_size < sizeof(const char*)) return 0;
            *((const char**)buffer) = val.as.string_val;
            return sizeof(const char*);

        case TYPE_OPAQUE:
            /* Opaque values are represented as pointer-sized ints in the interpreter */
            if (buffer_size < sizeof(int64_t)) return 0;
            *((int64_t*)buffer) = val.as.int_val;
            return sizeof(int64_t);
            
        case TYPE_VOID:
            return 0;  /* No marshaling needed */
            
        default:
            fprintf(stderr, "Error: Unsupported FFI type for marshaling: %d\n", expected_type);
            return 0;
    }
}

/* Marshal C return value back to nanolang Value */
static Value marshal_c_to_value(void *c_result, Type return_type) {
    switch (return_type) {
        case TYPE_INT:
            return create_int(*((int64_t*)c_result));
            
        case TYPE_FLOAT:
            return create_float(*((double*)c_result));
            
        case TYPE_BOOL:
            return create_bool(*((bool*)c_result));
            
        case TYPE_STRING: {
            const char *str = *((const char**)c_result);
            return str ? create_string(str) : create_void();
        }
            
        case TYPE_VOID:
            return create_void();

        case TYPE_OPAQUE:
            return create_int(*((int64_t*)c_result));

        case TYPE_ARRAY: {
            /* Arrays are represented as DynArray* in the C runtime/stdlib. */
            int64_t raw = *((int64_t*)c_result);
            DynArray *arr = (DynArray*)(intptr_t)raw;
            Value v;
            v.type = VAL_DYN_ARRAY;
            v.is_return = false;
            v.is_break = false;
            v.is_continue = false;
            v.as.dyn_array_val = arr;
            return v;
        }
            
        default:
            fprintf(stderr, "Error: Unsupported FFI return type: %d\n", return_type);
            return create_void();
    }
}

/* Call an extern function via FFI */
Value ffi_call_extern(const char *function_name, Value *args, int arg_count,
                      Function *func_info, Environment *env) {
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
            return v;
        }
        return create_void();
    }

    if (ffi_verbose) {
        if (module) {
            printf("[FFI] Calling %s from module %s\n", function_name, module->name);
        } else {
            printf("[FFI] Calling %s from RTLD_DEFAULT\n", function_name);
        }
    }
    
    /* Marshal arguments to C types */
    unsigned char arg_buffer[1024];  /* Stack buffer for marshaled args */
    size_t arg_offsets[16];           /* Track where each arg starts */
    size_t total_size = 0;
    
    if (arg_count > 16) {
        fprintf(stderr, "Error: Too many FFI arguments (%d > 16)\n", arg_count);
        return create_void();
    }
    
    for (int i = 0; i < arg_count; i++) {
        arg_offsets[i] = total_size;

        Type param_type = func_info->params[i].type;
        if (param_type == TYPE_STRUCT && func_info->params[i].struct_type_name) {
            if (env_get_opaque_type(env, func_info->params[i].struct_type_name)) {
                param_type = TYPE_OPAQUE;
            }
        }

        size_t size = marshal_value_to_c(args[i], param_type,
                                         arg_buffer + total_size,
                                         sizeof(arg_buffer) - total_size);
        if (size == 0) {
            fprintf(stderr, "Error: Failed to marshal argument %d for %s\n", 
                    i, function_name);
            return create_void();
        }
        total_size += size;
    }
    
    /* Call the C function based on signature 
     * Extract actual values from buffer for calling */
    unsigned char result_buffer[64];
    memset(result_buffer, 0, sizeof(result_buffer));
    
    /* Extract argument values based on their types */
    void *arg_ptrs[16];  /* Actual values to pass */
    for (int i = 0; i < arg_count; i++) {
        Type param_type = func_info->params[i].type;
        if (param_type == TYPE_STRUCT && func_info->params[i].struct_type_name) {
            if (env_get_opaque_type(env, func_info->params[i].struct_type_name)) {
                param_type = TYPE_OPAQUE;
            }
        }
        switch (param_type) {
            case TYPE_INT:
                /* Pass int64_t by value (cast to pointer-sized int) */
                arg_ptrs[i] = (void*)(*((int64_t*)(arg_buffer + arg_offsets[i])));
                break;
            case TYPE_FLOAT:
                /* Pass double by value - NOT SUPPORTED in simple casting */
                /* This is a limitation - need libffi for proper float support */
                arg_ptrs[i] = (void*)(arg_buffer + arg_offsets[i]);
                break;
            case TYPE_BOOL:
                /* Pass bool by value (cast to pointer-sized int) */
                arg_ptrs[i] = (void*)(intptr_t)(*((bool*)(arg_buffer + arg_offsets[i])) ? 1 : 0);
                break;
            case TYPE_STRING:
                /* Strings are already pointers - extract the pointer */
                arg_ptrs[i] = (void*)(*((const char**)(arg_buffer + arg_offsets[i])));
                break;

            case TYPE_OPAQUE: {
                /* Extract opaque pointer */
                void* opaque_ptr = (void*)(intptr_t)(*((int64_t*)(arg_buffer + arg_offsets[i])));

                /* ARC: Unwrap if it's a GC-managed wrapper */
                void* unwrapped = gc_unwrap(opaque_ptr);
                arg_ptrs[i] = unwrapped;
                break;
            }
            default:
                arg_ptrs[i] = NULL;
                break;
        }
    }
    
    /* Call function with extracted arguments */
    typedef int64_t (*FFI_Func_NoArgs)(void);
    typedef int64_t (*FFI_Func_1Arg)(void*);
    typedef int64_t (*FFI_Func_2Args)(void*, void*);
    typedef int64_t (*FFI_Func_3Args)(void*, void*, void*);
    typedef int64_t (*FFI_Func_4Args)(void*, void*, void*, void*);
    typedef int64_t (*FFI_Func_5Args)(void*, void*, void*, void*, void*);
    
    int64_t result = 0;
    
    switch (arg_count) {
        case 0:
            result = ((FFI_Func_NoArgs)func_ptr)();
            break;
        case 1:
            result = ((FFI_Func_1Arg)func_ptr)(arg_ptrs[0]);
            break;
        case 2:
            result = ((FFI_Func_2Args)func_ptr)(arg_ptrs[0], arg_ptrs[1]);
            break;
        case 3:
            result = ((FFI_Func_3Args)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2]);
            break;
        case 4:
            result = ((FFI_Func_4Args)func_ptr)(arg_ptrs[0], arg_ptrs[1], 
                                                arg_ptrs[2], arg_ptrs[3]);
            break;
        case 5:
            result = ((FFI_Func_5Args)func_ptr)(arg_ptrs[0], arg_ptrs[1], 
                                                arg_ptrs[2], arg_ptrs[3], arg_ptrs[4]);
            break;
        default:
            fprintf(stderr, "Error: FFI does not support %d arguments yet (max 5)\n", arg_count);
            return create_void();
    }
    
    /* Marshal result back */
    *((int64_t*)result_buffer) = result;

    Type ret_type = func_info->return_type;
    if (ret_type == TYPE_STRUCT && func_info->return_struct_type_name) {
        if (env_get_opaque_type(env, func_info->return_struct_type_name)) {
            ret_type = TYPE_OPAQUE;
        }
    }

    if (ret_type == TYPE_STRING) {
        const char *str = (const char*)(intptr_t)result;
        Value v = str ? create_string(str) : create_void();
        ModuleBuildMetadata *meta = module ? (ModuleBuildMetadata *)module->user_data : NULL;
        if (str && module_owns_string_return(meta, function_name)) {
            free((void*)str);
        }
        return v;
    }

    /* ARC: Wrap opaque return values if they require manual free */
    if (ret_type == TYPE_OPAQUE && func_info->requires_manual_free && !func_info->returns_borrowed) {
        void* external_ptr = (void*)(intptr_t)result;

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

    return marshal_c_to_value(result_buffer, ret_type);
}

