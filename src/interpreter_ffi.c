/**
 * interpreter_ffi.c - Foreign Function Interface for Interpreter
 * 
 * Enables the interpreter to call extern functions from compiled modules.
 */

#include "interpreter_ffi.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

/* Loaded module tracking */
typedef struct {
    char *name;
    char *path;
    void *handle;  /* dlopen handle */
} LoadedModule;

static LoadedModule *loaded_modules = NULL;
static int loaded_module_count = 0;
static int loaded_module_capacity = 0;
static bool ffi_initialized = false;
static bool ffi_verbose = false;

/* Initialize FFI system */
bool ffi_init(bool verbose) {
    if (ffi_initialized) {
        return true;  /* Already initialized */
    }
    
    ffi_verbose = verbose;
    loaded_modules = calloc(16, sizeof(LoadedModule));
    if (!loaded_modules) {
        fprintf(stderr, "Error: Failed to allocate FFI module tracking\n");
        return false;
    }
    loaded_module_capacity = 16;
    loaded_module_count = 0;
    ffi_initialized = true;
    
    if (ffi_verbose) {
        printf("[FFI] Initialized\n");
    }
    
    return true;
}

/* Cleanup FFI system */
void ffi_cleanup(void) {
    if (!ffi_initialized) return;
    
    for (int i = 0; i < loaded_module_count; i++) {
        if (loaded_modules[i].handle) {
            dlclose(loaded_modules[i].handle);
        }
        free(loaded_modules[i].name);
        free(loaded_modules[i].path);
    }
    
    free(loaded_modules);
    loaded_modules = NULL;
    loaded_module_count = 0;
    loaded_module_capacity = 0;
    ffi_initialized = false;
    
    if (ffi_verbose) {
        printf("[FFI] Cleaned up\n");
    }
}

/* Check if FFI is available */
bool ffi_is_available(void) {
    return ffi_initialized;
}

/* Find a loaded module by name */
static LoadedModule *find_loaded_module(const char *name) {
    for (int i = 0; i < loaded_module_count; i++) {
        if (strcmp(loaded_modules[i].name, name) == 0) {
            return &loaded_modules[i];
        }
    }
    return NULL;
}

/* Add a loaded module to tracking */
static bool add_loaded_module(const char *name, const char *path, void *handle) {
    if (loaded_module_count >= loaded_module_capacity) {
        int new_capacity = loaded_module_capacity * 2;
        LoadedModule *new_modules = realloc(loaded_modules, 
                                            new_capacity * sizeof(LoadedModule));
        if (!new_modules) {
            return false;
        }
        loaded_modules = new_modules;
        loaded_module_capacity = new_capacity;
    }
    
    loaded_modules[loaded_module_count].name = strdup(name);
    loaded_modules[loaded_module_count].path = strdup(path);
    loaded_modules[loaded_module_count].handle = handle;
    loaded_module_count++;
    
    return true;
}

/* Build shared library path for a module */
static bool get_module_library_path(const char *module_name, char *out_path, size_t path_size) {
    /* Try .build/libmodule.dylib (macOS) */
    #ifdef __APPLE__
    snprintf(out_path, path_size, "modules/%s/.build/lib%s.dylib", 
             module_name, module_name);
    if (access(out_path, F_OK) == 0) return true;
    #endif
    
    /* Try .build/libmodule.so (Linux) */
    snprintf(out_path, path_size, "modules/%s/.build/lib%s.so", 
             module_name, module_name);
    if (access(out_path, F_OK) == 0) return true;
    
    return false;
}

/* Load a module's shared library */
bool ffi_load_module(const char *module_name, const char *module_path, 
                     Environment *env, bool verbose) {
    if (!ffi_initialized) {
        fprintf(stderr, "Error: FFI not initialized\n");
        return false;
    }
    
    /* Check if already loaded */
    if (find_loaded_module(module_name)) {
        if (verbose) {
            printf("[FFI] Module '%s' already loaded\n", module_name);
        }
        return true;
    }
    
    /* Get shared library path */
    char lib_path[512];
    if (!get_module_library_path(module_name, lib_path, sizeof(lib_path))) {
        if (verbose) {
            printf("[FFI] No shared library found for module '%s'\n", module_name);
        }
        return false;  /* Not an error - module may not have C code */
    }
    
    /* Load the shared library */
    void *handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        if (verbose) {
            fprintf(stderr, "[FFI] Failed to load %s: %s\n", lib_path, dlerror());
        }
        return false;
    }
    
    /* Add to tracking */
    if (!add_loaded_module(module_name, lib_path, handle)) {
        dlclose(handle);
        return false;
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
            
        default:
            fprintf(stderr, "Error: Unsupported FFI return type: %d\n", return_type);
            return create_void();
    }
}

/* Call an extern function via FFI */
Value ffi_call_extern(const char *function_name, Value *args, int arg_count,
                      Function *func_info, Environment *env) {
    if (!ffi_initialized) {
        fprintf(stderr, "Error: FFI not initialized\n");
        return create_void();
    }
    
    /* Find which module this function belongs to */
    /* For now, try all loaded modules */
    void *func_ptr = NULL;
    LoadedModule *module = NULL;
    
    for (int i = 0; i < loaded_module_count; i++) {
        func_ptr = dlsym(loaded_modules[i].handle, function_name);
        if (func_ptr) {
            module = &loaded_modules[i];
            break;
        }
    }
    
    if (!func_ptr) {
        if (ffi_verbose) {
            fprintf(stderr, "[FFI] Function '%s' not found in loaded modules\n", 
                    function_name);
        }
        return create_void();
    }
    
    if (ffi_verbose) {
        printf("[FFI] Calling %s from module %s\n", function_name, module->name);
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
        size_t size = marshal_value_to_c(args[i], func_info->params[i].type,
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
                arg_ptrs[i] = (void*)(*((bool*)(arg_buffer + arg_offsets[i])) ? 1 : 0);
                break;
            case TYPE_STRING:
                /* Strings are already pointers - extract the pointer */
                arg_ptrs[i] = (void*)(*((const char**)(arg_buffer + arg_offsets[i])));
                break;
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
    return marshal_c_to_value(result_buffer, func_info->return_type);
}

