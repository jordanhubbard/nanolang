/*
 * NanoVM FFI Bridge - Call native C functions from the VM
 *
 * Loads shared libraries with dlopen, resolves functions with dlsym,
 * and marshals between NanoValue and C function signatures.
 */

#include "vm_ffi.h"
#include "runtime/dyn_array.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

/* ========================================================================
 * Module Registry
 * ======================================================================== */

typedef struct {
    char *name;
    void *handle;  /* dlopen handle */
} VmLoadedModule;

#define VM_FFI_MAX_MODULES 64

static VmLoadedModule loaded_modules[VM_FFI_MAX_MODULES];
static int loaded_module_count = 0;
static bool ffi_initialized = false;

void vm_ffi_init(void) {
    if (ffi_initialized) return;
    memset(loaded_modules, 0, sizeof(loaded_modules));
    loaded_module_count = 0;
    ffi_initialized = true;
}

void vm_ffi_shutdown(void) {
    for (int i = 0; i < loaded_module_count; i++) {
        if (loaded_modules[i].handle) {
            dlclose(loaded_modules[i].handle);
        }
        free(loaded_modules[i].name);
    }
    loaded_module_count = 0;
    ffi_initialized = false;
}

/* Check if a module is already loaded */
static bool module_already_loaded(const char *name) {
    for (int i = 0; i < loaded_module_count; i++) {
        if (strcmp(loaded_modules[i].name, name) == 0) return true;
    }
    return false;
}

bool vm_ffi_load_module(const char *module_name) {
    if (!ffi_initialized) vm_ffi_init();
    if (module_already_loaded(module_name)) return true;
    if (loaded_module_count >= VM_FFI_MAX_MODULES) return false;

    /* Try standard module paths */
    char path[1024];
    void *handle = NULL;

    /* Normalize module name: strip "modules/" prefix and ".nano" suffix.
     * E.g., "modules/std/json/json.nano" → "std/json/json"
     *        "std/fs" → "std/fs" (unchanged) */
    char normalized[512];
    const char *mn = module_name;
    if (strncmp(mn, "modules/", 8) == 0) mn += 8;
    size_t mn_len = strlen(mn);
    if (mn_len > 5 && strcmp(mn + mn_len - 5, ".nano") == 0) {
        if (mn_len - 5 < sizeof(normalized)) {
            memcpy(normalized, mn, mn_len - 5);
            normalized[mn_len - 5] = '\0';
            mn = normalized;
        }
    }

    /* For module names like "std/math/vector2d", extract the leaf name */
    const char *leaf = strrchr(mn, '/');
    const char *lib_name = leaf ? leaf + 1 : mn;

    /* Also extract the top-level dir (e.g., "std" from "std/fs") */
    char top_dir[256] = {0};
    const char *slash = strchr(mn, '/');
    if (slash) {
        size_t len = (size_t)(slash - mn);
        if (len < sizeof(top_dir)) {
            memcpy(top_dir, mn, len);
            top_dir[len] = '\0';
        }
    }

    /* Build parent dir path and underscore-joined lib name.
     * E.g., "std/json/json" → parent_dir="std/json", joined_name="std_json" */
    char parent_dir[512] = {0};
    char joined_name[256] = {0};
    if (leaf) {
        size_t plen = (size_t)(leaf - mn);
        if (plen < sizeof(parent_dir)) {
            memcpy(parent_dir, mn, plen);
            parent_dir[plen] = '\0';
        }
        /* Build underscore-joined name: "std/json" → "std_json" */
        size_t ji = 0;
        for (size_t k = 0; k < plen && ji < sizeof(joined_name) - 1; k++) {
            joined_name[ji++] = (mn[k] == '/') ? '_' : mn[k];
        }
        joined_name[ji] = '\0';
    }

    /* Path patterns to try - each with (dir, libname) args */
    const char *exts[] = { "dylib", "so", NULL };

    for (int ei = 0; exts[ei]; ei++) {
        /* Try: modules/<full_mn>/.build/lib<leaf>.<ext> */
        snprintf(path, sizeof(path), "modules/%s/.build/lib%s.%s", mn, lib_name, exts[ei]);
        if (access(path, F_OK) == 0) {
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
            if (handle) break;
        }

        /* Try: modules/<parent_dir>/.build/lib<joined_name>.<ext>
         * E.g., modules/std/json/.build/libstd_json.dylib */
        if (parent_dir[0]) {
            snprintf(path, sizeof(path), "modules/%s/.build/lib%s.%s",
                     parent_dir, joined_name, exts[ei]);
            if (access(path, F_OK) == 0) {
                handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
                if (handle) break;
            }
        }

        /* Try: modules/<top_dir>/.build/lib<top_dir>.<ext>
         * E.g., modules/std/.build/libstd.dylib */
        if (top_dir[0]) {
            snprintf(path, sizeof(path), "modules/%s/.build/lib%s.%s",
                     top_dir, top_dir, exts[ei]);
            if (access(path, F_OK) == 0) {
                handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
                if (handle) break;
            }
        }
    }

    if (!handle) {
        /* Not fatal - function might be in main executable or already-loaded lib */
        return false;
    }

    loaded_modules[loaded_module_count].name = strdup(module_name);
    loaded_modules[loaded_module_count].handle = handle;
    loaded_module_count++;
    return true;
}

/* ========================================================================
 * Function Resolution
 * ======================================================================== */

static void *resolve_function(const char *name) {
    /* Search loaded modules */
    for (int i = 0; i < loaded_module_count; i++) {
        void *ptr = dlsym(loaded_modules[i].handle, name);
        if (ptr) return ptr;
    }

    /* Fallback: search main executable and already-loaded libraries */
    void *self = dlopen(NULL, RTLD_LAZY);
    if (self) {
        void *ptr = dlsym(self, name);
        dlclose(self);
        if (ptr) return ptr;
    }

    return NULL;
}

/* ========================================================================
 * NanoValue ↔ C Marshaling
 * ======================================================================== */

/* Marshal NanoValue args to C void* array for polymorphic dispatch */
static int marshal_args(NanoValue *args, int arg_count,
                        const NvmImportEntry *imp, const uint8_t *param_types,
                        void **arg_ptrs) {
    for (int i = 0; i < arg_count; i++) {
        uint8_t expected_tag = (i < imp->param_count && param_types)
                               ? param_types[i] : args[i].tag;

        switch (expected_tag) {
            case TAG_INT:
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_FLOAT:
                /* Float args: store in stack buffer and pass pointer.
                 * NOTE: This is a known limitation - proper float passing
                 * requires libffi. For now, cast to intptr_t which works
                 * on arm64 for many C functions that take double. */
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
            default:
                /* Pass as raw int64 (best effort) */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
        }
    }
    return arg_count;
}

/* Convert C int64_t result to NanoValue based on return type tag */
static NanoValue marshal_result(int64_t raw_result, uint8_t return_tag,
                                VmHeap *heap) {
    switch (return_tag) {
        case TAG_INT:
            return val_int(raw_result);
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
                VmString *vs = vm_string_new(heap, str, (uint32_t)strlen(str));
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
            /* C functions return DynArray* - convert to VmArray */
            DynArray *darr = (DynArray *)(intptr_t)raw_result;
            if (!darr) return val_void();
            VmArray *varr = vm_array_new(heap, TAG_INT, (uint32_t)darr->length);
            for (int64_t ai = 0; ai < darr->length; ai++) {
                NanoValue elem;
                switch (darr->elem_type) {
                    case ELEM_INT:
                        elem = val_int(dyn_array_get_int(darr, ai));
                        break;
                    case ELEM_FLOAT:
                        elem = val_float(dyn_array_get_float(darr, ai));
                        break;
                    case ELEM_BOOL:
                        elem = val_bool(dyn_array_get_bool(darr, ai));
                        break;
                    case ELEM_STRING: {
                        const char *s = dyn_array_get_string(darr, ai);
                        if (s) {
                            VmString *vs = vm_string_new(heap, s, (uint32_t)strlen(s));
                            elem = val_string(vs);
                        } else {
                            elem = val_void();
                        }
                        break;
                    }
                    default:
                        elem = val_int(dyn_array_get_int(darr, ai));
                        break;
                }
                vm_array_push(varr, elem);
            }
            NanoValue v = {0};
            v.tag = TAG_ARRAY;
            v.as.array = varr;
            return v;
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

bool vm_ffi_call(const NvmModule *module, uint32_t import_idx,
                 NanoValue *args, int arg_count,
                 NanoValue *result, VmHeap *heap,
                 char *error_msg, size_t error_msg_size) {
    if (!ffi_initialized) vm_ffi_init();

    if (import_idx >= module->import_count) {
        snprintf(error_msg, error_msg_size, "Import index %u out of range", import_idx);
        return false;
    }

    const NvmImportEntry *imp = &module->imports[import_idx];
    const char *func_name = nvm_get_string(module, imp->function_name_idx);
    const char *mod_name = nvm_get_string(module, imp->module_name_idx);

    if (!func_name) {
        snprintf(error_msg, error_msg_size, "NULL function name for import %u", import_idx);
        return false;
    }

    /* Try to load the module if we have a module name */
    if (mod_name && mod_name[0] != '\0') {
        vm_ffi_load_module(mod_name);
    }

    /* Resolve the function */
    void *func_ptr = resolve_function(func_name);
    if (!func_ptr) {
        snprintf(error_msg, error_msg_size,
                 "FFI: function '%s' not found (module '%s')",
                 func_name, mod_name ? mod_name : "");
        return false;
    }

    /* Marshal arguments */
    void *arg_ptrs[16] = {0};
    if (arg_count > 16) {
        snprintf(error_msg, error_msg_size, "Too many FFI arguments (%d > 16)", arg_count);
        return false;
    }
    marshal_args(args, arg_count, imp,
                 module->import_param_types ? module->import_param_types[import_idx] : NULL,
                 arg_ptrs);

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
        default:
            snprintf(error_msg, error_msg_size,
                     "FFI: unsupported arg count %d (max 5)", arg_count);
            return false;
    }

    /* Marshal result */
    *result = marshal_result(raw_result, imp->return_type, heap);
    return true;
}
