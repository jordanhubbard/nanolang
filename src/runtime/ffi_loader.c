/**
 * ffi_loader.c - Unified FFI Module Loading
 *
 * Shared dlopen/dlsym plumbing for both the interpreter and VM.
 * See ffi_loader.h for the public API.
 */

#include "ffi_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

/* ── Internal state ──────────────────────────────────────────────── */

#define FFI_INITIAL_CAPACITY 16

static FfiModule *modules = NULL;
static int module_count = 0;
static int module_capacity = 0;
static bool initialized = false;
static bool verbose_mode = false;

/* ── Lifecycle ───────────────────────────────────────────────────── */

bool ffi_loader_init(bool verbose) {
    if (initialized) return true;

    verbose_mode = verbose;
    modules = calloc(FFI_INITIAL_CAPACITY, sizeof(FfiModule));
    if (!modules) {
        fprintf(stderr, "ffi_loader: allocation failed\n");
        return false;
    }
    module_capacity = FFI_INITIAL_CAPACITY;
    module_count = 0;
    initialized = true;

    if (verbose_mode) {
        fprintf(stderr, "[ffi_loader] Initialized\n");
    }
    return true;
}

void ffi_loader_shutdown(void) {
    if (!initialized) return;

    for (int i = 0; i < module_count; i++) {
        if (modules[i].handle) {
            dlclose(modules[i].handle);
        }
        free(modules[i].name);
        free(modules[i].path);
        /* NOTE: user_data is NOT freed here — caller's responsibility */
    }

    free(modules);
    modules = NULL;
    module_count = 0;
    module_capacity = 0;
    initialized = false;

    if (verbose_mode) {
        fprintf(stderr, "[ffi_loader] Shut down\n");
    }
}

bool ffi_loader_is_initialized(void) {
    return initialized;
}

/* ── Module tracking ─────────────────────────────────────────────── */

FfiModule *ffi_loader_find(const char *module_name) {
    if (!module_name) return NULL;
    for (int i = 0; i < module_count; i++) {
        if (strcmp(modules[i].name, module_name) == 0) {
            return &modules[i];
        }
    }
    return NULL;
}

bool ffi_loader_open(const char *module_name, const char *lib_path) {
    if (!initialized) {
        if (!ffi_loader_init(false)) return false;
    }

    /* Idempotent */
    if (ffi_loader_find(module_name)) return true;

    /* Grow array if needed */
    if (module_count >= module_capacity) {
        int new_cap = module_capacity * 2;
        FfiModule *new_arr = realloc(modules, (size_t)new_cap * sizeof(FfiModule));
        if (!new_arr) return false;
        modules = new_arr;
        module_capacity = new_cap;
    }

    /* Use RTLD_GLOBAL so module-to-module symbol deps can resolve */
    void *handle = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        if (verbose_mode) {
            fprintf(stderr, "[ffi_loader] Failed to load %s: %s\n",
                    lib_path, dlerror());
        }
        return false;
    }

    FfiModule *m = &modules[module_count];
    m->name = strdup(module_name);
    m->path = strdup(lib_path);
    m->handle = handle;
    m->user_data = NULL;
    module_count++;

    if (verbose_mode) {
        fprintf(stderr, "[ffi_loader] Loaded '%s' from %s\n", module_name, lib_path);
    }
    return true;
}

FfiModule *ffi_loader_get_modules(int *out_count) {
    if (out_count) *out_count = module_count;
    return modules;
}

/* ── Symbol resolution ───────────────────────────────────────────── */

void *ffi_loader_resolve(const char *symbol_name) {
    return ffi_loader_resolve_in(symbol_name, NULL);
}

void *ffi_loader_resolve_in(const char *symbol_name, FfiModule **out_module) {
    if (out_module) *out_module = NULL;

    /* Search loaded modules */
    for (int i = 0; i < module_count; i++) {
        void *ptr = dlsym(modules[i].handle, symbol_name);
        if (ptr) {
            if (out_module) *out_module = &modules[i];
            return ptr;
        }
    }

    /* Fallback: main executable + already-loaded libraries */
    void *self = dlopen(NULL, RTLD_LAZY);
    if (self) {
        void *ptr = dlsym(self, symbol_name);
        dlclose(self);
        if (ptr) return ptr;
    }

    return NULL;
}

/* ── Library search ──────────────────────────────────────────────── */

bool ffi_loader_find_library(const char *module_name, const char *module_dir,
                             char *out_path, size_t path_size) {
    if (!module_name) return false;

    /* Normalize: strip "modules/" prefix and ".nano" suffix */
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

    /* Extract leaf name (e.g., "vector2d" from "std/math/vector2d") */
    const char *leaf_slash = strrchr(mn, '/');
    const char *lib_name = leaf_slash ? leaf_slash + 1 : mn;

    /* Extract top-level dir (e.g., "std" from "std/fs") */
    char top_dir[256] = {0};
    const char *first_slash = strchr(mn, '/');
    if (first_slash) {
        size_t len = (size_t)(first_slash - mn);
        if (len < sizeof(top_dir)) {
            memcpy(top_dir, mn, len);
            top_dir[len] = '\0';
        }
    }

    /* Build parent dir and underscore-joined name.
     * E.g., "std/json/json" → parent_dir="std/json", joined="std_json" */
    char parent_dir[512] = {0};
    char joined_name[256] = {0};
    if (leaf_slash) {
        size_t plen = (size_t)(leaf_slash - mn);
        if (plen < sizeof(parent_dir)) {
            memcpy(parent_dir, mn, plen);
            parent_dir[plen] = '\0';
        }
        size_t ji = 0;
        for (size_t k = 0; k < plen && ji < sizeof(joined_name) - 1; k++) {
            joined_name[ji++] = (mn[k] == '/') ? '_' : mn[k];
        }
        joined_name[ji] = '\0';
    }

    /* Extensions to try */
    const char *exts[] = {
#ifdef __APPLE__
        "dylib",
#endif
        "so", NULL
    };

    /* Pattern 1: module_dir (interpreter supplies this from import path) */
    if (module_dir && module_dir[0] != '\0') {
        for (int ei = 0; exts[ei]; ei++) {
            snprintf(out_path, path_size, "%s/.build/lib%s.%s",
                     module_dir, lib_name, exts[ei]);
            if (access(out_path, F_OK) == 0) return true;
        }
    }

    for (int ei = 0; exts[ei]; ei++) {
        /* Pattern 2: modules/<full_normalized>/.build/lib<leaf>.<ext> */
        snprintf(out_path, path_size, "modules/%s/.build/lib%s.%s",
                 mn, lib_name, exts[ei]);
        if (access(out_path, F_OK) == 0) return true;

        /* Pattern 3: modules/<parent_dir>/.build/lib<joined>.<ext> */
        if (parent_dir[0]) {
            snprintf(out_path, path_size, "modules/%s/.build/lib%s.%s",
                     parent_dir, joined_name, exts[ei]);
            if (access(out_path, F_OK) == 0) return true;
        }

        /* Pattern 4: modules/<top_dir>/.build/lib<top_dir>.<ext> */
        if (top_dir[0]) {
            snprintf(out_path, path_size, "modules/%s/.build/lib%s.%s",
                     top_dir, top_dir, exts[ei]);
            if (access(out_path, F_OK) == 0) return true;
        }
    }

    return false;
}
