/**
 * ffi_loader.c - Unified FFI Module Loading
 *
 * Shared dlopen/dlsym plumbing for both the interpreter and VM.
 * See ffi_loader.h for the public API.
 *
 * Thread safety: protected by a pthread_rwlock_t.
 * - ffi_loader_open() / ffi_loader_shutdown() take a write lock
 * - ffi_loader_resolve() / ffi_loader_find() take a read lock
 * - The interpreter is single-threaded (lock has zero contention)
 * - The daemon runs concurrent threads (read lock allows parallel symbol resolution)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L  /* For strdup(), pthread_rwlock_t */

#include "ffi_loader.h"
#include "runtime/module_build_dir.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <limits.h>
#include <sched.h>

/* ── Internal state ──────────────────────────────────────────────── */

#define FFI_INITIAL_CAPACITY 16

static FfiModule *modules = NULL;
static int module_count = 0;
static int module_capacity = 0;
static bool initialized = false;
static bool verbose_mode = false;

static pthread_rwlock_t ffi_lock = PTHREAD_RWLOCK_INITIALIZER;

/* I require inline lock-free atomics: a hidden atomic-library lock would
 * defeat my fork admission boundary. GCC and Clang expose this same builtin. */
#if !defined(__GNUC__) && !defined(__clang__)
#error "I require compiler lock-free atomic support for loader fork admission"
#endif
_Static_assert(__atomic_always_lock_free(sizeof(unsigned), 0),
               "I require always-lock-free unsigned loader admission");
#define FFI_ADMISSION_CLOSED (UINT_MAX / 2u + 1u)
static unsigned ffi_admission;
static unsigned ffi_process;
static FfiLoaderFork *ffi_prepared;
static bool ffi_child_local;
static bool ffi_registered;
/* I never forget native image entry: even a failed load may run a constructor. */
static unsigned ffi_native_entered;

static void *ffi_native_open(const char *path, int flags) {
    __atomic_store_n(&ffi_native_entered, 1u, __ATOMIC_RELEASE);
    return dlopen(path, flags);
}

static void *ffi_native_symbol(void *image, const char *name) {
    __atomic_store_n(&ffi_native_entered, 1u, __ATOMIC_RELEASE);
    return dlsym(image, name);
}

static bool ffi_same_process(void) {
    unsigned current = (unsigned)getpid();
    unsigned owner = __atomic_load_n(&ffi_process, __ATOMIC_ACQUIRE);
    if (!owner) {
        unsigned empty = 0;
        if (__atomic_compare_exchange_n(&ffi_process, &empty, current, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return true;
        owner = empty;
    }
    return owner == current;
}

static bool ffi_registry_lock(bool writer) {
    /* I refuse an unprepared fork before touching a possibly inherited lock. */
    if (!ffi_same_process()) return false;
    unsigned count = __atomic_load_n(&ffi_admission, __ATOMIC_ACQUIRE);
    for (;;) {
        if (count & FFI_ADMISSION_CLOSED) {
            sched_yield();
            count = __atomic_load_n(&ffi_admission, __ATOMIC_ACQUIRE);
            continue;
        }
        if (count == FFI_ADMISSION_CLOSED - 1u) return false;
        if (__atomic_compare_exchange_n(&ffi_admission, &count, count + 1u,
                                        false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) break;
    }
    int result = writer ? pthread_rwlock_wrlock(&ffi_lock) : pthread_rwlock_rdlock(&ffi_lock);
    if (!result) return true;
    __atomic_fetch_sub(&ffi_admission, 1u, __ATOMIC_RELEASE);
    return false;
}

static void ffi_registry_unlock(void) {
    pthread_rwlock_unlock(&ffi_lock);
    __atomic_fetch_sub(&ffi_admission, 1u, __ATOMIC_RELEASE);
}

bool ffi_loader_fork_prepare(FfiLoaderFork *token) {
    if (!token || !ffi_same_process()) return false;
    unsigned empty = 0;
    if (!__atomic_compare_exchange_n(&ffi_admission, &empty, FFI_ADMISSION_CLOSED,
                                     false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return false;
    token->parent_pid = (unsigned)getpid();
    ffi_prepared = token;
    return true;
}

bool ffi_loader_fork_parent(FfiLoaderFork *token) {
    if (!token || token->parent_pid != (unsigned)getpid() ||
        __atomic_load_n(&ffi_admission, __ATOMIC_ACQUIRE) != FFI_ADMISSION_CLOSED ||
        ffi_prepared != token) return false;
    ffi_prepared = NULL;
    __atomic_store_n(&ffi_admission, 0u, __ATOMIC_RELEASE);
    return true;
}

bool ffi_loader_fork_child(FfiLoaderFork *token) {
    unsigned current = (unsigned)getpid();
    if (!token || token->parent_pid == current ||
        __atomic_load_n(&ffi_process, __ATOMIC_ACQUIRE) != token->parent_pid ||
        __atomic_load_n(&ffi_admission, __ATOMIC_ACQUIRE) != FFI_ADMISSION_CLOSED ||
        ffi_prepared != token) return false;
    /* My registry lock was unlocked and had no waiters at the prepared fork.
     * I never unlock, destroy or replace an inherited held pthread lock. */
    ffi_child_local = ffi_child_local || ffi_registered;
    ffi_prepared = NULL;
    __atomic_store_n(&ffi_process, current, __ATOMIC_RELEASE);
    __atomic_store_n(&ffi_admission, 0u, __ATOMIC_RELEASE);
    return true;
}

static bool ffi_register_shutdown(void) {
    if (ffi_child_local) return true;
    if (!nano_native_register_loader_shutdown(ffi_loader_shutdown)) return false;
    ffi_registered = true;
    return true;
}

typedef struct RetainedImage {
    void *handle;
    struct RetainedImage *next;
} RetainedImage;
/* I keep one additional loader reference per retained image, across registry
 * shutdown/reinitialization. A callback's final release can occur while native
 * code is still returning through that image, so it is not an unload boundary. */
static RetainedImage *retained_images;

bool ffi_loader_shadow_prepare(FfiLoaderFork *token) {
    if (!ffi_loader_fork_prepare(token)) return false;
    /* Admission excludes readers and writers while I inspect this history. */
    if (module_count || retained_images || __atomic_load_n(&ffi_native_entered, __ATOMIC_ACQUIRE)) {
        (void)ffi_loader_fork_parent(token);
        return false;
    }
    return true;
}

/* ── Lifecycle ───────────────────────────────────────────────────── */

bool ffi_loader_init(bool verbose) {
    if (!ffi_registry_lock(true)) return false;
    if (!ffi_register_shutdown()) {
        ffi_registry_unlock();
        return false;
    }

    if (initialized) {
        ffi_registry_unlock();
        return true;
    }

    verbose_mode = verbose;
    modules = calloc(FFI_INITIAL_CAPACITY, sizeof(FfiModule));
    if (!modules) {
        fprintf(stderr, "ffi_loader: allocation failed\n");
        ffi_registry_unlock();
        return false;
    }
    module_capacity = FFI_INITIAL_CAPACITY;
    module_count = 0;
    initialized = true;

    if (verbose_mode) {
        fprintf(stderr, "[ffi_loader] Initialized\n");
    }

    ffi_registry_unlock();
    return true;
}

void ffi_loader_shutdown(void) {
    if (!ffi_registry_lock(true)) return;

    if (!initialized) {
        ffi_registry_unlock();
        return;
    }

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

    ffi_registry_unlock();
}

bool ffi_loader_is_initialized(void) {
    if (!ffi_registry_lock(false)) return false;
    bool result = initialized;
    ffi_registry_unlock();
    return result;
}

/* ── Module tracking ─────────────────────────────────────────────── */

FfiModule *ffi_loader_find(const char *module_name) {
    if (!module_name) return NULL;

    if (!ffi_registry_lock(false)) return NULL;

    for (int i = 0; i < module_count; i++) {
        if (strcmp(modules[i].name, module_name) == 0) {
            FfiModule *result = &modules[i];
            ffi_registry_unlock();
            return result;
        }
    }

    ffi_registry_unlock();
    return NULL;
}

bool ffi_loader_open(const char *module_name, const char *lib_path) {
    if (!ffi_registry_lock(true)) return false;
    if (!ffi_register_shutdown()) {
        ffi_registry_unlock();
        return false;
    }

    if (!initialized) {
        /* Init under write lock */
        if (!modules) {
            verbose_mode = false;
            modules = calloc(FFI_INITIAL_CAPACITY, sizeof(FfiModule));
            if (!modules) {
                ffi_registry_unlock();
                return false;
            }
            module_capacity = FFI_INITIAL_CAPACITY;
            module_count = 0;
            initialized = true;
        }
    }

    /* Idempotent check (under lock) */
    for (int i = 0; i < module_count; i++) {
        if (strcmp(modules[i].name, module_name) == 0) {
            ffi_registry_unlock();
            return true;
        }
    }

    /* Grow array if needed */
    if (module_count >= module_capacity) {
        int new_cap = module_capacity * 2;
        FfiModule *new_arr = realloc(modules, (size_t)new_cap * sizeof(FfiModule));
        if (!new_arr) {
            ffi_registry_unlock();
            return false;
        }
        modules = new_arr;
        module_capacity = new_cap;
    }

    /* Use RTLD_GLOBAL so module-to-module symbol deps can resolve */
    void *handle = ffi_native_open(lib_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        if (verbose_mode) {
            fprintf(stderr, "[ffi_loader] Failed to load %s: %s\n",
                    lib_path, dlerror());
        }
        ffi_registry_unlock();
        return false;
    }

    FfiModule *m = &modules[module_count];
    char *name = strdup(module_name);
    char *path = strdup(lib_path);
    if (!name || !path) {
        free(name);
        free(path);
        dlclose(handle);
        ffi_registry_unlock();
        return false;
    }
    m->name = name;
    m->path = path;
    m->handle = handle;
    m->user_data = NULL;
    module_count++;

    if (verbose_mode) {
        fprintf(stderr, "[ffi_loader] Loaded '%s' from %s\n", module_name, lib_path);
    }

    ffi_registry_unlock();
    return true;
}

FfiModule *ffi_loader_get_modules(int *out_count) {
    /* My returned borrow still requires caller serialization after return. */
    if (out_count) *out_count = 0;
    if (!ffi_registry_lock(false)) return NULL;
    if (out_count) *out_count = module_count;
    FfiModule *result = modules;
    ffi_registry_unlock();
    return result;
}

/* ── Symbol resolution ───────────────────────────────────────────── */

void *ffi_loader_resolve(const char *symbol_name) {
    return ffi_loader_resolve_in(symbol_name, NULL);
}

void *ffi_loader_resolve_module(const char *symbol_name, const char *module_name) {
    if (!symbol_name || !module_name) return NULL;
    void *ptr = NULL;
    if (!ffi_registry_lock(false)) return NULL;
    for (int i = 0; i < module_count; i++) {
        if (strcmp(modules[i].name, module_name) == 0) {
            ptr = ffi_native_symbol(modules[i].handle, symbol_name);
            break;
        }
    }
    ffi_registry_unlock();
    return ptr;
}

bool ffi_loader_string_release(const char *module_name, const char *symbol_name,
                               void *function, void (**release)(const char *),
                               char *error, size_t error_size) {
    if (!release) return false;
    *release = NULL;
    if (!module_name || !symbol_name || !function) return false;
    const char suffix[] = "__nano_string_release_v1";
    size_t length = strlen(symbol_name);
    if (length > SIZE_MAX - sizeof suffix) return false;
    char *name = malloc(length + sizeof suffix);
    if (!name) {
        if (error && error_size) snprintf(error, error_size, "I could not allocate the string cleanup symbol");
        return false;
    }
    memcpy(name, symbol_name, length);
    memcpy(name + length, suffix, sizeof suffix);
    bool valid = false;
    if (!ffi_registry_lock(false)) { free(name); return false; }
    for (int i = 0; i < module_count; ++i) {
        if (strcmp(modules[i].name, module_name)) continue;
        void *cleanup = ffi_native_symbol(modules[i].handle, name);
        if (!cleanup) { valid = true; break; }
        Dl_info origin, companion;
        if (dladdr(function, &origin) && dladdr(cleanup, &companion) &&
            origin.dli_fbase == companion.dli_fbase) {
            *release = (void (*)(const char *))cleanup;
            valid = true;
        }
        break;
    }
    ffi_registry_unlock();
    free(name);
    if (!valid && error && error_size)
        snprintf(error, error_size, "I require string cleanup from the called function's own image");
    return valid;
}

bool ffi_loader_check_array_abi(const char *module_name, const char *symbol_name,
                                void *function, uint32_t expected,
                                char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!symbol_name || !function) return false;
    const char suffix[] = "__nano_array_abi";
    size_t length = strlen(symbol_name);
    if (length > SIZE_MAX - sizeof suffix) return false;
    char *name = malloc(length + sizeof suffix);
    if (!name) return false;
    memcpy(name, symbol_name, length);
    memcpy(name + length, suffix, sizeof suffix);
    bool found = false, valid = false;
    uint32_t actual = 1;
    if (!ffi_registry_lock(false)) { free(name); return false; }
    const uint32_t *declaration = NULL;
    if (!module_name) {
        found = true;
        declaration = ffi_native_symbol(RTLD_DEFAULT, name);
    }
    for (int i = 0; module_name && i < module_count; ++i) {
        if (strcmp(modules[i].name, module_name)) continue;
        found = true;
        declaration = ffi_native_symbol(modules[i].handle, name);
        break;
    }
    if (found) {
        if (!declaration) {
            valid = expected == 1;
        } else {
            Dl_info function_image, declaration_image;
            if (dladdr(function, &function_image) && dladdr(declaration, &declaration_image) &&
                function_image.dli_fbase == declaration_image.dli_fbase) {
                memcpy(&actual, declaration, sizeof actual);
                valid = actual == expected;
            }
        }
    }
    ffi_registry_unlock();
    free(name);
    if (!valid && error && error_size)
        snprintf(error, error_size,
                 "I require native array ABI %u for %s; its declaration is missing, incompatible or belongs to another image%s",
                 expected, symbol_name, found ? "" : " (module not loaded)");
    return valid;
}

void *ffi_loader_resolve_retained(const char *symbol_name, const char *module_name) {
    if (!symbol_name || !module_name) return NULL;
    void *ptr = NULL;
    if (!ffi_registry_lock(true)) return NULL;
    for (int i = 0; i < module_count; i++) {
        FfiModule *module = &modules[i];
        if (strcmp(module->name, module_name)) continue;
        ptr = ffi_native_symbol(module->handle, symbol_name);
        if (!ptr) break;
        RetainedImage *image = retained_images;
        while (image && image->handle != module->handle) image = image->next;
        if (!image) {
            image = malloc(sizeof(*image));
            if (!image) { ptr = NULL; break; }
            image->handle = ffi_native_open(module->path, RTLD_LAZY | RTLD_GLOBAL);
            if (image->handle != module->handle) {
                if (image->handle) dlclose(image->handle);
                free(image);
                ptr = NULL;
                break;
            }
            image->next = retained_images;
            retained_images = image;
        }
        break;
    }
    ffi_registry_unlock();
    return ptr;
}

void *ffi_loader_resolve_in(const char *symbol_name, FfiModule **out_module) {
    if (out_module) *out_module = NULL;

    if (!ffi_registry_lock(false)) return NULL;

    /* Search loaded modules */
    for (int i = 0; i < module_count; i++) {
        void *ptr = ffi_native_symbol(modules[i].handle, symbol_name);
        if (ptr) {
            if (out_module) *out_module = &modules[i];
            ffi_registry_unlock();
            return ptr;
        }
    }

    /* I mark native entry before releasing admission, so a concurrent shadow
     * preparation cannot slip between this registry scan and native fallback. */
    __atomic_store_n(&ffi_native_entered, 1u, __ATOMIC_RELEASE);
    ffi_registry_unlock();

    /* Fallback: main executable + already-loaded libraries (no lock needed) */
    void *self = ffi_native_open(NULL, RTLD_LAZY);
    if (self) {
        void *ptr = ffi_native_symbol(self, symbol_name);
        dlclose(self);
        if (ptr) return ptr;
    }

    return NULL;
}

/* ── Library search ──────────────────────────────────────────────── */

bool ffi_loader_find_library(const char *module_name, const char *module_dir,
                             char *out_path, size_t path_size) {
    if (!module_name || !out_path || path_size == 0) return false;

    /* Normalize: strip "modules/" prefix and ".nano" suffix */
    char normalized[512];
    const char *mn = module_name;
    /* Strip any leading "./" segments (module paths may arrive as
     * "./modules/std/fs.nano" from the VM import table). */
    while (mn[0] == '.' && mn[1] == '/') mn += 2;
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
        char bdir[1024];
        if (nano_module_artifact_dir(module_dir, bdir, sizeof(bdir))) {
            for (int ei = 0; exts[ei]; ei++) {
                snprintf(out_path, path_size, "%s/lib%s.%s",
                         bdir, lib_name, exts[ei]);
                if (access(out_path, F_OK) == 0) return true;
            }
        }
    }

    for (int ei = 0; exts[ei]; ei++) {
        char logical[1024];
        char bdir[1024];

        /* Pattern 2: modules/<full_normalized> */
        snprintf(logical, sizeof(logical), "modules/%s", mn);
        if (nano_module_artifact_dir(logical, bdir, sizeof(bdir))) {
            snprintf(out_path, path_size, "%s/lib%s.%s",
                     bdir, lib_name, exts[ei]);
            if (access(out_path, F_OK) == 0) return true;
        }

        /* Pattern 3: modules/<parent_dir> with joined lib name */
        if (parent_dir[0]) {
            snprintf(logical, sizeof(logical), "modules/%s", parent_dir);
            if (nano_module_artifact_dir(logical, bdir, sizeof(bdir))) {
                snprintf(out_path, path_size, "%s/lib%s.%s",
                         bdir, joined_name, exts[ei]);
                if (access(out_path, F_OK) == 0) return true;
            }
        }

        /* Pattern 4: modules/<top_dir> */
        if (top_dir[0]) {
            snprintf(logical, sizeof(logical), "modules/%s", top_dir);
            if (nano_module_artifact_dir(logical, bdir, sizeof(bdir))) {
                snprintf(out_path, path_size, "%s/lib%s.%s",
                         bdir, top_dir, exts[ei]);
                if (access(out_path, F_OK) == 0) return true;
            }
        }
    }

    return false;
}
