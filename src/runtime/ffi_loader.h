/**
 * ffi_loader.h - Unified FFI Module Loading
 *
 * Shared dlopen/dlsym logic used by both the interpreter (interpreter_ffi.c)
 * and the VM (vm_ffi.c). Each caller handles its own marshaling; this layer
 * manages module tracking, library search, and symbol resolution.
 */

#ifndef FFI_LOADER_H
#define FFI_LOADER_H

#include <stdbool.h>
#include <stddef.h>

/**
 * Loaded module handle (opaque to callers that just need symbol resolution).
 * Callers that need extra per-module data (e.g., interpreter's ModuleBuildMetadata)
 * can store it in user_data.
 */
typedef struct {
    char *name;
    char *path;       /* filesystem path of loaded .so/.dylib */
    void *handle;     /* dlopen handle */
    void *user_data;  /* caller-owned extra data (e.g., metadata) */
} FfiModule;

/**
 * Initialize the FFI loader.
 * Safe to call multiple times; subsequent calls are no-ops.
 * @param verbose  Enable diagnostic messages to stderr.
 * @return true on success.
 */
bool ffi_loader_init(bool verbose);

/**
 * Shut down the FFI loader and dlclose all modules.
 * Does NOT free user_data — callers must clean up their own data first
 * via ffi_loader_get_modules().
 */
void ffi_loader_shutdown(void);

/** @return true if the loader has been initialized. */
bool ffi_loader_is_initialized(void);

/**
 * Open a shared library at an explicit filesystem path and register it
 * under @p module_name. If a module with this name is already loaded,
 * returns true immediately (idempotent).
 *
 * @return true if the module is now loaded (either freshly or already).
 */
bool ffi_loader_open(const char *module_name, const char *lib_path);

/**
 * Look up a module by name.
 * @return pointer to the FfiModule entry, or NULL.
 */
FfiModule *ffi_loader_find(const char *module_name);

/**
 * Resolve a symbol by searching (in order):
 *   1. All loaded modules (most-recently-loaded first)
 *   2. The main executable + already-loaded libraries (RTLD_DEFAULT)
 *
 * @return function pointer, or NULL if not found.
 */
void *ffi_loader_resolve(const char *symbol_name);

/**
 * Resolve a symbol, returning which module it was found in (or NULL
 * if found via RTLD_DEFAULT / main executable).
 */
void *ffi_loader_resolve_in(const char *symbol_name, FfiModule **out_module);

/**
 * Access the loaded module array (for callers that need to iterate,
 * e.g., to free user_data before shutdown).
 * @param out_count  receives the number of loaded modules.
 * @return pointer to internal array (valid until next open/shutdown call).
 */
FfiModule *ffi_loader_get_modules(int *out_count);

/**
 * Utility: try to find a shared library for a module using the standard
 * nanolang search paths.  Writes the found path into @p out_path.
 *
 * Search order (for each extension .dylib then .so):
 *   1. <module_dir>/.build/lib<leaf>.<ext>    (if module_dir provided)
 *   2. modules/<normalized>/.build/lib<leaf>.<ext>
 *   3. modules/<parent_dir>/.build/lib<joined>.<ext>
 *   4. modules/<top_dir>/.build/lib<top_dir>.<ext>
 *
 * @param module_name   Raw module name (may include "modules/" prefix, ".nano" suffix)
 * @param module_dir    Optional explicit directory (from interpreter's module path).
 *                      Pass NULL to skip pattern 1.
 * @param out_path      Buffer to receive the found path.
 * @param path_size     Size of out_path buffer.
 * @return true if a library file was found on disk.
 */
bool ffi_loader_find_library(const char *module_name, const char *module_dir,
                             char *out_path, size_t path_size);

#endif /* FFI_LOADER_H */
