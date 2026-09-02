/**
 * @file stdlib_runtime.h
 * @brief Standard library runtime code generation for nanolang transpiler
 *
 * Provides functions to generate C code for nanolang's standard library operations
 * including string manipulation, file I/O, directory operations, and math utilities.
 */

#ifndef STDLIB_RUNTIME_H
#define STDLIB_RUNTIME_H

#include <stdbool.h>

/**
 * @brief String builder for efficient C code generation
 *
 * Dynamically growing buffer used throughout the transpiler to accumulate
 * generated C code without repeated reallocations.
 */
typedef struct {
    char *buffer;      /**< Dynamically allocated string buffer */
    int length;        /**< Current string length (excluding null terminator) */
    int capacity;      /**< Allocated buffer capacity */
} StringBuilder;

/**
 * @brief Create a new string builder
 * @return Pointer to newly allocated StringBuilder, or NULL on allocation failure
 */
StringBuilder *sb_create(void);

/**
 * @brief Append string to builder
 * @param sb StringBuilder to append to (must not be NULL)
 * @param str String to append (must be null-terminated, must not be NULL)
 *
 * Automatically grows buffer if needed. Performance: amortized O(1).
 */
void sb_append(StringBuilder *sb, const char *str);

/**
 * @brief Generate C code for string operations (concat, substring, length, etc.)
 * @param sb StringBuilder to append generated code to
 */
void generate_string_operations(StringBuilder *sb);

/**
 * @brief Generate C code for file I/O operations (read, write, exists, etc.)
 * @param sb StringBuilder to append generated code to
 */
void generate_file_operations(StringBuilder *sb);

/**
 * @brief Generate C code for directory operations (list, create, remove, etc.)
 * @param sb StringBuilder to append generated code to
 */
void generate_dir_operations(StringBuilder *sb);

/**
 * @brief Generate C code for path operations (join, dirname, basename, etc.)
 * @param sb StringBuilder to append generated code to
 */
void generate_path_operations(StringBuilder *sb);

/**
 * @brief Generate C code for math utility built-ins (abs, min, max, etc.)
 * @param sb StringBuilder to append generated code to
 */
void generate_math_utility_builtins(StringBuilder *sb);

/**
 * @brief Generate complete standard library runtime code
 * @param sb StringBuilder to append all generated code to
 *
 * Convenience function that calls all generate_*() functions in sequence
 * to produce a complete standard library implementation in C.
 */
void generate_stdlib_runtime(StringBuilder *sb);

/* generate_module_system_stubs: Generate stub implementations for module functions
 *
 * Provides fallback implementations when module system isn't linked.
 */
void generate_module_system_stubs(StringBuilder *sb);

/**
 * @brief Generate C code for timing utilities (microseconds, nanoseconds)
 * @param sb StringBuilder to append generated code to
 */
void generate_timing_utilities(StringBuilder *sb);

/**
 * @brief Generate C code for console I/O utilities (readline, etc.)
 * @param sb StringBuilder to append generated code to
 */
void generate_console_io_utilities(StringBuilder *sb);

/**
 * @brief Generate C code for cross-platform profiling system
 * @param sb StringBuilder to append generated code to
 *
 * When profiling is enabled (-pg flag), this generates code that:
 * - macOS: Uses 'sample' command on child process
 * - Linux: Uses 'gprofng collect' to wrap execution
 * Both output OS-neutral JSON analysis of performance hotspots
 */
void generate_profiling_system(StringBuilder *sb, const char *profile_output_path);

/**
 * @brief Generate instrumented profiling runtime for --profile flag
 * @param sb StringBuilder to append generated code to
 *
 * Injects clock_gettime-based timing accumulators into each function.
 * At program exit (via atexit), prints a hotspot table sorted by total time.
 */
/**
 * @brief Generate the shared diagnostics hook mechanism (tracing + profiling).
 * @param sb StringBuilder to append generated code to
 * @param want_profile Emit the profiling enable flag / accessor
 * @param want_trace   Emit the tracing enable flag, accessor, and trace hooks
 *
 * Both generated-C tracing (--trace) and profiling (--profile) share one
 * mechanism whose enable flags are resolved exactly once at process startup via
 * _nl_diag_init(). Disabled hooks read only a cached int, so they perform no
 * per-event environment lookups and no other work.
 */
void generate_diagnostics_runtime(StringBuilder *sb, bool want_profile, bool want_trace);

void generate_instrumented_profiling_system(StringBuilder *sb);
void generate_flamegraph_profiling_system(StringBuilder *sb, const char *flamegraph_path);

/**
 * @brief Resolve the effective backend name for a compilation
 * @param target Value of --target, or NULL when the flag was not given
 * @param llvm   True when --llvm was given
 * @return Static string naming the backend ("native", "wasm", "llvm", ...)
 *
 * --llvm short-circuits the transpiler in the same way an explicit --target
 * does, so it is reported as its own backend rather than as "native".
 */
const char *profile_runtime_backend_name(const char *target, bool llvm);

/**
 * @brief Can this backend emit a runtime flamegraph profile?
 * @param backend Backend name from profile_runtime_backend_name()
 * @return true only for backends that reach the transpiler's instrumentation
 *
 * --profile-runtime works by injecting clock_gettime counters into the C the
 * transpiler emits and registering an atexit hook that writes the collapsed
 * stack file. Every other backend (wasm, ptx, opencl, c, riscv, llvm) returns
 * from main() before that code generation runs and emits an artifact that the
 * compiler never executes, so none of them can produce a .nano.prof.
 */
bool profile_runtime_backend_supported(const char *backend);

void generate_coroutine_builtins(StringBuilder *sb);

#endif /* STDLIB_RUNTIME_H */
