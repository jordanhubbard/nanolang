/**
 * @file stdlib_runtime.h
 * @brief Standard library runtime code generation for nanolang transpiler
 *
 * Provides functions to generate C code for nanolang's standard library operations
 * including string manipulation, file I/O, directory operations, and math utilities.
 */

#ifndef STDLIB_RUNTIME_H
#define STDLIB_RUNTIME_H

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

#endif /* STDLIB_RUNTIME_H */

