/**
 * nl_string.h - Nanolang String Type
 * 
 * Length-explicit string structure supporting:
 * - Binary data (embedded nulls)
 * - UTF-8 validation and operations
 * - Efficient memory management
 * - C string interoperability
 */

#ifndef NL_STRING_H
#define NL_STRING_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * String structure with explicit length
 */
typedef struct {
    char *data;         // Raw byte data (may contain nulls)
    size_t length;      // Byte length (not including null terminator if present)
    size_t capacity;    // Allocated capacity
    bool is_utf8;       // True if validated as UTF-8
    bool null_terminated; // True if data has '\0' at data[length]
} nl_string_t;

/* ============================================================================
 * Creation & Destruction
 * ============================================================================ */

/**
 * Create string from C string (null-terminated)
 * Copies the data and validates UTF-8.
 */
nl_string_t* nl_string_new(const char *cstr);

/**
 * Create string from binary data with explicit length
 * No UTF-8 validation performed.
 */
nl_string_t* nl_string_new_binary(const void *data, size_t length);

/**
 * Create string from UTF-8 data with validation
 * Returns NULL if invalid UTF-8.
 */
nl_string_t* nl_string_from_utf8(const char *utf8_data, size_t length);

/**
 * Create empty string with given capacity
 */
nl_string_t* nl_string_with_capacity(size_t capacity);

/**
 * Free string and its data
 */
void nl_string_free(nl_string_t *str);

/* ============================================================================
 * Basic Operations
 * ============================================================================ */

/**
 * Get byte length
 */
size_t nl_string_length(const nl_string_t *str);

/**
 * Get byte at index (no bounds checking for performance)
 */
char nl_string_byte_at(const nl_string_t *str, size_t index);

/**
 * Get byte at index with bounds checking
 */
bool nl_string_byte_at_safe(const nl_string_t *str, size_t index, char *out);

/**
 * Concatenate two strings
 * Returns new string, originals unchanged.
 */
nl_string_t* nl_string_concat(const nl_string_t *a, const nl_string_t *b);

/**
 * Extract substring [start, start+length)
 * Returns new string.
 */
nl_string_t* nl_string_substring(const nl_string_t *str, size_t start, size_t length);

/**
 * Compare two strings for equality
 */
bool nl_string_equals(const nl_string_t *a, const nl_string_t *b);

/**
 * Compare string with C string
 */
bool nl_string_equals_cstr(const nl_string_t *str, const char *cstr);

/* ============================================================================
 * UTF-8 Operations
 * ============================================================================ */

/**
 * Validate UTF-8 encoding
 * Returns true if valid, updates str->is_utf8 flag.
 */
bool nl_string_validate_utf8(nl_string_t *str);

/**
 * Get UTF-8 character count (not byte count)
 * Only works if is_utf8 is true.
 * Returns -1 if not UTF-8.
 */
int64_t nl_string_utf8_length(const nl_string_t *str);

/**
 * Get UTF-8 codepoint at character index
 * Returns codepoint or -1 on error.
 */
int32_t nl_string_utf8_char_at(const nl_string_t *str, size_t char_index);

/**
 * UTF-8 substring by character positions
 */
nl_string_t* nl_string_utf8_substring(const nl_string_t *str, size_t char_start, size_t char_length);

/* ============================================================================
 * Conversion & Interoperability
 * ============================================================================ */

/**
 * Get as null-terminated C string
 * Returns internal buffer if already null-terminated,
 * otherwise creates temporary null-terminated copy.
 * WARNING: Pointer may become invalid after string modification!
 */
const char* nl_string_to_cstr(nl_string_t *str);

/**
 * Get raw binary data and length
 */
const void* nl_string_to_binary(const nl_string_t *str, size_t *out_length);

/**
 * Ensure string is null-terminated (for FFI)
 * Adds '\0' at str->data[str->length] if not present.
 */
void nl_string_ensure_null_terminated(nl_string_t *str);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * Reserve capacity (for efficient concatenation)
 */
void nl_string_reserve(nl_string_t *str, size_t new_capacity);

/**
 * Shrink capacity to fit current length
 */
void nl_string_shrink_to_fit(nl_string_t *str);

/**
 * Clone string (deep copy)
 */
nl_string_t* nl_string_clone(const nl_string_t *str);

/* ============================================================================
 * C String Convenience Wrappers
 *
 * These operate on plain `const char *` strings, delegating to the
 * nl_string_t implementation internally.  They provide a single shared
 * implementation that the interpreter (eval_string.c), VM (vm.c), and
 * VM builtins (vm_builtins.c) can all call, eliminating duplication.
 * ============================================================================ */

/** Byte length of a C string (NULL-safe, returns 0 for NULL). */
int64_t     nl_cstr_length(const char *s);

/** Concatenate two C strings. Caller must free() the result. */
char       *nl_cstr_concat(const char *a, const char *b);

/** Substring [start, start+len). Caller must free() the result. */
char       *nl_cstr_substring(const char *s, int64_t start, int64_t len);

/** True if haystack contains needle. */
bool        nl_cstr_contains(const char *haystack, const char *needle);

/** Index of needle in haystack, or -1. */
int64_t     nl_cstr_index_of(const char *haystack, const char *needle);

/** ASCII code of character at byte index, or -1 if out of range. */
int64_t     nl_cstr_char_at(const char *s, int64_t index);

/** Single-character string from ASCII code. Caller must free(). */
char       *nl_cstr_from_char(int64_t code);

#endif /* NL_STRING_H */
