#include "stdlib_runtime.h"
#include "nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* sb_append is defined in transpiler.c */
extern void sb_append(StringBuilder *sb, const char *str);

/* Generate math and utility built-in functions */
void generate_math_utility_builtins(StringBuilder *sb) {
    sb_append(sb, "/* ========== Math and Utility Built-in Functions ========== */\n\n");

    /* abs function - works with int and float via macro */
    sb_append(sb, "#define nl_abs(x) _Generic((x), \\\n");
    sb_append(sb, "    double: (double)((x) < 0.0 ? -(x) : (x)), \\\n");
    sb_append(sb, "    default: (int64_t)((x) < 0 ? -(x) : (x)))\n\n");

    /* min function */
    sb_append(sb, "#define nl_min(a, b) _Generic((a), \\\n");
    sb_append(sb, "    double: (double)((a) < (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    default: (int64_t)((a) < (b) ? (a) : (b)))\n\n");

    /* max function */
    sb_append(sb, "#define nl_max(a, b) _Generic((a), \\\n");
    sb_append(sb, "    double: (double)((a) > (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    default: (int64_t)((a) > (b) ? (a) : (b)))\n\n");

    /* Math functions - wrappers around C standard library math.h */
    sb_append(sb, "/* Trigonometric functions */\n");
    sb_append(sb, "static double nl_sin(double x) { return sin(x); }\n");
    sb_append(sb, "static double nl_cos(double x) { return cos(x); }\n");
    sb_append(sb, "static double nl_tan(double x) { return tan(x); }\n");
    sb_append(sb, "static double nl_atan2(double y, double x) { return atan2(y, x); }\n\n");
    
    sb_append(sb, "/* Power and root functions */\n");
    sb_append(sb, "static double nl_sqrt(double x) { return sqrt(x); }\n");
    sb_append(sb, "static double nl_pow(double base, double exp) { return pow(base, exp); }\n\n");
    
    sb_append(sb, "/* Rounding functions */\n");
    sb_append(sb, "static double nl_floor(double x) { return floor(x); }\n");
    sb_append(sb, "static double nl_ceil(double x) { return ceil(x); }\n");
    sb_append(sb, "static double nl_round(double x) { return round(x); }\n\n");

    /* Type casting functions */
    sb_append(sb, "static int64_t nl_cast_int(double x) { return (int64_t)x; }\n");
    sb_append(sb, "static int64_t nl_cast_int_from_int(int64_t x) { return x; }\n");
    sb_append(sb, "static double nl_cast_float(int64_t x) { return (double)x; }\n");
    sb_append(sb, "static double nl_cast_float_from_float(double x) { return x; }\n");
    sb_append(sb, "static void* nl_null_opaque() { return NULL; }\n");
    sb_append(sb, "static int64_t nl_cast_bool_to_int(bool x) { return x ? 1 : 0; }\n");
    sb_append(sb, "static bool nl_cast_bool(int64_t x) { return x != 0; }\n\n");

    /* println function - uses _Generic for type dispatch */
    sb_append(sb, "static void nl_println(void* value_ptr) {\n");
    sb_append(sb, "    (void)value_ptr; /* Unused - actual implementation uses type info from checker */\n");
    sb_append(sb, "}\n\n");

    /* Specialized print functions for each type (no newline) */
    sb_append(sb, "static void nl_print_int(int64_t value) {\n");
    sb_append(sb, "    printf(\"%lld\", (long long)value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_float(double value) {\n");
    sb_append(sb, "    printf(\"%g\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_string(const char* value) {\n");
    sb_append(sb, "    printf(\"%s\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_bool(bool value) {\n");
    sb_append(sb, "    printf(value ? \"true\" : \"false\");\n");
    sb_append(sb, "}\n\n");

    /* Specialized println functions for each type */
    sb_append(sb, "static void nl_println_int(int64_t value) {\n");
    sb_append(sb, "    printf(\"%lld\\n\", (long long)value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_float(double value) {\n");
    sb_append(sb, "    printf(\"%g\\n\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_string(const char* value) {\n");
    sb_append(sb, "    printf(\"%s\\n\", value);\n");
    sb_append(sb, "}\n\n");
    
    /* Dynamic array runtime - LEGACY (for old array<T> type) */
    sb_append(sb, "/* Dynamic array runtime (using GC) - LEGACY */\n");
    sb_append(sb, "#include \"runtime/gc.h\"\n");
    sb_append(sb, "#include \"runtime/dyn_array.h\"\n");
    sb_append(sb, "#include \"runtime/nl_string.h\"\n\n");
    
    /* Array literals create dynamic arrays - renamed to avoid conflicts */
    sb_append(sb, "static DynArray* dynarray_literal_int(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        int64_t val = va_arg(args, int64_t);\n");
    sb_append(sb, "        dyn_array_push_int(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* dynarray_literal_u8(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_U8);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        int val = va_arg(args, int); /* default promotion */\n");
    sb_append(sb, "        dyn_array_push_u8(arr, (uint8_t)val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* dynarray_literal_float(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        double val = va_arg(args, double);\n");
    sb_append(sb, "        dyn_array_push_float(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* dynarray_literal_string(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        const char* val = va_arg(args, const char*);\n");
    sb_append(sb, "        dyn_array_push_string(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* dynarray_literal_bool(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_BOOL);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        int val = va_arg(args, int); /* bool promotes to int */\n");
    sb_append(sb, "        dyn_array_push_bool(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    /* Array operations - renamed to avoid conflicts */
    sb_append(sb, "static DynArray* dynarray_push(DynArray* arr, double val) {\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_U8) {\n");
    sb_append(sb, "        return dyn_array_push_u8(arr, (uint8_t)val);\n");
    sb_append(sb, "    } else if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return dyn_array_push_int(arr, (int64_t)val);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_push_float(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    /* Wrappers for array push/pop that work with GC dynamic arrays */
    sb_append(sb, "static DynArray* nl_array_push(DynArray* arr, double val) {\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_U8) {\n");
    sb_append(sb, "        return dyn_array_push_u8(arr, (uint8_t)val);\n");
    sb_append(sb, "    } else if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return dyn_array_push_int(arr, (int64_t)val);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_push_float(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double nl_array_pop(DynArray* arr) {\n");
    sb_append(sb, "    bool success = false;\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_U8) {\n");
    sb_append(sb, "        return (double)dyn_array_pop_u8(arr, &success);\n");
    sb_append(sb, "    } else if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return (double)dyn_array_pop_int(arr, &success);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_pop_float(arr, &success);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    /* Array length wrapper */
    sb_append(sb, "static int64_t nl_array_length(DynArray* arr) {\n");
    sb_append(sb, "    return dyn_array_length(arr);\n");
    sb_append(sb, "}\n\n");
    
    /* Array remove_at wrapper */
    sb_append(sb, "static DynArray* nl_array_remove_at(DynArray* arr, int64_t index) {\n");
    sb_append(sb, "    return dyn_array_remove_at(arr, index);\n");
    sb_append(sb, "}\n\n");
    
    /* Array get (at) wrapper */
    sb_append(sb, "static int64_t nl_array_at_int(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_int(arr, idx);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static uint8_t nl_array_at_u8(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_u8(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double nl_array_at_float(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_float(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static const char* nl_array_at_string(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_string(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool nl_array_at_bool(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_bool(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    /* Array set wrapper */
    sb_append(sb, "static void nl_array_set_int(DynArray* arr, int64_t idx, int64_t val) {\n");
    sb_append(sb, "    dyn_array_set_int(arr, idx, val);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_array_set_u8(DynArray* arr, int64_t idx, uint8_t val) {\n");
    sb_append(sb, "    dyn_array_set_u8(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_float(DynArray* arr, int64_t idx, double val) {\n");
    sb_append(sb, "    dyn_array_set_float(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_string(DynArray* arr, int64_t idx, const char* val) {\n");
    sb_append(sb, "    dyn_array_set_string(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_bool(DynArray* arr, int64_t idx, bool val) {\n");
    sb_append(sb, "    dyn_array_set_bool(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    /* Nested array support */
    sb_append(sb, "static DynArray* nl_array_at_array(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_array(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_array(DynArray* arr, int64_t idx, DynArray* val) {\n");
    sb_append(sb, "    dyn_array_set_array(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    /* Array new wrapper - creates DynArray with specified size and default value */
    sb_append(sb, "static DynArray* nl_array_new_int(int64_t size, int64_t default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_int(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* nl_array_new_float(int64_t size, double default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_float(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* nl_array_new_string(int64_t size, const char* default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_string(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* nl_array_new_bool(int64_t size, bool default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_BOOL);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_bool(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t dynarray_length(DynArray* arr) {\n");
    sb_append(sb, "    return dyn_array_length(arr);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double dynarray_at_for_transpiler(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_U8) {\n");
    sb_append(sb, "        return (double)dyn_array_get_u8(arr, idx);\n");
    sb_append(sb, "    } else if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return (double)dyn_array_get_int(arr, idx);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_get_float(arr, idx);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    /* bstring helpers (nl_string_t wrappers) */
    sb_append(sb, "/* bstring helpers (nl_string_t wrappers) */\n");
    sb_append(sb, "static nl_string_t* bstr_new(const char* cstr) {\n");
    sb_append(sb, "    if (!cstr) cstr = \"\";\n");
    sb_append(sb, "    return nl_string_new(cstr);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static nl_string_t* bstr_new_binary(DynArray* bytes) {\n");
    sb_append(sb, "    if (!bytes || dyn_array_get_elem_type(bytes) != ELEM_U8) {\n");
    sb_append(sb, "        return nl_string_new_binary(\"\", 0);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    int64_t len = dyn_array_length(bytes);\n");
    sb_append(sb, "    if (len <= 0) {\n");
    sb_append(sb, "        return nl_string_new_binary(\"\", 0);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    uint8_t* buffer = malloc((size_t)len);\n");
    sb_append(sb, "    if (!buffer) {\n");
    sb_append(sb, "        return nl_string_new_binary(\"\", 0);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    for (int64_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        buffer[i] = dyn_array_get_u8(bytes, i);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    nl_string_t* result = nl_string_new_binary(buffer, (size_t)len);\n");
    sb_append(sb, "    free(buffer);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static size_t bstr_length(nl_string_t* str) {\n");
    sb_append(sb, "    if (!str) return 0;\n");
    sb_append(sb, "    return nl_string_length(str);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t bstr_byte_at(nl_string_t* str, int64_t index) {\n");
    sb_append(sb, "    if (!str || index < 0 || (size_t)index >= nl_string_length(str)) {\n");
    sb_append(sb, "        return 0;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return (unsigned char)nl_string_byte_at(str, (size_t)index);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static nl_string_t* bstr_concat(nl_string_t* a, nl_string_t* b) {\n");
    sb_append(sb, "    if (!a && !b) return nl_string_new(\"\");\n");
    sb_append(sb, "    if (!a) return nl_string_clone(b);\n");
    sb_append(sb, "    if (!b) return nl_string_clone(a);\n");
    sb_append(sb, "    return nl_string_concat(a, b);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static nl_string_t* bstr_substring(nl_string_t* str, int64_t start, int64_t length) {\n");
    sb_append(sb, "    if (!str || start < 0 || length < 0) {\n");
    sb_append(sb, "        return nl_string_new(\"\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    size_t len = nl_string_length(str);\n");
    sb_append(sb, "    if ((size_t)start > len) {\n");
    sb_append(sb, "        start = (int64_t)len;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    if ((size_t)(start + length) > len) {\n");
    sb_append(sb, "        length = (int64_t)len - start;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return nl_string_substring(str, (size_t)start, (size_t)length);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool bstr_equals(nl_string_t* a, nl_string_t* b) {\n");
    sb_append(sb, "    if (!a || !b) return a == b;\n");
    sb_append(sb, "    return nl_string_equals(a, b);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool bstr_validate_utf8(nl_string_t* str) {\n");
    sb_append(sb, "    if (!str) return false;\n");
    sb_append(sb, "    return nl_string_validate_utf8(str);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t bstr_utf8_length(nl_string_t* str) {\n");
    sb_append(sb, "    if (!str) return 0;\n");
    sb_append(sb, "    return nl_string_utf8_length(str);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t bstr_utf8_char_at(nl_string_t* str, int64_t char_index) {\n");
    sb_append(sb, "    if (!str || char_index < 0) return -1;\n");
    sb_append(sb, "    return nl_string_utf8_char_at(str, (size_t)char_index);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static const char* bstr_to_cstr(nl_string_t* str) {\n");
    sb_append(sb, "    if (!str) return \"\";\n");
    sb_append(sb, "    return nl_string_to_cstr(str);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void bstr_free(nl_string_t* str) {\n");
    sb_append(sb, "    if (str) {\n");
    sb_append(sb, "        nl_string_free(str);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");

    /* String operations */
    sb_append(sb, "/* String concatenation - use strnlen for safety */\n");
    sb_append(sb, "static const char* nl_str_concat(const char* s1, const char* s2) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to 1MB */\n");
    sb_append(sb, "    size_t len1 = strnlen(s1, 1024*1024);\n");
    sb_append(sb, "    size_t len2 = strnlen(s2, 1024*1024);\n");
    sb_append(sb, "    char* result = gc_alloc_string(len1 + len2);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, s1, len1);\n");
    sb_append(sb, "    memcpy(result + len1, s2, len2);\n");
    sb_append(sb, "    result[len1 + len2] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String substring - use strnlen for safety */\n");
    sb_append(sb, "static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to 1MB */\n");
    sb_append(sb, "    int64_t str_len = strnlen(str, 1024*1024);\n");
    sb_append(sb, "    if (start < 0 || start >= str_len || length < 0) return \"\";\n");
    sb_append(sb, "    if (start + length > str_len) length = str_len - start;\n");
    sb_append(sb, "    char* result = gc_alloc_string(length);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    strncpy(result, str + start, length);\n");
    sb_append(sb, "    result[length] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String contains */\n");
    sb_append(sb, "static bool nl_str_contains(const char* str, const char* substr) {\n");
    sb_append(sb, "    return strstr(str, substr) != NULL;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String equals */\n");
    sb_append(sb, "static bool nl_str_equals(const char* s1, const char* s2) {\n");
    sb_append(sb, "    return strcmp(s1, s2) == 0;\n");
    sb_append(sb, "}\n\n");

    /* Bytes (array<u8>) helpers */
    sb_append(sb, "static DynArray* nl_bytes_from_string(const char* s) {\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(ELEM_U8);\n");
    sb_append(sb, "    if (!out) return NULL;\n");
    sb_append(sb, "    if (!s) return out;\n");
    sb_append(sb, "    size_t len = strnlen(s, 1024*1024);\n");
    sb_append(sb, "    for (size_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        dyn_array_push_u8(out, (uint8_t)(unsigned char)s[i]);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static const char* nl_string_from_bytes(DynArray* bytes) {\n");
    sb_append(sb, "    if (!bytes) return \"\";\n");
    sb_append(sb, "    if (dyn_array_get_elem_type(bytes) != ELEM_U8) return \"\";\n");
    sb_append(sb, "    int64_t len = dyn_array_length(bytes);\n");
    sb_append(sb, "    if (len < 0) return \"\";\n");
    sb_append(sb, "    char* out = gc_alloc_string((size_t)len);\n");
    sb_append(sb, "    if (!out) return \"\";\n");
    sb_append(sb, "    for (int64_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        out[i] = (char)dyn_array_get_u8(bytes, i);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    out[len] = '\\0';\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    /* Array slicing: returns a copy of [start, start+length) */
    sb_append(sb, "static DynArray* nl_array_slice(DynArray* arr, int64_t start, int64_t length) {\n");
    sb_append(sb, "    if (!arr) return dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    if (start < 0) start = 0;\n");
    sb_append(sb, "    if (length < 0) length = 0;\n");
    sb_append(sb, "    int64_t len = dyn_array_length(arr);\n");
    sb_append(sb, "    if (start > len) start = len;\n");
    sb_append(sb, "    int64_t end = start + length;\n");
    sb_append(sb, "    if (end > len) end = len;\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(arr);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    if (!out) return NULL;\n");
    sb_append(sb, "    for (int64_t i = start; i < end; i++) {\n");
    sb_append(sb, "        switch (t) {\n");
    sb_append(sb, "            case ELEM_U8: dyn_array_push_u8(out, dyn_array_get_u8(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_INT: dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_FLOAT: dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_BOOL: dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_STRING: dyn_array_push_string(out, dyn_array_get_string(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_ARRAY: dyn_array_push_array(out, dyn_array_get_array(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_STRUCT: dyn_array_push_struct(out, dyn_array_get_struct(arr, i), (size_t)arr->elem_size); break;\n");
    sb_append(sb, "            default: assert(false && \"nl_array_slice: unsupported element type\");\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_bool(bool value) {\n");
    sb_append(sb, "    printf(\"%s\\n\", value ? \"true\" : \"false\");\n");
    sb_append(sb, "}\n\n");

    /* Array printing - supports DynArray */
    sb_append(sb, "static void nl_print_array(DynArray* arr) {\n");
    sb_append(sb, "    printf(\"[\");\n");
    sb_append(sb, "    for (int i = 0; i < arr->length; i++) {\n");
    sb_append(sb, "        if (i > 0) printf(\", \");\n");
    sb_append(sb, "        switch (arr->elem_type) {\n");
    sb_append(sb, "            case ELEM_INT:\n");
    sb_append(sb, "                printf(\"%lld\", (long long)((int64_t*)arr->data)[i]);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            case ELEM_U8:\n");
    sb_append(sb, "                printf(\"%u\", (unsigned)((uint8_t*)arr->data)[i]);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            case ELEM_FLOAT:\n");
    sb_append(sb, "                printf(\"%g\", ((double*)arr->data)[i]);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            default:\n");
    sb_append(sb, "                printf(\"?\");\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    printf(\"]\");\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_println_array(DynArray* arr) {\n");
    sb_append(sb, "    nl_print_array(arr);\n");
    sb_append(sb, "    printf(\"\\n\");\n");
    sb_append(sb, "}\n\n");

    /* Array operations */
    sb_append(sb, "/* ========== Array Operations (With Bounds Checking!) ========== */\n\n");
    
    sb_append(sb, "/* Array struct */\n");
    /* Old nl_array operations removed - now using DynArray exclusively */
    
    sb_append(sb, "/* ========== End Array Operations ========== */\n\n");

    sb_append(sb, "/* ========== End Math and Utility Built-in Functions ========== */\n\n");
}

/* Generate string operations */
void generate_string_operations(StringBuilder *sb) {
    sb_append(sb, "/* ========== Advanced String Operations ========== */\n\n");
    
    /* char_at - use strnlen for safety */
    sb_append(sb, "static int64_t char_at(const char* s, int64_t index) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to reasonable size (1MB) */\n");
    sb_append(sb, "    int len = strnlen(s, 1024*1024);\n");
    sb_append(sb, "    if (index < 0 || index >= len) {\n");
    sb_append(sb, "        fprintf(stderr, \"Error: Index %lld out of bounds (string length %d)\\n\", (long long)index, len);\n");
    sb_append(sb, "        return 0;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return (unsigned char)s[index];\n");
    sb_append(sb, "}\n\n");
    
    /* string_from_char */
    sb_append(sb, "static char* string_from_char(int64_t c) {\n");
    sb_append(sb, "    char* buffer = gc_alloc_string(1);\n");
    sb_append(sb, "    if (!buffer) return \"\";\n");
    sb_append(sb, "    buffer[0] = (char)c;\n");
    sb_append(sb, "    buffer[1] = '\\0';\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");
    
    /* Character classification */
    sb_append(sb, "static bool is_digit(int64_t c) {\n");
    sb_append(sb, "    return c >= '0' && c <= '9';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_alpha(int64_t c) {\n");
    sb_append(sb, "    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_alnum(int64_t c) {\n");
    sb_append(sb, "    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_whitespace(int64_t c) {\n");
    sb_append(sb, "    return c == ' ' || c == '\\t' || c == '\\n' || c == '\\r';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_upper(int64_t c) {\n");
    sb_append(sb, "    return c >= 'A' && c <= 'Z';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_lower(int64_t c) {\n");
    sb_append(sb, "    return c >= 'a' && c <= 'z';\n");
    sb_append(sb, "}\n\n");
    
    /* Type conversions */
    sb_append(sb, "static char* int_to_string(int64_t n) {\n");
    sb_append(sb, "    char* buffer = gc_alloc_string(31);\n");
    sb_append(sb, "    if (!buffer) return \"\";\n");
    sb_append(sb, "    snprintf(buffer, 32, \"%lld\", (long long)n);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* float_to_string(double x) {\n");
    sb_append(sb, "    char* buffer = gc_alloc_string(63);\n");
    sb_append(sb, "    if (!buffer) return \"\";\n");
    sb_append(sb, "    snprintf(buffer, 64, \"%g\", x);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "typedef struct {\n");
    sb_append(sb, "    char *buf;\n");
    sb_append(sb, "    size_t len;\n");
    sb_append(sb, "    size_t cap;\n");
    sb_append(sb, "} nl_fmt_sb_t;\n\n");

    sb_append(sb, "static void nl_fmt_sb_ensure(nl_fmt_sb_t *sb, size_t extra) {\n");
    sb_append(sb, "    if (!sb) return;\n");
    sb_append(sb, "    size_t needed = sb->len + extra + 1;\n");
    sb_append(sb, "    if (needed <= sb->cap) return;\n");
    sb_append(sb, "    size_t new_cap = sb->cap ? sb->cap : 128;\n");
    sb_append(sb, "    while (new_cap < needed) new_cap *= 2;\n");
    sb_append(sb, "    char *new_buf = realloc(sb->buf, new_cap);\n");
    sb_append(sb, "    if (!new_buf) return;\n");
    sb_append(sb, "    sb->buf = new_buf;\n");
    sb_append(sb, "    sb->cap = new_cap;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static nl_fmt_sb_t nl_fmt_sb_new(size_t initial_cap) {\n");
    sb_append(sb, "    nl_fmt_sb_t sb = {0};\n");
    sb_append(sb, "    sb.cap = initial_cap ? initial_cap : 128;\n");
    sb_append(sb, "    sb.buf = (char*)malloc(sb.cap);\n");
    sb_append(sb, "    sb.len = 0;\n");
    sb_append(sb, "    if (sb.buf) sb.buf[0] = '\\0';\n");
    sb_append(sb, "    return sb;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_fmt_sb_append_cstr(nl_fmt_sb_t *sb, const char *s) {\n");
    sb_append(sb, "    if (!sb || !s) return;\n");
    sb_append(sb, "    size_t n = strlen(s);\n");
    sb_append(sb, "    nl_fmt_sb_ensure(sb, n);\n");
    sb_append(sb, "    if (!sb->buf) return;\n");
    sb_append(sb, "    memcpy(sb->buf + sb->len, s, n);\n");
    sb_append(sb, "    sb->len += n;\n");
    sb_append(sb, "    sb->buf[sb->len] = '\\0';\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_fmt_sb_append_char(nl_fmt_sb_t *sb, char c) {\n");
    sb_append(sb, "    if (!sb) return;\n");
    sb_append(sb, "    nl_fmt_sb_ensure(sb, 1);\n");
    sb_append(sb, "    if (!sb->buf) return;\n");
    sb_append(sb, "    sb->buf[sb->len++] = c;\n");
    sb_append(sb, "    sb->buf[sb->len] = '\\0';\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_fmt_sb_build(nl_fmt_sb_t *sb) {\n");
    sb_append(sb, "    if (!sb || !sb->buf) return \"\";\n");
    sb_append(sb, "    return sb->buf;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static const char* nl_to_string_int(int64_t v) { return int_to_string(v); }\n");
    sb_append(sb, "static const char* nl_to_string_float(double v) { return float_to_string(v); }\n");
    sb_append(sb, "static const char* nl_to_string_bool(bool v) { return v ? \"true\" : \"false\"; }\n");
    sb_append(sb, "static const char* nl_to_string_string(const char* v) { return v ? v : \"\"; }\n\n");

    sb_append(sb, "static const char* nl_to_string_array(DynArray* arr) {\n");
    sb_append(sb, "    if (!arr) return \"[]\";\n");
    sb_append(sb, "    nl_fmt_sb_t sb = nl_fmt_sb_new(256);\n");
    sb_append(sb, "    nl_fmt_sb_append_char(&sb, '[');\n");
    sb_append(sb, "    int64_t len = dyn_array_length(arr);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(arr);\n");
    sb_append(sb, "    for (int64_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        if (i > 0) nl_fmt_sb_append_cstr(&sb, \", \");\n");
    sb_append(sb, "        switch (t) {\n");
    sb_append(sb, "            case ELEM_INT: {\n");
    sb_append(sb, "                const char* s = nl_to_string_int(dyn_array_get_int(arr, i));\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, s);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            case ELEM_U8: {\n");
    sb_append(sb, "                const char* s = nl_to_string_int((int64_t)dyn_array_get_u8(arr, i));\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, s);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            case ELEM_FLOAT: {\n");
    sb_append(sb, "                const char* s = nl_to_string_float(dyn_array_get_float(arr, i));\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, s);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            case ELEM_BOOL: {\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, nl_to_string_bool(dyn_array_get_bool(arr, i)));\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            case ELEM_STRING: {\n");
    sb_append(sb, "                nl_fmt_sb_append_char(&sb, '\"');\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, nl_to_string_string(dyn_array_get_string(arr, i)));\n");
    sb_append(sb, "                nl_fmt_sb_append_char(&sb, '\"');\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            case ELEM_ARRAY: {\n");
    sb_append(sb, "                const char* s = nl_to_string_array(dyn_array_get_array(arr, i));\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, s);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            case ELEM_STRUCT: {\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, \"<struct>\");\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "            default: {\n");
    sb_append(sb, "                nl_fmt_sb_append_cstr(&sb, \"?\");\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    nl_fmt_sb_append_char(&sb, ']');\n");
    sb_append(sb, "    return nl_fmt_sb_build(&sb);\n");
    sb_append(sb, "}\n\n");

    /* Array operators (elementwise) */
    sb_append(sb, "static const char* nl_str_concat(const char* s1, const char* s2);\n");
    sb_append(sb, "static DynArray* nl_array_add(DynArray* a, DynArray* b);\n");
    sb_append(sb, "static DynArray* nl_array_sub(DynArray* a, DynArray* b);\n");
    sb_append(sb, "static DynArray* nl_array_mul(DynArray* a, DynArray* b);\n");
    sb_append(sb, "static DynArray* nl_array_div(DynArray* a, DynArray* b);\n");
    sb_append(sb, "static DynArray* nl_array_mod(DynArray* a, DynArray* b);\n\n");

    sb_append(sb, "static void nl_array_assert_compatible(DynArray* a, DynArray* b) {\n");
    sb_append(sb, "    assert(a && b);\n");
    sb_append(sb, "    assert(dyn_array_length(a) == dyn_array_length(b));\n");
    sb_append(sb, "    assert(dyn_array_get_elem_type(a) == dyn_array_get_elem_type(b));\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_add(DynArray* a, DynArray* b) {\n");
    sb_append(sb, "    nl_array_assert_compatible(a, b);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(a);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    switch (t) {\n");
    sb_append(sb, "        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)+dyn_array_get_int(b,i)); break;\n");
    sb_append(sb, "        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)+dyn_array_get_float(b,i)); break;\n");
    sb_append(sb, "        case ELEM_STRING: for (int64_t i=0;i<len;i++) dyn_array_push_string(out, nl_str_concat(dyn_array_get_string(a,i), dyn_array_get_string(b,i))); break;\n");
    sb_append(sb, "        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_add(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;\n");
    sb_append(sb, "        default: assert(false && \"nl_array_add: unsupported element type\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_sub(DynArray* a, DynArray* b) {\n");
    sb_append(sb, "    nl_array_assert_compatible(a, b);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(a);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    switch (t) {\n");
    sb_append(sb, "        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)-dyn_array_get_int(b,i)); break;\n");
    sb_append(sb, "        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)-dyn_array_get_float(b,i)); break;\n");
    sb_append(sb, "        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_sub(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;\n");
    sb_append(sb, "        default: assert(false && \"nl_array_sub: unsupported element type\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_mul(DynArray* a, DynArray* b) {\n");
    sb_append(sb, "    nl_array_assert_compatible(a, b);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(a);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    switch (t) {\n");
    sb_append(sb, "        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)*dyn_array_get_int(b,i)); break;\n");
    sb_append(sb, "        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)*dyn_array_get_float(b,i)); break;\n");
    sb_append(sb, "        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_mul(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;\n");
    sb_append(sb, "        default: assert(false && \"nl_array_mul: unsupported element type\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_div(DynArray* a, DynArray* b) {\n");
    sb_append(sb, "    nl_array_assert_compatible(a, b);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(a);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    switch (t) {\n");
    sb_append(sb, "        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)/dyn_array_get_int(b,i)); break;\n");
    sb_append(sb, "        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)/dyn_array_get_float(b,i)); break;\n");
    sb_append(sb, "        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_div(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;\n");
    sb_append(sb, "        default: assert(false && \"nl_array_div: unsupported element type\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_mod(DynArray* a, DynArray* b) {\n");
    sb_append(sb, "    nl_array_assert_compatible(a, b);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(a);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    switch (t) {\n");
    sb_append(sb, "        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)%dyn_array_get_int(b,i)); break;\n");
    sb_append(sb, "        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_mod(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;\n");
    sb_append(sb, "        default: assert(false && \"nl_array_mod: unsupported element type\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_add_scalar_int(DynArray* a, int64_t s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) + s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_radd_scalar_int(int64_t s, DynArray* a) { return nl_array_add_scalar_int(a, s); }\n\n");
    sb_append(sb, "static DynArray* nl_array_sub_scalar_int(DynArray* a, int64_t s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) - s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rsub_scalar_int(int64_t s, DynArray* a) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, s - dyn_array_get_int(a,i));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_mul_scalar_int(DynArray* a, int64_t s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) * s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rmul_scalar_int(int64_t s, DynArray* a) { return nl_array_mul_scalar_int(a, s); }\n\n");
    sb_append(sb, "static DynArray* nl_array_div_scalar_int(DynArray* a, int64_t s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) / s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rdiv_scalar_int(int64_t s, DynArray* a) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, s / dyn_array_get_int(a,i));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_mod_scalar_int(DynArray* a, int64_t s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) % s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rmod_scalar_int(int64_t s, DynArray* a) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, s % dyn_array_get_int(a,i));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_add_scalar_float(DynArray* a, double s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) + s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_radd_scalar_float(double s, DynArray* a) { return nl_array_add_scalar_float(a, s); }\n\n");
    sb_append(sb, "static DynArray* nl_array_sub_scalar_float(DynArray* a, double s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) - s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rsub_scalar_float(double s, DynArray* a) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, s - dyn_array_get_float(a,i));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_mul_scalar_float(DynArray* a, double s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) * s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rmul_scalar_float(double s, DynArray* a) { return nl_array_mul_scalar_float(a, s); }\n\n");
    sb_append(sb, "static DynArray* nl_array_div_scalar_float(DynArray* a, double s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) / s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_rdiv_scalar_float(double s, DynArray* a) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, s / dyn_array_get_float(a,i));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_add_scalar_string(DynArray* a, const char* s) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_STRING);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_string(out, nl_str_concat(dyn_array_get_string(a,i), s));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static DynArray* nl_array_radd_scalar_string(const char* s, DynArray* a) {\n");
    sb_append(sb, "    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_STRING);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    for (int64_t i=0;i<len;i++) dyn_array_push_string(out, nl_str_concat(s, dyn_array_get_string(a,i)));\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t string_to_int(const char* s) {\n");
    sb_append(sb, "    return strtoll(s, NULL, 10);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t digit_value(int64_t c) {\n");
    sb_append(sb, "    if (c >= '0' && c <= '9') {\n");
    sb_append(sb, "        return c - '0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return -1;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t char_to_lower(int64_t c) {\n");
    sb_append(sb, "    if (c >= 'A' && c <= 'Z') {\n");
    sb_append(sb, "    return c + 32;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return c;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t char_to_upper(int64_t c) {\n");
    sb_append(sb, "    if (c >= 'a' && c <= 'z') {\n");
    sb_append(sb, "        return c - 32;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return c;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* ========== End Advanced String Operations ========== */\n\n");
}

/* Generate path operations for OS stdlib */
void generate_path_operations(StringBuilder *sb) {
    /* Path operations */
    sb_append(sb, "static bool nl_os_path_isfile(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISREG(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_path_isdir(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISDIR(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_join(const char* a, const char* b) {\n");
    sb_append(sb, "    char* buffer = malloc(2048);\n");
    sb_append(sb, "    if (!buffer) return \"\";\n");
    sb_append(sb, "    if (strlen(a) == 0) {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s\", b);\n");
    sb_append(sb, "    } else if (a[strlen(a) - 1] == '/') {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s%s\", a, b);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s/%s\", a, b);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_basename(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* base = basename(path_copy);\n");
    sb_append(sb, "    char* result = strdup(base);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_dirname(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* dir = dirname(path_copy);\n");
    sb_append(sb, "    char* result = strdup(dir);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_normalize(const char* path) {\n");
    sb_append(sb, "    if (!path) return \"\";\n");
    sb_append(sb, "    bool abs = (path[0] == '/');\n");
    sb_append(sb, "    char* copy = strdup(path);\n");
    sb_append(sb, "    if (!copy) return \"\";\n");
    sb_append(sb, "\n");
    sb_append(sb, "    const char* parts[512];\n");
    sb_append(sb, "    int count = 0;\n");
    sb_append(sb, "    char* save = NULL;\n");
    sb_append(sb, "    char* tok = strtok_r(copy, \"/\", &save);\n");
    sb_append(sb, "    while (tok) {\n");
    sb_append(sb, "        if (strcmp(tok, \"\") == 0 || strcmp(tok, \".\") == 0) {\n");
    sb_append(sb, "            /* skip */\n");
    sb_append(sb, "        } else if (strcmp(tok, \"..\") == 0) {\n");
    sb_append(sb, "            if (count > 0 && strcmp(parts[count - 1], \"..\") != 0) {\n");
    sb_append(sb, "                count--;\n");
    sb_append(sb, "            } else if (!abs) {\n");
    sb_append(sb, "                parts[count++] = tok;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "        } else {\n");
    sb_append(sb, "            if (count < 512) parts[count++] = tok;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        tok = strtok_r(NULL, \"/\", &save);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "\n");
    sb_append(sb, "    size_t cap = strlen(path) + 3;\n");
    sb_append(sb, "    char* out = malloc(cap);\n");
    sb_append(sb, "    if (!out) { free(copy); return \"\"; }\n");
    sb_append(sb, "    size_t pos = 0;\n");
    sb_append(sb, "    if (abs) out[pos++] = '/';\n");
    sb_append(sb, "\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        size_t len = strlen(parts[i]);\n");
    sb_append(sb, "        if (pos + len + 2 > cap) {\n");
    sb_append(sb, "            cap = (pos + len + 2) * 2;\n");
    sb_append(sb, "            char* n = realloc(out, cap);\n");
    sb_append(sb, "            if (!n) { free(out); free(copy); return \"\"; }\n");
    sb_append(sb, "            out = n;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        if (pos > 0 && out[pos - 1] != '/') out[pos++] = '/';\n");
    sb_append(sb, "        memcpy(out + pos, parts[i], len);\n");
    sb_append(sb, "        pos += len;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "\n");
    sb_append(sb, "    if (pos == 0) {\n");
    sb_append(sb, "        if (abs) { out[pos++] = '/'; } else { out[pos++] = '.'; }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    out[pos] = '\\0';\n");
    sb_append(sb, "\n");
    sb_append(sb, "    free(copy);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
}

/* Generate directory operations for OS stdlib */
void generate_dir_operations(StringBuilder *sb) {
    /* Directory operations */
    sb_append(sb, "static int64_t nl_os_dir_create(const char* path) {\n");
    sb_append(sb, "    return mkdir(path, 0755) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_dir_remove(const char* path) {\n");
    sb_append(sb, "    return rmdir(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_dir_list(const char* path) {\n");
    sb_append(sb, "    DIR* dir = opendir(path);\n");
    sb_append(sb, "    if (!dir) return \"\";\n");
    sb_append(sb, "    size_t capacity = 4096;\n");
    sb_append(sb, "    size_t used = 0;\n");
    sb_append(sb, "    char* buffer = malloc(capacity);\n");
    sb_append(sb, "    if (!buffer) { closedir(dir); return \"\"; }\n");
    sb_append(sb, "    buffer[0] = '\\0';\n");
    sb_append(sb, "    struct dirent* entry;\n");
    sb_append(sb, "    while ((entry = readdir(dir)) != NULL) {\n");
    sb_append(sb, "        if (strcmp(entry->d_name, \".\") == 0 || strcmp(entry->d_name, \"..\") == 0) continue;\n");
    sb_append(sb, "        size_t name_len = strlen(entry->d_name);\n");
    sb_append(sb, "        size_t needed = used + name_len + 2; /* +1 for newline, +1 for null */\n");
    sb_append(sb, "        if (needed > capacity) {\n");
    sb_append(sb, "            capacity = needed * 2;\n");
    sb_append(sb, "            char* new_buffer = realloc(buffer, capacity);\n");
    sb_append(sb, "            if (!new_buffer) { free(buffer); closedir(dir); return \"\"; }\n");
    sb_append(sb, "            buffer = new_buffer;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        memcpy(buffer + used, entry->d_name, name_len);\n");
    sb_append(sb, "        used += name_len;\n");
    sb_append(sb, "        buffer[used++] = '\\n';\n");
    sb_append(sb, "        buffer[used] = '\\0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    closedir(dir);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_dir_exists(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISDIR(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_getcwd(void) {\n");
    sb_append(sb, "    char* buffer = malloc(1024);\n");
    sb_append(sb, "    if (getcwd(buffer, 1024) == NULL) {\n");
    sb_append(sb, "        buffer[0] = '\\0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_chdir(const char* path) {\n");
    sb_append(sb, "    return chdir(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_os_walkdir_rec(const char* root, DynArray* out) {\n");
    sb_append(sb, "    DIR* dir = opendir(root);\n");
    sb_append(sb, "    if (!dir) return;\n");
    sb_append(sb, "    struct dirent* entry;\n");
    sb_append(sb, "    while ((entry = readdir(dir)) != NULL) {\n");
    sb_append(sb, "        if (strcmp(entry->d_name, \".\") == 0 || strcmp(entry->d_name, \"..\") == 0) continue;\n");
    sb_append(sb, "        size_t root_len = strlen(root);\n");
    sb_append(sb, "        size_t name_len = strlen(entry->d_name);\n");
    sb_append(sb, "        bool needs_slash = (root_len > 0 && root[root_len - 1] != '/');\n");
    sb_append(sb, "        size_t cap = root_len + (needs_slash ? 1 : 0) + name_len + 1;\n");
    sb_append(sb, "        char* path = malloc(cap);\n");
    sb_append(sb, "        if (!path) continue;\n");
    sb_append(sb, "        if (needs_slash) snprintf(path, cap, \"%s/%s\", root, entry->d_name);\n");
    sb_append(sb, "        else snprintf(path, cap, \"%s%s\", root, entry->d_name);\n");
    sb_append(sb, "\n");
    sb_append(sb, "        struct stat st;\n");
    sb_append(sb, "        if (stat(path, &st) != 0) { free(path); continue; }\n");
    sb_append(sb, "        if (S_ISDIR(st.st_mode)) {\n");
    sb_append(sb, "            nl_os_walkdir_rec(path, out);\n");
    sb_append(sb, "            free(path);\n");
    sb_append(sb, "        } else if (S_ISREG(st.st_mode)) {\n");
    sb_append(sb, "            dyn_array_push_string(out, path);\n");
    sb_append(sb, "        } else {\n");
    sb_append(sb, "            free(path);\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    closedir(dir);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_os_walkdir(const char* root) {\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    if (!root || root[0] == '\\0') return out;\n");
    sb_append(sb, "    nl_os_walkdir_rec(root, out);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
}

/* Generate file operations for OS stdlib */
void generate_file_operations(StringBuilder *sb) {
    /* File operations */
    sb_append(sb, "static char* nl_os_file_read(const char* path) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"rb\");  /* Binary mode for MOD files */\n");
    sb_append(sb, "    if (!f) return \"\";\n");
    sb_append(sb, "    fseek(f, 0, SEEK_END);\n");
    sb_append(sb, "    long size = ftell(f);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_SET);\n");
    sb_append(sb, "    char* buffer = malloc(size + 1);\n");
    sb_append(sb, "    if (!buffer) { fclose(f); return \"\"; }\n");
    sb_append(sb, "    fread(buffer, 1, size, f);\n");
    sb_append(sb, "    buffer[size] = '\\0';\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    /* Binary file reading - returns DynArray of bytes (0-255) */
    sb_append(sb, "static DynArray* nl_os_file_read_bytes(const char* path) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"rb\");\n");
    sb_append(sb, "    if (!f) {\n");
    sb_append(sb, "        /* Return empty array on error */\n");
    sb_append(sb, "        return dyn_array_new(ELEM_U8);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fseek(f, 0, SEEK_END);\n");
    sb_append(sb, "    long size = ftell(f);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_SET);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Create dynamic array for bytes */\n");
    sb_append(sb, "    DynArray* bytes = dyn_array_new(ELEM_U8);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Read bytes and add to array */\n");
    sb_append(sb, "    for (long i = 0; i < size; i++) {\n");
    sb_append(sb, "        int c = fgetc(f);\n");
    sb_append(sb, "        if (c == EOF) break;\n");
    sb_append(sb, "        dyn_array_push_u8(bytes, (uint8_t)(unsigned char)c);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return bytes;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_write(const char* path, const char* content) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"w\");\n");
    sb_append(sb, "    if (!f) return -1;\n");
    sb_append(sb, "    fputs(content, f);\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_append(const char* path, const char* content) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"a\");\n");
    sb_append(sb, "    if (!f) return -1;\n");
    sb_append(sb, "    fputs(content, f);\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_remove(const char* path) {\n");
    sb_append(sb, "    return remove(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    /* Backwards-compat alias */
    sb_append(sb, "static int64_t nl_os_file_delete(const char* path) {\n");
    sb_append(sb, "    return nl_os_file_remove(path);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_rename(const char* old_path, const char* new_path) {\n");
    sb_append(sb, "    return rename(old_path, new_path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_size(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return -1;\n");
    sb_append(sb, "    return (int64_t)st.st_size;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_file_exists(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    return stat(path, &st) == 0;\n");
    sb_append(sb, "}\n\n");

    /* Temp helpers */
    sb_append(sb, "static char* nl_os_tmp_dir(void) {\n");
    sb_append(sb, "    const char* tmp = getenv(\"TMPDIR\");\n");
    sb_append(sb, "    if (!tmp || tmp[0] == '\\0') tmp = \"/tmp\";\n");
    sb_append(sb, "    size_t len = strlen(tmp);\n");
    sb_append(sb, "    char* out = malloc(len + 1);\n");
    sb_append(sb, "    if (!out) return \"\";\n");
    sb_append(sb, "    memcpy(out, tmp, len);\n");
    sb_append(sb, "    out[len] = '\\0';\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_mktemp(const char* prefix) {\n");
    sb_append(sb, "    const char* tmp = getenv(\"TMPDIR\");\n");
    sb_append(sb, "    if (!tmp || tmp[0] == '\\0') tmp = \"/tmp\";\n");
    sb_append(sb, "    const char* p = (prefix && prefix[0]) ? prefix : \"nanolang_\";\n");
    sb_append(sb, "    char templ[1024];\n");
    sb_append(sb, "    snprintf(templ, sizeof(templ), \"%s/%sXXXXXX\", tmp, p);\n");
    sb_append(sb, "    int fd = mkstemp(templ);\n");
    sb_append(sb, "    if (fd < 0) return \"\";\n");
    sb_append(sb, "    close(fd);\n");
    sb_append(sb, "    size_t len = strlen(templ);\n");
    sb_append(sb, "    char* out = malloc(len + 1);\n");
    sb_append(sb, "    if (!out) return \"\";\n");
    sb_append(sb, "    memcpy(out, templ, len);\n");
    sb_append(sb, "    out[len] = '\\0';\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_mktemp_dir(const char* prefix) {\n");
    sb_append(sb, "    const char* tmp = getenv(\"TMPDIR\");\n");
    sb_append(sb, "    if (!tmp || tmp[0] == '\\0') tmp = \"/tmp\";\n");
    sb_append(sb, "    const char* p = (prefix && prefix[0]) ? prefix : \"nanolang_dir_\";\n");
    sb_append(sb, "    char path[1024];\n");
    sb_append(sb, "    for (int i = 0; i < 100; i++) {\n");
    sb_append(sb, "        snprintf(path, sizeof(path), \"%s/%s%lld_%d\", tmp, p, (long long)time(NULL), i);\n");
    sb_append(sb, "        if (mkdir(path, 0700) == 0) {\n");
    sb_append(sb, "            size_t len = strlen(path);\n");
    sb_append(sb, "            char* out = malloc(len + 1);\n");
    sb_append(sb, "            if (!out) return \"\";\n");
    sb_append(sb, "            memcpy(out, path, len);\n");
    sb_append(sb, "            out[len] = '\\0';\n");
    sb_append(sb, "            return out;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return \"\";\n");
    sb_append(sb, "}\n\n");
}

/* Generate complete stdlib runtime (convenience function) */
void generate_stdlib_runtime(StringBuilder *sb) {
    /* OS stdlib runtime library */
    sb_append(sb, "/* ========== OS Standard Library ========== */\n\n");
    
    /* Disable unused-function warnings - not all stdlib functions used in every program */
    sb_append(sb, "#pragma GCC diagnostic push\n");
    sb_append(sb, "#pragma GCC diagnostic ignored \"-Wunused-function\"\n\n");

    /* File operations */
    generate_file_operations(sb);

    /* Directory operations */
    generate_dir_operations(sb);

    /* Path operations */
    generate_path_operations(sb);

    /* Process operations */
    sb_append(sb, "static int64_t nl_os_system(const char* command) {\n");
    sb_append(sb, "    return system(command);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_os_exit(int64_t code) {\n");
    sb_append(sb, "    exit((int)code);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static const char* nl_os_getenv(const char* name) {\n");
    sb_append(sb, "    const char* value = getenv(name);\n");
    sb_append(sb, "    return value ? value : \"\";\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* system() wrapper - stdlib system() available via stdlib.h */\n");
    sb_append(sb, "int64_t nl_exec_shell(const char* cmd) {\n");
    sb_append(sb, "    return (int64_t)system(cmd);\n");
    sb_append(sb, "}\n\n");

    /* File I/O aliases for self-hosted compiler compatibility */
    sb_append(sb, "/* File I/O aliases (without nl_os_ prefix for compiler use) */\n");
    sb_append(sb, "static char* file_read(const char* path) {\n");
    sb_append(sb, "    return nl_os_file_read(path);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t file_write(const char* path, const char* content) {\n");
    sb_append(sb, "    return nl_os_file_write(path, content);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool file_exists(const char* path) {\n");
    sb_append(sb, "    return nl_os_file_exists(path);\n");
    sb_append(sb, "}\n\n");
    
    /* Re-enable warnings after stdlib functions */
    sb_append(sb, "#pragma GCC diagnostic pop\n\n");

    sb_append(sb, "/* ========== End OS Standard Library ========== */\n\n");

    /* String operations */
    generate_string_operations(sb);

    /* Math and utility built-in functions */
    generate_math_utility_builtins(sb);
}

/* =============================================================================
 * Module system stubs for runtime-only compilation
 * =============================================================================
 * These functions provide fallback implementations when the full module
 * system (module.c) isn't linked. Programs using imports will need the
 * full compiler linked.
 */

void generate_module_system_stubs(StringBuilder *sb) {
    sb_append(sb, "/* ========== Module System Stubs (runtime-only fallbacks) ========== */\n\n");
    
    sb_append(sb, "#ifndef MODULE_SYSTEM_AVAILABLE\n");
    sb_append(sb, "int64_t module_get_import_count(const char *module_path) {\n");
    sb_append(sb, "    (void)module_path;\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "const char *module_get_import_path(const char *module_path, int64_t index) {\n");
    sb_append(sb, "    (void)module_path; (void)index;\n");
    sb_append(sb, "    return \"\";\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "const char *module_generate_forward_declarations(const char *module_path) {\n");
    sb_append(sb, "    (void)module_path;\n");
    sb_append(sb, "    return \"\";\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "const char *resolve_module_path(const char *module_path, const char *current_file) {\n");
    sb_append(sb, "    (void)module_path; (void)current_file;\n");
    sb_append(sb, "    return \"\";\n");
    sb_append(sb, "}\n");
    sb_append(sb, "#endif\n\n");
    
    sb_append(sb, "/* ========== End Module System Stubs ========== */\n\n");
}

