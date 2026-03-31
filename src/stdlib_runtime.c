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
    sb_append(sb, "    /* Safety: Bound string scan to 64MB */\n");
    sb_append(sb, "    size_t len1 = strnlen(s1, 64*1024*1024);\n");
    sb_append(sb, "    size_t len2 = strnlen(s2, 64*1024*1024);\n");
    sb_append(sb, "    char* result = gc_alloc_string(len1 + len2);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, s1, len1);\n");
    sb_append(sb, "    memcpy(result + len1, s2, len2);\n");
    sb_append(sb, "    result[len1 + len2] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String substring - use strnlen for safety */\n");
    sb_append(sb, "static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to 64MB */\n");
    sb_append(sb, "    int64_t str_len = strnlen(str, 64*1024*1024);\n");
    sb_append(sb, "    if (start < 0 || start > str_len || length < 0) return \"\";\n");
    sb_append(sb, "    if (start == str_len) return \"\";\n");
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

    sb_append(sb, "/* String starts_with */\n");
    sb_append(sb, "static bool nl_str_starts_with(const char* s, const char* prefix) {\n");
    sb_append(sb, "    if (!s || !prefix) return false;\n");
    sb_append(sb, "    size_t slen = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    size_t plen = strnlen(prefix, 64*1024*1024);\n");
    sb_append(sb, "    if (plen > slen) return false;\n");
    sb_append(sb, "    return strncmp(s, prefix, plen) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* String ends_with */\n");
    sb_append(sb, "static bool nl_str_ends_with(const char* s, const char* suffix) {\n");
    sb_append(sb, "    if (!s || !suffix) return false;\n");
    sb_append(sb, "    size_t slen = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    size_t suflen = strnlen(suffix, 64*1024*1024);\n");
    sb_append(sb, "    if (suflen > slen) return false;\n");
    sb_append(sb, "    if (suflen == 0) return true;\n");
    sb_append(sb, "    return strncmp(s + slen - suflen, suffix, suflen) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* String index_of - returns index of first occurrence of needle, or -1 */\n");
    sb_append(sb, "static int64_t nl_str_index_of(const char* haystack, const char* needle) {\n");
    sb_append(sb, "    if (!haystack || !needle) return -1;\n");
    sb_append(sb, "    const char* p = strstr(haystack, needle);\n");
    sb_append(sb, "    if (!p) return -1;\n");
    sb_append(sb, "    return (int64_t)(p - haystack);\n");
    sb_append(sb, "}\n\n");

    /* Bytes (array<u8>) helpers */
    sb_append(sb, "static DynArray* nl_bytes_from_string(const char* s) {\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(ELEM_U8);\n");
    sb_append(sb, "    if (!out) return NULL;\n");
    sb_append(sb, "    if (!s) return out;\n");
    sb_append(sb, "    size_t len = strnlen(s, 64*1024*1024);\n");
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

    /* Array sort (integer ascending, in-place on a copy) */
    sb_append(sb, "static int nl_array_sort_cmp_int(const void* a, const void* b) {\n");
    sb_append(sb, "    int64_t x = *(const int64_t*)a;\n");
    sb_append(sb, "    int64_t y = *(const int64_t*)b;\n");
    sb_append(sb, "    return (x > y) - (x < y);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_array_sort(DynArray* arr) {\n");
    sb_append(sb, "    if (!arr) return dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    DynArray* out = dyn_array_clone(arr);\n");
    sb_append(sb, "    if (!out) return arr;\n");
    sb_append(sb, "    int64_t len = dyn_array_length(out);\n");
    sb_append(sb, "    if (len <= 1) return out;\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(out);\n");
    sb_append(sb, "    if (t == ELEM_INT) {\n");
    sb_append(sb, "        qsort(out->data, (size_t)len, sizeof(int64_t), nl_array_sort_cmp_int);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    /* Array reverse (returns a new array) */
    sb_append(sb, "static DynArray* nl_array_reverse(DynArray* arr) {\n");
    sb_append(sb, "    if (!arr) return dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    int64_t len = dyn_array_length(arr);\n");
    sb_append(sb, "    ElementType t = dyn_array_get_elem_type(arr);\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(t);\n");
    sb_append(sb, "    if (!out) return NULL;\n");
    sb_append(sb, "    for (int64_t i = len - 1; i >= 0; i--) {\n");
    sb_append(sb, "        switch (t) {\n");
    sb_append(sb, "            case ELEM_INT:    dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_FLOAT:  dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_BOOL:   dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;\n");
    sb_append(sb, "            case ELEM_STRING: dyn_array_push_string(out, dyn_array_get_string(arr, i)); break;\n");
    sb_append(sb, "            default: dyn_array_push_int(out, 0); break;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");

    /* Array contains (int elem) */
    sb_append(sb, "static bool nl_array_contains(DynArray* arr, int64_t elem) {\n");
    sb_append(sb, "    if (!arr) return false;\n");
    sb_append(sb, "    int64_t len = dyn_array_length(arr);\n");
    sb_append(sb, "    for (int64_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        if (dyn_array_get_int(arr, i) == elem) return true;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return false;\n");
    sb_append(sb, "}\n\n");

    /* Array index_of (int elem, returns -1 if not found) */
    sb_append(sb, "static int64_t nl_array_index_of(DynArray* arr, int64_t elem) {\n");
    sb_append(sb, "    if (!arr) return -1;\n");
    sb_append(sb, "    int64_t len = dyn_array_length(arr);\n");
    sb_append(sb, "    for (int64_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        if (dyn_array_get_int(arr, i) == elem) return i;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* ========== End Array Operations ========== */\n\n");

    sb_append(sb, "/* ========== End Math and Utility Built-in Functions ========== */\n\n");
}

/* Generate string operations */
void generate_string_operations(StringBuilder *sb) {
    sb_append(sb, "/* ========== Advanced String Operations ========== */\n\n");
    
    /* char_at - use strnlen for safety */
    sb_append(sb, "static int64_t char_at(const char* s, int64_t index) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to 64MB */\n");
    sb_append(sb, "    int len = strnlen(s, 64*1024*1024);\n");
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
    
    /* str_trim */
    sb_append(sb, "static const char* nl_str_trim(const char* s) {\n");
    sb_append(sb, "    if (!s) return \"\";\n");
    sb_append(sb, "    size_t len = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    size_t start = 0;\n");
    sb_append(sb, "    while (start < len && (s[start] == ' ' || s[start] == '\\t' || s[start] == '\\n' || s[start] == '\\r')) start++;\n");
    sb_append(sb, "    size_t end = len;\n");
    sb_append(sb, "    while (end > start && (s[end-1] == ' ' || s[end-1] == '\\t' || s[end-1] == '\\n' || s[end-1] == '\\r')) end--;\n");
    sb_append(sb, "    size_t new_len = end - start;\n");
    sb_append(sb, "    char* result = gc_alloc_string(new_len);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, s + start, new_len);\n");
    sb_append(sb, "    result[new_len] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_trim_left */
    sb_append(sb, "static const char* nl_str_trim_left(const char* s) {\n");
    sb_append(sb, "    if (!s) return \"\";\n");
    sb_append(sb, "    size_t len = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    size_t start = 0;\n");
    sb_append(sb, "    while (start < len && (s[start] == ' ' || s[start] == '\\t' || s[start] == '\\n' || s[start] == '\\r')) start++;\n");
    sb_append(sb, "    size_t new_len = len - start;\n");
    sb_append(sb, "    char* result = gc_alloc_string(new_len);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, s + start, new_len);\n");
    sb_append(sb, "    result[new_len] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_trim_right */
    sb_append(sb, "static const char* nl_str_trim_right(const char* s) {\n");
    sb_append(sb, "    if (!s) return \"\";\n");
    sb_append(sb, "    size_t len = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    size_t end = len;\n");
    sb_append(sb, "    while (end > 0 && (s[end-1] == ' ' || s[end-1] == '\\t' || s[end-1] == '\\n' || s[end-1] == '\\r')) end--;\n");
    sb_append(sb, "    char* result = gc_alloc_string(end);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, s, end);\n");
    sb_append(sb, "    result[end] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_to_lower */
    sb_append(sb, "static const char* nl_str_to_lower(const char* s) {\n");
    sb_append(sb, "    if (!s) return \"\";\n");
    sb_append(sb, "    size_t len = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    char* result = gc_alloc_string(len);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    for (size_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        unsigned char c = (unsigned char)s[i];\n");
    sb_append(sb, "        result[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    result[len] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_to_upper */
    sb_append(sb, "static const char* nl_str_to_upper(const char* s) {\n");
    sb_append(sb, "    if (!s) return \"\";\n");
    sb_append(sb, "    size_t len = strnlen(s, 64*1024*1024);\n");
    sb_append(sb, "    char* result = gc_alloc_string(len);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    for (size_t i = 0; i < len; i++) {\n");
    sb_append(sb, "        unsigned char c = (unsigned char)s[i];\n");
    sb_append(sb, "        result[i] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : (char)c;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    result[len] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_replace */
    sb_append(sb, "static const char* nl_str_replace(const char* s, const char* old_str, const char* new_str) {\n");
    sb_append(sb, "    if (!s || !old_str || !new_str) return s ? s : \"\";\n");
    sb_append(sb, "    size_t old_len = strlen(old_str);\n");
    sb_append(sb, "    size_t s_len = strlen(s);\n");
    sb_append(sb, "    if (old_len == 0) {\n");
    sb_append(sb, "        char* copy = gc_alloc_string(s_len);\n");
    sb_append(sb, "        if (!copy) return s;\n");
    sb_append(sb, "        memcpy(copy, s, s_len + 1);\n");
    sb_append(sb, "        return copy;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    size_t new_len = strlen(new_str);\n");
    sb_append(sb, "    int64_t count = 0;\n");
    sb_append(sb, "    const char* p = s;\n");
    sb_append(sb, "    const char* found;\n");
    sb_append(sb, "    while ((found = strstr(p, old_str)) != NULL) { count++; p = found + old_len; }\n");
    sb_append(sb, "    if (count == 0) {\n");
    sb_append(sb, "        char* copy = gc_alloc_string(s_len);\n");
    sb_append(sb, "        if (!copy) return s;\n");
    sb_append(sb, "        memcpy(copy, s, s_len + 1);\n");
    sb_append(sb, "        return copy;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    int64_t result_len = (int64_t)s_len + count * ((int64_t)new_len - (int64_t)old_len);\n");
    sb_append(sb, "    if (result_len < 0) return \"\";\n");
    sb_append(sb, "    char* result = gc_alloc_string((size_t)result_len);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    const char* src = s;\n");
    sb_append(sb, "    char* dst = result;\n");
    sb_append(sb, "    while ((found = strstr(src, old_str)) != NULL) {\n");
    sb_append(sb, "        size_t seg_len = (size_t)(found - src);\n");
    sb_append(sb, "        memcpy(dst, src, seg_len);\n");
    sb_append(sb, "        dst += seg_len;\n");
    sb_append(sb, "        memcpy(dst, new_str, new_len);\n");
    sb_append(sb, "        dst += new_len;\n");
    sb_append(sb, "        src = found + old_len;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    size_t rest = strlen(src);\n");
    sb_append(sb, "    memcpy(dst, src, rest);\n");
    sb_append(sb, "    dst[rest] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_split */
    sb_append(sb, "static DynArray* nl_str_split(const char* str, const char* delim) {\n");
    sb_append(sb, "    DynArray* result = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    if (!result) return NULL;\n");
    sb_append(sb, "    if (!str) return result;\n");
    sb_append(sb, "    size_t delim_len = strlen(delim);\n");
    sb_append(sb, "    if (delim_len == 0) {\n");
    sb_append(sb, "        size_t str_len = strnlen(str, 64*1024*1024);\n");
    sb_append(sb, "        for (size_t i = 0; i < str_len; i++) {\n");
    sb_append(sb, "            char* ch = gc_alloc_string(1);\n");
    sb_append(sb, "            if (!ch) break;\n");
    sb_append(sb, "            ch[0] = str[i]; ch[1] = '\\0';\n");
    sb_append(sb, "            dyn_array_push_string(result, ch);\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        return result;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    const char* start = str;\n");
    sb_append(sb, "    const char* found;\n");
    sb_append(sb, "    while ((found = strstr(start, delim)) != NULL) {\n");
    sb_append(sb, "        size_t seg_len = (size_t)(found - start);\n");
    sb_append(sb, "        char* seg = gc_alloc_string(seg_len);\n");
    sb_append(sb, "        if (!seg) break;\n");
    sb_append(sb, "        memcpy(seg, start, seg_len);\n");
    sb_append(sb, "        seg[seg_len] = '\\0';\n");
    sb_append(sb, "        dyn_array_push_string(result, seg);\n");
    sb_append(sb, "        start = found + delim_len;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    size_t rest_len = strlen(start);\n");
    sb_append(sb, "    char* seg = gc_alloc_string(rest_len);\n");
    sb_append(sb, "    if (seg) {\n");
    sb_append(sb, "        memcpy(seg, start, rest_len);\n");
    sb_append(sb, "        seg[rest_len] = '\\0';\n");
    sb_append(sb, "        dyn_array_push_string(result, seg);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* str_join */
    sb_append(sb, "static const char* nl_str_join(DynArray* arr, const char* delim) {\n");
    sb_append(sb, "    if (!arr) return \"\";\n");
    sb_append(sb, "    int64_t count = dyn_array_length(arr);\n");
    sb_append(sb, "    if (count == 0) return \"\";\n");
    sb_append(sb, "    size_t delim_len = strlen(delim);\n");
    sb_append(sb, "    size_t total = 0;\n");
    sb_append(sb, "    for (int64_t i = 0; i < count; i++) {\n");
    sb_append(sb, "        const char* s = dyn_array_get_string(arr, i);\n");
    sb_append(sb, "        if (s) total += strlen(s);\n");
    sb_append(sb, "        if (i < count - 1) total += delim_len;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    char* result = gc_alloc_string(total);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    size_t pos = 0;\n");
    sb_append(sb, "    for (int64_t i = 0; i < count; i++) {\n");
    sb_append(sb, "        const char* s = dyn_array_get_string(arr, i);\n");
    sb_append(sb, "        if (s) { size_t slen = strlen(s); memcpy(result + pos, s, slen); pos += slen; }\n");
    sb_append(sb, "        if (i < count - 1) { memcpy(result + pos, delim, delim_len); pos += delim_len; }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    result[pos] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* format - string interpolation: format(template, str_arg1, str_arg2, ...) */
    sb_append(sb, "static const char* nl_format(const char *fmt, int n_args, ...) {\n");
    sb_append(sb, "    if (!fmt) return \"\";\n");
    sb_append(sb, "    va_list ap;\n");
    sb_append(sb, "    va_start(ap, n_args);\n");
    sb_append(sb, "    nl_fmt_sb_t out = nl_fmt_sb_new(128);\n");
    sb_append(sb, "    const char *p = fmt;\n");
    sb_append(sb, "    int used = 0;\n");
    sb_append(sb, "    while (*p) {\n");
    sb_append(sb, "        if (*p == '%' && (p[1] == 's' || p[1] == 'd' || p[1] == 'f' || p[1] == 'g') && used < n_args) {\n");
    sb_append(sb, "            const char *arg = va_arg(ap, const char *);\n");
    sb_append(sb, "            if (arg) nl_fmt_sb_append_cstr(&out, arg);\n");
    sb_append(sb, "            used++;\n");
    sb_append(sb, "            p += 2;\n");
    sb_append(sb, "        } else {\n");
    sb_append(sb, "            nl_fmt_sb_append_char(&out, *p++);\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(ap);\n");
    sb_append(sb, "    char *result = gc_alloc_string(out.len);\n");
    sb_append(sb, "    if (result && out.buf) { memcpy(result, out.buf, out.len + 1); }\n");
    sb_append(sb, "    if (out.buf) free(out.buf);\n");
    sb_append(sb, "    return result ? result : \"\";\n");
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
    sb_append(sb, "    size_t len_a = strlen(a);\n");
    sb_append(sb, "    size_t len_b = strlen(b);\n");
    sb_append(sb, "    size_t total_len = len_a + len_b + 2; /* +1 for '/', +1 for null */\n");
    sb_append(sb, "    char* buffer = gc_alloc_string(total_len);\n");
    sb_append(sb, "    if (!buffer) return gc_alloc_string(0);\n");
    sb_append(sb, "    if (len_a == 0) {\n");
    sb_append(sb, "        snprintf(buffer, total_len, \"%s\", b);\n");
    sb_append(sb, "    } else if (a[len_a - 1] == '/') {\n");
    sb_append(sb, "        snprintf(buffer, total_len, \"%s%s\", a, b);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        snprintf(buffer, total_len, \"%s/%s\", a, b);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_basename(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* base = basename(path_copy);\n");
    sb_append(sb, "    size_t len = strlen(base);\n");
    sb_append(sb, "    char* result = gc_alloc_string(len);\n");
    sb_append(sb, "    if (result) memcpy(result, base, len + 1);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result ? result : gc_alloc_string(0);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_dirname(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* dir = dirname(path_copy);\n");
    sb_append(sb, "    size_t len = strlen(dir);\n");
    sb_append(sb, "    char* result = gc_alloc_string(len);\n");
    sb_append(sb, "    if (result) memcpy(result, dir, len + 1);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result ? result : gc_alloc_string(0);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_normalize(const char* path) {\n");
    sb_append(sb, "    if (!path) return gc_alloc_string(0);\n");
    sb_append(sb, "    bool abs = (path[0] == '/');\n");
    sb_append(sb, "    char* copy = strdup(path);\n");
    sb_append(sb, "    if (!copy) return gc_alloc_string(0);\n");
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
    sb_append(sb, "    /* Allocate GC string with max possible size */\n");
    sb_append(sb, "    size_t cap = strlen(path) + 3;\n");
    sb_append(sb, "    char* out = gc_alloc_string(cap);\n");
    sb_append(sb, "    if (!out) { free(copy); return gc_alloc_string(0); }\n");
    sb_append(sb, "    size_t pos = 0;\n");
    sb_append(sb, "    if (abs) out[pos++] = '/';\n");
    sb_append(sb, "\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        size_t len = strlen(parts[i]);\n");
    sb_append(sb, "        /* Check if we have enough space */\n");
    sb_append(sb, "        if (pos + len + 2 > cap) {\n");
    sb_append(sb, "            /* Need more space - allocate new GC string and copy */\n");
    sb_append(sb, "            size_t new_cap = (pos + len + 2) * 2;\n");
    sb_append(sb, "            char* new_out = gc_alloc_string(new_cap);\n");
    sb_append(sb, "            if (!new_out) { gc_release(out); free(copy); return gc_alloc_string(0); }\n");
    sb_append(sb, "            memcpy(new_out, out, pos);\n");
    sb_append(sb, "            gc_release(out);\n");
    sb_append(sb, "            out = new_out;\n");
    sb_append(sb, "            cap = new_cap;\n");
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
    sb_append(sb, "    if (!dir) return gc_alloc_string(0);\n");
    sb_append(sb, "    size_t capacity = 4096;\n");
    sb_append(sb, "    size_t used = 0;\n");
    sb_append(sb, "    char* buffer = gc_alloc_string(capacity);\n");
    sb_append(sb, "    if (!buffer) { closedir(dir); return gc_alloc_string(0); }\n");
    sb_append(sb, "    buffer[0] = '\\0';\n");
    sb_append(sb, "    struct dirent* entry;\n");
    sb_append(sb, "    while ((entry = readdir(dir)) != NULL) {\n");
    sb_append(sb, "        if (strcmp(entry->d_name, \".\") == 0 || strcmp(entry->d_name, \"..\") == 0) continue;\n");
    sb_append(sb, "        size_t name_len = strlen(entry->d_name);\n");
    sb_append(sb, "        size_t needed = used + name_len + 2; /* +1 for newline, +1 for null */\n");
    sb_append(sb, "        if (needed > capacity) {\n");
    sb_append(sb, "            capacity = needed * 2;\n");
    sb_append(sb, "            char* new_buffer = gc_alloc_string(capacity);\n");
    sb_append(sb, "            if (!new_buffer) { gc_release(buffer); closedir(dir); return gc_alloc_string(0); }\n");
    sb_append(sb, "            memcpy(new_buffer, buffer, used);\n");
    sb_append(sb, "            gc_release(buffer);\n");
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
    sb_append(sb, "    char temp_buffer[1024];\n");
    sb_append(sb, "    if (getcwd(temp_buffer, sizeof(temp_buffer)) == NULL) {\n");
    sb_append(sb, "        return gc_alloc_string(0);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    size_t len = strlen(temp_buffer);\n");
    sb_append(sb, "    char* buffer = gc_alloc_string(len);\n");
    sb_append(sb, "    if (buffer) memcpy(buffer, temp_buffer, len + 1);\n");
    sb_append(sb, "    return buffer ? buffer : gc_alloc_string(0);\n");
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
    sb_append(sb, "    if (!f) return gc_alloc_string(0);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_END);\n");
    sb_append(sb, "    long size = ftell(f);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_SET);\n");
    sb_append(sb, "    char* buffer = gc_alloc_string((size_t)size);\n");
    sb_append(sb, "    if (!buffer) { fclose(f); return gc_alloc_string(0); }\n");
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
    sb_append(sb, "    char* out = gc_alloc_string(len);\n");
    sb_append(sb, "    if (!out) return gc_alloc_string(0);\n");
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
    sb_append(sb, "    if (fd < 0) return gc_alloc_string(0);\n");
    sb_append(sb, "    close(fd);\n");
    sb_append(sb, "    size_t len = strlen(templ);\n");
    sb_append(sb, "    char* out = gc_alloc_string(len);\n");
    sb_append(sb, "    if (!out) return gc_alloc_string(0);\n");
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
    sb_append(sb, "            char* out = gc_alloc_string(len);\n");
    sb_append(sb, "            if (!out) return gc_alloc_string(0);\n");
    sb_append(sb, "            memcpy(out, path, len);\n");
    sb_append(sb, "            out[len] = '\\0';\n");
    sb_append(sb, "            return out;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return gc_alloc_string(0);\n");
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

    sb_append(sb, "/* Capture stdout from a shell command */\n");
    sb_append(sb, "const char* nl_exec_capture(const char* cmd) {\n");
    sb_append(sb, "    FILE* pipe = popen(cmd, \"r\");\n");
    sb_append(sb, "    if (!pipe) return \"\";\n");
    sb_append(sb, "    char* out = (char*)malloc(65536);\n");
    sb_append(sb, "    if (!out) { pclose(pipe); return \"\"; }\n");
    sb_append(sb, "    size_t total = 0;\n");
    sb_append(sb, "    while (total < 65535) {\n");
    sb_append(sb, "        size_t n = fread(out + total, 1, 65535 - total, pipe);\n");
    sb_append(sb, "        if (n == 0) break;\n");
    sb_append(sb, "        total += n;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    out[total] = '\\0';\n");
    sb_append(sb, "    pclose(pipe);\n");
    sb_append(sb, "    return out;\n");
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

    /* Timing utilities */
    generate_timing_utilities(sb);

    /* Console I/O utilities */
    generate_console_io_utilities(sb);

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

/* =============================================================================
 * Timing utilities for performance measurement
 * =============================================================================
 */

void generate_timing_utilities(StringBuilder *sb) {
    sb_append(sb, "/* ========== Timing Utilities ========== */\n\n");

    sb_append(sb, "#include <sys/time.h>\n");
    sb_append(sb, "#ifdef __MACH__\n");
    sb_append(sb, "#include <mach/mach_time.h>\n");
    sb_append(sb, "#endif\n\n");

    sb_append(sb, "/* Get current time in microseconds since epoch */\n");
    sb_append(sb, "static int64_t nl_timing_get_microseconds(void) {\n");
    sb_append(sb, "#ifdef CLOCK_REALTIME\n");
    sb_append(sb, "    struct timespec ts;\n");
    sb_append(sb, "    clock_gettime(CLOCK_REALTIME, &ts);\n");
    sb_append(sb, "    return ((int64_t)ts.tv_sec * 1000000LL) + (int64_t)(ts.tv_nsec / 1000);\n");
    sb_append(sb, "#else\n");
    sb_append(sb, "    struct timeval tv;\n");
    sb_append(sb, "    gettimeofday(&tv, NULL);\n");
    sb_append(sb, "    return ((int64_t)tv.tv_sec * 1000000LL) + (int64_t)tv.tv_usec;\n");
    sb_append(sb, "#endif\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* Get high-resolution time in nanoseconds */\n");
    sb_append(sb, "static int64_t nl_timing_get_nanoseconds(void) {\n");
    sb_append(sb, "#ifdef __MACH__\n");
    sb_append(sb, "    static mach_timebase_info_data_t timebase;\n");
    sb_append(sb, "    static int initialized = 0;\n");
    sb_append(sb, "    if (!initialized) {\n");
    sb_append(sb, "        mach_timebase_info(&timebase);\n");
    sb_append(sb, "        initialized = 1;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    uint64_t mach_time = mach_absolute_time();\n");
    sb_append(sb, "    return (int64_t)((mach_time * timebase.numer) / timebase.denom);\n");
    sb_append(sb, "#elif defined(CLOCK_MONOTONIC)\n");
    sb_append(sb, "    struct timespec ts;\n");
    sb_append(sb, "    clock_gettime(CLOCK_MONOTONIC, &ts);\n");
    sb_append(sb, "    return ((int64_t)ts.tv_sec * 1000000000LL) + (int64_t)ts.tv_nsec;\n");
    sb_append(sb, "#else\n");
    sb_append(sb, "    return nl_timing_get_microseconds() * 1000LL;\n");
    sb_append(sb, "#endif\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* Convenience: current time in milliseconds */\n");
    sb_append(sb, "static int64_t nl_get_time_ms(void) { return nl_timing_get_microseconds() / 1000LL; }\n\n");

    sb_append(sb, "/* ========== End Timing Utilities ========== */\n\n");
}

/* =============================================================================
 * Console I/O utilities for REPL and interactive programs
 * =============================================================================
 */

void generate_console_io_utilities(StringBuilder *sb) {
    sb_append(sb, "/* ========== Console I/O Utilities ========== */\n\n");

    sb_append(sb, "/* Read a line from stdin, returns heap-allocated string */\n");
    sb_append(sb, "/* Static to avoid duplicate symbols when linking multiple modules */\n");
    sb_append(sb, "static const char* nl_read_line(void) {\n");
    sb_append(sb, "    char buffer[4096];\n");
    sb_append(sb, "    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {\n");
    sb_append(sb, "        char* empty = malloc(1);\n");
    sb_append(sb, "        if (empty) empty[0] = '\\0';\n");
    sb_append(sb, "        return empty ? empty : \"\";\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    /* Remove trailing newline if present */\n");
    sb_append(sb, "    size_t len = strlen(buffer);\n");
    sb_append(sb, "    if (len > 0 && buffer[len-1] == '\\n') {\n");
    sb_append(sb, "        buffer[len-1] = '\\0';\n");
    sb_append(sb, "        len--;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    char* result = malloc(len + 1);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, buffer, len + 1);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* ========== End Console I/O Utilities ========== */\n\n");
}

/* =============================================================================
 * Cross-Platform Profiling System
 * macOS: Uses 'sample' command on child process
 * Linux: Uses 'gprofng collect' to wrap execution
 * Both output OS-neutral JSON for LLM analysis
 * =============================================================================
 */

void generate_profiling_system(StringBuilder *sb, const char *profile_output_path) {
    sb_append(sb, "/* ========== Cross-Platform Profiling System ========== */\n\n");
    
    sb_append(sb, "#include <sys/types.h>\n");
    sb_append(sb, "#include <sys/wait.h>\n");
    sb_append(sb, "#include <unistd.h>\n");
    sb_append(sb, "#include <signal.h>\n");
    sb_append(sb, "#include <stdarg.h>\n\n");

    /* Embed output path as a compile-time constant in the generated C */
    if (profile_output_path) {
        char path_decl[4096];
        snprintf(path_decl, sizeof(path_decl),
                 "static const char* _nl_profile_output_path = \"%s\";\n", profile_output_path);
        sb_append(sb, path_decl);
    } else {
        sb_append(sb, "static const char* _nl_profile_output_path = NULL;\n");
    }
    sb_append(sb, "static FILE* _nl_profile_file = NULL;\n\n");

    /* Helper: emit a formatted string to both stdout (decorated) and the profile file (clean) */
    sb_append(sb, "static void _nl_profile_emit(FILE* f, const char* fmt, ...) {\n");
    sb_append(sb, "    va_list ap;\n");
    sb_append(sb, "    va_start(ap, fmt);\n");
    sb_append(sb, "    vfprintf(stdout, fmt, ap);\n");
    sb_append(sb, "    va_end(ap);\n");
    sb_append(sb, "    if (f) {\n");
    sb_append(sb, "        va_start(ap, fmt);\n");
    sb_append(sb, "        vfprintf(f, fmt, ap);\n");
    sb_append(sb, "        va_end(ap);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");

    /* Common JSON output helpers */
    sb_append(sb, "/* Output JSON header for profile results */\n");
    sb_append(sb, "static void _nl_profile_json_header(const char* binary, const char* platform, const char* tool) {\n");
    sb_append(sb, "    if (_nl_profile_output_path && !_nl_profile_file) {\n");
    sb_append(sb, "        _nl_profile_file = fopen(_nl_profile_output_path, \"w\");\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    printf(\"\\n========== PROFILE ANALYSIS (LLM-READY JSON) ==========\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"{\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  \\\"profile_type\\\": \\\"sampling\\\",\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  \\\"platform\\\": \\\"%s\\\",\\n\", platform);\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  \\\"tool\\\": \\\"%s\\\",\\n\", tool);\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  \\\"binary\\\": \\\"%s\\\",\\n\", binary);\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  \\\"hotspots\\\": [\\n\");\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void _nl_profile_json_footer(void) {\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"\\n  ],\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  \\\"analysis_hints\\\": [\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"    \\\"Functions with high sample counts are hot spots\\\",\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"    \\\"Look for nl_ prefixed functions (NanoLang generated)\\\",\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"    \\\"str_ and array_ functions often indicate algorithmic issues\\\",\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"    \\\"Deep call stacks may indicate recursion or callback chains\\\"\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"  ]\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"}\\n\");\n");
    sb_append(sb, "    printf(\"========== END PROFILE ANALYSIS ==========\\n\\n\");\n");
    sb_append(sb, "    if (_nl_profile_file) { fclose(_nl_profile_file); _nl_profile_file = NULL; }\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void _nl_profile_json_entry(int* count, const char* func, int samples, double pct) {\n");
    sb_append(sb, "    if (*count > 0) _nl_profile_emit(_nl_profile_file, \",\\n\");\n");
    sb_append(sb, "    _nl_profile_emit(_nl_profile_file, \"    {\\\"function\\\": \\\"%s\\\", \\\"samples\\\": %d, \\\"pct_time\\\": %.1f}\", func, samples, pct);\n");
    sb_append(sb, "    (*count)++;\n");
    sb_append(sb, "}\n\n");

    /* macOS-specific: parse 'xctrace' output (with fallback to sample) */
    sb_append(sb, "#ifdef __APPLE__\n\n");

    sb_append(sb, "/* Parse macOS xctrace table output and convert to JSON */\n");
    sb_append(sb, "static void _nl_parse_xctrace_output(const char* trace_file, const char* binary) {\n");
    sb_append(sb, "    FILE* f = fopen(trace_file, \"r\");\n");
    sb_append(sb, "    if (!f) {\n");
    sb_append(sb, "        fprintf(stderr, \"\\n[profile] Could not read xctrace output.\\n\");\n");
    sb_append(sb, "        return;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    _nl_profile_json_header(binary, \"macOS\", \"xctrace\");\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parse xctrace table output */\n");
    sb_append(sb, "    char line[4096];\n");
    sb_append(sb, "    int count = 0;\n");
    sb_append(sb, "    int max_entries = 20;\n");
    sb_append(sb, "    int in_table = 0;\n");
    sb_append(sb, "    double total_time_ms = 0.0;\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parse table format: Symbol Name, Self Weight (ms), Total Weight (ms) */\n");
    sb_append(sb, "    while (fgets(line, sizeof(line), f) && count < max_entries) {\n");
    sb_append(sb, "        /* Look for table header */\n");
    sb_append(sb, "        if (strstr(line, \"Symbol Name\") && strstr(line, \"Weight\")) {\n");
    sb_append(sb, "            in_table = 1;\n");
    sb_append(sb, "            continue;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        if (!in_table) continue;\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Parse data lines: func_name  self_ms  total_ms */\n");
    sb_append(sb, "        char func[512] = {0};\n");
    sb_append(sb, "        double self_ms = 0.0, total_ms = 0.0;\n");
    sb_append(sb, "        if (sscanf(line, \"%511[^\\t]\\t%lf\\t%lf\", func, &self_ms, &total_ms) >= 2) {\n");
    sb_append(sb, "            /* Use self time for accurate hotspot identification */\n");
    sb_append(sb, "            if (self_ms > 0.0) {\n");
    sb_append(sb, "                total_time_ms += self_ms;\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Second pass: emit entries with >= 1% time */\n");
    sb_append(sb, "    if (total_time_ms == 0.0) total_time_ms = 1.0;\n");
    sb_append(sb, "    rewind(f);\n");
    sb_append(sb, "    in_table = 0;\n");
    sb_append(sb, "    count = 0;\n");
    sb_append(sb, "    while (fgets(line, sizeof(line), f) && count < max_entries) {\n");
    sb_append(sb, "        if (strstr(line, \"Symbol Name\") && strstr(line, \"Weight\")) {\n");
    sb_append(sb, "            in_table = 1;\n");
    sb_append(sb, "            continue;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        if (!in_table) continue;\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        char func[512] = {0};\n");
    sb_append(sb, "        double self_ms = 0.0, total_ms = 0.0;\n");
    sb_append(sb, "        if (sscanf(line, \"%511[^\\t]\\t%lf\\t%lf\", func, &self_ms, &total_ms) >= 2 && self_ms > 0.0) {\n");
    sb_append(sb, "            double pct = (100.0 * self_ms) / total_time_ms;\n");
    sb_append(sb, "            if (pct >= 1.0) {\n");
    sb_append(sb, "                /* Emit as call count (approximate from time) */\n");
    sb_append(sb, "                int call_count = (int)(self_ms * 1000); /* Rough estimate */\n");
    sb_append(sb, "                _nl_profile_json_entry(&count, func, call_count, pct);\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    _nl_profile_json_footer();\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* Parse macOS sample output and convert to JSON (fallback) */\n");
    sb_append(sb, "static void _nl_parse_sample_output(const char* sample_file, const char* binary) {\n");
    sb_append(sb, "    FILE* f = fopen(sample_file, \"r\");\n");
    sb_append(sb, "    if (!f) {\n");
    sb_append(sb, "        fprintf(stderr, \"\\n[profile] Could not read sample output.\\n\");\n");
    sb_append(sb, "        fprintf(stderr, \"[profile] The sample command may have failed or the process exited too quickly.\\n\");\n");
    sb_append(sb, "        fprintf(stderr, \"[profile] You can also use Instruments.app for detailed profiling.\\n\\n\");\n");
    sb_append(sb, "        return;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    _nl_profile_json_header(binary, \"macOS\", \"sample\");\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parse sample output - look for heaviest stack frames */\n");
    sb_append(sb, "    char line[4096];\n");
    sb_append(sb, "    int count = 0;\n");
    sb_append(sb, "    int max_entries = 20;\n");
    sb_append(sb, "    int total_samples = 0;\n");
    sb_append(sb, "    int in_call_graph = 0;\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* First pass: count total samples */\n");
    sb_append(sb, "    while (fgets(line, sizeof(line), f)) {\n");
    sb_append(sb, "        if (strstr(line, \"Total number in stack\")) {\n");
    sb_append(sb, "            /* Parse: Total number in stack (self-sampling) = 1234 */\n");
    sb_append(sb, "            char* eq = strchr(line, '=');\n");
    sb_append(sb, "            if (eq) total_samples = atoi(eq + 1);\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    if (total_samples == 0) total_samples = 1; /* Avoid div by zero */\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Rewind and parse call graph */\n");
    sb_append(sb, "    rewind(f);\n");
    sb_append(sb, "    while (fgets(line, sizeof(line), f) && count < max_entries) {\n");
    sb_append(sb, "        /* Look for Call graph section */\n");
    sb_append(sb, "        if (strstr(line, \"Call graph:\")) {\n");
    sb_append(sb, "            in_call_graph = 1;\n");
    sb_append(sb, "            continue;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        if (!in_call_graph) continue;\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Parse lines like: 1234 func_name (in binary) */\n");
    sb_append(sb, "        int samples = 0;\n");
    sb_append(sb, "        char func[512] = {0};\n");
    sb_append(sb, "        /* Skip leading whitespace and parse sample count + function */\n");
    sb_append(sb, "        char* p = line;\n");
    sb_append(sb, "        while (*p == ' ' || *p == '+' || *p == '!' || *p == '|') p++;\n");
    sb_append(sb, "        if (sscanf(p, \"%d %511[^( \\n]\", &samples, func) == 2 && samples > 0) {\n");
    sb_append(sb, "            double pct = (100.0 * samples) / total_samples;\n");
    sb_append(sb, "            if (pct >= 1.0) { /* Only show functions with >= 1% time */\n");
    sb_append(sb, "                _nl_profile_json_entry(&count, func, samples, pct);\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    _nl_profile_json_footer();\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* Check if xctrace is available AND functional (requires full Xcode) */\n");
    sb_append(sb, "static int _nl_has_xctrace(void) {\n");
    sb_append(sb, "    /* Check if xctrace exists */\n");
    sb_append(sb, "    int status = system(\"which xctrace >/dev/null 2>&1\");\n");
    sb_append(sb, "    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {\n");
    sb_append(sb, "        return 0; /* Not found */\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    /* Check if xctrace can actually run (needs Xcode, not just CLT) */\n");
    sb_append(sb, "    status = system(\"xctrace version >/dev/null 2>&1\");\n");
    sb_append(sb, "    return WIFEXITED(status) && WEXITSTATUS(status) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* macOS profiling wrapper using 'xctrace' (preferred) or 'sample' (fallback) */\n");
    sb_append(sb, "static int _nl_run_with_profiling(int argc, char** argv, int64_t (*real_main)(void)) {\n");
    sb_append(sb, "    (void)argc; /* unused */\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Try xctrace first (instrumentation-based profiling) */\n");
    sb_append(sb, "    if (_nl_has_xctrace()) {\n");
    sb_append(sb, "        fprintf(stderr, \"\\n[profile] Auto-profiling with xctrace (instrumentation)...\\n\\n\");\n");
    sb_append(sb, "        fflush(stderr);\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Generate unique trace file */\n");
    sb_append(sb, "        char trace_file[256], table_file[256];\n");
    sb_append(sb, "        pid_t pid = getpid();\n");
    sb_append(sb, "        snprintf(trace_file, sizeof(trace_file), \"/tmp/nanolang_trace_%d.trace\", pid);\n");
    sb_append(sb, "        snprintf(table_file, sizeof(table_file), \"/tmp/nanolang_table_%d.txt\", pid);\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Run program with xctrace */\n");
    sb_append(sb, "        pid_t child = fork();\n");
    sb_append(sb, "        if (child == 0) {\n");
    sb_append(sb, "            /* Child: run xctrace */\n");
    sb_append(sb, "            freopen(\"/dev/null\", \"w\", stdout); /* Suppress xctrace output */\n");
    sb_append(sb, "            freopen(\"/dev/null\", \"w\", stderr);\n");
    sb_append(sb, "            execlp(\"xctrace\", \"xctrace\", \"record\",\n");
    sb_append(sb, "                   \"--template\", \"Time Profiler\",\n");
    sb_append(sb, "                   \"--output\", trace_file,\n");
    sb_append(sb, "                   \"--launch\", \"--\", argv[0], (char*)NULL);\n");
    sb_append(sb, "            _exit(1);\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Parent: wait for xctrace to complete */\n");
    sb_append(sb, "        int status = 0;\n");
    sb_append(sb, "        waitpid(child, &status, 0);\n");
    sb_append(sb, "        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Export trace to table format */\n");
    sb_append(sb, "        char export_cmd[1024];\n");
    sb_append(sb, "        snprintf(export_cmd, sizeof(export_cmd),\n");
    sb_append(sb, "                 \"xctrace export --input %s --toc 2>/dev/null | \"\n");
    sb_append(sb, "                 \"grep -A 1000 'Time Profiler' | \"\n");
    sb_append(sb, "                 \"xctrace export --input %s --xpath '/trace-toc/run[@number=\\\"1\\\"]/data/table[@schema=\\\"time-profile\\\"]' > %s 2>/dev/null\",\n");
    sb_append(sb, "                 trace_file, trace_file, table_file);\n");
    sb_append(sb, "        system(export_cmd);\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Parse and output results */\n");
    sb_append(sb, "        _nl_parse_xctrace_output(table_file, argv[0]);\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Clean up */\n");
    sb_append(sb, "        unlink(trace_file);\n");
    sb_append(sb, "        unlink(table_file);\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        return exit_code;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Fallback to sample (sampling-based profiling) */\n");
    sb_append(sb, "    fprintf(stderr, \"\\n[profile] xctrace not found, falling back to sample (sampling)...\\n\\n\");\n");
    sb_append(sb, "    fflush(stderr);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Generate unique sample output file */\n");
    sb_append(sb, "    char sample_file[256];\n");
    sb_append(sb, "    pid_t main_pid = getpid();\n");
    sb_append(sb, "    snprintf(sample_file, sizeof(sample_file), \"/tmp/nanolang_sample_%d.txt\", main_pid);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fprintf(stderr, \"\\n[profile] Auto-profiling with macOS sample(1)...\\n\\n\");\n");
    sb_append(sb, "    fflush(stderr);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Use pipe to synchronize: child waits for sampler to attach */\n");
    sb_append(sb, "    int sync_pipe[2];\n");
    sb_append(sb, "    if (pipe(sync_pipe) < 0) {\n");
    sb_append(sb, "        perror(\"pipe failed\");\n");
    sb_append(sb, "        return (int)real_main(); /* Fall back to direct execution */\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Fork child first - it will wait for signal to start */\n");
    sb_append(sb, "    pid_t child = fork();\n");
    sb_append(sb, "    if (child < 0) {\n");
    sb_append(sb, "        perror(\"fork failed\");\n");
    sb_append(sb, "        close(sync_pipe[0]); close(sync_pipe[1]);\n");
    sb_append(sb, "        return (int)real_main(); /* Fall back to direct execution */\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    if (child == 0) {\n");
    sb_append(sb, "        /* Child process: wait for sampler to attach, then run */\n");
    sb_append(sb, "        close(sync_pipe[1]); /* Close write end */\n");
    sb_append(sb, "        char buf[1];\n");
    sb_append(sb, "        read(sync_pipe[0], buf, 1); /* Block until parent signals */\n");
    sb_append(sb, "        close(sync_pipe[0]);\n");
    sb_append(sb, "        int64_t result = real_main();\n");
    sb_append(sb, "        _exit((int)result);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parent: close read end */\n");
    sb_append(sb, "    close(sync_pipe[0]);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Fork sampler with child's PID */\n");
    sb_append(sb, "    pid_t sampler = fork();\n");
    sb_append(sb, "    if (sampler < 0) {\n");
    sb_append(sb, "        perror(\"fork failed\");\n");
    sb_append(sb, "        close(sync_pipe[1]);\n");
    sb_append(sb, "        waitpid(child, NULL, 0);\n");
    sb_append(sb, "        return 1;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    if (sampler == 0) {\n");
    sb_append(sb, "        /* Sampler child: run sample on child's PID */\n");
    sb_append(sb, "        close(sync_pipe[1]);\n");
    sb_append(sb, "        char pid_str[32];\n");
    sb_append(sb, "        snprintf(pid_str, sizeof(pid_str), \"%d\", child);\n");
    sb_append(sb, "        /* sample <pid> <duration> -f <output> -mayDie */\n");
    sb_append(sb, "        execlp(\"sample\", \"sample\", pid_str, \"60\", \"-f\", sample_file, \"-mayDie\", (char*)NULL);\n");
    sb_append(sb, "        _exit(1); /* exec failed */\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Give sample time to attach */\n");
    sb_append(sb, "    usleep(200000); /* 200ms */\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Signal child to start running */\n");
    sb_append(sb, "    write(sync_pipe[1], \"g\", 1);\n");
    sb_append(sb, "    close(sync_pipe[1]);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parent: wait for child to complete */\n");
    sb_append(sb, "    int status = 0;\n");
    sb_append(sb, "    waitpid(child, &status, 0);\n");
    sb_append(sb, "    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Give sample time to finish writing */\n");
    sb_append(sb, "    usleep(500000); /* 500ms */\n");
    sb_append(sb, "    kill(sampler, SIGTERM);\n");
    sb_append(sb, "    waitpid(sampler, NULL, 0);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parse and output results */\n");
    sb_append(sb, "    _nl_parse_sample_output(sample_file, argv[0]);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Clean up */\n");
    sb_append(sb, "    unlink(sample_file);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    return exit_code;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "#endif /* __APPLE__ */\n\n");
    
    /* Linux-specific: use gprofng */
    sb_append(sb, "#ifdef __linux__\n\n");
    
    sb_append(sb, "/* Parse gprofng output directory and convert to JSON */\n");
    sb_append(sb, "static void _nl_parse_gprofng_output(const char* exp_dir, const char* binary) {\n");
    sb_append(sb, "    _nl_profile_json_header(binary, \"Linux\", \"gprofng\");\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Run gprofng display to get function list */\n");
    sb_append(sb, "    char cmd[1024];\n");
    sb_append(sb, "    snprintf(cmd, sizeof(cmd), \"gprofng display text -functions '%s' 2>/dev/null\", exp_dir);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    FILE* pipe = popen(cmd, \"r\");\n");
    sb_append(sb, "    if (!pipe) {\n");
    sb_append(sb, "        fprintf(stderr, \"[profile] Failed to run gprofng display\\n\");\n");
    sb_append(sb, "        _nl_profile_json_footer();\n");
    sb_append(sb, "        return;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    char line[1024];\n");
    sb_append(sb, "    int count = 0;\n");
    sb_append(sb, "    int max_entries = 20;\n");
    sb_append(sb, "    int in_data = 0;\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    while (fgets(line, sizeof(line), pipe) && count < max_entries) {\n");
    sb_append(sb, "        /* Skip header lines until we see data */\n");
    sb_append(sb, "        if (strstr(line, \"Excl.\") || strstr(line, \"-----\")) {\n");
    sb_append(sb, "            in_data = 1;\n");
    sb_append(sb, "            continue;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        if (!in_data) continue;\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Parse: Excl_sec  Excl_%  Incl_sec  Incl_%  Name */\n");
    sb_append(sb, "        double excl_sec = 0, excl_pct = 0, incl_sec = 0, incl_pct = 0;\n");
    sb_append(sb, "        char func[512] = {0};\n");
    sb_append(sb, "        if (sscanf(line, \" %lf %lf %lf %lf %511s\", &excl_sec, &excl_pct, &incl_sec, &incl_pct, func) >= 5) {\n");
    sb_append(sb, "            if (excl_pct >= 1.0 && strlen(func) > 0) {\n");
    sb_append(sb, "                _nl_profile_json_entry(&count, func, (int)(excl_pct * 10), excl_pct);\n");
    sb_append(sb, "            }\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    pclose(pipe);\n");
    sb_append(sb, "    _nl_profile_json_footer();\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* Linux profiling wrapper using gprofng collect */\n");
    sb_append(sb, "static int _nl_run_with_profiling(int argc, char** argv, int64_t (*real_main)(void)) {\n");
    sb_append(sb, "    (void)real_main; /* Not used directly - we exec through gprofng */\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Check if we're already being profiled (avoid recursion) */\n");
    sb_append(sb, "    const char* ld_preload = getenv(\"LD_PRELOAD\");\n");
    sb_append(sb, "    if (getenv(\"_NL_PROFILING_CHILD\") || (ld_preload && strstr(ld_preload, \"libgp-collector\"))) {\n");
    sb_append(sb, "        /* We're already being profiled - just run normally */\n");
    sb_append(sb, "        return (int)real_main();\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Generate unique experiment directory */\n");
    sb_append(sb, "    char exp_dir[256];\n");
    sb_append(sb, "    snprintf(exp_dir, sizeof(exp_dir), \"/tmp/nanolang_gprofng_%d.er\", getpid());\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    pid_t child = fork();\n");
    sb_append(sb, "    if (child < 0) {\n");
    sb_append(sb, "        perror(\"fork failed\");\n");
    sb_append(sb, "        return 1;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    if (child == 0) {\n");
    sb_append(sb, "        /* Child: exec through gprofng collect */\n");
    sb_append(sb, "        setenv(\"_NL_PROFILING_CHILD\", \"1\", 1);\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        /* Build argv for gprofng: gprofng collect app -o <dir> <binary> <args...> */\n");
    sb_append(sb, "        char** new_argv = malloc((argc + 6) * sizeof(char*));\n");
    sb_append(sb, "        new_argv[0] = \"gprofng\";\n");
    sb_append(sb, "        new_argv[1] = \"collect\";\n");
    sb_append(sb, "        new_argv[2] = \"app\";\n");
    sb_append(sb, "        new_argv[3] = \"-o\";\n");
    sb_append(sb, "        new_argv[4] = exp_dir;\n");
    sb_append(sb, "        for (int i = 0; i < argc; i++) {\n");
    sb_append(sb, "            new_argv[5 + i] = argv[i];\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        new_argv[5 + argc] = NULL;\n");
    sb_append(sb, "        \n");
    sb_append(sb, "        execvp(\"gprofng\", new_argv);\n");
    sb_append(sb, "        /* If gprofng not available, just run directly */\n");
    sb_append(sb, "        _exit((int)real_main());\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parent: wait for child and parse results */\n");
    sb_append(sb, "    int status = 0;\n");
    sb_append(sb, "    waitpid(child, &status, 0);\n");
    sb_append(sb, "    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Parse and output results */\n");
    sb_append(sb, "    _nl_parse_gprofng_output(exp_dir, argv[0]);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Clean up experiment directory */\n");
    sb_append(sb, "    char rm_cmd[512];\n");
    sb_append(sb, "    snprintf(rm_cmd, sizeof(rm_cmd), \"rm -rf '%s' 2>/dev/null\", exp_dir);\n");
    sb_append(sb, "    system(rm_cmd);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    return exit_code;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "#endif /* __linux__ */\n\n");
    
    /* Fallback for other platforms */
    sb_append(sb, "#if !defined(__APPLE__) && !defined(__linux__)\n");
    sb_append(sb, "/* Fallback: just run the program without profiling */\n");
    sb_append(sb, "static int _nl_run_with_profiling(int argc, char** argv, int64_t (*real_main)(void)) {\n");
    sb_append(sb, "    (void)argc; (void)argv;\n");
    sb_append(sb, "    fprintf(stderr, \"[profile] Profiling not supported on this platform\\n\");\n");
    sb_append(sb, "    return (int)real_main();\n");
    sb_append(sb, "}\n");
    sb_append(sb, "#endif\n\n");
    
    sb_append(sb, "/* ========== End Cross-Platform Profiling System ========== */\n\n");
}

void generate_instrumented_profiling_system(StringBuilder *sb) {
    sb_append(sb, "/* ========== Instrumented Profiling System (--profile) ========== */\n\n");
    sb_append(sb, "#include <time.h>\n\n");
    sb_append(sb, "#define _NL_PROF_MAX_FUNCS 512\n\n");
    sb_append(sb, "typedef struct {\n");
    sb_append(sb, "    const char *name;\n");
    sb_append(sb, "    int64_t calls;\n");
    sb_append(sb, "    int64_t total_ns;\n");
    sb_append(sb, "} _NlProfEntry;\n\n");
    sb_append(sb, "static _NlProfEntry _nl_prof_table[_NL_PROF_MAX_FUNCS];\n");
    sb_append(sb, "static int _nl_prof_count = 0;\n\n");
    sb_append(sb, "static _NlProfEntry *_nl_prof_get_entry(const char *name) {\n");
    sb_append(sb, "    for (int i = 0; i < _nl_prof_count; i++) {\n");
    sb_append(sb, "        if (strcmp(_nl_prof_table[i].name, name) == 0) return &_nl_prof_table[i];\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    if (_nl_prof_count < _NL_PROF_MAX_FUNCS) {\n");
    sb_append(sb, "        _nl_prof_table[_nl_prof_count].name = name;\n");
    sb_append(sb, "        _nl_prof_table[_nl_prof_count].calls = 0;\n");
    sb_append(sb, "        _nl_prof_table[_nl_prof_count].total_ns = 0;\n");
    sb_append(sb, "        return &_nl_prof_table[_nl_prof_count++];\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return (void*)0;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static int _nl_prof_cmp(const void *a, const void *b) {\n");
    sb_append(sb, "    const _NlProfEntry *ea = (const _NlProfEntry *)a;\n");
    sb_append(sb, "    const _NlProfEntry *eb = (const _NlProfEntry *)b;\n");
    sb_append(sb, "    if (eb->total_ns > ea->total_ns) return 1;\n");
    sb_append(sb, "    if (eb->total_ns < ea->total_ns) return -1;\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static void _nl_prof_report(void) {\n");
    sb_append(sb, "    if (_nl_prof_count == 0) return;\n");
    sb_append(sb, "    qsort(_nl_prof_table, (size_t)_nl_prof_count, sizeof(_NlProfEntry), _nl_prof_cmp);\n");
    sb_append(sb, "    fprintf(stderr, \"\\n\");\n");
    sb_append(sb, "    fprintf(stderr, \"--- NanoLang Profile Report ---\\n\");\n");
    sb_append(sb, "    fprintf(stderr, \"%-32s  %10s  %14s  %14s\\n\",\n");
    sb_append(sb, "        \"Function\", \"Calls\", \"Total (ms)\", \"Avg (us)\");\n");
    sb_append(sb, "    fprintf(stderr, \"%-32s  %10s  %14s  %14s\\n\",\n");
    sb_append(sb, "        \"--------\", \"-----\", \"----------\", \"--------\");\n");
    sb_append(sb, "    for (int i = 0; i < _nl_prof_count; i++) {\n");
    sb_append(sb, "        if (_nl_prof_table[i].calls == 0) continue;\n");
    sb_append(sb, "        double total_ms = (double)_nl_prof_table[i].total_ns / 1000000.0;\n");
    sb_append(sb, "        double avg_us = (double)_nl_prof_table[i].total_ns\n");
    sb_append(sb, "            / (double)_nl_prof_table[i].calls / 1000.0;\n");
    sb_append(sb, "        fprintf(stderr, \"%-32s  %10lld  %14.3f  %14.3f\\n\",\n");
    sb_append(sb, "            _nl_prof_table[i].name,\n");
    sb_append(sb, "            (long long)_nl_prof_table[i].calls,\n");
    sb_append(sb, "            total_ms, avg_us);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    fprintf(stderr, \"--- End Profile Report ---\\n\");\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "typedef struct {\n");
    sb_append(sb, "    _NlProfEntry *entry;\n");
    sb_append(sb, "    struct timespec start;\n");
    sb_append(sb, "} _NlProfGuard;\n\n");
    sb_append(sb, "static void _nl_prof_guard_exit(_NlProfGuard *g) {\n");
    sb_append(sb, "    if (!g->entry) return;\n");
    sb_append(sb, "    struct timespec _nl_prof_end;\n");
    sb_append(sb, "    clock_gettime(CLOCK_MONOTONIC, &_nl_prof_end);\n");
    sb_append(sb, "    int64_t ns = (int64_t)(_nl_prof_end.tv_sec - g->start.tv_sec) * 1000000000LL\n");
    sb_append(sb, "               + (int64_t)(_nl_prof_end.tv_nsec - g->start.tv_nsec);\n");
    sb_append(sb, "    g->entry->calls++;\n");
    sb_append(sb, "    g->entry->total_ns += ns;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "/* ========== End Instrumented Profiling System ========== */\n\n");
}

/* Generate flamegraph collapsed-stack profiling report function.
 * Emits a _nl_prof_flamegraph_report() that writes flamegraph.pl-compatible
 * collapsed stack format to the specified file path (or <prog>.nano.prof).
 *
 * Flamegraph collapsed format (one line per sample):
 *   fn_name_a;fn_name_b count
 * For flat (non-nested) profiling we emit each function's call count as:
 *   fn_name count
 * Sorted by call count descending (hottest first).
 *
 * Usage: flamegraph.pl <input>.nano.prof > flame.svg && open flame.svg
 */
void generate_flamegraph_profiling_system(StringBuilder *sb, const char *flamegraph_path) {
    sb_append(sb, "\n/* ========== Flamegraph Profiling Output (--profile-runtime) ========== */\n\n");

    /* Emit the output path as a C string literal, or NULL for auto-derive */
    if (flamegraph_path && flamegraph_path[0]) {
        /* Build path literal inline; avoid sb_appendf (not in scope here) */
        char path_lit[4096 + 64];
        snprintf(path_lit, sizeof(path_lit),
                 "static const char *_nl_flamegraph_path = \"%s\";\n\n", flamegraph_path);
        sb_append(sb, path_lit);
    } else {
        sb_append(sb, "static const char *_nl_flamegraph_path = NULL; /* auto: <argv[0]>.nano.prof */\n\n");
    }

    /* Comparator by calls descending */
    sb_append(sb,
        "static int _nl_prof_cmp_calls(const void *a, const void *b) {\n"
        "    const _NlProfEntry *ea = (const _NlProfEntry *)a;\n"
        "    const _NlProfEntry *eb = (const _NlProfEntry *)b;\n"
        "    if (eb->calls > ea->calls) return 1;\n"
        "    if (eb->calls < ea->calls) return -1;\n"
        "    return 0;\n"
        "}\n\n"
    );

    sb_append(sb,
        "static void _nl_prof_flamegraph_report(void) {\n"
        "    if (_nl_prof_count == 0) return;\n"
        "\n"
        "    /* Determine output path */\n"
        "    char path_buf[4096];\n"
        "    const char *out_path = _nl_flamegraph_path;\n"
        "    if (!out_path || out_path[0] == '\\0') {\n"
        "        /* Derive from argv[0]: <prog>.nano.prof */\n"
        "        extern int g_argc;\n"
        "        extern char **g_argv;\n"
        "        const char *prog = (g_argc > 0 && g_argv) ? g_argv[0] : \"program\";\n"
        "        snprintf(path_buf, sizeof(path_buf), \"%s.nano.prof\", prog);\n"
        "        out_path = path_buf;\n"
        "    }\n"
        "\n"
        "    FILE *f = fopen(out_path, \"w\");\n"
        "    if (!f) {\n"
        "        fprintf(stderr, \"[nano-prof] warning: cannot open %s for writing\\n\", out_path);\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    /* Sort by calls descending */\n"
        "    qsort(_nl_prof_table, (size_t)_nl_prof_count, sizeof(_NlProfEntry), _nl_prof_cmp_calls);\n"
        "\n"
        "    /* Emit collapsed stack format: 'fn_name count\\n' */\n"
        "    int wrote = 0;\n"
        "    for (int i = 0; i < _nl_prof_count; i++) {\n"
        "        if (_nl_prof_table[i].calls == 0) continue;\n"
        "        fprintf(f, \"%s %lld\\n\",\n"
        "            _nl_prof_table[i].name,\n"
        "            (long long)_nl_prof_table[i].calls);\n"
        "        wrote++;\n"
        "    }\n"
        "    fclose(f);\n"
        "    if (wrote > 0) {\n"
        "        fprintf(stderr, \"[nano-prof] flamegraph data: %s (%d functions)\\n\", out_path, wrote);\n"
        "        fprintf(stderr, \"[nano-prof] render: flamegraph.pl %s > flame.svg\\n\", out_path);\n"
        "    }\n"
        "}\n\n"
    );

    sb_append(sb, "/* ========== End Flamegraph Profiling Output ========== */\n\n");
}
