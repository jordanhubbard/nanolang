/* eval_string.c - String operation built-in functions for interpreter
 * Extracted from eval.c for better organization
 *
 * Now delegates to nl_cstr_* shared primitives in runtime/nl_string.c.
 */

#define _POSIX_C_SOURCE 200809L

#include "eval_string.h"
#include "../nanolang.h"
#include "../runtime/nl_string.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ============================================================================
 * String Operations
 * ========================================================================== */

Value builtin_str_length(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: str_length requires string argument\n");
        return create_void();
    }
    assert(args[0].as.string_val != NULL);
    return create_int(nl_cstr_length(args[0].as.string_val));
}

Value builtin_str_concat(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_concat requires two string arguments\n");
        return create_void();
    }

    assert(args[0].as.string_val != NULL);
    assert(args[1].as.string_val != NULL);
    char *result = nl_cstr_concat(args[0].as.string_val, args[1].as.string_val);
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed in str_concat\n");
        return create_void();
    }

    return create_string(result);
}

Value builtin_str_substring(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: str_substring requires string as first argument\n");
        return create_void();
    }
    if (args[1].type != VAL_INT || args[2].type != VAL_INT) {
        fprintf(stderr, "Error: str_substring requires integer start and length\n");
        return create_void();
    }

    const char *str = args[0].as.string_val;
    assert(str != NULL);
    long long start = args[1].as.int_val;
    long long length = args[2].as.int_val;
    long long str_len = nl_cstr_length(str);

    if (start < 0 || start > str_len) {
        fprintf(stderr, "Error: str_substring start index out of bounds\n");
        return create_void();
    }

    if (length < 0) {
        fprintf(stderr, "Error: str_substring length cannot be negative\n");
        return create_void();
    }

    if (start == str_len) {
        if (length == 0) {
            return create_string("");
        }
        fprintf(stderr, "Error: str_substring start index out of bounds\n");
        return create_void();
    }

    char *result = nl_cstr_substring(str, start, length);
    return create_string(result);
}

Value builtin_str_contains(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_contains requires two string arguments\n");
        return create_void();
    }

    return create_bool(nl_cstr_contains(args[0].as.string_val, args[1].as.string_val));
}

Value builtin_str_equals(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_equals requires two string arguments\n");
        return create_void();
    }

    return create_bool(strcmp(args[0].as.string_val, args[1].as.string_val) == 0);
}

Value builtin_str_starts_with(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_starts_with requires two string arguments\n");
        return create_void();
    }
    const char *s = args[0].as.string_val;
    const char *prefix = args[1].as.string_val;
    size_t slen = strlen(s);
    size_t plen = strlen(prefix);
    if (plen > slen) return create_bool(false);
    return create_bool(strncmp(s, prefix, plen) == 0);
}

Value builtin_str_ends_with(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_ends_with requires two string arguments\n");
        return create_void();
    }
    const char *s = args[0].as.string_val;
    const char *suffix = args[1].as.string_val;
    size_t slen = strlen(s);
    size_t suflen = strlen(suffix);
    if (suflen > slen) return create_bool(false);
    if (suflen == 0) return create_bool(true);
    return create_bool(strncmp(s + slen - suflen, suffix, suflen) == 0);
}

Value builtin_str_index_of(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_index_of requires two string arguments\n");
        return create_void();
    }
    const char *haystack = args[0].as.string_val;
    const char *needle = args[1].as.string_val;
    const char *p = strstr(haystack, needle);
    if (!p) return create_int(-1);
    return create_int((long long)(p - haystack));
}

Value builtin_str_trim(Value *args) {
    if (args[0].type != VAL_STRING) { fprintf(stderr, "Error: str_trim requires string\n"); return create_void(); }
    const char *s = args[0].as.string_val;
    size_t len = strlen(s);
    size_t start = 0;
    while (start < len && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
    size_t end = len;
    while (end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r')) end--;
    size_t new_len = end - start;
    char *result = malloc(new_len + 1);
    if (!result) return create_string("");
    memcpy(result, s + start, new_len);
    result[new_len] = '\0';
    Value v = create_string(result);
    free(result);
    return v;
}

Value builtin_str_trim_left(Value *args) {
    if (args[0].type != VAL_STRING) { fprintf(stderr, "Error: str_trim_left requires string\n"); return create_void(); }
    const char *s = args[0].as.string_val;
    size_t len = strlen(s);
    size_t start = 0;
    while (start < len && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
    size_t new_len = len - start;
    char *result = malloc(new_len + 1);
    if (!result) return create_string("");
    memcpy(result, s + start, new_len);
    result[new_len] = '\0';
    Value v = create_string(result);
    free(result);
    return v;
}

Value builtin_str_trim_right(Value *args) {
    if (args[0].type != VAL_STRING) { fprintf(stderr, "Error: str_trim_right requires string\n"); return create_void(); }
    const char *s = args[0].as.string_val;
    size_t len = strlen(s);
    size_t end = len;
    while (end > 0 && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r')) end--;
    char *result = malloc(end + 1);
    if (!result) return create_string("");
    memcpy(result, s, end);
    result[end] = '\0';
    Value v = create_string(result);
    free(result);
    return v;
}

Value builtin_str_to_lower(Value *args) {
    if (args[0].type != VAL_STRING) { fprintf(stderr, "Error: str_to_lower requires string\n"); return create_void(); }
    const char *s = args[0].as.string_val;
    size_t len = strlen(s);
    char *result = malloc(len + 1);
    if (!result) return create_string("");
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        result[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
    }
    result[len] = '\0';
    Value v = create_string(result);
    free(result);
    return v;
}

Value builtin_str_to_upper(Value *args) {
    if (args[0].type != VAL_STRING) { fprintf(stderr, "Error: str_to_upper requires string\n"); return create_void(); }
    const char *s = args[0].as.string_val;
    size_t len = strlen(s);
    char *result = malloc(len + 1);
    if (!result) return create_string("");
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        result[i] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : (char)c;
    }
    result[len] = '\0';
    Value v = create_string(result);
    free(result);
    return v;
}

Value builtin_str_replace(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_STRING) {
        fprintf(stderr, "Error: str_replace requires three string arguments\n");
        return create_void();
    }
    const char *s = args[0].as.string_val;
    const char *old_str = args[1].as.string_val;
    const char *new_str = args[2].as.string_val;
    size_t old_len = strlen(old_str);
    size_t new_len = strlen(new_str);
    size_t s_len = strlen(s);
    if (old_len == 0) return create_string(s);
    /* Count occurrences */
    int64_t count = 0;
    const char *p = s;
    const char *found;
    while ((found = strstr(p, old_str)) != NULL) { count++; p = found + old_len; }
    if (count == 0) return create_string(s);
    int64_t result_len = (int64_t)s_len + count * ((int64_t)new_len - (int64_t)old_len);
    if (result_len < 0) return create_string("");
    char *result = malloc((size_t)result_len + 1);
    if (!result) return create_string(s);
    const char *src = s;
    char *dst = result;
    while ((found = strstr(src, old_str)) != NULL) {
        size_t seg_len = (size_t)(found - src);
        memcpy(dst, src, seg_len); dst += seg_len;
        memcpy(dst, new_str, new_len); dst += new_len;
        src = found + old_len;
    }
    size_t rest = strlen(src);
    memcpy(dst, src, rest);
    dst[rest] = '\0';
    Value v = create_string(result);
    free(result);
    return v;
}
