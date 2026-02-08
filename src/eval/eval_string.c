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
