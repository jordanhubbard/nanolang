/* I exercise my actual interpreted payload dispatcher and GC finalizer. */
#include "nanolang.h"
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool g_in_shadow_tests = true;
static int g_shadow_current_fail_count;
static int g_shadow_current_first_line;
static int g_shadow_current_first_column;
#include "eval/bstring_intrinsics.inc"

static Value invoke(const char *name, Value *args, int count) {
    ASTNode node = {0};
    node.type = AST_CALL;
    node.line = 7; node.column = 2;
    node.as.call.arg_count = count;
    const NanoBstringContract *contract = nano_bstring_source_contract(name);
    assert(contract);
    return eval_bstring_intrinsic(&node, contract, args);
}

static Value text(const char *bytes) {
    Value string = create_string(bytes);
    Value value = invoke("bstr_new", &string, 1);
    assert(value.type == VAL_BSTRING);
    return value;
}

int main(void) {
    gc_init();
    Value value = text("Aé茶");
    assert(invoke("bstr_length", &value, 1).as.int_val == 6);
    assert(invoke("bstr_utf8_length", &value, 1).as.int_val == -1);
    assert(invoke("bstr_validate_utf8", &value, 1).as.bool_val);
    assert(invoke("bstr_utf8_length", &value, 1).as.int_val == 3);
    Value args[] = {value, create_int(2), create_int(INT64_MAX)};
    assert(invoke("bstr_utf8_char_at", args, 2).as.int_val == 33590);
    args[1] = create_int(1);
    assert(invoke("bstr_byte_at", args, 2).as.int_val == 195);
    Value tail = invoke("bstr_substring", args, 3);
    assert(invoke("bstr_length", &tail, 1).as.int_val == 5);
    Value snapshot = invoke("bstr_to_cstr", &value, 1);
    Value equal[] = {value, value};
    assert(invoke("bstr_equals", equal, 2).as.bool_val);
    Value concatenated = invoke("bstr_concat", equal, 2);
    assert(invoke("bstr_length", &concatenated, 1).as.int_val == 12);
    invoke("bstr_free", &value, 1);
    assert(!strcmp(snapshot.as.string_val, "Aé茶"));
    assert(invoke("bstr_length", &value, 1).type == VAL_VOID);
    assert(g_shadow_current_fail_count == 1);
    assert(g_shadow_current_first_line == 7 && g_shadow_current_first_column == 2);
    invoke("bstr_free", &value, 1);
    assert(g_shadow_current_fail_count == 2);
    Value wrong = create_int(1);
    assert(invoke("bstr_length", &wrong, 1).type == VAL_VOID);
    Value forged = eval_bstring_borrow((void *)(uintptr_t)1);
    assert(invoke("bstr_length", &forged, 1).type == VAL_VOID);
    Value wrong_owner = eval_bstring_borrow(snapshot.as.string_val);
    assert(invoke("bstr_length", &wrong_owner, 1).type == VAL_VOID);
    assert(invoke("bstr_new", NULL, 0).type == VAL_VOID);
    assert(g_shadow_current_fail_count == 6);
    assert(invoke("bstr_new", NULL, 1).type == VAL_VOID);
    assert(g_shadow_current_fail_count == 7);
    Value array = create_array(VAL_INT, 3, 3);
    long long *bytes = array.as.array_val->data;
    bytes[0] = 65; bytes[1] = 0; bytes[2] = 66;
    Value fallback = invoke("bstr_new_binary", &array, 1);
    assert(invoke("bstr_length", &fallback, 1).as.int_val == 0);
    array.as.array_val->is_byte_array = true;
    Value binary = invoke("bstr_new_binary", &array, 1);
    assert(invoke("bstr_length", &binary, 1).as.int_val == 3);
    Value binary_args[] = {binary, create_int(1), create_int(2)};
    assert(invoke("bstr_byte_at", binary_args, 2).as.int_val == 0);
    Value suffix = invoke("bstr_substring", binary_args, 3);
    assert(invoke("bstr_length", &suffix, 1).as.int_val == 2);
    bytes[0] = 255;
    Value invalid = invoke("bstr_new_binary", &array, 1);
    assert(!invoke("bstr_validate_utf8", &invalid, 1).as.bool_val);
    assert(invoke("bstr_utf8_length", &invalid, 1).as.int_val == -1);
    free(array.as.array_val->data); free(array.as.array_val);
    invoke("bstr_free", &binary, 1); invoke("bstr_free", &suffix, 1);
    invoke("bstr_free", &tail, 1); invoke("bstr_free", &fallback, 1);
    invoke("bstr_free", &invalid, 1); invoke("bstr_free", &concatenated, 1);
    /* I leave a live owner to exercise shutdown finalization as well as explicit free. */
    Value live = text("shutdown");
    assert(invoke("bstr_length", &live, 1).as.int_val == 8);
    gc_shutdown();
    GCStats stats = gc_get_stats();
    assert(stats.num_objects == 0 && stats.current_usage == 0);
    assert(stats.total_allocated == stats.total_freed);
    gc_shutdown();
    GCStats twice = gc_get_stats();
    assert(twice.total_allocated == stats.total_allocated && twice.total_freed == stats.total_freed);
    assert(twice.num_objects == 0 && twice.current_usage == 0);
    puts("I preserve interpreted binary payloads, type/owner checks, shadow failures and finalizer cleanup.");
    return 0;
}
