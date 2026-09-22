#define _POSIX_C_SOURCE 200809L  /* For mkstemp/mkdtemp */

#include "nanolang.h"
#include "eval_u8.h"
#include "string_literal_decode.h"
#include "binary64_bits.h"
#include "binary64_format.h"
#include "binary64_arithmetic.h"
#include "runtime/binary64_parse.h"
#include "coroutine.h"
#include "effects.h"
#include "runtime/list_int.h"
#include "runtime/list_string.h"
#include "runtime/list_token.h"
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
#include "runtime/shadow_timing.h"
#include "tracing.h"
#include "interpreter_ffi.h"
#include "eval/eval_hashmap.h"
#include "eval/eval_math.h"
#include "eval/eval_string.h"
#include "eval/eval_io.h"
#include "utf8.h"
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <libgen.h>
#include <sys/wait.h>
#include <spawn.h>
#include <math.h>
#include <limits.h>

/* g_argc/g_argv are defined in main.c / nano_main.c */
extern int g_argc;
extern char **g_argv;

/* DAP hook — set by dap_server.c to intercept each statement for breakpoints/stepping.
 * NULL when not debugging. */
void (*g_dap_statement_hook)(ASTNode *stmt, Environment *env) = NULL;

/* ── Coroutine spawn helpers ─────────────────────────────────────────── */

/* Argument bundle for spawned coroutines */
typedef struct {
    char *func_name;    /* Name of the async function to call */
    Value *args;        /* Argument array (malloc'd copy) */
    bool *owned_args;   /* Immediate-await borrow formals retain caller identity. */
    int arg_count;
    Environment *env;   /* One checked lease after preparation. */
    bool leased;
} CoroCallArgs;

/* Top-level callable values already own metadata at the public call boundary.
 * Nested callable/reference fields keep the separate borrowed graph contract. */
static void eval_owned_task_drop(Value value) {
    if (value.type == VAL_FUNCTION) {
        free((char *)value.as.function_val.function_name);
        free_function_signature(value.as.function_val.signature);
    } else env_discard_value_snapshot(value);
}

static bool eval_owned_task_clone(Value source, Value *out) {
    if (!out) return false;
    if (source.type != VAL_FUNCTION) return env_clone_value_snapshot(source, out);
    if (!source.as.function_val.function_name) return false;
    char *name = strdup(source.as.function_val.function_name);
    if (!name) return false;
    FunctionSignature *signature = NULL;
    if (!copy_function_signature_checked(source.as.function_val.signature, &signature)) {
        free(name);
        return false;
    }
    Value copy = {0};
    copy.type = VAL_FUNCTION;
    copy.as.function_val.function_name = name;
    copy.as.function_val.signature = signature;
    *out = copy;
    return true;
}

static void coro_bundle_drop(void *raw) {
    CoroCallArgs *ca = raw;
    if (!ca) return;
    for (int i = 0; i < ca->arg_count; ++i) {
        if (!ca->owned_args[i]) continue;
        Value value = ca->args[i];
        if (value.type == VAL_FUNCTION) {
            free((char *)value.as.function_val.function_name);
            free_function_signature(value.as.function_val.signature);
        } else env_discard_value_snapshot(value);
    }
    free(ca->args);
    free(ca->owned_args);
    free(ca->func_name);
    if (ca->leased) env_release_evaluation_lease(ca->env);
    free(ca);
}

static CoroCallArgs *coro_bundle_new(Environment *env, const char *name, int argc) {
    if (!env || !name || argc < 0 || (size_t)argc > SIZE_MAX / sizeof(Value)) return NULL;
    CoroCallArgs *ca = calloc(1, sizeof(*ca));
    if (!ca) return NULL;
    ca->env = env;
    ca->func_name = strdup(name);
    ca->args = calloc(argc ? (size_t)argc : 1, sizeof(Value));
    ca->owned_args = calloc(argc ? (size_t)argc : 1, sizeof(bool));
    if (!ca->func_name || !ca->args || !ca->owned_args) { coro_bundle_drop(ca); return NULL; }
    ca->arg_count = argc;
    for (int i = 0; i < argc; ++i) ca->args[i].type = VAL_VOID;
    return ca;
}

static bool coro_bundle_argument(CoroCallArgs *ca, int index, Value value, bool borrowed) {
    if (borrowed) { ca->args[index] = value; return true; }
    if (!eval_owned_task_clone(value, &ca->args[index])) return false;
    ca->owned_args[index] = true;
    return true;
}

/* The scheduler owns cleanup after successful enqueue, including ERROR/cancel. */
static Value coro_trampoline(void *raw_arg, int coro_id) {
    (void)coro_id;
    CoroCallArgs *ca = raw_arg;
    return call_function(ca->func_name, ca->args, ca->arg_count, ca->env);
}

static int coro_bundle_enqueue(CoroCallArgs *ca) {
    if (!env_acquire_evaluation_lease(ca->env)) return -1;
    ca->leased = true;
    return nano_coro_spawn_owned(coro_trampoline, ca, coro_bundle_drop,
        eval_owned_task_drop, eval_owned_task_clone);
}

static Value eval_task_result(Environment *env, int id, bool await) {
    Value result;
    bool ok = await ? nano_coro_await_copy(id, &result) : nano_coro_result_copy(id, &result);
    if (!ok) {
        fprintf(stderr, "I cannot copy a completed owned task result.\n"); exit(1);
    }
    if (result.type == VAL_STRUCT || result.type == VAL_TUPLE) {
        if (!env_retire_value(env, result)) {
            env_discard_value_snapshot(result);
            fprintf(stderr, "I cannot retain a copied task result.\n"); exit(1);
        }
    }
    return result;
}

typedef struct {
    const char *test_name;
    const char *source_file;
    int first_line;
    int first_column;
    int fail_count;
} ShadowFailure;



static bool g_in_shadow_tests = false;
static int g_shadow_current_fail_count = 0;
static int g_shadow_current_first_line = 0;
static int g_shadow_current_first_column = 0;
/* Like shadow accounting, my interpreted call context is sequential. */
static ASTNode *g_eval_call_site = NULL;
/* I borrow stack-local call identities only while their activations are live. */
static const void *g_eval_return_target = NULL;

static Value call_function_at(const char *name, Value *args, int arg_count,
                             Environment *env, int line, int column);

/* I keep checked dispatch and shadow failure accounting shared by every call route. */
static Value eval_foreign_call(Function *func, Value *args, int arg_count,
                               Environment *env, int line, int column) {
    bool success = false;
    Value result = ffi_call_extern_checked(func->name, args, arg_count, func, env, &success);
    if (!success && g_in_shadow_tests) {
        g_shadow_current_fail_count++;
        if (g_shadow_current_first_line == 0) {
            g_shadow_current_first_line = line;
            g_shadow_current_first_column = column;
        }
    }
    return result;
}

static void shadow_json_escape(FILE *out, const char *s) {
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", (unsigned int)c);
                else fputc((int)c, out);
        }
    }
}

static bool shadow_write_json_file(const char *path, const ShadowFailure *fails, int fail_len, bool success, int test_count) {
    if (!path || path[0] == '\0') return true;
    FILE *f = fopen(path, "w");
    if (!f) return false;

    fprintf(f, "{");
    fprintf(f, "\"tool\":\"nanoc_c\",");
    fprintf(f, "\"success\":%s,", success ? "true" : "false");
    fprintf(f, "\"completed\":true,");
    fprintf(f, "\"test_count\":%d,", test_count);
    fprintf(f, "\"failures\":[");
    for (int i = 0; i < fail_len; i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "{");
        fprintf(f, "\"test\":\""); shadow_json_escape(f, fails[i].test_name); fprintf(f, "\",");
        fprintf(f, "\"source_file\":\""); shadow_json_escape(f, fails[i].source_file); fprintf(f, "\",");
        fprintf(f, "\"fail_count\":%d,", fails[i].fail_count);
        fprintf(f, "\"first_location\":{");
        fprintf(f, "\"line\":%d,", fails[i].first_line);
        fprintf(f, "\"column\":%d", fails[i].first_column);
        fprintf(f, "}");
        fprintf(f, "}");
    }
    fprintf(f, "]}");
    bool written = !ferror(f);
    if (fclose(f) != 0) written = false;
    return written;
}


/* Process escape sequences in a raw lexer string into actual characters */
char *nl_unescape_string(const char *raw) {
    return nl_decode_string_literal(raw);
}

/* Forward declarations */
static Value eval_expression(ASTNode *expr, Environment *env);
static Value eval_statement(ASTNode *stmt, Environment *env);

static bool eval_match_or_pattern(const char *pattern, const char *variant) {
    if (!pattern || !variant || strncmp(pattern, "OR:", 3) != 0) return false;
    const char *part = pattern + 3;
    size_t variant_length = strlen(variant);
    while (*part) {
        const char *end = strchr(part, ':');
        size_t part_length = end ? (size_t)(end - part) : strlen(part);
        if (part_length == variant_length &&
            strncmp(part, variant, part_length) == 0) return true;
        if (!end) break;
        part = end + 1;
    }
    return false;
}

static bool eval_match_pattern(const Value *value, const char *pattern) {
    if (!value || !pattern) return false;
    if (strcmp(pattern, "_") == 0) return true;
    if (value->type == VAL_UNION) {
        UnionValue *union_value = value->as.union_val;
        if (!union_value || !union_value->variant_name) return false;
        return eval_match_or_pattern(pattern, union_value->variant_name) ||
               strcmp(pattern, union_value->variant_name) == 0;
    }
    if (strncmp(pattern, "INT:", 4) == 0) {
        long long expected = strtoll(pattern + 4, NULL, 10);
        if (value->type == VAL_INT) return value->as.int_val == expected;
        if (value->type == VAL_FLOAT) return (long long)value->as.float_val == expected;
        return false;
    }
    if (value->type == VAL_BOOL)
        return strcmp(pattern, value->as.bool_val ? "true" : "false") == 0;
    if (value->type == VAL_STRING && value->as.string_val)
        return strcmp(pattern, value->as.string_val) == 0;
    return false;
}

static Value eval_match_invariant_failure(const char *reason) {
    fprintf(stderr, "I cannot continue: %s.\n", reason);
    fflush(stderr);
    exit(EXIT_FAILURE);
    return create_void();
}

/* I own only a directly constructed empty union, never an alias or payload. */
static bool eval_match_owns_empty_literal(const ASTNode *scrutinee, Value value) {
    if (!scrutinee || value.type != VAL_UNION || !value.as.union_val)
        return false;
    bool literal =
        (scrutinee->type == AST_STRUCT_LITERAL &&
         scrutinee->as.struct_literal.field_count == 0) ||
        (scrutinee->type == AST_UNION_CONSTRUCT &&
         scrutinee->as.union_construct.field_count == 0);
    UnionValue *u = value.as.union_val;
    return literal && u->field_count == 0 &&
           !u->field_names && !u->field_values;
}

static void eval_match_release_empty_literal(Value value, bool owned, Value result) {
    if (!owned || (result.type == VAL_UNION &&
                   result.as.union_val == value.as.union_val)) return;
    UnionValue *u = value.as.union_val;
    free(u->union_name);
    free(u->variant_name);
    free(u);
}

/* I retire only owned names; values and declaration type facts are separate. */
static void eval_match_pop_metadata(Environment *env, int first) {
    for (int i = first; i < env->symbol_count; ++i) {
        free(env->symbols[i].name);
        free(env->symbols[i].struct_type_name);
        env->symbols[i].name = NULL;
        env->symbols[i].struct_type_name = NULL;
    }
    env->symbol_count = first;
}

/* I restore lexical bindings on every exit, retaining a yielded local string. */
/* I release only binding-owned storage. Registry result snapshots are never
 * installed directly into an owning record binding. Borrow formals stay borrowed. */
static void eval_scope_release(Environment *env, int first, bool functions) {
    for (int i = first; i < env->symbol_count; ++i) {
        Symbol *symbol = &env->symbols[i];
        bool borrowed = symbol->type == TYPE_BORROW_SHARED || symbol->type == TYPE_BORROW_MUT;
        Value value = symbol->value;
        if (!borrowed && (value.type == VAL_STRUCT || value.type == VAL_TUPLE)) {
            if (!env_retire_value(env, value)) {
                fprintf(stderr, "I cannot retire an owned record binding.\n"); exit(1);
            }
            symbol->value = create_void();
        }
        free(symbol->name);
        free(symbol->struct_type_name);
        if (borrowed) continue;
        if (value.type == VAL_STRUCT || value.type == VAL_TUPLE) { /* Its unique owner is now my retirement entry. */ }
        else if (value.type == VAL_STRING) {
            if (gc_is_managed(value.as.string_val)) gc_release(value.as.string_val);
            else free(value.as.string_val);
        } else if (functions && value.type == VAL_FUNCTION) {
            free((char *)value.as.function_val.function_name);
            free_function_signature(value.as.function_val.signature);
        }
    }
    env->symbol_count = first;
    /* My index keeps numeric links and hashes, not freed names. Its next sync
     * pops these slots before lookup or ordinary insertion reuses them. */
}

static Value eval_preserve_value(Environment *env, Value value) {
    if ((value.type != VAL_STRUCT && value.type != VAL_TUPLE && value.type != VAL_STRING) || env_record_result_borrowed(env, value)) return value;
    Value copy;
    if (!env_value_snapshot(env, value, &copy)) {
        fprintf(stderr, "I cannot preserve a record result across its scope.\n");
        exit(1);
    }
    copy.is_return = value.is_return;
    copy.return_target = value.return_target;
    copy.is_break = value.is_break;
    copy.is_continue = value.is_continue;
    return copy;
}

/* I capture the formal kind before evaluation can move function tables. */
static Value eval_staged_argument(ASTNode *expression, Environment *env,
                                  const char *callee, int index) {
    Function *function = env_get_function(env, callee);
    Type formal = function && function->params && index >= 0 && index < function->param_count
        ? function->params[index].type : TYPE_UNKNOWN;
    Value value = eval_expression(expression, env);
    if (value.is_return || value.is_break || value.is_continue ||
        formal == TYPE_BORROW_SHARED || formal == TYPE_BORROW_MUT) return value;
    return eval_preserve_value(env, value);
}

static Value eval_scoped_block(ASTNode **statements, int count, Environment *env) {
    int first = env->symbol_count;
    Value result = create_void();
    for (int i = 0; i < count; ++i) {
        result = eval_statement(statements[i], env);
        if (result.is_return || result.is_break || result.is_continue) break;
    }
    if (result.type == VAL_STRING) {
        for (int i = first; i < env->symbol_count; ++i) {
            if (env->symbols[i].value.type == VAL_STRING &&
                env->symbols[i].value.as.string_val == result.as.string_val) {
                Value copy = create_string(result.as.string_val);
                result.as.string_val = copy.as.string_val;
                break;
            }
        }
    }
    result = eval_preserve_value(env, result);
    /* Existing callable values may also borrow a local binding. */
    if (result.type == VAL_FUNCTION) {
        Value copy = create_function(result.as.function_val.function_name,
            copy_function_signature(result.as.function_val.signature));
        copy.is_return = result.is_return;
        copy.return_target = result.return_target;
        result = copy;
    }
    eval_scope_release(env, first, false);
    return result;
}
static Value create_dyn_array(DynArray *arr);

/* I reconstruct the modular signed value without an out-of-range cast. */
static int64_t eval_int_bits(uint64_t bits) {
    return bits <= INT64_MAX ? (int64_t)bits : -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
}
static int64_t eval_int_add(int64_t a, int64_t b) { return eval_int_bits((uint64_t)a + (uint64_t)b); }
static int64_t eval_int_sub(int64_t a, int64_t b) { return eval_int_bits((uint64_t)a - (uint64_t)b); }
static int64_t eval_int_mul(int64_t a, int64_t b) { return eval_int_bits((uint64_t)a * (uint64_t)b); }
static int64_t eval_int_div(int64_t a, int64_t b) {
    if (!b) return 0;
    return a == INT64_MIN && b == -1 ? INT64_MIN : a / b;
}
static int64_t eval_int_rem(int64_t a, int64_t b) {
    if (!b || (a == INT64_MIN && b == -1)) return 0;
    return a % b;
}

static DynArray* eval_dyn_array_binop(DynArray *a, DynArray *b, TokenType op);
static DynArray* eval_dyn_array_scalar_right(DynArray *a, Value scalar, TokenType op);
static DynArray* eval_dyn_array_scalar_left(Value scalar, DynArray *a, TokenType op);

static DynArray* eval_dyn_array_binop(DynArray *a, DynArray *b, TokenType op) {
    if (!a || !b) return NULL;
    int64_t len = dyn_array_length(a);
    if (len != dyn_array_length(b)) return NULL;
    ElementType t = dyn_array_get_elem_type(a);
    if (t != dyn_array_get_elem_type(b)) return NULL;

    DynArray *out = dyn_array_new_with_capacity(t, len);
    if (!out) return NULL;

    /* Switch on type/op outside the loop so the compiler sees a clean vectorizable loop */
    if (t == ELEM_INT) {
        int64_t *__restrict__ pa = (int64_t*)a->data;
        int64_t *__restrict__ pb = (int64_t*)b->data;
        int64_t *__restrict__ po = (int64_t*)out->data;
        switch (op) {
            case TOKEN_PLUS:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_add(pa[i], pb[i]);
                break;
            case TOKEN_MINUS:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_sub(pa[i], pb[i]);
                break;
            case TOKEN_STAR:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_mul(pa[i], pb[i]);
                break;
            case TOKEN_SLASH:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_div(pa[i], pb[i]);
                break;
            case TOKEN_PERCENT:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_rem(pa[i], pb[i]);
                break;
            default: break;
        }
        out->length = len;
    } else if (t == ELEM_FLOAT) {
        double *__restrict__ pa = (double*)a->data;
        double *__restrict__ pb = (double*)b->data;
        double *__restrict__ po = (double*)out->data;
        switch (op) {
            case TOKEN_PLUS:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_add(pa[i], pb[i]);
                break;
            case TOKEN_MINUS:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_sub(pa[i], pb[i]);
                break;
            case TOKEN_STAR:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_mul(pa[i], pb[i]);
                break;
            case TOKEN_SLASH:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_div(pa[i], pb[i]);
                break;
            default: break;
        }
        out->length = len;
    } else if (t == ELEM_STRING) {
        if (op != TOKEN_PLUS) return NULL;
        for (int64_t i = 0; i < len; i++) {
            const char *x = dyn_array_get_string(a, i);
            const char *y = dyn_array_get_string(b, i);
            size_t lx = strlen(x);
            size_t ly = strlen(y);
            char *buf = malloc(lx + ly + 1);
            memcpy(buf, x, lx);
            memcpy(buf + lx, y, ly);
            buf[lx + ly] = '\0';
            dyn_array_push_string(out, buf);
        }
    } else if (t == ELEM_ARRAY) {
        for (int64_t i = 0; i < len; i++) {
            DynArray *x = dyn_array_get_array(a, i);
            DynArray *y = dyn_array_get_array(b, i);
            DynArray *r = eval_dyn_array_binop(x, y, op);
            if (!r) return NULL;
            dyn_array_push_array(out, r);
        }
    } else {
        return NULL;
    }
    return out;
}

static DynArray* eval_dyn_array_scalar_right(DynArray *a, Value scalar, TokenType op) {
    if (!a) return NULL;
    int64_t len = dyn_array_length(a);
    ElementType t = dyn_array_get_elem_type(a);
    DynArray *out = dyn_array_new_with_capacity(t, len);
    if (!out) return NULL;

    if (t == ELEM_ARRAY) {
        for (int64_t i = 0; i < len; i++) {
            DynArray *inner = dyn_array_get_array(a, i);
            DynArray *r = eval_dyn_array_scalar_right(inner, scalar, op);
            if (!r) return NULL;
            dyn_array_push_array(out, r);
        }
    } else if (t == ELEM_INT && scalar.type == VAL_INT) {
        int64_t *__restrict__ pa = (int64_t*)a->data;
        int64_t *__restrict__ po = (int64_t*)out->data;
        int64_t s = scalar.as.int_val;
        switch (op) {
            case TOKEN_PLUS:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_add(pa[i], s);
                break;
            case TOKEN_MINUS:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_sub(pa[i], s);
                break;
            case TOKEN_STAR:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_mul(pa[i], s);
                break;
            case TOKEN_SLASH:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_div(pa[i], s);
                break;
            case TOKEN_PERCENT:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_rem(pa[i], s);
                break;
            default: break;
        }
        out->length = len;
    } else if (t == ELEM_FLOAT && scalar.type == VAL_FLOAT) {
        double *__restrict__ pa = (double*)a->data;
        double *__restrict__ po = (double*)out->data;
        double s = scalar.as.float_val;
        switch (op) {
            case TOKEN_PLUS:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_add(pa[i], s);
                break;
            case TOKEN_MINUS:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_sub(pa[i], s);
                break;
            case TOKEN_STAR:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_mul(pa[i], s);
                break;
            case TOKEN_SLASH:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_div(pa[i], s);
                break;
            default: break;
        }
        out->length = len;
    } else if (t == ELEM_STRING && scalar.type == VAL_STRING) {
        if (op != TOKEN_PLUS) return NULL;
        const char *s = scalar.as.string_val;
        size_t ls = strlen(s);
        for (int64_t i = 0; i < len; i++) {
            const char *x = dyn_array_get_string(a, i);
            size_t lx = strlen(x);
            char *buf = malloc(lx + ls + 1);
            memcpy(buf, x, lx);
            memcpy(buf + lx, s, ls);
            buf[lx + ls] = '\0';
            dyn_array_push_string(out, buf);
        }
    } else {
        return NULL;
    }
    return out;
}

static DynArray* eval_dyn_array_scalar_left(Value scalar, DynArray *a, TokenType op) {
    if (!a) return NULL;
    int64_t len = dyn_array_length(a);
    ElementType t = dyn_array_get_elem_type(a);
    DynArray *out = dyn_array_new_with_capacity(t, len);
    if (!out) return NULL;

    if (t == ELEM_ARRAY) {
        for (int64_t i = 0; i < len; i++) {
            DynArray *inner = dyn_array_get_array(a, i);
            DynArray *r = eval_dyn_array_scalar_left(scalar, inner, op);
            if (!r) return NULL;
            dyn_array_push_array(out, r);
        }
    } else if (t == ELEM_INT && scalar.type == VAL_INT) {
        int64_t *__restrict__ pa = (int64_t*)a->data;
        int64_t *__restrict__ po = (int64_t*)out->data;
        int64_t s = scalar.as.int_val;
        switch (op) {
            case TOKEN_PLUS:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_add(s, pa[i]);
                break;
            case TOKEN_MINUS:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_sub(s, pa[i]);
                break;
            case TOKEN_STAR:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_mul(s, pa[i]);
                break;
            case TOKEN_SLASH:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_div(s, pa[i]);
                break;
            case TOKEN_PERCENT:
                for (int64_t i = 0; i < len; i++) po[i] = eval_int_rem(s, pa[i]);
                break;
            default: break;
        }
        out->length = len;
    } else if (t == ELEM_FLOAT && scalar.type == VAL_FLOAT) {
        double *__restrict__ pa = (double*)a->data;
        double *__restrict__ po = (double*)out->data;
        double s = scalar.as.float_val;
        switch (op) {
            case TOKEN_PLUS:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_add(s, pa[i]);
                break;
            case TOKEN_MINUS:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_sub(s, pa[i]);
                break;
            case TOKEN_STAR:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_mul(s, pa[i]);
                break;
            case TOKEN_SLASH:
                for (int64_t i = 0; i < len; i++) po[i] = nano_rt_f64_div(s, pa[i]);
                break;
            default: break;
        }
        out->length = len;
    } else if (t == ELEM_STRING && scalar.type == VAL_STRING) {
        if (op != TOKEN_PLUS) return NULL;
        const char *s = scalar.as.string_val;
        size_t ls = strlen(s);
        for (int64_t i = 0; i < len; i++) {
            const char *y = dyn_array_get_string(a, i);
            size_t ly = strlen(y);
            char *buf = malloc(ls + ly + 1);
            memcpy(buf, s, ls);
            memcpy(buf + ls, y, ly);
            buf[ls + ly] = '\0';
            dyn_array_push_string(out, buf);
        }
    } else {
        return NULL;
    }
    return out;
}

/* ==========================================================================
 * Print and Evaluation Helper Functions
 * ========================================================================== */

/* Print a value (used by println and eval) */
static void print_value(Value val) {
    switch (val.type) {
        case VAL_INT:
            printf("%lld", (long long)val.as.int_val);
            break;
        case VAL_FLOAT:
            nano_rt_f64_print(stdout, val.as.float_val);
            break;
        case VAL_BOOL:
            printf("%s", val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            printf("%s", val.as.string_val);
            break;
        case VAL_ARRAY: {
            /* Print array as [elem1, elem2, ...] */
            Array *arr = val.as.array_val;
            printf("[");
            for (int i = 0; i < arr->length; i++) {
                if (i > 0) printf(", ");
                switch (arr->element_type) {
                    case VAL_INT:
                        printf("%lld", ((long long*)arr->data)[i]);
                        break;
                    case VAL_FLOAT:
                        nano_rt_f64_print(stdout, ((double*)arr->data)[i]);
                        break;
                    case VAL_BOOL:
                        printf("%s", ((bool*)arr->data)[i] ? "true" : "false");
                        break;
                    case VAL_STRING:
                        printf("\"%s\"", ((char**)arr->data)[i]);
                        break;
                    default:
                        break;
                }
            }
            printf("]");
            break;
        }
        case VAL_DYN_ARRAY: {
            /* Print dynamic array as [elem1, elem2, ...] */
            DynArray *arr = val.as.dyn_array_val;
            printf("[");
            int64_t len = dyn_array_length(arr);
            ElementType elem_type = dyn_array_get_elem_type(arr);
            for (int64_t i = 0; i < len; i++) {
                if (i > 0) printf(", ");
                switch (elem_type) {
                    case ELEM_INT:
                        printf("%lld", (long long)dyn_array_get_int(arr, i));
                        break;
                    case ELEM_FLOAT:
                        nano_rt_f64_print(stdout, dyn_array_get_float(arr, i));
                        break;
                    case ELEM_BOOL:
                        printf("%s", dyn_array_get_bool(arr, i) ? "true" : "false");
                        break;
                    case ELEM_STRING:
                        printf("\"%s\"", dyn_array_get_string(arr, i));
                        break;
                    default:
                        printf("?");
                        break;
                }
            }
            printf("]");
            break;
        }
        case VAL_STRUCT: {
            /* Print struct as StructName { field1: value1, field2: value2 } */
            StructValue *sv = val.as.struct_val;
            printf("%s { ", sv->struct_name);
            for (int i = 0; i < sv->field_count; i++) {
                if (i > 0) printf(", ");
                printf("%s: ", sv->field_names[i]);
                print_value(sv->field_values[i]);
            }
            printf(" }");
            break;
        }
        case VAL_FUNCTION: {
            /* Print function value */
            printf("<function %s>", val.as.function_val.function_name);
            break;
        }
        case VAL_GC_STRUCT: {
            /* Print GC struct */
            printf("<gc_struct>");
            break;
        }
        case VAL_UNION: {
            /* Print union value */
            UnionValue *uv = val.as.union_val;
            printf("%s.%s { ", uv->union_name, uv->variant_name);
            for (int i = 0; i < uv->field_count; i++) {
                if (i > 0) printf(", ");
                printf("%s: ", uv->field_names[i]);
                print_value(uv->field_values[i]);
            }
            printf(" }");
            break;
        }
        case VAL_TUPLE: {
            /* Print tuple value */
            TupleValue *tv = val.as.tuple_val;
            printf("(");
            for (int i = 0; i < tv->element_count; i++) {
                if (i > 0) printf(", ");
                print_value(tv->elements[i]);
            }
            printf(")");
            break;
        }
        case VAL_COROUTINE:
            printf("<coroutine:%lld>", val.as.int_val);
            break;
        case VAL_VOID:
            printf("void");
            break;
    }
}

/* ============================================================================
 * Type Casting Functions
 * ========================================================================== */

static Value builtin_cast_int(Value *args) {
    Value arg = args[0];
    
    if (arg.type == VAL_INT) {
        return arg;  /* Already an int */
    } else if (arg.type == VAL_FLOAT) {
        if (!(arg.as.float_val >= -0x1p63 && arg.as.float_val < 0x1p63)) {
            fprintf(stderr, "I cannot convert this float to int: I require a finite value in [-2^63, 2^63).\n");
            exit(EXIT_FAILURE);
        }
        return create_int((long long)arg.as.float_val);  /* Truncate */
    } else if (arg.type == VAL_BOOL) {
        return create_int(arg.as.bool_val ? 1 : 0);
    } else if (arg.type == VAL_STRING) {
        /* Parse string to int */
        char *endptr;
        long long val = strtoll(arg.as.string_val, &endptr, 10);
        if (endptr == arg.as.string_val || *endptr != '\0') {
            fprintf(stderr, "Error: cast_int cannot parse '%s' as integer\n", arg.as.string_val);
            return create_int(0);
        }
        return create_int(val);
    } else {
        fprintf(stderr, "Error: cast_int cannot convert type to int\n");
        return create_void();
    }
}

static Value builtin_cast_float(Value *args) {
    Value arg = args[0];
    
    if (arg.type == VAL_FLOAT) {
        return arg;  /* Already a float */
    } else if (arg.type == VAL_INT) {
        return create_float((double)arg.as.int_val);
    } else if (arg.type == VAL_BOOL) {
        return create_float(arg.as.bool_val ? 1.0 : 0.0);
    } else if (arg.type == VAL_STRING) {
        /* Parse string to float */
        uint32_t consumed;
        double val;
        if (!nl_binary64_parse(arg.as.string_val, &val, &consumed) ||
            consumed == 0 || arg.as.string_val[consumed] != '\0') {
            fprintf(stderr, "Error: cast_float cannot parse '%s' as float\n", arg.as.string_val);
            return create_float(0.0);
        }
        return create_float(val);
    } else {
        fprintf(stderr, "Error: cast_float cannot convert type to float\n");
        return create_void();
    }
}

static Value builtin_cast_bool(Value *args) {
    Value arg = args[0];
    
    if (arg.type == VAL_BOOL) {
        return arg;  /* Already a bool */
    } else if (arg.type == VAL_INT) {
        return create_bool(arg.as.int_val != 0);
    } else if (arg.type == VAL_FLOAT) {
        return create_bool(arg.as.float_val != 0.0);
    } else if (arg.type == VAL_STRING) {
        /* Parse string to bool */
        if (strcmp(arg.as.string_val, "true") == 0 || strcmp(arg.as.string_val, "1") == 0) {
            return create_bool(true);
        } else {
            return create_bool(false);
        }
    } else {
        fprintf(stderr, "Error: cast_bool cannot convert type to bool\n");
        return create_void();
    }
}

static Value builtin_null_opaque(Value *args) {
    (void)args;  /* No arguments needed */
    /* Opaque types are represented as integers (pointers cast to int64_t) */
    return create_int(0);
}

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} EvalSB;

static void eval_sb_ensure(EvalSB *sb, size_t extra) {
    if (!sb) return;
    size_t needed = sb->len + extra + 1;
    if (needed <= sb->cap) return;
    size_t new_cap = sb->cap ? sb->cap : 128;
    while (new_cap < needed) new_cap *= 2;
    char *new_buf = realloc(sb->buf, new_cap);
    if (!new_buf) return;
    sb->buf = new_buf;
    sb->cap = new_cap;
}

static EvalSB eval_sb_new(size_t initial_cap) {
    EvalSB sb = {0};
    sb.cap = initial_cap ? initial_cap : 128;
    sb.buf = malloc(sb.cap);
    sb.len = 0;
    if (sb.buf) sb.buf[0] = '\0';
    return sb;
}

static void eval_sb_append_cstr(EvalSB *sb, const char *s) {
    if (!sb || !s) return;
    size_t n = strlen(s);
    eval_sb_ensure(sb, n);
    if (!sb->buf) return;
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
}

static void eval_sb_append_char(EvalSB *sb, char c) {
    if (!sb) return;
    eval_sb_ensure(sb, 1);
    if (!sb->buf) return;
    sb->buf[sb->len++] = c;
    sb->buf[sb->len] = '\0';
}

static void eval_sb_append_value(EvalSB *sb, Value val);

static void eval_sb_append_dyn_array(EvalSB *sb, DynArray *arr) {
    eval_sb_append_char(sb, '[');
    int64_t len = dyn_array_length(arr);
    ElementType elem_type = dyn_array_get_elem_type(arr);
    for (int64_t i = 0; i < len; i++) {
        if (i > 0) eval_sb_append_cstr(sb, ", ");
        switch (elem_type) {
            case ELEM_INT: {
                char tmp[64];
                snprintf(tmp, sizeof(tmp), "%lld", (long long)dyn_array_get_int(arr, i));
                eval_sb_append_cstr(sb, tmp);
                break;
            }
            case ELEM_FLOAT: {
                char tmp[64];
                nano_rt_f64_format(tmp, sizeof(tmp), dyn_array_get_float(arr, i));
                eval_sb_append_cstr(sb, tmp);
                break;
            }
            case ELEM_BOOL:
                eval_sb_append_cstr(sb, dyn_array_get_bool(arr, i) ? "true" : "false");
                break;
            case ELEM_STRING:
                eval_sb_append_char(sb, '"');
                eval_sb_append_cstr(sb, dyn_array_get_string(arr, i));
                eval_sb_append_char(sb, '"');
                break;
            case ELEM_ARRAY:
                eval_sb_append_dyn_array(sb, dyn_array_get_array(arr, i));
                break;
            default:
                eval_sb_append_cstr(sb, "?");
                break;
        }
    }
    eval_sb_append_char(sb, ']');
}

static void eval_sb_append_value(EvalSB *sb, Value val) {
    switch (val.type) {
        case VAL_INT: {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%lld", (long long)val.as.int_val);
            eval_sb_append_cstr(sb, tmp);
            break;
        }
        case VAL_FLOAT: {
            char tmp[64];
            nano_rt_f64_format(tmp, sizeof(tmp), val.as.float_val);
            /* Ensure at least one decimal place for whole-number floats
             * so 0.0 → "0.0" rather than "0" (matches Python/JS behaviour) */
            if (strchr(tmp, '.') == NULL && strchr(tmp, 'e') == NULL
                    && strchr(tmp, 'n') == NULL && strchr(tmp, 'i') == NULL) {
                size_t len = strlen(tmp);
                if (len + 2 < sizeof(tmp)) {
                    tmp[len]     = '.';
                    tmp[len + 1] = '0';
                    tmp[len + 2] = '\0';
                }
            }
            eval_sb_append_cstr(sb, tmp);
            break;
        }
        case VAL_BOOL:
            eval_sb_append_cstr(sb, val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            eval_sb_append_cstr(sb, val.as.string_val ? val.as.string_val : "");
            break;
        case VAL_ARRAY: {
            Array *arr = val.as.array_val;
            eval_sb_append_char(sb, '[');
            for (int i = 0; i < arr->length; i++) {
                if (i > 0) eval_sb_append_cstr(sb, ", ");
                switch (arr->element_type) {
                    case VAL_INT: {
                        char tmp[64];
                        snprintf(tmp, sizeof(tmp), "%lld", (long long)((long long*)arr->data)[i]);
                        eval_sb_append_cstr(sb, tmp);
                        break;
                    }
                    case VAL_FLOAT: {
                        char tmp[64];
                        nano_rt_f64_format(tmp, sizeof(tmp), ((double*)arr->data)[i]);
                        eval_sb_append_cstr(sb, tmp);
                        break;
                    }
                    case VAL_BOOL:
                        eval_sb_append_cstr(sb, ((bool*)arr->data)[i] ? "true" : "false");
                        break;
                    case VAL_STRING:
                        eval_sb_append_char(sb, '"');
                        eval_sb_append_cstr(sb, ((char**)arr->data)[i]);
                        eval_sb_append_char(sb, '"');
                        break;
                    default:
                        eval_sb_append_cstr(sb, "?");
                        break;
                }
            }
            eval_sb_append_char(sb, ']');
            break;
        }
        case VAL_DYN_ARRAY: {
            DynArray *arr = val.as.dyn_array_val;
            eval_sb_append_dyn_array(sb, arr);
            break;
        }
        case VAL_STRUCT: {
            StructValue *sv = val.as.struct_val;
            eval_sb_append_cstr(sb, sv->struct_name);
            eval_sb_append_cstr(sb, " { ");
            for (int i = 0; i < sv->field_count; i++) {
                if (i > 0) eval_sb_append_cstr(sb, ", ");
                eval_sb_append_cstr(sb, sv->field_names[i]);
                eval_sb_append_cstr(sb, ": ");
                eval_sb_append_value(sb, sv->field_values[i]);
            }
            eval_sb_append_cstr(sb, " }");
            break;
        }
        case VAL_UNION: {
            UnionValue *uv = val.as.union_val;
            eval_sb_append_cstr(sb, uv->union_name);
            eval_sb_append_char(sb, '.');
            eval_sb_append_cstr(sb, uv->variant_name);
            if (uv->field_count > 0) {
                eval_sb_append_cstr(sb, " { ");
                for (int i = 0; i < uv->field_count; i++) {
                    if (i > 0) eval_sb_append_cstr(sb, ", ");
                    eval_sb_append_cstr(sb, uv->field_names[i]);
                    eval_sb_append_cstr(sb, ": ");
                    eval_sb_append_value(sb, uv->field_values[i]);
                }
                eval_sb_append_cstr(sb, " }");
            }
            break;
        }
        case VAL_TUPLE: {
            TupleValue *tv = val.as.tuple_val;
            eval_sb_append_char(sb, '(');
            for (int i = 0; i < tv->element_count; i++) {
                if (i > 0) eval_sb_append_cstr(sb, ", ");
                eval_sb_append_value(sb, tv->elements[i]);
            }
            eval_sb_append_char(sb, ')');
            break;
        }
        case VAL_FUNCTION:
            eval_sb_append_cstr(sb, "<function>");
            break;
        case VAL_GC_STRUCT:
            eval_sb_append_cstr(sb, "<gc_struct>");
            break;
        case VAL_COROUTINE: {
            char coro_buf[32];
            snprintf(coro_buf, sizeof(coro_buf), "<coroutine:%lld>", val.as.int_val);
            eval_sb_append_cstr(sb, coro_buf);
            break;
        }
        case VAL_VOID:
            eval_sb_append_cstr(sb, "void");
            break;
    }
}

static Value builtin_to_string(Value *args) {
    Value arg = args[0];
    if (arg.type == VAL_STRING) return arg;

    EvalSB sb = eval_sb_new(256);
    eval_sb_append_value(&sb, arg);
    const char *out = sb.buf ? sb.buf : "";
    Value v = create_string(out);
    free(sb.buf);
    return v;
}

static Value builtin_cast_string(Value *args) {
    return builtin_to_string(args);
}

static Value builtin_print(Value *args) {
    print_value(args[0]);
    return create_void();
}

static Value builtin_println(Value *args) {
    print_value(args[0]);
    printf("\n");
    return create_void();
}

/* ==========================================================================
 * Array Built-in Functions (With Bounds Checking!)
 * ========================================================================== */

static Value builtin_at(Value *args) {
    /* at(array, index) -> element */
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: at() requires an integer index\n");
        return create_void();
    }
    
    long long index = args[1].as.int_val;
    
    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        
        /* BOUNDS CHECKING - This is the safety guarantee! */
        if (index < 0 || index >= arr->length) {
            fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%d)\n",
                    (long long)index, arr->length);
            exit(1);  /* Fail fast - no undefined behavior! */
        }
        
        /* Return element based on type */
        switch (arr->element_type) {
            case VAL_ARRAY:
                return ((Value*)arr->data)[index];
            case VAL_INT:
                return create_int(((long long*)arr->data)[index]);
            case VAL_FLOAT:
                return create_float(((double*)arr->data)[index]);
            case VAL_BOOL:
                return create_bool(((bool*)arr->data)[index]);
            case VAL_STRING:
                return create_string(((char**)arr->data)[index]);
            case VAL_STRUCT: {
                StructValue *sv = ((StructValue**)arr->data)[index];
                return create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
            }
            default:
                fprintf(stderr, "Error: Unsupported array element type\n");
                return create_void();
        }
    }
    
    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(arr);
        
        /* BOUNDS CHECKING */
        if (index < 0 || index >= len) {
            fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n",
                    (long long)index, (long long)len);
            exit(1);
        }
        
        /* Return element based on type */
        ElementType elem_type = dyn_array_get_elem_type(arr);
        switch (elem_type) {
            case ELEM_U8:
                return create_int(dyn_array_get_u8(arr, index));
            case ELEM_INT:
                return create_int(dyn_array_get_int(arr, index));
            case ELEM_FLOAT:
                return create_float(dyn_array_get_float(arr, index));
            case ELEM_BOOL:
                return create_bool(dyn_array_get_bool(arr, index));
            case ELEM_STRING:
                return create_string(dyn_array_get_string(arr, index));
            case ELEM_ARRAY:
                return create_dyn_array(dyn_array_get_array(arr, index));
            case ELEM_STRUCT: {
                void *raw = dyn_array_get_struct(arr, index);
                if (!raw) return create_void();
                StructValue *sv = *(StructValue**)raw;
                return create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
            }
            default:
                fprintf(stderr, "Error: Unsupported array element type\n");
                return create_void();
        }
    }
    
    fprintf(stderr, "Error: at() requires an array as first argument\n");
    return create_void();
}

static Value builtin_array_length(Value *args) {
    /* array_length(array) -> int */
    if (args[0].type == VAL_ARRAY) {
        return create_int(args[0].as.array_val->length);
    }
    if (args[0].type == VAL_DYN_ARRAY) {
        return create_int(dyn_array_length(args[0].as.dyn_array_val));
    }
    
    fprintf(stderr, "Error: array_length() requires an array argument\n");
    return create_void();
}

static Value builtin_array_new(Value *args) {
    /* array_new(size, default_value) -> array */
    if (args[0].type != VAL_INT) {
        fprintf(stderr, "Error: array_new() requires an integer size\n");
        return create_void();
    }
    
    long long size = args[0].as.int_val;
    if (size < 0) {
        fprintf(stderr, "Error: array_new() size must be non-negative\n");
        return create_void();
    }
    
    ValueType elem_type = args[1].type;
    Value arr = create_array(elem_type, size, size);
    
    /* Initialize all elements with default value */
    for (long long i = 0; i < size; i++) {
        switch (elem_type) {
            case VAL_INT:
                ((long long*)arr.as.array_val->data)[i] = args[1].as.int_val;
                break;
            case VAL_FLOAT:
                ((double*)arr.as.array_val->data)[i] = args[1].as.float_val;
                break;
            case VAL_BOOL:
                ((bool*)arr.as.array_val->data)[i] = args[1].as.bool_val;
                break;
            case VAL_STRING:
                ((char**)arr.as.array_val->data)[i] = strdup(args[1].as.string_val);
                break;
            case VAL_STRUCT: {
                Value copy = create_struct(args[1].as.struct_val->struct_name,
                    args[1].as.struct_val->field_names,
                    args[1].as.struct_val->field_values,
                    args[1].as.struct_val->field_count);
                ((StructValue**)arr.as.array_val->data)[i] = copy.as.struct_val;
                break;
            }
            default:
                break;
        }
    }
    
    return arr;
}

static ElementType value_type_to_elem_type(ValueType vtype);

static Value builtin_array_set(Value *args) {
    /* array_set(array, index, value) -> void */
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: array_set() requires an integer index\n");
        return create_void();
    }
    if (args[0].type != VAL_ARRAY && args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_set() requires an array as first argument\n");
        return create_void();
    }

    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        long long index = args[1].as.int_val;
        if (index < 0 || index >= dyn_array_length(arr)) {
            fprintf(stderr, "I cannot write array index %lld outside [0..%lld).\n",
                    index, (long long)dyn_array_length(arr));
            exit(1);
        }
        if (dyn_array_get_elem_type(arr) == ELEM_U8 && args[2].type == VAL_INT) {
            dyn_array_set_u8(arr, index, (uint8_t)args[2].as.int_val);
            return create_void();
        }
        if (value_type_to_elem_type(args[2].type) != dyn_array_get_elem_type(arr)) {
            fprintf(stderr, "I cannot assign a different element type to this array.\n");
            exit(1);
        }
        switch (args[2].type) {
            case VAL_INT: dyn_array_set_int(arr, index, args[2].as.int_val); break;
            case VAL_FLOAT: dyn_array_set_float(arr, index, args[2].as.float_val); break;
            case VAL_BOOL: dyn_array_set_bool(arr, index, args[2].as.bool_val); break;
            case VAL_STRING: {
                char *copy = strdup(args[2].as.string_val);
                if (!copy) { fprintf(stderr, "I cannot allocate an array string.\n"); exit(1); }
                dyn_array_set_string(arr, index, copy);
                break;
            }
            case VAL_DYN_ARRAY: dyn_array_set_array(arr, index, args[2].as.dyn_array_val); break;
            case VAL_STRUCT: {
                StructValue *sv = args[2].as.struct_val;
                Value copy = create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
                StructValue *stored = copy.as.struct_val;
                dyn_array_set_struct(arr, index, &stored, sizeof(stored));
                break;
            }
            default:
                fprintf(stderr, "I cannot write this array element representation.\n");
                exit(1);
        }
        return create_void();
    }
    
    Array *arr = args[0].as.array_val;
    long long index = args[1].as.int_val;
    
    /* BOUNDS CHECKING */
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%d)\n",
                (long long)index, arr->length);
        exit(1);  /* Fail fast! */
    }
    
    /* Set element based on type */
    switch (arr->element_type) {
        case VAL_ARRAY:
            if (args[2].type != VAL_ARRAY && args[2].type != VAL_DYN_ARRAY) {
                fprintf(stderr, "I require an array value for a nested array element.\n");
                exit(1);
            }
            ((Value*)arr->data)[index] = args[2];
            break;
        case VAL_INT:
            if (args[2].type != VAL_INT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((long long*)arr->data)[index] = args[2].as.int_val;
            break;
        case VAL_FLOAT:
            if (args[2].type != VAL_FLOAT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((double*)arr->data)[index] = args[2].as.float_val;
            break;
        case VAL_BOOL:
            if (args[2].type != VAL_BOOL) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((bool*)arr->data)[index] = args[2].as.bool_val;
            break;
        case VAL_STRING:
            if (args[2].type != VAL_STRING) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            /* Free old string if exists */
            if (((char**)arr->data)[index]) {
                free(((char**)arr->data)[index]);
            }
            ((char**)arr->data)[index] = strdup(args[2].as.string_val);
            break;
        case VAL_STRUCT:
            if (args[2].type != VAL_STRUCT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            {
                Value copy = create_struct(args[2].as.struct_val->struct_name,
                    args[2].as.struct_val->field_names,
                    args[2].as.struct_val->field_values,
                    args[2].as.struct_val->field_count);
                ((StructValue**)arr->data)[index] = copy.as.struct_val;
            }
            break;
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            break;
    }
    
    return create_void();
}

static Value builtin_array_slice(Value *args) {
    /* array_slice(array, start, length) -> array */
    if (args[1].type != VAL_INT || args[2].type != VAL_INT) {
        fprintf(stderr, "Error: array_slice() requires integer start and length\n");
        return create_void();
    }

    int64_t start = args[1].as.int_val;
    int64_t length = args[2].as.int_val;
    if (start < 0) start = 0;
    if (length < 0) length = 0;

    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        int64_t len = arr->length;
        if (start > len) start = len;
        if (length > len - start) length = len - start;
        int64_t end = start + length;
        if (end > len) end = len;
        int64_t out_len = end - start;

        Value out = create_array(arr->element_type, out_len, out_len);
        switch (arr->element_type) {
            case VAL_ARRAY:
                for (int64_t i = 0; i < out_len; i++)
                    ((Value*)out.as.array_val->data)[i] = ((Value*)arr->data)[start + i];
                break;
            case VAL_INT:
                for (int64_t i = 0; i < out_len; i++) {
                    ((long long*)out.as.array_val->data)[i] = ((long long*)arr->data)[start + i];
                }
                break;
            case VAL_FLOAT:
                for (int64_t i = 0; i < out_len; i++) {
                    ((double*)out.as.array_val->data)[i] = ((double*)arr->data)[start + i];
                }
                break;
            case VAL_BOOL:
                for (int64_t i = 0; i < out_len; i++) {
                    ((bool*)out.as.array_val->data)[i] = ((bool*)arr->data)[start + i];
                }
                break;
            case VAL_STRING:
                for (int64_t i = 0; i < out_len; i++) {
                    ((char**)out.as.array_val->data)[i] = strdup(((char**)arr->data)[start + i]);
                }
                break;
            case VAL_STRUCT:
                for (int64_t i = 0; i < out_len; i++) {
                    StructValue *sv = ((StructValue**)arr->data)[start + i];
                    Value copy = create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
                    ((StructValue**)out.as.array_val->data)[i] = copy.as.struct_val;
                }
                break;
            default:
                break;
        }
        return out;
    }

    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(arr);
        if (start > len) start = len;
        if (length > len - start) length = len - start;
        int64_t end = start + length;
        if (end > len) end = len;

        ElementType t = dyn_array_get_elem_type(arr);
        DynArray *out = dyn_array_new(t);
        for (int64_t i = start; i < end; i++) {
            switch (t) {
                case ELEM_INT: dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;
                case ELEM_U8: dyn_array_push_u8(out, dyn_array_get_u8(arr, i)); break;
                case ELEM_FLOAT: dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;
                case ELEM_BOOL: dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;
                case ELEM_STRING: dyn_array_push_string_copy(out, dyn_array_get_string(arr, i)); break;
                case ELEM_ARRAY: dyn_array_push_array(out, dyn_array_get_array(arr, i)); break;
                case ELEM_STRUCT: {
                    void *raw = dyn_array_get_struct(arr, i);
                    if (!raw) {
                        fprintf(stderr, "Error: array_slice unsupported element type\n");
                        return create_void();
                    }
                    StructValue *sv = *(StructValue**)raw;
                    Value copy = create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
                    StructValue *sv_copy = copy.as.struct_val;
                    dyn_array_push_struct(out, &sv_copy, sizeof(StructValue*));
                    break;
                }
                default:
                    fprintf(stderr, "Error: array_slice unsupported element type\n");
                    return create_void();
            }
        }
        return create_dyn_array(out);
    }

    fprintf(stderr, "Error: array_slice() requires an array argument\n");
    return create_void();
}

/* ==========================================================================
 * Dynamic Array Operations (GC-Managed)
 * ========================================================================== */

/* Helper to create a dynamic array Value */
static Value create_dyn_array(DynArray *arr) {
    Value val;
    val.type = VAL_DYN_ARRAY;
    val.is_return = false;
    val.is_break = false;
    val.is_continue = false;
    val.as.dyn_array_val = arr;
    return val;
}

/* Helper to map ValueType to ElementType */
static ElementType value_type_to_elem_type(ValueType vtype) {
    switch (vtype) {
        case VAL_INT: return ELEM_INT;
        case VAL_FLOAT: return ELEM_FLOAT;
        case VAL_BOOL: return ELEM_BOOL;
        case VAL_STRING: return ELEM_STRING;
        case VAL_DYN_ARRAY: return ELEM_ARRAY;  /* Nested arrays */
        case VAL_STRUCT:
        case VAL_GC_STRUCT: return ELEM_STRUCT;  /* Structs */
        default: return ELEM_INT; /* Default */
    }
}

static void discard_literal_record(StructValue *record);

static size_t static_array_element_width(ValueType type) {
    switch (type) {
        case VAL_ARRAY: return sizeof(Value);
        case VAL_INT: return sizeof(long long);
        case VAL_FLOAT: return sizeof(double);
        case VAL_BOOL: return sizeof(bool);
        case VAL_STRING: return sizeof(char*);
        case VAL_STRUCT: return sizeof(StructValue*);
        default:
            fprintf(stderr, "I cannot mutate this array element representation.\n");
            exit(1);
    }
}

/* I release only storage owned by a static array slot. Nested array Values
 * keep their shared identity; create_struct clones record/string fields. */
static void static_array_remove(Array *arr, int index) {
    size_t width = static_array_element_width(arr->element_type);
    if (arr->element_type == VAL_STRING) free(((char**)arr->data)[index]);
    else if (arr->element_type == VAL_STRUCT)
        discard_literal_record(((StructValue**)arr->data)[index]);
    memmove((char*)arr->data + (size_t)index * width,
            (char*)arr->data + ((size_t)index + 1) * width,
            (size_t)(arr->length - index - 1) * width);
    arr->length--;
    memset((char*)arr->data + (size_t)arr->length * width, 0, width);
}

static Value builtin_array_push(Value *args) {
    /* array_push(array, value) -> array
     * For empty array literal [], infers type from first push
     * For dynamic arrays, appends element
     */
    
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        /* I keep the shared Array identity when its first value establishes
         * storage. Empty literals start with an integer placeholder type. */
        ValueType element_type = arr->length == 0
            ? (args[1].type == VAL_DYN_ARRAY ? VAL_ARRAY : args[1].type)
            : arr->element_type;
        size_t width = static_array_element_width(element_type);
        if (arr->length == 0 && element_type != arr->element_type) {
            free(arr->data);
            arr->data = NULL;
            arr->capacity = 0;
            arr->element_type = element_type;
        }
        bool nested_value = arr->element_type == VAL_ARRAY &&
            (args[1].type == VAL_ARRAY || args[1].type == VAL_DYN_ARRAY);
        if ((!nested_value && args[1].type != arr->element_type) || arr->length == INT_MAX ||
            (size_t)arr->length + 1 > SIZE_MAX / width) {
            fprintf(stderr, "I cannot append this value to the array.\n");
            exit(1);
        }
        if (arr->length == arr->capacity) {
            void *data = realloc(arr->data, ((size_t)arr->length + 1) * width);
            if (!data) { fprintf(stderr, "I cannot grow the array.\n"); exit(1); }
            arr->data = data;
            arr->capacity++;
        }
        memset((char*)arr->data + (size_t)arr->length * width, 0, width);
        Value set_args[] = {args[0], create_int(arr->length), args[1]};
        arr->length++;
        builtin_array_set(set_args);
        return args[0];
    }

    /* Must be a dynamic array */
    if (args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_push() requires a dynamic array (use [] to create one)\n");
        return create_void();
    }
    
    DynArray *arr = args[0].as.dyn_array_val;
    
    /* Type check */
    ElementType expected_type = dyn_array_get_elem_type(arr);
    ValueType value_type = args[1].type;
    
    if (expected_type == ELEM_U8 && value_type == VAL_INT) {
        dyn_array_push_u8(arr, (uint8_t)args[1].as.int_val);
        return args[0];
    }
    if (value_type_to_elem_type(value_type) != expected_type) {
        fprintf(stderr, "Error: Type mismatch in array_push\n");
        return create_void();
    }
    
    /* Push element */
    switch (value_type) {
        case VAL_INT:
            dyn_array_push_int(arr, args[1].as.int_val);
            break;
        case VAL_FLOAT:
            dyn_array_push_float(arr, args[1].as.float_val);
            break;
        case VAL_BOOL:
            dyn_array_push_bool(arr, args[1].as.bool_val);
            break;
        case VAL_STRING:
            dyn_array_push_string_copy(arr, args[1].as.string_val);
            break;
        case VAL_DYN_ARRAY:
            dyn_array_push_array(arr, args[1].as.dyn_array_val);
            break;
        case VAL_STRUCT: {
            Value copy = create_struct(args[1].as.struct_val->struct_name,
                args[1].as.struct_val->field_names,
                args[1].as.struct_val->field_values,
                args[1].as.struct_val->field_count);
            StructValue *sv_copy = copy.as.struct_val;
            dyn_array_push_struct(arr, &sv_copy, sizeof(StructValue*));
            break;
        }
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            return create_void();
    }
    
    /* Return the same array (it's mutated in-place) */
    return args[0];
}

static Value builtin_array_pop(Value *args) {
    /* array_pop(array) -> value */
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        if (!arr->length) {
            fprintf(stderr, "I cannot pop an empty array.\n");
            return create_void();
        }
        Value read_args[] = {args[0], create_int(arr->length - 1)};
        Value result = builtin_at(read_args);
        static_array_remove(arr, arr->length - 1);
        return result;
    }
    if (args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_pop() requires a dynamic array\n");
        return create_void();
    }
    
    DynArray *arr = args[0].as.dyn_array_val;
    
    if (dyn_array_length(arr) == 0) {
        fprintf(stderr, "Error: array_pop() on empty array\n");
        return create_void();
    }
    
    /* Pop element based on type */
    bool success = false;
    ElementType elem_type = dyn_array_get_elem_type(arr);
    switch (elem_type) {
        case ELEM_INT: {
            int64_t val = dyn_array_pop_int(arr, &success);
            return success ? create_int(val) : create_void();
        }
        case ELEM_FLOAT: {
            double val = dyn_array_pop_float(arr, &success);
            return success ? create_float(val) : create_void();
        }
        case ELEM_BOOL: {
            bool val = dyn_array_pop_bool(arr, &success);
            return success ? create_bool(val) : create_void();
        }
        case ELEM_STRING: {
            const char *val = dyn_array_pop_string(arr, &success);
            return success ? create_string(val) : create_void();
        }
        case ELEM_ARRAY: {
            DynArray *val = dyn_array_pop_array(arr, &success);
            return success ? create_dyn_array(val) : create_void();
        }
        case ELEM_STRUCT: {
            StructValue *sv = NULL;
            dyn_array_pop_struct(arr, &sv, sizeof(StructValue*), &success);
            if (!success || !sv) return create_void();
            return create_struct(sv->struct_name, sv->field_names, sv->field_values, sv->field_count);
        }
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            return create_void();
    }
}

static Value builtin_array_remove_at(Value *args) {
    /* array_remove_at(array, index) -> array */
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        if (args[1].type != VAL_INT) {
            fprintf(stderr, "I require an integer array removal index.\n");
            return create_void();
        }
        long long index = args[1].as.int_val;
        if (index < 0 || index >= arr->length) {
            fprintf(stderr, "I cannot remove array index %lld outside [0..%d).\n",
                    index, arr->length);
            exit(1);
        }
        static_array_remove(arr, (int)index);
        return args[0];
    }
    if (args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_remove_at() requires a dynamic array\n");
        return create_void();
    }
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: array_remove_at() requires an integer index\n");
        return create_void();
    }
    
    DynArray *arr = args[0].as.dyn_array_val;
    int64_t index = args[1].as.int_val;
    
    if (index < 0 || index >= dyn_array_length(arr)) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds\n", (long long)index);
        exit(1);
    }
    
    dyn_array_remove_at(arr, index);
    
    /* Return the modified array */
    return args[0];
}

/* I normalize scalar literals without confusing them with empty arrays. */
static DynArray *builtin_scalar_array(Value value) {
    if (value.type == VAL_DYN_ARRAY) return value.as.dyn_array_val;
    if (value.type != VAL_ARRAY || !value.as.array_val) return NULL;
    Array *source = value.as.array_val;
    if (source->length < 0 || (source->length && !source->data)) return NULL;
    ElementType type;
    switch (source->element_type) {
        case VAL_INT: type = ELEM_INT; break;
        case VAL_FLOAT: type = ELEM_FLOAT; break;
        case VAL_BOOL: type = ELEM_BOOL; break;
        case VAL_STRING: type = ELEM_STRING; break;
        default: return NULL;
    }
    DynArray *result = dyn_array_new_with_capacity(type, source->length);
    if (!result) return NULL;
    for (int i = 0; i < source->length; i++) {
        switch (source->element_type) {
            case VAL_INT: dyn_array_push_int(result, ((long long *)source->data)[i]); break;
            case VAL_FLOAT: dyn_array_push_float(result, ((double *)source->data)[i]); break;
            case VAL_BOOL: dyn_array_push_bool(result, ((bool *)source->data)[i]); break;
            case VAL_STRING: dyn_array_push_string(result, ((char **)source->data)[i]); break;
            default: break;
        }
    }
    return result;
}

static Value builtin_array_sort(Value *args) {
    /* I share scalar ordering with native code and the VM. */
    DynArray *arr = builtin_scalar_array(args[0]);
    if (!arr) {
        fprintf(stderr, "I require a supported array for array_sort.\n");
        return create_void();
    }
    DynArray *out = dyn_array_sorted(arr);
    if (!out) {
        fprintf(stderr, "I could not sort this array.\n");
        return create_void();
    }
    return create_dyn_array(out);
}

static Value builtin_array_reverse(Value *args) {
    /* array_reverse(array) -> array — returns reversed copy */
    DynArray *arr = builtin_scalar_array(args[0]);
    if (!arr) {
        fprintf(stderr, "I require a supported array for array_reverse.\n");
        return create_void();
    }
    int64_t len = dyn_array_length(arr);
    ElementType t = dyn_array_get_elem_type(arr);
    DynArray *out = dyn_array_new(t);
    if (!out) return create_void();
    for (int64_t i = len - 1; i >= 0; i--) {
        switch (t) {
            case ELEM_INT:    dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;
            case ELEM_FLOAT:  dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;
            case ELEM_BOOL:   dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;
            case ELEM_STRING: dyn_array_push_string(out, dyn_array_get_string(arr, i)); break;
            default: dyn_array_push_int(out, 0); break;
        }
    }
    return create_dyn_array(out);
}

static Value builtin_array_contains(Value *args) {
    /* array_contains(array, elem) -> bool */
    DynArray *arr = builtin_scalar_array(args[0]);
    if (!arr) {
        fprintf(stderr, "I require a supported array for array_contains.\n");
        return create_bool(false);
    }
    int64_t len = dyn_array_length(arr);
    ElementType t = dyn_array_get_elem_type(arr);
    for (int64_t i = 0; i < len; i++) {
        switch (t) {
            case ELEM_INT:
                if (args[1].type == VAL_INT && dyn_array_get_int(arr, i) == args[1].as.int_val)
                    return create_bool(true);
                break;
            case ELEM_FLOAT:
                if (args[1].type == VAL_FLOAT && dyn_array_get_float(arr, i) == args[1].as.float_val)
                    return create_bool(true);
                break;
            case ELEM_BOOL:
                if (args[1].type == VAL_BOOL && dyn_array_get_bool(arr, i) == args[1].as.bool_val)
                    return create_bool(true);
                break;
            case ELEM_STRING:
                if (args[1].type == VAL_STRING && strcmp(dyn_array_get_string(arr, i), args[1].as.string_val) == 0)
                    return create_bool(true);
                break;
            default: break;
        }
    }
    return create_bool(false);
}

static Value builtin_array_index_of(Value *args) {
    /* array_index_of(array, elem) -> int (-1 if not found) */
    DynArray *arr = builtin_scalar_array(args[0]);
    if (!arr) {
        fprintf(stderr, "I require a supported array for array_index_of.\n");
        return create_int(-1);
    }
    int64_t len = dyn_array_length(arr);
    ElementType t = dyn_array_get_elem_type(arr);
    for (int64_t i = 0; i < len; i++) {
        switch (t) {
            case ELEM_INT:
                if (args[1].type == VAL_INT && dyn_array_get_int(arr, i) == args[1].as.int_val)
                    return create_int(i);
                break;
            case ELEM_FLOAT:
                if (args[1].type == VAL_FLOAT && dyn_array_get_float(arr, i) == args[1].as.float_val)
                    return create_int(i);
                break;
            case ELEM_BOOL:
                if (args[1].type == VAL_BOOL && dyn_array_get_bool(arr, i) == args[1].as.bool_val)
                    return create_int(i);
                break;
            case ELEM_STRING:
                if (args[1].type == VAL_STRING && strcmp(dyn_array_get_string(arr, i), args[1].as.string_val) == 0)
                    return create_int(i);
                break;
            default: break;
        }
    }
    return create_int(-1);
}

/* ==========================================================================
 * Higher-Order Array Functions (map, filter, reduce)
 * ========================================================================== */

/* Pure arithmetic lambda detection for map/reduce fast paths.
 * Returns true if expr contains only: arithmetic binary/unary ops, numeric
 * literals, and identifier references (assumed to be function parameters).
 * No function calls, conditionals, or assignments.
 */
static bool is_pure_arithmetic_expr(ASTNode *expr) {
    if (!expr) return false;
    switch (expr->type) {
        case AST_NUMBER:
        case AST_FLOAT:
            return true;
        case AST_IDENTIFIER:
            return true;  /* Assume it's a parameter — caller verifies */
        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            /* Allow arithmetic operators only */
            if (op != TOKEN_PLUS && op != TOKEN_MINUS &&
                op != TOKEN_STAR && op != TOKEN_SLASH && op != TOKEN_PERCENT)
                return false;
            for (int i = 0; i < expr->as.prefix_op.arg_count; i++) {
                if (!is_pure_arithmetic_expr(expr->as.prefix_op.args[i]))
                    return false;
            }
            return true;
        }
        default:
            return false;
    }
}

/* Returns true when the function body is a single return of a pure arithmetic expr. */
static bool is_pure_arithmetic_lambda(ASTNode *fn_body) {
    if (!fn_body || fn_body->type != AST_BLOCK) return false;
    if (fn_body->as.block.count != 1) return false;
    ASTNode *stmt = fn_body->as.block.statements[0];
    if (!stmt || stmt->type != AST_RETURN) return false;
    return is_pure_arithmetic_expr(stmt->as.return_stmt.value);
}

/* I match NanoVM integer negation without evaluating signed overflow. */
static int64_t eval_negate_int(int64_t value) {
    return value == INT64_MIN ? INT64_MIN : -value;
}

/* Evaluate a pure arithmetic expression for int64_t.
 * param_val is the value to substitute for any identifier matching param_name.
 */
static int64_t eval_pure_expr_int(ASTNode *expr, int64_t param_val, const char *param_name) {
    if (!expr) return 0;
    switch (expr->type) {
        case AST_NUMBER: return expr->as.number;
        case AST_FLOAT:  return (int64_t)expr->as.float_val;
        case AST_IDENTIFIER:
            return (param_name && strcmp(expr->as.identifier, param_name) == 0)
                   ? param_val : 0;
        case AST_PREFIX_OP: {
            if (expr->as.prefix_op.arg_count == 1) {
                int64_t a = eval_pure_expr_int(expr->as.prefix_op.args[0], param_val, param_name);
                return (expr->as.prefix_op.op == TOKEN_MINUS) ? eval_negate_int(a) : a;
            }
            if (expr->as.prefix_op.arg_count == 2) {
                int64_t a = eval_pure_expr_int(expr->as.prefix_op.args[0], param_val, param_name);
                int64_t b = eval_pure_expr_int(expr->as.prefix_op.args[1], param_val, param_name);
                switch (expr->as.prefix_op.op) {
                    case TOKEN_PLUS:    return eval_int_add(a, b);
                    case TOKEN_MINUS:   return eval_int_sub(a, b);
                    case TOKEN_STAR:    return eval_int_mul(a, b);
                    case TOKEN_SLASH:   return eval_int_div(a, b);
                    case TOKEN_PERCENT: return eval_int_rem(a, b);
                    default: return 0;
                }
            }
            return 0;
        }
        default: return 0;
    }
}

/* Evaluate a pure arithmetic expression for int64_t with two parameter substitutions. */
static int64_t eval_pure_expr_int2(ASTNode *expr,
                                   int64_t p0_val, const char *p0_name,
                                   int64_t p1_val, const char *p1_name) {
    if (!expr) return 0;
    switch (expr->type) {
        case AST_NUMBER: return expr->as.number;
        case AST_FLOAT:  return (int64_t)expr->as.float_val;
        case AST_IDENTIFIER:
            if (p0_name && strcmp(expr->as.identifier, p0_name) == 0) return p0_val;
            if (p1_name && strcmp(expr->as.identifier, p1_name) == 0) return p1_val;
            return 0;
        case AST_PREFIX_OP: {
            if (expr->as.prefix_op.arg_count == 1) {
                int64_t a = eval_pure_expr_int2(expr->as.prefix_op.args[0], p0_val, p0_name, p1_val, p1_name);
                return (expr->as.prefix_op.op == TOKEN_MINUS) ? eval_negate_int(a) : a;
            }
            if (expr->as.prefix_op.arg_count == 2) {
                int64_t a = eval_pure_expr_int2(expr->as.prefix_op.args[0], p0_val, p0_name, p1_val, p1_name);
                int64_t b = eval_pure_expr_int2(expr->as.prefix_op.args[1], p0_val, p0_name, p1_val, p1_name);
                switch (expr->as.prefix_op.op) {
                    case TOKEN_PLUS:    return eval_int_add(a, b);
                    case TOKEN_MINUS:   return eval_int_sub(a, b);
                    case TOKEN_STAR:    return eval_int_mul(a, b);
                    case TOKEN_SLASH:   return eval_int_div(a, b);
                    case TOKEN_PERCENT: return eval_int_rem(a, b);
                    default: return 0;
                }
            }
            return 0;
        }
        default: return 0;
    }
}

/* Evaluate a pure arithmetic expression for double with two parameter substitutions. */
static double eval_pure_expr_float2(ASTNode *expr,
                                    double p0_val, const char *p0_name,
                                    double p1_val, const char *p1_name) {
    if (!expr) return 0.0;
    switch (expr->type) {
        case AST_NUMBER: return (double)expr->as.number;
        case AST_FLOAT:  return expr->as.float_val;
        case AST_IDENTIFIER:
            if (p0_name && strcmp(expr->as.identifier, p0_name) == 0) return p0_val;
            if (p1_name && strcmp(expr->as.identifier, p1_name) == 0) return p1_val;
            return 0.0;
        case AST_PREFIX_OP: {
            if (expr->as.prefix_op.arg_count == 1) {
                double a = eval_pure_expr_float2(expr->as.prefix_op.args[0], p0_val, p0_name, p1_val, p1_name);
                return (expr->as.prefix_op.op == TOKEN_MINUS) ? -a : a;
            }
            if (expr->as.prefix_op.arg_count == 2) {
                double a = eval_pure_expr_float2(expr->as.prefix_op.args[0], p0_val, p0_name, p1_val, p1_name);
                double b = eval_pure_expr_float2(expr->as.prefix_op.args[1], p0_val, p0_name, p1_val, p1_name);
                switch (expr->as.prefix_op.op) {
                    case TOKEN_PLUS:  return nano_rt_f64_add(a, b);
                    case TOKEN_MINUS: return nano_rt_f64_sub(a, b);
                    case TOKEN_STAR:  return nano_rt_f64_mul(a, b);
                    case TOKEN_SLASH: return nano_rt_f64_div(a, b);
                    default: return 0.0;
                }
            }
            return 0.0;
        }
        default: return 0.0;
    }
}

/* Evaluate a pure arithmetic expression for double. */
static double eval_pure_expr_float(ASTNode *expr, double param_val, const char *param_name) {
    if (!expr) return 0.0;
    switch (expr->type) {
        case AST_NUMBER: return (double)expr->as.number;
        case AST_FLOAT:  return expr->as.float_val;
        case AST_IDENTIFIER:
            return (param_name && strcmp(expr->as.identifier, param_name) == 0)
                   ? param_val : 0.0;
        case AST_PREFIX_OP: {
            if (expr->as.prefix_op.arg_count == 1) {
                double a = eval_pure_expr_float(expr->as.prefix_op.args[0], param_val, param_name);
                return (expr->as.prefix_op.op == TOKEN_MINUS) ? -a : a;
            }
            if (expr->as.prefix_op.arg_count == 2) {
                double a = eval_pure_expr_float(expr->as.prefix_op.args[0], param_val, param_name);
                double b = eval_pure_expr_float(expr->as.prefix_op.args[1], param_val, param_name);
                switch (expr->as.prefix_op.op) {
                    case TOKEN_PLUS:  return nano_rt_f64_add(a, b);
                    case TOKEN_MINUS: return nano_rt_f64_sub(a, b);
                    case TOKEN_STAR:  return nano_rt_f64_mul(a, b);
                    case TOKEN_SLASH: return nano_rt_f64_div(a, b);
                    default: return 0.0;
                }
            }
            return 0.0;
        }
        default: return 0.0;
    }
}

static void discard_partial_owned_array(Array *array, int initialized);

static Value builtin_map(Value *args, Environment *env) {
    /* map(array, transform_fn) -> array
     * Applies transform_fn to each element and returns a new array
     */
    if (args[1].type != VAL_FUNCTION) {
        fprintf(stderr, "Error: map() requires a function as second argument\n");
        return create_void();
    }
    
    const char *transform_fn_name = args[1].as.function_val.function_name;
    Function *transform = env_get_function(env, transform_fn_name);
    ValueType result_type = VAL_VOID;
    if (transform) {
        switch (transform->return_type) {
            case TYPE_INT: result_type = VAL_INT; break;
            case TYPE_FLOAT: result_type = VAL_FLOAT; break;
            case TYPE_BOOL: result_type = VAL_BOOL; break;
            case TYPE_STRING: result_type = VAL_STRING; break;
            default: break;
        }
    }
    
    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *input_arr = args[0].as.array_val;
        int64_t len = input_arr->length;
        
        /* I retain declared scalar output types even when no callback runs. */
        Value result = create_array(result_type == VAL_VOID ? input_arr->element_type : result_type, len, len);
        Array *output_arr = result.as.array_val;
        
        /* Apply transform to each element */
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.type = input_arr->element_type;
            elem.is_return = false;
            elem.is_break = false;
            elem.is_continue = false;
            
            /* Get element from input array */
            switch (input_arr->element_type) {
                case VAL_INT:
                    elem.as.int_val = ((long long*)input_arr->data)[i];
                    break;
                case VAL_FLOAT:
                    elem.as.float_val = ((double*)input_arr->data)[i];
                    break;
                case VAL_BOOL:
                    elem.as.bool_val = ((bool*)input_arr->data)[i];
                    break;
                case VAL_STRING:
                    elem.as.string_val = ((char**)input_arr->data)[i];
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in map\n");
                    return create_void();
            }
            
            /* Call transform function with this element */
            Value call_args[1];
            call_args[0] = elem;
            Value transformed = call_function(transform_fn_name, call_args, 1, env);
            if (transformed.is_return) {
                discard_partial_owned_array(output_arr, (int)i);
                return transformed;
            }
            
            /* Store transformed value in output array */
            switch (output_arr->element_type) {
                case VAL_INT:
                    if (transformed.type != VAL_INT) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    ((long long*)output_arr->data)[i] = transformed.as.int_val;
                    break;
                case VAL_FLOAT:
                    if (transformed.type != VAL_FLOAT) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    ((double*)output_arr->data)[i] = transformed.as.float_val;
                    break;
                case VAL_BOOL:
                    if (transformed.type != VAL_BOOL) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    ((bool*)output_arr->data)[i] = transformed.as.bool_val;
                    break;
                case VAL_STRING:
                    if (transformed.type != VAL_STRING) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    ((char**)output_arr->data)[i] = strdup(transformed.as.string_val);
                    break;
                default:
                    break;
            }
        }
        
        return result;
    }
    
    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *input_arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(input_arr);
        ElementType elem_type = dyn_array_get_elem_type(input_arr);

        /* Fast path: pure arithmetic lambda — bypass call_function overhead.
         * Pre-allocate full output, extract restrict pointers, inline the expression. */
        if ((elem_type == ELEM_INT && result_type == VAL_INT) ||
            (elem_type == ELEM_FLOAT && result_type == VAL_FLOAT)) {
            Function *fn = env_get_function(env, transform_fn_name);
            if (fn && fn->param_count == 1 && fn->body &&
                is_pure_arithmetic_lambda(fn->body)) {
                ASTNode *ret_expr = fn->body->as.block.statements[0]->as.return_stmt.value;
                const char *param_name = fn->params[0].name;
                DynArray *output_arr = dyn_array_new_with_capacity(elem_type, len);
                if (output_arr) {
                    if (elem_type == ELEM_INT) {
                        int64_t *__restrict__ pin  = (int64_t*)input_arr->data;
                        int64_t *__restrict__ pout = (int64_t*)output_arr->data;
                        for (int64_t i = 0; i < len; i++) {
                            pout[i] = eval_pure_expr_int(ret_expr, pin[i], param_name);
                        }
                    } else {
                        double *__restrict__ pin  = (double*)input_arr->data;
                        double *__restrict__ pout = (double*)output_arr->data;
                        for (int64_t i = 0; i < len; i++) {
                            pout[i] = eval_pure_expr_float(ret_expr, pin[i], param_name);
                        }
                    }
                    output_arr->length = len;
                    return create_dyn_array(output_arr);
                }
            }
        }

        ElementType output_type = result_type == VAL_VOID ? elem_type : value_type_to_elem_type(result_type);
        DynArray *output_arr = dyn_array_new(output_type);

        /* Apply transform to each element */
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.is_return = false;
            elem.is_break = false;
            elem.is_continue = false;
            
            /* Get element from input array */
            switch (elem_type) {
                case ELEM_INT:
                    elem.type = VAL_INT;
                    elem.as.int_val = dyn_array_get_int(input_arr, i);
                    break;
                case ELEM_FLOAT:
                    elem.type = VAL_FLOAT;
                    elem.as.float_val = dyn_array_get_float(input_arr, i);
                    break;
                case ELEM_BOOL:
                    elem.type = VAL_BOOL;
                    elem.as.bool_val = dyn_array_get_bool(input_arr, i);
                    break;
                case ELEM_STRING:
                    elem.type = VAL_STRING;
                    elem.as.string_val = (char*)dyn_array_get_string(input_arr, i);
                    break;
                case ELEM_ARRAY:
                    elem.type = VAL_DYN_ARRAY;
                    elem.as.dyn_array_val = dyn_array_get_array(input_arr, i);
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in map\n");
                    return create_void();
            }
            
            /* Call transform function */
            Value call_args[1];
            call_args[0] = elem;
            Value transformed = call_function(transform_fn_name, call_args, 1, env);
            if (transformed.is_return) {
                gc_release(output_arr);
                return transformed;
            }
            
            /* Push transformed value to output array */
            switch (output_type) {
                case ELEM_INT:
                    if (transformed.type != VAL_INT) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    dyn_array_push_int(output_arr, transformed.as.int_val);
                    break;
                case ELEM_FLOAT:
                    if (transformed.type != VAL_FLOAT) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    dyn_array_push_float(output_arr, transformed.as.float_val);
                    break;
                case ELEM_BOOL:
                    if (transformed.type != VAL_BOOL) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    dyn_array_push_bool(output_arr, transformed.as.bool_val);
                    break;
                case ELEM_STRING:
                    if (transformed.type != VAL_STRING) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    dyn_array_push_string_copy(output_arr, transformed.as.string_val);
                    break;
                case ELEM_ARRAY:
                    if (transformed.type != VAL_DYN_ARRAY) {
                        fprintf(stderr, "I require the transform's declared result type in map.\n");
                        return create_void();
                    }
                    dyn_array_push_array(output_arr, transformed.as.dyn_array_val);
                    break;
                default:
                    break;
            }
        }
        
        return create_dyn_array(output_arr);
    }
    
    fprintf(stderr, "Error: map() requires an array as first argument\n");
    return create_void();
}

static Value builtin_filter(Value *args, Environment *env) {
    /* filter(array, predicate_fn) -> array
     * Returns a new array containing only elements where predicate_fn(elem) is true.
     */
    if (args[1].type != VAL_FUNCTION) {
        fprintf(stderr, "Error: filter() requires a function as second argument\n");
        return create_void();
    }

    const char *pred_fn_name = args[1].as.function_val.function_name;

    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *input_arr = args[0].as.array_val;
        int64_t len = input_arr->length;

        bool *keep = (bool*)calloc((size_t)len, sizeof(bool));
        if (!keep) {
            fprintf(stderr, "Error: Out of memory in filter()\n");
            return create_void();
        }

        int64_t out_len = 0;
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.type = input_arr->element_type;
            elem.is_return = false;
            elem.is_break = false;
            elem.is_continue = false;

            switch (input_arr->element_type) {
                case VAL_INT:
                    elem.as.int_val = ((long long*)input_arr->data)[i];
                    break;
                case VAL_FLOAT:
                    elem.as.float_val = ((double*)input_arr->data)[i];
                    break;
                case VAL_BOOL:
                    elem.as.bool_val = ((bool*)input_arr->data)[i];
                    break;
                case VAL_STRING:
                    elem.as.string_val = ((char**)input_arr->data)[i];
                    break;
                default:
                    free(keep);
                    fprintf(stderr, "Error: Unsupported array element type in filter\n");
                    return create_void();
            }

            Value call_args[1];
            call_args[0] = elem;
            Value pred = call_function(pred_fn_name, call_args, 1, env);
            if (pred.is_return) {
                free(keep);
                return pred;
            }
            if (pred.type != VAL_BOOL) {
                free(keep);
                fprintf(stderr, "Error: filter predicate must return bool\n");
                return create_void();
            }
            keep[i] = pred.as.bool_val;
            if (keep[i]) out_len++;
        }

        Value result = create_array(input_arr->element_type, out_len, out_len);
        Array *output_arr = result.as.array_val;

        int64_t out_i = 0;
        for (int64_t i = 0; i < len; i++) {
            if (!keep[i]) continue;

            switch (input_arr->element_type) {
                case VAL_INT:
                    ((long long*)output_arr->data)[out_i] = ((long long*)input_arr->data)[i];
                    break;
                case VAL_FLOAT:
                    ((double*)output_arr->data)[out_i] = ((double*)input_arr->data)[i];
                    break;
                case VAL_BOOL:
                    ((bool*)output_arr->data)[out_i] = ((bool*)input_arr->data)[i];
                    break;
                case VAL_STRING:
                    ((char**)output_arr->data)[out_i] = strdup(((char**)input_arr->data)[i]);
                    break;
                default:
                    break;
            }
            out_i++;
        }

        free(keep);
        return result;
    }

    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *input_arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(input_arr);
        ElementType elem_type = dyn_array_get_elem_type(input_arr);

        DynArray *output_arr = dyn_array_new(elem_type);
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.is_return = false;
            elem.is_break = false;
            elem.is_continue = false;

            switch (elem_type) {
                case ELEM_INT:
                    elem.type = VAL_INT;
                    elem.as.int_val = dyn_array_get_int(input_arr, i);
                    break;
                case ELEM_FLOAT:
                    elem.type = VAL_FLOAT;
                    elem.as.float_val = dyn_array_get_float(input_arr, i);
                    break;
                case ELEM_BOOL:
                    elem.type = VAL_BOOL;
                    elem.as.bool_val = dyn_array_get_bool(input_arr, i);
                    break;
                case ELEM_STRING:
                    elem.type = VAL_STRING;
                    elem.as.string_val = (char*)dyn_array_get_string(input_arr, i);
                    break;
                case ELEM_ARRAY:
                    elem.type = VAL_DYN_ARRAY;
                    elem.as.dyn_array_val = dyn_array_get_array(input_arr, i);
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in filter\n");
                    return create_void();
            }

            Value call_args[1];
            call_args[0] = elem;
            Value pred = call_function(pred_fn_name, call_args, 1, env);
            if (pred.is_return) {
                gc_release(output_arr);
                return pred;
            }
            if (pred.type != VAL_BOOL) {
                fprintf(stderr, "Error: filter predicate must return bool\n");
                return create_void();
            }

            if (!pred.as.bool_val) continue;

            switch (elem_type) {
                case ELEM_INT:
                    dyn_array_push_int(output_arr, elem.as.int_val);
                    break;
                case ELEM_FLOAT:
                    dyn_array_push_float(output_arr, elem.as.float_val);
                    break;
                case ELEM_BOOL:
                    dyn_array_push_bool(output_arr, elem.as.bool_val);
                    break;
                case ELEM_STRING:
                    dyn_array_push_string_copy(output_arr, elem.as.string_val);
                    break;
                case ELEM_ARRAY:
                    dyn_array_push_array(output_arr, elem.as.dyn_array_val);
                    break;
                default:
                    break;
            }
        }

        return create_dyn_array(output_arr);
    }

    fprintf(stderr, "Error: filter() requires an array as first argument\n");
    return create_void();
}

static Value builtin_reduce(Value *args, Environment *env) {
    /* reduce(array, initial_value, combine_fn) -> value
     * Combines all elements using combine_fn, starting with initial_value
     */
    if (args[2].type != VAL_FUNCTION) {
        fprintf(stderr, "Error: reduce() requires a function as third argument\n");
        return create_void();
    }
    
    const char *combine_fn_name = args[2].as.function_val.function_name;
    Value accumulator = args[1];  /* Initial value */
    
    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        int64_t len = arr->length;
        
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.type = arr->element_type;
            elem.is_return = false;
            elem.is_break = false;
            elem.is_continue = false;
            
            /* Get element */
            switch (arr->element_type) {
                case VAL_INT:
                    elem.as.int_val = ((long long*)arr->data)[i];
                    break;
                case VAL_FLOAT:
                    elem.as.float_val = ((double*)arr->data)[i];
                    break;
                case VAL_BOOL:
                    elem.as.bool_val = ((bool*)arr->data)[i];
                    break;
                case VAL_STRING:
                    elem.as.string_val = ((char**)arr->data)[i];
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in reduce\n");
                    return create_void();
            }
            
            /* Call combine function with accumulator and element */
            Value call_args[2];
            call_args[0] = accumulator;
            call_args[1] = elem;
            accumulator = call_function(combine_fn_name, call_args, 2, env);
            if (accumulator.is_return) return accumulator;
        }
        
        return accumulator;
    }
    
    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(arr);
        ElementType elem_type = dyn_array_get_elem_type(arr);

        /* Fast path: pure arithmetic 2-param lambda — bypass call_function overhead.
         * Evaluates combine(acc, elem) inline using typed direct loop. */
        if (elem_type == ELEM_INT || elem_type == ELEM_FLOAT) {
            Function *fn = env_get_function(env, combine_fn_name);
            if (fn && fn->param_count == 2 && fn->body &&
                is_pure_arithmetic_lambda(fn->body)) {
                ASTNode *ret_expr = fn->body->as.block.statements[0]->as.return_stmt.value;
                const char *acc_name  = fn->params[0].name;
                const char *elem_name = fn->params[1].name;
                if (elem_type == ELEM_INT && accumulator.type == VAL_INT) {
                    int64_t *__restrict__ pa = (int64_t*)arr->data;
                    int64_t acc = accumulator.as.int_val;
                    for (int64_t i = 0; i < len; i++) {
                        acc = eval_pure_expr_int2(ret_expr,
                                                  acc,    acc_name,
                                                  pa[i],  elem_name);
                    }
                    return create_int(acc);
                } else if (elem_type == ELEM_FLOAT && accumulator.type == VAL_FLOAT) {
                    double *__restrict__ pa = (double*)arr->data;
                    double acc = accumulator.as.float_val;
                    for (int64_t i = 0; i < len; i++) {
                        acc = eval_pure_expr_float2(ret_expr,
                                                    acc,    acc_name,
                                                    pa[i],  elem_name);
                    }
                    return create_float(acc);
                }
            }
        }

        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.is_return = false;
            elem.is_break = false;
            elem.is_continue = false;

            /* Get element */
            switch (elem_type) {
                case ELEM_INT:
                    elem.type = VAL_INT;
                    elem.as.int_val = dyn_array_get_int(arr, i);
                    break;
                case ELEM_FLOAT:
                    elem.type = VAL_FLOAT;
                    elem.as.float_val = dyn_array_get_float(arr, i);
                    break;
                case ELEM_BOOL:
                    elem.type = VAL_BOOL;
                    elem.as.bool_val = dyn_array_get_bool(arr, i);
                    break;
                case ELEM_STRING:
                    elem.type = VAL_STRING;
                    elem.as.string_val = (char*)dyn_array_get_string(arr, i);
                    break;
                case ELEM_ARRAY:
                    elem.type = VAL_DYN_ARRAY;
                    elem.as.dyn_array_val = dyn_array_get_array(arr, i);
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in reduce\n");
                    return create_void();
            }

            /* Call combine function */
            Value call_args[2];
            call_args[0] = accumulator;
            call_args[1] = elem;
            accumulator = call_function(combine_fn_name, call_args, 2, env);
            if (accumulator.is_return) return accumulator;
        }

        return accumulator;
    }

    fprintf(stderr, "Error: reduce() requires an array as first argument\n");
    return create_void();
}

/* ==========================================================================
 * End of Math and Utility Built-in Functions
 * ========================================================================== */

/* Helper to convert value to boolean */
static bool is_truthy(Value val) {
    switch (val.type) {
        case VAL_BOOL:
            return val.as.bool_val;
        case VAL_INT:
            return val.as.int_val != 0;
        case VAL_FLOAT:
            return val.as.float_val != 0.0;
        case VAL_VOID:
            return false;
        default:
            return true; /* Strings are truthy if non-null */
    }
}

/* Evaluate prefix operation */
static Value eval_prefix_op(ASTNode *node, Environment *env) {
    TokenType op = node->as.prefix_op.op;
    int arg_count = node->as.prefix_op.arg_count;


    /* Arithmetic operators */
    if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
        op == TOKEN_SLASH || op == TOKEN_PERCENT) {
        
        /* Handle unary minus: (- x) */
        if (op == TOKEN_MINUS && arg_count == 1) {
            Value arg = eval_expression(node->as.prefix_op.args[0], env);
            if (arg.is_return) return arg;
            if (arg.type == VAL_INT) {
                return create_int(eval_negate_int(arg.as.int_val));
            } else if (arg.type == VAL_FLOAT) {
                return create_float(-arg.as.float_val);
            } else if (arg.type == VAL_DYN_ARRAY) {
                DynArray *a = arg.as.dyn_array_val;
                assert(a);
                ElementType t = dyn_array_get_elem_type(a);
                int64_t len = dyn_array_length(a);
                if (t == ELEM_INT) {
                    DynArray *out = dyn_array_new(ELEM_INT);
                    for (int64_t i = 0; i < len; i++) dyn_array_push_int(out, eval_negate_int(dyn_array_get_int(a, i)));
                    return create_dyn_array(out);
                } else if (t == ELEM_FLOAT) {
                    DynArray *out = dyn_array_new(ELEM_FLOAT);
                    for (int64_t i = 0; i < len; i++) dyn_array_push_float(out, -dyn_array_get_float(a, i));
                    return create_dyn_array(out);
                }
                fprintf(stderr, "Error: Unary minus requires array<int> or array<float>\n");
                return create_void();
            } else if (arg.type == VAL_ARRAY) {
                Array *a = arg.as.array_val;
                if (!a) return create_void();
                if (a->element_type == VAL_INT) {
                    Value out = create_array(VAL_INT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) ((long long*)out.as.array_val->data)[i] = eval_negate_int(((long long*)a->data)[i]);
                    return out;
                } else if (a->element_type == VAL_FLOAT) {
                    Value out = create_array(VAL_FLOAT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) ((double*)out.as.array_val->data)[i] = -((double*)a->data)[i];
                    return out;
                }
                fprintf(stderr, "Error: Unary minus requires array<int> or array<float>\n");
                return create_void();
            } else {
                fprintf(stderr, "Error: Unary minus requires numeric argument\n");
                return create_void();
            }
        }
        
        /* Binary arithmetic operations */
        if (arg_count != 2) {
            fprintf(stderr, "Error: Binary arithmetic operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        if (left.is_return) return left;
        Value right = eval_expression(node->as.prefix_op.args[1], env);
        if (right.is_return) return right;

        /* Array arithmetic (elementwise) */
        if (left.type == VAL_DYN_ARRAY || right.type == VAL_DYN_ARRAY || left.type == VAL_ARRAY || right.type == VAL_ARRAY) {
            /* DynArray path */
            if (left.type == VAL_DYN_ARRAY || right.type == VAL_DYN_ARRAY) {
                if (left.type == VAL_DYN_ARRAY && right.type == VAL_DYN_ARRAY) {
                    DynArray *a = left.as.dyn_array_val;
                    DynArray *b = right.as.dyn_array_val;
                    assert(a && b);
                    int64_t len = dyn_array_length(a);
                    if (len != dyn_array_length(b)) {
                        fprintf(stderr, "Error: Array length mismatch in operator\n");
                        return create_void();
                    }
                    ElementType t = dyn_array_get_elem_type(a);
                    if (t != dyn_array_get_elem_type(b)) {
                        fprintf(stderr, "Error: Array element type mismatch in operator\n");
                        return create_void();
                    }
                    DynArray *out = eval_dyn_array_binop(a, b, op);
                    if (!out) {
                        fprintf(stderr, "Error: Array mismatch in operator\n");
                        return create_void();
                    }
                    return create_dyn_array(out);
                }

                /* Broadcast scalar over array */
                if (left.type == VAL_DYN_ARRAY) {
                    DynArray *a = left.as.dyn_array_val;
                    DynArray *out = eval_dyn_array_scalar_right(a, right, op);
                    if (!out) {
                        fprintf(stderr, "Error: Type mismatch in array-scalar operator\n");
                        return create_void();
                    }
                    return create_dyn_array(out);
                } else if (right.type == VAL_DYN_ARRAY) {
                    DynArray *a = right.as.dyn_array_val;
                    DynArray *out = eval_dyn_array_scalar_left(left, a, op);
                    if (!out) {
                        fprintf(stderr, "Error: Type mismatch in scalar-array operator\n");
                        return create_void();
                    }
                    return create_dyn_array(out);
                }
            }

            /* Static Array path */
            if (left.type == VAL_ARRAY && right.type == VAL_ARRAY) {
                Array *a = left.as.array_val;
                Array *b = right.as.array_val;
                if (!a || !b || a->length != b->length || a->element_type != b->element_type) {
                    fprintf(stderr, "Error: Array mismatch in operator\n");
                    return create_void();
                }
                Value out = create_array(a->element_type, a->length, a->length);
                for (int i = 0; i < a->length; i++) {
                    switch (a->element_type) {
                        case VAL_INT: {
                            long long x = ((long long*)a->data)[i];
                            long long y = ((long long*)b->data)[i];
                            long long r = 0;
                            switch (op) {
                                case TOKEN_PLUS: r = eval_int_add(x, y); break;
                                case TOKEN_MINUS: r = eval_int_sub(x, y); break;
                                case TOKEN_STAR: r = eval_int_mul(x, y); break;
                                case TOKEN_SLASH: r = eval_int_div(x, y); break;
                                case TOKEN_PERCENT: r = eval_int_rem(x, y); break;
                                default: break;
                            }
                            ((long long*)out.as.array_val->data)[i] = r;
                            break;
                        }
                        case VAL_FLOAT: {
                            double x = ((double*)a->data)[i];
                            double y = ((double*)b->data)[i];
                            double r = 0.0;
                            switch (op) {
                                case TOKEN_PLUS: r = nano_rt_f64_add(x, y); break;
                                case TOKEN_MINUS: r = nano_rt_f64_sub(x, y); break;
                                case TOKEN_STAR: r = nano_rt_f64_mul(x, y); break;
                                case TOKEN_SLASH: r = nano_rt_f64_div(x, y); break;
                                default: break;
                            }
                            ((double*)out.as.array_val->data)[i] = r;
                            break;
                        }
                        case VAL_STRING: {
                            if (op != TOKEN_PLUS) {
                                fprintf(stderr, "Error: string arrays only support +\n");
                                return create_void();
                            }
                            const char *x = ((char**)a->data)[i];
                            const char *y = ((char**)b->data)[i];
                            size_t lx = strlen(x);
                            size_t ly = strlen(y);
                            char *buf = malloc(lx + ly + 1);
                            memcpy(buf, x, lx);
                            memcpy(buf + lx, y, ly);
                            buf[lx + ly] = '\0';
                            ((char**)out.as.array_val->data)[i] = buf;
                            break;
                        }
                        default:
                            fprintf(stderr, "Error: unsupported array element type in operator\n");
                            return create_void();
                    }
                }
                return out;
            }

            /* Static array + scalar broadcast */
            if (left.type == VAL_ARRAY && (right.type == VAL_INT || right.type == VAL_FLOAT || right.type == VAL_STRING)) {
                Array *a = left.as.array_val;
                if (!a) return create_void();

                if (a->element_type == VAL_INT && right.type == VAL_INT) {
                    Value out = create_array(VAL_INT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        long long x = ((long long*)a->data)[i];
                        long long s = right.as.int_val;
                        long long r = 0;
                        switch (op) {
                            case TOKEN_PLUS: r = eval_int_add(x, s); break;
                            case TOKEN_MINUS: r = eval_int_sub(x, s); break;
                            case TOKEN_STAR: r = eval_int_mul(x, s); break;
                            case TOKEN_SLASH: r = eval_int_div(x, s); break;
                            case TOKEN_PERCENT: r = eval_int_rem(x, s); break;
                            default: break;
                        }
                        ((long long*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_FLOAT && right.type == VAL_FLOAT) {
                    Value out = create_array(VAL_FLOAT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        double x = ((double*)a->data)[i];
                        double s = right.as.float_val;
                        double r = 0.0;
                        switch (op) {
                            case TOKEN_PLUS: r = nano_rt_f64_add(x, s); break;
                            case TOKEN_MINUS: r = nano_rt_f64_sub(x, s); break;
                            case TOKEN_STAR: r = nano_rt_f64_mul(x, s); break;
                            case TOKEN_SLASH: r = nano_rt_f64_div(x, s); break;
                            default: break;
                        }
                        ((double*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_STRING && right.type == VAL_STRING) {
                    if (op != TOKEN_PLUS) {
                        fprintf(stderr, "Error: string arrays only support +\n");
                        return create_void();
                    }
                    Value out = create_array(VAL_STRING, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        const char *x = ((char**)a->data)[i];
                        const char *s = right.as.string_val;
                        size_t lx = strlen(x);
                        size_t ls = strlen(s);
                        char *buf = malloc(lx + ls + 1);
                        memcpy(buf, x, lx);
                        memcpy(buf + lx, s, ls);
                        buf[lx + ls] = '\0';
                        ((char**)out.as.array_val->data)[i] = buf;
                    }
                    return out;
                }
            }

            /* Scalar + static array broadcast */
            if (right.type == VAL_ARRAY && (left.type == VAL_INT || left.type == VAL_FLOAT || left.type == VAL_STRING)) {
                Array *a = right.as.array_val;
                if (!a) return create_void();

                if (a->element_type == VAL_INT && left.type == VAL_INT) {
                    Value out = create_array(VAL_INT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        long long s = left.as.int_val;
                        long long y = ((long long*)a->data)[i];
                        long long r = 0;
                        switch (op) {
                            case TOKEN_PLUS: r = eval_int_add(s, y); break;
                            case TOKEN_MINUS: r = eval_int_sub(s, y); break;
                            case TOKEN_STAR: r = eval_int_mul(s, y); break;
                            case TOKEN_SLASH: r = eval_int_div(s, y); break;
                            case TOKEN_PERCENT: r = eval_int_rem(s, y); break;
                            default: break;
                        }
                        ((long long*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_FLOAT && left.type == VAL_FLOAT) {
                    Value out = create_array(VAL_FLOAT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        double s = left.as.float_val;
                        double y = ((double*)a->data)[i];
                        double r = 0.0;
                        switch (op) {
                            case TOKEN_PLUS: r = nano_rt_f64_add(s, y); break;
                            case TOKEN_MINUS: r = nano_rt_f64_sub(s, y); break;
                            case TOKEN_STAR: r = nano_rt_f64_mul(s, y); break;
                            case TOKEN_SLASH: r = nano_rt_f64_div(s, y); break;
                            default: break;
                        }
                        ((double*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_STRING && left.type == VAL_STRING) {
                    if (op != TOKEN_PLUS) {
                        fprintf(stderr, "Error: string arrays only support +\n");
                        return create_void();
                    }
                    Value out = create_array(VAL_STRING, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        const char *s = left.as.string_val;
                        const char *y = ((char**)a->data)[i];
                        size_t ls = strlen(s);
                        size_t ly = strlen(y);
                        char *buf = malloc(ls + ly + 1);
                        memcpy(buf, s, ls);
                        memcpy(buf + ls, y, ly);
                        buf[ls + ly] = '\0';
                        ((char**)out.as.array_val->data)[i] = buf;
                    }
                    return out;
                }
            }
        }

        if (left.type == VAL_INT && right.type == VAL_INT) {
            long long result;
            switch (op) {
                case TOKEN_PLUS: result = eval_int_add(left.as.int_val, right.as.int_val); break;
                case TOKEN_MINUS: result = eval_int_sub(left.as.int_val, right.as.int_val); break;
                case TOKEN_STAR: result = eval_int_mul(left.as.int_val, right.as.int_val); break;
                case TOKEN_SLASH: result = eval_int_div(left.as.int_val, right.as.int_val); break;
                case TOKEN_PERCENT: result = eval_int_rem(left.as.int_val, right.as.int_val); break;
                default: result = 0;
            }
            return create_int(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_FLOAT) {
            double result;
            switch (op) {
                case TOKEN_PLUS: result = nano_rt_f64_add(left.as.float_val, right.as.float_val); break;
                case TOKEN_MINUS: result = nano_rt_f64_sub(left.as.float_val, right.as.float_val); break;
                case TOKEN_STAR: result = nano_rt_f64_mul(left.as.float_val, right.as.float_val); break;
                case TOKEN_SLASH:
                    /* Total float division = 0.0 by zero, matching the VM. */
                    result = nano_rt_f64_div(left.as.float_val, right.as.float_val);
                    break;
                default: result = 0.0;
            }
            return create_float(result);
        } else if (left.type == VAL_STRING && right.type == VAL_STRING) {
            /* String concatenation with + operator */
            if (op == TOKEN_PLUS) {
                size_t len1 = safe_strlen(left.as.string_val);
                size_t len2 = safe_strlen(right.as.string_val);
                char *result = malloc(len1 + len2 + 1);
                if (!result) {
                    safe_fprintf(stderr, "Error: Memory allocation failed in string concatenation\n");
                    return create_void();
                }
                safe_strncpy(result, left.as.string_val, len1 + len2 + 1);
                safe_strncat(result, right.as.string_val, len1 + len2 + 1);
                result[len1 + len2] = '\0';
                Value v = create_string(result);
                free(result);
                return v;
            } else {
                fprintf(stderr, "Error: Strings only support + operator\n");
                return create_void();
            }
        }
    }

    /* Comparison operators */
    if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Comparison operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        if (left.is_return) return left;
        Value right = eval_expression(node->as.prefix_op.args[1], env);
        if (right.is_return) return right;

        if (left.type == VAL_INT && right.type == VAL_INT) {
            bool result;
            switch (op) {
                case TOKEN_LT: result = left.as.int_val < right.as.int_val; break;
                case TOKEN_LE: result = left.as.int_val <= right.as.int_val; break;
                case TOKEN_GT: result = left.as.int_val > right.as.int_val; break;
                case TOKEN_GE: result = left.as.int_val >= right.as.int_val; break;
                default: result = false;
            }
            return create_bool(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_FLOAT) {
            bool result;
            switch (op) {
                case TOKEN_LT: result = left.as.float_val < right.as.float_val; break;
                case TOKEN_LE: result = left.as.float_val <= right.as.float_val; break;
                case TOKEN_GT: result = left.as.float_val > right.as.float_val; break;
                case TOKEN_GE: result = left.as.float_val >= right.as.float_val; break;
                default: result = false;
            }
            return create_bool(result);
        } else if (left.type == VAL_INT && right.type == VAL_FLOAT) {
            /* Mixed int/float comparison: convert int to float */
            bool result;
            double left_f = (double)left.as.int_val;
            switch (op) {
                case TOKEN_LT: result = left_f < right.as.float_val; break;
                case TOKEN_LE: result = left_f <= right.as.float_val; break;
                case TOKEN_GT: result = left_f > right.as.float_val; break;
                case TOKEN_GE: result = left_f >= right.as.float_val; break;
                default: result = false;
            }
            return create_bool(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_INT) {
            /* Mixed float/int comparison: convert int to float */
            bool result;
            double right_f = (double)right.as.int_val;
            switch (op) {
                case TOKEN_LT: result = left.as.float_val < right_f; break;
                case TOKEN_LE: result = left.as.float_val <= right_f; break;
                case TOKEN_GT: result = left.as.float_val > right_f; break;
                case TOKEN_GE: result = left.as.float_val >= right_f; break;
                default: result = false;
            }
            return create_bool(result);
        }
    }

    /* Equality operators */
    if (op == TOKEN_EQ || op == TOKEN_NE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Equality operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        if (left.is_return) return left;
        Value right = eval_expression(node->as.prefix_op.args[1], env);
        if (right.is_return) return right;

        bool equal = false;
        if (left.type == right.type) {
            switch (left.type) {
                case VAL_INT: equal = left.as.int_val == right.as.int_val; break;
                case VAL_FLOAT: equal = left.as.float_val == right.as.float_val; break;
                case VAL_BOOL: equal = left.as.bool_val == right.as.bool_val; break;
                case VAL_STRING: equal = strcmp(left.as.string_val, right.as.string_val) == 0; break;
                case VAL_STRUCT: {
                    /* Structs are equal if they're the same type and all fields are equal */
                    StructValue *left_sv = left.as.struct_val;
                    StructValue *right_sv = right.as.struct_val;
                    if (strcmp(left_sv->struct_name, right_sv->struct_name) != 0 ||
                        left_sv->field_count != right_sv->field_count) {
                        equal = false;
                    } else {
                        equal = true;
                        for (int i = 0; i < left_sv->field_count && equal; i++) {
                            Value left_field = left_sv->field_values[i];
                            Value right_field = right_sv->field_values[i];
                            /* Recursively compare field values (simplified - only int/float/bool/string) */
                            if (left_field.type != right_field.type) {
                                equal = false;
                            } else if (left_field.type == VAL_INT) {
                                equal = left_field.as.int_val == right_field.as.int_val;
                            } else if (left_field.type == VAL_FLOAT) {
                                equal = left_field.as.float_val == right_field.as.float_val;
                            } else if (left_field.type == VAL_BOOL) {
                                equal = left_field.as.bool_val == right_field.as.bool_val;
                            } else if (left_field.type == VAL_STRING) {
                                equal = strcmp(left_field.as.string_val, right_field.as.string_val) == 0;
                            }
                        }
                    }
                    break;
                }
                case VAL_VOID: equal = true; break;  /* void == void */
                case VAL_ARRAY: {
                    /* Arrays are equal if they have same length and all elements equal */
                    Array *left_arr = left.as.array_val;
                    Array *right_arr = right.as.array_val;
                    if (left_arr->length != right_arr->length) {
                        equal = false;
                    } else {
                        equal = true;
                        for (int i = 0; i < left_arr->length && equal; i++) {
                            switch (left_arr->element_type) {
                                case VAL_INT:
                                    equal = ((long long*)left_arr->data)[i] == ((long long*)right_arr->data)[i];
                                    break;
                                case VAL_FLOAT:
                                    equal = ((double*)left_arr->data)[i] == ((double*)right_arr->data)[i];
                                    break;
                                case VAL_BOOL:
                                    equal = ((bool*)left_arr->data)[i] == ((bool*)right_arr->data)[i];
                                    break;
                                case VAL_STRING:
                                    equal = strcmp(((char**)left_arr->data)[i], ((char**)right_arr->data)[i]) == 0;
                                    break;
                                default:
                                    equal = false;
                                    break;
                            }
                        }
                    }
                    break;
                }
                case VAL_DYN_ARRAY:
                case VAL_GC_STRUCT:
                case VAL_UNION:
                case VAL_TUPLE:
                case VAL_FUNCTION:
                case VAL_COROUTINE:
                    /* These types don't support equality comparison yet */
                    equal = false;
                    break;
            }
        }

        return create_bool(op == TOKEN_EQ ? equal : !equal);
    }

    /* Logical operators */
    if (op == TOKEN_AND || op == TOKEN_OR) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Logical operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        if (left.is_return || left.is_break || left.is_continue) return left;

        if (op == TOKEN_AND) {
            if (!is_truthy(left)) return create_bool(false);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            if (right.is_return || right.is_break || right.is_continue) return right;
            return create_bool(is_truthy(right));
        } else { /* OR */
            if (is_truthy(left)) return create_bool(true);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            if (right.is_return || right.is_break || right.is_continue) return right;
            return create_bool(is_truthy(right));
        }
    }

    if (op == TOKEN_NOT) {
        if (arg_count != 1) {
            fprintf(stderr, "Error: 'not' requires 1 argument\n");
            return create_void();
        }
        Value arg = eval_expression(node->as.prefix_op.args[0], env);
        if (arg.is_return) return arg;
        return create_bool(!is_truthy(arg));
    }

    return create_void();
}

/* I select only my generated declaration or an undeclared lowercase operation.
 * User functions retain precedence. A spelling alone never identifies a handle. */
static bool eval_record_list_call(const char *name, Value *args, int argc,
                                 Environment *env, Value *out) {
    if (!name || (strncmp(name, "list_", 5) && strncmp(name, "List_", 5))) return false;
    Function *function = env_get_function(env, name);
    NominalIdentity element = env_generated_list_element(env, function);
    if (function && !element.ordinal) return false;
    if (!element.ordinal && strncmp(name, "list_", 5)) return false;
    static const char *operations[] = {"with_capacity", "is_empty", "new", "push", "pop",
        "get", "set", "insert", "remove", "length", "capacity", "clear", "free"};
    const char *operation = NULL;
    size_t length = strlen(name), type_length = 0;
    for (size_t i = 0; i < sizeof(operations) / sizeof(operations[0]); ++i) {
        size_t n = strlen(operations[i]);
        if (length > 6 + n && name[length - n - 1] == '_' &&
            !strcmp(name + length - n, operations[i])) {
            operation = operations[i];
            type_length = length - n - 6;
            break;
        }
    }
    if (!operation) return false;
    if (!element.ordinal) {
        char *type_name = strndup(name + 5, type_length);
        if (!type_name) { fprintf(stderr, "I cannot allocate list declaration metadata.\n"); exit(1); }
        element = env_nominal_identity(env, type_name, env->current_module, TYPE_STRUCT);
        free(type_name);
    }
    if (!element.ordinal || !env_record_list_apply(env, element, operation, args, argc, out)) {
        fprintf(stderr, "I cannot perform '%s': I require a live matching record list, valid arguments and bounds, and available snapshot storage.\n", name);
        exit(1);
    }
    return true;
}

/* Synchronous array callbacks borrow their descriptor while running. Only a
 * declaration identifier creates a fresh descriptor here; a variable read
 * borrows the Symbol's descriptor. Other expression ownership stays unchanged. */
static bool owns_declared_callback(ASTNode *expression, Environment *env, Value value) {
    if (!expression || expression->type != AST_IDENTIFIER || value.type != VAL_FUNCTION)
        return false;
    /* Runtime lookup may skip a later checker-only placeholder. I compare
     * actual live owners rather than repeating a different name lookup. */
    for (int i = 0; i < env->symbol_count; ++i) {
        Value owner = env->symbols[i].value;
        if (owner.type == VAL_FUNCTION &&
            owner.as.function_val.function_name == value.as.function_val.function_name)
            return false;
    }
    return true;
}

static void discard_declared_callback(Value value) {
    if (value.type != VAL_FUNCTION) return;
    free(value.as.function_val.function_name);
    free_function_signature(value.as.function_val.signature);
}

static Value eval_call_impl(ASTNode *node, Environment *env, const char *bound_name);

/* I retain the invoking source node while native builtins call back into me. */
static Value eval_call(ASTNode *node, Environment *env) {
    ASTNode *saved_site = g_eval_call_site;
    g_eval_call_site = node;
    char *bound_name = NULL;
    if (!node->as.call.func_expr && node->as.call.name) {
        Symbol *binding = env_get_var(env, node->as.call.name);
        if (binding && binding->value.type == VAL_FUNCTION) {
            bound_name = strdup(binding->value.as.function_val.function_name);
            if (!bound_name) {
                fprintf(stderr, "I could not retain the function name.\n");
                g_eval_call_site = saved_site;
                return create_void();
            }
        }
    }
    Value result = eval_call_impl(node, env, bound_name);
    free(bound_name);
    g_eval_call_site = saved_site;
    return result;
}

/* Evaluate function call */
static Value eval_call_impl(ASTNode *node, Environment *env, const char *bound_name) {
    /* Check if this is a function call returning a function: ((func_call) arg1 arg2) */
    if (node->as.call.func_expr) {
        /* Evaluate the inner function call to get the function */
        Value func_val = eval_expression(node->as.call.func_expr, env);
        if (func_val.is_return) return func_val;
        if (func_val.type != VAL_FUNCTION) {
            fprintf(stderr, "Error: Expression does not return a function\n");
            return create_void();
        }
        
        /* Get the function name from the function value */
        const char *borrowed_name = func_val.as.function_val.function_name;
        char *func_name = borrowed_name ? strdup(borrowed_name) : NULL;
        if (!func_name) {
            fprintf(stderr, "Error: Cannot get function name from function value\n");
            return create_void();
        }
        
        /* Call the function */
        Function *func = env_get_function(env, func_name);

        /* Infer anonymous struct literal names from parameter types before evaluating */
        for (int i = 0; i < node->as.call.arg_count && func && i < func->param_count; i++) {
            ASTNode *arg = node->as.call.args[i];
            if (arg->type == AST_STRUCT_LITERAL && arg->as.struct_literal.struct_name == NULL) {
                if (func->params[i].type == TYPE_STRUCT && func->params[i].struct_type_name) {
                    arg->as.struct_literal.struct_name = strdup(func->params[i].struct_type_name);
                }
            }
        }

        /* Evaluate arguments */
        Value *args = malloc(sizeof(Value) * node->as.call.arg_count);
        for (int i = 0; i < node->as.call.arg_count; i++) {
            args[i] = eval_staged_argument(node->as.call.args[i], env, func_name, i);
            if (args[i].is_return) {
                Value result = args[i];
                free(args);
                free(func_name);
                return result;
            }
        }
        if (!func) {
            fprintf(stderr, "Error: Function '%s' not found\n", func_name);
            free(args);
            free(func_name);
            return create_void();
        }
        
        Value result = call_function_at(func_name, args, node->as.call.arg_count, env,
                                        node->line, node->column);
        free(args);
        free(func_name);
        return result;
    }
    
    const char *name = bound_name ? bound_name : node->as.call.name;

    /* Special built-in: range (used in for loops only) */
    if (strcmp(name, "range") == 0) {
        /* This should not be called directly */
        return create_void();
    }

    /* ── Coroutine builtins ─────────────────────────────────────────── */

    /* spawn(fn_name, arg1, arg2, ...) — spawn an async function as a coroutine.
     * Returns a VAL_COROUTINE value (int_val = coroutine id).
     */
    if (strcmp(name, "coro_spawn") == 0) {
        if (node->as.call.arg_count < 1) {
            fprintf(stderr, "Error: spawn() requires at least a function name argument\n");
            return create_void();
        }
        ASTNode *fn_arg = node->as.call.args[0];
        const char *async_fn_name = NULL;
        if (fn_arg->type == AST_IDENTIFIER) {
            async_fn_name = fn_arg->as.identifier;
        } else if (fn_arg->type == AST_STRING) {
            async_fn_name = fn_arg->as.string_val;
        } else {
            Value fn_val = eval_expression(fn_arg, env);
            if (fn_val.type == VAL_FUNCTION) {
                async_fn_name = fn_val.as.function_val.function_name;
            }
        }
        if (!async_fn_name) {
            fprintf(stderr, "Error: spawn() first argument must be a function\n");
            return create_void();
        }

        int extra_args = node->as.call.arg_count - 1;
        Function *deferred = env_get_function(env, async_fn_name);
        for (int i = 0; deferred && deferred->params && i < deferred->param_count; ++i) {
            Type type = deferred->params[i].type;
            if (type == TYPE_BORROW_SHARED || type == TYPE_BORROW_MUT) {
                fprintf(stderr, "I cannot enqueue a deferred borrowed argument.\n"); exit(1);
            }
        }
        CoroCallArgs *ca = coro_bundle_new(env, async_fn_name, extra_args);
        if (!ca) { fprintf(stderr, "I cannot prepare task argument storage.\n"); exit(1); }
        for (int i = 0; i < extra_args; ++i) {
            Value value = eval_expression(node->as.call.args[i + 1], env);
            if (value.is_return || value.is_break || value.is_continue) {
                coro_bundle_drop(ca);
                return value;
            }
            if (!coro_bundle_argument(ca, i, value, false)) {
                coro_bundle_drop(ca);
                fprintf(stderr, "I cannot copy a pending task argument.\n"); exit(1);
            }
        }
        int coro_id = coro_bundle_enqueue(ca);
        if (coro_id < 0) {
            coro_bundle_drop(ca);
            fprintf(stderr, "Error: spawn() failed — scheduler full or lease unavailable\n");
            return create_void();
        }

        Value coro_val;
        memset(&coro_val, 0, sizeof(coro_val));
        coro_val.type = VAL_COROUTINE;
        coro_val.as.int_val = (long long)coro_id;
        return coro_val;
    }

    /* coro_yield() — cooperatively suspend the current coroutine */
    if (strcmp(name, "coro_yield") == 0) {
        nano_coro_yield();
        return create_void();
    }

    /* coro_done(handle) — returns true if the coroutine is done */
    if (strcmp(name, "coro_done") == 0) {
        if (node->as.call.arg_count < 1) return create_void();
        Value h = eval_expression(node->as.call.args[0], env);
        bool done = (h.type == VAL_COROUTINE)
            ? nano_coro_is_done((int)h.as.int_val)
            : true;
        return create_bool(done);
    }

    /* coro_result(handle) — returns result of a completed coroutine */
    if (strcmp(name, "coro_result") == 0) {
        if (node->as.call.arg_count < 1) return create_void();
        Value h = eval_expression(node->as.call.args[0], env);
        return (h.type == VAL_COROUTINE)
            ? eval_task_result(env, (int)h.as.int_val, false)
            : create_void();
    }

    /* scheduler_run() — drain all pending coroutines */
    if (strcmp(name, "scheduler_run") == 0) {
        nano_scheduler_run_until_done();
        return create_void();
    }

    /* scheduler_step() — run one scheduler step */
    if (strcmp(name, "scheduler_step") == 0) {
        bool did_work = nano_scheduler_step();
        return create_bool(did_work);
    }

    /* Infer anonymous struct literal names from parameter types before evaluating */
    Function *named_func = env_get_function(env, name);
    bool is_builtin_array_push = env_function_is_named_builtin(named_func, "array_push");
    for (int i = 0; i < node->as.call.arg_count && named_func && i < named_func->param_count; i++) {
        ASTNode *arg = node->as.call.args[i];
        if (arg->type == AST_STRUCT_LITERAL && arg->as.struct_literal.struct_name == NULL) {
            if (named_func->params[i].type == TYPE_STRUCT && named_func->params[i].struct_type_name) {
                arg->as.struct_literal.struct_name = strdup(named_func->params[i].struct_type_name);
            }
        }
    }

    /* These existing synchronous consumers never publish their callback descriptor. */
    int callback_kind = 0;
    if (strcmp(name, "map") == 0 || strcmp(name, "array_map") == 0) callback_kind = 1;
    else if (strcmp(name, "filter") == 0 || strcmp(name, "array_filter") == 0) callback_kind = 2;
    else if (strcmp(name, "reduce") == 0 || strcmp(name, "array_fold") == 0) callback_kind = 3;
    int callback_index = callback_kind == 3 ? 2 : 1;
    Value owned_callback = create_void();

    /* Evaluate arguments in the original order. */
    Value args[16];  /* Max args for function calls */
    for (int i = 0; i < node->as.call.arg_count; i++) {
        args[i] = eval_staged_argument(node->as.call.args[i], env, name, i);
        if (callback_kind && i == callback_index &&
            owns_declared_callback(node->as.call.args[i], env, args[i]))
            owned_callback = args[i];
        if (args[i].is_return) {
            discard_declared_callback(owned_callback);
            return args[i];
        }
    }

    /* File operations */
    if (strcmp(name, "file_read") == 0) return builtin_file_read(args);
    if (strcmp(name, "file_read_bytes") == 0) return builtin_file_read_bytes(args);
    if (strcmp(name, "file_write") == 0) return builtin_file_write(args);
    if (strcmp(name, "file_append") == 0) return builtin_file_append(args);
    if (strcmp(name, "file_remove") == 0) return builtin_file_remove(args);
    if (strcmp(name, "file_rename") == 0) return builtin_file_rename(args);
    if (strcmp(name, "file_exists") == 0) return builtin_file_exists(args);
    if (strcmp(name, "file_size") == 0) return builtin_file_size(args);

    /* Temp helpers */
    if (strcmp(name, "tmp_dir") == 0) return builtin_tmp_dir(args);
    if (strcmp(name, "mktemp") == 0) return builtin_mktemp(args);
    if (strcmp(name, "mktemp_dir") == 0) return builtin_mktemp_dir(args);

    /* Directory operations */
    if (strcmp(name, "dir_create") == 0) return builtin_dir_create(args);
    if (strcmp(name, "dir_remove") == 0) return builtin_dir_remove(args);
    if (strcmp(name, "dir_list") == 0) return builtin_dir_list(args);
    if (strcmp(name, "dir_exists") == 0) return builtin_dir_exists(args);
    if (strcmp(name, "getcwd") == 0) return builtin_getcwd(args);
    if (strcmp(name, "chdir") == 0) return builtin_chdir(args);
    if (strcmp(name, "fs_walkdir") == 0) return builtin_fs_walkdir(args);

    /* Path operations */
    if (strcmp(name, "path_isfile") == 0) return builtin_path_isfile(args);
    if (strcmp(name, "path_isdir") == 0) return builtin_path_isdir(args);
    if (strcmp(name, "path_join") == 0) return builtin_path_join(args);
    if (strcmp(name, "path_basename") == 0) return builtin_path_basename(args);
    if (strcmp(name, "path_dirname") == 0) return builtin_path_dirname(args);
    if (strcmp(name, "path_normalize") == 0) return builtin_path_normalize(args);

    /* Process operations */
    if (strcmp(name, "system") == 0) return builtin_system(args);
    if (strcmp(name, "exit") == 0) return builtin_exit(args);
    if (strcmp(name, "getenv") == 0) return builtin_getenv(args);
    if (strcmp(name, "setenv") == 0) return builtin_setenv(args);
    if (strcmp(name, "unsetenv") == 0) return builtin_unsetenv(args);
    if (strcmp(name, "process_run") == 0) return builtin_process_run(args);

    /* Result helpers */
    if (strcmp(name, "result_is_ok") == 0) return builtin_result_is_ok(args);
    if (strcmp(name, "result_is_err") == 0) return builtin_result_is_err(args);
    if (strcmp(name, "result_unwrap") == 0) return builtin_result_unwrap(args);
    if (strcmp(name, "result_unwrap_err") == 0) return builtin_result_unwrap_err(args);
    if (strcmp(name, "result_unwrap_or") == 0) return builtin_result_unwrap_or(args);
    if (strcmp(name, "result_map") == 0) return builtin_result_map(args, env);
    if (strcmp(name, "result_and_then") == 0) return builtin_result_and_then(args, env);

    /* GPU built-in stubs — return 0 in interpreter; real impl is in PTX backend */
    if (strcmp(name, "thread_id_x") == 0) return create_int(0);
    if (strcmp(name, "thread_id_y") == 0) return create_int(0);
    if (strcmp(name, "thread_id_z") == 0) return create_int(0);
    if (strcmp(name, "block_id_x")  == 0) return create_int(0);
    if (strcmp(name, "block_id_y")  == 0) return create_int(0);
    if (strcmp(name, "block_id_z")  == 0) return create_int(0);
    if (strcmp(name, "block_dim_x") == 0) return create_int(256);
    if (strcmp(name, "block_dim_y") == 0) return create_int(256);
    if (strcmp(name, "block_dim_z") == 0) return create_int(1);
    if (strcmp(name, "grid_dim_x")  == 0) return create_int(1);
    if (strcmp(name, "grid_dim_y")  == 0) return create_int(1);
    if (strcmp(name, "grid_dim_z")  == 0) return create_int(1);
    if (strcmp(name, "global_id_x") == 0) return create_int(0);
    if (strcmp(name, "global_id_y") == 0) return create_int(0);
    if (strcmp(name, "gpu_barrier")    == 0) return create_void();

    /* Math and utility functions */
    if (strcmp(name, "abs") == 0) return builtin_abs(args);
    if (strcmp(name, "min") == 0) return builtin_min(args);
    if (strcmp(name, "max") == 0) return builtin_max(args);
    if (strcmp(name, "print") == 0) return builtin_print(args);
    if (strcmp(name, "println") == 0) return builtin_println(args);
    
    /* Advanced math functions */
    if (strcmp(name, "sqrt") == 0) return builtin_sqrt(args);
    if (strcmp(name, "pow") == 0) return builtin_pow(args);
    if (strcmp(name, "floor") == 0) return builtin_floor(args);
    if (strcmp(name, "ceil") == 0) return builtin_ceil(args);
    if (strcmp(name, "round") == 0) return builtin_round(args);
    
    /* Trigonometric functions */
    if (strcmp(name, "sin") == 0) return builtin_sin(args);
    if (strcmp(name, "cos") == 0) return builtin_cos(args);
    if (strcmp(name, "tan") == 0) return builtin_tan(args);
    if (strcmp(name, "atan2") == 0) return builtin_atan2(args);
    
    /* Type casting functions */
    if (strcmp(name, "cast_int") == 0) return builtin_cast_int(args);
    if (strcmp(name, "cast_float") == 0) return builtin_cast_float(args);
    if (strcmp(name, "float_from_bits") == 0 || strcmp(name, "float_to_bits") == 0) {
        bool from = strcmp(name, "float_from_bits") == 0;
        if (args[0].type != (from ? VAL_INT : VAL_FLOAT)) {
            fputs("I require the exact input type for binary64 bit transport.\n", stderr);
            exit(EXIT_FAILURE);
        }
        return from ? create_float(nl_float_from_bits(args[0].as.int_val))
                    : create_int(nl_float_to_bits(args[0].as.float_val));
    }
    if (strcmp(name, "cast_bool") == 0) return builtin_cast_bool(args);
    if (strcmp(name, "cast_string") == 0) return builtin_cast_string(args);
    if (strcmp(name, "null_opaque") == 0) return builtin_null_opaque(args);
    if (strcmp(name, "to_string") == 0) return builtin_to_string(args);

    /* Additional type conversion functions */
    if (strcmp(name, "float_to_string") == 0) {
        if (args[0].type != VAL_FLOAT && args[0].type != VAL_INT) {
            return create_string("0.0");
        }
        double v = args[0].type == VAL_FLOAT ? args[0].as.float_val : (double)args[0].as.int_val;
        char buffer[64];
        nano_rt_f64_format(buffer, sizeof(buffer), v);
        if (strchr(buffer, '.') == NULL && strchr(buffer, 'e') == NULL
                && strchr(buffer, 'n') == NULL && strchr(buffer, 'i') == NULL) {
            size_t len = strlen(buffer);
            if (len + 2 < sizeof(buffer)) {
                buffer[len]     = '.';
                buffer[len + 1] = '0';
                buffer[len + 2] = '\0';
            }
        }
        return create_string(buffer);
    }
    if (strcmp(name, "bool_to_string") == 0) {
        if (args[0].type == VAL_BOOL) {
            return create_string(args[0].as.bool_val ? "true" : "false");
        }
        return create_string("false");
    }
    if (strcmp(name, "string_to_float") == 0) {
        if (args[0].type != VAL_STRING) return create_float(0.0);
        return create_float(nl_binary64_prefix(args[0].as.string_val));
    }

    /* get_argc / get_argv for CLI programs */
    if (strcmp(name, "get_argc") == 0) {
        return create_int(g_argc);
    }
    if (strcmp(name, "get_argv") == 0) {
        if (args[0].type != VAL_INT) return create_string("");
        int idx = (int)args[0].as.int_val;
        if (idx < 0 || idx >= g_argc || !g_argv[idx]) return create_string("");
        return create_string(g_argv[idx]);
    }

    /* Timing utilities */
    if (strcmp(name, "nl_get_time_ms") == 0) {
        struct timespec ts;
        if (node->as.call.arg_count != 0 || clock_gettime(CLOCK_REALTIME, &ts) != 0) {
            fprintf(stderr, "I cannot read epoch milliseconds.\n");
            exit(1);
        }
        return create_int((long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
    }
    if (strcmp(name, "nl_timing_get_nanoseconds") == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        long long ns = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
        return create_int(ns);
    }

    /* String builder builtins (interpreter-native implementations) */
    if (strcmp(name, "nl_sb_new") == 0) {
        /* Allocate an EvalSB on the heap; store pointer as int */
        EvalSB *sb = malloc(sizeof(EvalSB));
        *sb = eval_sb_new(256);
        return create_int((long long)(intptr_t)sb);
    }
    if (strcmp(name, "nl_sb_with_capacity") == 0) {
        size_t cap = (args[0].type == VAL_INT) ? (size_t)args[0].as.int_val : 256;
        EvalSB *sb = malloc(sizeof(EvalSB));
        *sb = eval_sb_new(cap);
        return create_int((long long)(intptr_t)sb);
    }
    if (strcmp(name, "nl_sb_append") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        if (sb && args[1].type == VAL_STRING && args[1].as.string_val)
            eval_sb_append_cstr(sb, args[1].as.string_val);
        return create_void();
    }
    if (strcmp(name, "nl_sb_append_char") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        if (sb && args[1].type == VAL_INT) {
            char c = (char)args[1].as.int_val;
            eval_sb_append_char(sb, c);
        }
        return create_void();
    }
    if (strcmp(name, "nl_sb_clear") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        if (sb) { sb->len = 0; if (sb->buf) sb->buf[0] = '\0'; }
        return create_void();
    }
    if (strcmp(name, "nl_sb_length") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        return create_int(sb ? (long long)sb->len : 0);
    }
    if (strcmp(name, "nl_sb_capacity") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        return create_int(sb ? (long long)sb->cap : 0);
    }
    if (strcmp(name, "nl_sb_to_string") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        if (!sb || !sb->buf) return create_string("");
        return create_string(sb->buf);
    }
    if (strcmp(name, "nl_sb_free") == 0) {
        EvalSB *sb = (EvalSB*)(intptr_t)args[0].as.int_val;
        if (sb) { free(sb->buf); free(sb); }
        return create_void();
    }
    
    /* String operations */
    if (strcmp(name, "str_length") == 0) return builtin_str_length(args);
    if (strcmp(name, "str_concat") == 0) return builtin_str_concat(args);
    if (strcmp(name, "str_substring") == 0) return builtin_str_substring(args);
    if (strcmp(name, "str_contains") == 0) return builtin_str_contains(args);
    if (strcmp(name, "str_equals") == 0) return builtin_str_equals(args);
    if (strcmp(name, "str_starts_with") == 0) return builtin_str_starts_with(args);
    if (strcmp(name, "str_ends_with") == 0) return builtin_str_ends_with(args);
    if (strcmp(name, "str_index_of") == 0) return builtin_str_index_of(args);
    if (strcmp(name, "str_last_index_of") == 0) return builtin_str_last_index_of(args);
    if (strcmp(name, "str_trim") == 0) return builtin_str_trim(args);
    if (strcmp(name, "str_trim_left") == 0) return builtin_str_trim_left(args);
    if (strcmp(name, "str_trim_right") == 0) return builtin_str_trim_right(args);
    if (strcmp(name, "str_to_lower") == 0) return builtin_str_to_lower(args);
    if (strcmp(name, "str_to_upper") == 0) return builtin_str_to_upper(args);
    if (strcmp(name, "str_replace") == 0) return builtin_str_replace(args);
    if (strcmp(name, "format") == 0) {
        int arg_count = node->as.call.arg_count;
        if (arg_count < 1 || args[0].type != VAL_STRING) {
            fprintf(stderr, "Error: format requires a string template as first argument\n");
            return create_string("");
        }
        const char *fmt = args[0].as.string_val;
        /* Build result by scanning format string and substituting %s/%d/%f/%g */
        size_t buf_cap = 256;
        char *buf = malloc(buf_cap);
        if (!buf) return create_string("");
        size_t buf_len = 0;
        int arg_idx = 1;
        const char *p = fmt;
        while (*p) {
            if (*p == '%' && (p[1] == 's' || p[1] == 'd' || p[1] == 'f' || p[1] == 'g') && arg_idx < arg_count) {
                /* Convert arg to string */
                char tmp[64];
                const char *s = NULL;
                if (args[arg_idx].type == VAL_STRING) {
                    s = args[arg_idx].as.string_val ? args[arg_idx].as.string_val : "";
                } else if (args[arg_idx].type == VAL_INT) {
                    snprintf(tmp, sizeof(tmp), "%lld", (long long)args[arg_idx].as.int_val);
                    s = tmp;
                } else if (args[arg_idx].type == VAL_FLOAT) {
                    nano_rt_f64_format(tmp, sizeof(tmp), args[arg_idx].as.float_val);
                    s = tmp;
                } else if (args[arg_idx].type == VAL_BOOL) {
                    s = args[arg_idx].as.bool_val ? "true" : "false";
                } else {
                    s = "";
                }
                size_t slen = strlen(s);
                if (buf_len + slen + 1 > buf_cap) {
                    buf_cap = (buf_len + slen + 1) * 2;
                    char *new_buf = realloc(buf, buf_cap);
                    if (!new_buf) { free(buf); return create_string(""); }
                    buf = new_buf;
                }
                memcpy(buf + buf_len, s, slen);
                buf_len += slen;
                arg_idx++;
                p += 2;
            } else {
                if (buf_len + 2 > buf_cap) {
                    buf_cap *= 2;
                    char *new_buf = realloc(buf, buf_cap);
                    if (!new_buf) { free(buf); return create_string(""); }
                    buf = new_buf;
                }
                buf[buf_len++] = *p++;
            }
        }
        buf[buf_len] = '\0';
        Value v = create_string(buf);
        free(buf);
        return v;
    }
    if (strcmp(name, "str_split") == 0 &&
        env_native_array_is_builtin(env, name, node->line, node->column)) {
        if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
            fprintf(stderr, "Error: str_split requires two string arguments\n");
            return create_void();
        }
        const char *str = args[0].as.string_val;
        const char *delim = args[1].as.string_val;
        DynArray *result = dyn_array_new(ELEM_STRING);
        if (!result) {
            fprintf(stderr, "I cannot allocate a complete split-string result.\n");
            abort();
        }
        size_t delim_len = strlen(delim);
        if (delim_len == 0) {
            size_t str_len = strlen(str);
            for (size_t i = 0; i < str_len; i++) {
                char *ch = gc_alloc_string(1);
                if (!ch) {
                    fprintf(stderr, "I cannot allocate a complete split-string result.\n");
                    abort();
                }
                ch[0] = str[i];
                ch[1] = '\0';
                dyn_array_push_string(result, ch);
            }
        } else {
            const char *start = str;
            const char *found;
            while ((found = strstr(start, delim)) != NULL) {
                size_t seg_len = (size_t)(found - start);
                char *seg = gc_alloc_string(seg_len);
                if (!seg) {
                    fprintf(stderr, "I cannot allocate a complete split-string result.\n");
                    abort();
                }
                memcpy(seg, start, seg_len);
                seg[seg_len] = '\0';
                dyn_array_push_string(result, seg);
                start = found + delim_len;
            }
            size_t rest_len = strlen(start);
            char *tail = gc_alloc_string(rest_len);
            if (!tail) {
                fprintf(stderr, "I cannot allocate a complete split-string result.\n");
                abort();
            }
            memcpy(tail, start, rest_len + 1);
            dyn_array_push_string(result, tail);
        }
        return create_dyn_array(result);
    }
    if (strcmp(name, "str_join") == 0) {
        if ((args[0].type != VAL_DYN_ARRAY && args[0].type != VAL_ARRAY) || args[1].type != VAL_STRING) {
            fprintf(stderr, "Error: str_join requires array<string> and string\n");
            return create_void();
        }
        DynArray *arr = args[0].type == VAL_DYN_ARRAY ? args[0].as.dyn_array_val : NULL;
        Array *literal = args[0].type == VAL_ARRAY ? args[0].as.array_val : NULL;
        if (!arr && !literal) return create_void();
        const char *delim = args[1].as.string_val;
        int64_t count = arr ? dyn_array_length(arr) : literal->length;
        if (count == 0) return create_string("");
        if (count < 0 || (arr && arr->elem_type != ELEM_STRING) ||
            (literal && literal->element_type != VAL_STRING)) return create_void();
        size_t delim_len = strlen(delim);
        size_t total = 0;
        for (int64_t i = 0; i < count; i++) {
            const char *s = arr ? dyn_array_get_string(arr, i) : ((char **)literal->data)[i];
            size_t length = s ? strlen(s) : 0;
            if (length > SIZE_MAX - 1 - total) return create_void();
            total += length;
            if (i < count - 1) {
                if (delim_len > SIZE_MAX - 1 - total) return create_void();
                total += delim_len;
            }
        }
        char *buf = malloc(total + 1);
        if (!buf) return create_string("");
        size_t pos = 0;
        for (int64_t i = 0; i < count; i++) {
            const char *s = arr ? dyn_array_get_string(arr, i) : ((char **)literal->data)[i];
            if (s) { size_t slen = strlen(s); memcpy(buf + pos, s, slen); pos += slen; }
            if (i < count - 1) { memcpy(buf + pos, delim, delim_len); pos += delim_len; }
        }
        buf[pos] = '\0';
        Value v = create_string(buf);
        free(buf);
        return v;
    }

    /* Bytes helpers */
    if (strcmp(name, "bytes_from_string") == 0) return builtin_bytes_from_string(args);
    if (strcmp(name, "string_from_bytes") == 0) return builtin_string_from_bytes(args);
    
    /* Advanced string operations */
    if (strcmp(name, "char_at") == 0) {
        if (args[0].type != VAL_STRING || args[1].type != VAL_INT) {
            fprintf(stderr, "Error: char_at requires string and int\n");
            return create_void();
        }
        const char *str = args[0].as.string_val;
        long long index = args[1].as.int_val;
        /* Safety: Bound string scan to 64MB (large enough for self-hosting) */
        int len = strnlen(str, 64*1024*1024);
        if (index < 0 || index >= len) {
            fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", (long long)index, len);
            return create_void();
        }
        return create_int((unsigned char)str[index]);
    }
    
    if (strcmp(name, "string_from_char") == 0) {
        if (args[0].type != VAL_INT) {
            fprintf(stderr, "Error: string_from_char requires int\n");
            return create_void();
        }
        char buffer[2];
        buffer[0] = (char)args[0].as.int_val;
        buffer[1] = '\0';
        return create_string(buffer);
    }
    
    /* Character classification */
    if (strcmp(name, "is_digit") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= '0' && c <= '9');
    }
    
    if (strcmp(name, "is_alpha") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
    }
    
    if (strcmp(name, "is_alnum") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool((c >= '0' && c <= '9') || 
                           (c >= 'a' && c <= 'z') || 
                           (c >= 'A' && c <= 'Z'));
    }
    
    if (strcmp(name, "is_whitespace") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c == ' ' || c == '\t' || c == '\n' || c == '\r');
    }
    
    if (strcmp(name, "is_upper") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= 'A' && c <= 'Z');
    }
    
    if (strcmp(name, "is_lower") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= 'a' && c <= 'z');
    }
    
    /* Type conversions */
    if (strcmp(name, "int_to_string") == 0) {
        if (args[0].type != VAL_INT) {
            return create_string("0");
        }
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%lld", (long long)args[0].as.int_val);
        return create_string(buffer);
    }
    
    if (strcmp(name, "string_to_int") == 0) {
        if (args[0].type != VAL_STRING) {
            return create_int(0);
        }
        long long result = strtoll(args[0].as.string_val, NULL, 10);
        return create_int(result);
    }
    
    if (strcmp(name, "digit_value") == 0) {
        if (args[0].type != VAL_INT) return create_int(-1);
        int c = (int)args[0].as.int_val;
        if (c >= '0' && c <= '9') {
            return create_int(c - '0');
        }
        return create_int(-1);
    }
    
    if (strcmp(name, "char_to_lower") == 0) {
        if (args[0].type != VAL_INT) return create_int(args[0].as.int_val);
        int c = (int)args[0].as.int_val;
        if (c >= 'A' && c <= 'Z') {
            return create_int(c + 32);
        }
        return create_int(c);
    }
    
    if (strcmp(name, "char_to_upper") == 0) {
        if (args[0].type != VAL_INT) return create_int(args[0].as.int_val);
        int c = (int)args[0].as.int_val;
        if (c >= 'a' && c <= 'z') {
            return create_int(c - 32);
        }
        return create_int(c);
    }
    
    /* Array operations */
    if (strcmp(name, "at") == 0 || strcmp(name, "array_get") == 0) return builtin_at(args);
    if (strcmp(name, "array_length") == 0) return builtin_array_length(args);
    if (strcmp(name, "array_new") == 0) return builtin_array_new(args);
    if (strcmp(name, "array_set") == 0) {
        if (node->as.call.checked_u8_array_mutation && !bound_name &&
            node->as.call.arg_count == 3)
            args[2] = eval_checked_scalar_destination(TYPE_U8, args[2]);
        return builtin_array_set(args);
    }
    if (strcmp(name, "array_slice") == 0) return builtin_array_slice(args);
    
    /* Higher-order array functions */
    if (callback_kind) {
        char *callback_name = NULL;
        if (args[callback_index].type == VAL_FUNCTION) {
            const char *name_to_copy = args[callback_index].as.function_val.function_name;
#ifdef NANO_TEST_CALLBACK_SNAPSHOT
            extern char *nano_test_callback_name(const char *name);
            callback_name = name_to_copy ? nano_test_callback_name(name_to_copy) : NULL;
#else
            callback_name = name_to_copy ? strdup(name_to_copy) : NULL;
#endif
            if (!callback_name) {
                discard_declared_callback(owned_callback);
                fprintf(stderr, "I could not retain the array callback name.\n");
                return create_void();
            }
            /* A callback can replace its own owning Symbol. These three
             * builtins use only the name; their local descriptor borrows this
             * snapshot until every callback and early return is finished. */
            args[callback_index].as.function_val.function_name = callback_name;
            args[callback_index].as.function_val.signature = NULL;
        }
        Value result = callback_kind == 1 ? builtin_map(args, env) :
            callback_kind == 2 ? builtin_filter(args, env) : builtin_reduce(args, env);
        free(callback_name);
        discard_declared_callback(owned_callback);
        return result;
    }
    
    /* Dynamic array operations (GC-managed) */
    if (strcmp(name, "array_push") == 0 && is_builtin_array_push) {
        if (node->as.call.checked_u8_array_mutation && !bound_name &&
            node->as.call.arg_count == 2)
            args[1] = eval_checked_scalar_destination(TYPE_U8, args[1]);
        return builtin_array_push(args);
    }
    if (strcmp(name, "array_pop") == 0) return builtin_array_pop(args);
    if (strcmp(name, "array_remove_at") == 0) return builtin_array_remove_at(args);
    if (strcmp(name, "array_sort") == 0) return builtin_array_sort(args);
    if (strcmp(name, "array_reverse") == 0) return builtin_array_reverse(args);
    if (strcmp(name, "array_contains") == 0) return builtin_array_contains(args);
    if (strcmp(name, "array_index_of") == 0) return builtin_array_index_of(args);
    
    /* list_int operations - delegate to C runtime */
    if (strcmp(name, "list_int_new") == 0) {
        List_int *list = list_int_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_int_with_capacity") == 0) {
        List_int *list = list_int_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_int_push") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_push(list, args[1].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_pop") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_pop(list));
    }
    if (strcmp(name, "list_int_get") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_get(list, args[1].as.int_val));
    }
    if (strcmp(name, "list_int_set") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_set(list, args[1].as.int_val, args[2].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_insert") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_insert(list, args[1].as.int_val, args[2].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_remove") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_remove(list, args[1].as.int_val));
    }
    if (strcmp(name, "list_int_length") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_length(list));
    }
    if (strcmp(name, "list_int_capacity") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_capacity(list));
    }
    if (strcmp(name, "list_int_is_empty") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_bool(list_int_is_empty(list));
    }
    if (strcmp(name, "list_int_clear") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_int_free") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_free(list);
        return create_void();
    }

    /* list_string operations - delegate to C runtime */
    if (strcmp(name, "list_string_new") == 0) {
        List_string *list = list_string_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_string_with_capacity") == 0) {
        List_string *list = list_string_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_string_push") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_push(list, args[1].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_pop") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_pop(list);
        Value result = create_string(str);
        free(str);  /* list_string_pop returns strdup'd string */
        return result;
    }
    if (strcmp(name, "list_string_get") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_get(list, args[1].as.int_val);
        return create_string(str);
    }
    if (strcmp(name, "list_string_set") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_set(list, args[1].as.int_val, args[2].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_insert") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_insert(list, args[1].as.int_val, args[2].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_remove") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_remove(list, args[1].as.int_val);
        Value result = create_string(str);
        free(str);  /* list_string_remove returns strdup'd string */
        return result;
    }
    if (strcmp(name, "list_string_length") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_int(list_string_length(list));
    }
    if (strcmp(name, "list_string_capacity") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_int(list_string_capacity(list));
    }
    if (strcmp(name, "list_string_is_empty") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_bool(list_string_is_empty(list));
    }
    if (strcmp(name, "list_string_clear") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_string_free") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_free(list);
        return create_void();
    }

    /* list_Token operations - delegate to C runtime */
    /* Note: Token structs are stored as pointers for now */
    /* When we rewrite lexer in nanolang, we'll use proper Token struct values */
    if (strcmp(name, "nl_list_Token_new") == 0) {
        List_Token *list = nl_list_Token_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "nl_list_Token_with_capacity") == 0) {
        List_Token *list = nl_list_Token_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "nl_list_Token_push") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        /* For now, args[1] should be a Token struct pointer */
        /* When we have proper Token struct support, this will change */
        Token *token = (Token*)args[1].as.int_val;
        if (token) {
            nl_list_Token_push(list, *token);
        }
        return create_void();
    }
    if (strcmp(name, "nl_list_Token_pop") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        Token token = nl_list_Token_pop(list);
        /* Return token as struct value - for now return pointer */
        /* TODO: Convert Token to proper struct value when we have Token struct support */
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "nl_list_Token_get") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        Token token = nl_list_Token_get(list, args[1].as.int_val);
        /* Return token as struct value - for now return pointer */
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "nl_list_Token_set") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        Token *token = (Token*)args[2].as.int_val;
        if (token) {
            nl_list_Token_set(list, args[1].as.int_val, *token);
        }
        return create_void();
    }
    if (strcmp(name, "nl_list_Token_insert") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        Token *token = (Token*)args[2].as.int_val;
        if (token) {
            nl_list_Token_insert(list, args[1].as.int_val, *token);
        }
        return create_void();
    }
    if (strcmp(name, "nl_list_Token_remove") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        Token token = nl_list_Token_remove(list, args[1].as.int_val);
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "nl_list_Token_length") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        return create_int(nl_list_Token_length(list));
    }
    if (strcmp(name, "nl_list_Token_capacity") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        return create_int(nl_list_Token_capacity(list));
    }
    if (strcmp(name, "nl_list_Token_is_empty") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        return create_bool(nl_list_Token_is_empty(list));
    }
    if (strcmp(name, "nl_list_Token_clear") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        nl_list_Token_clear(list);
        return create_void();
    }
    if (strcmp(name, "nl_list_Token_free") == 0) {
        List_Token *list = (List_Token*)args[0].as.int_val;
        nl_list_Token_free(list);
        return create_void();
    }

    Value record_list_result;
    if (eval_record_list_call(name, args, node->as.call.arg_count, env, &record_list_result))
        return record_list_result;

    /* External C library functions - provide interpreter implementations */
    if (strcmp(name, "rand") == 0) {
        return create_int(rand());
    }
    if (strcmp(name, "srand") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: srand expects 1 int argument\n");
            return create_void();
        }
        srand((unsigned int)args[0].as.int_val);
        return create_void();
    }
    if (strcmp(name, "time") == 0) {
        /* Simplified: ignore the argument, just return current time */
        return create_int((long long)time(NULL));
    }
    
    /* C string functions - map to interpreter built-ins */
    if (strcmp(name, "strlen") == 0) {
        return builtin_str_length(args);
    }
    if (strcmp(name, "strcmp") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
            fprintf(stderr, "Error: strcmp requires 2 string arguments\n");
            return create_void();
        }
        int result = strcmp(args[0].as.string_val, args[1].as.string_val);
        return create_int(result);
    }
    if (strcmp(name, "strncmp") == 0) {
        if (node->as.call.arg_count < 3 || args[0].type != VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_INT) {
            fprintf(stderr, "Error: strncmp requires 2 string arguments and 1 int argument\n");
            return create_void();
        }
        int n = (int)args[2].as.int_val;
        if (n < 0) n = 0;
        int result = strncmp(args[0].as.string_val, args[1].as.string_val, (size_t)n);
        return create_int(result);
    }
    
    /* C math functions - provide interpreter implementations */
    if (strcmp(name, "asin") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: asin requires 1 float argument\n");
            return create_void();
        }
        return create_float(asin(args[0].as.float_val));
    }
    if (strcmp(name, "acos") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: acos requires 1 float argument\n");
            return create_void();
        }
        return create_float(acos(args[0].as.float_val));
    }
    if (strcmp(name, "atan") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: atan requires 1 float argument\n");
            return create_void();
        }
        return create_float(atan(args[0].as.float_val));
    }
    if (strcmp(name, "exp") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: exp requires 1 float argument\n");
            return create_void();
        }
        return create_float(exp(args[0].as.float_val));
    }
    if (strcmp(name, "exp2") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: exp2 requires 1 float argument\n");
            return create_void();
        }
        return create_float(exp2(args[0].as.float_val));
    }
    if (strcmp(name, "log") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log requires 1 float argument\n");
            return create_void();
        }
        return create_float(log(args[0].as.float_val));
    }
    if (strcmp(name, "log10") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log10 requires 1 float argument\n");
            return create_void();
        }
        return create_float(log10(args[0].as.float_val));
    }
    if (strcmp(name, "log2") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log2 requires 1 float argument\n");
            return create_void();
        }
        return create_float(log2(args[0].as.float_val));
    }
    if (strcmp(name, "log1p") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log1p requires 1 float argument\n");
            return create_void();
        }
        return create_float(log1p(args[0].as.float_val));
    }
    if (strcmp(name, "expm1") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: expm1 requires 1 float argument\n");
            return create_void();
        }
        return create_float(expm1(args[0].as.float_val));
    }
    if (strcmp(name, "cbrt") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: cbrt requires 1 float argument\n");
            return create_void();
        }
        return create_float(cbrt(args[0].as.float_val));
    }
    if (strcmp(name, "hypot") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: hypot requires 2 float arguments\n");
            return create_void();
        }
        return create_float(hypot(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "sinh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: sinh requires 1 float argument\n");
            return create_void();
        }
        return create_float(sinh(args[0].as.float_val));
    }
    if (strcmp(name, "cosh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: cosh requires 1 float argument\n");
            return create_void();
        }
        return create_float(cosh(args[0].as.float_val));
    }
    if (strcmp(name, "tanh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: tanh requires 1 float argument\n");
            return create_void();
        }
        return create_float(tanh(args[0].as.float_val));
    }
    if (strcmp(name, "asinh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: asinh requires 1 float argument\n");
            return create_void();
        }
        return create_float(asinh(args[0].as.float_val));
    }
    if (strcmp(name, "acosh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: acosh requires 1 float argument\n");
            return create_void();
        }
        return create_float(acosh(args[0].as.float_val));
    }
    if (strcmp(name, "atanh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: atanh requires 1 float argument\n");
            return create_void();
        }
        return create_float(atanh(args[0].as.float_val));
    }
    if (strcmp(name, "fmod") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fmod requires 2 float arguments\n");
            return create_void();
        }
        return create_float(fmod(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "trunc") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: trunc requires 1 float argument\n");
            return create_void();
        }
        return create_float(trunc(args[0].as.float_val));
    }
    if (strcmp(name, "rint") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: rint requires 1 float argument\n");
            return create_void();
        }
        return create_float(rint(args[0].as.float_val));
    }
    if (strcmp(name, "nearbyint") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: nearbyint requires 1 float argument\n");
            return create_void();
        }
        return create_float(nearbyint(args[0].as.float_val));
    }
    if (strcmp(name, "remainder") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: remainder requires 2 float arguments\n");
            return create_void();
        }
        return create_float(remainder(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "fmin") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fmin requires 2 float arguments\n");
            return create_void();
        }
        return create_float(fmin(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "fmax") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fmax requires 2 float arguments\n");
            return create_void();
        }
        return create_float(fmax(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "copysign") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: copysign requires 2 float arguments\n");
            return create_void();
        }
        return create_float(copysign(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "fabs") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fabs requires 1 float argument\n");
            return create_void();
        }
        return create_float(fabs(args[0].as.float_val));
    }
    
    /* C character functions */
    if (strcmp(name, "getchar") == 0) {
        int c = getchar();
        return create_int(c);
    }
    if (strcmp(name, "putchar") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: putchar requires 1 int argument\n");
            return create_void();
        }
        int c = putchar((int)args[0].as.int_val);
        return create_int(c);
    }
    if (strcmp(name, "isalpha") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isalpha requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_isalpha((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isdigit") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isdigit requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_isdigit((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isalnum") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isalnum requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_isalnum((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "islower") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: islower requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_islower((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isupper") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isupper requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_isupper((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "tolower") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: tolower requires 1 int argument\n");
            return create_void();
        }
        return create_int(nl_ascii_tolower((int)args[0].as.int_val));
    }
    if (strcmp(name, "toupper") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: toupper requires 1 int argument\n");
            return create_void();
        }
        return create_int(nl_ascii_toupper((int)args[0].as.int_val));
    }
    if (strcmp(name, "isspace") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isspace requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_isspace((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isprint") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isprint requires 1 int argument\n");
            return create_void();
        }
        return create_bool(nl_ascii_isprint((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "ispunct") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: ispunct requires 1 int argument\n");
            return create_void();
        }
        return create_bool(ispunct((int)args[0].as.int_val) != 0);
    }

    /* Get user-defined function */
    Function *func = env_get_function(env, name);

    /* HashMap<K,V> core built-ins (only when not shadowed by a user-defined function) */
    if (!func) {
        if (strcmp(name, "map_new") == 0) {
            if (node->as.call.arg_count != 0) {
                fprintf(stderr, "Error: map_new requires 0 arguments\n");
                return create_void();
            }

            NLHashMapKeyType kt;
            NLHashMapValType vt;
            const char *mono = node->as.call.return_struct_type_name;
            if (!eval_hm_parse_monomorph(mono, &kt, &vt)) {
                fprintf(stderr, "Error: map_new requires HashMap<K,V> type context\n");
                return create_void();
            }

            NLHashMapCore *hm = eval_hm_alloc(kt, vt, 16);
            return create_int((long long)hm);
        }

        if (strcmp(name, "map_put") == 0 || strcmp(name, "map_set") == 0) {
            if (node->as.call.arg_count != 3) {
                fprintf(stderr, "Error: %s requires 3 arguments\n", name);
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: %s expects HashMap as first argument\n", name);
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            if (!hm) return create_void();

            /* Resize */
            if ((hm->size + hm->tombstones) * 10 >= hm->capacity * 7) {
                eval_hm_rehash(hm, hm->capacity * 2);
            }

            /* Type checks */
            if (hm->key_type == NL_HM_KEY_INT && args[1].type != VAL_INT) {
                fprintf(stderr, "Error: %s expects int key\n", name);
                return create_void();
            }
            if (hm->key_type == NL_HM_KEY_STRING && args[1].type != VAL_STRING) {
                fprintf(stderr, "Error: %s expects string key\n", name);
                return create_void();
            }
            if (hm->val_type == NL_HM_VAL_INT && args[2].type != VAL_INT) {
                fprintf(stderr, "Error: %s expects int value\n", name);
                return create_void();
            }
            if (hm->val_type == NL_HM_VAL_STRING && args[2].type != VAL_STRING) {
                fprintf(stderr, "Error: %s expects string value\n", name);
                return create_void();
            }

            bool found = false;
            int64_t idx = eval_hm_find_slot(hm, &args[1], &found);
            if (idx < 0) return create_void();
            NLHashMapEntry *e = &hm->entries[idx];

            if (found) {
                if (hm->val_type == NL_HM_VAL_STRING) {
                    if (e->value.s) free(e->value.s);
                    e->value.s = args[2].as.string_val ? strdup(args[2].as.string_val) : strdup("");
                } else {
                    e->value.i = args[2].as.int_val;
                }
                return create_void();
            }

            if (e->state == 2) hm->tombstones--;
            e->state = 1;
            if (hm->key_type == NL_HM_KEY_STRING) {
                e->key.s = args[1].as.string_val ? strdup(args[1].as.string_val) : strdup("");
            } else {
                e->key.i = args[1].as.int_val;
            }
            if (hm->val_type == NL_HM_VAL_STRING) {
                e->value.s = args[2].as.string_val ? strdup(args[2].as.string_val) : strdup("");
            } else {
                e->value.i = args[2].as.int_val;
            }
            hm->size++;
            return create_void();
        }

        if (strcmp(name, "map_get") == 0) {
            if (node->as.call.arg_count != 2) {
                fprintf(stderr, "Error: map_get requires 2 arguments\n");
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: map_get expects HashMap as first argument\n");
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            if (!hm) {
                return (Value){ .type = VAL_VOID };
            }
            bool found = false;
            int64_t idx = eval_hm_find_slot(hm, &args[1], &found);
            if (!found || idx < 0) {
                if (hm->val_type == NL_HM_VAL_STRING) return create_string("");
                return create_int(0);
            }
            NLHashMapEntry *e = &hm->entries[idx];
            if (hm->val_type == NL_HM_VAL_STRING) return create_string(e->value.s ? e->value.s : "");
            return create_int(e->value.i);
        }

        if (strcmp(name, "map_has") == 0) {
            if (node->as.call.arg_count != 2) {
                fprintf(stderr, "Error: map_has requires 2 arguments\n");
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: map_has expects HashMap as first argument\n");
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            if (!hm) return create_bool(false);
            bool found = false;
            (void)eval_hm_find_slot(hm, &args[1], &found);
            return create_bool(found);
        }

        if (strcmp(name, "map_remove") == 0) {
            if (node->as.call.arg_count != 2) {
                fprintf(stderr, "Error: map_remove requires 2 arguments\n");
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: map_remove expects HashMap as first argument\n");
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            if (!hm) return create_void();
            bool found = false;
            int64_t idx = eval_hm_find_slot(hm, &args[1], &found);
            if (!found || idx < 0) return create_void();
            NLHashMapEntry *e = &hm->entries[idx];
            eval_hm_free_entry(hm, e);
            e->state = 2;
            hm->size--;
            hm->tombstones++;
            return create_void();
        }

        if (strcmp(name, "map_length") == 0 || strcmp(name, "map_size") == 0) {
            if (node->as.call.arg_count != 1) {
                fprintf(stderr, "Error: %s requires 1 argument\n", name);
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: %s expects HashMap as first argument\n", name);
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            return create_int(hm ? hm->size : 0);
        }

        if (strcmp(name, "map_clear") == 0 || strcmp(name, "map_free") == 0) {
            if (node->as.call.arg_count != 1) {
                fprintf(stderr, "Error: %s requires 1 argument\n", name);
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: %s expects HashMap as first argument\n", name);
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            if (!hm) return create_void();
            if (strcmp(name, "map_free") == 0) {
                eval_hm_free(hm);
            } else {
                eval_hm_clear(hm);
            }
            return create_void();
        }

        if (strcmp(name, "map_keys") == 0 || strcmp(name, "map_values") == 0) {
            if (node->as.call.arg_count != 1) {
                fprintf(stderr, "Error: %s requires 1 argument\n", name);
                return create_void();
            }
            if (args[0].type != VAL_INT) {
                fprintf(stderr, "Error: %s expects HashMap as first argument\n", name);
                return create_void();
            }
            NLHashMapCore *hm = (NLHashMapCore*)args[0].as.int_val;
            if (!hm) return create_array(VAL_INT, 0, 0);

            bool is_keys = (strcmp(name, "map_keys") == 0);
            ValueType elem_type;
            if (is_keys) {
                elem_type = (hm->key_type == NL_HM_KEY_STRING) ? VAL_STRING : VAL_INT;
            } else {
                elem_type = (hm->val_type == NL_HM_VAL_STRING) ? VAL_STRING : VAL_INT;
            }

            Value out = create_array(elem_type, (int)hm->size, (int)hm->size);
            int out_idx = 0;
            for (int64_t i = 0; i < hm->capacity; i++) {
                NLHashMapEntry *e = &hm->entries[i];
                if (e->state != 1) continue;
                if (elem_type == VAL_STRING) {
                    char *s = NULL;
                    if (is_keys) s = e->key.s; else s = e->value.s;
                    ((char**)out.as.array_val->data)[out_idx++] = s;
                } else {
                    int64_t v = 0;
                    if (is_keys) v = e->key.i; else v = e->value.i;
                    ((long long*)out.as.array_val->data)[out_idx++] = v;
                }
            }
            out.as.array_val->length = out_idx;
            return out;
        }
    }
    
    if (!func) {
        fprintf(stderr, "Error: Undefined function '%s'\n", name);
        return create_void();
    }

    /* Coroutine scheduler: route async function calls through the run queue.
     * When an async fn is called, spawn a coroutine and run it to completion.
     * In synchronous/test mode the behaviour is identical to before. */
    if (func->is_async && func->body != NULL) {
        CoroCallArgs *ca = coro_bundle_new(env, name, node->as.call.arg_count);
        if (ca) {
            bool copied = true;
            for (int i = 0; copied && i < ca->arg_count; ++i) {
                Type formal = func->params && i < func->param_count ? func->params[i].type : TYPE_UNKNOWN;
                bool borrowed = formal == TYPE_BORROW_SHARED || formal == TYPE_BORROW_MUT;
                copied = coro_bundle_argument(ca, i, args[i], borrowed);
            }
            int coro_id = copied ? coro_bundle_enqueue(ca) : -1;
            if (coro_id >= 0) {
                Value result = eval_task_result(env, coro_id, true);
                if (!nano_coro_release(coro_id)) {
                    fprintf(stderr, "I cannot release a completed inactive task.\n"); exit(1);
                }
                return result;
            }
            coro_bundle_drop(ca);
        }
        /* I preserve the existing synchronous fallback when preparation fails. */
    }

    /* If built-in with no body, already handled above */
    if (func->body == NULL) {
        /* Try FFI for extern functions */
        if (func->is_extern) {
            Value result = eval_foreign_call(func, args, node->as.call.arg_count,
                                            env, node->line, node->column);
            return result;
        }

        fprintf(stderr, "Error: Built-in function '%s' not implemented in interpreter\n", name);
        return create_void();
    }

    /* Trace function call */
    const char **param_names = NULL;
    if (func->params) {
        param_names = malloc(sizeof(char*) * func->param_count);
        for (int i = 0; i < func->param_count; i++) {
            param_names[i] = func->params[i].name;
        }
    }
    tracing_push_call(name);
    trace_function_call(name, args, node->as.call.arg_count, param_names, 
                        node->line, node->column);
    if (param_names) free(param_names);

    /* Create new environment for function */
    int old_symbol_count = env->symbol_count;

    /* Bind parameters with copies of string values */
    for (int i = 0; i < func->param_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }
        if (param_value.type == VAL_FUNCTION) {
            FunctionSignature *sig_copy = copy_function_signature(param_value.as.function_val.signature);
            param_value = create_function(param_value.as.function_val.function_name, sig_copy);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute function body */
    char return_boundary;
    const void *saved_return_target = g_eval_return_target;
    g_eval_return_target = &return_boundary;
    char *saved_module_context = env->current_module;
    env->current_module = func->module_name;
    Value result = create_void();
    for (int i = 0; i < func->body->as.block.count; i++) {
        ASTNode *stmt = func->body->as.block.statements[i];
        if (stmt->type == AST_RETURN) {
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            }
            break;
        }
        result = eval_statement(stmt, env);
        /* If statement returned a value (e.g., from if block with return), propagate it */
        if (result.is_return) {
            break;
        }
    }

    /* Pop call stack */
    g_eval_return_target = saved_return_target;
    env->current_module = saved_module_context;
    tracing_pop_call();

    /*
     * Make a copy of the result BEFORE cleaning up parameters.
     *
     * Function-local variables (including string temporaries) are freed when we
     * unwind the call frame, so any returned value that references them must
     * deep-copy those strings.
     */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    } else if (result.type == VAL_FUNCTION) {
        FunctionSignature *sig_copy = copy_function_signature(result.as.function_val.signature);
        return_value = create_function(result.as.function_val.function_name, sig_copy);
    } else if (result.type == VAL_STRUCT || result.type == VAL_TUPLE) {
        if (!env_value_snapshot(env, result, &return_value)) {
            fprintf(stderr, "I cannot copy a returned record.\n");
            exit(1);
        }
    }

    /* I consume only returns addressed to this call, not an enclosing handler owner. */
    return_value.is_return = result.is_return && result.return_target &&
        result.return_target != &return_boundary;
    return_value.return_target = return_value.is_return ? result.return_target : NULL;
    return_value.is_break = false;
    return_value.is_continue = false;

    eval_scope_release(env, old_symbol_count, true);

    /* An outer handler's return has not reached its destination yet. */
    if (!return_value.is_return)
        return_value = eval_checked_scalar_destination(func->return_type, return_value);
    return return_value;
}

/* I discard only record/string storage cloned by create_struct. Arrays and
 * other referenced fields remain borrowed, as in that constructor. */
static void discard_literal_record(StructValue *record) {
    env_discard_record(record);
}

static void discard_partial_owned_array(Array *array, int initialized) {
    for (int i = 0; i < initialized; i++) {
        if (array->element_type == VAL_STRING) free(((char **)array->data)[i]);
        else if (array->element_type == VAL_STRUCT)
            discard_literal_record(((StructValue **)array->data)[i]);
    }
    free(array->data);
    free(array);
}

/* Evaluate expression */
static Value eval_expression(ASTNode *expr, Environment *env) {
    if (!expr) return create_void();


    switch (expr->type) {
        case AST_NUMBER:
            return create_int(expr->as.number);

        case AST_FLOAT:
            return create_float(expr->as.float_val);

        case AST_STRING: {
            char *unescaped = nl_unescape_string(expr->as.string_val);
            Value v = create_string(unescaped);
            free(unescaped);
            return v;
        }

        case AST_BOOL:
            return create_bool(expr->as.bool_val);

        case AST_IDENTIFIER: {
            /* First check if it's a variable */
            Symbol *sym = env_get_var(env, expr->as.identifier);
            if (strcmp(expr->as.identifier, "array_push") == 0) {
                Function *declared = env_get_function(env, expr->as.identifier);
                if (declared && !declared->is_extern && declared->body &&
                    ((!env->current_module && !declared->module_name) ||
                     (env->current_module && declared->module_name &&
                      strcmp(env->current_module, declared->module_name) == 0))) {
                    /* I retain checker facts without reading them as runtime bindings.
                     * Evaluated locals/parameters have def_line == 0, even for VOID. */
                    sym = NULL;
                    for (int i = env->symbol_count - 1; i >= 0; i--) {
                        Symbol *candidate = &env->symbols[i];
                        if (!candidate->name ||
                            strcmp(candidate->name, expr->as.identifier) != 0) continue;
                        if (!candidate->is_global && candidate->def_line > 0 &&
                            candidate->value.type == VAL_VOID) continue;
                        sym = candidate;
                        break;
                    }
                }
            }
            if (sym) {
                /* Trace variable read */
#ifdef TRACING_ENABLED
                const char *scope = (g_tracing_config.call_stack_size > 0) ?
                    g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
                trace_var_read(expr->as.identifier, sym->value, expr->line, expr->column, scope);
#else
                trace_var_read(expr->as.identifier, sym->value, expr->line, expr->column, NULL);
#endif
                return sym->value;
            }
            
            /* If not a variable, check if it's a function (for first-class function support) */
            Function *func = env_get_function(env, expr->as.identifier);
            if (func) {
                /* Return a function value - create signature from function's parameters */
                Type *param_types = NULL;
                int actual_param_count = 0;
                if (func->param_count > 0 && func->params) {
                    param_types = malloc(sizeof(Type) * func->param_count);
                    for (int i = 0; i < func->param_count; i++) {
                        param_types[i] = func->params[i].type;
                    }
                    actual_param_count = func->param_count;
                }
                FunctionSignature *sig = create_function_signature(
                    param_types,
                    actual_param_count,
                    func->return_type
                );
                free(param_types); /* The signature owns its independent copy. */
                return create_function(expr->as.identifier, sig);
            }
            
            /* Neither variable nor function */
            fprintf(stderr, "Error: Undefined variable or function '%s'\n", expr->as.identifier);
            return create_void();
        }

        case AST_PREFIX_OP:
            return eval_prefix_op(expr, env);

        case AST_CALL:
            if (expr->as.call.borrow_mode) return eval_expression(expr->as.call.args[0], env);
            return eval_call(expr, env);

        case AST_MODULE_QUALIFIED_CALL: {
            const char *module_alias = expr->as.module_qualified_call.module_alias;
            const char *function_name = expr->as.module_qualified_call.function_name;
            int arg_count = expr->as.module_qualified_call.arg_count;

            size_t qlen = strlen(module_alias) + strlen(function_name) + 2;
            char *qualified_name = malloc(qlen);
            snprintf(qualified_name, qlen, "%s.%s", module_alias, function_name);

            Value *args = NULL;
            if (arg_count > 0) {
                args = malloc(sizeof(Value) * (size_t)arg_count);
                for (int i = 0; i < arg_count; i++) {
                    args[i] = eval_staged_argument(expr->as.module_qualified_call.args[i], env, qualified_name, i);
                    if (args[i].is_return) {
                        Value result = args[i];
                        free(args);
                        free(qualified_name);
                        return result;
                    }
                }
            }

            Value result = call_function_at(qualified_name, args, arg_count, env,
                                            expr->line, expr->column);
            free(args);
            free(qualified_name);
            return result;
        }

        case AST_ARRAY_LITERAL: {
            /* Evaluate array literal: [1, 2, 3] */
            int count = expr->as.array_literal.element_count;
            
            /* Empty array */
            if (count == 0) {
                ValueType element = VAL_INT;
                switch (expr->as.array_literal.element_type) {
                    case TYPE_FLOAT: element = VAL_FLOAT; break;
                    case TYPE_BOOL: element = VAL_BOOL; break;
                    case TYPE_STRING: element = VAL_STRING; break;
                    case TYPE_ARRAY: element = VAL_ARRAY; break;
                    case TYPE_STRUCT: element = VAL_STRUCT; break;
                    default: break;
                }
                return create_array(element, 0, 0);
            }
            
            /* Evaluate first element to determine type */
            Value first = eval_expression(expr->as.array_literal.elements[0], env);
            if (first.is_return) return first;
            ValueType elem_type = first.type;
            if (elem_type == VAL_DYN_ARRAY) elem_type = VAL_ARRAY;
            
            /* Create array */
            Value arr = create_array(elem_type, count, count);
            
            /* Set elements */
            for (int i = 0; i < count; i++) {
                Value elem = i == 0 ? first : eval_expression(expr->as.array_literal.elements[i], env);
                if (elem.is_return) {
                    discard_partial_owned_array(arr.as.array_val, i);
                    return elem;
                }
                /* I convert each checked byte destination after its single evaluation. */
                elem = eval_checked_scalar_destination(expr->as.array_literal.element_type, elem);
                
                /* Store element in array data */
                switch (elem_type) {
                    case VAL_ARRAY:
                        if (elem.type != VAL_ARRAY && elem.type != VAL_DYN_ARRAY) {
                            fprintf(stderr, "I require array values in a nested array literal.\n");
                            exit(1);
                        }
                        ((Value*)arr.as.array_val->data)[i] = elem;
                        break;
                    case VAL_INT:
                        ((long long*)arr.as.array_val->data)[i] = elem.as.int_val;
                        break;
                    case VAL_FLOAT:
                        ((double*)arr.as.array_val->data)[i] = elem.as.float_val;
                        break;
                    case VAL_BOOL:
                        ((bool*)arr.as.array_val->data)[i] = elem.as.bool_val;
                        break;
                    case VAL_STRING:
                        ((char**)arr.as.array_val->data)[i] = strdup(elem.as.string_val);
                        break;
                    case VAL_STRUCT: {
                        StructValue *sv = elem.as.struct_val;
                        Value copy = create_struct(sv->struct_name, sv->field_names,
                                                   sv->field_values, sv->field_count);
                        ((StructValue**)arr.as.array_val->data)[i] = copy.as.struct_val;
                        break;
                    }
                    default:
                        fprintf(stderr, "Error: Unsupported array element type\n");
                        break;
                }
            }
            
            return arr;
        }

        case AST_IF: {
            Value cond = eval_expression(expr->as.if_stmt.condition, env);
            if (cond.is_return) return cond;
            if (is_truthy(cond)) {
                return eval_statement(expr->as.if_stmt.then_branch, env);
            } else {
                return eval_statement(expr->as.if_stmt.else_branch, env);
            }
        }

        case AST_COND: {
            /* Evaluate cond expression: (cond (pred1 val1) (pred2 val2) ... (else valN)) */
            /* Check each condition in order and return the corresponding value */
            for (int i = 0; i < expr->as.cond_expr.clause_count; i++) {
                Value cond = eval_expression(expr->as.cond_expr.conditions[i], env);
                if (cond.is_return) return cond;
                if (is_truthy(cond)) {
                    return eval_expression(expr->as.cond_expr.values[i], env);
                }
            }
            /* If no condition matched, return the else value */
            return eval_expression(expr->as.cond_expr.else_value, env);
        }

        case AST_STRUCT_LITERAL: {
            /* Evaluate struct literal: Point { x: 10, y: 20 } */
            
            const char *struct_name = expr->as.struct_literal.struct_name;
            int field_count = expr->as.struct_literal.field_count;

            /* Union variant construction is currently parsed as a struct literal with a dotted name:
             *   UnionName.Variant { ... }
             * If UnionName is a known union, evaluate it as a union value.
             */
            if (struct_name) {
                const char *dot = strchr(struct_name, '.');
                if (dot) {
                    char union_name_buf[256];
                    size_t union_len = (size_t)(dot - struct_name);
                    if (union_len > 0 && union_len < sizeof(union_name_buf)) {
                        memcpy(union_name_buf, struct_name, union_len);
                        union_name_buf[union_len] = '\0';
                        const char *variant_name = dot + 1;

                        UnionDef *udef = env_get_union(env, union_name_buf);
                        if (udef) {
                            int variant_idx = env_get_union_variant_index(env, union_name_buf, variant_name);
                            if (variant_idx < 0) {
                                fprintf(stderr, "Error: Unknown variant '%s' in union '%s'\n", variant_name, union_name_buf);
                                return create_void();
                            }

                            if (field_count != udef->variant_field_counts[variant_idx]) {
                                fprintf(stderr, "Error: Variant '%s.%s' expects %d fields, got %d\n",
                                        union_name_buf, variant_name, udef->variant_field_counts[variant_idx], field_count);
                                return create_void();
                            }

                            Value *field_values = NULL;
                            if (field_count > 0) {
                                field_values = malloc(sizeof(Value) * field_count);
                                for (int i = 0; i < field_count; i++) {
                                    field_values[i] = eval_preserve_value(env, eval_expression(expr->as.struct_literal.field_values[i], env));
                                    if (field_values[i].is_return) {
                                        Value result = field_values[i];
                                        free(field_values);
                                        return result;
                                    }
                                }
                            }

                            Value result = create_union(union_name_buf, variant_idx, variant_name,
                                                      expr->as.struct_literal.field_names, field_values, field_count);

                            if (field_values) free(field_values);
                            return result;
                        }
                    }
                }
            }
            
            
            /* Handle spread syntax: {..base, extra: val}
             * Note: struct_name may be non-NULL here if the typechecker inferred the
             * target type from a 'let x: T = {..base, ...}' declaration. We must
             * handle spread regardless of whether struct_name is set. */
            if (expr->as.struct_literal.spread_source) {
                Value base_val = eval_preserve_value(env, eval_expression(expr->as.struct_literal.spread_source, env));
                if (base_val.is_return) return base_val;
                StructValue *base_sv = base_val.type == VAL_STRUCT ? base_val.as.struct_val : NULL;
                int base_count = base_sv ? base_sv->field_count : 0;
                int over_count = expr->as.struct_literal.field_count;
                int merged_cap = base_count + over_count;
                char **merged_names  = malloc(sizeof(char*) * (merged_cap + 1));
                Value *merged_values = malloc(sizeof(Value) * (merged_cap + 1));
                int merged_count = 0;
                /* Copy base fields not overridden */
                for (int bi = 0; bi < base_count; bi++) {
                    bool overridden = false;
                    for (int oi = 0; oi < over_count; oi++) {
                        if (strcmp(base_sv->field_names[bi],
                                   expr->as.struct_literal.field_names[oi]) == 0) {
                            overridden = true; break;
                        }
                    }
                    if (!overridden) {
                        merged_names[merged_count]  = base_sv->field_names[bi];
                        merged_values[merged_count] = base_sv->field_values[bi];
                        merged_count++;
                    }
                }
                /* Add override/new fields */
                for (int oi = 0; oi < over_count; oi++) {
                    merged_names[merged_count]  = expr->as.struct_literal.field_names[oi];
                    merged_values[merged_count] = eval_preserve_value(env, eval_expression(
                        expr->as.struct_literal.field_values[oi], env));
                    if (merged_values[merged_count].is_return) {
                        Value result = merged_values[merged_count];
                        free(merged_names);
                        free(merged_values);
                        return result;
                    }
                    merged_count++;
                }
                /* Prefer the declared struct_name (set by typechecker) over the
                 * base value's name, so typed spreads like 'let x: T = {..base,...}'
                 * produce a struct named T rather than whatever base's type was. */
                const char *res_name = struct_name ? struct_name
                                     : (base_sv ? base_sv->struct_name : NULL);
                NominalIdentity spread_identity = env_nominal_identity(env, res_name,
                    env->current_module, TYPE_STRUCT);
                const char *spread_name = env_nominal_name(env, spread_identity);
                Value result = create_struct(spread_name ? spread_name : res_name ? res_name : "anonymous",
                                             merged_names, merged_values, merged_count);
                free(merged_names);
                free(merged_values);
                return result;
            }

            /* Get struct definition to verify field order */
            StructDef *struct_def = env_get_struct(env, struct_name);
            if (!struct_def) {
                fprintf(stderr, "Error: Undefined struct '%s'\n", struct_name);
                return create_void();
            }
            
            
            char *canonical_name = strdup(struct_def->name);
            if (!canonical_name) { fprintf(stderr, "I cannot retain record declaration identity.\n"); exit(1); }
            /* Allocate arrays for field names and values */
            char **field_names = malloc(sizeof(char*) * field_count);
            Value *field_values = malloc(sizeof(Value) * field_count);
            
            
            /* Evaluate each field value */
            for (int i = 0; i < field_count; i++) {
                field_names[i] = expr->as.struct_literal.field_names[i];
                field_values[i] = eval_preserve_value(env, eval_expression(expr->as.struct_literal.field_values[i], env));
                if (field_values[i].is_return) {
                    Value result = field_values[i];
                    free(canonical_name);
                    free(field_names);
                    free(field_values);
                    return result;
                }
            }
            
            
            /* Create struct value */
            Value result = create_struct(canonical_name, field_names, field_values, field_count);
            
            
            /* Free temporary arrays (create_struct makes copies) */
            free(canonical_name);
            free(field_names);
            free(field_values);
            
            
            return result;
        }

        case AST_FIELD_ACCESS: {
            /* Check object is not NULL */
            if (!expr->as.field_access.object) {
                fprintf(stderr, "Error: NULL object in field access\n");
                return create_void();
            }
            
            /* Special case: Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                assert(enum_name != NULL);
                if (!enum_name) {
                    safe_fprintf(stderr, "Error: NULL enum name in field access\n");
                    return create_void();
                }
                EnumDef *enum_def = env_get_enum(env, enum_name);
                
                if (enum_def && enum_def->variant_names) {
                    /* This is an enum variant access (e.g., Color.Red) */
                    const char *variant_name = expr->as.field_access.field_name;
                    
                    assert(variant_name != NULL);
                    if (!variant_name) {
                        safe_fprintf(stderr, "Error: NULL variant name in enum access\n");
                        return create_void();
                    }
                    
                    /* Lookup variant value */
                    for (int i = 0; i < enum_def->variant_count; i++) {
                        if (safe_strcmp(enum_def->variant_names[i], variant_name) == 0) {
                            return create_int(enum_def->variant_values ? enum_def->variant_values[i] : i);
                        }
                    }
                    
                    safe_fprintf(stderr, "Error: Enum '%s' has no variant '%s'\n",
                            safe_format_string(enum_name), safe_format_string(variant_name));
                    return create_void();
                }
            }
            
            /* Regular struct field access */
            /* Evaluate field access: point.x */
            Value obj = eval_expression(expr->as.field_access.object, env);
            if (obj.is_return) return obj;
            
            if (obj.type != VAL_STRUCT) {
                fprintf(stderr, "Error: Cannot access field on non-struct value\n");
                return create_void();
            }
            
            const char *field_name = expr->as.field_access.field_name;
            StructValue *sv = obj.as.struct_val;
            
            /* Find field in struct */
            for (int i = 0; i < sv->field_count; i++) {
                if (strcmp(sv->field_names[i], field_name) == 0) {
                    /* I return an owned string, not a record's borrowed storage.
                     * A local binding releases its value when its call ends. */
                    if (sv->field_values[i].type == VAL_STRING) {
                        return create_string(sv->field_values[i].as.string_val);
                    }
                    return sv->field_values[i];
                }
            }
            
            fprintf(stderr, "Error: Struct '%s' has no field '%s'\n", 
                    sv->struct_name, field_name);
            return create_void();
        }

        case AST_UNION_CONSTRUCT: {
            /* Evaluate union construction: Status.Ok {} or Result.Error { code: 404 } */
            const char *union_name = expr->as.union_construct.union_name;
            const char *variant_name = expr->as.union_construct.variant_name;
            
            /* Get variant index */
            int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
            if (variant_idx < 0) {
                fprintf(stderr, "Error: Unknown variant '%s' for union '%s'\n", variant_name, union_name);
                return create_void();
            }
            
            /* Evaluate field values */
            int field_count = expr->as.union_construct.field_count;
            char **field_names = NULL;
            Value *field_values = NULL;
            
            if (field_count > 0) {
                field_names = malloc(sizeof(char*) * field_count);
                field_values = malloc(sizeof(Value) * field_count);
                
                for (int i = 0; i < field_count; i++) {
                    field_names[i] = expr->as.union_construct.field_names[i];
                    field_values[i] = eval_preserve_value(env, eval_expression(expr->as.union_construct.field_values[i], env));
                    if (field_values[i].is_return) {
                        Value result = field_values[i];
                        free(field_names);
                        free(field_values);
                        return result;
                    }
                }
            }
            
            Value result = create_union(union_name, variant_idx, variant_name, 
                                       field_names, field_values, field_count);
            
            /* Free temporary arrays (create_union makes copies) */
            if (field_count > 0) {
                free(field_names);
                free(field_values);
            }
            
            return result;
        }

        case AST_MATCH: {
            /* Evaluate match expression: match status { Ok(x) => 1, Error(e) => 0 }
             * Also supports integer literal patterns: match n { 0 => "zero", 1 => "one", _ => "many" }
             */
            Value match_val = eval_expression(expr->as.match_expr.expr, env);
            if (match_val.is_return || match_val.is_break || match_val.is_continue)
                return match_val;
            if (match_val.type == VAL_UNION && !match_val.as.union_val)
                return eval_match_invariant_failure("a union match received no value");

            bool owns_empty = eval_match_owns_empty_literal(
                expr->as.match_expr.expr, match_val);

            /* Every pattern, including a wildcard, participates in source order. */
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                const char *pattern_variant = expr->as.match_expr.pattern_variants[i];
                if (!eval_match_pattern(&match_val, pattern_variant)) continue;

                int saved_symbol_count = env->symbol_count;
                if (match_val.type == VAL_UNION && strcmp(pattern_variant, "_") != 0) {
                    UnionValue *union_value = match_val.as.union_val;
                    const char *binding = expr->as.match_expr.pattern_bindings[i];
                    if (binding && !*binding && union_value->field_count != 0)
                        return eval_match_invariant_failure("I require a zero-field variant for an empty match binding");
                    /* I create no local for () or underscore discard. */
                    if (binding && *binding && strcmp(binding, "_") != 0) {
                        Value binding_value;
                        if (union_value->field_count > 0) {
                            char **field_names = malloc(sizeof(char *) * (size_t)union_value->field_count);
                            Value *field_values = malloc(sizeof(Value) * (size_t)union_value->field_count);
                            if (!field_names || !field_values) {
                                free(field_names);
                                free(field_values);
                                return eval_match_invariant_failure(
                                    "I could not allocate a match payload binding");
                            }
                            for (int field = 0; field < union_value->field_count; ++field) {
                                field_names[field] = union_value->field_names[field];
                                field_values[field] = union_value->field_values[field];
                            }
                            binding_value = create_struct(
                                union_value->union_name, field_names, field_values,
                                union_value->field_count);
                        } else {
                            binding_value = create_void();
                        }
                        env_define_var(env, binding, TYPE_STRUCT, false, binding_value);
                    }
                }

                ASTNode *guard = expr->as.match_expr.guard_exprs
                    ? expr->as.match_expr.guard_exprs[i] : NULL;
                if (guard) {
                    Value guard_value = eval_expression(guard, env);
                    if (guard_value.is_return || guard_value.is_break || guard_value.is_continue) {
                        eval_match_pop_metadata(env, saved_symbol_count);
                        eval_match_release_empty_literal(match_val, owns_empty, guard_value);
                        return guard_value;
                    }
                    if (guard_value.type != VAL_BOOL) {
                        eval_match_pop_metadata(env, saved_symbol_count);
                        eval_match_release_empty_literal(match_val, owns_empty, create_void());
                        return eval_match_invariant_failure(
                            "a checked match guard did not produce bool");
                    }
                    if (!guard_value.as.bool_val) {
                        eval_match_pop_metadata(env, saved_symbol_count);
                        continue;
                    }
                }

                Value result = eval_expression(expr->as.match_expr.arm_bodies[i], env);
                eval_match_pop_metadata(env, saved_symbol_count);
                eval_match_release_empty_literal(match_val, owns_empty, result);
                return result;
            }

            eval_match_release_empty_literal(match_val, owns_empty, create_void());
            return eval_match_invariant_failure(
                "a checked match reached no successful arm");
        }

        case AST_BLOCK: {
            /* Blocks can be used as expressions in match arms
             * I yield the final expression and preserve function-scoped control flow.
             */
            return eval_scoped_block(expr->as.block.statements, expr->as.block.count, env);
        }

        case AST_RETURN: {
            /* Return statements can appear in blocks that are used as expressions */
            Value result;
            if (expr->as.return_stmt.value) {
                result = eval_expression(expr->as.return_stmt.value, env);
            } else {
                result = create_void();
            }
            if (result.is_return) return result;
            result.is_return = true;
            result.return_target = g_eval_return_target;
            return result;
        }

        case AST_TUPLE_LITERAL: {
            /* Evaluate tuple literal: (1, "hello", true) */
            int element_count = expr->as.tuple_literal.element_count;
            
            /* Empty tuple */
            if (element_count == 0) {
                return create_tuple(NULL, 0);
            }
            
            /* Evaluate each element */
            Value *elements = malloc(sizeof(Value) * element_count);
            for (int i = 0; i < element_count; i++) {
                elements[i] = eval_preserve_value(env, eval_expression(expr->as.tuple_literal.elements[i], env));
                if (elements[i].is_return) {
                    Value result = elements[i];
                    free(elements);
                    return result;
                }
            }
            
            /* Create tuple value */
            Value result = create_tuple(elements, element_count);
            free(elements);  /* create_tuple makes a copy */
            
            return result;
        }

        case AST_TUPLE_INDEX: {
            /* Evaluate tuple index access: tuple.0, tuple.1 */
            Value tuple = eval_expression(expr->as.tuple_index.tuple, env);
            if (tuple.is_return) return tuple;
            
            if (tuple.type != VAL_TUPLE) {
                fprintf(stderr, "Error: Tuple index access on non-tuple value (type %d)\n", tuple.type);
                return create_void();
            }
            
            int index = expr->as.tuple_index.index;
            TupleValue *tv = tuple.as.tuple_val;
            
            if (index < 0 || index >= tv->element_count) {
                fprintf(stderr, "Error: Tuple index %d out of bounds (tuple has %d elements)\n",
                        index, tv->element_count);
                return create_void();
            }
            
            Value item = tv->elements[index];
            if (item.type == VAL_STRING) return create_string(item.as.string_val);
            return eval_preserve_value(env, item);
        }

        case AST_TRY_OP: {
            /* Desugar expr? in the interpreter:
             * evaluate operand; if Err variant, propagate as return; else return Ok's first field. */
            Value inner = eval_expression(expr->as.try_op.operand, env);
            if (inner.is_return) return inner;
            if (inner.type != VAL_UNION || !inner.as.union_val) {
                fprintf(stderr, "Error at line %d, column %d: '?' operator requires a union value\n",
                        expr->line, expr->column);
                return create_void();
            }
            UnionValue *uv = inner.as.union_val;
            if (strcmp(uv->variant_name, "Err") == 0) {
                /* Propagate the Err as a return value */
                inner.is_return = true;
                inner.return_target = g_eval_return_target;
                inner.is_break = false;
                inner.is_continue = false;
                return inner;
            }
            /* Ok variant: return the first field value */
            if (uv->field_count == 0) {
                fprintf(stderr, "Error at line %d, column %d: '?' operator: Ok variant has no fields\n",
                        expr->line, expr->column);
                return create_void();
            }
            return uv->field_values[0];
        }
        case AST_AWAIT: {
            /*
             * await expr — evaluate the inner expression.
             * If it yields a VAL_COROUTINE (from spawn), run the scheduler
             * until that coroutine completes and return its result.
             * Otherwise fall through (synchronous transparent await).
             */
            Value inner = eval_expression(expr->as.await_expr.expr, env);
            if (inner.is_return) return inner;
            if (inner.type == VAL_COROUTINE) {
                int coro_id = (int)inner.as.int_val;
                return eval_task_result(env, coro_id, true);
            }
            return inner;
        }

        case AST_EFFECT_DECL:
            /* Effect declarations are registered at program-level; no runtime work. */
            return create_void();

        case AST_HANDLE_EXPR: {
            int count = expr->as.handle_expr.handler_count;
            if (count <= 0 || !expr->as.handle_expr.effect_name) {
                fprintf(stderr, "I require a resolved effect and nonempty handler clauses.\n");
                return create_void();
            }
            EffectHandlerFrame frame = {0};
            frame.effect_name = expr->as.handle_expr.effect_name;
            frame.handler_op_names = expr->as.handle_expr.handler_op_names;
            frame.handler_param_groups = expr->as.handle_expr.handler_param_names;
            frame.handler_param_counts = expr->as.handle_expr.handler_param_counts;
            frame.handler_bodies = expr->as.handle_expr.handler_bodies;
            frame.handler_count = count;
            frame.env = env;
            frame.return_target = g_eval_return_target;
            nl_effect_frame_push(&frame);
            Value result = eval_expression(expr->as.handle_expr.body, env);
            nl_effect_frame_pop();
            return result;
        }

        case AST_EFFECT_HANDLER: {
            /* handle <body> with { Effect.op(param) -> handler_body }
             * Push a handler frame, evaluate body, pop the frame, return body result. */
            EffectHandlerFrame frame;
            memset(&frame, 0, sizeof(frame));
            frame.effect_name         = expr->as.effect_handler.effect_name;
            frame.handler_op_names    = expr->as.effect_handler.handler_op_names;
            frame.handler_param_names = expr->as.effect_handler.handler_param_names;
            frame.handler_bodies      = expr->as.effect_handler.handler_bodies;
            frame.handler_count       = expr->as.effect_handler.handler_count;
            frame.env                 = env;
            frame.return_target       = g_eval_return_target;

            nl_effect_frame_push(&frame);
            Value result = eval_expression(expr->as.effect_handler.body, env);
            nl_effect_frame_pop();
            return result;
        }

        case AST_EFFECT_OP: {
            /* perform Effect.op(arg) — dispatch to the nearest matching handler. */
            int arm_idx = -1;
            EffectHandlerFrame *frame = nl_effect_find_handler(
                expr->as.effect_op.effect_name,
                expr->as.effect_op.op_name,
                &arm_idx);

            if (!frame || arm_idx < 0) {
                fprintf(stderr,
                    "Error at line %d: unhandled effect %s.%s\n",
                    expr->line,
                    expr->as.effect_op.effect_name ? expr->as.effect_op.effect_name : "?",
                    expr->as.effect_op.op_name     ? expr->as.effect_op.op_name     : "?");
                return create_void();
            }

            const char *legacy_param = frame->handler_param_names
                ? frame->handler_param_names[arm_idx] : NULL;
            int count = frame->handler_param_counts ? frame->handler_param_counts[arm_idx]
                : (legacy_param && legacy_param[0] ? 1 : 0);
            if (count != expr->as.effect_op.arg_count) {
                fprintf(stderr, "I require matching perform and handler argument counts.\n");
                return create_void();
            }
            Value *args = count ? calloc((size_t)count, sizeof(*args)) : NULL;
            if (count && !args) return create_void();
            /* I evaluate in the caller before handler names can shadow arguments. */
            for (int i = 0; i < count; i++) {
                args[i] = eval_expression(expr->as.effect_op.args[i], env);
                if (args[i].is_return) {
                    Value result = args[i];
                    free(args);
                    return result;
                }
            }
            Environment *henv = frame->env;
            int saved_sym = henv->symbol_count;
            for (int i = 0; i < count; i++) {
                const char *param = frame->handler_param_groups
                    ? frame->handler_param_groups[arm_idx][i] : legacy_param;
                env_define_var(henv, param, TYPE_UNKNOWN, false, args[i]);
            }
            free(args);

            const void *saved_return_target = g_eval_return_target;
            g_eval_return_target = frame->return_target;
            Value handler_result = eval_statement(frame->handler_bodies[arm_idx], henv);
            g_eval_return_target = saved_return_target;

            /* Restore scope. */
            henv->symbol_count = saved_sym;

            /* I preserve lexical returns; ordinary final values resume perform. */
            handler_result.is_break    = false;
            handler_result.is_continue = false;
            return handler_result;
        }

        default:
            return create_void();
    }
}

/* An identifier read borrows its environment value. A new binding owns its
 * value, so function identifiers need the same explicit copy that call
 * parameters and returned function values already receive. */
static Value own_function_identifier(ASTNode *expression, Environment *env,
                                     Value value) {
    if (value.type != VAL_FUNCTION || !expression ||
        expression->type != AST_IDENTIFIER) return value;
    Symbol *source = env_get_var(env, expression->as.identifier);
    if (!source || source->value.type != VAL_FUNCTION ||
        source->value.as.function_val.function_name !=
            value.as.function_val.function_name) return value;
    return create_function(value.as.function_val.function_name,
        copy_function_signature(value.as.function_val.signature));
}

/* Evaluate statement */
static Value eval_statement(ASTNode *stmt, Environment *env) {
    if (!stmt) return create_void();

    /* DAP breakpoint/step hook — fires before each statement when debugging */
    if (g_dap_statement_hook) g_dap_statement_hook(stmt, env);

    switch (stmt->type) {
        case AST_LET: {
            /* For HashMap<K,V>, ensure (map_new) has concrete type context during interpretation. */
            if (stmt->as.let.var_type == TYPE_HASHMAP &&
                stmt->as.let.type_info &&
                stmt->as.let.value &&
                stmt->as.let.value->type == AST_CALL &&
                stmt->as.let.value->as.call.name &&
                strcmp(stmt->as.let.value->as.call.name, "map_new") == 0 &&
                stmt->as.let.value->as.call.return_struct_type_name == NULL) {
                TypeInfo *info = stmt->as.let.type_info;
                if (info->generic_name && strcmp(info->generic_name, "HashMap") == 0 && info->type_param_count == 2) {
                    const char *k = eval_hm_typeinfo_arg_name(info->type_params[0]);
                    const char *v = eval_hm_typeinfo_arg_name(info->type_params[1]);
                    if (k && v) {
                        char mono[128];
                        snprintf(mono, sizeof(mono), "HashMap_%s_%s", k, v);
                        stmt->as.let.value->as.call.return_struct_type_name = strdup(mono);
                    }
                }
            }

            Value value = eval_expression(stmt->as.let.value, env);
            /* If the RHS propagates a return (e.g. via the ? operator), forward it. */
            if (value.is_return || value.is_break || value.is_continue) {
                return value;
            }
            value = own_function_identifier(stmt->as.let.value, env, value);
            env_define_var_with_type_info(env,
                                         stmt->as.let.name,
                                         stmt->as.let.var_type,
                                         stmt->as.let.element_type,
                                         stmt->as.let.type_info,
                                         stmt->as.let.is_mut,
                                         value);
            
            /* Trace variable declaration */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_decl(stmt->as.let.name, stmt->as.let.var_type, value, 
                          stmt->as.let.is_mut, stmt->line, stmt->column, scope);
#else
            trace_var_decl(stmt->as.let.name, stmt->as.let.var_type, value, 
                          stmt->as.let.is_mut, stmt->line, stmt->column, NULL);
#endif
            
            return create_void();
        }

        case AST_SET: {
            Value value = eval_expression(stmt->as.set.value, env);
            if (value.is_return) return value;
            if (stmt->as.set.field_name) {
                Symbol *owner = env_get_var(env, stmt->as.set.name);
                if (owner && owner->value.type == VAL_STRUCT) {
                    StructValue *record = owner->value.as.struct_val;
                    for (int i = 0; i < record->field_count; ++i) {
                        if (!strcmp(record->field_names[i], stmt->as.set.field_name)) {
                            Value staged = value;
                            if (value.type == VAL_STRUCT || value.type == VAL_TUPLE) {
                                if (!env_clone_value_snapshot(value, &staged)) {
                                    fprintf(stderr, "I cannot copy a replacement record field.\n"); exit(1);
                                }
                            } else if (value.type == VAL_STRING) {
                                staged.as.string_val = strdup(value.as.string_val ? value.as.string_val : "");
                                if (!staged.as.string_val) {
                                    fprintf(stderr, "I cannot copy a replacement string field.\n"); exit(1);
                                }
                            }
                            Value previous = record->field_values[i];
                            record->field_values[i] = staged;
                            if (previous.type == VAL_STRUCT || previous.type == VAL_TUPLE) env_discard_value_snapshot(previous);
                            else if (previous.type == VAL_STRING) {
                                if (gc_is_managed(previous.as.string_val)) gc_release(previous.as.string_val);
                                else free(previous.as.string_val);
                            }
                            return create_void();
                        }
                    }
                }
                fprintf(stderr, "I cannot resolve a borrowed field during evaluation\n");
                return create_void();
            }
            value = own_function_identifier(stmt->as.set.value, env, value);
            env_set_var(env, stmt->as.set.name, value);
            
            /* Trace variable assignment */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_set(stmt->as.set.name, old_value, value, 
                         stmt->line, stmt->column, scope);
#else
            trace_var_set(stmt->as.set.name, old_value, value, 
                         stmt->line, stmt->column, NULL);
#endif
            
            return create_void();
        }

        case AST_WHILE: {
            Value result = create_void();
            for (;;) {
                Value condition = eval_expression(stmt->as.while_stmt.condition, env);
                if (condition.is_return) return condition;
                if (!is_truthy(condition)) break;
                result = eval_statement(stmt->as.while_stmt.body, env);
                /* If body returned a value, propagate it immediately */
                if (result.is_return) {
                    return result;
                }
                if (result.is_break) {
                    result = create_void();
                    break;
                }
                if (result.is_continue) {
                    result = create_void();
                    continue;
                }
            }
            return result;
        }

        case AST_FOR: {
            ASTNode *range_expr = stmt->as.for_stmt.range_expr;
            const char *loop_var = stmt->as.for_stmt.var_name;

            /* Check if it's a range(start, end) call */
            if (range_expr->type == AST_CALL && range_expr->as.call.name &&
                strcmp(range_expr->as.call.name, "range") == 0 &&
                range_expr->as.call.arg_count == 2) {

                Value start_val = eval_expression(range_expr->as.call.args[0], env);
                if (start_val.is_return) return start_val;
                Value end_val = eval_expression(range_expr->as.call.args[1], env);
                if (end_val.is_return) return end_val;

                if (start_val.type != VAL_INT || end_val.type != VAL_INT) {
                    fprintf(stderr, "Error: range requires int arguments\n");
                    return create_void();
                }

                long long start = start_val.as.int_val;
                long long end = end_val.as.int_val;

                int loop_var_index = env->symbol_count;
                env_define_var(env, loop_var, TYPE_INT, false, create_int(start));

                Value result = create_void();
                for (long long i = start; i < end; i++) {
                    env->symbols[loop_var_index].value = create_int(i);
                    result = eval_statement(stmt->as.for_stmt.body, env);
                    if (result.is_return) {
                        env->symbol_count = loop_var_index;
                        return result;
                    }
                    if (result.is_break) { result = create_void(); break; }
                    if (result.is_continue) { result = create_void(); continue; }
                }
                env->symbol_count = loop_var_index;
                return result;

            } else {
                /* List iteration: look up list type from symbol table */
                Symbol *iterable_sym = NULL;
                if (range_expr->type == AST_IDENTIFIER) {
                    iterable_sym = env_get_var(env, range_expr->as.identifier);
                }

                Type list_type = iterable_sym ? iterable_sym->type : TYPE_UNKNOWN;
                Value iter_val = eval_expression(range_expr, env);
                if (iter_val.is_return) return iter_val;
                NominalIdentity element;
                if (env_record_list_identity(env, iter_val, &element)) {
                    Value length, list_arg[] = {iter_val};
                    if (!env_record_list_apply(env, element, "length", list_arg, 1, &length)) {
                        fprintf(stderr, "I cannot read a record-list iteration length.\n"); exit(1);
                    }
                    int first = env->symbol_count;
                    env_define_var(env, loop_var, TYPE_STRUCT, true, create_void());
                    env->symbols[first].struct_type_name = strdup(env_nominal_name(env, element));
                    if (!env->symbols[first].struct_type_name) {
                        fprintf(stderr, "I cannot retain loop element identity.\n"); exit(1);
                    }
                    env->symbols[first].nominal_owner = env_nominal_owner(env, element);
                    Value result = create_void();
                    for (long long index = 0; index < length.as.int_val; ++index) {
                        Value args[] = {iter_val, create_int(index)}, item;
                        if (!env_record_list_apply(env, element, "get", args, 2, &item)) {
                            fprintf(stderr, "I cannot read this record-list iteration index.\n"); exit(1);
                        }
                        /* env_set_var clones; my loop binding never owns an arena result. */
                        env_set_var(env, loop_var, item);
                        result = eval_statement(stmt->as.for_stmt.body, env);
                        if (result.is_return || result.is_break) break;
                        if (result.is_continue) result = create_void();
                    }
                    result = eval_preserve_value(env, result);
                    if (result.is_break) result = create_void();
                    eval_scope_release(env, first, false);
                    return result;
                }
                if (list_type == TYPE_LIST_GENERIC) {
                    fprintf(stderr, "I require a live record-list handle for iteration.\n"); exit(1);
                }

                int loop_var_index = env->symbol_count;
                env_define_var(env, loop_var, TYPE_INT, true, create_void());

                Value result = create_void();

                if (iter_val.type == VAL_ARRAY) {
                    /* Static array iteration */
                    Array *arr = iter_val.as.array_val;
                    if (!arr) { env->symbol_count = loop_var_index; return create_void(); }
                    for (int idx = 0; idx < arr->length; idx++) {
                        Value elem = create_void();
                        switch (arr->element_type) {
                            case VAL_INT:    elem = create_int(((long long*)arr->data)[idx]); break;
                            case VAL_FLOAT:  elem = create_float(((double*)arr->data)[idx]); break;
                            case VAL_BOOL:   elem = create_bool(((bool*)arr->data)[idx]); break;
                            case VAL_STRING: elem = create_string(((char**)arr->data)[idx]); break;
                            default: break;
                        }
                        env->symbols[loop_var_index].value = elem;
                        result = eval_statement(stmt->as.for_stmt.body, env);
                        if (result.is_return) { env->symbol_count = loop_var_index; return result; }
                        if (result.is_break) { result = create_void(); break; }
                        if (result.is_continue) { result = create_void(); continue; }
                    }
                } else if (iter_val.type == VAL_DYN_ARRAY) {
                    /* Dynamic array iteration */
                    DynArray *arr = iter_val.as.dyn_array_val;
                    if (!arr) { env->symbol_count = loop_var_index; return create_void(); }
                    int64_t len = dyn_array_length(arr);
                    ElementType et = dyn_array_get_elem_type(arr);
                    for (int64_t idx = 0; idx < len; idx++) {
                        Value elem = create_void();
                        switch (et) {
                            case ELEM_INT:    elem = create_int(dyn_array_get_int(arr, idx)); break;
                            case ELEM_FLOAT:  elem = create_float(dyn_array_get_float(arr, idx)); break;
                            case ELEM_BOOL:   elem = create_bool(dyn_array_get_bool(arr, idx)); break;
                            case ELEM_STRING: elem = create_string(dyn_array_get_string(arr, idx)); break;
                            default: break;
                        }
                        env->symbols[loop_var_index].value = elem;
                        result = eval_statement(stmt->as.for_stmt.body, env);
                        if (result.is_return) { env->symbol_count = loop_var_index; return result; }
                        if (result.is_break) { result = create_void(); break; }
                        if (result.is_continue) { result = create_void(); continue; }
                    }
                } else if (list_type == TYPE_LIST_INT) {
                    List_int *lst = (List_int*)(intptr_t)iter_val.as.int_val;
                    if (!lst) { env->symbol_count = loop_var_index; return create_void(); }
                    for (int idx = 0; idx < lst->length; idx++) {
                        env->symbols[loop_var_index].value = create_int(lst->data[idx]);
                        result = eval_statement(stmt->as.for_stmt.body, env);
                        if (result.is_return) { env->symbol_count = loop_var_index; return result; }
                        if (result.is_break) { result = create_void(); break; }
                        if (result.is_continue) { result = create_void(); continue; }
                    }
                } else if (list_type == TYPE_LIST_STRING) {
                    List_string *lst = (List_string*)(intptr_t)iter_val.as.int_val;
                    if (!lst) { env->symbol_count = loop_var_index; return create_void(); }
                    for (int idx = 0; idx < lst->length; idx++) {
                        env->symbols[loop_var_index].value = create_string(lst->data[idx]);
                        result = eval_statement(stmt->as.for_stmt.body, env);
                        if (result.is_return) { env->symbol_count = loop_var_index; return result; }
                        if (result.is_break) { result = create_void(); break; }
                        if (result.is_continue) { result = create_void(); continue; }
                    }
                } else {
                    fprintf(stderr, "Error: for-in requires a list, array, or range expression\n");
                }

                env->symbol_count = loop_var_index;
                return result;
            }
        }

        case AST_RETURN: {
            Value result;
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            } else {
                result = create_void();
            }
            if (result.is_return) return result;
            result.is_return = true;  /* Mark as return value */
            result.return_target = g_eval_return_target;
            result.is_break = false;
            result.is_continue = false;
            return result;
        }

        case AST_BLOCK: {
            return eval_scoped_block(stmt->as.block.statements, stmt->as.block.count, env);
        }

        case AST_BREAK: {
            Value v = create_void();
            v.is_break = true;
            return v;
        }

        case AST_CONTINUE: {
            Value v = create_void();
            v.is_continue = true;
            return v;
        }

        case AST_PRINT: {
            Value value = eval_expression(stmt->as.print.expr, env);
            if (value.is_return) return value;
            print_value(value);
            if (stmt->as.print.is_println) printf("\n");
            return create_void();
        }

        case AST_ASSERT: {
            Value cond = eval_expression(stmt->as.assert.condition, env);
            if (cond.is_return) return cond;
            if (!is_truthy(cond)) {
                if (g_in_shadow_tests) {
                    g_shadow_current_fail_count++;
                    if (g_shadow_current_first_line == 0) {
                        g_shadow_current_first_line = stmt->line;
                        g_shadow_current_first_column = stmt->column;
                    }
                    return create_void();
                }

                fprintf(stderr, "Assertion failed at line %d, column %d\n", stmt->line, stmt->column);
                exit(1);
            }
            return create_void();
        }

        case AST_PAR_BLOCK: {
            Value par_result = create_void();
            int *order = stmt->as.par_block.is_flow ? passive_binding_order(stmt) : NULL;
            if (stmt->as.par_block.is_flow && !order) {
                fprintf(stderr, "I cannot establish a valid flow execution order.\n");
                exit(1);
            }
            for (int i = 0; i < stmt->as.par_block.count; i++) {
                par_result = eval_statement(stmt->as.par_block.bindings[order ? order[i] : i], env);
                if (par_result.is_return || par_result.is_break || par_result.is_continue) break;
            }
            free(order);
            return par_result;
        }

        case AST_PAR_LET: {
            /* Evaluate all bindings (sequentially; parallelism is the runtime's job) */
            for (int i = 0; i < stmt->as.par_let.count; i++) {
                Value v = eval_expression(stmt->as.par_let.values[i], env);
                if (v.is_return || v.is_break || v.is_continue) return v;
                env_define_var_with_type_info(env,
                    stmt->as.par_let.names[i],
                    TYPE_UNKNOWN, TYPE_UNKNOWN, NULL, false, v);
            }
            /* Evaluate body with all bindings in scope */
            return eval_expression(stmt->as.par_let.body, env);
        }

        case AST_UNSAFE_BLOCK: {
            /* Unsafe blocks are treated like regular blocks in the interpreter */
            return eval_scoped_block(stmt->as.unsafe_block.statements, stmt->as.unsafe_block.count, env);
        }

        case AST_STRUCT_DEF:
            /* Struct definitions are handled at program level (typechecker) */
            return create_void();
        
        case AST_ENUM_DEF: {
            /* Register enum in interpreter environment for enum variant access */
            if (!stmt->as.enum_def.name) {
                fprintf(stderr, "Error: Enum definition has NULL name\n");
                return create_void();
            }
            
            EnumDef edef;
            edef.name = strdup(stmt->as.enum_def.name);
            edef.variant_count = stmt->as.enum_def.variant_count;
            
            if (edef.variant_count <= 0) {
                fprintf(stderr, "Error: Enum '%s' has invalid variant count: %d\n", edef.name, edef.variant_count);
                free(edef.name);
                return create_void();
            }
            
            /* Duplicate variant names */
            edef.variant_names = malloc(sizeof(char*) * edef.variant_count);
            if (!edef.variant_names) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant names\n");
                free(edef.name);
                return create_void();
            }
            
            for (int j = 0; j < edef.variant_count; j++) {
                if (stmt->as.enum_def.variant_names && stmt->as.enum_def.variant_names[j]) {
                    edef.variant_names[j] = strdup(stmt->as.enum_def.variant_names[j]);
                    if (!edef.variant_names[j]) {
                        fprintf(stderr, "Error: Failed to duplicate variant name at index %d\n", j);
                        edef.variant_names[j] = NULL;
                    }
                } else {
                    fprintf(stderr, "Warning: Enum '%s' has NULL variant name at index %d\n", edef.name, j);
                    edef.variant_names[j] = NULL;
                }
            }
            
            /* Duplicate variant values */
            edef.variant_values = malloc(sizeof(int) * edef.variant_count);
            if (!edef.variant_values) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant values\n");
                free(edef.name);
                for (int j = 0; j < edef.variant_count; j++) {
                    free(edef.variant_names[j]);
                }
                free(edef.variant_names);
                return create_void();
            }
            
            if (stmt->as.enum_def.variant_values) {
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = stmt->as.enum_def.variant_values[j];
                }
            } else {
                /* No explicit values - use index as value */
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = j;
                }
            }
            
            env_define_enum(env, edef);
            return create_void();
        }
        
        case AST_UNION_DEF: {
            /* Register union in interpreter environment for union construction */
            if (!stmt->as.union_def.name) {
                fprintf(stderr, "Error: Union definition has NULL name\n");
                return create_void();
            }
            
            /* Typechecking may already have registered this declaration. */
            if (env_get_union(env, stmt->as.union_def.name)) return create_void();
            UnionDef udef = {0};
            udef.name = strdup(stmt->as.union_def.name);
            udef.variant_count = stmt->as.union_def.variant_count;
            
            /* Copy generic parameters */
            udef.generic_param_count = stmt->as.union_def.generic_param_count;
            if (udef.generic_param_count > 0) {
                udef.generic_params = malloc(sizeof(char*) * udef.generic_param_count);
                for (int j = 0; j < udef.generic_param_count; j++) {
                    udef.generic_params[j] = strdup(stmt->as.union_def.generic_params[j]);
                }
            } else {
                udef.generic_params = NULL;
            }
            udef.is_pub = stmt->as.union_def.is_pub;
            udef.module_name = NULL;
            
            if (udef.variant_count <= 0) {
                fprintf(stderr, "Error: Union '%s' has invalid variant count: %d\n", udef.name, udef.variant_count);
                free(udef.name);
                return create_void();
            }
            
            /* Duplicate variant names */
            udef.variant_names = malloc(sizeof(char*) * udef.variant_count);
            udef.variant_field_counts = malloc(sizeof(int) * udef.variant_count);
            udef.variant_field_names = malloc(sizeof(char**) * udef.variant_count);
            udef.variant_field_types = malloc(sizeof(Type*) * udef.variant_count);
            
            for (int j = 0; j < udef.variant_count; j++) {
                udef.variant_names[j] = strdup(stmt->as.union_def.variant_names[j]);
                udef.variant_field_counts[j] = stmt->as.union_def.variant_field_counts[j];
                
                /* Duplicate field names and types for this variant */
                int field_count = udef.variant_field_counts[j];
                if (field_count > 0) {
                    udef.variant_field_names[j] = malloc(sizeof(char*) * field_count);
                    udef.variant_field_types[j] = malloc(sizeof(Type) * field_count);
                    
                    for (int k = 0; k < field_count; k++) {
                        udef.variant_field_names[j][k] = strdup(stmt->as.union_def.variant_field_names[j][k]);
                        udef.variant_field_types[j][k] = stmt->as.union_def.variant_field_types[j][k];
                    }
                } else {
                    udef.variant_field_names[j] = NULL;
                    udef.variant_field_types[j] = NULL;
                }
            }
            
            udef.variant_field_type_info = calloc((size_t)udef.variant_count, sizeof(TypeInfo **));
            for (int j = 0; j < udef.variant_count; ++j) {
                int fields = udef.variant_field_counts[j];
                udef.variant_field_type_info[j] = calloc((size_t)fields, sizeof(TypeInfo *));
                for (int k = 0; k < fields; ++k)
                    if (stmt->as.union_def.variant_field_type_info && stmt->as.union_def.variant_field_type_info[j])
                        udef.variant_field_type_info[j][k] = copy_payload_type_info(stmt->as.union_def.variant_field_type_info[j][k]);
            }
            env_define_union(env, udef);
            return create_void();
        }
        
        case AST_SERVICE_DECL:
            return eval_match_invariant_failure("I have not resolved File service declarations for this consumer");
        case AST_FUNCTION:
        case AST_SHADOW:
            /* Function and shadow definitions are handled at program level */
            return create_void();

        case AST_ASYNC_FN: {
            /* async fn — register the inner function, then mark it as async
             * so the coroutine scheduler can route calls through it. */
            if (stmt->as.async_fn.function) {
                Value r = eval_statement(stmt->as.async_fn.function, env);
                /* Mark the function as async in the environment */
                ASTNode *fn_node = stmt->as.async_fn.function;
                if (fn_node && fn_node->type == AST_FUNCTION && fn_node->as.function.name) {
                    Function *fn = env_get_function(env, fn_node->as.function.name);
                    if (fn) fn->is_async = true;
                }
                return r;
            }
            return create_void();
        }

        case AST_EFFECT_DECL:
            /* Effect declarations are registered at program-level; no runtime work. */
            return create_void();

        case AST_EFFECT_HANDLER:
        case AST_EFFECT_OP:
            /* Delegate to eval_expression — these produce values. */
            return eval_expression(stmt, env);

        default:
            /* Expression statements */
            return eval_expression(stmt, env);
    }
}


/* Run shadow tests */
bool run_shadow_tests(ASTNode *program, Environment *env, bool verbose) {
    return run_shadow_tests_scope(program, env, NULL, env_current_file(env), false, verbose);
}

bool run_shadow_tests_scope(ASTNode *program, Environment *env, ModuleList *modules,
                            const char *input_file, bool include_imports, bool verbose) {
    if (ast_has_service_declaration(program)) { fprintf(stderr, "I have not resolved File service declarations for this consumer.\n"); return false; }
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program for shadow tests\n");
        return false;
    }

    if (verbose) {
        fprintf(stdout, "Running shadow tests...\n");
    }

    bool all_passed = true;
    g_in_shadow_tests = true;

    ShadowFailure *failures = NULL;
    int failure_count = 0;
    int failure_cap = 0;
    int test_count = 0;
    const char *shadow_json_path = getenv("NANO_LLM_SHADOW_JSON");
    ASTNode *root_program = program;
    char *root_owner = env->current_module;
    const char *root_file = env_current_file(env);
    int imported_count = include_imports && modules ? modules->count : 0;

    nl_shadow_timing("interpreter_start", -1, -1, 0, 0, 0);
    for (int source = 0; source <= imported_count; source++) {
        bool imported = source < imported_count;
        const char *file = imported ? modules->module_paths[source] : input_file;
        program = imported ? get_cached_module_ast(file) : root_program;
        char *owner = imported ? module_program_name(program, file) : NULL;
        if (!program || (imported && !owner)) {
            fprintf(stderr, "I cannot load a selected shadow module: %s\n", file ? file : "");
            free(owner);
            all_passed = false;
            break;
        }
        env->current_module = imported ? owner : root_owner;
        env_set_current_file(env, file);

        nl_shadow_timing("module_init_start", source, -1, test_count, 0, 0);
        /* First pass: Evaluate top-level constants */
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
        
            if (item->type == AST_LET) {
                eval_statement(item, env);  /* Evaluate the constant */
            }
        }

        /* Second pass: Register all enum definitions so they're available in shadow tests */
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
        
            if (item->type == AST_ENUM_DEF) {
                eval_statement(item, env);  /* This will register the enum */
            }
        }

        /* Third pass: Register all union definitions so they're available in shadow tests */
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
        
            if (item->type == AST_UNION_DEF) {
                eval_statement(item, env);  /* This will register the union */
            }
        }

        nl_shadow_timing("module_init_end", source, -1, test_count, 0, 0);
        /* Fourth pass: Run each shadow test */
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
        
            if (item->type == AST_SHADOW) {
                const char *func_name = item->as.shadow.function_name;
                /* I execute explicit shadows; foreign syntax does not exempt them. */
            
                test_count++;
                if (verbose) {
                    fprintf(stdout, "Testing %s... ", func_name);
                }
            
                /* Execute shadow test */
                g_shadow_current_fail_count = 0;
                g_shadow_current_first_line = 0;
                g_shadow_current_first_column = 0;

                /* When not verbose, suppress stdout from test body execution */
                int saved_stdout_fd = -1;
                if (!verbose) {
                    fflush(stdout);
                    saved_stdout_fd = dup(STDOUT_FILENO);
                    int devnull = open("/dev/null", O_WRONLY);
                    if (devnull >= 0) {
                        dup2(devnull, STDOUT_FILENO);
                        close(devnull);
                    }
                }

                nl_shadow_timing("shadow_start", source, i, test_count, 0, 0);
                eval_statement(item->as.shadow.body, env);
                nl_shadow_timing("shadow_end", source, i, test_count, 0, g_shadow_current_fail_count);

                if (!verbose && saved_stdout_fd >= 0) {
                    fflush(stdout);
                    dup2(saved_stdout_fd, STDOUT_FILENO);
                    close(saved_stdout_fd);
                }

                if (g_shadow_current_fail_count > 0) {
                    all_passed = false;
                    if (verbose) {
                        fprintf(stdout, "FAILED\n");
                    }
                    fprintf(stdout, "  Shadow test '%s' FAILED: %d failure(s)\n", func_name, g_shadow_current_fail_count);
                    if (g_shadow_current_first_line > 0) {
                        fprintf(stdout, "  First failure at line %d, column %d\n", g_shadow_current_first_line, g_shadow_current_first_column);
                    }

                    if (failure_count >= failure_cap) {
                        int new_cap = failure_cap == 0 ? 8 : failure_cap * 2;
                        ShadowFailure *new_arr = realloc(failures, sizeof(ShadowFailure) * (size_t)new_cap);
                        if (new_arr) {
                            failures = new_arr;
                            failure_cap = new_cap;
                        }
                    }
                    if (failure_count < failure_cap) {
                        failures[failure_count].test_name = func_name;
                        failures[failure_count].source_file = file;
                        failures[failure_count].fail_count = g_shadow_current_fail_count;
                        failures[failure_count].first_line = g_shadow_current_first_line;
                        failures[failure_count].first_column = g_shadow_current_first_column;
                        failure_count++;
                    }
                } else {
                    if (verbose) {
                        fprintf(stdout, "PASSED\n");
                    }
                }
            }
            /* Note: We do NOT execute non-shadow items here - they're already registered
             * in the environment by the type checker. Only shadow test bodies need execution. */
        }
        env->current_module = root_owner;
        free(owner);
    }
    env_set_current_file(env, root_file);

    if (all_passed) {
        if (verbose) {
            fprintf(stdout, "All shadow tests passed! (%d tests", test_count);
            fprintf(stdout, ")\n");
        }
    }

    if (!shadow_write_json_file(shadow_json_path, failures, failure_count, all_passed, test_count)) {
        fprintf(stderr, "I cannot write the completed shadow report.\n");
        all_passed = false;
    }
    free(failures);
    g_in_shadow_tests = false;

    nl_shadow_timing("interpreter_end", -1, -1, test_count, 0, all_passed ? 0 : 1);
    return all_passed;
}

/* Run the entire program (interpreter mode) */
bool run_program(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program\n");
        return false;
    }

    /* First pass: evaluate top-level constants before functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* Evaluate top-level constants */
        if (item->type == AST_LET) {
            eval_statement(item, env);
        }
    }

    /* Second pass: execute all other top-level items (functions, statements, etc.) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* Skip constants - already processed */
        if (item->type == AST_LET) {
            continue;
        }

        /* Skip shadow tests in interpreter mode - they're for compiler validation */
        if (item->type == AST_SHADOW) {
            continue;
        }

        /* Skip imports - they're handled separately before execution */
        if (item->type == AST_IMPORT) {
            continue;
        }

        /* Execute the item */
        eval_statement(item, env);
    }

    return true;
}

/* Call a function by name with arguments */
static Value call_function_at(const char *name, Value *args, int arg_count,
                             Environment *env, int line, int column) {
    Value record_list_result;
    if (eval_record_list_call(name, args, arg_count, env, &record_list_result))
        return record_list_result;

    Function *func = env_get_function(env, name);
    if (!func) {
        fprintf(stderr, "Error: Function '%s' not found\n", name);
        return create_void();
    }

    if (func->is_extern && func->body == NULL) {
        return eval_foreign_call(func, args, arg_count, env, line, column);
    }

    /* Check argument count */
    if (arg_count != func->param_count) {
        fprintf(stderr, "Error: Function '%s' expects %d arguments, got %d\n",
                name, func->param_count, arg_count);
        return create_void();
    }

    /* Save original symbol count to restore environment after function call */
    int original_symbol_count = env->symbol_count;

    /* Add function parameters to environment with copies of string values */
    for (int i = 0; i < arg_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        if (param_value.type == VAL_FUNCTION) {
            param_value = create_function(param_value.as.function_val.function_name,
                copy_function_signature(param_value.as.function_val.signature));
        }
        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute the function body */
    char return_boundary;
    const void *saved_return_target = g_eval_return_target;
    g_eval_return_target = &return_boundary;
    char *saved_module_context = env->current_module;
    env->current_module = func->module_name;
    Value result = eval_statement(func->body, env);
    g_eval_return_target = saved_return_target;
    env->current_module = saved_module_context;

    /* I copy records recursively before dropping parameters, including returns
     * addressed to an enclosing handler. My Environment owns this snapshot. */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    } else if (result.type == VAL_STRUCT || result.type == VAL_TUPLE) {
        if (!env_value_snapshot(env, result, &return_value)) {
            fprintf(stderr, "I cannot copy a returned record.\n");
            exit(1);
        }
    } else if (result.type == VAL_FUNCTION) {
        return_value = create_function(result.as.function_val.function_name,
            copy_function_signature(result.as.function_val.signature));
    }

    /* I consume only this activation's return, after preserving its value. */
    return_value.is_return = result.is_return && result.return_target &&
        result.return_target != &return_boundary;
    return_value.return_target = return_value.is_return ? result.return_target : NULL;
    return_value.is_break = false;
    return_value.is_continue = false;

    eval_scope_release(env, original_symbol_count, false);

    /* An outer handler's return has not reached its destination yet. */
    if (!return_value.is_return)
        return_value = eval_checked_scalar_destination(func->return_type, return_value);
    return return_value;
}

Value call_function(const char *name, Value *args, int arg_count, Environment *env) {
    /* A builtin callback inherits its invoking call. Host calls have no location. */
    Value result = call_function_at(name, args, arg_count, env,
                            g_eval_call_site ? g_eval_call_site->line : 0,
                            g_eval_call_site ? g_eval_call_site->column : 0);
    /* Only this public boundary publishes an independently owned record. Both
     * internal call paths and direct generated operations use my result arena. */
    if (env_record_result_borrowed(env, result)) {
        Value copy;
        if (!env_clone_value_snapshot(result, &copy)) { fprintf(stderr, "I cannot copy an escaping record.\n"); exit(1); }
        copy.is_return = result.is_return;
        copy.return_target = result.return_target;
        result = copy;
    }
    return result;
}

/* ============================================================================
 * REPL support: public wrappers around static functions
 * ========================================================================== */

Value repl_eval_node(ASTNode *node, Environment *env) {
    return eval_statement(node, env);
}

void repl_print_value(Value val) {
    print_value(val);
}
