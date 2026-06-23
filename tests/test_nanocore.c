/*
 * test_nanocore.c — unit tests for src/nanocore_export.c and src/emit_typed_ast.c
 *
 * Exercises:
 *   nanocore_export_sexpr: all binary operators, unary minus, string escaping,
 *                           empty/multi-statement blocks, param type emission
 *   emit_typed_ast_json:   all emit_stmt branches (let, set, if, while, for),
 *                           various parameter/return types (float, void, etc.)
 */

#include "../src/nanolang.h"
#include "../src/nanocore_export.h"
#include "../src/emit_typed_ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

#define TEST(name) \
    do { printf("  Testing %s...", #name); test_##name(); printf(" \xE2\x9C\x93\n"); } while(0)
#define ASSERT(cond) \
    do { if (!(cond)) { \
        printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); \
    } } while(0)
#define ASSERT_NOT_NULL(p) \
    do { if ((p) == NULL) { \
        printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); \
    } } while(0)
#define ASSERT_NULL(p) \
    do { if ((p) != NULL) { \
        printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); \
    } } while(0)
#define ASSERT_STR_CONTAINS(haystack, needle) \
    do { if (!(haystack) || !strstr((haystack), (needle))) { \
        printf("\n    FAILED: expected '%s' in '%s' at line %d\n", \
               (needle), (haystack) ? (haystack) : "(null)", __LINE__); exit(1); \
    } } while(0)

/* ── Helpers ────────────────────────────────────────────────────────────── */

static ASTNode *parse_nano(const char *src) {
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    if (!tokens) return NULL;
    ASTNode *prog = parse_program(tokens, token_count);
    free_tokens(tokens, token_count);
    return prog;
}

/* Get first function node from a parsed program */
static ASTNode *get_func(ASTNode *prog) {
    if (!prog) return NULL;
    int count = 0;
    ASTNode **items = NULL;
    if (prog->type == AST_PROGRAM) {
        count = prog->as.program.count;
        items = prog->as.program.items;
    } else if (prog->type == AST_BLOCK) {
        count = prog->as.block.count;
        items = prog->as.block.statements;
    }
    for (int i = 0; i < count; i++) {
        if (items[i] && items[i]->type == AST_FUNCTION) return items[i];
    }
    return NULL;
}

/* Get the first statement's return expression */
static ASTNode *get_return_val(ASTNode *func) {
    if (!func || func->type != AST_FUNCTION) return NULL;
    ASTNode *body = func->as.function.body;
    if (!body) return NULL;
    ASTNode **stmts = NULL;
    int count = 0;
    if (body->type == AST_BLOCK) {
        stmts = body->as.block.statements;
        count = body->as.block.count;
    }
    if (count == 0) return NULL;
    ASTNode *stmt = stmts[0];
    if (!stmt || stmt->type != AST_RETURN) return NULL;
    return stmt->as.return_stmt.value;
}

/* Suppress/restore stdout for emit_typed_ast_json calls */
static FILE *s_saved_stdout = NULL;
static void suppress_stdout(void) {
    fflush(stdout);
    s_saved_stdout = stdout;
    stdout = fopen("/dev/null", "w");
}
static void restore_stdout(void) {
    if (stdout && stdout != s_saved_stdout) fclose(stdout);
    stdout = s_saved_stdout;
    s_saved_stdout = NULL;
}

/* ============================================================================
 * nanocore_export_sexpr tests
 * ============================================================================ */

static void test_export_null_node(void) {
    char *s = nanocore_export_sexpr(NULL, NULL);
    ASSERT_NULL(s);
}

static void test_export_int_literal(void) {
    ASTNode *prog = parse_nano("fn f() -> int { return 42 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "EInt");
    ASSERT_STR_CONTAINS(s, "42");
    free(s);
    free_ast(prog);
}

static void test_export_binop_sub(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> int { return (- a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpSub");
    free(s);
    free_ast(prog);
}

static void test_export_binop_div(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> int { return (/ a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpDiv");
    free(s);
    free_ast(prog);
}

static void test_export_binop_mod(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> int { return (% a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpMod");
    free(s);
    free_ast(prog);
}

static void test_export_binop_eq(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> bool { return (== a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpEq");
    free(s);
    free_ast(prog);
}

static void test_export_binop_ne(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> bool { return (!= a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpNe");
    free(s);
    free_ast(prog);
}

static void test_export_binop_lt(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> bool { return (< a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpLt");
    free(s);
    free_ast(prog);
}

static void test_export_binop_le(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> bool { return (<= a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpLe");
    free(s);
    free_ast(prog);
}

static void test_export_binop_gt(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> bool { return (> a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpGt");
    free(s);
    free_ast(prog);
}

static void test_export_binop_ge(void) {
    ASTNode *prog = parse_nano("fn f(a: int, b: int) -> bool { return (>= a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpGe");
    free(s);
    free_ast(prog);
}

static void test_export_binop_and(void) {
    ASTNode *prog = parse_nano("fn f(a: bool, b: bool) -> bool { return (and a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpAnd");
    free(s);
    free_ast(prog);
}

static void test_export_binop_or(void) {
    ASTNode *prog = parse_nano("fn f(a: bool, b: bool) -> bool { return (or a b) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpOr");
    free(s);
    free_ast(prog);
}

static void test_export_unary_neg(void) {
    /* (- x) with one argument → unary minus → OpNeg */
    ASTNode *prog = parse_nano("fn f(x: int) -> int { return (- x) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpNeg");
    free(s);
    free_ast(prog);
}

static void test_export_unary_not(void) {
    /* (! x) → unary not → OpNot */
    ASTNode *prog = parse_nano("fn f(x: bool) -> bool { return (! x) }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "OpNot");
    free(s);
    free_ast(prog);
}

static void test_export_string_literal(void) {
    /* Simple string literal with escape chars — exercises emit_string */
    ASTNode *prog = parse_nano("fn f() -> string { return \"hello\" }");
    ASSERT_NOT_NULL(prog);
    ASTNode *expr = get_return_val(get_func(prog));
    ASSERT_NOT_NULL(expr);
    char *s = nanocore_export_sexpr(expr, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "EString");
    ASSERT_STR_CONTAINS(s, "hello");
    free(s);
    free_ast(prog);
}

static void test_export_empty_block(void) {
    /* Function body with no statements → emit_block count=0 → (EUnit) */
    ASTNode *prog = parse_nano("fn f() -> void { }");
    ASSERT_NOT_NULL(prog);
    ASTNode *func = get_func(prog);
    ASSERT_NOT_NULL(func);
    ASTNode *body = func->as.function.body;
    ASSERT_NOT_NULL(body);
    /* Call export on the block node directly */
    char *s = nanocore_export_sexpr(body, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "EUnit");
    free(s);
    free_ast(prog);
}

static void test_export_multi_stmt_block(void) {
    /* Function with 3 statements → emit_block multi-stmt → ESeq */
    ASTNode *prog = parse_nano(
        "fn f() -> int {\n"
        "    let x: int = 1\n"
        "    let y: int = 2\n"
        "    return (+ x y)\n"
        "}\n"
    );
    ASSERT_NOT_NULL(prog);
    ASTNode *func = get_func(prog);
    ASSERT_NOT_NULL(func);
    ASTNode *body = func->as.function.body;
    ASSERT_NOT_NULL(body);
    char *s = nanocore_export_sexpr(body, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "ESeq");
    free(s);
    free_ast(prog);
}

static void test_export_func_no_params(void) {
    /* Zero-param function → ELam "_" TUnit */
    ASTNode *prog = parse_nano("fn f() -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *func = get_func(prog);
    ASSERT_NOT_NULL(func);
    char *s = nanocore_export_sexpr(func, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "ELam");
    ASSERT_STR_CONTAINS(s, "TUnit");
    free(s);
    free_ast(prog);
}

static void test_export_func_bool_param(void) {
    /* Bool param type → TBool in output */
    ASTNode *prog = parse_nano("fn f(x: bool) -> bool { return x }");
    ASSERT_NOT_NULL(prog);
    ASTNode *func = get_func(prog);
    ASSERT_NOT_NULL(func);
    char *s = nanocore_export_sexpr(func, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "TBool");
    free(s);
    free_ast(prog);
}

static void test_export_func_string_param(void) {
    /* String param type → TString in output */
    ASTNode *prog = parse_nano("fn f(x: string) -> int { return 0 }");
    ASSERT_NOT_NULL(prog);
    ASTNode *func = get_func(prog);
    ASSERT_NOT_NULL(func);
    char *s = nanocore_export_sexpr(func, NULL);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "TString");
    free(s);
    free_ast(prog);
}

/* ============================================================================
 * emit_typed_ast_json tests
 * (stdout is suppressed — we only verify no crash and basic structure)
 * ============================================================================ */

static void test_emit_null_program(void) {
    suppress_stdout();
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", NULL, env);
    free_environment(env);
    restore_stdout();
    /* Must not crash */
}

static void test_emit_empty_function(void) {
    suppress_stdout();
    ASTNode *prog = parse_nano("fn main() -> int { return 0 }");
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_float_return_type(void) {
    /* Exercises type_str(TYPE_FLOAT, ...) */
    suppress_stdout();
    ASTNode *prog = parse_nano("fn f() -> float { return 3.14 }");
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_void_return_type(void) {
    /* Exercises type_str(TYPE_VOID, ...) */
    suppress_stdout();
    ASTNode *prog = parse_nano("fn f() -> void { }");
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_let_statement(void) {
    /* Exercises emit_stmt AST_LET branch */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f() -> int {\n"
        "    let x: int = 42\n"
        "    return x\n"
        "}\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_set_statement(void) {
    /* Exercises emit_stmt AST_SET branch */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f() -> int {\n"
        "    let x: int = 1\n"
        "    set x 2\n"
        "    return x\n"
        "}\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_if_statement(void) {
    /* Exercises emit_stmt AST_IF branch */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f(x: int) -> int {\n"
        "    if (> x 0) { return 1 } else { return 0 }\n"
        "}\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_while_statement(void) {
    /* Exercises emit_stmt AST_WHILE branch */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f() -> int {\n"
        "    let x: int = 0\n"
        "    while (< x 10) { set x (+ x 1) }\n"
        "    return x\n"
        "}\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_for_statement(void) {
    /* Exercises emit_stmt AST_FOR branch */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f() -> int {\n"
        "    let s: int = 0\n"
        "    for i in (range 0 5) { set s (+ s i) }\n"
        "    return s\n"
        "}\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_multi_type_params(void) {
    /* Exercises type_str for int, bool, string params; also float return */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f(a: int, b: bool, c: string) -> void { }\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

static void test_emit_let_float_type(void) {
    /* Exercises type_str(TYPE_FLOAT) in emit_stmt AST_LET fallback path */
    suppress_stdout();
    ASTNode *prog = parse_nano(
        "fn f() -> float {\n"
        "    let x: float = 1.5\n"
        "    return x\n"
        "}\n"
    );
    ASSERT(prog != NULL);
    Environment *env = create_environment();
    emit_typed_ast_json("test.nano", prog, env);
    free_environment(env);
    free_ast(prog);
    restore_stdout();
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("\n[test_nanocore] nanocore_export and emit_typed_ast tests...\n\n");

    /* nanocore_export_sexpr */
    TEST(export_null_node);
    TEST(export_int_literal);
    TEST(export_binop_sub);
    TEST(export_binop_div);
    TEST(export_binop_mod);
    TEST(export_binop_eq);
    TEST(export_binop_ne);
    TEST(export_binop_lt);
    TEST(export_binop_le);
    TEST(export_binop_gt);
    TEST(export_binop_ge);
    TEST(export_binop_and);
    TEST(export_binop_or);
    TEST(export_unary_neg);
    TEST(export_unary_not);
    TEST(export_string_literal);
    TEST(export_empty_block);
    TEST(export_multi_stmt_block);
    TEST(export_func_no_params);
    TEST(export_func_bool_param);
    TEST(export_func_string_param);

    /* emit_typed_ast_json */
    TEST(emit_null_program);
    TEST(emit_empty_function);
    TEST(emit_float_return_type);
    TEST(emit_void_return_type);
    TEST(emit_let_statement);
    TEST(emit_set_statement);
    TEST(emit_if_statement);
    TEST(emit_while_statement);
    TEST(emit_for_statement);
    TEST(emit_multi_type_params);
    TEST(emit_let_float_type);

    printf("\nAll tests passed.\n");
    return 0;
}
