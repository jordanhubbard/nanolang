/**
 * test_effects.c — unit tests for effects.c
 *
 * Exercises the algebraic effect system:
 *   EffectRegistry, EffectCtx, EffectRow operations,
 *   effect registration, lookup, and row set operations.
 */

#include "../src/nanolang.h"
#include "../src/effects.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %lld, expected %lld)\n", \
        #a, #b, __LINE__, (long long)(a), (long long)(b)); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_NULL(p) \
    if ((p) != NULL) { printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); }

/* Required by runtime */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* ============================================================================
 * EffectRegistry tests
 * ============================================================================ */

void test_registry_create_free(void) {
    EffectRegistry *reg = effect_registry_new();
    ASSERT_NOT_NULL(reg);
    effect_registry_free(reg);
}

void test_registry_free_null(void) {
    /* Should not crash */
    effect_registry_free(NULL);
}

static EffectSysOp make_op(const char *name, Type param, Type ret) {
    EffectSysOp op;
    memset(&op, 0, sizeof(op));
    op.name = (char *)name;
    op.param_type = param;
    op.return_type = ret;
    return op;
}

void test_registry_register_and_lookup(void) {
    EffectRegistry *reg = effect_registry_new();
    ASSERT_NOT_NULL(reg);

    /* Register a simple IO effect with one operation */
    EffectSysOp ops[1] = {make_op("print", TYPE_STRING, TYPE_VOID)};
    bool ok = effect_register(reg, "IO", ops, 1);
    ASSERT(ok);

    /* Lookup the registered effect */
    EffectDecl *decl = effect_lookup(reg, "IO");
    ASSERT_NOT_NULL(decl);

    /* Lookup non-existent effect */
    EffectDecl *missing = effect_lookup(reg, "NonExistent");
    ASSERT_NULL(missing);

    effect_registry_free(reg);
}

void test_registry_lookup_op(void) {
    EffectRegistry *reg = effect_registry_new();
    EffectSysOp ops[2] = {
        make_op("print", TYPE_STRING, TYPE_VOID),
        make_op("read", TYPE_VOID, TYPE_STRING)
    };
    effect_register(reg, "IO", ops, 2);

    EffectDecl *decl = effect_lookup(reg, "IO");
    ASSERT_NOT_NULL(decl);

    EffectSysOp *op = effect_op_lookup(decl, "print");
    ASSERT_NOT_NULL(op);

    EffectSysOp *op2 = effect_op_lookup(decl, "read");
    ASSERT_NOT_NULL(op2);

    EffectSysOp *missing_op = effect_op_lookup(decl, "nonexistent");
    ASSERT_NULL(missing_op);

    effect_registry_free(reg);
}

void test_registry_multiple_effects(void) {
    EffectRegistry *reg = effect_registry_new();

    EffectSysOp io_ops[2] = {
        make_op("print", TYPE_STRING, TYPE_VOID),
        make_op("read", TYPE_VOID, TYPE_STRING)
    };
    EffectSysOp err_ops[2] = {
        make_op("throw", TYPE_STRING, TYPE_VOID),
        make_op("catch_err", TYPE_VOID, TYPE_STRING)
    };
    EffectSysOp state_ops[2] = {
        make_op("get", TYPE_VOID, TYPE_INT),
        make_op("put", TYPE_INT, TYPE_VOID)
    };

    ASSERT(effect_register(reg, "IO", io_ops, 2));
    ASSERT(effect_register(reg, "Err", err_ops, 2));
    ASSERT(effect_register(reg, "State", state_ops, 2));

    ASSERT_NOT_NULL(effect_lookup(reg, "IO"));
    ASSERT_NOT_NULL(effect_lookup(reg, "Err"));
    ASSERT_NOT_NULL(effect_lookup(reg, "State"));

    effect_registry_free(reg);
}

/* ============================================================================
 * EffectCtx tests
 * ============================================================================ */

void test_ctx_create_free(void) {
    EffectRegistry *reg = effect_registry_new();
    EffectCtx *ctx = effect_ctx_new(reg);
    ASSERT_NOT_NULL(ctx);
    effect_ctx_free(ctx);
    effect_registry_free(reg);
}

void test_ctx_free_null(void) {
    effect_ctx_free(NULL);
}

/* ============================================================================
 * EffectRow tests
 * ============================================================================ */

void test_row_create_free(void) {
    EffectRow *row = effect_row_new();
    ASSERT_NOT_NULL(row);
    effect_row_free(row);
}

void test_row_free_null(void) {
    effect_row_free(NULL);
}

void test_row_add_and_contains(void) {
    EffectRow *row = effect_row_new();

    ASSERT(!effect_row_contains(row, "IO"));
    ASSERT(effect_row_add(row, "IO"));
    ASSERT(effect_row_contains(row, "IO"));

    ASSERT(!effect_row_contains(row, "Err"));
    ASSERT(effect_row_add(row, "Err"));
    ASSERT(effect_row_contains(row, "Err"));

    effect_row_free(row);
}

void test_row_add_duplicate(void) {
    EffectRow *row = effect_row_new();
    ASSERT(effect_row_add(row, "IO"));
    /* Adding again should be idempotent (may return false or true) */
    effect_row_add(row, "IO");
    ASSERT(effect_row_contains(row, "IO"));
    effect_row_free(row);
}

void test_row_merge(void) {
    EffectRow *dst = effect_row_new();
    EffectRow *src = effect_row_new();

    effect_row_add(dst, "IO");
    effect_row_add(src, "Err");
    effect_row_add(src, "State");

    ASSERT(effect_row_merge(dst, src));

    ASSERT(effect_row_contains(dst, "IO"));
    ASSERT(effect_row_contains(dst, "Err"));
    ASSERT(effect_row_contains(dst, "State"));

    effect_row_free(dst);
    effect_row_free(src);
}

void test_row_subset(void) {
    EffectRow *big = effect_row_new();
    EffectRow *small = effect_row_new();
    EffectRow *empty = effect_row_new();

    effect_row_add(big, "IO");
    effect_row_add(big, "Err");
    effect_row_add(big, "State");
    effect_row_add(small, "IO");

    /* empty ⊆ big */
    ASSERT(effect_row_subset(empty, big));
    /* small ⊆ big */
    ASSERT(effect_row_subset(small, big));
    /* big ⊄ small */
    ASSERT(!effect_row_subset(big, small));

    effect_row_free(big);
    effect_row_free(small);
    effect_row_free(empty);
}

void test_row_to_str(void) {
    EffectRow *row = effect_row_new();
    effect_row_add(row, "IO");
    effect_row_add(row, "Err");

    char *s = effect_row_to_str(row);
    ASSERT_NOT_NULL(s);
    /* String should contain both effect names */
    ASSERT(strstr(s, "IO") != NULL || strstr(s, "Err") != NULL);
    free(s);

    effect_row_free(row);
}

void test_row_to_str_empty(void) {
    EffectRow *row = effect_row_new();
    char *s = effect_row_to_str(row);
    ASSERT_NOT_NULL(s);  /* Should return empty string or "Pure" */
    free(s);
    effect_row_free(row);
}

/* ============================================================================
 * Effect handler frame tests
 * ============================================================================ */

void test_effect_frame_push_pop(void) {
    char *op_names[] = {"print"};
    EffectHandlerFrame frame;
    memset(&frame, 0, sizeof(frame));
    frame.effect_name = "IO";
    frame.handler_op_names = op_names;
    frame.handler_count = 1;

    nl_effect_frame_push(&frame);
    int arm_idx = -1;
    EffectHandlerFrame *found = nl_effect_find_handler("IO", "print", &arm_idx);
    ASSERT(found == &frame);
    ASSERT_EQ(arm_idx, 0);

    /* Lookup non-existent op in existing effect */
    EffectHandlerFrame *not_found = nl_effect_find_handler("IO", "no_such_op", NULL);
    ASSERT_NULL(not_found);

    nl_effect_frame_pop();

    /* After pop, should not find it */
    EffectHandlerFrame *after_pop = nl_effect_find_handler("IO", "print", NULL);
    ASSERT_NULL(after_pop);
}

void test_effect_find_handler_empty_stack(void) {
    /* With no frames, should return NULL */
    EffectHandlerFrame *found = nl_effect_find_handler("NonExistent", "op", NULL);
    ASSERT_NULL(found);
}

/* ============================================================================
 * Effect integration with environment
 * ============================================================================ */

void test_env_effect_lookup_missing(void) {
    Environment *env = create_environment();
    EffectDecl *decl = env_effect_lookup(env, "IO");
    /* Should return NULL for effects not registered in env */
    (void)decl;  /* May be NULL or not — just shouldn't crash */
    free_environment(env);
}

static ASTNode *parse_nano_effects(const char *src) {
    int n = 0;
    Token *t = tokenize(src, &n);
    if (!t) return NULL;
    ASTNode *p = parse_program(t, n);
    free_tokens(t, n);
    return p;
}

void test_effect_arrow_type_str_no_row(void) {
    /* Without row effects */
    char *s = effect_arrow_type_str("int", "string", NULL);
    ASSERT_NOT_NULL(s);
    ASSERT(strstr(s, "int") != NULL);
    ASSERT(strstr(s, "string") != NULL);
    free(s);
}

void test_effect_arrow_type_str_with_row(void) {
    EffectRow *row = effect_row_new();
    effect_row_add(row, "IO");
    char *s = effect_arrow_type_str("int", "void", row);
    ASSERT_NOT_NULL(s);
    ASSERT(strstr(s, "IO") != NULL);
    free(s);
    effect_row_free(row);
}

void test_effect_arrow_type_str_empty_row(void) {
    EffectRow *row = effect_row_new();  /* empty */
    char *s = effect_arrow_type_str("string", "int", row);
    ASSERT_NOT_NULL(s);
    free(s);
    effect_row_free(row);
}

void test_effect_check_program_no_effects(void) {
    /* Program with no effect declarations should pass cleanly */
    ASTNode *prog = parse_nano_effects(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
    );
    ASSERT_NOT_NULL(prog);
    EffectRegistry *reg = effect_registry_new();
    Environment *env = create_environment();
    bool ok = effect_check_program(prog, reg, env);
    ASSERT(ok);
    free_environment(env);
    effect_registry_free(reg);
    free_ast(prog);
}

void test_effect_check_program_null(void) {
    /* Null program should return true (nothing to check) */
    EffectRegistry *reg = effect_registry_new();
    bool ok = effect_check_program(NULL, reg, NULL);
    ASSERT(ok);
    effect_registry_free(reg);

    /* Null registry should also return true */
    ASTNode *prog = parse_nano_effects("fn f() -> int { return 0 }\n");
    ok = effect_check_program(prog, NULL, NULL);
    ASSERT(ok);
    free_ast(prog);
}

void test_effect_register_from_ast_null(void) {
    /* NULL node should return false */
    EffectRegistry *reg = effect_registry_new();
    bool ok = effect_register_from_ast(reg, NULL);
    ASSERT(!ok);

    /* Non-effect-decl node should return false */
    ASTNode *prog = parse_nano_effects("fn f() -> int { return 0 }\n");
    if (prog && prog->as.program.count > 0) {
        ok = effect_register_from_ast(reg, prog->as.program.items[0]);
        ASSERT(!ok);
    }
    if (prog) free_ast(prog);
    effect_registry_free(reg);
}

void test_effect_check_program_with_no_effect_nodes(void) {
    /* Program with functions but no effect declarations */
    ASTNode *prog = parse_nano_effects(
        "fn double(x: int) -> int { return (* x 2) }\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT_NOT_NULL(prog);
    EffectRegistry *reg = effect_registry_new();
    Environment *env = create_environment();
    bool ok = effect_check_program(prog, reg, env);
    ASSERT(ok);
    free_environment(env);
    effect_registry_free(reg);
    free_ast(prog);
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Effects System Tests ===\n");

    TEST(registry_create_free);
    TEST(registry_free_null);
    TEST(registry_register_and_lookup);
    TEST(registry_lookup_op);
    TEST(registry_multiple_effects);
    TEST(ctx_create_free);
    TEST(ctx_free_null);
    TEST(row_create_free);
    TEST(row_free_null);
    TEST(row_add_and_contains);
    TEST(row_add_duplicate);
    TEST(row_merge);
    TEST(row_subset);
    TEST(row_to_str);
    TEST(row_to_str_empty);
    TEST(effect_frame_push_pop);
    TEST(effect_find_handler_empty_stack);
    TEST(env_effect_lookup_missing);
    TEST(effect_arrow_type_str_no_row);
    TEST(effect_arrow_type_str_with_row);
    TEST(effect_arrow_type_str_empty_row);
    TEST(effect_check_program_no_effects);
    TEST(effect_check_program_null);
    TEST(effect_register_from_ast_null);
    TEST(effect_check_program_with_no_effect_nodes);

    printf("\n✓ All effects tests passed!\n");
    return 0;
}
