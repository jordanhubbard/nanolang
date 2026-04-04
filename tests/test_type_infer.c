/**
 * test_type_infer.c — unit tests for type_infer.c (Hindley-Milner engine)
 *
 * Exercises the HM API: context management, type construction, unification,
 * substitution, record types, and whole-program inference.
 */

#include "../src/nanolang.h"
#include "../src/type_infer.h"
#include "../src/effects.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %d, expected %d)\n", \
        #a, #b, __LINE__, (int)(a), (int)(b)); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_NULL(p) \
    if ((p) != NULL) { printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_STR_EQ(a, b) \
    if (strcmp((a), (b)) != 0) { printf("\n    FAILED: %s == %s at line %d\n    got: \"%s\"\n    expected: \"%s\"\n", \
        #a, #b, __LINE__, (a), (b)); exit(1); }
#define ASSERT_STR_CONTAINS(s, sub) \
    if (strstr((s), (sub)) == NULL) { printf("\n    FAILED: expected \"%s\" in \"%s\" at line %d\n", \
        (sub), (s), __LINE__); exit(1); }

/* Required by runtime/cli.c (extern in eval.c) */
int g_argc = 0;
char **g_argv = NULL;

/* Suppress stderr for expected-error tests */
static FILE *s_orig_stderr = NULL;
static void suppress_stderr(void) {
    fflush(stderr);
    s_orig_stderr = stderr;
    stderr = fopen("/dev/null", "w");
}
static void restore_stderr(void) {
    if (stderr && stderr != s_orig_stderr) fclose(stderr);
    stderr = s_orig_stderr;
    s_orig_stderr = NULL;
}

/* Global error count — reset between tests */

/* ============================================================================
 * Context management
 * ============================================================================ */

void test_ctx_create_and_free(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    ASSERT_NOT_NULL(ctx);
    ASSERT(!ctx->has_error);
    ASSERT_EQ(ctx->next_var_id, 1);  /* starts at 1 by convention */
    hm_ctx_free(ctx);
}

void test_ctx_null_source_file(void) {
    InferCtx *ctx = hm_ctx_new(NULL);
    ASSERT_NOT_NULL(ctx);
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Type variable creation
 * ============================================================================ */

void test_tv_fresh_unique_ids(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t1 = hm_tv_fresh(ctx);
    HMType *t2 = hm_tv_fresh(ctx);
    HMType *t3 = hm_tv_fresh(ctx);
    ASSERT_NOT_NULL(t1);
    ASSERT_NOT_NULL(t2);
    ASSERT_NOT_NULL(t3);
    ASSERT_EQ(t1->kind, HM_VAR);
    ASSERT_EQ(t2->kind, HM_VAR);
    ASSERT_EQ(t3->kind, HM_VAR);
    ASSERT(t1->as.var_id != t2->as.var_id);
    ASSERT(t2->as.var_id != t3->as.var_id);
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Concrete type construction
 * ============================================================================ */

void test_con_type_int(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t = hm_con_type(ctx, "int");
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(t->kind, HM_CON);
    ASSERT_STR_EQ(t->as.con_name, "int");
    hm_ctx_free(ctx);
}

void test_con_type_all_primitives(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    const char *prims[] = {"int", "float", "bool", "string", "void"};
    for (int i = 0; i < 5; i++) {
        HMType *t = hm_con_type(ctx, prims[i]);
        ASSERT_NOT_NULL(t);
        ASSERT_EQ(t->kind, HM_CON);
        ASSERT_STR_EQ(t->as.con_name, prims[i]);
    }
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Arrow type construction
 * ============================================================================ */

void test_arrow_type_basic(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *param = hm_con_type(ctx, "int");
    HMType *ret   = hm_con_type(ctx, "bool");
    HMType *arrow = hm_arrow_type(ctx, param, ret);
    ASSERT_NOT_NULL(arrow);
    ASSERT_EQ(arrow->kind, HM_ARROW);
    ASSERT(arrow->as.arrow.param == param);
    ASSERT(arrow->as.arrow.ret   == ret);
    ASSERT_NULL(arrow->as.arrow.row);   /* pure — no effect row */
    hm_ctx_free(ctx);
}

void test_arrow_type_with_effects(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *param = hm_con_type(ctx, "string");
    HMType *ret   = hm_con_type(ctx, "void");
    EffectRow *row = effect_row_new();
    effect_row_add(row, "IO");
    HMType *arrow = hm_arrow_type_with_effects(ctx, param, ret, row);
    ASSERT_NOT_NULL(arrow);
    ASSERT_EQ(arrow->kind, HM_ARROW);
    ASSERT_NOT_NULL(arrow->as.arrow.row);
    effect_row_free(row);
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Unification
 * ============================================================================ */

void test_unify_same_con(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t1 = hm_con_type(ctx, "int");
    HMType *t2 = hm_con_type(ctx, "int");
    bool ok = hm_unify(ctx, t1, t2, 1, 1);
    ASSERT(ok);
    ASSERT(!ctx->has_error);
    hm_ctx_free(ctx);
}

void test_unify_var_with_con(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *alpha = hm_tv_fresh(ctx);
    HMType *t_int = hm_con_type(ctx, "int");
    bool ok = hm_unify(ctx, alpha, t_int, 1, 1);
    ASSERT(ok);
    ASSERT(!ctx->has_error);
    HMType *resolved = hm_subst_apply(ctx, alpha);
    ASSERT_NOT_NULL(resolved);
    ASSERT_EQ(resolved->kind, HM_CON);
    ASSERT_STR_EQ(resolved->as.con_name, "int");
    hm_ctx_free(ctx);
}

void test_unify_con_with_var(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t_str = hm_con_type(ctx, "string");
    HMType *alpha = hm_tv_fresh(ctx);
    bool ok = hm_unify(ctx, t_str, alpha, 1, 1);
    ASSERT(ok);
    HMType *resolved = hm_subst_apply(ctx, alpha);
    ASSERT_EQ(resolved->kind, HM_CON);
    ASSERT_STR_EQ(resolved->as.con_name, "string");
    hm_ctx_free(ctx);
}

void test_unify_var_with_var(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *alpha = hm_tv_fresh(ctx);
    HMType *beta  = hm_tv_fresh(ctx);
    bool ok = hm_unify(ctx, alpha, beta, 1, 1);
    ASSERT(ok);
    /* After unifying two vars, one should resolve to the other */
    HMType *ra = hm_subst_apply(ctx, alpha);
    HMType *rb = hm_subst_apply(ctx, beta);
    ASSERT(ra->as.var_id == rb->as.var_id);
    hm_ctx_free(ctx);
}

void test_unify_mismatch_errors(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t_int  = hm_con_type(ctx, "int");
    HMType *t_bool = hm_con_type(ctx, "bool");
    suppress_stderr();
    bool ok = hm_unify(ctx, t_int, t_bool, 5, 3);
    restore_stderr();
    ASSERT(!ok);
    ASSERT(ctx->has_error);
    hm_ctx_free(ctx);

}

void test_unify_arrow_arrow(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *int_t  = hm_con_type(ctx, "int");
    HMType *bool_t = hm_con_type(ctx, "bool");
    HMType *a1 = hm_arrow_type(ctx, int_t, bool_t);
    HMType *a2 = hm_arrow_type(ctx, int_t, bool_t);
    bool ok = hm_unify(ctx, a1, a2, 1, 1);
    ASSERT(ok);
    ASSERT(!ctx->has_error);
    hm_ctx_free(ctx);
}

void test_unify_arrow_param_mismatch(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *int_t  = hm_con_type(ctx, "int");
    HMType *bool_t = hm_con_type(ctx, "bool");
    HMType *str_t  = hm_con_type(ctx, "string");
    HMType *a1 = hm_arrow_type(ctx, int_t,  bool_t);
    HMType *a2 = hm_arrow_type(ctx, str_t, bool_t);
    suppress_stderr();
    bool ok = hm_unify(ctx, a1, a2, 1, 1);
    restore_stderr();
    ASSERT(!ok);
    hm_ctx_free(ctx);

}

/* ============================================================================
 * Substitution apply
 * ============================================================================ */

void test_subst_apply_unbound_var(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *alpha = hm_tv_fresh(ctx);
    /* Unbound var should apply to itself */
    HMType *result = hm_subst_apply(ctx, alpha);
    ASSERT_NOT_NULL(result);
    ASSERT_EQ(result->kind, HM_VAR);
    ASSERT_EQ(result->as.var_id, alpha->as.var_id);
    hm_ctx_free(ctx);
}

void test_subst_apply_con_type(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t_int = hm_con_type(ctx, "int");
    HMType *result = hm_subst_apply(ctx, t_int);
    ASSERT_EQ(result->kind, HM_CON);
    ASSERT_STR_EQ(result->as.con_name, "int");
    hm_ctx_free(ctx);
}

void test_subst_apply_after_unify(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *alpha = hm_tv_fresh(ctx);
    HMType *t_float = hm_con_type(ctx, "float");
    hm_unify(ctx, alpha, t_float, 1, 1);
    HMType *result = hm_subst_apply(ctx, alpha);
    ASSERT_EQ(result->kind, HM_CON);
    ASSERT_STR_EQ(result->as.con_name, "float");
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Type to string
 * ============================================================================ */

void test_type_to_str_con(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *t = hm_con_type(ctx, "int");
    char *s = hm_type_to_str(ctx, t);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_EQ(s, "int");
    hm_ctx_free(ctx);
}

void test_type_to_str_var(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *alpha = hm_tv_fresh(ctx);
    char *s = hm_type_to_str(ctx, alpha);
    ASSERT_NOT_NULL(s);
    /* Should be something like "α0" */
    ASSERT(strlen(s) > 0);
    hm_ctx_free(ctx);
}

void test_type_to_str_arrow(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *param = hm_con_type(ctx, "int");
    HMType *ret   = hm_con_type(ctx, "bool");
    HMType *arrow = hm_arrow_type(ctx, param, ret);
    char *s = hm_type_to_str(ctx, arrow);
    ASSERT_NOT_NULL(s);
    ASSERT_STR_CONTAINS(s, "int");
    ASSERT_STR_CONTAINS(s, "bool");
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Record types
 * ============================================================================ */

void test_record_closed_empty(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    HMType *rec = hm_record_closed(ctx, NULL, NULL, 0);
    ASSERT_NOT_NULL(rec);
    ASSERT_EQ(rec->kind, HM_RECORD);
    ASSERT_EQ(rec->as.record.field_count, 0);
    ASSERT_NULL(rec->as.record.row_tail);
    hm_ctx_free(ctx);
}

void test_record_closed_two_fields(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    const char *names[] = {"x", "y"};
    HMType *t_int = hm_con_type(ctx, "int");
    HMType *types[] = {t_int, t_int};
    HMType *rec = hm_record_closed(ctx, names, types, 2);
    ASSERT_NOT_NULL(rec);
    ASSERT_EQ(rec->kind, HM_RECORD);
    ASSERT_EQ(rec->as.record.field_count, 2);
    ASSERT_NULL(rec->as.record.row_tail);
    hm_ctx_free(ctx);
}

void test_record_open(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    const char *names[] = {"name"};
    HMType *t_str = hm_con_type(ctx, "string");
    HMType *types[] = {t_str};
    HMType *rec = hm_record_open(ctx, names, types, 1);
    ASSERT_NOT_NULL(rec);
    ASSERT_EQ(rec->kind, HM_RECORD);
    ASSERT_NOT_NULL(rec->as.record.row_tail);
    ASSERT_EQ(rec->as.record.row_tail->kind, HM_VAR);
    hm_ctx_free(ctx);
}

void test_record_field_lookup_existing(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    const char *names[] = {"age", "name"};
    HMType *t_int = hm_con_type(ctx, "int");
    HMType *t_str = hm_con_type(ctx, "string");
    HMType *types[] = {t_int, t_str};
    HMType *rec = hm_record_closed(ctx, names, types, 2);
    HMType *age_t = hm_record_field_type(ctx, rec, "age", 1, 1);
    ASSERT_NOT_NULL(age_t);
    HMType *resolved = hm_subst_apply(ctx, age_t);
    ASSERT_EQ(resolved->kind, HM_CON);
    ASSERT_STR_EQ(resolved->as.con_name, "int");
    hm_ctx_free(ctx);
}

void test_record_field_lookup_missing_closed(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    const char *names[] = {"x"};
    HMType *t_int = hm_con_type(ctx, "int");
    HMType *types[] = {t_int};
    HMType *rec = hm_record_closed(ctx, names, types, 1);
    suppress_stderr();
    HMType *missing = hm_record_field_type(ctx, rec, "z", 1, 1);
    restore_stderr();
    /* Missing field in closed record: returns a fresh type var and sets has_error */
    ASSERT_NOT_NULL(missing);
    ASSERT(ctx->has_error);
    hm_ctx_free(ctx);

}

void test_record_spread(void) {
    InferCtx *ctx = hm_ctx_new("test.nano");
    const char *base_names[] = {"x"};
    HMType *t_int = hm_con_type(ctx, "int");
    HMType *base_types[] = {t_int};
    HMType *base = hm_record_closed(ctx, base_names, base_types, 1);

    const char *extra_names[] = {"y"};
    HMType *t_str = hm_con_type(ctx, "string");
    HMType *extra_types[] = {t_str};
    HMType *spread = hm_record_spread(ctx, base, extra_names, extra_types, 1);
    ASSERT_NOT_NULL(spread);
    ASSERT_EQ(spread->kind, HM_RECORD);
    /* Spread should have both x and y */
    ASSERT(spread->as.record.field_count >= 2);
    hm_ctx_free(ctx);
}

/* ============================================================================
 * Full program inference
 * ============================================================================ */

void test_infer_empty_program(void) {
    /* Build an empty program AST node directly */
    ASTNode prog;
    memset(&prog, 0, sizeof(prog));
    prog.type = AST_PROGRAM;
    prog.as.program.items = NULL;
    prog.as.program.count = 0;
    bool ok = hm_infer_program(&prog, "empty.nano");
    ASSERT(ok);  /* empty program has no type errors */
}

void test_infer_simple_parsed_program(void) {
    /* Lex and parse a simple nano program, run HM inference on it */
    const char *src =
        "fn identity(x: int) -> int { x }\n"
        "fn main() -> void { (print \"hello\") }\n";
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    ASSERT_NOT_NULL(tokens);
    ASTNode *program = parse_program(tokens, token_count);
    ASSERT_NOT_NULL(program);
    /* Run HM inference — just verify it doesn't crash */
    suppress_stderr();
    (void)hm_infer_program(program, "simple.nano");
    restore_stderr();
    free_ast(program);
    free_tokens(tokens, token_count);

}

void test_infer_with_effects(void) {
    ASTNode prog;
    memset(&prog, 0, sizeof(prog));
    prog.type = AST_PROGRAM;
    prog.as.program.items = NULL;
    prog.as.program.count = 0;
    EffectRegistry *reg = effect_registry_new();
    bool ok = hm_infer_program_with_effects(&prog, "fx.nano", reg);
    ASSERT(ok);
    effect_registry_free(reg);
}

void test_infer_for_lsp_and_free(void) {
    ASTNode prog;
    memset(&prog, 0, sizeof(prog));
    prog.type = AST_PROGRAM;
    prog.as.program.items = NULL;
    prog.as.program.count = 0;
    HMInferResult result = hm_infer_program_for_lsp(&prog, "lsp.nano");
    ASSERT(result.ok);
    ASSERT_NOT_NULL(result.ctx);
    /* Lookup a name that doesn't exist */
    TypeScheme *scheme = hm_env_lookup_scheme(result.env, "nonexistent");
    ASSERT_NULL(scheme);
    hm_infer_result_free(&result);
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Context Management Tests ===\n");
    TEST(ctx_create_and_free);
    TEST(ctx_null_source_file);

    printf("\n=== Type Variable Tests ===\n");
    TEST(tv_fresh_unique_ids);

    printf("\n=== Concrete Type Tests ===\n");
    TEST(con_type_int);
    TEST(con_type_all_primitives);

    printf("\n=== Arrow Type Tests ===\n");
    TEST(arrow_type_basic);
    TEST(arrow_type_with_effects);

    printf("\n=== Unification Tests ===\n");
    TEST(unify_same_con);
    TEST(unify_var_with_con);
    TEST(unify_con_with_var);
    TEST(unify_var_with_var);
    TEST(unify_mismatch_errors);
    TEST(unify_arrow_arrow);
    TEST(unify_arrow_param_mismatch);

    printf("\n=== Substitution Tests ===\n");
    TEST(subst_apply_unbound_var);
    TEST(subst_apply_con_type);
    TEST(subst_apply_after_unify);

    printf("\n=== Type-to-String Tests ===\n");
    TEST(type_to_str_con);
    TEST(type_to_str_var);
    TEST(type_to_str_arrow);

    printf("\n=== Record Type Tests ===\n");
    TEST(record_closed_empty);
    TEST(record_closed_two_fields);
    TEST(record_open);
    TEST(record_field_lookup_existing);
    TEST(record_field_lookup_missing_closed);
    TEST(record_spread);

    printf("\n=== Program Inference Tests ===\n");
    TEST(infer_empty_program);
    TEST(infer_simple_parsed_program);
    TEST(infer_with_effects);
    TEST(infer_for_lsp_and_free);

    printf("\n✓ All type inference tests passed!\n");
    return 0;
}
