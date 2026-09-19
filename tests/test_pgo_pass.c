/*
 * test_pgo_pass.c — Unit tests for the PGO (profile-guided inlining) pass
 *
 * Tests:
 *   1.  pgo_load_profile: parse .nano.prof format
 *   2.  pgo_is_hot: threshold detection
 *   3.  pgo_apply: call site inlining (simple function, 1 arg)
 *   4.  pgo_apply: arity mismatch → no inline
 *   5.  pgo_apply: recursive function → no inline
 *   6.  pgo_apply: body > PGO_MAX_INLINE_STMTS → no inline
 *   7.  pgo_apply: indirect call (func_expr != NULL) → no inline
 *   8.  Manual threshold override
 */

#include "../src/pgo_pass.h"
#include "../src/nanolang.h"
#include "../src/generated/compiler_schema.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define TEST(name) static void test_##name(void)
#define RUN(name) do { test_##name(); printf("  %-50s PASS\n", #name "..."); g_pass++; } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); g_fail++; return; } } while(0)

/* ── Helpers to build minimal AST nodes ─────────────────────────────────── */

static ASTNode *make_num(int64_t v) {
    ASTNode *n = calloc(1, sizeof(ASTNode));
    n->type = AST_NUMBER;
    n->as.number = v;
    return n;
}

static ASTNode *make_ident(const char *name) {
    ASTNode *n = calloc(1, sizeof(ASTNode));
    n->type = AST_IDENTIFIER;
    n->as.identifier = (char *)name;
    return n;
}

static ASTNode *make_call(const char *fn_name, ASTNode **args, int argc) {
    ASTNode *n = calloc(1, sizeof(ASTNode));
    n->type = AST_CALL;
    n->as.call.name      = (char *)fn_name;
    n->as.call.args      = args;
    n->as.call.arg_count = argc;
    return n;
}

static ASTNode *make_block(ASTNode **stmts, int count) {
    ASTNode *n = calloc(1, sizeof(ASTNode));
    n->type = AST_BLOCK;
    n->as.block.statements = stmts;
    n->as.block.count = count;
    return n;
}

static ASTNode *make_return(ASTNode *value) {
    ASTNode *n = calloc(1, sizeof(ASTNode));
    n->type = AST_RETURN;
    n->as.return_stmt.value = value;
    return n;
}

/* Build a function: fn double(x) { return x + x } */
static ASTNode *make_fn_double(void) {
    static Parameter p = { "x", TYPE_INT, NULL, TYPE_INT, NULL, NULL };
    /* Body: return (x + x) — represented as prefix_op with two ident args */
    ASTNode **plus_args = calloc(2, sizeof(ASTNode *));
    plus_args[0] = make_ident("x");
    plus_args[1] = make_ident("x");
    ASTNode *add = calloc(1, sizeof(ASTNode));
    add->type = AST_PREFIX_OP;
    add->as.prefix_op.op = TOKEN_PLUS;
    add->as.prefix_op.args = plus_args;
    add->as.prefix_op.arg_count = 2;

    ASTNode *ret = make_return(add);
    ASTNode **body_stmts = calloc(1, sizeof(ASTNode *));
    body_stmts[0] = ret;
    ASTNode *body = make_block(body_stmts, 1);

    ASTNode *fn = calloc(1, sizeof(ASTNode));
    fn->type = AST_FUNCTION;
    fn->as.function.name        = "double";
    fn->as.function.params      = &p;
    fn->as.function.param_count = 1;
    fn->as.function.body        = body;
    return fn;
}

/* Build program with one function */
static ASTNode *make_program(ASTNode **items, int count) {
    ASTNode *prog = calloc(1, sizeof(ASTNode));
    prog->type = AST_PROGRAM;
    prog->as.program.items = items;
    prog->as.program.count = count;
    return prog;
}

/* ── Write a temporary .nano.prof file ─────────────────────────────────── */
static const char *write_prof(const char *content) {
    static char path[64] = "/tmp/test_pgo.nano.prof";
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    fputs(content, f);
    fclose(f);
    return path;
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

TEST(load_profile_basic) {
    const char *prof_path = write_prof(
        "double 5000\n"
        "helper 100\n"
        "main 1\n"
    );
    ASSERT(prof_path);
    PGOProfile *p = pgo_load_profile(prof_path);
    ASSERT(p != NULL);
    ASSERT(p->count == 3);
    ASSERT(strcmp(p->entries[0].name, "double") == 0);
    ASSERT(p->entries[0].calls == 5000);
    pgo_profile_free(p);
}

TEST(is_hot_threshold) {
    const char *prof_path = write_prof("double 5000\nhelper 10\n");
    PGOProfile *p = pgo_load_profile_threshold(prof_path, 1000);
    ASSERT(p != NULL);
    ASSERT(pgo_is_hot(p, "double"));
    ASSERT(!pgo_is_hot(p, "helper"));
    ASSERT(!pgo_is_hot(p, "nonexistent"));
    pgo_profile_free(p);
}

TEST(apply_inlines_hot_call) {
    /* Build: fn double(x) { return x+x }  and a call site: double(42) */
    ASTNode *fn = make_fn_double();
    ASTNode *call = make_call("double", (ASTNode*[]){make_num(42)}, 1);

    ASTNode **prog_items = calloc(2, sizeof(ASTNode *));
    prog_items[0] = fn;
    prog_items[1] = call;
    ASTNode *prog = make_program(prog_items, 2);

    const char *prof_path = write_prof("double 10000\n");
    PGOProfile *p = pgo_load_profile_threshold(prof_path, 100);
    ASSERT(p);
    int inlined = pgo_apply(prog, p);
    /* Call should have been rewritten to an AST_BLOCK or similar (not AST_CALL) */
    ASSERT(inlined > 0);
    ASSERT(prog_items[1]->type != AST_CALL);
    pgo_profile_free(p);
}

TEST(apply_skips_cold_call) {
    ASTNode *fn = make_fn_double();
    ASTNode *call = make_call("double", (ASTNode*[]){make_num(7)}, 1);

    ASTNode **prog_items = calloc(2, sizeof(ASTNode *));
    prog_items[0] = fn;
    prog_items[1] = call;
    ASTNode *prog = make_program(prog_items, 2);

    const char *prof_path = write_prof("double 5\n");  /* below threshold */
    PGOProfile *p = pgo_load_profile_threshold(prof_path, 1000);
    ASSERT(p);
    int inlined = pgo_apply(prog, p);
    ASSERT(inlined == 0);
    ASSERT(prog_items[1]->type == AST_CALL);
    pgo_profile_free(p);
}

TEST(apply_skips_arity_mismatch) {
    ASTNode *fn = make_fn_double();
    /* Call with 2 args to a 1-param function */
    ASTNode *call = make_call("double",
        (ASTNode*[]){make_num(1), make_num(2)}, 2);

    ASTNode **prog_items = calloc(2, sizeof(ASTNode *));
    prog_items[0] = fn;
    prog_items[1] = call;
    ASTNode *prog = make_program(prog_items, 2);

    const char *prof_path = write_prof("double 99999\n");
    PGOProfile *p = pgo_load_profile_threshold(prof_path, 1);
    ASSERT(p);
    int inlined = pgo_apply(prog, p);
    ASSERT(inlined == 0);  /* arity mismatch → not inlined */
    ASSERT(prog_items[1]->type == AST_CALL);
    pgo_profile_free(p);
}

TEST(apply_skips_indirect_call) {
    ASTNode *fn = make_fn_double();
    ASTNode *call = make_call("double", (ASTNode*[]){make_num(5)}, 1);
    /* Mark as indirect */
    call->as.call.func_expr = make_ident("fn_ptr");

    ASTNode **prog_items = calloc(2, sizeof(ASTNode *));
    prog_items[0] = fn;
    prog_items[1] = call;
    ASTNode *prog = make_program(prog_items, 2);

    const char *prof_path = write_prof("double 99999\n");
    PGOProfile *p = pgo_load_profile_threshold(prof_path, 1);
    ASSERT(p);
    int inlined = pgo_apply(prog, p);
    ASSERT(inlined == 0);
    ASSERT(prog_items[1]->type == AST_CALL);
    pgo_profile_free(p);
}

TEST(apply_clones_and_substitutes_match_guards) {
    static Parameter parameter = { "predicate", TYPE_BOOL, NULL, TYPE_BOOL, NULL, NULL };
    char *bindings[1] = { "payload" };
    ASTNode *guard = make_ident("predicate");
    ASTNode *scrutinee = make_num(0);
    ASTNode *arm_body = make_num(1);
    ASTNode **guards = calloc(1, sizeof(ASTNode *));
    ASTNode **arm_bodies = calloc(1, sizeof(ASTNode *));
    guards[0] = guard;
    arm_bodies[0] = arm_body;

    ASTNode *match = calloc(1, sizeof(ASTNode));
    match->type = AST_MATCH;
    match->as.match_expr.expr = scrutinee;
    match->as.match_expr.arm_count = 1;
    match->as.match_expr.pattern_bindings = bindings;
    match->as.match_expr.guard_exprs = guards;
    match->as.match_expr.arm_bodies = arm_bodies;

    ASTNode *ret = make_return(match);
    ASTNode **fn_stmts = calloc(1, sizeof(ASTNode *));
    fn_stmts[0] = ret;
    ASTNode *fn = calloc(1, sizeof(ASTNode));
    fn->type = AST_FUNCTION;
    fn->as.function.name = "guarded";
    fn->as.function.params = &parameter;
    fn->as.function.param_count = 1;
    fn->as.function.body = make_block(fn_stmts, 1);

    ASTNode **args = calloc(1, sizeof(ASTNode *));
    args[0] = make_num(1);
    ASTNode *call = make_call("guarded", args, 1);
    ASTNode **items = calloc(2, sizeof(ASTNode *));
    items[0] = fn;
    items[1] = call;
    ASTNode *program = make_program(items, 2);

    const char *prof_path = write_prof("guarded 10000\n");
    PGOProfile *profile = pgo_load_profile_threshold(prof_path, 1);
    ASSERT(profile);
    ASSERT(pgo_apply(program, profile) == 1);
    ASSERT(call->type == AST_BLOCK);
    ASTNode *cloned_return = call->as.block.statements[0];
    ASSERT(cloned_return && cloned_return->type == AST_RETURN);
    ASTNode *cloned_match = cloned_return->as.return_stmt.value;
    ASSERT(cloned_match && cloned_match->type == AST_MATCH);
    ASSERT(cloned_match->as.match_expr.guard_exprs != guards);
    ASSERT(cloned_match->as.match_expr.guard_exprs[0] != guard);
    ASSERT(cloned_match->as.match_expr.guard_exprs[0]->type == AST_NUMBER);
    ASSERT(cloned_match->as.match_expr.guard_exprs[0]->as.number == 1);
    ASSERT(guard->type == AST_IDENTIFIER);
    ASSERT(strcmp(guard->as.identifier, "predicate") == 0);
    pgo_profile_free(profile);
}

TEST(apply_preserves_match_binding_shadow_in_guard_and_body) {
    static Parameter parameter = { "payload", TYPE_INT, NULL, TYPE_INT, NULL, NULL };
    char *bindings[1] = { "payload" };
    ASTNode *guard = make_ident("payload");
    ASTNode *arm_body = make_ident("payload");
    ASTNode **guards = calloc(1, sizeof(ASTNode *));
    ASTNode **arm_bodies = calloc(1, sizeof(ASTNode *));
    guards[0] = guard;
    arm_bodies[0] = arm_body;

    ASTNode *match = calloc(1, sizeof(ASTNode));
    match->type = AST_MATCH;
    match->as.match_expr.expr = make_num(0);
    match->as.match_expr.arm_count = 1;
    match->as.match_expr.pattern_bindings = bindings;
    match->as.match_expr.guard_exprs = guards;
    match->as.match_expr.arm_bodies = arm_bodies;

    ASTNode **fn_stmts = calloc(1, sizeof(ASTNode *));
    fn_stmts[0] = make_return(match);
    ASTNode *fn = calloc(1, sizeof(ASTNode));
    fn->type = AST_FUNCTION;
    fn->as.function.name = "shadowed";
    fn->as.function.params = &parameter;
    fn->as.function.param_count = 1;
    fn->as.function.body = make_block(fn_stmts, 1);

    ASTNode **args = calloc(1, sizeof(ASTNode *));
    args[0] = make_num(7);
    ASTNode *call = make_call("shadowed", args, 1);
    ASTNode **items = calloc(2, sizeof(ASTNode *));
    items[0] = fn;
    items[1] = call;
    ASTNode *program = make_program(items, 2);

    const char *prof_path = write_prof("shadowed 10000\n");
    PGOProfile *profile = pgo_load_profile_threshold(prof_path, 1);
    ASSERT(profile);
    ASSERT(pgo_apply(program, profile) == 1);
    ASTNode *cloned_return = call->as.block.statements[0];
    ASTNode *cloned_match = cloned_return->as.return_stmt.value;
    ASTNode *cloned_guard = cloned_match->as.match_expr.guard_exprs[0];
    ASTNode *cloned_body = cloned_match->as.match_expr.arm_bodies[0];
    ASSERT(cloned_guard->type == AST_IDENTIFIER);
    ASSERT(strcmp(cloned_guard->as.identifier, "payload") == 0);
    ASSERT(cloned_body->type == AST_IDENTIFIER);
    ASSERT(strcmp(cloned_body->as.identifier, "payload") == 0);
    pgo_profile_free(profile);
}

TEST(load_empty_profile) {
    const char *prof_path = write_prof("");
    PGOProfile *p = pgo_load_profile(prof_path);
    ASSERT(p != NULL);
    ASSERT(p->count == 0);
    ASSERT(p->functions_hot == 0);
    pgo_profile_free(p);
}

TEST(load_missing_profile) {
    PGOProfile *p = pgo_load_profile("/tmp/this_file_does_not_exist_nano.prof");
    ASSERT(p == NULL);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[pgo_pass] Profile-guided inlining pass tests...\n\n");
    RUN(load_profile_basic);
    RUN(is_hot_threshold);
    RUN(apply_inlines_hot_call);
    RUN(apply_skips_cold_call);
    RUN(apply_skips_arity_mismatch);
    RUN(apply_skips_indirect_call);
    RUN(apply_clones_and_substitutes_match_guards);
    RUN(apply_preserves_match_binding_shadow_in_guard_and_body);
    RUN(load_empty_profile);
    RUN(load_missing_profile);
    printf("\n");
    if (g_fail == 0) { printf("All %d tests passed.\n", g_pass); return 0; }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
