/**
 * test_optimizer.c — Unit tests for constant folding + dead code elimination.
 *
 * Tests are self-contained: each test builds a minimal AST by hand, runs the
 * relevant pass, and inspects the result.  No file I/O or subprocess invocation.
 *
 * Build:
 *   cc -Wall -Isrc -o tests/test_optimizer tests/test_optimizer.c \
 *       $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) -lm
 *
 * Run:
 *   ./tests/test_optimizer
 */

#include "../src/nanolang.h"
#include "../src/fold_constants.h"
#include "../src/dce_pass.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test helpers ─────────────────────────────────────────────────────────── */

#define TEST(name) \
    do { printf("  %-55s", "test_" #name "..."); test_##name(); printf("PASS\n"); } while (0)

#define ASSERT(cond) \
    do { if (!(cond)) { \
        printf("FAIL\n    Assertion failed: %s  (line %d)\n", #cond, __LINE__); \
        exit(1); \
    } } while (0)

#define ASSERT_EQ_INT(a, b) \
    do { long long _a = (a); long long _b = (b); \
        if (_a != _b) { \
            printf("FAIL\n    %s == %s  got %lld, expected %lld  (line %d)\n", \
                   #a, #b, _a, _b, __LINE__); exit(1); \
        } \
    } while (0)

#define ASSERT_EQ_BOOL(a, b) \
    do { bool _a = (a); bool _b = (b); \
        if (_a != _b) { \
            printf("FAIL\n    %s == %s  got %s, expected %s  (line %d)\n", \
                   #a, #b, _a?"true":"false", _b?"true":"false", __LINE__); exit(1); \
        } \
    } while (0)

/* ── AST construction helpers ─────────────────────────────────────────────── */

/** Allocate a zeroed ASTNode and set type/line/column. */
static ASTNode *make_node(ASTNodeType type) {
    ASTNode *n = calloc(1, sizeof(ASTNode));
    n->type   = type;
    n->line   = 1;
    n->column = 1;
    return n;
}

static ASTNode *make_int(long long v) {
    ASTNode *n = make_node(AST_NUMBER);
    n->as.number = v;
    return n;
}

static ASTNode *make_float(double v) {
    ASTNode *n = make_node(AST_FLOAT);
    n->as.float_val = v;
    return n;
}

static ASTNode *make_bool(bool v) {
    ASTNode *n = make_node(AST_BOOL);
    n->as.bool_val = v;
    return n;
}

static ASTNode *make_ident(const char *name) {
    ASTNode *n = make_node(AST_IDENTIFIER);
    n->as.identifier = strdup(name);
    return n;
}

/**
 * make_binop(op, left, right) — builds an AST_PREFIX_OP node with 2 args.
 * Mirrors how the parser creates binary expressions.
 */
static ASTNode *make_binop(TokenType op, ASTNode *left, ASTNode *right) {
    ASTNode *n = make_node(AST_PREFIX_OP);
    n->as.prefix_op.op        = op;
    n->as.prefix_op.arg_count = 2;
    n->as.prefix_op.args      = malloc(sizeof(ASTNode *) * 2);
    n->as.prefix_op.args[0]   = left;
    n->as.prefix_op.args[1]   = right;
    return n;
}

static ASTNode *make_unop(TokenType op, ASTNode *operand) {
    ASTNode *n = make_node(AST_PREFIX_OP);
    n->as.prefix_op.op        = op;
    n->as.prefix_op.arg_count = 1;
    n->as.prefix_op.args      = malloc(sizeof(ASTNode *));
    n->as.prefix_op.args[0]   = operand;
    return n;
}

/**
 * Wrap a single expression in an AST_PROGRAM node so the passes can walk it.
 * The program takes ownership of *expr*.
 */
static ASTNode *wrap_program(ASTNode *expr) {
    ASTNode *prog = make_node(AST_PROGRAM);
    prog->as.program.count = 1;
    prog->as.program.items = malloc(sizeof(ASTNode *));
    prog->as.program.items[0] = expr;
    return prog;
}

/**
 * Build a block with *count* statements taken from va-list style array.
 */
static ASTNode *make_block(ASTNode **stmts, int count) {
    ASTNode *blk = make_node(AST_BLOCK);
    blk->as.block.count      = count;
    blk->as.block.statements = malloc(sizeof(ASTNode *) * (count > 0 ? count : 1));
    for (int i = 0; i < count; i++) blk->as.block.statements[i] = stmts[i];
    return blk;
}

static ASTNode *make_let(const char *name, ASTNode *value) {
    ASTNode *n = make_node(AST_LET);
    n->as.let.name      = strdup(name);
    n->as.let.var_type  = TYPE_INT;
    n->as.let.value     = value;
    return n;
}

static ASTNode *make_if(ASTNode *cond, ASTNode *then_br, ASTNode *else_br) {
    ASTNode *n = make_node(AST_IF);
    n->as.if_stmt.condition   = cond;
    n->as.if_stmt.then_branch = then_br;
    n->as.if_stmt.else_branch = else_br;
    return n;
}

/* ── Test 1: constant fold 2 + 3 → 5 ─────────────────────────────────────── */
static void test_fold_add_ints(void) {
    /* (+ 2 3) */
    ASTNode *expr = make_binop(TOKEN_PLUS, make_int(2), make_int(3));
    ASTNode *prog = wrap_program(expr);

    int folded = fold_constants(prog, false);

    ASSERT_EQ_INT(folded, 1);
    /* prog->items[0] should now be an AST_NUMBER with value 5 */
    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_NUMBER);
    ASSERT_EQ_INT(result->as.number, 5);

    free_ast(prog);
}

/* ── Test 2: constant fold 10 * 10 - 5 → 95 ──────────────────────────────── */
static void test_fold_mul_sub(void) {
    /* (- (* 10 10) 5) */
    ASTNode *inner = make_binop(TOKEN_STAR, make_int(10), make_int(10));
    ASTNode *outer = make_binop(TOKEN_MINUS, inner, make_int(5));
    ASTNode *prog  = wrap_program(outer);

    int folded = fold_constants(prog, false);

    /* Two folds: inner (* 10 10)=100, outer (- 100 5)=95 */
    ASSERT(folded >= 2);
    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_NUMBER);
    ASSERT_EQ_INT(result->as.number, 95);

    free_ast(prog);
}

/* ── Test 3: constant fold true && false → false ──────────────────────────── */
static void test_fold_bool_and(void) {
    /* (&& true false) */
    ASTNode *expr = make_binop(TOKEN_AND, make_bool(true), make_bool(false));
    ASTNode *prog = wrap_program(expr);

    int folded = fold_constants(prog, false);

    ASSERT_EQ_INT(folded, 1);
    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_BOOL);
    ASSERT_EQ_BOOL(result->as.bool_val, false);

    free_ast(prog);
}

/* ── Test 4: DCE if(true) A else B → A ───────────────────────────────────── */
static void test_dce_if_true(void) {
    /* if (true) 42 else 99 */
    ASTNode *then_br = make_int(42);
    ASTNode *else_br = make_int(99);
    ASTNode *if_node = make_if(make_bool(true), then_br, else_br);
    ASTNode *prog    = wrap_program(if_node);

    int elim = dce_pass(prog, false);

    ASSERT_EQ_INT(elim, 1);
    ASTNode *result = prog->as.program.items[0];
    /* Node should now be the then-branch: integer 42 */
    ASSERT(result->type == AST_NUMBER);
    ASSERT_EQ_INT(result->as.number, 42);

    free_ast(prog);
}

/* ── Test 5: DCE if(false) A else B → B ──────────────────────────────────── */
static void test_dce_if_false(void) {
    /* if (false) 42 else 99 */
    ASTNode *if_node = make_if(make_bool(false), make_int(42), make_int(99));
    ASTNode *prog    = wrap_program(if_node);

    int elim = dce_pass(prog, false);

    ASSERT_EQ_INT(elim, 1);
    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_NUMBER);
    ASSERT_EQ_INT(result->as.number, 99);

    free_ast(prog);
}

/* ── Test 6: combined — folded condition drives DCE ──────────────────────── */
/*
 * Equivalent to:
 *   if (1 == 1) 7 else 0
 *
 * fold_constants should reduce (1 == 1) → true, then dce_pass should
 * eliminate the else branch and keep 7.
 */
static void test_combined_fold_then_dce(void) {
    /* (== 1 1) */
    ASTNode *cond    = make_binop(TOKEN_EQ, make_int(1), make_int(1));
    ASTNode *if_node = make_if(cond, make_int(7), make_int(0));
    ASTNode *prog    = wrap_program(if_node);

    int folded = fold_constants(prog, false);
    ASSERT(folded >= 1);  /* (== 1 1) folded to true */

    int elim = dce_pass(prog, false);
    ASSERT_EQ_INT(elim, 1);  /* if(true) eliminated */

    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_NUMBER);
    ASSERT_EQ_INT(result->as.number, 7);

    free_ast(prog);
}

/* ── Test 7: dead let binding removed ────────────────────────────────────── */
/*
 * {
 *   let unused = 42      ← dead: never referenced
 *   let kept   = 1       ← referenced (just to have something live)
 * }
 * After DCE: block should have 1 statement (kept).
 *
 * Note: "kept" is referenced by a later print or identifier;
 * we simulate that by adding an identifier node after it.
 */
static void test_dce_dead_let(void) {
    /* block: [ let unused = 42, let kept = 1, kept ] */
    ASTNode *stmts[3];
    stmts[0] = make_let("unused", make_int(42));
    stmts[1] = make_let("kept",   make_int(1));
    stmts[2] = make_ident("kept");   /* reference to 'kept' keeps it alive */

    ASTNode *blk  = make_block(stmts, 3);
    ASTNode *prog = wrap_program(blk);

    int elim = dce_pass(prog, false);

    ASSERT(elim >= 1);  /* 'unused' was eliminated */

    /* Block should now have 2 statements: let kept + the identifier */
    ASTNode *block = prog->as.program.items[0];
    ASSERT(block->type == AST_BLOCK);
    ASSERT_EQ_INT(block->as.block.count, 2);
    ASSERT(block->as.block.statements[0]->type == AST_LET);
    ASSERT(strcmp(block->as.block.statements[0]->as.let.name, "kept") == 0);

    free_ast(prog);
}

/* ── Test 8: unary minus fold ────────────────────────────────────────────── */
static void test_fold_unary_minus(void) {
    /* (- 7)  →  -7 */
    ASTNode *expr = make_unop(TOKEN_MINUS, make_int(7));
    ASTNode *prog = wrap_program(expr);

    int folded = fold_constants(prog, false);

    ASSERT_EQ_INT(folded, 1);
    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_NUMBER);
    ASSERT_EQ_INT(result->as.number, -7);

    free_ast(prog);
}

/* ── Test 9: comparison fold with floats ─────────────────────────────────── */
static void test_fold_float_lt(void) {
    /* (< 1.5 2.5) → true */
    ASTNode *expr = make_binop(TOKEN_LT, make_float(1.5), make_float(2.5));
    ASTNode *prog = wrap_program(expr);

    int folded = fold_constants(prog, false);

    ASSERT_EQ_INT(folded, 1);
    ASTNode *result = prog->as.program.items[0];
    ASSERT(result->type == AST_BOOL);
    ASSERT_EQ_BOOL(result->as.bool_val, true);

    free_ast(prog);
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("Optimizer tests\n");
    printf("---------------\n");

    TEST(fold_add_ints);
    TEST(fold_mul_sub);
    TEST(fold_bool_and);
    TEST(dce_if_true);
    TEST(dce_if_false);
    TEST(combined_fold_then_dce);
    TEST(dce_dead_let);
    TEST(fold_unary_minus);
    TEST(fold_float_lt);

    printf("---------------\n");
    printf("All optimizer tests passed.\n");
    return 0;
}
