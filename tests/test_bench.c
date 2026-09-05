/**
 * test_bench.c — bench_native_run must call the interpreter, not a no-op stub.
 */

#include "../src/nanolang.h"
#include "../src/bench.h"
#include "../src/builtins_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#define ASSERT(cond) \
    if (!(cond)) { \
        printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); \
        exit(1); \
    }

typedef struct {
    Environment *env;
    ASTNode     *program;
    Token       *tokens;
    int          token_count;
} RunCtx;

static bool run_ctx_init(RunCtx *ctx, const char *src) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->tokens = tokenize(src, &ctx->token_count);
    if (!ctx->tokens) return false;

    ctx->program = parse_program(ctx->tokens, ctx->token_count);
    if (!ctx->program) return false;

    clear_module_cache();
    ctx->env = create_environment();

    typecheck_set_current_file("<bench-test>");
    if (!type_check(ctx->program, ctx->env)) return false;
    return run_program(ctx->program, ctx->env);
}

static void run_ctx_free(RunCtx *ctx) {
    if (ctx->env)     free_environment(ctx->env);
    if (ctx->program) free_ast(ctx->program);
    if (ctx->tokens)  free_tokens(ctx->tokens, ctx->token_count);
    clear_module_cache();
    memset(ctx, 0, sizeof(*ctx));
}

static void test_bench_native_run_invokes_function(void) {
    RunCtx ctx;
    bool ok = run_ctx_init(&ctx,
        "let mut hits: int = 0\n"
        "fn bench_mark() -> int {\n"
        "  set hits (+ hits 1)\n"
        "  return hits\n"
        "}\n"
        "shadow bench_mark {\n"
        "  assert true\n"
        "}\n"
        "fn main() -> int { return 0 }\n"
    );
    ASSERT(ok);

    BenchNativeCtx bctx = {
        .program = ctx.program,
        .fn_name = "bench_mark",
        .fn_node = NULL,
        .env     = ctx.env,
    };
    bench_native_run(&bctx, 3);

    Value after = call_function("bench_mark", NULL, 0, ctx.env);
    ASSERT(after.type == VAL_INT);
    ASSERT(after.as.int_val == 4);

    run_ctx_free(&ctx);
}

int main(void) {
    printf("Running bench runner tests...\n");
    test_bench_native_run_invokes_function();
    printf("  Testing bench_native_run invokes function... ✓\n");
    printf("All bench runner tests passed.\n");
    return 0;
}
