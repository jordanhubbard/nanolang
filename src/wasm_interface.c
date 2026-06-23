/*
 * wasm_interface.c — Emscripten entry points for the NanoLang browser playground
 *
 * Exposes the NanoLang interpreter to JavaScript via EMSCRIPTEN_KEEPALIVE exports.
 *
 * Output capture:  In Emscripten, printf/fprintf(stderr,...) automatically
 * routes to Module.print / Module.printErr (configurable JS callbacks).
 * The JS wrapper in playground.js temporarily replaces those callbacks before
 * calling nl_run() and restores them after — no C-side redirection needed.
 *
 * Pipeline (mirrors nano_main.c interpret_file, minus file I/O and FFI loading):
 *   source string → lex → parse → env → type_check → run_program → main()
 *
 * For programs with `from ... import ...`, process_imports() is called but
 * silently skips modules that cannot be resolved in the browser environment.
 */

#include <emscripten.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanolang.h"
#include "version.h"

/* ── Globals required by the interpreter (normally set by nano_main.c) ─── */
int   g_argc = 0;
char **g_argv = NULL;

char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* ── JS-callable API ─────────────────────────────────────────────────────── */

/* Returns the NanoLang version string as "major.minor.patch". */
EMSCRIPTEN_KEEPALIVE
const char *nl_version(void) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "%d.%d.%d",
             NANOLANG_VERSION_MAJOR,
             NANOLANG_VERSION_MINOR,
             NANOLANG_VERSION_PATCH);
    return buf;
}

/*
 * nl_run(source) — compile and execute a NanoLang source string.
 *
 * Returns 0 on success, non-zero on error.
 *
 * stdout/stderr output reaches the browser via Emscripten's Module.print /
 * Module.printErr.  The JS caller should intercept those before calling here
 * and restore them after (see playground.js: runNano()).
 */
EMSCRIPTEN_KEEPALIVE
int nl_run(const char *source) {
    if (!source) return 1;

    /* Phase 1: Lex */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Error: lexing failed\n");
        return 1;
    }

    /* Phase 2: Parse */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Error: parsing failed\n");
        free_tokens(tokens, token_count);
        return 1;
    }

    /* Phase 3: Environment + imports */
    clear_module_cache();
    Environment *env     = create_environment();
    ModuleList  *modules = create_module_list();

    /* process_imports is a no-op for programs with no import statements.
     * For programs that do import, it resolves relative to get_project_root()
     * (".") — missing modules are reported to stderr but don't abort.      */
    process_imports(program, env, modules, "<browser>");

    /* Phase 4: Type-check */
    typecheck_set_current_file("<browser>");
    if (!type_check(program, env)) {
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        return 1;
    }

    /* Phase 5: Register top-level functions / run top-level statements */
    if (!run_program(program, env)) {
        fprintf(stderr, "Error: runtime error\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        return 1;
    }

    /* Phase 6: Call main() if present */
    int rc = 0;
    Function *main_fn = env_get_function(env, "main");
    if (main_fn) {
        Value result = call_function("main", NULL, 0, env);
        if (result.type == VAL_INT)
            rc = (int)result.as.int_val;
    }

    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    clear_module_cache();
    return rc;
}

/*
 * nl_check(source) — type-check only, no execution.
 *
 * Returns 0 if type-correct, 1 on error.
 * Diagnostics go to stderr (Module.printErr on the JS side).
 */
EMSCRIPTEN_KEEPALIVE
int nl_check(const char *source) {
    if (!source) return 1;

    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "Error: lexing failed\n");
        return 1;
    }

    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "Error: parsing failed\n");
        free_tokens(tokens, token_count);
        return 1;
    }

    clear_module_cache();
    Environment *env     = create_environment();
    ModuleList  *modules = create_module_list();
    process_imports(program, env, modules, "<browser>");

    typecheck_set_current_file("<browser>");
    int rc = type_check(program, env) ? 0 : 1;

    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    clear_module_cache();
    return rc;
}
