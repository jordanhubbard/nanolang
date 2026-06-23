/* repl.c -- Interactive REPL for nanolang
 *
 * run_repl() is called by:
 *   - nanorepl standalone binary (repl_main.c)
 *   - nano --repl flag            (nano_main.c)
 *
 * Each input line is parsed and evaluated in a persistent environment so that
 * variables and functions defined in earlier lines remain visible.  Readline
 * is used for line editing and history when available.
 *
 * Multi-line input: if the line has unbalanced (, [ or { the REPL keeps
 * reading (with a continuation prompt) until the brackets are balanced.
 *
 * Hot-reload commands:
 *   :load <file.nano>   -- load and eval a .nano file into the current session
 *   :save <file.nano>   -- save all source fragments evaluated this session
 *   :reload <file.nano> -- recompile file and hot-patch changed function
 *                          bindings in the running environment (no restart)
 */

#define _POSIX_C_SOURCE 200809L

#include "nanolang.h"
#include "interpreter_ffi.h"
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_READLINE
/* readline's deprecated 'Function' typedef clashes with nanolang's Function type */
#  define Function RL_Function_deprecated
#  include <readline/readline.h>
#  include <readline/history.h>
#  undef Function
#endif

/* --- bracket balance helper ----------------------------------------------- */

/* Returns >0 if more input is needed (unbalanced open), 0 if balanced. */
static int bracket_balance(const char *src) {
    int depth = 0;
    bool in_string = false;
    char str_delim = 0;
    for (const char *p = src; *p; p++) {
        if (in_string) {
            if (*p == '\\') { p++; continue; }
            if (*p == str_delim) in_string = false;
            continue;
        }
        if (*p == '"' || *p == '\'') { in_string = true; str_delim = *p; continue; }
        if (*p == '(' || *p == '[' || *p == '{') depth++;
        if (*p == ')' || *p == ']' || *p == '}') depth--;
    }
    return depth > 0 ? depth : 0;
}

/* --- line reading ---------------------------------------------------------- */

static char *read_line_repl(const char *prompt) {
#ifdef HAVE_READLINE
    char *line = readline(prompt);
    if (line && *line) add_history(line);
    return line;   /* caller must free() */
#else
    printf("%s", prompt);
    fflush(stdout);
    char buf[4096];
    if (!fgets(buf, sizeof(buf), stdin)) return NULL;
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
    return strdup(buf);
#endif
}

/* Read a (possibly multi-line) input.  Returns heap-allocated source or NULL
 * on EOF. */
static char *read_input(void) {
    char *line = read_line_repl("nano> ");
    if (!line) return NULL;

    char *src = line;
    while (bracket_balance(src) > 0) {
        char *cont = read_line_repl("...   ");
        if (!cont) break;
        size_t old_len = strlen(src);
        size_t add_len = strlen(cont);
        char *next = realloc(src, old_len + add_len + 2);
        if (!next) { free(cont); break; }
        src = next;
        src[old_len] = '\n';
        memcpy(src + old_len + 1, cont, add_len + 1);
        free(cont);
    }
    return src;
}

/* --- value display --------------------------------------------------------- */

static void show_result(Value val) {
    if (val.type == VAL_VOID) return;
    printf("=> ");
    repl_print_value(val);
    printf("\n");
}

/* --- help / banner --------------------------------------------------------- */

static void print_banner(void) {
    printf("nanolang REPL  (type :help for commands, Ctrl-D to exit)\n");
}

static void print_help(void) {
    printf(":help              -- show this message\n");
    printf(":quit              -- exit the REPL\n");
    printf(":env               -- list bound variable names\n");
    printf(":clear             -- clear variable bindings (functions are retained)\n");
    printf(":load <file.nano>  -- load and evaluate a .nano file into this session\n");
    printf(":save <file.nano>  -- save all source typed/loaded this session to a file\n");
    printf(":reload <file.nano>-- recompile file and hot-patch changed function bindings\n");
    printf("\n");
    printf("Any nanolang expression or statement is evaluated immediately.\n");
    printf("Variables and functions persist across lines.\n");
    printf("Function calls use prefix syntax: (func arg1 arg2)\n");
}

/* --- AST lifetime management ---------------------------------------------- */
/*
 * Function bodies registered in the environment reference nodes inside the
 * program ASTs.  We must keep every program AST alive as long as the
 * environment is alive, then free them all at session end.
 */
#define REPL_MAX_PROGRAMS 4096

static ASTNode *g_repl_programs[REPL_MAX_PROGRAMS];
static char    *g_repl_sources[REPL_MAX_PROGRAMS];
static int      g_repl_program_count = 0;

static void repl_track(ASTNode *prog, char *src) {
    if (g_repl_program_count < REPL_MAX_PROGRAMS) {
        g_repl_programs[g_repl_program_count] = prog;
        g_repl_sources[g_repl_program_count]  = src;
        g_repl_program_count++;
    }
    /* Beyond the limit: AST leaks for the session -- acceptable in a REPL. */
}

static void repl_free_all(void) {
    for (int i = 0; i < g_repl_program_count; i++) {
        free_ast(g_repl_programs[i]);
        free(g_repl_sources[i]);
    }
    g_repl_program_count = 0;
}

/* ── Source history for :save ──────────────────────────────────────────────
 *
 * We keep a parallel list of source strings that were typed interactively
 * (not loaded from files — those have their own source on disk).  This lets
 * :save write a file that can be :load-ed in a new session to restore state.
 */
#define REPL_MAX_HISTORY 4096
static char *g_repl_history[REPL_MAX_HISTORY]; /* raw source strings, heap-alloc */
static int   g_repl_history_count = 0;

static void repl_history_append(const char *src) {
    if (g_repl_history_count < REPL_MAX_HISTORY) {
        g_repl_history[g_repl_history_count++] = strdup(src);
    }
}

static void repl_history_free(void) {
    for (int i = 0; i < g_repl_history_count; i++) free(g_repl_history[i]);
    g_repl_history_count = 0;
}

/* ── Read entire file into heap-allocated string ───────────────────────── */

static char *read_file_contents(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

/* ── Shared: parse + eval file source into an env ──────────────────────── */

static int eval_source_into_env(const char *path, const char *src, Environment *env,
                                 bool adopt_ast) {
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    if (!tokens) {
        fprintf(stderr, "error: lexing '%s' failed\n", path);
        return -1;
    }
    ASTNode *program = parse_repl_input(tokens, token_count);
    free_tokens(tokens, token_count);
    if (!program) {
        fprintf(stderr, "error: parsing '%s' failed\n", path);
        return -1;
    }
    if (program->type != AST_PROGRAM || program->as.program.count == 0) {
        free_ast(program);
        return 0;
    }
    if (adopt_ast) {
        repl_track(program, (char *)src); /* caller must pass heap-alloc src */
    }
    int count = program->as.program.count;
    for (int i = 0; i < count; i++) {
        Value result = repl_eval_node(program->as.program.items[i], env);
        show_result(result);
    }
    if (!adopt_ast) free_ast(program);
    return count;
}

/* ── :load <file> — eval file contents in current env ─────────────────── */

static void cmd_load(const char *path, Environment *env) {
    char *src = read_file_contents(path);
    if (!src) { fprintf(stderr, "error: cannot open '%s'\n", path); return; }
    typecheck_set_current_file(path);
    int n = eval_source_into_env(path, src, env, /*adopt_ast=*/true);
    typecheck_set_current_file("<repl>");
    if (n > 0) printf("; loaded %d definition(s) from '%s'\n", n, path);
    else if (n == 0) printf("(empty file '%s' — nothing loaded)\n", path);
    /* src ownership transferred to repl_track on success; on error n==-1 and src leaked
     * (acceptable — error path is rare). */
}

/* ── :save <file> — write session source history to file ──────────────── */

static void cmd_save(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "error: cannot write to '%s'\n", path); return; }
    int written = 0;
    for (int i = 0; i < g_repl_history_count; i++) {
        if (g_repl_history[i] && *g_repl_history[i]) {
            fprintf(f, "%s\n", g_repl_history[i]);
            written++;
        }
    }
    fclose(f);
    printf("; saved %d fragment(s) to '%s'\n", written, path);
}

/* ── :reload <file> — hot-patch changed function bindings ─────────────── *
 *
 * Algorithm:
 *   1. Parse + eval the file into a temporary environment (parent = live env,
 *      so cross-references resolve correctly).
 *   2. For each function defined in the temp env that also exists in the live
 *      env, overwrite the live binding with the new one (hot-patch in-place).
 *   3. New functions (not previously defined) are added to the live env.
 *   4. The parsed AST is adopted into g_repl_programs so function bodies
 *      that reference AST nodes remain valid after tmp env is freed.
 *
 * Variable state is preserved across reload — only function bindings change.
 */

static void cmd_reload(const char *path, Environment *env) {
    char *src = read_file_contents(path);
    if (!src) { fprintf(stderr, "error: cannot open '%s'\n", path); return; }

    typecheck_set_current_file(path);
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    if (!tokens) {
        fprintf(stderr, "error: lexing '%s' failed\n", path);
        free(src); return;
    }
    ASTNode *program = parse_repl_input(tokens, token_count);
    free_tokens(tokens, token_count);
    if (!program) {
        fprintf(stderr, "error: parsing '%s' failed\n", path);
        free(src); return;
    }
    if (program->type != AST_PROGRAM || program->as.program.count == 0) {
        free_ast(program); free(src);
        printf("(empty file '%s' — nothing reloaded)\n", path);
        return;
    }

    /* Temporary env with live env as parent so cross-refs resolve */
    Environment *tmp = create_environment();
    tmp->parent = env;

    /* Adopt AST + src before eval so function bodies stay live */
    repl_track(program, src);

    for (int i = 0; i < program->as.program.count; i++) {
        repl_eval_node(program->as.program.items[i], tmp);
    }

    /* Hot-patch: walk tmp function table, apply to live env */
    int patched = 0, added = 0;
    for (int fi = 0; fi < tmp->function_count; fi++) {
        Function *new_fn = &tmp->functions[fi];
        if (!new_fn->name) continue;

        bool found = false;
        for (int li = 0; li < env->function_count; li++) {
            if (env->functions[li].name &&
                strcmp(env->functions[li].name, new_fn->name) == 0) {
                env->functions[li] = *new_fn;
                patched++;
                found = true;
                break;
            }
        }
        if (!found) {
            env_define_function(env, *new_fn);
            added++;
        }
    }

    tmp->parent = NULL; /* unlink before free to avoid double-free of shared nodes */
    free_environment(tmp);
    typecheck_set_current_file("<repl>");
    printf("; reload '%s': %d function(s) hot-patched, %d new\n", path, patched, added);
}

/* --- main loop ------------------------------------------------------------ */

int run_repl(void) {
    print_banner();
    g_repl_program_count = 0;
    g_mod_count = 0;

    clear_module_cache();
    Environment *env = create_environment();
    (void)ffi_init(false);
    typecheck_set_current_file("<repl>");

    bool running = true;
    while (running) {
        char *src = read_input();
        if (!src) {
            printf("\n");
            break;
        }

        /* Trim whitespace in-place for command detection */
        char *trimmed = src;
        while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
        size_t tlen = strlen(trimmed);
        while (tlen > 0 && (trimmed[tlen-1] == ' '  || trimmed[tlen-1] == '\t' ||
                            trimmed[tlen-1] == '\r'  || trimmed[tlen-1] == '\n')) {
            trimmed[--tlen] = '\0';
        }

        if (tlen == 0) { free(src); continue; }

        /* Meta-commands */
        if (strcmp(trimmed, ":quit") == 0 || strcmp(trimmed, ":exit") == 0 ||
            strcmp(trimmed, ":q")   == 0) {
            running = false;
            free(src);
            continue;
        }
        if (strcmp(trimmed, ":help") == 0 || strcmp(trimmed, ":h") == 0 ||
            strcmp(trimmed, ":?")   == 0) {
            print_help();
            free(src);
            continue;
        }
        if (strcmp(trimmed, ":env") == 0) {
            printf("Bound variables:\n");
            for (int i = 0; i < env->symbol_count; i++)
                printf("  %s\n", env->symbols[i].name);
            free(src);
            continue;
        }
        if (strcmp(trimmed, ":clear") == 0) {
            env->symbol_count = 0;
            printf("(bindings cleared)\n");
            free(src);
            continue;
        }
        if (strcmp(trimmed, ":modules") == 0) {
            cmd_modules(env);
            free(src);
            continue;
        }
        /* :reload <path> */
        if (strncmp(trimmed, ":reload", 7) == 0 &&
            (trimmed[7] == ' ' || trimmed[7] == '\t')) {
            const char *path = trimmed + 8;
            while (*path == ' ' || *path == '\t') path++;
            if (*path == '\0') {
                fprintf(stderr, "Usage: :reload <file.nano>\n");
            } else {
                /* [wasm] note: WASM hot-reload is not supported; use interpreter mode */
                cmd_reload(path, env);
            }
            free(src);
            continue;
        }

        /* :load <file.nano> */
        if (strncmp(trimmed, ":load ", 6) == 0) {
            const char *path = trimmed + 6;
            while (*path == ' ' || *path == '\t') path++;
            cmd_load(path, env);
            free(src);
            continue;
        }

        /* :save <file.nano> */
        if (strncmp(trimmed, ":save ", 6) == 0) {
            const char *path = trimmed + 6;
            while (*path == ' ' || *path == '\t') path++;
            cmd_save(path);
            free(src);
            continue;
        }

        /* :reload <file.nano> */
        if (strncmp(trimmed, ":reload ", 8) == 0) {
            const char *path = trimmed + 8;
            while (*path == ' ' || *path == '\t') path++;
            cmd_reload(path, env);
            free(src);
            continue;
        }

        /* Append interactive input to session history (for :save) */
        repl_history_append(trimmed);

        /* Lex */
        int token_count = 0;
        Token *tokens = tokenize(src, &token_count);
        if (!tokens) {
            fprintf(stderr, "error: lexing failed\n");
            free(src);
            continue;
        }

        /* Parse (REPL mode: accepts any statement, not just top-level decls) */
        ASTNode *program = parse_repl_input(tokens, token_count);
        free_tokens(tokens, token_count);  /* tokens not needed after parse */
        if (!program) {
            fprintf(stderr, "error: parse failed\n");
            free(src);
            continue;
        }

        if (program->type != AST_PROGRAM || program->as.program.count == 0) {
            free_ast(program);
            free(src);
            continue;
        }

        /* Keep the AST and source alive: function bodies registered in the env
         * reference nodes inside this program -- freeing early causes crashes. */
        repl_track(program, src);

        /* Process any import statements in the input */
        ModuleList *mods = create_module_list();
        process_imports(program, env, mods, "<repl>");
        free_module_list(mods);

        /* Typecheck: registers function/struct/enum definitions in env */
        type_check_module(program, env);

        /* Evaluate each top-level node in the persistent environment */
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *node = program->as.program.items[i];
            /* Imports already handled above; function defs registered by typecheck */
            if (node->type == AST_IMPORT) continue;
            Value result = repl_eval_node(node, env);
            show_result(result);
        }
    }

    free_environment(env);
    repl_free_all();
    repl_history_free();
    clear_module_cache();
    return 0;
}
