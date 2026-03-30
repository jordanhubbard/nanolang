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
 */

#define _POSIX_C_SOURCE 200809L

#include "nanolang.h"
#include "interpreter_ffi.h"
#include <limits.h>
#include <string.h>
#include <unistd.h>

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
    printf(":help   -- show this message\n");
    printf(":quit   -- exit the REPL\n");
    printf(":env    -- list bound variable names\n");
    printf(":clear  -- clear variable bindings (functions are retained)\n");
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

/* --- main loop ------------------------------------------------------------ */

int run_repl(void) {
    print_banner();
    g_repl_program_count = 0;

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

        /* Evaluate each top-level node in the persistent environment */
        for (int i = 0; i < program->as.program.count; i++) {
            Value result = repl_eval_node(program->as.program.items[i], env);
            show_result(result);
        }
    }

    free_environment(env);
    repl_free_all();
    clear_module_cache();
    return 0;
}
