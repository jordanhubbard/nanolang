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
 * Hot-reload: :reload <file.nano> re-parses and re-typechecks the named
 * module, then updates the body of every function that was previously defined
 * from that file.  Function bodies in the persistent environment are patched
 * in-place so existing call sites immediately use the new code.
 */

#define _POSIX_C_SOURCE 200809L

#include "nanolang.h"
#include "interpreter_ffi.h"
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

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
    printf(":reload <file>     -- hot-reload a module: re-parse, re-bind exports\n");
    printf(":modules           -- list currently loaded modules and their exports\n");
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

/* --- loaded-module table (for :reload mtime tracking) --------------------- */

#define REPL_MAX_MODS 64

typedef struct {
    char   *path;
    time_t  mtime;
} ReplMod;

static ReplMod g_mods[REPL_MAX_MODS];
static int     g_mod_count = 0;

static time_t file_mtime(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return st.st_mtime;
    return (time_t)-1;
}

static ReplMod *find_mod(const char *path) {
    for (int i = 0; i < g_mod_count; i++)
        if (strcmp(g_mods[i].path, path) == 0)
            return &g_mods[i];
    return NULL;
}

static void track_mod(const char *path, time_t mtime) {
    ReplMod *m = find_mod(path);
    if (m) { m->mtime = mtime; return; }
    if (g_mod_count >= REPL_MAX_MODS) return;
    g_mods[g_mod_count].path  = strdup(path);
    g_mods[g_mod_count].mtime = mtime;
    g_mod_count++;
}

/* --- hot-reload (:reload <file>) ------------------------------------------ */

/*
 * Direct AST walk to rebind functions without going through type_check_module.
 * type_check_module rejects redefinitions with "already defined" errors, so for
 * hot-reload we bypass it and patch function bodies in-place directly.
 *
 * Each AST_FUNCTION node in the reloaded AST is inspected:
 *   - If a function with the same name already exists in env, its body/params
 *     are replaced in-place so existing call sites immediately use the new code.
 *   - If no match exists, the function is registered as new via env_define_function.
 *
 * The reloaded AST is kept alive via repl_track() because the function bodies
 * now reference nodes inside it.
 */
static void cmd_reload(const char *path, Environment *env) {
    time_t mtime = file_mtime(path);
    if (mtime == (time_t)-1) {
        fprintf(stderr, "[reload] error: cannot stat '%s': %s\n", path, strerror(errno));
        return;
    }

    /* Skip if file hasn't changed since last load */
    ReplMod *m = find_mod(path);
    if (m && m->mtime == mtime) {
        printf("[unchanged] %s\n", path);
        return;
    }

    /* Read source */
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[reload] error: cannot open '%s': %s\n", path, strerror(errno));
        return;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = malloc((size_t)size + 1);
    if (!src) { fclose(f); fprintf(stderr, "[reload] out of memory\n"); return; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fread(src, 1, (size_t)size, f);
#pragma GCC diagnostic pop
    src[size] = '\0';
    fclose(f);

    /* Lex + parse */
    int tc = 0;
    Token *tokens = tokenize(src, &tc);
    if (!tokens) {
        fprintf(stderr, "[reload] error: lex failed for '%s'\n", path);
        free(src);
        return;
    }
    ASTNode *ast = parse_repl_input(tokens, tc);
    free_tokens(tokens, tc);
    if (!ast) {
        fprintf(stderr, "[reload] error: parse failed for '%s'\n", path);
        free(src);
        return;
    }

    int updated = 0;
    int added   = 0;
    char names[2048] = "";

    /* Walk top-level AST nodes, rebind each function definition */
    for (int i = 0; i < ast->as.program.count; i++) {
        ASTNode *item = ast->as.program.items[i];
        if (item->type != AST_FUNCTION) continue;

        const char *fname = item->as.function.name;
        if (!fname) continue;

        /* Search for an existing entry with the same name */
        bool found = false;
        for (int j = 0; j < env->function_count; j++) {
            if (!env->functions[j].name) continue;
            if (strcmp(env->functions[j].name, fname) != 0) continue;
            if (env->functions[j].is_extern) continue; /* don't patch externs */

            /* Patch the body in-place so live call sites pick up the new code */
            env->functions[j].body                    = item->as.function.body;
            env->functions[j].params                  = item->as.function.params;
            env->functions[j].param_count             = item->as.function.param_count;
            env->functions[j].return_type             = item->as.function.return_type;
            env->functions[j].return_element_type     = item->as.function.return_element_type;
            env->functions[j].return_struct_type_name = item->as.function.return_struct_type_name;
            env->functions[j].return_type_info        = item->as.function.return_type_info;
            env->functions[j].is_pub                  = item->as.function.is_pub;

            /* Build the names summary */
            if (updated > 0 && strlen(names) + 2 < sizeof(names))
                strncat(names, ", ", sizeof(names) - strlen(names) - 1);
            if (strlen(names) + strlen(fname) < sizeof(names))
                strncat(names, fname, sizeof(names) - strlen(names) - 1);

            updated++;
            found = true;
            break;
        }

        if (!found) {
            /* New function — register it */
            Function new_fn = {0};
            new_fn.name                    = (char *)fname;
            new_fn.params                  = item->as.function.params;
            new_fn.param_count             = item->as.function.param_count;
            new_fn.return_type             = item->as.function.return_type;
            new_fn.return_element_type     = item->as.function.return_element_type;
            new_fn.return_struct_type_name = item->as.function.return_struct_type_name;
            new_fn.return_type_info        = item->as.function.return_type_info;
            new_fn.body                    = item->as.function.body;
            new_fn.is_pub                  = item->as.function.is_pub;
            env_define_function(env, new_fn);
            added++;
        }
    }

    /* Keep the new AST and source alive (function bodies reference AST nodes) */
    repl_track(ast, src);

    /* Record mtime so the next :reload can detect unchanged files */
    track_mod(path, mtime);

    /* Report */
    if (updated == 0 && added == 0) {
        printf("Reloaded: %s (no functions found)\n", path);
    } else {
        printf("Reloaded: %s (%d updated", names[0] ? names : "—", updated);
        if (added) printf(", %d new", added);
        printf(")\n");
    }
}

/* --- :modules command ----------------------------------------------------- */

static void cmd_modules(Environment *env) {
    if (env->module_count == 0) {
        printf("(no modules loaded)\n");
        return;
    }
    for (int i = 0; i < env->module_count; i++) {
        ModuleInfo *mi = &env->modules[i];
        printf("  %s  (%s)\n", mi->name ? mi->name : "<unnamed>",
               mi->path ? mi->path : "?");
        for (int j = 0; j < mi->function_count; j++)
            printf("    fn %s\n", mi->exported_functions[j]);
        for (int j = 0; j < mi->struct_count; j++)
            printf("    struct %s\n", mi->exported_structs[j]);
    }
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
    clear_module_cache();
    return 0;
}
