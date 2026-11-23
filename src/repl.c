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
            if (*p == '\\') { if (p[1]) p++; continue; }
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

/* I consume only the callable metadata owned by an expression result. */
static void retire_callable_result(Value value) {
    if (value.type != VAL_FUNCTION) return;
    free(value.as.function_val.function_name);
    free_function_signature(value.as.function_val.signature);
}

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
    printf(":modules           -- list loaded modules\n");
    printf(":clear             -- clear variable bindings (functions are retained)\n");
    printf(":load <file.nano>  -- load and evaluate a .nano file into this session\n");
    printf(":save <file.nano>  -- save all source typed/loaded this session to a file\n");
    printf(":reload <file.nano>-- recompile file and hot-patch changed function bindings\n");
    printf("I preserve variables and signatures on :reload; top-level effects are not replayed.\n");
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
static char    *g_repl_paths[REPL_MAX_PROGRAMS];
static int      g_repl_program_count = 0;

static bool repl_track(ASTNode *prog, char *src, const char *path) {
    if (g_repl_program_count == REPL_MAX_PROGRAMS) return false;
    char *owned = strdup(path);
    if (!owned) return false;
    g_repl_programs[g_repl_program_count] = prog;
    g_repl_sources[g_repl_program_count] = src;
    g_repl_paths[g_repl_program_count++] = owned;
    return true;
}

static void repl_free_all(void) {
    for (int i = 0; i < g_repl_program_count; i++) {
        free_ast(g_repl_programs[i]);
        free(g_repl_sources[i]);
        free(g_repl_paths[i]);
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

/* I keep checker placeholders out of the live variable state between inputs. */
static bool check_source(ASTNode *program, Environment *env) {
    int symbols = env->symbol_count, functions = env->function_count;
    Symbol *saved = symbols ? calloc((size_t)symbols, sizeof *saved) : NULL;
    Function *prior = functions ? malloc((size_t)functions * sizeof *prior) : NULL;
    if ((symbols && !saved) || (functions && !prior)) { free(saved); free(prior); return false; }
    if (functions) memcpy(prior, env->functions, (size_t)functions * sizeof *prior);
    int copied = 0;
    for (; copied < symbols; copied++) {
        saved[copied] = env->symbols[copied];
        saved[copied].name = env->symbols[copied].name ? strdup(env->symbols[copied].name) : NULL;
        saved[copied].struct_type_name = env->symbols[copied].struct_type_name ? strdup(env->symbols[copied].struct_type_name) : NULL;
        if ((env->symbols[copied].name && !saved[copied].name) ||
            (env->symbols[copied].struct_type_name && !saved[copied].struct_type_name)) break;
    }
    if (copied != symbols) {
        for (int i = 0; i <= copied; i++) { free(saved[i].name); free(saved[i].struct_type_name); }
        free(saved); free(prior); return false;
    }
    bool ok = type_check_repl(program, env);
    if (ok) for (int i = 0; i < symbols; i++) {
        saved[i].is_used = env->symbols[i].is_used;
        saved[i].resource_state = env->symbols[i].resource_state;
    }
    env_symbol_index_invalidate(env);
    for (int i = 0; i < env->symbol_count; i++) {
        free(env->symbols[i].name); free(env->symbols[i].struct_type_name);
    }
    if (symbols) memcpy(env->symbols, saved, (size_t)symbols * sizeof *saved);
    env->symbol_count = symbols;
    free(saved);
    if (!ok) {
        if (functions) memcpy(env->functions, prior, (size_t)functions * sizeof *prior);
        /* I retain valid generated helper caches, but publish no failed body. */
        for (int i = functions; i < env->function_count; i++)
            if (env->functions[i].body || env->functions[i].source_file == env_current_file(env))
                env->functions[i].name = NULL;
    }
    free(prior);
    return ok;
}

/* I own every parsed input before a checker can retain any of its annotations. */
static ASTNode *parse_source(const char *path, char *src) {
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    if (!tokens) { fprintf(stderr, "I cannot lex '%s'.\n", path); free(src); return NULL; }
    ASTNode *program = parse_repl_input(tokens, token_count);
    free_tokens(tokens, token_count);
    if (!program) { fprintf(stderr, "I cannot parse '%s'.\n", path); free(src); return NULL; }
    if (program->type != AST_PROGRAM || !program->as.program.count) {
        free_ast(program); free(src); return NULL;
    }
    if (!repl_track(program, src, path)) {
        fprintf(stderr, "I cannot retain another REPL input in this session.\n");
        free_ast(program); free(src); return NULL;
    }
    return program;
}

static void evaluate_source(ASTNode *program, Environment *env) {
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *node = program->as.program.items[i];
        if (node->type == AST_IMPORT || node->type == AST_MODULE_DECL ||
            node->type == AST_FUNCTION || node->type == AST_SHADOW ||
            node->type == AST_STRUCT_DEF || node->type == AST_ENUM_DEF ||
            node->type == AST_UNION_DEF || node->type == AST_EFFECT_DECL ||
            node->type == AST_OPAQUE_TYPE) continue;
        Value result = repl_eval_node(node, env);
        show_result(result); retire_callable_result(result);
    }
}

static int eval_source_into_env(const char *path, char *src, Environment *env) {
    ASTNode *program = parse_source(path, src);
    if (!program) return -1;
    const char *owner = g_repl_paths[g_repl_program_count - 1];
    env_set_current_file(env, owner); typecheck_set_current_file(owner);
    ModuleList *modules = create_module_list();
    bool ok = process_imports(program, env, modules, owner) && check_source(program, env);
    free_module_list(modules);
    if (ok) evaluate_source(program, env);
    env_set_current_file(env, "<repl>"); typecheck_set_current_file("<repl>");
    return ok ? program->as.program.count : -1;
}

static void cmd_load(const char *path, Environment *env) {
    char *src = read_file_contents(path);
    if (!src) { fprintf(stderr, "I cannot open '%s'.\n", path); return; }
    int n = eval_source_into_env(path, src, env);
    if (n > 0) printf("; I loaded %d item(s) from '%s'.\n", n, path);
}

static void cmd_modules(Environment *env) {
    printf("Modules (%d):\n", env->module_count);
    for (int i = 0; i < env->module_count; i++) {
        ModuleInfo *module = &env->modules[i];
        printf("  %s%s%s\n", module->name, module->path ? " — " : "", module->path ? module->path : "");
    }
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

/* I replace checked function slots without replaying top-level state changes. */
static bool same_text(const char *a, const char *b) {
    return (!a && !b) || (a && b && !strcmp(a,b));
}
static bool same_info(const TypeInfo *a, const TypeInfo *b) {
    return (!a && !b) || (a && b && type_infos_equal(a,b));
}
static bool same_signature(FunctionSignature *a, FunctionSignature *b) {
    return (!a && !b) || (a && b && function_signatures_equal(a,b));
}
static bool compatible_function(const Function *a, const Function *b) {
    if (a->is_extern || b->is_extern || a->param_count != b->param_count ||
        a->return_type != b->return_type || a->return_element_type != b->return_element_type ||
        !same_text(a->return_struct_type_name,b->return_struct_type_name) ||
        !same_info(a->return_type_info,b->return_type_info) ||
        !same_signature(a->return_fn_sig,b->return_fn_sig) ||
        a->is_pure != b->is_pure || a->is_async != b->is_async ||
        a->is_gpu != b->is_gpu || a->is_pub != b->is_pub || a->effect_count != b->effect_count) return false;
    for (int i = 0; i < a->effect_count; i++)
        if (!same_text(a->effect_names[i],b->effect_names[i])) return false;
    for (int i = 0; i < a->param_count; i++) {
        Parameter *x = &a->params[i], *y = &b->params[i];
        if (x->type != y->type || x->element_type != y->element_type ||
            !same_text(x->struct_type_name,y->struct_type_name) ||
            !same_info(x->type_info,y->type_info) || !same_signature(x->fn_sig,y->fn_sig)) return false;
    }
    return true;
}
static ASTNode *function_node(ASTNode *node) {
    return node->type == AST_ASYNC_FN ? node->as.async_fn.function : node;
}
static void cmd_reload(const char *path, Environment *env) {
    char *src = read_file_contents(path);
    if (!src) { fprintf(stderr, "I cannot open '%s'.\n", path); return; }
    ASTNode *program = parse_source(path, src);
    if (!program) return;
    const char *owner = g_repl_paths[g_repl_program_count - 1];
    char *saved_module = env->current_module;
    for (int i = 0; i < program->as.program.count; i++)
        if (program->as.program.items[i]->type == AST_MODULE_DECL)
            env->current_module = program->as.program.items[i]->as.module_decl.name;
    env_set_current_file(env, owner); typecheck_set_current_file(owner);
    ASTNode selected = {0}; selected.type = AST_PROGRAM;
    selected.as.program.items = calloc((size_t)program->as.program.count, sizeof(ASTNode *));
    if (!selected.as.program.items) goto restore_context;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i], *fn = function_node(item);
        if ((fn->type == AST_FUNCTION && !fn->as.function.is_extern) || item->type == AST_SHADOW || item->type == AST_IMPORT)
            selected.as.program.items[selected.as.program.count++] = item;
    }
    ModuleList *modules = create_module_list();
    bool imported = process_imports(&selected, env, modules, owner);
    free_module_list(modules);
    if (!imported) { free(selected.as.program.items); goto restore_context; }
    int before = env->function_count, module_count = env->module_count;
    Function *saved = before ? malloc((size_t)before * sizeof *saved) : NULL;
    int *exports = module_count ? malloc((size_t)module_count * sizeof *exports) : NULL;
    if ((before && !saved) || (module_count && !exports)) {
        free(saved); free(exports); free(selected.as.program.items); goto restore_context;
    }
    if (before) memcpy(saved,env->functions,(size_t)before*sizeof *saved);
    for (int i = 0; i < module_count; i++) exports[i] = env->modules[i].function_count;
    for (int i = 0; i < selected.as.program.count; i++) {
        ASTNode *fn = function_node(selected.as.program.items[i]);
        if (fn->type != AST_FUNCTION) continue;
        Function *old = env_get_function_owned(env,fn->as.function.name,env->current_module);
        if (old && !old->is_extern) old->name = NULL;
    }
    bool ok = check_source(&selected, env);
    for (int i = 0; ok && i < selected.as.program.count; i++) {
        ASTNode *item = selected.as.program.items[i], *fn = function_node(item);
        if (fn->type != AST_FUNCTION) continue;
        Function *replacement = NULL;
        for (int j = before; j < env->function_count; j++)
            if (env->functions[j].body == fn->as.function.body && env->functions[j].name)
                replacement = &env->functions[j];
        if (!replacement) { ok = false; break; }
        replacement->is_async = item->type == AST_ASYNC_FN;
        for (int j = 0; j < before; j++)
            if (same_text(saved[j].name,fn->as.function.name) && same_text(saved[j].module_name,env->current_module) &&
                !compatible_function(&saved[j],replacement)) {
                fprintf(stderr,"I retain the existing signature of '%s' when reloading.\n",saved[j].name);
                ok = false; break;
            }
    }
    if (before) memcpy(env->functions,saved,(size_t)before*sizeof *saved);
    int patched = 0, added = 0;
    if (ok) {
        for (int j = before; j < env->function_count; j++) {
            Function fresh = env->functions[j];
            if (!fresh.body || fresh.source_file != owner || !fresh.name) continue;
            bool found = false;
            for (int k = 0; k < before; k++)
                if (same_text(saved[k].name,fresh.name) && same_text(saved[k].module_name,fresh.module_name)) {
                    env->functions[k] = fresh; env->functions[j].name = NULL;
                    patched++; found = true; break;
                }
            if (!found) added++;
        }
        printf("; reload '%s': %d function(s) hot-patched, %d new\n",path,patched,added);
    } else {
        for (int j = before; j < env->function_count; j++)
            if (env->functions[j].body || env->functions[j].source_file == owner) env->functions[j].name = NULL;
        for (int i = 0; i < module_count; i++)
            while (env->modules[i].function_count > exports[i])
                free(env->modules[i].exported_functions[--env->modules[i].function_count]);
        fprintf(stderr,"I retain the previous functions after this reload refusal.\n");
    }
    free(saved); free(exports); free(selected.as.program.items);
restore_context:
    env->current_module = saved_module;
    env_set_current_file(env,"<repl>"); typecheck_set_current_file("<repl>");
}

/* --- main loop ------------------------------------------------------------ */

int run_repl(void) {
    print_banner();
    g_repl_program_count = 0;
    typecheck_set_current_file("<repl>");

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
            env_clear_variables(env);
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

        (void)eval_source_into_env("<repl>", src, env);
    }

    free_environment(env);
    repl_free_all();
    repl_history_free();
    clear_module_cache();
    return 0;
}
