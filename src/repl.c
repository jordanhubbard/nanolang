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
#include "eval/carrier_runtime.h"
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
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
    printf(":load <path>       -- evaluate NanoLang source or replay a saved session\n");
    printf(":save <path>       -- save accepted input, load, reload and clear operations\n");
    printf(":reload <file.nano>-- recompile file and hot-patch changed function bindings\n");
    printf("I preserve variables and signatures on :reload; top-level effects are not replayed.\n");
    printf("Saved sessions replay source operations in order with :load.\n");
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

/* I replay accepted source operations in order, including reload and clear. */
#define REPL_MAX_HISTORY 4096
typedef enum { REPL_INPUT, REPL_LOAD, REPL_RELOAD, REPL_CLEAR } ReplOperation;
typedef struct {
    ReplOperation operation;
    char *path, *source;
} ReplFragment;
static ReplFragment g_repl_history[REPL_MAX_HISTORY];
static int   g_repl_history_count = 0;
static const char *session_magic = "# nanolang-repl-session v1\n";
static const char *operation_names[] = {"input", "load", "reload", "clear"};

static void fragment_free(ReplFragment *fragment) {
    free(fragment->path); free(fragment->source);
    fragment->path = fragment->source = NULL;
}

static bool history_prepare(ReplFragment *fragment, ReplOperation operation,
                            const char *path, const char *source, bool retained_origin) {
    memset(fragment, 0, sizeof *fragment); fragment->operation = operation;
    if (g_repl_history_count == REPL_MAX_HISTORY) {
        fprintf(stderr, "I cannot retain another REPL input in this session.\n"); return false;
    }
    if (operation == REPL_CLEAR) return true;
    if (!retained_origin && operation != REPL_INPUT) fragment->path = realpath(path, NULL);
    if (!fragment->path && path[0] == '/') fragment->path = strdup(path);
    if (!fragment->path) {
        char *directory = getcwd(NULL, 0);
        if (directory) {
            size_t a = strlen(directory), b = strlen(path);
            if (b <= SIZE_MAX - 2 && a <= SIZE_MAX - b - 2) {
                fragment->path = malloc(a + b + 2);
                if (fragment->path) snprintf(fragment->path, a + b + 2, "%s/%s", directory, path);
            }
            free(directory);
        }
    }
    fragment->source = strdup(source);
    if (fragment->path && fragment->source) return true;
    fragment_free(fragment);
    fprintf(stderr, "I cannot retain REPL session source.\n"); return false;
}

static void history_finish(ReplFragment *fragment, bool accepted) {
    if (accepted) g_repl_history[g_repl_history_count++] = *fragment;
    else fragment_free(fragment);
}

static void repl_history_free(void) {
    for (int i = 0; i < g_repl_history_count; i++) fragment_free(&g_repl_history[i]);
    g_repl_history_count = 0;
}

/* ── Read entire file into heap-allocated string ───────────────────────── */

static char *read_file_contents(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    if ((uintmax_t)sz >= SIZE_MAX) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    bool ok = got == (size_t)sz && !ferror(f) && !memchr(buf, '\0', got);
    buf[got] = '\0';
    if (fclose(f)) ok = false;
    if (!ok) { free(buf); return NULL; }
    if (bytes) *bytes = got;
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
        char *nominal = node->type == AST_LET && node->as.let.type_name ? strdup(node->as.let.type_name) : NULL;
        if (node->type == AST_LET && node->as.let.type_name && !nominal) {
            fprintf(stderr, "I cannot retain checked REPL binding metadata.\n"); exit(1);
        }
        int before = env->symbol_count;
        /* I keep owned results in the caller until display has finished. */
        env_carrier_enter(env);
        Value result = repl_eval_node(node, env);
        if (node->type == AST_LET && env->symbol_count > before) {
            Symbol *binding = &env->symbols[env->symbol_count - 1];
            free(binding->struct_type_name); binding->struct_type_name = nominal;
        } else free(nominal);
        show_result(result); retire_callable_result(result);
        env_carrier_leave(env);
    }
}

static int eval_source_into_env(const char *path, char *src, Environment *env) {
    ASTNode *program = parse_source(path, src);
    if (!program) return -1;
    const char *owner = g_repl_paths[g_repl_program_count - 1];
    env_set_current_file(env, owner); typecheck_set_current_file(owner);
    const char *logical_owner = env->current_module;
    for (int i = 0; i < program->as.program.count; ++i)
        if (program->as.program.items[i]->type == AST_MODULE_DECL) {
            logical_owner = program->as.program.items[i]->as.module_decl.name;
            break; /* I match the import pass's first explicit owner. */
        }
    char *prior_owner = env->current_module;
    struct EnvModuleCheckpoint *admission = env_module_checkpoint(env, logical_owner, owner);
    if (!admission) {
        env_module_commit(admission);
        fprintf(stderr, "I cannot retain REPL admission state.\n");
        env_set_current_file(env, "<repl>"); typecheck_set_current_file("<repl>");
        return -1;
    }
    ModuleList *modules = create_module_list();
    bool ok = process_imports(program, env, modules, owner) &&
              env_module_checkpoint_declarations(env, admission) && check_source(program, env);
    free_module_list(modules);
    if (ok) {
        env_module_commit(admission); evaluate_source(program, env);
    } else {
        env->current_module = prior_owner;
        env_module_rollback(env, admission);
    }
    env_set_current_file(env, "<repl>"); typecheck_set_current_file("<repl>");
    return ok ? program->as.program.count : -1;
}

static int recorded_source(ReplOperation operation, const char *path, char *source, Environment *env, bool retained_origin) {
    ReplFragment fragment;
    if (!history_prepare(&fragment, operation, path, source, retained_origin)) { free(source); return -1; }
    int result = eval_source_into_env(fragment.path, source, env);
    history_finish(&fragment, result > 0);
    return result;
}

static bool replay_session(const char *source, size_t bytes, Environment *env);

static bool is_session(const char *source) {
    const char *prefix = "# nanolang-repl-session ";
    return !strncmp(source, prefix, strlen(prefix));
}

static void cmd_load(const char *path, Environment *env) {
    size_t bytes = 0;
    char *src = read_file_contents(path, &bytes);
    if (!src) { fprintf(stderr, "I cannot open '%s'.\n", path); return; }
    if (is_session(src)) { (void)replay_session(src, bytes, env); free(src); return; }
    int n = recorded_source(REPL_LOAD, path, src, env, false);
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
    size_t length = strlen(path);
    if (length > SIZE_MAX - 32) { fprintf(stderr, "I cannot name this saved session.\n"); return; }
    char *temporary = malloc(length + 32);
    if (!temporary) { fprintf(stderr, "I cannot retain a save destination.\n"); return; }
    snprintf(temporary, length + 32, "%s.nano-repl-XXXXXX", path);
    int descriptor = mkstemp(temporary);
    if (descriptor < 0) { fprintf(stderr, "I cannot save '%s': %s.\n", path, strerror(errno)); free(temporary); return; }
    FILE *f = fdopen(descriptor, "w");
    if (!f) { close(descriptor); unlink(temporary); free(temporary); fprintf(stderr, "I cannot open my session output stream.\n"); return; }
    int written = 0;
    bool ok = fwrite(session_magic, 1, strlen(session_magic), f) == strlen(session_magic);
    for (int i = 0; i < g_repl_history_count; i++) {
        ReplFragment *fragment = &g_repl_history[i];
        size_t origin = fragment->path ? strlen(fragment->path) : 0;
        size_t source = fragment->source ? strlen(fragment->source) : 0;
        if (fprintf(f, "%s %zu %zu\n", operation_names[fragment->operation], origin, source) < 0 ||
            (origin && fwrite(fragment->path, 1, origin, f) != origin) ||
            (source && fwrite(fragment->source, 1, source, f) != source) || fputc('\n', f) == EOF) ok = false;
        written++;
    }
    int failure = ok ? 0 : (errno ? errno : EIO);
    if (fflush(f) && !failure) failure = errno ? errno : EIO;
    if (fclose(f) && !failure) failure = errno ? errno : EIO;
    if (!failure && rename(temporary, path)) failure = errno;
    if (failure) {
        unlink(temporary); fprintf(stderr, "I cannot save '%s': %s.\n", path, strerror(failure));
    } else printf("; saved %d fragment(s) to '%s'\n", written, path);
    free(temporary);
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
static bool reload_source(const char *path, char *src, Environment *env) {
    ASTNode *program = parse_source(path, src);
    if (!program) return false;
    const char *owner = g_repl_paths[g_repl_program_count - 1];
    char *saved_module = env->current_module;
    struct EnvModuleCheckpoint *admission = NULL;
    bool published = false;
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
    admission = env_module_checkpoint(env, env->current_module, owner);
    if (!admission) { free(selected.as.program.items); goto restore_context; }
    ModuleList *modules = create_module_list();
    bool imported = process_imports(&selected, env, modules, owner);
    free_module_list(modules);
    if (!imported) { free(selected.as.program.items); goto restore_context; }
    if (!env_module_checkpoint_declarations(env, admission)) {
        free(selected.as.program.items); goto restore_context;
    }
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
        published = true;
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
    if (published) env_module_commit(admission);
    else env_module_rollback(env, admission);
    env->current_module = saved_module;
    env_set_current_file(env,"<repl>"); typecheck_set_current_file("<repl>");
    return published;
}

static bool recorded_reload(const char *path, char *source, Environment *env, bool retained_origin) {
    ReplFragment fragment;
    if (!history_prepare(&fragment, REPL_RELOAD, path, source, retained_origin)) { free(source); return false; }
    bool accepted = reload_source(fragment.path, source, env);
    history_finish(&fragment, accepted); return accepted;
}

static void cmd_reload(const char *path, Environment *env) {
    char *source = read_file_contents(path, NULL);
    if (!source) { fprintf(stderr, "I cannot open '%s'.\n", path); return; }
    if (is_session(source)) {
        fprintf(stderr, "I replay a saved session with :load; :reload accepts function source.\n"); free(source); return;
    }
    (void)recorded_reload(path, source, env, false);
}

static bool session_size(const char **cursor, const char *end, size_t *value) {
    const char *p = *cursor; size_t result = 0;
    if (p == end || *p < '0' || *p > '9') return false;
    do {
        unsigned digit = (unsigned)(*p++ - '0');
        if (result > (SIZE_MAX - digit) / 10) return false;
        result = result * 10 + digit;
    } while (p != end && *p >= '0' && *p <= '9');
    *cursor = p; *value = result; return true;
}

static bool replay_session(const char *source, size_t bytes, Environment *env) {
    size_t magic = strlen(session_magic), count = 0, programs = 0;
    ReplFragment *records = calloc(REPL_MAX_HISTORY, sizeof *records);
    if (!records) { fprintf(stderr, "I cannot retain saved session framing.\n"); return false; }
    const char *cursor = source, *end = source + bytes;
    bool valid = bytes >= magic && !memcmp(source, session_magic, magic);
    if (valid) cursor += magic;
    while (valid && cursor != end) {
        if (count == REPL_MAX_HISTORY) { valid = false; break; }
        ReplFragment *record = &records[count++];
        bool operation = false;
        for (unsigned k = 0; k < sizeof operation_names / sizeof *operation_names; ++k) {
            size_t n = strlen(operation_names[k]);
            if ((size_t)(end - cursor) > n && !memcmp(cursor, operation_names[k], n) && cursor[n] == ' ') {
                record->operation = (ReplOperation)k; cursor += n + 1; operation = true; break;
            }
        }
        size_t origin = 0, text = 0;
        if (!operation || !session_size(&cursor, end, &origin) || cursor == end || *cursor++ != ' ' ||
            !session_size(&cursor, end, &text) || cursor == end || *cursor++ != '\n' ||
            origin > (size_t)(end - cursor)) { valid = false; break; }
        record->path = strndup(cursor, origin); cursor += origin;
        if (text > (size_t)(end - cursor)) { valid = false; break; }
        record->source = strndup(cursor, text); cursor += text;
        if (!record->path || !record->source || cursor == end || *cursor++ != '\n') { valid = false; break; }
        if (record->operation == REPL_CLEAR) {
            if (origin || text) valid = false;
        } else {
            ++programs;
            if (!origin || !text || record->path[0] != '/' || is_session(record->source)) valid = false;
        }
    }
    if (count > (size_t)(REPL_MAX_HISTORY - g_repl_history_count) ||
        programs > (size_t)(REPL_MAX_PROGRAMS - g_repl_program_count)) valid = false;
    if (!valid) fprintf(stderr, "I cannot replay this saved session: invalid framing, version or capacity.\n");
    bool framed = valid;
    size_t applied = 0;
    while (valid && applied < count) {
        ReplFragment *record = &records[applied];
        if (record->operation == REPL_CLEAR) {
            ReplFragment fragment;
            valid = history_prepare(&fragment, REPL_CLEAR, NULL, NULL, true);
            if (valid) { env_clear_variables(env); history_finish(&fragment, true); }
        } else {
            char *text = record->source; record->source = NULL;
            valid = record->operation == REPL_RELOAD ? recorded_reload(record->path, text, env, true) :
                    recorded_source(record->operation, record->path, text, env, true) > 0;
        }
        if (valid) ++applied;
    }
    if (valid) printf("; I replayed %zu session fragment(s).\n", applied);
    else if (framed) fprintf(stderr, "I stopped session replay after %zu accepted fragment(s).\n", applied);
    for (size_t i = 0; i < count; ++i) fragment_free(&records[i]);
    free(records); return valid;
}

/* --- main loop ------------------------------------------------------------ */

int run_repl(void) {
    print_banner();
    g_repl_program_count = 0;
    g_repl_history_count = 0;
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
            ReplFragment fragment;
            if (history_prepare(&fragment, REPL_CLEAR, NULL, NULL, false)) {
                env_clear_variables(env); history_finish(&fragment, true);
                printf("(bindings cleared)\n");
            }
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

        (void)recorded_source(REPL_INPUT, "<repl>", src, env, false);
    }

    free_environment(env);
    repl_free_all();
    repl_history_free();
    clear_module_cache();
    return 0;
}
