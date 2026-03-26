/* nano_main.c — Tree-walking interpreter entry point for nanolang
 *
 * Usage: nano script.nano [args...]
 *
 * Runs nanolang programs directly via the eval.c interpreter, skipping the
 * C transpiler.  The pipeline is: lex → parse → import → typecheck →
 * run_program (registers top-level items) → call main().
 */

#define _POSIX_C_SOURCE 200809L

#include "nanolang.h"
#include "module_builder.h"
#include "interpreter_ffi.h"
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

/* Runtime access for interpreted programs that call argc()/argv() */
int g_argc = 0;
char **g_argv = NULL;

/* Project root (resolved from binary location so module loading works) */
char g_project_root[PATH_MAX] = "";

const char *get_project_root(void) {
    return g_project_root[0] ? g_project_root : ".";
}

static void resolve_project_root(const char *argv0) {
    char exe_path[PATH_MAX];
    if (realpath(argv0, exe_path) == NULL) {
        if (getcwd(g_project_root, sizeof(g_project_root)) == NULL)
            strcpy(g_project_root, ".");
        return;
    }
    char *slash = strrchr(exe_path, '/');
    if (slash) {
        *slash = '\0';
        slash = strrchr(exe_path, '/');
        if (slash) *slash = '\0';
    }
    strncpy(g_project_root, exe_path, sizeof(g_project_root) - 1);
    g_project_root[sizeof(g_project_root) - 1] = '\0';
}

/* Run a nanolang source file through the interpreter */
static int interpret_file(const char *input_file) {
    /* Read source */
    FILE *file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "nano: cannot open '%s': %s\n", input_file, strerror(errno));
        return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *source = malloc((size_t)size + 1);
    if (!source) { fclose(file); fprintf(stderr, "nano: out of memory\n"); return 1; }
    fread(source, 1, (size_t)size, file);
    source[size] = '\0';
    fclose(file);

    /* Phase 1: Lex */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "nano: lexing failed\n");
        free(source);
        return 1;
    }

    /* Phase 2: Parse */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "nano: parsing failed\n");
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Phase 3: Create environment and load imports */
    clear_module_cache();
    Environment *env = create_environment();
    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, input_file)) {
        fprintf(stderr, "nano: module loading failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        return 1;
    }

    /* Phase 4: Type check (also registers all functions/structs/enums/unions) */
    typecheck_set_current_file(input_file);
    if (!type_check(program, env)) {
        fprintf(stderr, "nano: type checking failed\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        return 1;
    }

    /* Phase 4.5: Initialize FFI and load module shared libraries */
    (void)ffi_init(false);
    for (int i = 0; i < modules->count; i++) {
        const char *module_path = modules->module_paths[i];
        char *module_dir = strdup(module_path);
        char *last_slash = strrchr(module_dir, '/');
        if (last_slash) {
            *last_slash = '\0';
        } else {
            free(module_dir);
            module_dir = strdup(".");
        }
        ModuleBuildMetadata *meta = module_load_metadata(module_dir);
        char mod_name[256];
        if (meta && meta->name) {
            snprintf(mod_name, sizeof(mod_name), "%s", meta->name);
        } else {
            const char *base = last_slash ? (last_slash + 1) : module_path;
            snprintf(mod_name, sizeof(mod_name), "%s", base);
            char *dot = strrchr(mod_name, '.');
            if (dot) *dot = '\0';
        }
        (void)ffi_load_module(mod_name, module_path, env, false);
        if (meta) module_metadata_free(meta);
        free(module_dir);
    }

    /* Phase 5: Run program (evaluates top-level lets, registers structs/enums/unions) */
    if (!run_program(program, env)) {
        fprintf(stderr, "nano: runtime error\n");
        free_ast(program);
        free_tokens(tokens, token_count);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free(source);
        return 1;
    }

    /* Phase 6: Call main() */
    Function *main_fn = env_get_function(env, "main");
    int exit_code = 0;
    if (main_fn) {
        Value result = call_function("main", NULL, 0, env);
        if (result.type == VAL_INT) {
            exit_code = (int)result.as.int_val;
        }
    }
    /* If no main function, the program ran as a script and we're done */

    /* Cleanup */
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    clear_module_cache();
    free(source);

    return exit_code;
}

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = argv;

    resolve_project_root(argv[0]);

    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        printf("nano - Nanolang tree-walking interpreter\n\n");
        printf("Usage: %s <script.nano> [args...]\n\n", argv[0]);
        printf("Runs a nanolang program directly via the interpreter,\n");
        printf("without compiling to C.  Use 'nanoc' to compile instead.\n\n");
        printf("Options:\n");
        printf("  --help, -h    Show this help message\n");
        printf("  --version     Show version\n");
        return argc < 2 ? 1 : 0;
    }

    if (strcmp(argv[1], "--version") == 0) {
        printf("nano (nanolang interpreter)\n");
        return 0;
    }

    return interpret_file(argv[1]);
}
