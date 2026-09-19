/**
 * test_affine_c_frontend.c -- Exercise the C frontend without publishing a backend artifact.
 */

#include "../src/nanolang.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by the runtime objects linked into this test-only runner. */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

enum {
    AFFINE_FRONTEND_ACCEPTED = 0,
    AFFINE_FRONTEND_REFUSED = 1,
    AFFINE_FRONTEND_SETUP_FAILURE = 2,
};

static char *read_source(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:read:%s\n", strerror(errno));
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:seek:%s\n", strerror(errno));
        fclose(file);
        return NULL;
    }
    long end = ftell(file);
    if (end < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:size:%s\n", strerror(errno));
        fclose(file);
        return NULL;
    }
    size_t size = (size_t)end;
    char *source = malloc(size + 1);
    if (!source) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:allocation\n");
        fclose(file);
        return NULL;
    }
    if (size > 0 && fread(source, 1, size, file) != size) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:read:%s\n", strerror(errno));
        free(source);
        fclose(file);
        return NULL;
    }
    source[size] = '\0';
    if (fclose(file) != 0) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:close:%s\n", strerror(errno));
        free(source);
        return NULL;
    }
    return source;
}

static void release_frontend(char *source, Token *tokens, int token_count,
                             ASTNode *program, Environment *env, ModuleList *modules) {
    if (program) free_ast(program);
    if (tokens) free_tokens(tokens, token_count);
    if (env) free_environment(env);
    if (modules) free_module_list(modules);
    clear_module_cache();
    free(source);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s SOURCE.nano\n", argv[0]);
        return AFFINE_FRONTEND_SETUP_FAILURE;
    }

    const char *path = argv[1];
    char *source = read_source(path);
    if (!source) return AFFINE_FRONTEND_SETUP_FAILURE;

    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:lexer\n");
        release_frontend(source, NULL, 0, NULL, NULL, NULL);
        return AFFINE_FRONTEND_SETUP_FAILURE;
    }

    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:parser\n");
        release_frontend(source, tokens, token_count, NULL, NULL, NULL);
        return AFFINE_FRONTEND_SETUP_FAILURE;
    }

    clear_module_cache();
    Environment *env = create_environment();
    ModuleList *modules = create_module_list();
    if (!env || !modules) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:environment\n");
        release_frontend(source, tokens, token_count, program, env, modules);
        return AFFINE_FRONTEND_SETUP_FAILURE;
    }
    if (!process_imports(program, env, modules, path)) {
        fprintf(stderr, "AFFINE_C_FRONTEND_FAILED:imports\n");
        release_frontend(source, tokens, token_count, program, env, modules);
        return AFFINE_FRONTEND_SETUP_FAILURE;
    }

    typecheck_set_current_file(path);
    env_set_current_file(env, path);
    bool accepted = type_check(program, env);
    if (accepted) {
        int production_symbols = env->symbol_count;
        accepted = type_check_shadow_scope(program, env, modules, path, true);
        for (int i = production_symbols; i < env->symbol_count; i++) {
            free(env->symbols[i].name);
            free(env->symbols[i].struct_type_name);
        }
        env->symbol_count = production_symbols;
    }

    release_frontend(source, tokens, token_count, program, env, modules);
    if (!accepted) {
        fprintf(stderr, "AFFINE_C_FRONTEND_REFUSED:typecheck\n");
        return AFFINE_FRONTEND_REFUSED;
    }
    puts("AFFINE_C_FRONTEND_ACCEPTED");
    return AFFINE_FRONTEND_ACCEPTED;
}
