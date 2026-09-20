/* I expose cache identity observations without replacing loader/clear/compile code. */
#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include <stdlib.h>
#include <string.h>
#ifdef EVALUATOR_ALLOCATION_HOOKS
void *lifetime_alloc_malloc(size_t);
void *lifetime_alloc_calloc(size_t, size_t);
void *lifetime_alloc_realloc(void *, size_t);
char *lifetime_alloc_strdup(const char *);
void lifetime_alloc_free(void *);
#define malloc lifetime_alloc_malloc
#define calloc lifetime_alloc_calloc
#define realloc lifetime_alloc_realloc
#define strdup lifetime_alloc_strdup
#define free lifetime_alloc_free
#endif
static int lifetime_private_fault;
static int lifetime_private_closes;
static FILE *lifetime_private_output;
static Environment *lifetime_private_environment(void) {
    return lifetime_private_fault == 6 ? NULL : create_environment();
}
static char *lifetime_private_transpile(ASTNode *ast, Environment *env, const char *path) {
    return lifetime_private_fault == 1 ? NULL : transpile_to_c(ast, env, path);
}
static int lifetime_private_fputs(const char *text, FILE *stream) {
    lifetime_private_output = stream;
    return lifetime_private_fault == 3 ? EOF : fputs(text, stream);
}
static int lifetime_private_fclose(FILE *stream) {
    bool output = stream == lifetime_private_output;
    int result = fclose(stream);
    if (output) { ++lifetime_private_closes; lifetime_private_output = NULL; }
    return lifetime_private_fault == 4 ? EOF : result;
}
#define create_environment lifetime_private_environment
#define transpile_to_c lifetime_private_transpile
#define fputs lifetime_private_fputs
#define fclose lifetime_private_fclose
#include "../src/module.c"
#undef create_environment
#undef transpile_to_c
#undef fputs
#undef fclose
#ifdef EVALUATOR_ALLOCATION_HOOKS
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free
#endif
EnvEvaluationProvider *lifetime_cache_provider(void) {
    return module_cache ? module_cache->provider : NULL;
}
int lifetime_cache_count(void) { return module_cache ? module_cache->count : 0; }

void lifetime_cache_initialize(void) { init_module_cache(); }

bool lifetime_private_compile_fault(const char *path, const char *output, Environment *env, int fault) {
    lifetime_private_fault = fault;
    lifetime_private_closes = 0;
    lifetime_private_output = NULL;
    char *invalid_flags[] = {"-fno-nanolang-fixture-absent-option"};
    bool result = compile_module_to_object(path, output, env, false,
        fault == 5 ? invalid_flags : NULL, fault == 5 ? 1 : 0);
    lifetime_private_fault = 0;
    return result;
}
int lifetime_private_close_count(void) { return lifetime_private_closes; }
