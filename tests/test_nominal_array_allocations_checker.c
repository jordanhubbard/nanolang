/* I compile the actual checker under a separate allocation domain. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/nanolang.h"
#include <stdlib.h>
#include <string.h>
void *array_alloc_malloc(size_t);
void *array_alloc_calloc(size_t, size_t);
void *array_alloc_realloc(void *, size_t);
char *array_alloc_strdup(const char *);
void array_alloc_free(void *);
#define malloc array_alloc_malloc
#define calloc array_alloc_calloc
#define realloc array_alloc_realloc
#define strdup array_alloc_strdup
#define free array_alloc_free
#include "../src/typechecker.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

bool array_test_view(Environment *env, ASTNode *expression, unsigned depth,
                     TypeInfo **info, const char **owner) {
    NominalView view = {.info = *info, .owner = *owner};
    if (!nominal_value_view(expression, env, depth, &view)) {
        /* I check the actual private output, not only my wrapper's outputs. */
        if (view.info != *info || view.owner != *owner) abort();
        return false;
    }
    /* This wrapper's original vectors have only root-module scalar leaves. */
    if (view.owner || view.owned_context || view.payload) abort();
    *info = view.info; *owner = NULL;
    view.info = NULL;
    nominal_view_discard(&view);
    return true;
}

/* I exercise owned context copying, materialization and both registrations. */
bool array_test_owned_context(Environment *env, Symbol *output) {
    char formal_name[] = "T", record_name[] = "Item", owner[] = "Caller";
    char *formals[] = {formal_name};
    UnionDef declaration = {.generic_param_count = 1, .generic_params = formals};
    TypeInfo leaf = {.base_type = TYPE_STRUCT, .generic_name = record_name};
    TypeInfo *arguments[] = {&leaf};
    TypeInfo instance = {.type_param_count = 1, .type_params = arguments};
    NominalSubstitution context = {&declaration, &instance, owner, NULL};
    TypeInfo compact = {.base_type = TYPE_LIST_GENERIC, .generic_name = formal_name};
    TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = &compact};
    NominalView view = {0};
    if (!nominal_view_copy_context(env, &array, "Definitions", &context, 0, &view)) return false;
    formal_name[0] = 'X'; record_name[0] = 'X'; owner[0] = 'X';
    bool ok = nominal_view_retain(env, output, &view);
    nominal_view_discard(&view);
    if (ok) {
        const NominalView *retained = output->checker_nominal_view;
        if (!retained || !retained->owned_context || strcmp(retained->owner, "Definitions") ||
            strcmp(retained->owned_context->context.argument_owner, "Caller") ||
            strcmp(output->type_info->element_type->type_params[0]->generic_name, "Item")) abort();
    }
    return ok;
}
