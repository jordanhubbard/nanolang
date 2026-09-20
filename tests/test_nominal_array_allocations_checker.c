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
    NominalView view = {*info, *owner};
    if (!nominal_value_view(expression, env, depth, &view)) {
        /* I check the actual private output, not only my wrapper's outputs. */
        if (view.info != *info || view.owner != *owner) abort();
        return false;
    }
    *info = view.info; *owner = view.owner;
    return true;
}
