/* I exercise the actual evaluator task snapshot hook, not a duplicate copier. */
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
#include "../src/eval.c"
#ifdef EVALUATOR_ALLOCATION_HOOKS
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free
#endif
bool lifetime_task_clone(Value source, Value *out) { return eval_owned_task_clone(source, out); }
void lifetime_task_drop(Value value) { eval_owned_task_drop(value); }

/* I use the real bundle preparation/rollback functions; this queued-only control
 * cancels before attempting to call its deliberately absent fixture target. */
bool lifetime_prepare_bundle(Environment *env, Value source, int *out) {
    CoroCallArgs *bundle = coro_bundle_new(env, "queued_fixture_target", 1);
    if (!bundle) return false;
    if (!coro_bundle_argument(bundle, 0, source, false)) { coro_bundle_drop(bundle); return false; }
    int id = coro_bundle_enqueue(bundle);
    if (id < 0) { coro_bundle_drop(bundle); return false; }
    *out = id;
    return true;
}
Value lifetime_stage_argument(Environment *env, ASTNode *expression, const char *name) {
    return eval_staged_argument(expression, env, name, 0);
}
