#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
/* I preserve every original assertion and declare custody before C calls. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>
/* I fail only private ownership metadata; payload allocations remain ordinary. */
static long allocation_budget = -1;
static size_t refused_allocations;
static bool allow_allocation(void) {
    if (allocation_budget < 0) return true;
    if (!allocation_budget) { ++refused_allocations; return false; }
    --allocation_budget; return true;
}
void *nano_carrier_test_malloc(size_t n) { return allow_allocation() ? malloc(n) : NULL; }
void *nano_carrier_test_calloc(size_t n, size_t w) { return allow_allocation() ? calloc(n,w) : NULL; }
void *nano_carrier_test_realloc(void *p, size_t n) { return allow_allocation() ? realloc(p,n) : NULL; }
void nano_carrier_test_free(void *p) { free(p); }
static bool qualified_type_check(ASTNode *program, Environment *env) {
    bool ok = type_check(program, env);
    return ok && env_carrier_enable(env);
}
static Value qualified_call_function(const char *name, Value *args, int count, Environment *env) {
    if (count == 1 && (!strcmp(name, "consume") || !strcmp(name, "install"))) {
        NlCarrierRef root = args[0].type == VAL_INT ?
            nl_carrier_legacy_record_list((List_int *)(intptr_t)args[0].as.int_val) :
            nl_carrier_value(args[0]);
        assert(env_carrier_set_call_roots(env, &root, 1));
    }
    return call_function(name, args, count, env);
}
#define type_check qualified_type_check
#define call_function qualified_call_function
#define main ephemeral_cases_main
#include "support/ephemeral_array_cases.inc"
#undef main
#undef call_function
#undef type_check

static void declaration_guards(void) {
    Environment *env = create_environment(); assert(env_carrier_enable(env));
    Value outer = create_array(VAL_ARRAY, 0, 0);
    NlCarrierRef root = nl_carrier_value(outer);
    assert(env_carrier_set_call_roots(env, &root, 1));
    Value nil = create_void(); nil.type = VAL_ARRAY; nil.as.array_val = NULL;
    NlCarrierRef null_root = nl_carrier_value(nil);
    assert(!env_carrier_set_call_roots(env, &null_root, 1));
    NlCarrierRef plain = nl_carrier_value(create_int(7));
    assert(!env_carrier_set_call_roots(env, &plain, 1));
    assert(!env_carrier_set_call_roots(env, NULL, 1));
    assert(!env_carrier_set_call_roots(env, &root, SIZE_MAX));
    env_carrier_enter(env);
    assert(!env_carrier_set_call_roots(env, NULL, 0));
    env_carrier_leave(env);
    Value child = create_array(VAL_INT, 1, 1);
    ((long long *)child.as.array_val->data)[0] = 7;
    env_carrier_track(env, child, false);
    assert(!env_carrier_set_call_roots(env, &(NlCarrierRef){child,0}, 1));
    Value malformed = create_array(VAL_ARRAY, 0, 0);
    malformed.as.array_val->length = -1;
    NlCarrierRef bad = nl_carrier_value(malformed);
    assert(!env_carrier_set_call_roots(env, &bad, 1));
    malformed.as.array_val->length = 0;
    free(malformed.as.array_val->data); free(malformed.as.array_val);
    List_int *list = list_int_new(); assert(list);
    NlCarrierRef conflict[] = {nl_carrier_legacy_record_list(list), nl_carrier_legacy_scalar_list(list)};
    assert(!env_carrier_set_call_roots(env, conflict, 2));
    list_int_free(list);
    /* A rejected replacement keeps my previous caller root alive and owned. */
    free(outer.as.array_val->data);
    outer.as.array_val->data = malloc(sizeof(Value)); assert(outer.as.array_val->data);
    ((Value *)outer.as.array_val->data)[0] = child;
    outer.as.array_val->length = outer.as.array_val->capacity = 1;
    assert(env_carrier_enable_collection(env, NULL, 0));
    assert(env_carrier_collect(env)); assert(env_carrier_owns(env, child));
    env_carrier_export_call(env, create_int(0), false, false);
    assert(!env_carrier_owns(env, child));
    assert(env_carrier_set_call_roots(env, NULL, 0));
    Value borrowed = create_array(VAL_INT, 0, 0);
    env_carrier_track(env, borrowed, false);
    env_carrier_export_call(env, borrowed, false, false);
    assert(env_carrier_owns(env, borrowed));
    free_environment(env);
    release_caller_nested(outer);
    gc_shutdown();
    puts("I preserve caller declarations across refusals, collect reachable children and hand off without owning borrowed results.");
}
static void overlapping_cyclic_roots(void) {
    Environment *env = create_environment(); assert(env_carrier_enable(env));
    Value cycle = create_array(VAL_ARRAY, 1, 1);
    ((Value *)cycle.as.array_val->data)[0] = cycle;
    NlCarrierRef roots[] = {nl_carrier_value(cycle), nl_carrier_value(cycle)};
    assert(env_carrier_set_call_roots(env, roots, 2));
    assert(env_carrier_enable_collection(env, NULL, 0));
    assert(env_carrier_collect(env));
    env_carrier_export_call(env, create_void(), false, false);
    assert(env_carrier_set_call_roots(env, NULL, 0));
    free_environment(env);
    assert(((Value *)cycle.as.array_val->data)[0].as.array_val == cycle.as.array_val);
    free(cycle.as.array_val->data); free(cycle.as.array_val); gc_shutdown();
    puts("I preserve overlapping cyclic caller roots without adopting storage.");
}
static void scalar_list_and_keepalive_roles(void) {
    Environment *env = create_environment(); assert(env_carrier_enable(env));
    Value record = create_struct("Empty", NULL, NULL, 0);
    env_carrier_track(env, record, false);
    List_int *bits = list_int_new(); assert(bits);
    list_int_push(bits, (int64_t)(intptr_t)record.as.struct_val);
    NlCarrierRef records = nl_carrier_legacy_record_list(bits);
    assert(!env_carrier_set_call_roots(env, &records, 1));
    NlCarrierRef scalar = nl_carrier_legacy_scalar_list(bits);
    assert(env_carrier_set_call_roots(env, &scalar, 1));
    env_carrier_export_call(env, create_void(), false, false);
    assert(env_carrier_owns(env, record));
    assert(env_carrier_set_call_roots(env, NULL, 0));
    Value keepalive = create_array(VAL_ARRAY, 1, 1);
    ((Value *)keepalive.as.array_val->data)[0] = record;
    NlCarrierRef borrowed = nl_carrier_value(keepalive);
    assert(env_carrier_enable_collection(env, &borrowed, 1));
    assert(env_carrier_collect(env));
    env_carrier_export_call(env, create_void(), false, false);
    assert(env_carrier_owns(env, record));
    assert(env_carrier_enable_collection(env, NULL, 0));
    free_environment(env);
    free(keepalive.as.array_val->data); free(keepalive.as.array_val);
    list_int_free(bits); gc_shutdown();
    puts("I refuse owned record graphs while scalar list cells and borrow-only keepalive roots grant no custody handoff.");
}
static void declaration_allocation_failures(void) {
    size_t failures = 0;
    for (long budget = 0; budget < 32; ++budget) {
        Environment *env = create_environment(); assert(env_carrier_enable(env));
        Value previous = create_array(VAL_ARRAY, 0, 0);
        NlCarrierRef old = nl_carrier_value(previous);
        assert(env_carrier_set_call_roots(env, &old, 1));
        Value wide = create_array(VAL_ARRAY, 64, 64);
        for (int i = 0; i < 64; ++i)
            ((Value *)wide.as.array_val->data)[i] = create_array(VAL_ARRAY, 0, 0);
        NlCarrierRef next = nl_carrier_value(wide);
        size_t before = refused_allocations;
        allocation_budget = budget;
        bool ok = env_carrier_set_call_roots(env, &next, 1);
        allocation_budget = -1;
        if (!ok) {
            assert(refused_allocations == before + 1); ++failures;
            Value child = create_array(VAL_INT, 1, 1);
            ((long long *)child.as.array_val->data)[0] = 7;
            env_carrier_track(env, child, false);
            free(previous.as.array_val->data);
            previous.as.array_val->data = malloc(sizeof(Value)); assert(previous.as.array_val->data);
            ((Value *)previous.as.array_val->data)[0] = child;
            previous.as.array_val->length = previous.as.array_val->capacity = 1;
            assert(env_carrier_enable_collection(env, NULL, 0));
            assert(env_carrier_collect(env)); assert(env_carrier_owns(env, child));
            env_carrier_export_call(env, create_void(), false, false);
            assert(!env_carrier_owns(env, child));
            assert(env_carrier_set_call_roots(env, &next, 1));
            release_caller_nested(previous);
        } else { free(previous.as.array_val->data); free(previous.as.array_val); }
        assert(env_carrier_set_call_roots(env, NULL, 0));
        free_environment(env);
        for (int i = 0; i < 64; ++i) {
            Array *a = ((Value *)wide.as.array_val->data)[i].as.array_val;
            free(a->data); free(a);
        }
        free(wide.as.array_val->data); free(wide.as.array_val);
        gc_shutdown();
        if (ok) break;
    }
    assert(failures >= 3);
    printf("I preserve custody declarations through %zu allocation failures and recover.\n", failures);
}
int main(void) {
    declaration_guards();
    overlapping_cyclic_roots();
    scalar_list_and_keepalive_roles();
    declaration_allocation_failures();
    return ephemeral_cases_main();
}
