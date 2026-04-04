/**
 * test_coroutine_scheduler.c — unit tests for coroutine.c
 *
 * Exercises the cooperative coroutine scheduler API directly:
 *   nano_scheduler_init, nano_coro_spawn, nano_scheduler_step,
 *   nano_scheduler_run_until_done, nano_coro_await_id, nano_coro_result,
 *   nano_coro_is_done, nano_scheduler_pending_count, nano_coro_yield,
 *   nano_coro_complete, nano_coro_error, nano_coro_current_id
 */

#include "../src/nanolang.h"
#include "../src/coroutine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %lld, expected %lld)\n", \
        #a, #b, __LINE__, (long long)(a), (long long)(b)); exit(1); }

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* Helper: force-reset the scheduler between tests */
static void reset_scheduler(void) {
    g_scheduler.initialized = false;
    nano_scheduler_init();
}

/* Helper: Value factory for int */
static Value make_int(long long n) {
    Value v; v.type = VAL_INT; v.as.int_val = n; return v;
}

/* Helper: Value factory for void */
static Value make_void(void) {
    Value v; v.type = VAL_VOID; return v;
}

/* ============================================================================
 * Simple coroutine functions for testing
 * ============================================================================ */

static Value coro_return_42(void *arg, int coro_id) {
    (void)arg; (void)coro_id;
    return make_int(42);
}

static Value coro_return_arg_int(void *arg, int coro_id) {
    (void)coro_id;
    long long *n = (long long *)arg;
    return make_int(*n);
}

static Value coro_yield_and_return(void *arg, int coro_id) {
    (void)arg; (void)coro_id;
    nano_coro_yield();
    return make_int(99);
}

static Value coro_set_done(void *arg, int coro_id) {
    (void)arg; (void)coro_id;
    Value result = make_int(77);
    nano_coro_complete(result);
    return make_void();  /* not reached */
}

static Value coro_set_error(void *arg, int coro_id) {
    (void)arg; (void)coro_id;
    nano_coro_error("test error");
    return make_void();  /* not reached */
}

/* ============================================================================
 * Tests
 * ============================================================================ */

void test_scheduler_init(void) {
    reset_scheduler();
    ASSERT(g_scheduler.initialized);
    ASSERT_EQ(g_scheduler.count, 0);
    ASSERT_EQ(nano_scheduler_pending_count(), 0);
}

void test_scheduler_pending_count_empty(void) {
    reset_scheduler();
    ASSERT_EQ(nano_scheduler_pending_count(), 0);
}

void test_coro_spawn_simple(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_return_42, NULL);
    ASSERT(id >= 0);
    ASSERT_EQ(nano_scheduler_pending_count(), 1);
}

void test_coro_spawn_multiple(void) {
    reset_scheduler();
    int id1 = nano_coro_spawn(coro_return_42, NULL);
    int id2 = nano_coro_spawn(coro_return_42, NULL);
    ASSERT(id1 >= 0);
    ASSERT(id2 >= 0);
    ASSERT(id1 != id2);
    ASSERT_EQ(nano_scheduler_pending_count(), 2);
}

void test_scheduler_step_runs_one(void) {
    reset_scheduler();
    nano_coro_spawn(coro_return_42, NULL);
    bool stepped = nano_scheduler_step();
    ASSERT(stepped);
    ASSERT_EQ(nano_scheduler_pending_count(), 0);
}

void test_scheduler_step_empty_returns_false(void) {
    reset_scheduler();
    bool stepped = nano_scheduler_step();
    ASSERT(!stepped);
}

void test_coro_is_done_after_step(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_return_42, NULL);
    ASSERT(!nano_coro_is_done(id));
    nano_scheduler_step();
    ASSERT(nano_coro_is_done(id));
}

void test_coro_result_after_done(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_return_42, NULL);
    nano_scheduler_step();
    Value r = nano_coro_result(id);
    ASSERT_EQ(r.as.int_val, 42);
}

void test_coro_result_with_arg(void) {
    reset_scheduler();
    long long n = 123;
    int id = nano_coro_spawn(coro_return_arg_int, &n);
    nano_scheduler_step();
    Value r = nano_coro_result(id);
    ASSERT_EQ(r.as.int_val, 123);
}

void test_coro_await_id(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_return_42, NULL);
    Value r = nano_coro_await_id(id);
    ASSERT_EQ(r.as.int_val, 42);
    ASSERT(nano_coro_is_done(id));
}

void test_scheduler_run_until_done(void) {
    reset_scheduler();
    nano_coro_spawn(coro_return_42, NULL);
    nano_coro_spawn(coro_return_42, NULL);
    nano_coro_spawn(coro_return_42, NULL);
    nano_scheduler_run_until_done();
    ASSERT_EQ(nano_scheduler_pending_count(), 0);
}

void test_coro_yield(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_yield_and_return, NULL);
    /* Run until done */
    nano_coro_await_id(id);
    Value r = nano_coro_result(id);
    ASSERT_EQ(r.as.int_val, 99);
}

void test_coro_complete_sets_result(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_set_done, NULL);
    nano_scheduler_step();
    ASSERT(nano_coro_is_done(id));
    Value r = nano_coro_result(id);
    ASSERT_EQ(r.as.int_val, 77);
}

void test_coro_error(void) {
    reset_scheduler();
    int id = nano_coro_spawn(coro_set_error, NULL);
    nano_scheduler_step();
    /* After error, coro should be in error state (is_done returns true for error) */
    ASSERT(nano_coro_is_done(id));
}

void test_coro_current_id_outside_coro(void) {
    reset_scheduler();
    /* Outside a coroutine, current_id should return -1 */
    ASSERT_EQ(nano_coro_current_id(), -1);
}

void test_scheduler_multiple_sequential(void) {
    /* Run 5 coroutines sequentially, verify all complete correctly */
    reset_scheduler();
    long long vals[5] = {10, 20, 30, 40, 50};
    int ids[5];
    for (int i = 0; i < 5; i++) {
        ids[i] = nano_coro_spawn(coro_return_arg_int, &vals[i]);
    }
    nano_scheduler_run_until_done();
    for (int i = 0; i < 5; i++) {
        ASSERT(nano_coro_is_done(ids[i]));
        Value r = nano_coro_result(ids[i]);
        ASSERT_EQ(r.as.int_val, vals[i]);
    }
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Coroutine Scheduler Tests ===\n");
    TEST(scheduler_init);
    TEST(scheduler_pending_count_empty);
    TEST(coro_spawn_simple);
    TEST(coro_spawn_multiple);
    TEST(scheduler_step_runs_one);
    TEST(scheduler_step_empty_returns_false);
    TEST(coro_is_done_after_step);
    TEST(coro_result_after_done);
    TEST(coro_result_with_arg);
    TEST(coro_await_id);
    TEST(scheduler_run_until_done);
    TEST(coro_yield);
    TEST(coro_complete_sets_result);
    TEST(coro_error);
    TEST(coro_current_id_outside_coro);
    TEST(scheduler_multiple_sequential);

    printf("\n✓ All coroutine scheduler tests passed!\n");
    return 0;
}
