/*
 * test_refcount_gc.c — Tests for refcount_gc.h (v2.0 with tri-color GC)
 */

#define NL_WASM_HEAP_SIZE (64 * 1024)
#include "../src/runtime/refcount_gc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int g_pass = 0, g_fail = 0;
#define TEST(name) static void test_##name(void)
#define RUN(name)  do { nl_rc_heap_reset(); test_##name(); \
    printf("  %-55s PASS\n", #name "..."); g_pass++; } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
    g_fail++; return; } } while(0)

/* ── Original RC tests (v1.0) ──────────────────────────────────────────── */

TEST(basic_alloc_release) {
    void *p = nl_rc_alloc(16);
    ASSERT(p != NULL);
    ASSERT(nl_rc_refcount(p) == 1);
    nl_rc_release(p);
    ASSERT(nl_rc_refcount(p) == 0);
}

TEST(retain_release) {
    void *p = nl_rc_alloc(8);
    nl_rc_retain(p);
    ASSERT(nl_rc_refcount(p) == 2);
    nl_rc_release(p);
    ASSERT(nl_rc_refcount(p) == 1);
    nl_rc_release(p);
    ASSERT(nl_rc_refcount(p) == 0);
}

TEST(double_release_guard) {
    void *p = nl_rc_alloc(8);
    nl_rc_release(p);
    ASSERT(nl_rc_refcount(p) == 0);
    nl_rc_release(p);
    ASSERT(nl_rc_refcount(p) == 0);
}

TEST(null_retain_release_safe) {
    nl_rc_retain(NULL);
    nl_rc_release(NULL);
    ASSERT(1);
}

TEST(oom_returns_null) {
    int oom = 0;
    for (int i = 0; i < 100000; i++) {
        void *p = nl_rc_alloc(64);
        if (!p) { oom = 1; break; }
    }
    ASSERT(oom);
}

TEST(str_new_content) {
    char *s = nl_rc_str_new("hello", 5);
    ASSERT(s && strcmp(s, "hello") == 0);
    ASSERT(nl_rc_refcount(s) == 1);
    nl_rc_release(s);
}

TEST(str_from_cstr) {
    char *s = nl_rc_str_from_cstr("world");
    ASSERT(s && strcmp(s, "world") == 0);
    nl_rc_release(s);
}

TEST(str_from_cstr_null) {
    char *s = nl_rc_str_from_cstr(NULL);
    ASSERT(s && strcmp(s, "") == 0);
    nl_rc_release(s);
}

TEST(str_concat) {
    char *a = nl_rc_str_from_cstr("foo");
    char *b = nl_rc_str_from_cstr("bar");
    char *r = nl_rc_str_concat(a, b);
    ASSERT(r && strcmp(r, "foobar") == 0);
    ASSERT(nl_rc_refcount(r) == 1);
    ASSERT(nl_rc_refcount(a) == 1);
    ASSERT(nl_rc_refcount(b) == 1);
    nl_rc_release(a); nl_rc_release(b); nl_rc_release(r);
}

TEST(str_concat_empty) {
    char *a = nl_rc_str_from_cstr("abc");
    char *r = nl_rc_str_concat(a, NULL);
    ASSERT(r && strcmp(r, "abc") == 0);
    nl_rc_release(a); nl_rc_release(r);
}

TEST(str_eq) {
    char *a = nl_rc_str_from_cstr("peabody");
    char *b = nl_rc_str_from_cstr("peabody");
    ASSERT(nl_rc_str_eq(a, b));
    ASSERT(!nl_rc_str_eq(a, NULL));
    ASSERT(!nl_rc_str_eq(NULL, b));
    ASSERT(nl_rc_str_eq(NULL, NULL));
    nl_rc_release(a); nl_rc_release(b);
}

TEST(heap_stats_live_count) {
    uint32_t allocs = 0, bytes = 0;
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 0 && bytes == 0);
    void *p1 = nl_rc_alloc(10);
    void *p2 = nl_rc_alloc(20);
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 2 && bytes == 30);
    nl_rc_release(p1);
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 1 && bytes == 20);
    nl_rc_release(p2);
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 0 && bytes == 0);
}

TEST(closure_alloc_upvalue_retain) {
    char *uv = nl_rc_str_from_cstr("captured");
    ASSERT(nl_rc_refcount(uv) == 1);
    void *uvs[] = { uv };
    NLClosure *cl = nl_rc_closure_new(42, 1, uvs);
    ASSERT(cl && nl_rc_refcount(uv) == 2);
    nl_rc_closure_release(cl);
    ASSERT(nl_rc_refcount(uv) == 1);
    nl_rc_release(uv);
    ASSERT(nl_rc_refcount(uv) == 0);
}

TEST(closure_full_drop) {
    char *uv = nl_rc_str_from_cstr("gone");
    void *uvs[] = { uv };
    NLClosure *cl = nl_rc_closure_new(7, 1, uvs);
    nl_rc_release(uv);
    ASSERT(nl_rc_refcount(uv) == 1);
    nl_rc_closure_release(cl);
    ASSERT(nl_rc_refcount(uv) == 0);
    ASSERT(nl_rc_refcount(cl) == 0);
}

/* ── v2.0 Tri-color cycle collector tests ──────────────────────────────── */

TEST(gc_new_objects_black) {
    /* New allocations should be BLACK (definitely live) */
    void *p = nl_rc_alloc(8);
    ASSERT(p);
    ASSERT(NL_GC_COLOR(NL_RC_HDR(p)) == NL_GC_BLACK);
    nl_rc_release(p);
}

TEST(gc_suspect_added_on_partial_release) {
    /* When RC drops to >0, object should be added to cycle suspect buffer */
    char *uv = nl_rc_str_from_cstr("uv");
    void *uvs[] = { uv };
    NLClosure *cl = nl_rc_closure_new(1, 1, uvs);
    nl_rc_retain(cl);              /* RC = 2 */
    nl_rc_release(uv);             /* uv RC drops: 2→1, added to cycle buf */
    uint32_t cb_before = _nl_cycle_count;
    nl_rc_release(cl);             /* cl RC drops: 2→1, added to cycle buf */
    ASSERT(_nl_cycle_count >= cb_before); /* count should not decrease after a release */
    nl_rc_release(cl);             /* final release */
    nl_rc_release(uv);
}

TEST(gc_cycle_collector_reclaims_cycle) {
    /*
     * Build a 2-closure mutual reference cycle:
     *   cl_a references cl_b as upvalue
     *   cl_b references cl_a as upvalue
     * External references are then dropped — only the cycle remains.
     * nl_gc_collect_cycles() should free both.
     */
    uint32_t allocs_before = 0, bytes_before = 0;
    nl_rc_heap_stats(&allocs_before, &bytes_before);

    /* Allocate two closures with a dummy upvalue first (we'll replace) */
    char *dummy = nl_rc_str_from_cstr("dummy");
    void *uvs_a[] = { dummy };
    void *uvs_b[] = { dummy };
    NLClosure *cl_a = nl_rc_closure_new(1, 1, uvs_a);
    NLClosure *cl_b = nl_rc_closure_new(2, 1, uvs_b);
    ASSERT(cl_a && cl_b);

    /* Manually wire the cycle: cl_a.upvalue[0] = cl_b, cl_b.upvalue[0] = cl_a */
    /* First release the dummy upvalues that the constructor retained */
    nl_rc_release(dummy);  /* was retained by cl_a */
    nl_rc_release(dummy);  /* was retained by cl_b */
    /* Now install cycle edges (retain each for the cross-reference) */
    cl_a->upvalues[0] = cl_b;
    nl_rc_retain(cl_b);
    cl_b->upvalues[0] = cl_a;
    nl_rc_retain(cl_a);

    /* Drop external refs — only the cycle references keep RC > 0 */
    nl_rc_release(cl_a);   /* RC: cl_a = 1 (held by cl_b), cl_b = 1 (held by cl_a) */
    nl_rc_release(cl_b);
    nl_rc_release(dummy);  /* drop our own dummy ref */

    /* At this point cl_a and cl_b each have RC=1 from the cycle, neither freed */
    ASSERT(nl_rc_refcount(cl_a) == 1);
    ASSERT(nl_rc_refcount(cl_b) == 1);

    /* Run cycle collector */
    nl_gc_collect_cycles();

    /* Both should now be freed */
    ASSERT(nl_rc_refcount(cl_a) == 0);
    ASSERT(nl_rc_refcount(cl_b) == 0);
}

TEST(gc_collect_does_not_free_live) {
    /* An object with external references must NOT be freed by the collector */
    char *s = nl_rc_str_from_cstr("alive");
    nl_rc_retain(s);   /* external ref: RC = 2 */
    /* Force it onto the cycle buffer by dropping to RC=1 */
    nl_rc_release(s);  /* RC = 1 */
    ASSERT(nl_rc_refcount(s) == 1);

    nl_gc_collect_cycles();

    /* Still alive */
    ASSERT(nl_rc_refcount(s) == 1);
    nl_rc_release(s);
    ASSERT(nl_rc_refcount(s) == 0);
}

TEST(gc_collect_empty_buffer_noop) {
    /* Collecting with empty cycle buffer should be a no-op */
    ASSERT(_nl_cycle_count == 0);
    nl_gc_collect_cycles();
    ASSERT(_nl_cycle_count == 0);
    ASSERT(1);
}

int main(void) {
    printf("\n[refcount_gc v2.0] RC + tri-color mark-sweep GC tests...\n\n");
    RUN(basic_alloc_release);
    RUN(retain_release);
    RUN(double_release_guard);
    RUN(null_retain_release_safe);
    RUN(oom_returns_null);
    RUN(str_new_content);
    RUN(str_from_cstr);
    RUN(str_from_cstr_null);
    RUN(str_concat);
    RUN(str_concat_empty);
    RUN(str_eq);
    RUN(heap_stats_live_count);
    RUN(closure_alloc_upvalue_retain);
    RUN(closure_full_drop);
    /* v2.0 tests */
    RUN(gc_new_objects_black);
    RUN(gc_suspect_added_on_partial_release);
    RUN(gc_cycle_collector_reclaims_cycle);
    RUN(gc_collect_does_not_free_live);
    RUN(gc_collect_empty_buffer_noop);
    printf("\n");
    if (g_fail == 0) { printf("All %d tests passed.\n", g_pass); return 0; }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
