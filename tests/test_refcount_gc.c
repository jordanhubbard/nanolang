/*
 * test_refcount_gc.c — Unit tests for the WASM reference-counting GC
 *
 * Tests cover:
 *   1.  Basic alloc/release (single allocation, rc goes 1→0)
 *   2.  Retain/release (rc 1→2→1→0)
 *   3.  Double-release guard (no crash on extra release)
 *   4.  OOM: heap full returns NULL
 *   5.  String allocation: null-terminated, correct content
 *   6.  String from C literal
 *   7.  String concatenation
 *   8.  nl_rc_str_eq
 *   9.  Heap stats: live alloc count + live bytes
 *   10. Closure alloc: upvalues retained
 *   11. Closure release: upvalues released
 *   12. Heap reset (test isolation)
 */

/* Pull in the GC as header-only */
#define NL_WASM_HEAP_SIZE (64 * 1024)   /* 64KB for tests */
#include "../src/runtime/refcount_gc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Test runner helpers ─────────────────────────────────────────────────── */

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name)  do { \
    nl_rc_heap_reset(); \
    test_##name(); \
    printf("  %-55s PASS\n", #name "..."); \
    g_pass++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("  ASSERTION FAILED: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
        g_fail++; \
        return; \
    } \
} while(0)

/* ── Tests ───────────────────────────────────────────────────────────────── */

TEST(basic_alloc_release) {
    void *p = nl_rc_alloc(16);
    ASSERT(p != NULL);
    ASSERT(nl_rc_refcount(p) == 1);
    nl_rc_release(p);
    ASSERT(nl_rc_refcount(p) == 0);
}

TEST(retain_release) {
    void *p = nl_rc_alloc(8);
    ASSERT(nl_rc_refcount(p) == 1);
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
    nl_rc_release(p);  /* should not crash or corrupt */
    ASSERT(nl_rc_refcount(p) == 0);
}

TEST(null_retain_release_safe) {
    nl_rc_retain(NULL);  /* no-op */
    nl_rc_release(NULL); /* no-op */
    ASSERT(1); /* no crash */
}

TEST(oom_returns_null) {
    /* Exhaust the 64KB test heap with many small allocs */
    int oom_seen = 0;
    for (int i = 0; i < 100000; i++) {
        void *p = nl_rc_alloc(64);
        if (!p) { oom_seen = 1; break; }
    }
    ASSERT(oom_seen);
}

TEST(str_new_content) {
    char *s = nl_rc_str_new("hello", 5);
    ASSERT(s != NULL);
    ASSERT(strcmp(s, "hello") == 0);
    ASSERT(nl_rc_refcount(s) == 1);
    nl_rc_release(s);
}

TEST(str_from_cstr) {
    char *s = nl_rc_str_from_cstr("world");
    ASSERT(s != NULL);
    ASSERT(strcmp(s, "world") == 0);
    nl_rc_release(s);
}

TEST(str_from_cstr_null) {
    char *s = nl_rc_str_from_cstr(NULL);
    ASSERT(s != NULL);
    ASSERT(strcmp(s, "") == 0);
    nl_rc_release(s);
}

TEST(str_concat) {
    char *a = nl_rc_str_from_cstr("foo");
    char *b = nl_rc_str_from_cstr("bar");
    char *r = nl_rc_str_concat(a, b);
    ASSERT(r != NULL);
    ASSERT(strcmp(r, "foobar") == 0);
    ASSERT(nl_rc_refcount(r) == 1);
    /* a and b are unchanged */
    ASSERT(nl_rc_refcount(a) == 1);
    ASSERT(nl_rc_refcount(b) == 1);
    nl_rc_release(a);
    nl_rc_release(b);
    nl_rc_release(r);
}

TEST(str_concat_empty) {
    char *a = nl_rc_str_from_cstr("abc");
    char *r = nl_rc_str_concat(a, NULL);
    ASSERT(r != NULL);
    ASSERT(strcmp(r, "abc") == 0);
    nl_rc_release(a);
    nl_rc_release(r);
}

TEST(str_eq) {
    char *a = nl_rc_str_from_cstr("peabody");
    char *b = nl_rc_str_from_cstr("peabody");
    ASSERT(nl_rc_str_eq(a, b));
    ASSERT(!nl_rc_str_eq(a, NULL));
    ASSERT(!nl_rc_str_eq(NULL, b));
    ASSERT(nl_rc_str_eq(NULL, NULL));
    nl_rc_release(a);
    nl_rc_release(b);
}

TEST(heap_stats_live_count) {
    uint32_t allocs = 0, bytes = 0;
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 0);
    ASSERT(bytes == 0);

    void *p1 = nl_rc_alloc(10);
    void *p2 = nl_rc_alloc(20);
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 2);
    ASSERT(bytes == 30);

    nl_rc_release(p1);
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 1);
    ASSERT(bytes == 20);

    nl_rc_release(p2);
    nl_rc_heap_stats(&allocs, &bytes);
    ASSERT(allocs == 0);
    ASSERT(bytes == 0);
}

TEST(closure_alloc_upvalue_retain) {
    char *upval = nl_rc_str_from_cstr("captured");
    ASSERT(nl_rc_refcount(upval) == 1);

    void *uvs[] = { upval };
    NLClosure *cl = nl_rc_closure_new(42, 1, uvs);
    ASSERT(cl != NULL);
    ASSERT(cl->fn_index == 42);
    ASSERT(cl->upvalue_count == 1);
    ASSERT(cl->upvalues[0] == upval);
    /* closure should have retained the upvalue */
    ASSERT(nl_rc_refcount(upval) == 2);

    nl_rc_closure_release(cl);
    /* closure released: upvalue RC back to 1 */
    ASSERT(nl_rc_refcount(upval) == 1);

    nl_rc_release(upval);
    ASSERT(nl_rc_refcount(upval) == 0);
}

TEST(closure_full_drop) {
    char *upval = nl_rc_str_from_cstr("gone");
    void *uvs[] = { upval };
    NLClosure *cl = nl_rc_closure_new(7, 1, uvs);
    nl_rc_release(upval);   /* caller drops its ref; closure holds the last */
    ASSERT(nl_rc_refcount(upval) == 1);

    nl_rc_closure_release(cl);
    /* both closure and upval should be at rc=0 */
    ASSERT(nl_rc_refcount(upval) == 0);
    ASSERT(nl_rc_refcount(cl)    == 0);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[refcount_gc] Running WASM reference-counting GC tests...\n\n");

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

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    } else {
        printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
        return 1;
    }
}
