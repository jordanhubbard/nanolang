/*
 * test_gc_struct.c — unit tests for src/runtime/gc_struct.c
 *
 * Exercises: gc_struct_new, gc_struct_free, gc_struct_set_field,
 * gc_struct_get_field, gc_struct_get_field_by_name, gc_struct_get_field_index,
 * gc_struct_clone.
 */

#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include "../src/runtime/gc_struct.h"
#include "../src/runtime/gc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-55s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-55s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { FAIL(test_name, (msg)); return; } } while(0)
#define ASSERT_STR(a, b, msg) do { if (!a || strcmp((a),(b))!=0) { FAIL(test_name, (msg)); return; } } while(0)

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_new_and_free(void) {
    const char *test_name = "gc_struct_new_and_free";
    gc_init();

    GCStruct *s = gc_struct_new("Point", 2);
    ASSERT(s != NULL, "gc_struct_new returned NULL");
    ASSERT_STR(s->struct_name, "Point", "struct_name wrong");
    ASSERT_EQ(s->field_count, 2, "field_count wrong");
    ASSERT(s->field_names != NULL, "field_names NULL");
    ASSERT(s->field_values != NULL, "field_values NULL");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

static void test_set_and_get_primitive(void) {
    const char *test_name = "gc_struct_set_get_primitive";
    gc_init();

    GCStruct *s = gc_struct_new("Rect", 3);
    ASSERT(s != NULL, "alloc failed");

    /* Store primitive ints as pointer-sized values */
    intptr_t w = 100, h = 200, depth = 3;
    gc_struct_set_field(s, 0, "width",  (void*)w, FIELD_INT, false);
    gc_struct_set_field(s, 1, "height", (void*)h, FIELD_INT, false);
    gc_struct_set_field(s, 2, "depth",  (void*)depth, FIELD_INT, false);

    ASSERT_EQ((intptr_t)gc_struct_get_field(s, 0), w, "width wrong");
    ASSERT_EQ((intptr_t)gc_struct_get_field(s, 1), h, "height wrong");
    ASSERT_EQ((intptr_t)gc_struct_get_field(s, 2), depth, "depth wrong");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

static void test_get_field_by_name(void) {
    const char *test_name = "gc_struct_get_field_by_name";
    gc_init();

    GCStruct *s = gc_struct_new("Color", 3);
    ASSERT(s != NULL, "alloc failed");

    gc_struct_set_field(s, 0, "r", (void*)(intptr_t)255, FIELD_INT, false);
    gc_struct_set_field(s, 1, "g", (void*)(intptr_t)128, FIELD_INT, false);
    gc_struct_set_field(s, 2, "b", (void*)(intptr_t)64,  FIELD_INT, false);

    ASSERT_EQ((intptr_t)gc_struct_get_field_by_name(s, "r"), 255, "r wrong");
    ASSERT_EQ((intptr_t)gc_struct_get_field_by_name(s, "g"), 128, "g wrong");
    ASSERT_EQ((intptr_t)gc_struct_get_field_by_name(s, "b"), 64,  "b wrong");
    ASSERT(gc_struct_get_field_by_name(s, "missing") == NULL, "missing should be NULL");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

static void test_get_field_index(void) {
    const char *test_name = "gc_struct_get_field_index";
    gc_init();

    GCStruct *s = gc_struct_new("Vec2", 2);
    ASSERT(s != NULL, "alloc failed");

    gc_struct_set_field(s, 0, "x", (void*)(intptr_t)10, FIELD_INT, false);
    gc_struct_set_field(s, 1, "y", (void*)(intptr_t)20, FIELD_INT, false);

    ASSERT_EQ(gc_struct_get_field_index(s, "x"), 0, "x index wrong");
    ASSERT_EQ(gc_struct_get_field_index(s, "y"), 1, "y index wrong");
    ASSERT_EQ(gc_struct_get_field_index(s, "z"), -1, "z should be -1");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

static void test_get_field_out_of_bounds(void) {
    const char *test_name = "gc_struct_get_field_out_of_bounds";
    gc_init();

    GCStruct *s = gc_struct_new("Small", 1);
    ASSERT(s != NULL, "alloc failed");
    gc_struct_set_field(s, 0, "only", (void*)(intptr_t)42, FIELD_INT, false);

    /* Out of bounds returns NULL (not crash) */
    ASSERT(gc_struct_get_field(s, 5) == NULL, "out-of-bounds should return NULL");
    ASSERT(gc_struct_get_field(s, -1) == NULL, "negative index should return NULL");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

static void test_set_field_update(void) {
    const char *test_name = "gc_struct_set_field_update";
    gc_init();

    GCStruct *s = gc_struct_new("Counter", 1);
    ASSERT(s != NULL, "alloc failed");

    gc_struct_set_field(s, 0, "count", (void*)(intptr_t)1, FIELD_INT, false);
    ASSERT_EQ((intptr_t)gc_struct_get_field(s, 0), 1, "initial wrong");

    /* Update same field — old non-GC value just replaced */
    gc_struct_set_field(s, 0, "count", (void*)(intptr_t)99, FIELD_INT, false);
    ASSERT_EQ((intptr_t)gc_struct_get_field(s, 0), 99, "update wrong");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

static void test_null_free(void) {
    const char *test_name = "gc_struct_free_null";
    gc_init();
    gc_release(NULL);  /* Should not crash */
    gc_struct_free(NULL);  /* Direct call with NULL should not crash */
    gc_shutdown();
    PASS(test_name);
}

static void test_clone_primitive(void) {
    const char *test_name = "gc_struct_clone_primitive";
    gc_init();

    GCStruct *orig = gc_struct_new("Pair", 2);
    ASSERT(orig != NULL, "alloc failed");
    gc_struct_set_field(orig, 0, "first",  (void*)(intptr_t)10, FIELD_INT, false);
    gc_struct_set_field(orig, 1, "second", (void*)(intptr_t)20, FIELD_INT, false);

    GCStruct *clone = gc_struct_clone(orig);
    ASSERT(clone != NULL, "clone failed");
    ASSERT(clone != orig, "clone is same pointer");
    ASSERT_STR(clone->struct_name, "Pair", "clone name wrong");
    ASSERT_EQ((intptr_t)gc_struct_get_field(clone, 0), 10, "clone first wrong");
    ASSERT_EQ((intptr_t)gc_struct_get_field(clone, 1), 20, "clone second wrong");

    gc_release(orig);
    gc_release(clone);
    gc_shutdown();
    PASS(test_name);
}

static void test_gc_object_field_retain_release(void) {
    const char *test_name = "gc_struct_gc_object_field";
    gc_init();

    /* Allocate a GC string to store as a struct field */
    char *str = (char*)gc_alloc(8, GC_TYPE_STRING);
    ASSERT(str != NULL, "gc_alloc failed");
    strcpy(str, "hello");

    GCStruct *s = gc_struct_new("Named", 1);
    ASSERT(s != NULL, "struct alloc failed");

    /* Set GC field — should retain str (rc goes 1→2) */
    gc_struct_set_field(s, 0, "name", str, FIELD_STRING, true);

    /* Release our own reference; struct still holds it */
    gc_release(str);

    /* Release struct via GC — it will call gc_struct_free which releases str */
    gc_release(s);

    gc_shutdown();
    PASS(test_name);
}

static void test_field_types(void) {
    const char *test_name = "gc_struct_field_types";
    gc_init();

    GCStruct *s = gc_struct_new("Mixed", 3);
    ASSERT(s != NULL, "alloc failed");

    gc_struct_set_field(s, 0, "i", (void*)(intptr_t)42,  FIELD_INT,   false);
    gc_struct_set_field(s, 1, "b", (void*)(intptr_t)1,   FIELD_BOOL,  false);
    gc_struct_set_field(s, 2, "f", (void*)(intptr_t)0,   FIELD_FLOAT, false);

    ASSERT_EQ(s->field_types[0], (uint8_t)FIELD_INT,   "type[0] wrong");
    ASSERT_EQ(s->field_types[1], (uint8_t)FIELD_BOOL,  "type[1] wrong");
    ASSERT_EQ(s->field_types[2], (uint8_t)FIELD_FLOAT, "type[2] wrong");

    gc_release(s);
    gc_shutdown();
    PASS(test_name);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[gc_struct] GC-managed struct tests...\n\n");

    test_new_and_free();
    test_set_and_get_primitive();
    test_get_field_by_name();
    test_get_field_index();
    test_get_field_out_of_bounds();
    test_set_field_update();
    test_null_free();
    test_clone_primitive();
    test_gc_object_field_retain_release();
    test_field_types();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d gc_struct tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d gc_struct tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
