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

static int shutdown_finalizations;

static void count_shutdown_finalizer(void *object) {
    (void)object;
    shutdown_finalizations++;
}

static void test_shutdown_statistics(void) {
    const char *test_name = "gc_shutdown_statistics_and_finalization";
    gc_init();
    char *transient = gc_alloc_string(3);
    ASSERT(transient != NULL, "transient allocation failed");
    gc_release(transient);
    ASSERT(gc_alloc_string(8) != NULL, "live string allocation failed");
    ASSERT(gc_alloc_opaque(1, count_shutdown_finalizer) != NULL, "live owner allocation failed");
    GCStats before = gc_get_stats();
    ASSERT(before.num_objects == 2, "live object setup mismatch");
    gc_shutdown();
    GCStats after = gc_get_stats();
    ASSERT_EQ(after.current_usage, 0, "shutdown retains current bytes");
    ASSERT_EQ(after.num_objects, 0, "shutdown retains live objects");
    ASSERT_EQ(after.total_allocated, before.total_allocated, "shutdown changed allocated total");
    ASSERT_EQ(after.total_freed, after.total_allocated, "shutdown omitted reclaimed bytes");
    ASSERT_EQ(shutdown_finalizations, 1, "shutdown finalizer count mismatch");
    gc_shutdown();
    GCStats twice = gc_get_stats();
    ASSERT_EQ(twice.total_freed, after.total_freed, "repeated shutdown double-counted bytes");
    ASSERT_EQ(shutdown_finalizations, 1, "repeated shutdown finalized twice");
    gc_init();
    ASSERT_EQ(gc_get_stats().total_allocated, 0, "new GC lifetime kept old allocated total");
    gc_shutdown();
    PASS(test_name);
}

static void *registry_owners[3];
static int registry_finalizations;

static void inspect_shutdown_registry(void *object) {
    const char *test_name = "gc_shutdown_registry_visibility";
    int index = *(unsigned char *)object;
    ASSERT_EQ(index, 2 - registry_finalizations, "I changed shutdown finalizer order.");
    ASSERT(gc_is_managed(object), "I hid an owner during its own finalizer.");
    for (int i = 0; i < 3; ++i)
        ASSERT(gc_is_managed(registry_owners[i]) == (i <= index),
               "I confused a released owner with a still-live owner during shutdown.");
    registry_finalizations++;
}

static void test_shutdown_registry_visibility(void) {
    const char *test_name = "gc_shutdown_registry_visibility";
    gc_init();
    registry_finalizations = 0;
    for (int i = 0; i < 3; ++i) {
        registry_owners[i] = gc_alloc_opaque(1, inspect_shutdown_registry);
        ASSERT(registry_owners[i] != NULL, "I could not allocate my finalizer control.");
        *(unsigned char *)registry_owners[i] = (unsigned char)i;
    }
    gc_shutdown();
    ASSERT_EQ(registry_finalizations, 3, "I lost a shutdown finalizer.");
    for (int i = 0; i < 3; ++i)
        ASSERT(!gc_is_managed(registry_owners[i]), "I retain a shutdown registry entry.");
    GCStats after = gc_get_stats();
    ASSERT_EQ(after.current_usage, 0, "I retain bytes after registry shutdown.");
    ASSERT_EQ(after.num_objects, 0, "I retain objects after registry shutdown.");
    ASSERT_EQ(after.total_freed, after.total_allocated, "I omitted shutdown reclamation.");
    gc_shutdown();
    ASSERT_EQ(registry_finalizations, 3, "I repeated shutdown finalizers.");
    gc_init();
    void *fresh = gc_alloc(1, GC_TYPE_OPAQUE);
    ASSERT(fresh != NULL && gc_is_managed(fresh), "I lost registration after reinitializing.");
    gc_release(fresh);
    gc_shutdown();
    PASS(test_name);
}

static void test_allocation_size_boundary(void) {
    const char *test_name = "gc_allocation_size_boundary";
    gc_init();
    GCStats before = gc_get_stats();
    ASSERT(gc_alloc(SIZE_MAX, GC_TYPE_OPAQUE) == NULL, "I accepted an overflowing allocation.");
    ASSERT(gc_alloc(SIZE_MAX - sizeof(GCHeader) + 1, GC_TYPE_OPAQUE) == NULL,
           "I accepted the first overflowing header size.");
    ASSERT(gc_alloc_string(SIZE_MAX) == NULL, "I accepted an overflowing string terminator.");
    GCStats refused = gc_get_stats();
    ASSERT_EQ(refused.total_allocated, before.total_allocated, "I changed allocated bytes on refusal.");
    ASSERT_EQ(refused.total_freed, before.total_freed, "I changed freed bytes on refusal.");
    ASSERT_EQ(refused.current_usage, before.current_usage, "I changed live bytes on refusal.");
    ASSERT_EQ(refused.num_objects, before.num_objects, "I registered an object on refusal.");
    char *empty = gc_alloc_string(0);
    ASSERT(empty != NULL && empty[0] == '\0', "I lost my empty string terminator.");
    gc_release(empty);
    char *text = gc_alloc_string(3);
    ASSERT(text != NULL && text[3] == '\0', "I lost my nonempty string terminator.");
    memcpy(text, "abc", 3);
    ASSERT_STR(text, "abc", "I changed valid string storage.");
    gc_release(text);
    size_t sizes[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 4095, 4096, 4097};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; ++i) {
        void *value = gc_alloc(sizes[i], GC_TYPE_OPAQUE);
        ASSERT(value != NULL && gc_is_managed(value), "I did not register a valid allocation.");
        ASSERT_EQ(gc_get_header(value)->size, sizeof(GCHeader) + sizes[i], "I changed the logical allocation size.");
        gc_retain(value);
        ASSERT_EQ(gc_get_header(value)->ref_count, 2, "I changed the retained reference count.");
        gc_release(value);
        ASSERT(gc_is_managed(value), "I lost a retained value.");
        gc_release(value);
        ASSERT(!gc_is_managed(value), "I kept a released value registered.");
    }
    int foreign = 7;
    gc_release(&foreign);
    ASSERT(!gc_is_managed(&foreign), "I registered a foreign pointer.");
    ASSERT_EQ(gc_get_stats().current_usage, 0, "I retain managed bytes after my boundary controls.");
    ASSERT_EQ(gc_get_stats().num_objects, 0, "I retain managed objects after my boundary controls.");
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
    test_shutdown_statistics();
    test_allocation_size_boundary();
    test_shutdown_registry_visibility();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d gc_struct tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d gc_struct tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
