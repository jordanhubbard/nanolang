/*
 * test_dyn_array.c — unit tests for src/runtime/dyn_array.c
 *
 * Exercises: dyn_array_new, dyn_array_new_with_capacity, push/pop/get/set
 * for all element types, clone, remove_at, clear, reserve, struct ops.
 */

#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include "../src/runtime/dyn_array.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

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

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_new_int(void) {
    const char *test_name = "dyn_array_new: int array";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");
    ASSERT_EQ(dyn_array_length(arr), 0, "length should be 0");
    ASSERT(dyn_array_capacity(arr) > 0, "capacity should be positive");
    ASSERT_EQ(dyn_array_get_elem_type(arr), ELEM_INT, "elem_type should be ELEM_INT");
    PASS(test_name);
}

static void test_new_with_capacity(void) {
    const char *test_name = "dyn_array_new_with_capacity: custom capacity";
    DynArray *arr = dyn_array_new_with_capacity(ELEM_FLOAT, 32);
    ASSERT(arr != NULL, "dyn_array_new_with_capacity returned NULL");
    ASSERT(dyn_array_capacity(arr) >= 32, "capacity should be >= 32");
    ASSERT_EQ(dyn_array_length(arr), 0, "length should be 0");
    PASS(test_name);
}

static void test_push_pop_int(void) {
    const char *test_name = "push/pop int";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 42);
    dyn_array_push_int(arr, 100);
    dyn_array_push_int(arr, -7);
    ASSERT_EQ(dyn_array_length(arr), 3, "length should be 3");

    bool ok = false;
    int64_t v = dyn_array_pop_int(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT_EQ(v, -7, "popped value should be -7");
    ASSERT_EQ(dyn_array_length(arr), 2, "length should be 2 after pop");

    v = dyn_array_pop_int(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT_EQ(v, 100, "popped value should be 100");

    v = dyn_array_pop_int(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT_EQ(v, 42, "popped value should be 42");

    /* Pop from empty */
    ok = true;
    dyn_array_pop_int(arr, &ok);
    ASSERT(!ok, "pop from empty should fail");

    PASS(test_name);
}

static void test_push_pop_u8(void) {
    const char *test_name = "push/pop u8";
    DynArray *arr = dyn_array_new(ELEM_U8);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_u8(arr, 0);
    dyn_array_push_u8(arr, 255);
    dyn_array_push_u8(arr, 128);
    ASSERT_EQ(dyn_array_length(arr), 3, "length should be 3");

    bool ok = false;
    uint8_t v = dyn_array_pop_u8(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT_EQ(v, 128, "popped value should be 128");

    PASS(test_name);
}

static void test_push_pop_float(void) {
    const char *test_name = "push/pop float";
    DynArray *arr = dyn_array_new(ELEM_FLOAT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_float(arr, 3.14);
    dyn_array_push_float(arr, -2.71);
    ASSERT_EQ(dyn_array_length(arr), 2, "length should be 2");

    bool ok = false;
    double v = dyn_array_pop_float(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT(v == -2.71, "popped value should be -2.71");

    PASS(test_name);
}

static void test_push_pop_bool(void) {
    const char *test_name = "push/pop bool";
    DynArray *arr = dyn_array_new(ELEM_BOOL);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_bool(arr, true);
    dyn_array_push_bool(arr, false);
    dyn_array_push_bool(arr, true);
    ASSERT_EQ(dyn_array_length(arr), 3, "length should be 3");

    bool ok = false;
    bool v = dyn_array_pop_bool(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT(v == true, "popped value should be true");

    PASS(test_name);
}

static void test_push_pop_string(void) {
    const char *test_name = "push/pop string";
    DynArray *arr = dyn_array_new(ELEM_STRING);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    const char *s1 = "hello";
    const char *s2 = "world";
    dyn_array_push_string(arr, s1);
    dyn_array_push_string(arr, s2);
    ASSERT_EQ(dyn_array_length(arr), 2, "length should be 2");

    bool ok = false;
    const char *v = dyn_array_pop_string(arr, &ok);
    ASSERT(ok, "pop should succeed");
    ASSERT(strcmp(v, "world") == 0, "popped value should be 'world'");

    PASS(test_name);
}

static void test_push_string_copy(void) {
    const char *test_name = "push_string_copy: copies the string";
    DynArray *arr = dyn_array_new(ELEM_STRING);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    char buf[32];
    strcpy(buf, "original");
    dyn_array_push_string_copy(arr, buf);
    strcpy(buf, "modified");  /* Modify original buffer */

    char *stored = dyn_array_get_string(arr, 0);
    ASSERT(stored != NULL, "stored string should not be NULL");
    ASSERT(strcmp(stored, "original") == 0, "stored string should be 'original' (copy)");
    free(stored);

    PASS(test_name);
}

static void test_get_set_int(void) {
    const char *test_name = "get/set int";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 10);
    dyn_array_push_int(arr, 20);
    dyn_array_push_int(arr, 30);

    ASSERT_EQ(dyn_array_get_int(arr, 0), 10, "get[0] should be 10");
    ASSERT_EQ(dyn_array_get_int(arr, 1), 20, "get[1] should be 20");
    ASSERT_EQ(dyn_array_get_int(arr, 2), 30, "get[2] should be 30");

    dyn_array_set_int(arr, 1, 99);
    ASSERT_EQ(dyn_array_get_int(arr, 1), 99, "get[1] should be 99 after set");

    PASS(test_name);
}

static void test_get_set_float(void) {
    const char *test_name = "get/set float";
    DynArray *arr = dyn_array_new(ELEM_FLOAT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_float(arr, 1.5);
    dyn_array_push_float(arr, 2.5);

    ASSERT(dyn_array_get_float(arr, 0) == 1.5, "get[0] should be 1.5");
    dyn_array_set_float(arr, 0, 9.9);
    ASSERT(dyn_array_get_float(arr, 0) == 9.9, "get[0] should be 9.9 after set");

    PASS(test_name);
}

static void test_get_set_bool(void) {
    const char *test_name = "get/set bool";
    DynArray *arr = dyn_array_new(ELEM_BOOL);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_bool(arr, false);
    ASSERT(dyn_array_get_bool(arr, 0) == false, "get[0] should be false");
    dyn_array_set_bool(arr, 0, true);
    ASSERT(dyn_array_get_bool(arr, 0) == true, "get[0] should be true after set");

    PASS(test_name);
}

static void test_get_set_u8(void) {
    const char *test_name = "get/set u8";
    DynArray *arr = dyn_array_new(ELEM_U8);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_u8(arr, 10);
    dyn_array_push_u8(arr, 20);

    ASSERT_EQ((int)dyn_array_get_u8(arr, 0), 10, "get[0] should be 10");
    ASSERT_EQ((int)dyn_array_get_u8(arr, 1), 20, "get[1] should be 20");
    dyn_array_set_u8(arr, 0, 55);
    ASSERT_EQ((int)dyn_array_get_u8(arr, 0), 55, "get[0] should be 55 after set");

    PASS(test_name);
}

static void test_growth_beyond_initial_capacity(void) {
    const char *test_name = "growth: push beyond initial capacity";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    /* Initial capacity is 8, push 20 elements to trigger growth */
    for (int i = 0; i < 20; i++) {
        dyn_array_push_int(arr, (int64_t)i * 100);
    }
    ASSERT_EQ(dyn_array_length(arr), 20, "length should be 20");
    ASSERT(dyn_array_capacity(arr) >= 20, "capacity should be >= 20");

    /* Verify all values */
    for (int i = 0; i < 20; i++) {
        int64_t v = dyn_array_get_int(arr, (int64_t)i);
        ASSERT(v == (int64_t)i * 100, "value mismatch after growth");
    }

    PASS(test_name);
}

static void test_remove_at(void) {
    const char *test_name = "dyn_array_remove_at: remove middle element";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 1);
    dyn_array_push_int(arr, 2);
    dyn_array_push_int(arr, 3);
    dyn_array_push_int(arr, 4);

    dyn_array_remove_at(arr, 1);  /* Remove '2' */
    ASSERT_EQ(dyn_array_length(arr), 3, "length should be 3 after remove");
    ASSERT_EQ(dyn_array_get_int(arr, 0), 1, "arr[0] should be 1");
    ASSERT_EQ(dyn_array_get_int(arr, 1), 3, "arr[1] should be 3");
    ASSERT_EQ(dyn_array_get_int(arr, 2), 4, "arr[2] should be 4");

    /* Remove first element */
    dyn_array_remove_at(arr, 0);
    ASSERT_EQ(dyn_array_length(arr), 2, "length should be 2");
    ASSERT_EQ(dyn_array_get_int(arr, 0), 3, "arr[0] should be 3");

    /* Remove last element */
    dyn_array_remove_at(arr, 1);
    ASSERT_EQ(dyn_array_length(arr), 1, "length should be 1");

    PASS(test_name);
}

static void test_clear(void) {
    const char *test_name = "dyn_array_clear: clears all elements";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 1);
    dyn_array_push_int(arr, 2);
    dyn_array_push_int(arr, 3);

    int64_t cap_before = dyn_array_capacity(arr);
    dyn_array_clear(arr);

    ASSERT_EQ(dyn_array_length(arr), 0, "length should be 0 after clear");
    ASSERT_EQ(dyn_array_capacity(arr), cap_before, "capacity should not change after clear");

    PASS(test_name);
}

static void test_reserve(void) {
    const char *test_name = "dyn_array_reserve: increases capacity";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 1);
    ASSERT_EQ(dyn_array_length(arr), 1, "length should be 1");

    dyn_array_reserve(arr, 100);
    ASSERT(dyn_array_capacity(arr) >= 100, "capacity should be >= 100 after reserve");
    ASSERT_EQ(dyn_array_length(arr), 1, "length should still be 1 after reserve");
    ASSERT_EQ(dyn_array_get_int(arr, 0), 1, "existing element should be preserved");

    /* Reserve less than current capacity — should be no-op */
    int64_t cap = dyn_array_capacity(arr);
    dyn_array_reserve(arr, 10);
    ASSERT_EQ(dyn_array_capacity(arr), cap, "capacity should not decrease");

    PASS(test_name);
}

static void test_clone_int(void) {
    const char *test_name = "dyn_array_clone: int array deep copy";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 10);
    dyn_array_push_int(arr, 20);
    dyn_array_push_int(arr, 30);

    DynArray *clone = dyn_array_clone(arr);
    ASSERT(clone != NULL, "dyn_array_clone returned NULL");
    ASSERT(clone != arr, "clone should be a different pointer");
    ASSERT_EQ(dyn_array_length(clone), 3, "clone length should be 3");
    ASSERT_EQ(dyn_array_get_int(clone, 0), 10, "clone[0] should be 10");
    ASSERT_EQ(dyn_array_get_int(clone, 1), 20, "clone[1] should be 20");
    ASSERT_EQ(dyn_array_get_int(clone, 2), 30, "clone[2] should be 30");

    /* Modifying original should not affect clone */
    dyn_array_set_int(arr, 0, 999);
    ASSERT_EQ(dyn_array_get_int(clone, 0), 10, "clone should be independent of original");

    PASS(test_name);
}

static void test_nested_arrays(void) {
    const char *test_name = "push/pop/get nested arrays";
    DynArray *outer = dyn_array_new(ELEM_ARRAY);
    ASSERT(outer != NULL, "outer array creation failed");

    DynArray *inner1 = dyn_array_new(ELEM_INT);
    dyn_array_push_int(inner1, 1);
    dyn_array_push_int(inner1, 2);

    DynArray *inner2 = dyn_array_new(ELEM_INT);
    dyn_array_push_int(inner2, 3);
    dyn_array_push_int(inner2, 4);

    dyn_array_push_array(outer, inner1);
    dyn_array_push_array(outer, inner2);
    ASSERT_EQ(dyn_array_length(outer), 2, "outer length should be 2");

    DynArray *got = dyn_array_get_array(outer, 0);
    ASSERT(got != NULL, "get_array should return non-NULL");
    ASSERT_EQ(dyn_array_length(got), 2, "inner array should have 2 elements");
    ASSERT_EQ(dyn_array_get_int(got, 0), 1, "inner[0][0] should be 1");

    /* Test set_array */
    DynArray *inner3 = dyn_array_new(ELEM_INT);
    dyn_array_push_int(inner3, 99);
    dyn_array_set_array(outer, 0, inner3);
    DynArray *got2 = dyn_array_get_array(outer, 0);
    ASSERT(got2 != NULL, "get_array after set should return non-NULL");
    ASSERT_EQ(dyn_array_get_int(got2, 0), 99, "replaced inner[0] should be 99");

    /* Pop nested array */
    bool ok = false;
    DynArray *popped = dyn_array_pop_array(outer, &ok);
    ASSERT(ok, "pop_array should succeed");
    ASSERT(popped != NULL, "popped array should not be NULL");
    ASSERT_EQ(dyn_array_length(outer), 1, "outer length should be 1 after pop");

    PASS(test_name);
}

static void test_struct_operations(void) {
    const char *test_name = "struct push/get/set/pop operations";
    typedef struct { int x; int y; } Point;

    DynArray *arr = dyn_array_new(ELEM_STRUCT);
    ASSERT(arr != NULL, "dyn_array_new struct returned NULL");

    Point p1 = {1, 2};
    Point p2 = {3, 4};
    Point p3 = {5, 6};

    dyn_array_push_struct(arr, &p1, sizeof(Point));
    dyn_array_push_struct(arr, &p2, sizeof(Point));
    dyn_array_push_struct(arr, &p3, sizeof(Point));
    ASSERT_EQ(dyn_array_length(arr), 3, "length should be 3");

    Point *got = (Point *)dyn_array_get_struct(arr, 0);
    ASSERT(got != NULL, "get_struct should return non-NULL");
    ASSERT_EQ(got->x, 1, "p[0].x should be 1");
    ASSERT_EQ(got->y, 2, "p[0].y should be 2");

    /* Set struct */
    Point p_new = {10, 20};
    dyn_array_set_struct(arr, 1, &p_new, sizeof(Point));
    Point *got2 = (Point *)dyn_array_get_struct(arr, 1);
    ASSERT(got2 != NULL, "get_struct after set should return non-NULL");
    ASSERT_EQ(got2->x, 10, "p[1].x should be 10 after set");

    /* Pop struct */
    Point out = {0, 0};
    bool ok = false;
    dyn_array_pop_struct(arr, &out, sizeof(Point), &ok);
    ASSERT(ok, "pop_struct should succeed");
    ASSERT_EQ(out.x, 5, "popped.x should be 5");
    ASSERT_EQ(out.y, 6, "popped.y should be 6");
    ASSERT_EQ(dyn_array_length(arr), 2, "length should be 2 after pop");

    /* Pop from empty fails */
    dyn_array_clear(arr);
    ok = true;
    dyn_array_pop_struct(arr, &out, sizeof(Point), &ok);
    ASSERT(!ok, "pop_struct from empty should fail");

    PASS(test_name);
}

static void test_struct_auto_promote(void) {
    const char *test_name = "struct push: auto-promotes empty ELEM_INT array";
    typedef struct { double val; } Wrapper;

    /* Create an int array but push a struct — should auto-promote */
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");
    ASSERT_EQ(dyn_array_length(arr), 0, "length should be 0");

    Wrapper w = {3.14};
    dyn_array_push_struct(arr, &w, sizeof(Wrapper));
    ASSERT_EQ(dyn_array_length(arr), 1, "length should be 1 after push");
    ASSERT_EQ(dyn_array_get_elem_type(arr), ELEM_STRUCT, "elem_type should be ELEM_STRUCT after auto-promote");

    PASS(test_name);
}

static void test_pop_nested_empty(void) {
    const char *test_name = "pop_array: empty array returns false";
    DynArray *arr = dyn_array_new(ELEM_ARRAY);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    bool ok = true;
    DynArray *result = dyn_array_pop_array(arr, &ok);
    ASSERT(!ok, "pop_array from empty should return false");
    (void)result;

    PASS(test_name);
}

static void test_remove_at_last(void) {
    const char *test_name = "remove_at: remove last element";
    DynArray *arr = dyn_array_new(ELEM_INT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");

    dyn_array_push_int(arr, 10);
    dyn_array_push_int(arr, 20);
    dyn_array_push_int(arr, 30);

    dyn_array_remove_at(arr, 2);  /* Remove last */
    ASSERT_EQ(dyn_array_length(arr), 2, "length should be 2");
    ASSERT_EQ(dyn_array_get_int(arr, 0), 10, "arr[0] should be 10");
    ASSERT_EQ(dyn_array_get_int(arr, 1), 20, "arr[1] should be 20");

    PASS(test_name);
}

static void test_pop_float_empty(void) {
    const char *test_name = "pop float from empty array";
    DynArray *arr = dyn_array_new(ELEM_FLOAT);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");
    bool ok = true;
    dyn_array_pop_float(arr, &ok);
    ASSERT(!ok, "pop from empty should fail");
    PASS(test_name);
}

static void test_pop_bool_empty(void) {
    const char *test_name = "pop bool from empty array";
    DynArray *arr = dyn_array_new(ELEM_BOOL);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");
    bool ok = true;
    dyn_array_pop_bool(arr, &ok);
    ASSERT(!ok, "pop from empty should fail");
    PASS(test_name);
}

static void test_pop_string_empty(void) {
    const char *test_name = "pop string from empty array";
    DynArray *arr = dyn_array_new(ELEM_STRING);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");
    bool ok = true;
    dyn_array_pop_string(arr, &ok);
    ASSERT(!ok, "pop from empty should fail");
    PASS(test_name);
}

static void test_pop_u8_empty(void) {
    const char *test_name = "pop u8 from empty array";
    DynArray *arr = dyn_array_new(ELEM_U8);
    ASSERT(arr != NULL, "dyn_array_new returned NULL");
    bool ok = true;
    dyn_array_pop_u8(arr, &ok);
    ASSERT(!ok, "pop from empty should fail");
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[dyn_array] Dynamic array unit tests...\n\n");

    test_new_int();
    test_new_with_capacity();
    test_push_pop_int();
    test_push_pop_u8();
    test_push_pop_float();
    test_push_pop_bool();
    test_push_pop_string();
    test_push_string_copy();
    test_get_set_int();
    test_get_set_float();
    test_get_set_bool();
    test_get_set_u8();
    test_growth_beyond_initial_capacity();
    test_remove_at();
    test_remove_at_last();
    test_clear();
    test_reserve();
    test_clone_int();
    test_nested_arrays();
    test_struct_operations();
    test_struct_auto_promote();
    test_pop_nested_empty();
    test_pop_float_empty();
    test_pop_bool_empty();
    test_pop_string_empty();
    test_pop_u8_empty();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
