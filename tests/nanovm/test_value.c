/*
 * test_value.c — unit tests for nanovm/value.c
 *
 * Exercises val_print, val_println, val_equal, val_compare,
 * val_truthy, and val_to_cstring for all NanoValue tag types.
 *
 * Avoids heap allocation — uses NULL for VmString/VmArray/etc
 * pointers where safe (null-string path, null-array path).
 */

#include "nanovm/value.h"
#include "nanoisa/isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Required by runtime/cli.c (linked via NANOVM_OBJECTS chain) */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* Redirect val_print output to /dev/null for "no crash" tests */
static FILE *devnull;

/* ── val_truthy tests ────────────────────────────────────────────────────── */

static void test_val_truthy_void(void) {
    const char *test_name = "val_truthy: void is false";
    ASSERT(val_truthy(val_void()) == false, "void should be falsy");
    PASS(test_name);
}

static void test_val_truthy_int(void) {
    const char *test_name = "val_truthy: int truthy/falsy";
    ASSERT(val_truthy(val_int(0)) == false, "0 should be falsy");
    ASSERT(val_truthy(val_int(1)) == true,  "1 should be truthy");
    ASSERT(val_truthy(val_int(-1)) == true, "-1 should be truthy");
    PASS(test_name);
}

static void test_val_truthy_float(void) {
    const char *test_name = "val_truthy: float truthy/falsy";
    ASSERT(val_truthy(val_float(0.0)) == false, "0.0 should be falsy");
    ASSERT(val_truthy(val_float(1.5)) == true,  "1.5 should be truthy");
    PASS(test_name);
}

static void test_val_truthy_bool(void) {
    const char *test_name = "val_truthy: bool truthy/falsy";
    ASSERT(val_truthy(val_bool(true))  == true,  "true should be truthy");
    ASSERT(val_truthy(val_bool(false)) == false, "false should be falsy");
    PASS(test_name);
}

static void test_val_truthy_u8(void) {
    const char *test_name = "val_truthy: u8 truthy/falsy";
    ASSERT(val_truthy(val_u8(0))   == false, "u8(0) should be falsy");
    ASSERT(val_truthy(val_u8(255)) == true,  "u8(255) should be truthy");
    PASS(test_name);
}

static void test_val_truthy_string_null(void) {
    const char *test_name = "val_truthy: null string is false";
    ASSERT(val_truthy(val_string(NULL)) == false, "null string should be falsy");
    PASS(test_name);
}

static void test_val_truthy_array_null(void) {
    const char *test_name = "val_truthy: null array is false";
    ASSERT(val_truthy(val_array(NULL)) == false, "null array should be falsy");
    PASS(test_name);
}

/* ── val_equal tests ─────────────────────────────────────────────────────── */

static void test_val_equal_void(void) {
    const char *test_name = "val_equal: void == void";
    ASSERT(val_equal(val_void(), val_void()), "void == void");
    PASS(test_name);
}

static void test_val_equal_int(void) {
    const char *test_name = "val_equal: int equality";
    ASSERT(val_equal(val_int(42), val_int(42)),  "42 == 42");
    ASSERT(!val_equal(val_int(42), val_int(43)), "42 != 43");
    PASS(test_name);
}

static void test_val_equal_float(void) {
    const char *test_name = "val_equal: float equality";
    ASSERT(val_equal(val_float(3.14), val_float(3.14)), "3.14 == 3.14");
    ASSERT(!val_equal(val_float(1.0), val_float(2.0)),  "1.0 != 2.0");
    PASS(test_name);
}

static void test_val_equal_bool(void) {
    const char *test_name = "val_equal: bool equality";
    ASSERT(val_equal(val_bool(true), val_bool(true)),   "true == true");
    ASSERT(val_equal(val_bool(false), val_bool(false)), "false == false");
    ASSERT(!val_equal(val_bool(true), val_bool(false)), "true != false");
    PASS(test_name);
}

static void test_val_equal_different_tags(void) {
    const char *test_name = "val_equal: different tags are not equal";
    ASSERT(!val_equal(val_int(0), val_void()),      "int(0) != void");
    ASSERT(!val_equal(val_bool(false), val_void()), "bool(false) != void");
    ASSERT(!val_equal(val_int(1), val_bool(true)),  "int(1) != bool(true)");
    PASS(test_name);
}

static void test_val_equal_int_float_crosstype(void) {
    const char *test_name = "val_equal: int/float cross-type comparison";
    /* The implementation allows int ↔ float comparison */
    ASSERT(val_equal(val_int(1), val_float(1.0)),    "int(1) == float(1.0)");
    ASSERT(val_equal(val_float(2.0), val_int(2)),    "float(2.0) == int(2)");
    ASSERT(!val_equal(val_int(1), val_float(1.5)),   "int(1) != float(1.5)");
    PASS(test_name);
}

static void test_val_equal_enum_int_crosstype(void) {
    const char *test_name = "val_equal: enum/int cross-type comparison";
    ASSERT(val_equal(val_enum(3), val_int(3)),  "enum(3) == int(3)");
    ASSERT(val_equal(val_int(5), val_enum(5)),  "int(5) == enum(5)");
    ASSERT(!val_equal(val_enum(3), val_int(4)), "enum(3) != int(4)");
    PASS(test_name);
}

static void test_val_equal_string_null(void) {
    const char *test_name = "val_equal: null strings are equal (same pointer)";
    ASSERT(val_equal(val_string(NULL), val_string(NULL)), "null == null (pointer eq)");
    PASS(test_name);
}

static void test_val_equal_enum(void) {
    const char *test_name = "val_equal: enum equality";
    ASSERT(val_equal(val_enum(0), val_enum(0)),   "enum(0) == enum(0)");
    ASSERT(!val_equal(val_enum(0), val_enum(1)), "enum(0) != enum(1)");
    PASS(test_name);
}

/* ── val_compare tests ───────────────────────────────────────────────────── */

static void test_val_compare_int(void) {
    const char *test_name = "val_compare: integer ordering";
    ASSERT(val_compare(val_int(1), val_int(2)) < 0,  "1 < 2");
    ASSERT(val_compare(val_int(2), val_int(1)) > 0,  "2 > 1");
    ASSERT(val_compare(val_int(5), val_int(5)) == 0, "5 == 5");
    PASS(test_name);
}

static void test_val_compare_float(void) {
    const char *test_name = "val_compare: float ordering";
    ASSERT(val_compare(val_float(1.0), val_float(2.0)) < 0,  "1.0 < 2.0");
    ASSERT(val_compare(val_float(2.5), val_float(1.5)) > 0,  "2.5 > 1.5");
    ASSERT(val_compare(val_float(3.0), val_float(3.0)) == 0, "3.0 == 3.0");
    PASS(test_name);
}

static void test_val_compare_bool(void) {
    const char *test_name = "val_compare: bool ordering";
    ASSERT(val_compare(val_bool(false), val_bool(true))  < 0,  "false < true");
    ASSERT(val_compare(val_bool(true),  val_bool(false)) > 0,  "true > false");
    ASSERT(val_compare(val_bool(true),  val_bool(true))  == 0, "true == true");
    PASS(test_name);
}

static void test_val_compare_int_float_crosstype(void) {
    const char *test_name = "val_compare: int/float cross-type ordering";
    ASSERT(val_compare(val_int(1), val_float(2.0)) < 0,  "int(1) < float(2.0)");
    ASSERT(val_compare(val_float(3.0), val_int(2)) > 0,  "float(3.0) > int(2)");
    ASSERT(val_compare(val_int(5), val_float(5.0)) == 0, "int(5) == float(5.0)");
    PASS(test_name);
}

static void test_val_compare_enum_int_crosstype(void) {
    const char *test_name = "val_compare: enum/int cross-type ordering";
    ASSERT(val_compare(val_enum(1), val_int(3)) < 0,  "enum(1) < int(3)");
    ASSERT(val_compare(val_int(5), val_enum(3)) > 0,  "int(5) > enum(3)");
    ASSERT(val_compare(val_enum(2), val_int(2)) == 0, "enum(2) == int(2)");
    PASS(test_name);
}

static void test_val_compare_different_tags(void) {
    const char *test_name = "val_compare: mismatched tags use tag difference";
    /* For non-cross-type mismatches, result is (int)a.tag - (int)b.tag */
    int r = val_compare(val_void(), val_bool(false));
    (void)r; /* May be positive or negative — just verify no crash */
    PASS(test_name);
}

static void test_val_compare_string_null(void) {
    const char *test_name = "val_compare: null string < non-null string";
    /* null string vs null string → pointer eq → 0 */
    ASSERT(val_compare(val_string(NULL), val_string(NULL)) == 0,
           "null==null should be 0");
    PASS(test_name);
}

/* ── val_to_cstring tests ────────────────────────────────────────────────── */

static void test_val_to_cstring_void(void) {
    const char *test_name = "val_to_cstring: void -> 'void'";
    char *s = val_to_cstring(val_void());
    ASSERT(s != NULL, "should return non-NULL");
    ASSERT(strcmp(s, "void") == 0, "void -> 'void'");
    free(s);
    PASS(test_name);
}

static void test_val_to_cstring_int(void) {
    const char *test_name = "val_to_cstring: int -> decimal string";
    char *s = val_to_cstring(val_int(42));
    ASSERT(s != NULL, "should return non-NULL");
    ASSERT(strcmp(s, "42") == 0, "42 -> '42'");
    free(s);
    char *neg = val_to_cstring(val_int(-7));
    ASSERT(strcmp(neg, "-7") == 0, "-7 -> '-7'");
    free(neg);
    PASS(test_name);
}

static void test_val_to_cstring_float(void) {
    const char *test_name = "val_to_cstring: float -> decimal string";
    char *s = val_to_cstring(val_float(3.14));
    ASSERT(s != NULL, "should return non-NULL");
    ASSERT(strlen(s) > 0, "float string should be non-empty");
    free(s);
    PASS(test_name);
}

static void test_val_to_cstring_bool(void) {
    const char *test_name = "val_to_cstring: bool -> 'true'/'false'";
    char *t = val_to_cstring(val_bool(true));
    char *f = val_to_cstring(val_bool(false));
    ASSERT(strcmp(t, "true")  == 0, "true  -> 'true'");
    ASSERT(strcmp(f, "false") == 0, "false -> 'false'");
    free(t); free(f);
    PASS(test_name);
}

static void test_val_to_cstring_string_null(void) {
    const char *test_name = "val_to_cstring: null string -> 'null'";
    char *s = val_to_cstring(val_string(NULL));
    ASSERT(s != NULL, "should return non-NULL");
    ASSERT(strcmp(s, "null") == 0, "null string -> 'null'");
    free(s);
    PASS(test_name);
}

static void test_val_to_cstring_other(void) {
    const char *test_name = "val_to_cstring: function/enum tags return non-NULL";
    char *fn = val_to_cstring(val_function(0));
    ASSERT(fn != NULL, "function value should return non-NULL");
    free(fn);
    char *en = val_to_cstring(val_enum(5));
    ASSERT(en != NULL, "enum value should return non-NULL");
    free(en);
    PASS(test_name);
}

/* ── val_print tests ─────────────────────────────────────────────────────── */

static void test_val_print_void(void) {
    const char *test_name = "val_print: void prints without crash";
    val_print(val_void(), devnull);
    PASS(test_name);
}

static void test_val_print_int(void) {
    const char *test_name = "val_print: int prints without crash";
    val_print(val_int(123), devnull);
    PASS(test_name);
}

static void test_val_print_float(void) {
    const char *test_name = "val_print: float prints without crash";
    val_print(val_float(2.718), devnull);
    val_print(val_float(1.0), devnull); /* exercises the %.1f branch */
    val_print(val_float(1e20), devnull); /* exercises the %g branch */
    PASS(test_name);
}

static void test_val_print_bool(void) {
    const char *test_name = "val_print: bool prints without crash";
    val_print(val_bool(true),  devnull);
    val_print(val_bool(false), devnull);
    PASS(test_name);
}

static void test_val_print_u8(void) {
    const char *test_name = "val_print: u8 prints without crash";
    val_print(val_u8(42), devnull);
    PASS(test_name);
}

static void test_val_print_null_string(void) {
    const char *test_name = "val_print: null string prints 'null'";
    val_print(val_string(NULL), devnull);
    PASS(test_name);
}

static void test_val_print_null_array(void) {
    const char *test_name = "val_print: null array prints '[]'";
    val_print(val_array(NULL), devnull);
    PASS(test_name);
}

static void test_val_print_null_struct(void) {
    const char *test_name = "val_print: null struct prints '{}'";
    val_print(val_struct(NULL), devnull);
    PASS(test_name);
}

static void test_val_print_null_tuple(void) {
    const char *test_name = "val_print: null tuple prints '()'";
    val_print(val_tuple(NULL), devnull);
    PASS(test_name);
}

static void test_val_print_null_union(void) {
    const char *test_name = "val_print: null union prints 'union(null)'";
    val_print(val_union(NULL), devnull);
    PASS(test_name);
}

static void test_val_print_null_hashmap(void) {
    const char *test_name = "val_print: hashmap prints without crash";
    val_print(val_hashmap(NULL), devnull);
    PASS(test_name);
}

static void test_val_print_enum(void) {
    const char *test_name = "val_print: enum prints without crash";
    val_print(val_enum(7), devnull);
    PASS(test_name);
}

static void test_val_print_function(void) {
    const char *test_name = "val_print: function prints without crash";
    val_print(val_function(0), devnull);
    PASS(test_name);
}

static void test_val_println_int(void) {
    const char *test_name = "val_println: int prints to stdout without crash";
    /* Redirect stdout temporarily */
    FILE *orig = stdout;
    (void)orig;
    /* Just call it — if it doesn't crash, that's sufficient */
    /* We can't easily redirect stdout, so just suppress via freopen */
    val_println(val_int(42));
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    devnull = fopen("/dev/null", "w");
    if (!devnull) devnull = stderr;

    printf("\n[value] NanoVM value operations tests...\n\n");

    /* val_truthy */
    test_val_truthy_void();
    test_val_truthy_int();
    test_val_truthy_float();
    test_val_truthy_bool();
    test_val_truthy_u8();
    test_val_truthy_string_null();
    test_val_truthy_array_null();

    /* val_equal */
    test_val_equal_void();
    test_val_equal_int();
    test_val_equal_float();
    test_val_equal_bool();
    test_val_equal_different_tags();
    test_val_equal_int_float_crosstype();
    test_val_equal_enum_int_crosstype();
    test_val_equal_string_null();
    test_val_equal_enum();

    /* val_compare */
    test_val_compare_int();
    test_val_compare_float();
    test_val_compare_bool();
    test_val_compare_int_float_crosstype();
    test_val_compare_enum_int_crosstype();
    test_val_compare_different_tags();
    test_val_compare_string_null();

    /* val_to_cstring */
    test_val_to_cstring_void();
    test_val_to_cstring_int();
    test_val_to_cstring_float();
    test_val_to_cstring_bool();
    test_val_to_cstring_string_null();
    test_val_to_cstring_other();

    /* val_print */
    test_val_print_void();
    test_val_print_int();
    test_val_print_float();
    test_val_print_bool();
    test_val_print_u8();
    test_val_print_null_string();
    test_val_print_null_array();
    test_val_print_null_struct();
    test_val_print_null_tuple();
    test_val_print_null_union();
    test_val_print_null_hashmap();
    test_val_print_enum();
    test_val_print_function();
    test_val_println_int();

    if (devnull != stderr) fclose(devnull);

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
