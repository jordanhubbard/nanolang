/**
 * test_builtins_direct.c — direct unit tests for eval builtin functions
 *
 * Calls eval_string.c, eval_math.c, eval_io.c, and eval_hashmap.c
 * builtin functions directly (bypassing the interpreter pipeline) to
 * maximize line coverage of those modules.
 */

#define _POSIX_C_SOURCE 200809L

#include "../src/nanolang.h"
#include "../src/eval/eval_string.h"
#include "../src/eval/eval_math.h"
#include "../src/eval/eval_io.h"
#include "../src/eval/eval_hashmap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { printf("\n    FAILED: %s == %s at line %d (got %lld, expected %lld)\n", \
        #a, #b, __LINE__, (long long)(a), (long long)(b)); exit(1); }
#define ASSERT_STREQ(a, b) \
    if (strcmp((a), (b)) != 0) { printf("\n    FAILED: \"%s\" != \"%s\" at line %d\n", (a), (b), __LINE__); exit(1); }
#define ASSERT_NEAR(a, b, eps) \
    if (fabs((double)(a) - (double)(b)) > (eps)) { \
        printf("\n    FAILED: %f not near %f (eps=%f) at line %d\n", (double)(a), (double)(b), (eps), __LINE__); exit(1); }

/* Required by runtime */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

/* Suppress stderr during expected-error paths */
static FILE *s_orig_stderr = NULL;
static void suppress_stderr(void) {
    fflush(stderr);
    s_orig_stderr = stderr;
    stderr = fopen("/dev/null", "w");
}
static void restore_stderr(void) {
    if (stderr && stderr != s_orig_stderr) fclose(stderr);
    stderr = s_orig_stderr;
    s_orig_stderr = NULL;
}

/* Helper: create a Value with a raw string pointer (GC not needed for args) */
static Value make_str(const char *s) {
    Value v;
    memset(&v, 0, sizeof(v));
    v.type = VAL_STRING;
    v.as.string_val = (char *)s;
    return v;
}

static Value make_int(long long n) {
    Value v;
    memset(&v, 0, sizeof(v));
    v.type = VAL_INT;
    v.as.int_val = n;
    return v;
}

static Value make_float(double f) {
    Value v;
    memset(&v, 0, sizeof(v));
    v.type = VAL_FLOAT;
    v.as.float_val = f;
    return v;
}

/* ============================================================================
 * eval_string.c tests
 * ============================================================================ */

void test_str_length(void) {
    Value args[1] = { make_str("hello") };
    Value r = builtin_str_length(args);
    ASSERT_EQ(r.type, VAL_INT);
    ASSERT_EQ(r.as.int_val, 5);

    args[0] = make_str("");
    r = builtin_str_length(args);
    ASSERT_EQ(r.as.int_val, 0);
}

void test_str_length_type_error(void) {
    Value args[1] = { make_int(42) };
    suppress_stderr();
    Value r = builtin_str_length(args);
    restore_stderr();
    ASSERT_EQ(r.type, VAL_VOID);
}

void test_str_concat(void) {
    Value args[2] = { make_str("foo"), make_str("bar") };
    Value r = builtin_str_concat(args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT_STREQ(r.as.string_val, "foobar");
}

void test_str_concat_type_error(void) {
    Value args[2] = { make_int(1), make_str("bar") };
    suppress_stderr();
    Value r = builtin_str_concat(args);
    restore_stderr();
    ASSERT_EQ(r.type, VAL_VOID);
}

void test_str_substring(void) {
    Value args[3] = { make_str("hello"), make_int(1), make_int(3) };
    Value r = builtin_str_substring(args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT_STREQ(r.as.string_val, "ell");
}

void test_str_substring_edge(void) {
    /* start == end of string, length == 0 → empty string */
    Value args[3] = { make_str("abc"), make_int(3), make_int(0) };
    Value r = builtin_str_substring(args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT_STREQ(r.as.string_val, "");
}

void test_str_substring_errors(void) {
    suppress_stderr();
    /* negative start */
    Value a1[3] = { make_str("hello"), make_int(-1), make_int(2) };
    Value r1 = builtin_str_substring(a1);
    ASSERT_EQ(r1.type, VAL_VOID);

    /* negative length */
    Value a2[3] = { make_str("hello"), make_int(0), make_int(-1) };
    Value r2 = builtin_str_substring(a2);
    ASSERT_EQ(r2.type, VAL_VOID);

    /* start past end */
    Value a3[3] = { make_str("hi"), make_int(5), make_int(1) };
    Value r3 = builtin_str_substring(a3);
    ASSERT_EQ(r3.type, VAL_VOID);

    /* non-string first arg */
    Value a4[3] = { make_int(1), make_int(0), make_int(1) };
    Value r4 = builtin_str_substring(a4);
    ASSERT_EQ(r4.type, VAL_VOID);

    /* non-int second arg */
    Value a5[3] = { make_str("hello"), make_str("x"), make_int(1) };
    Value r5 = builtin_str_substring(a5);
    ASSERT_EQ(r5.type, VAL_VOID);
    restore_stderr();
}

void test_str_contains(void) {
    Value args[2] = { make_str("foobar"), make_str("oba") };
    Value r = builtin_str_contains(args);
    ASSERT(r.as.bool_val == true);

    args[1] = make_str("xyz");
    r = builtin_str_contains(args);
    ASSERT(r.as.bool_val == false);
}

void test_str_equals(void) {
    Value args[2] = { make_str("abc"), make_str("abc") };
    Value r = builtin_str_equals(args);
    ASSERT(r.as.bool_val == true);

    args[1] = make_str("ABC");
    r = builtin_str_equals(args);
    ASSERT(r.as.bool_val == false);
}

void test_str_starts_ends_with(void) {
    Value a1[2] = { make_str("hello world"), make_str("hello") };
    ASSERT(builtin_str_starts_with(a1).as.bool_val == true);

    Value a2[2] = { make_str("hello world"), make_str("world") };
    ASSERT(builtin_str_ends_with(a2).as.bool_val == true);

    Value a3[2] = { make_str("hello"), make_str("hello world") };  /* prefix longer than s */
    ASSERT(builtin_str_starts_with(a3).as.bool_val == false);

    Value a4[2] = { make_str("hi"), make_str("hello") };  /* suffix longer than s */
    ASSERT(builtin_str_ends_with(a4).as.bool_val == false);

    /* empty suffix always matches */
    Value a5[2] = { make_str("abc"), make_str("") };
    ASSERT(builtin_str_ends_with(a5).as.bool_val == true);
}

void test_str_index_of(void) {
    Value args[2] = { make_str("foobar"), make_str("bar") };
    Value r = builtin_str_index_of(args);
    ASSERT_EQ(r.as.int_val, 3);

    args[1] = make_str("xyz");
    r = builtin_str_index_of(args);
    ASSERT_EQ(r.as.int_val, -1);
}

void test_str_trim(void) {
    Value args[1] = { make_str("  hello  ") };
    Value r = builtin_str_trim(args);
    ASSERT_STREQ(r.as.string_val, "hello");
}

void test_str_trim_left_right(void) {
    Value a1[1] = { make_str("  abc") };
    Value r1 = builtin_str_trim_left(a1);
    ASSERT_STREQ(r1.as.string_val, "abc");

    Value a2[1] = { make_str("abc  ") };
    Value r2 = builtin_str_trim_right(a2);
    ASSERT_STREQ(r2.as.string_val, "abc");
}

void test_str_case(void) {
    Value a1[1] = { make_str("Hello World") };
    Value r1 = builtin_str_to_lower(a1);
    ASSERT_STREQ(r1.as.string_val, "hello world");

    Value a2[1] = { make_str("Hello World") };
    Value r2 = builtin_str_to_upper(a2);
    ASSERT_STREQ(r2.as.string_val, "HELLO WORLD");
}

void test_str_replace(void) {
    /* builtin_str_replace: args[0]=haystack, args[1]=needle, args[2]=replacement */
    Value args[3] = { make_str("hello world"), make_str("world"), make_str("nanolang") };
    Value r = builtin_str_replace(args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT(strstr(r.as.string_val, "nanolang") != NULL);
}

void test_str_type_errors(void) {
    suppress_stderr();
    Value bad[2] = { make_int(1), make_str("x") };
    ASSERT_EQ(builtin_str_contains(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_str_equals(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_str_starts_with(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_str_ends_with(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_str_index_of(bad).type, VAL_VOID);
    Value bad1[1] = { make_int(1) };
    ASSERT_EQ(builtin_str_trim(bad1).type, VAL_VOID);
    ASSERT_EQ(builtin_str_trim_left(bad1).type, VAL_VOID);
    ASSERT_EQ(builtin_str_trim_right(bad1).type, VAL_VOID);
    ASSERT_EQ(builtin_str_to_lower(bad1).type, VAL_VOID);
    ASSERT_EQ(builtin_str_to_upper(bad1).type, VAL_VOID);
    restore_stderr();
}

/* ============================================================================
 * eval_math.c tests
 * ============================================================================ */

void test_math_abs(void) {
    Value a1[1] = { make_int(-7) };
    ASSERT_EQ(builtin_abs(a1).as.int_val, 7);

    Value a2[1] = { make_float(-3.5) };
    ASSERT_NEAR(builtin_abs(a2).as.float_val, 3.5, 1e-9);
}

void test_math_min_max(void) {
    Value a1[2] = { make_int(3), make_int(7) };
    ASSERT_EQ(builtin_min(a1).as.int_val, 3);
    ASSERT_EQ(builtin_max(a1).as.int_val, 7);

    Value a2[2] = { make_float(1.5), make_float(2.5) };
    ASSERT_NEAR(builtin_min(a2).as.float_val, 1.5, 1e-9);
    ASSERT_NEAR(builtin_max(a2).as.float_val, 2.5, 1e-9);
}

void test_math_sqrt(void) {
    Value a1[1] = { make_float(9.0) };
    ASSERT_NEAR(builtin_sqrt(a1).as.float_val, 3.0, 1e-9);

    Value a2[1] = { make_int(4) };
    ASSERT_NEAR(builtin_sqrt(a2).as.float_val, 2.0, 1e-9);
}

void test_math_pow(void) {
    Value a1[2] = { make_float(2.0), make_float(10.0) };
    ASSERT_NEAR(builtin_pow(a1).as.float_val, 1024.0, 1e-6);

    Value a2[2] = { make_int(3), make_int(3) };
    ASSERT_NEAR(builtin_pow(a2).as.float_val, 27.0, 1e-9);
}

void test_math_floor_ceil_round(void) {
    Value a1[1] = { make_float(2.7) };
    ASSERT_NEAR(builtin_floor(a1).as.float_val, 2.0, 1e-9);
    ASSERT_NEAR(builtin_ceil(a1).as.float_val, 3.0, 1e-9);
    ASSERT_NEAR(builtin_round(a1).as.float_val, 3.0, 1e-9);

    /* Int versions just return the int */
    Value a2[1] = { make_int(5) };
    ASSERT_EQ(builtin_floor(a2).as.int_val, 5);
    ASSERT_EQ(builtin_ceil(a2).as.int_val, 5);
    ASSERT_EQ(builtin_round(a2).as.int_val, 5);
}

void test_math_trig(void) {
    Value zero[1] = { make_float(0.0) };
    ASSERT_NEAR(builtin_sin(zero).as.float_val, 0.0, 1e-9);
    ASSERT_NEAR(builtin_cos(zero).as.float_val, 1.0, 1e-9);
    ASSERT_NEAR(builtin_tan(zero).as.float_val, 0.0, 1e-9);

    /* atan2(0,1) = 0 */
    Value atan_args[2] = { make_float(0.0), make_float(1.0) };
    ASSERT_NEAR(builtin_atan2(atan_args).as.float_val, 0.0, 1e-9);
}

void test_math_trig_int_args(void) {
    Value i0[1] = { make_int(0) };
    ASSERT_NEAR(builtin_sin(i0).as.float_val, 0.0, 1e-9);
    ASSERT_NEAR(builtin_cos(i0).as.float_val, 1.0, 1e-9);
    ASSERT_NEAR(builtin_tan(i0).as.float_val, 0.0, 1e-9);

    Value atan_args[2] = { make_int(0), make_int(1) };
    ASSERT_NEAR(builtin_atan2(atan_args).as.float_val, 0.0, 1e-9);
}

void test_math_type_errors(void) {
    suppress_stderr();
    Value bad[2] = { make_str("x"), make_str("y") };
    ASSERT_EQ(builtin_abs(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_min(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_max(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_sqrt(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_floor(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_ceil(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_round(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_sin(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_cos(bad).type, VAL_VOID);
    ASSERT_EQ(builtin_tan(bad).type, VAL_VOID);

    Value bad_atan[2] = { make_str("x"), make_float(1.0) };
    ASSERT_EQ(builtin_atan2(bad_atan).type, VAL_VOID);
    Value bad_atan2[2] = { make_float(0.0), make_str("x") };
    ASSERT_EQ(builtin_atan2(bad_atan2).type, VAL_VOID);

    /* pow type errors */
    ASSERT_EQ(builtin_pow(bad).type, VAL_VOID);
    Value bad_pow2[2] = { make_float(2.0), make_str("x") };
    ASSERT_EQ(builtin_pow(bad_pow2).type, VAL_VOID);
    restore_stderr();
}

/* ============================================================================
 * eval_io.c tests
 * ============================================================================ */

void test_io_tmp_dir(void) {
    Value args[1] = { make_str("") };  /* unused */
    Value r = builtin_tmp_dir(args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT(r.as.string_val != NULL && r.as.string_val[0] != '\0');
}

void test_io_getcwd(void) {
    Value args[1] = { make_str("") };  /* unused */
    Value r = builtin_getcwd(args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT(r.as.string_val != NULL && r.as.string_val[0] == '/');
}

void test_io_file_exists(void) {
    Value args_yes[1] = { make_str("/tmp") };
    Value r1 = builtin_file_exists(args_yes);
    ASSERT_EQ(r1.type, VAL_BOOL);
    ASSERT(r1.as.bool_val == true);

    Value args_no[1] = { make_str("/nonexistent_nanolang_test_path_xyz") };
    Value r2 = builtin_file_exists(args_no);
    ASSERT(r2.as.bool_val == false);
}

void test_io_dir_exists(void) {
    Value args_yes[1] = { make_str("/tmp") };
    Value r1 = builtin_dir_exists(args_yes);
    ASSERT(r1.as.bool_val == true);

    Value args_no[1] = { make_str("/nonexistent_dir_xyz_nanolang") };
    Value r2 = builtin_dir_exists(args_no);
    ASSERT(r2.as.bool_val == false);
}

void test_io_path_isfile_isdir(void) {
    Value tmp[1] = { make_str("/tmp") };
    ASSERT(builtin_path_isdir(tmp).as.bool_val == true);
    ASSERT(builtin_path_isfile(tmp).as.bool_val == false);

    Value noexist[1] = { make_str("/no_such_path_nanolang") };
    ASSERT(builtin_path_isdir(noexist).as.bool_val == false);
    ASSERT(builtin_path_isfile(noexist).as.bool_val == false);
}

void test_io_path_join(void) {
    Value a1[2] = { make_str("/foo"), make_str("bar") };
    Value r1 = builtin_path_join(a1);
    ASSERT_STREQ(r1.as.string_val, "/foo/bar");

    /* Already has trailing slash */
    Value a2[2] = { make_str("/foo/"), make_str("bar") };
    Value r2 = builtin_path_join(a2);
    ASSERT_STREQ(r2.as.string_val, "/foo/bar");

    /* Empty first component */
    Value a3[2] = { make_str(""), make_str("bar") };
    Value r3 = builtin_path_join(a3);
    ASSERT_STREQ(r3.as.string_val, "bar");
}

void test_io_path_basename_dirname(void) {
    Value a1[1] = { make_str("/foo/bar/baz.txt") };
    Value r1 = builtin_path_basename(a1);
    ASSERT_STREQ(r1.as.string_val, "baz.txt");

    Value a2[1] = { make_str("/foo/bar/baz.txt") };
    Value r2 = builtin_path_dirname(a2);
    ASSERT_STREQ(r2.as.string_val, "/foo/bar");
}

void test_io_path_normalize(void) {
    Value a1[1] = { make_str("/foo/../bar/./baz") };
    Value r1 = builtin_path_normalize(a1);
    ASSERT_STREQ(r1.as.string_val, "/bar/baz");

    /* Relative path */
    Value a2[1] = { make_str("foo/./bar/../baz") };
    Value r2 = builtin_path_normalize(a2);
    ASSERT_STREQ(r2.as.string_val, "foo/baz");
}

void test_io_file_size_nonexistent(void) {
    Value args[1] = { make_str("/nonexistent_nanolang_file_xyz") };
    Value r = builtin_file_size(args);
    ASSERT_EQ(r.as.int_val, -1);
}

void test_io_dir_list(void) {
    Value args[1] = { make_str("/nonexistent_dir_xyz") };
    suppress_stderr();
    Value r = builtin_dir_list(args);
    restore_stderr();
    /* Returns empty string on failure */
    ASSERT_EQ(r.type, VAL_STRING);
}

void test_io_mktemp(void) {
    Value args[1] = { make_str("nanotest_") };
    Value r = builtin_mktemp(args);
    ASSERT_EQ(r.type, VAL_STRING);
    /* Clean up the temp file */
    if (r.as.string_val[0] != '\0') {
        remove(r.as.string_val);
    }
}

void test_io_mktemp_dir(void) {
    Value args[1] = { make_str("nanotest_dir_") };
    Value r = builtin_mktemp_dir(args);
    ASSERT_EQ(r.type, VAL_STRING);
    /* Clean up */
    if (r.as.string_val[0] != '\0') {
        rmdir(r.as.string_val);
    }
}

void test_io_file_write_read_remove(void) {
    /* Write a temp file, read it back, then remove it */
    Value tmp_args[1] = { make_str("nanotest_io_") };
    Value tmp = builtin_mktemp(tmp_args);
    ASSERT_EQ(tmp.type, VAL_STRING);
    const char *path = tmp.as.string_val;

    Value write_args[2] = { make_str(path), make_str("hello test\n") };
    builtin_file_write(write_args);

    Value read_args[1] = { make_str(path) };
    Value content = builtin_file_read(read_args);
    ASSERT_EQ(content.type, VAL_STRING);
    ASSERT(strstr(content.as.string_val, "hello test") != NULL);

    /* append */
    Value append_args[2] = { make_str(path), make_str("more\n") };
    builtin_file_append(append_args);

    Value content2 = builtin_file_read(read_args);
    ASSERT(strstr(content2.as.string_val, "more") != NULL);

    /* size > 0 */
    Value r = builtin_file_size(read_args);
    ASSERT(r.as.int_val > 0);

    /* remove */
    Value remove_args[1] = { make_str(path) };
    builtin_file_remove(remove_args);

    /* file gone */
    Value gone = builtin_file_exists(read_args);
    ASSERT(gone.as.bool_val == false);
}

void test_io_dir_create_remove(void) {
    Value tmp_args[1] = { make_str("nanotest_dc_") };
    Value tmp = builtin_mktemp_dir(tmp_args);
    ASSERT_EQ(tmp.type, VAL_STRING);
    const char *dir = tmp.as.string_val;

    /* Create a subdir inside */
    char subdir[1024];
    snprintf(subdir, sizeof(subdir), "%s/subdir", dir);
    Value create_args[1] = { make_str(subdir) };
    Value cr = builtin_dir_create(create_args);
    ASSERT_EQ(cr.as.int_val, 0);

    /* Verify it exists */
    Value de = builtin_dir_exists(create_args);
    ASSERT(de.as.bool_val == true);

    /* dir_list the parent */
    Value list_args[1] = { make_str(dir) };
    Value listing = builtin_dir_list(list_args);
    ASSERT(strstr(listing.as.string_val, "subdir") != NULL);

    /* Remove subdir */
    Value rm_args[1] = { make_str(subdir) };
    builtin_dir_remove(rm_args);

    /* Remove parent */
    Value rm_parent[1] = { make_str(dir) };
    builtin_dir_remove(rm_parent);
}

/* ============================================================================
 * eval_io.c — additional uncovered functions
 * ============================================================================ */

void test_io_file_read_bytes(void) {
    /* Write a temp file, read as bytes */
    Value tmp_args[1] = { make_str("nanotest_rb_") };
    Value tmp = builtin_mktemp(tmp_args);
    ASSERT_EQ(tmp.type, VAL_STRING);
    const char *path = tmp.as.string_val;

    Value write_args[2] = { make_str(path), make_str("ABC") };
    builtin_file_write(write_args);

    Value read_args[1] = { make_str(path) };
    Value r = builtin_file_read_bytes(read_args);
    ASSERT_EQ(r.type, VAL_DYN_ARRAY);
    ASSERT(r.as.dyn_array_val != NULL);
    ASSERT(dyn_array_length(r.as.dyn_array_val) == 3);
    ASSERT_EQ(dyn_array_get_int(r.as.dyn_array_val, 0), 'A');
    ASSERT_EQ(dyn_array_get_int(r.as.dyn_array_val, 1), 'B');
    ASSERT_EQ(dyn_array_get_int(r.as.dyn_array_val, 2), 'C');

    /* Non-existent file → empty array */
    Value no_args[1] = { make_str("/no_such_file_nanolang_xyz") };
    Value r2 = builtin_file_read_bytes(no_args);
    ASSERT_EQ(r2.type, VAL_DYN_ARRAY);
    ASSERT(dyn_array_length(r2.as.dyn_array_val) == 0);

    remove(path);
}

void test_io_bytes_string_conversions(void) {
    /* bytes_from_string → string_from_bytes round-trip */
    Value str_args[1] = { make_str("Hello") };
    Value bytes = builtin_bytes_from_string(str_args);
    ASSERT_EQ(bytes.type, VAL_DYN_ARRAY);
    ASSERT(dyn_array_length(bytes.as.dyn_array_val) == 5);
    ASSERT_EQ(dyn_array_get_int(bytes.as.dyn_array_val, 0), 'H');

    /* string_from_bytes */
    Value str2_args[1] = { bytes };
    Value str2 = builtin_string_from_bytes(str2_args);
    ASSERT_EQ(str2.type, VAL_STRING);
    ASSERT(strcmp(str2.as.string_val, "Hello") == 0);

    /* bytes_from_string with non-string → VAL_VOID */
    Value bad_args[1] = { make_int(42) };
    suppress_stderr();
    Value bad = builtin_bytes_from_string(bad_args);
    restore_stderr();
    ASSERT_EQ(bad.type, VAL_VOID);

    /* string_from_bytes with non-array → VAL_VOID */
    Value bad2_args[1] = { make_int(42) };
    suppress_stderr();
    Value bad2 = builtin_string_from_bytes(bad2_args);
    restore_stderr();
    ASSERT_EQ(bad2.type, VAL_VOID);
}

void test_io_file_rename(void) {
    /* Create a temp file, rename it */
    Value tmp_args[1] = { make_str("nanotest_rn_") };
    Value tmp = builtin_mktemp(tmp_args);
    const char *src = tmp.as.string_val;

    Value write_args[2] = { make_str(src), make_str("data") };
    builtin_file_write(write_args);

    char dst[256];
    snprintf(dst, sizeof(dst), "%s_renamed", src);

    Value rename_args[2] = { make_str(src), make_str(dst) };
    Value r = builtin_file_rename(rename_args);
    ASSERT_EQ(r.as.int_val, 0);  /* 0 = success */

    Value exists_args[1] = { make_str(dst) };
    ASSERT(builtin_file_exists(exists_args).as.bool_val == true);

    remove(dst);
}

void test_io_chdir(void) {
    /* Save current dir, chdir to /tmp, restore */
    Value cwd_args[1] = { make_str("") };
    Value orig_cwd = builtin_getcwd(cwd_args);
    ASSERT_EQ(orig_cwd.type, VAL_STRING);

    Value chdir_args[1] = { make_str("/tmp") };
    Value r = builtin_chdir(chdir_args);
    ASSERT(r.as.int_val == 0);

    Value new_cwd = builtin_getcwd(cwd_args);
    ASSERT(strstr(new_cwd.as.string_val, "tmp") != NULL);

    /* Restore */
    Value restore_args[1] = { make_str(orig_cwd.as.string_val) };
    builtin_chdir(restore_args);
}

void test_io_walkdir(void) {
    /* Walk /tmp - should return a non-empty array */
    Value args[1] = { make_str("/tmp") };
    Value r = builtin_fs_walkdir(args);
    ASSERT_EQ(r.type, VAL_DYN_ARRAY);
    ASSERT(r.as.dyn_array_val != NULL);
    /* Just verify it doesn't crash - may have 0+ entries */
}

void test_io_system(void) {
    /* Run a simple command that should succeed (exit 0) */
    Value args[1] = { make_str("true") };
    Value r = builtin_system(args);
    ASSERT_EQ(r.type, VAL_INT);
    ASSERT_EQ(r.as.int_val, 0);
}

void test_io_getenv_setenv_unsetenv(void) {
    /* Set a test env var */
    Value set_args[3] = { make_str("NANO_TEST_VAR_XYZ"), make_str("testval"), make_int(1) };
    Value sr = builtin_setenv(set_args);
    ASSERT_EQ(sr.as.int_val, 0);

    /* Get it back */
    Value get_args[1] = { make_str("NANO_TEST_VAR_XYZ") };
    Value gr = builtin_getenv(get_args);
    ASSERT_EQ(gr.type, VAL_STRING);
    ASSERT(strcmp(gr.as.string_val, "testval") == 0);

    /* Unset */
    Value unset_args[1] = { make_str("NANO_TEST_VAR_XYZ") };
    Value ur = builtin_unsetenv(unset_args);
    ASSERT_EQ(ur.as.int_val, 0);

    /* Now getenv should return empty string */
    Value gr2 = builtin_getenv(get_args);
    ASSERT(strcmp(gr2.as.string_val, "") == 0);
}

void test_io_process_run(void) {
    /* Run echo and verify output */
    Value args[1] = { make_str("echo hello") };
    Value r = builtin_process_run(args);
    ASSERT_EQ(r.type, VAL_DYN_ARRAY);
    ASSERT(dyn_array_length(r.as.dyn_array_val) == 3);

    /* r[0] = exit code string, r[1] = stdout, r[2] = stderr */
    char *code_str = dyn_array_get_string(r.as.dyn_array_val, 0);
    ASSERT(strcmp(code_str, "0") == 0);
    char *stdout_str = dyn_array_get_string(r.as.dyn_array_val, 1);
    ASSERT(strstr(stdout_str, "hello") != NULL);
}

void test_io_result_non_union(void) {
    /* result_is_ok / result_is_err on non-union → false */
    Value args[1] = { make_int(42) };
    ASSERT(builtin_result_is_ok(args).as.bool_val == false);
    ASSERT(builtin_result_is_err(args).as.bool_val == false);

    /* result_unwrap_or on non-union → returns default */
    Value or_args[2] = { make_int(42), make_str("default") };
    Value r = builtin_result_unwrap_or(or_args);
    ASSERT_EQ(r.type, VAL_STRING);
    ASSERT(strcmp(r.as.string_val, "default") == 0);
}

void test_io_result_with_union(void) {
    /* Create an Ok union value manually */
    char *field_names[1] = { "value" };
    Value field_values[1] = { make_int(99) };
    Value ok_val = create_union("Result", 0, "Ok", field_names, field_values, 1);
    ASSERT_EQ(ok_val.type, VAL_UNION);

    /* result_is_ok → true, result_is_err → false */
    Value args[1] = { ok_val };
    ASSERT(builtin_result_is_ok(args).as.bool_val == true);
    ASSERT(builtin_result_is_err(args).as.bool_val == false);

    /* result_unwrap → returns the value */
    Value unwrapped = builtin_result_unwrap(args);
    ASSERT_EQ(unwrapped.as.int_val, 99);

    /* result_unwrap_or with Ok → returns Ok value */
    Value or_args[2] = { ok_val, make_int(0) };
    Value or_result = builtin_result_unwrap_or(or_args);
    ASSERT_EQ(or_result.as.int_val, 99);

    /* Create an Err union value */
    char *err_names[1] = { "error" };
    Value err_values[1] = { make_str("oh no") };
    Value err_val = create_union("Result", 1, "Err", err_names, err_values, 1);

    /* result_is_err → true, result_is_ok → false */
    Value err_args[1] = { err_val };
    ASSERT(builtin_result_is_err(err_args).as.bool_val == true);
    ASSERT(builtin_result_is_ok(err_args).as.bool_val == false);

    /* result_unwrap_err → returns the error value */
    Value err_unwrapped = builtin_result_unwrap_err(err_args);
    ASSERT_EQ(err_unwrapped.type, VAL_STRING);
    ASSERT(strcmp(err_unwrapped.as.string_val, "oh no") == 0);

    /* result_unwrap_or on Err → returns default */
    Value err_or_args[2] = { err_val, make_int(42) };
    Value err_or = builtin_result_unwrap_or(err_or_args);
    ASSERT_EQ(err_or.as.int_val, 42);
}

/* ============================================================================
 * eval_hashmap.c tests
 * ============================================================================ */

void test_hm_hash_functions(void) {
    /* nl_hm_hash_string: different strings → different hashes */
    uint64_t h1 = nl_hm_hash_string("hello");
    uint64_t h2 = nl_hm_hash_string("world");
    ASSERT(h1 != h2);

    uint64_t h_null = nl_hm_hash_string(NULL);
    ASSERT_EQ(h_null, 0);

    /* nl_hm_hash_int */
    uint64_t hi1 = nl_hm_hash_int(42);
    uint64_t hi2 = nl_hm_hash_int(43);
    ASSERT(hi1 != hi2);
}

void test_hm_parse_monomorph(void) {
    NLHashMapKeyType k;
    NLHashMapValType v;

    ASSERT(nl_hm_parse_monomorph("HashMap_string_int", &k, &v));
    ASSERT_EQ(k, NL_HM_KEY_STRING);
    ASSERT_EQ(v, NL_HM_VAL_INT);

    ASSERT(nl_hm_parse_monomorph("HashMap_int_string", &k, &v));
    ASSERT_EQ(k, NL_HM_KEY_INT);
    ASSERT_EQ(v, NL_HM_VAL_STRING);

    ASSERT(nl_hm_parse_monomorph("HashMap_int_int", &k, &v));
    ASSERT_EQ(k, NL_HM_KEY_INT);
    ASSERT_EQ(v, NL_HM_VAL_INT);

    ASSERT(nl_hm_parse_monomorph("HashMap_string_string", &k, &v));
    ASSERT_EQ(k, NL_HM_KEY_STRING);
    ASSERT_EQ(v, NL_HM_VAL_STRING);

    /* Invalid forms */
    ASSERT(!nl_hm_parse_monomorph(NULL, &k, &v));
    ASSERT(!nl_hm_parse_monomorph("NotAHashMap", &k, &v));
    ASSERT(!nl_hm_parse_monomorph("HashMap_", &k, &v));
    ASSERT(!nl_hm_parse_monomorph("HashMap_float_int", &k, &v));
    ASSERT(!nl_hm_parse_monomorph("HashMap_int_float", &k, &v));
}

void test_hm_alloc_free(void) {
    NLHashMapCore *hm = nl_hm_alloc(NL_HM_KEY_STRING, NL_HM_VAL_INT, 0);
    ASSERT(hm != NULL);
    ASSERT_EQ(hm->capacity, 16);  /* min capacity */
    ASSERT_EQ(hm->size, 0);
    nl_hm_free(hm);

    NLHashMapCore *hm2 = nl_hm_alloc(NL_HM_KEY_INT, NL_HM_VAL_STRING, 100);
    ASSERT(hm2 != NULL);
    ASSERT(hm2->capacity >= 100);
    nl_hm_free(hm2);
}

void test_hm_key_equals(void) {
    NLHashMapCore *hm = nl_hm_alloc(NL_HM_KEY_INT, NL_HM_VAL_INT, 16);
    NLHashMapEntry e;
    memset(&e, 0, sizeof(e));
    e.key.i = 42;

    Value key = make_int(42);
    ASSERT(nl_hm_key_equals(hm, &e, &key));

    Value key2 = make_int(99);
    ASSERT(!nl_hm_key_equals(hm, &e, &key2));

    /* NULL safety */
    ASSERT(!nl_hm_key_equals(NULL, &e, &key));
    ASSERT(!nl_hm_key_equals(hm, NULL, &key));
    ASSERT(!nl_hm_key_equals(hm, &e, NULL));

    nl_hm_free(hm);
}

void test_hm_find_slot(void) {
    NLHashMapCore *hm = nl_hm_alloc(NL_HM_KEY_INT, NL_HM_VAL_INT, 16);

    Value key = make_int(7);
    bool found = false;
    int64_t slot = nl_hm_find_slot(hm, &key, &found);
    ASSERT(!found);
    ASSERT(slot >= 0 && slot < hm->capacity);

    nl_hm_free(hm);
}

void test_hm_clear(void) {
    NLHashMapCore *hm = nl_hm_alloc(NL_HM_KEY_INT, NL_HM_VAL_INT, 16);
    /* Mark a few slots as filled */
    hm->entries[0].state = 1;
    hm->entries[0].key.i = 1;
    hm->entries[0].value.i = 100;
    hm->size = 1;

    nl_hm_clear(hm);
    ASSERT_EQ(hm->size, 0);
    ASSERT_EQ(hm->entries[0].state, 0);

    nl_hm_free(hm);
}

void test_hm_free_null(void) {
    nl_hm_free(NULL);  /* should not crash */
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Builtin Functions Direct Tests ===\n");

    printf("\n--- eval_string.c ---\n");
    TEST(str_length);
    TEST(str_length_type_error);
    TEST(str_concat);
    TEST(str_concat_type_error);
    TEST(str_substring);
    TEST(str_substring_edge);
    TEST(str_substring_errors);
    TEST(str_contains);
    TEST(str_equals);
    TEST(str_starts_ends_with);
    TEST(str_index_of);
    TEST(str_trim);
    TEST(str_trim_left_right);
    TEST(str_case);
    TEST(str_replace);
    TEST(str_type_errors);

    printf("\n--- eval_math.c ---\n");
    TEST(math_abs);
    TEST(math_min_max);
    TEST(math_sqrt);
    TEST(math_pow);
    TEST(math_floor_ceil_round);
    TEST(math_trig);
    TEST(math_trig_int_args);
    TEST(math_type_errors);

    printf("\n--- eval_io.c ---\n");
    TEST(io_tmp_dir);
    TEST(io_getcwd);
    TEST(io_file_exists);
    TEST(io_dir_exists);
    TEST(io_path_isfile_isdir);
    TEST(io_path_join);
    TEST(io_path_basename_dirname);
    TEST(io_path_normalize);
    TEST(io_file_size_nonexistent);
    TEST(io_dir_list);
    TEST(io_mktemp);
    TEST(io_mktemp_dir);
    TEST(io_file_write_read_remove);
    TEST(io_dir_create_remove);
    TEST(io_file_read_bytes);
    TEST(io_bytes_string_conversions);
    TEST(io_file_rename);
    TEST(io_chdir);
    TEST(io_walkdir);
    TEST(io_system);
    TEST(io_getenv_setenv_unsetenv);
    TEST(io_process_run);
    TEST(io_result_non_union);
    TEST(io_result_with_union);

    printf("\n--- eval_hashmap.c ---\n");
    TEST(hm_hash_functions);
    TEST(hm_parse_monomorph);
    TEST(hm_alloc_free);
    TEST(hm_key_equals);
    TEST(hm_find_slot);
    TEST(hm_clear);
    TEST(hm_free_null);

    printf("\n✓ All builtin direct tests passed!\n");
    return 0;
}
