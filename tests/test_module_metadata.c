/**
 * test_module_metadata.c — unit tests for module_metadata.c
 *
 * Tests serialize_module_metadata_to_c, embed_metadata_in_module_c, and
 * deserialize_module_metadata_from_c with a variety of module shapes.
 */

#include "../src/nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#define TEST(name) printf("  Testing %s...", #name); test_##name(); printf(" ✓\n")
#define ASSERT(cond) \
    if (!(cond)) { printf("\n    FAILED: %s at line %d\n", #cond, __LINE__); exit(1); }
#define ASSERT_NOT_NULL(p) \
    if ((p) == NULL) { printf("\n    FAILED: unexpected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_NULL(p) \
    if ((p) != NULL) { printf("\n    FAILED: expected NULL at line %d\n", __LINE__); exit(1); }
#define ASSERT_CONTAINS(haystack, needle) \
    if (strstr((haystack), (needle)) == NULL) { \
        printf("\n    FAILED: expected \"%s\" in output at line %d\n    got:\n%s\n", \
            (needle), __LINE__, (haystack)); exit(1); }

/* Required by runtime/cli.c (extern in eval.c) */
int g_argc = 0;
char **g_argv = NULL;

/* Helper: build a zero-initialised ModuleMetadata on the stack */
static ModuleMetadata make_empty_meta(const char *name) {
    ModuleMetadata m;
    memset(&m, 0, sizeof(m));
    m.module_name = (char *)name;
    return m;
}

/* Helper: build a single void→void function entry */
static Function make_simple_fn(const char *name) {
    Function f;
    memset(&f, 0, sizeof(f));
    f.name = (char *)name;
    f.return_type = TYPE_VOID;
    return f;
}

/* ============================================================================
 * Tests
 * ============================================================================ */

void test_serialize_null_returns_null(void) {
    char *result = serialize_module_metadata_to_c(NULL);
    ASSERT_NULL(result);
}

void test_serialize_empty_module(void) {
    ModuleMetadata meta = make_empty_meta("mymod");
    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "mymod");
    ASSERT_CONTAINS(out, "#include \"nanolang.h\"");
    ASSERT_CONTAINS(out, "_module_functions[0]");  /* static array of size 0 */
    free(out);
}

void test_serialize_single_void_function(void) {
    Function fn = make_simple_fn("my_func");
    ModuleMetadata meta = make_empty_meta("testmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "my_func");
    ASSERT_CONTAINS(out, "function_count = 1");
    ASSERT_CONTAINS(out, "return_type");
    ASSERT_CONTAINS(out, "_init_module_metadata");
    free(out);
}

void test_serialize_function_is_extern(void) {
    Function fn = make_simple_fn("ext_fn");
    fn.is_extern = true;
    ModuleMetadata meta = make_empty_meta("extmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "is_extern = true");
    free(out);
}

void test_serialize_function_with_params(void) {
    Parameter params[2];
    memset(params, 0, sizeof(params));
    params[0].name = "x";
    params[0].type = TYPE_INT;
    params[1].name = "s";
    params[1].type = TYPE_STRING;

    Function fn = make_simple_fn("add");
    fn.param_count = 2;
    fn.params = params;
    fn.return_type = TYPE_INT;

    ModuleMetadata meta = make_empty_meta("mathmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "\"x\"");
    ASSERT_CONTAINS(out, "\"s\"");
    ASSERT_CONTAINS(out, "param_count = 2");
    free(out);
}

void test_serialize_function_with_struct_return(void) {
    Function fn = make_simple_fn("get_point");
    fn.return_type = TYPE_STRUCT;
    fn.return_struct_type_name = "Point";

    ModuleMetadata meta = make_empty_meta("geomod");
    meta.function_count = 1;
    meta.functions = &fn;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "\"Point\"");
    ASSERT_CONTAINS(out, "return_struct_type_name");
    free(out);
}

void test_serialize_memory_annotations(void) {
    Function fn = make_simple_fn("alloc_handle");
    fn.returns_gc_managed = true;
    fn.requires_manual_free = true;
    fn.cleanup_function = "handle_free";
    fn.returns_borrowed = false;

    ModuleMetadata meta = make_empty_meta("memmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "returns_gc_managed = true");
    ASSERT_CONTAINS(out, "requires_manual_free = true");
    ASSERT_CONTAINS(out, "\"handle_free\"");
    ASSERT_CONTAINS(out, "returns_borrowed = false");
    free(out);
}

void test_serialize_int_constant(void) {
    ConstantDef c;
    memset(&c, 0, sizeof(c));
    c.name = "MAX_SIZE";
    c.type = TYPE_INT;
    c.value = 1024;

    ModuleMetadata meta = make_empty_meta("constmod");
    meta.constant_count = 1;
    meta.constants = &c;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "MAX_SIZE");
    ASSERT_CONTAINS(out, "1024");
    free(out);
}

void test_serialize_float_constant(void) {
    ConstantDef c;
    memset(&c, 0, sizeof(c));
    c.name = "PI";
    c.type = TYPE_FLOAT;
    /* Store 3.14159 as bit pattern in int64 */
    union { double d; int64_t i; } u;
    u.d = 3.14159;
    c.value = u.i;

    ModuleMetadata meta = make_empty_meta("mathconst");
    meta.constant_count = 1;
    meta.constants = &c;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "PI");
    ASSERT_CONTAINS(out, "const double");
    free(out);
}

void test_serialize_multiple_functions(void) {
    Function fns[3];
    memset(fns, 0, sizeof(fns));
    fns[0].name = "alpha";  fns[0].return_type = TYPE_VOID;
    fns[1].name = "beta";   fns[1].return_type = TYPE_INT;
    fns[2].name = "gamma";  fns[2].return_type = TYPE_STRING;

    ModuleMetadata meta = make_empty_meta("multimod");
    meta.function_count = 3;
    meta.functions = fns;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "\"alpha\"");
    ASSERT_CONTAINS(out, "\"beta\"");
    ASSERT_CONTAINS(out, "\"gamma\"");
    ASSERT_CONTAINS(out, "function_count = 3");
    free(out);
}

void test_serialize_param_with_struct_type(void) {
    Parameter p;
    memset(&p, 0, sizeof(p));
    p.name = "pt";
    p.type = TYPE_STRUCT;
    p.struct_type_name = "Vec3";

    Function fn = make_simple_fn("normalize");
    fn.param_count = 1;
    fn.params = &p;

    ModuleMetadata meta = make_empty_meta("vecmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char *out = serialize_module_metadata_to_c(&meta);
    ASSERT_NOT_NULL(out);
    ASSERT_CONTAINS(out, "\"Vec3\"");
    ASSERT_CONTAINS(out, "\"pt\"");
    free(out);
}

void test_embed_null_inputs_return_false(void) {
    ASSERT(!embed_metadata_in_module_c(NULL, NULL, 0));

    ModuleMetadata meta = make_empty_meta("x");
    ASSERT(!embed_metadata_in_module_c(NULL, &meta, 1024));

    char buf[16] = "code";
    ASSERT(!embed_metadata_in_module_c(buf, NULL, sizeof(buf)));
}

void test_embed_buffer_too_small(void) {
    /* Buffer must be large enough: code + metadata + 100 bytes */
    Function fn = make_simple_fn("f");
    ModuleMetadata meta = make_empty_meta("small");
    meta.function_count = 1;
    meta.functions = &fn;

    char small_buf[32] = "int main() {}";
    bool ok = embed_metadata_in_module_c(small_buf, &meta, sizeof(small_buf));
    ASSERT(!ok);  /* too small */
}

void test_embed_metadata_at_end(void) {
    /* Code without main() — metadata goes at the end */
    Function fn = make_simple_fn("helper");
    ModuleMetadata meta = make_empty_meta("embedmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char buf[65536] = "/* preamble */\n";
    bool ok = embed_metadata_in_module_c(buf, &meta, sizeof(buf));
    ASSERT(ok);
    ASSERT(strstr(buf, "helper") != NULL);
    ASSERT(strstr(buf, "_init_module_metadata") != NULL);
}

void test_embed_metadata_before_main(void) {
    /* Code with int main() — metadata should be inserted before it */
    Function fn = make_simple_fn("do_work");
    ModuleMetadata meta = make_empty_meta("mainmod");
    meta.function_count = 1;
    meta.functions = &fn;

    char buf[65536];
    snprintf(buf, sizeof(buf), "/* top */\nint main() { return 0; }\n");
    bool ok = embed_metadata_in_module_c(buf, &meta, sizeof(buf));
    ASSERT(ok);
    /* Metadata should appear before int main() */
    char *meta_pos = strstr(buf, "do_work");
    char *main_pos = strstr(buf, "int main()");
    ASSERT_NOT_NULL(meta_pos);
    ASSERT_NOT_NULL(main_pos);
    ASSERT(meta_pos < main_pos);
}

void test_deserialize_stub_returns_false(void) {
    ModuleMetadata *out = NULL;
    bool ok = deserialize_module_metadata_from_c("/* some c code */", &out);
    ASSERT(!ok);
    ASSERT_NULL(out);
}

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("=== Module Metadata Tests ===\n");
    TEST(serialize_null_returns_null);
    TEST(serialize_empty_module);
    TEST(serialize_single_void_function);
    TEST(serialize_function_is_extern);
    TEST(serialize_function_with_params);
    TEST(serialize_function_with_struct_return);
    TEST(serialize_memory_annotations);
    TEST(serialize_int_constant);
    TEST(serialize_float_constant);
    TEST(serialize_multiple_functions);
    TEST(serialize_param_with_struct_type);
    TEST(embed_null_inputs_return_false);
    TEST(embed_buffer_too_small);
    TEST(embed_metadata_at_end);
    TEST(embed_metadata_before_main);
    TEST(deserialize_stub_returns_false);

    printf("\n✓ All module metadata tests passed!\n");
    return 0;
}
