#include "../modules/std/peg/peg.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static int fail_copy_at, copies, fail_table_at, tables, fail_array;
static void *copy_alloc(size_t bytes) {
    return ++copies == fail_copy_at ? NULL : malloc(bytes);
}
static void *table_alloc(void *ptr, size_t bytes) {
    return ++tables == fail_table_at ? NULL : realloc(ptr, bytes);
}
static DynArray *array_alloc(ElementType type, int64_t capacity) {
    return fail_array ? NULL : dyn_array_new_with_capacity(type, capacity);
}
#define malloc copy_alloc
#define realloc table_alloc
#define dyn_array_new_with_capacity array_alloc
#include "../modules/std/peg/peg.c"
#undef malloc
#undef realloc
#undef dyn_array_new_with_capacity

static void release_copies(DynArray *a) {
    assert(a && a->elem_type == ELEM_STRING && a->elem_size == sizeof(char *));
    for (int64_t i = 0; i < a->length; ++i) free((void *)dyn_array_get_string(a, i));
    gc_release(a);
}
int main(void) {
    gc_init();
    assert(nl_peg_captures__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    void *peg = nl_peg_compile("([a-z]+) \":\" ([0-9]+)");
    assert(peg);
    char input[] = "word:42";
    DynArray *a = nl_peg_captures(peg, input);
    assert(a && a->length == 2);
    memset(input, 'x', sizeof input - 1);
    assert(!strcmp(dyn_array_get_string(a, 0), "word"));
    assert(!strcmp(dyn_array_get_string(a, 1), "42"));
    for (int fail = 1; fail <= 2; ++fail) {
        copies = 0; fail_copy_at = fail;
        assert(!nl_peg_captures(peg, "word:42"));
        fail_copy_at = 0;
        assert(gc_get_stats().num_objects == 1);
    }
    fail_array = 1;
    assert(!nl_peg_captures(peg, "word:42"));
    fail_array = 0;
    nl_peg_free(peg);
    assert(!strcmp(dyn_array_get_string(a, 0), "word"));
    release_copies(a);
    a = nl_peg_captures(NULL, NULL); assert(a && a->length == 0); release_copies(a);
    peg = nl_peg_compile("([a-z])+");
    assert(peg);
    for (int fail = 1; fail <= 2; ++fail) {
        tables = 0; fail_table_at = fail;
        assert(!nl_peg_captures(peg, "abcdefghijklmnop"));
        fail_table_at = 0;
        assert(gc_get_stats().num_objects == 0);
    }
    a = nl_peg_captures(peg, "abcdefghijklmnop");
    assert(a && a->length == 16); release_copies(a);
    a = nl_peg_captures(peg, "123"); assert(a && a->length == 0); release_copies(a);
    nl_peg_free(peg);
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    return 0;
}
