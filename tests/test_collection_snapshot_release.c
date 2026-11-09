#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fail_copy = -1, copy_index;
static char *snapshot_strdup(const char *text) {
    if (fail_copy >= 0 && copy_index++ == fail_copy) return NULL;
    return strdup(text);
}
#define strdup snapshot_strdup
#include "../modules/std/collections/collections.c"
#undef strdup

static void check_empty(void) {
    assert(!nl_hm_keys__nano_array_release_v1(NULL));
    assert(!nl_hm_values__nano_array_release_v1((void *)1));
    DynArray *array = nl_hm_keys(NULL); assert(array && array->length == 0);
    assert(nl_hm_keys__nano_array_release_v1(array));
    assert(!nl_hm_keys__nano_array_release_v1(array));
    assert(nl_hm_values__nano_array_release_v1(nl_hm_values(NULL)));
    assert(nl_set_values__nano_array_release_v1(nl_set_values(NULL)));
}
static void check_refusals(void) {
    DynArray *borrowed = dyn_array_new(ELEM_STRING); assert(borrowed);
    dyn_array_push_string(borrowed, "I retain this borrowed string.");
    assert(!nl_hm_keys__nano_array_release_v1(borrowed));
    assert(!strcmp(dyn_array_get_string(borrowed, 0), "I retain this borrowed string."));
    gc_release(borrowed);
    void *other = gc_alloc(16, GC_TYPE_OPAQUE); assert(other);
    assert(!nl_hm_values__nano_array_release_v1(other)); gc_release(other);
    DynArray *array = nl_hm_keys(NULL); assert(array);
    gc_retain(array); assert(!nl_hm_keys__nano_array_release_v1(array)); gc_release(array);
    array->elem_type = ELEM_INT; assert(!nl_hm_keys__nano_array_release_v1(array)); array->elem_type = ELEM_STRING;
    array->elem_size = 1; assert(!nl_hm_keys__nano_array_release_v1(array)); array->elem_size = sizeof(char *);
    array->length = -1; assert(!nl_hm_keys__nano_array_release_v1(array)); array->length = 0;
    int64_t capacity = array->capacity; array->capacity = -1;
    assert(!nl_hm_keys__nano_array_release_v1(array)); array->capacity = capacity;
    void *data = array->data; array->data = NULL;
    assert(!nl_hm_keys__nano_array_release_v1(array)); array->data = data;
    assert(nl_hm_keys__nano_array_release_v1(array));
}
static void check_independent_snapshots(void) {
    void *map = nl_hm_new(), *set = nl_set_new(); assert(map && set);
    nl_hm_put(map, "alpha", "one"); nl_hm_put(map, "beta", "two"); nl_set_add(set, "unique");
    DynArray *keys = nl_hm_keys(map), *values = nl_hm_values(map), *members = nl_set_values(set);
    assert(keys && values && members && keys->length == 2 && values->length == 2 && members->length == 1);
    for (int64_t i = 0; i < keys->length; i++)
        assert(!strcmp(nl_hm_get(map, dyn_array_get_string(keys, i)), dyn_array_get_string(values, i)));
    nl_hm_clear(map); nl_hm_free(map); nl_set_free(set);
    assert(!strcmp(dyn_array_get_string(members, 0), "unique"));
    assert(nl_hm_keys__nano_array_release_v1(keys));
    assert(nl_hm_values__nano_array_release_v1(values));
    assert(nl_set_values__nano_array_release_v1(members));
    assert(!collection_handles);
}
static void check_copy_refusals(void) {
    void *map = nl_hm_new(); assert(map);
    nl_hm_put(map, "alpha", "one"); nl_hm_put(map, "beta", "two");
    for (int family = 0; family < 2; family++) for (int fail = 0; fail < 2; fail++) {
        size_t before = gc_get_stats().num_objects;
        copy_index = 0; fail_copy = fail;
        DynArray *array = family ? nl_hm_values(map) : nl_hm_keys(map);
        fail_copy = -1;
        assert(!array && gc_get_stats().num_objects == before);
        assert(nl_hm_size(map) == 2 && !strcmp(nl_hm_get(map, "beta"), "two"));
    }
    nl_hm_free(map);
    void *set = nl_set_new(); assert(set); nl_set_add(set, "member");
    copy_index = 0; fail_copy = 0; DynArray *members = nl_set_values(set); fail_copy = -1;
    assert(!members && gc_get_stats().num_objects == 0 && nl_set_size(set) == 1); nl_set_free(set);
}
static void check_legacy_cleanup(void) {
    void *map = nl_hm_new(); assert(map); nl_hm_put(map, "legacy", "retained");
    DynArray *keys = nl_hm_keys(map); assert(keys && keys->length == 1); nl_hm_free(map);
    free(dyn_array_get_string(keys, 0)); gc_release(keys);
}
int main(void) {
    gc_init(); check_empty(); check_refusals(); check_independent_snapshots();
    check_copy_refusals(); check_legacy_cleanup();
    assert(gc_get_stats().num_objects == 0 && !collection_handles); gc_shutdown();
    puts("I pass collection snapshot provenance, unique-owner, metadata, copy-failure, independent-lifetime and original C cleanup controls.");
    return 0;
}
