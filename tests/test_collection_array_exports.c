#include "../modules/std/collections/collections.h"
#include "../modules/std/json/json.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

extern const uint32_t nl_hm_keys__nano_array_abi, nl_hm_values__nano_array_abi;
extern const uint32_t nl_set_values__nano_array_abi, nl_json_object_keys__nano_array_abi;

static void check(DynArray *a, int64_t length) {
    assert(a && a->elem_type == ELEM_STRING && a->elem_size == sizeof(char *));
    assert(a->length == length && a->capacity >= length);
    for (int64_t i = 0; i < length; ++i) assert(dyn_array_get_string(a, i));
}
static bool contains(DynArray *a, const char *value) {
    for (int64_t i = 0; i < a->length; ++i)
        if (!strcmp(dyn_array_get_string(a, i), value)) return true;
    return false;
}
/* I explicitly free this fixture's known copied strings. The runtime's
 * unresolved mixed borrowed/owned string contract is not certified here. */
static void release_snapshot(DynArray *a) {
    for (int64_t i = 0; i < a->length; ++i) free((void *)dyn_array_get_string(a, i));
    gc_release(a);
}

int main(void) {
    gc_init();
    assert(nl_hm_keys__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_hm_values__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_set_values__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    assert(nl_json_object_keys__nano_array_abi == NANO_DYN_ARRAY_ABI_VERSION);
    DynArray *empty[] = {nl_hm_keys(NULL), nl_hm_values(NULL),
                        nl_set_values(NULL), nl_json_object_keys(NULL)};
    for (unsigned i = 0; i < sizeof empty / sizeof *empty; ++i) {
        check(empty[i], 0); release_snapshot(empty[i]);
    }
    void *hm = nl_hm_new();
    assert(hm);
    nl_hm_put(hm, "alpha", "one"); nl_hm_put(hm, "beta", "two");
    DynArray *keys = nl_hm_keys(hm), *values = nl_hm_values(hm);
    nl_hm_free(hm);
    check(keys, 2); check(values, 2);
    assert(contains(keys, "alpha") && contains(keys, "beta"));
    assert(contains(values, "one") && contains(values, "two"));
    release_snapshot(keys); release_snapshot(values);
    void *set = nl_set_new();
    assert(set);
    nl_set_add(set, "unique"); nl_set_add(set, "unique");
    keys = nl_set_values(set); nl_set_free(set);
    check(keys, 1); assert(contains(keys, "unique")); release_snapshot(keys);
    void *json = nl_json_parse("{\"alpha\":1,\"beta\":2}");
    assert(json);
    keys = nl_json_object_keys(json); nl_json_free(json);
    check(keys, 2);
    assert(contains(keys, "alpha") && contains(keys, "beta"));
    release_snapshot(keys);
    assert(gc_get_stats().num_objects == 0);
    gc_shutdown();
    return 0;
}
