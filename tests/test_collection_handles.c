#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static int fail_allocation = -1, allocation_index;
static void *checked_calloc(size_t count, size_t width) {
    if (fail_allocation >= 0 && allocation_index++ == fail_allocation) return NULL;
    return calloc(count, width);
}
/* I exercise the actual implementation, including private publication refusal,
 * without adding test hooks or changing the normal library build. */
#define calloc checked_calloc
#include "../modules/std/collections/collections.c"
#undef calloc

static void release_snapshot(DynArray *array) {
    assert(array && array->elem_type == ELEM_STRING);
    for (int64_t i = 0; i < array->length; i++) free(dyn_array_get_string(array, i));
    gc_release(array);
}
static void empty_snapshot(DynArray *array) {
    assert(array && array->length == 0);
    release_snapshot(array);
}
static void refused_map(void *identity) {
    nl_hm_put(identity, "key", "changed");
    assert(!nl_hm_has(identity, "key"));
    assert(!strcmp(nl_hm_get(identity, "key"), ""));
    assert(nl_hm_size(identity) == 0);
    empty_snapshot(nl_hm_keys(identity)); empty_snapshot(nl_hm_values(identity));
    nl_hm_remove(identity, "key"); nl_hm_clear(identity); nl_hm_free(identity);
}
static void refused_set(void *identity) {
    nl_set_add(identity, "key"); assert(!nl_set_has(identity, "key"));
    assert(nl_set_size(identity) == 0); empty_snapshot(nl_set_values(identity));
    nl_set_remove(identity, "key"); nl_set_clear(identity); nl_set_free(identity);
}
static void lifetime(void) {
    int foreign = 0;
    refused_map(NULL); refused_set(NULL);
    refused_map(&foreign); refused_set(&foreign);
    refused_map((void *)UINTPTR_MAX); refused_set((void *)UINTPTR_MAX);
    void *map = nl_hm_new(), *set = nl_set_new(); assert(map && set && map != set);
    void *alias = map;
    nl_hm_put(map, "key", "live"); assert(!strcmp(nl_hm_get(alias, "key"), "live"));
    nl_set_add(set, "key");
    refused_map(set); refused_set(map);
    assert(nl_hm_size(map) == 1 && nl_set_size(set) == 1);
    nl_hm_free(alias); refused_map(map); nl_set_free(set); refused_set(set);
    for (int i = 0; i < 1000; i++) {
        void *fresh_map = nl_hm_new(), *fresh_set = nl_set_new();
        assert(fresh_map && fresh_set && fresh_map != map && fresh_set != set);
        nl_hm_put(fresh_map, "key", "new"); nl_set_add(fresh_set, "key");
        assert(nl_hm_size(map) == 0 && nl_set_size(set) == 0);
        nl_hm_free(map); nl_set_free(set);
        assert(nl_hm_size(fresh_map) == 1 && nl_set_size(fresh_set) == 1);
        nl_hm_free(fresh_map); nl_set_free(fresh_set);
    }
    assert(!collection_handles);
}
static void contents(void) {
    void *map = nl_hm_new(); assert(map);
    nl_hm_put(map, "d", "1"); nl_hm_put(map, "aa", "2");
    nl_hm_remove(map, "d"); assert(nl_hm_has(map, "aa") && !nl_hm_has(map, "d"));
    for (int i = 0; i < 200; i++) {
        char key[32], value[32]; snprintf(key, sizeof key, "k%d", i);
        snprintf(value, sizeof value, "%d", i * 2); nl_hm_put(map, key, value);
    }
    assert(nl_hm_size(map) == 201);
    for (int i = 0; i < 100; i++) {
        char key[32]; snprintf(key, sizeof key, "k%d", i); nl_hm_remove(map, key);
    }
    assert(nl_hm_size(map) == 101 && !strcmp(nl_hm_get(map, "k199"), "398"));
    DynArray *keys = nl_hm_keys(map), *values = nl_hm_values(map), *again = nl_hm_keys(map);
    assert(keys->length == 101 && values->length == 101 && again->length == 101);
    for (int64_t i = 0; i < keys->length; i++) {
        assert(!strcmp(dyn_array_get_string(keys, i), dyn_array_get_string(again, i)));
        assert(!strcmp(nl_hm_get(map, dyn_array_get_string(keys, i)), dyn_array_get_string(values, i)));
    }
    nl_hm_clear(map); assert(nl_hm_size(map) == 0); nl_hm_free(map);
    assert(!strcmp(dyn_array_get_string(keys, 0), dyn_array_get_string(again, 0)));
    release_snapshot(keys); release_snapshot(values); release_snapshot(again);
    void *set = nl_set_new(); assert(set);
    nl_set_add(set, "a"); nl_set_add(set, "a"); nl_set_add(set, "b");
    assert(nl_set_size(set) == 2); nl_set_remove(set, "a");
    keys = nl_set_values(set); assert(keys->length == 1);
    nl_set_clear(set); assert(nl_set_size(set) == 0); nl_set_free(set);
    assert(!strcmp(dyn_array_get_string(keys, 0), "b")); release_snapshot(keys);
    assert(!collection_handles);
}
static void allocation_refusals(void) {
    for (int family = 0; family < 2; family++) {
        for (int fail = 0; fail < (family ? 5 : 3); fail++) {
            fail_allocation = fail; allocation_index = 0;
            void *identity = family ? nl_set_new() : nl_hm_new();
            fail_allocation = -1;
            assert(!identity && !collection_handles);
            identity = family ? nl_set_new() : nl_hm_new(); assert(identity);
            if (family) nl_set_free(identity); else nl_hm_free(identity);
            assert(!collection_handles);
        }
    }
}
static void *independent_worker(void *unused) {
    (void)unused;
    for (int i = 0; i < 300; i++) {
        void *map = nl_hm_new(), *set = nl_set_new(); assert(map && set);
        nl_hm_put(map, "key", "value"); nl_set_add(set, "key");
        assert(!strcmp(nl_hm_get(map, "key"), "value") && nl_set_has(set, "key"));
        nl_hm_free(map); nl_set_free(set);
    }
    return NULL;
}
static void independent_threads(void) {
    pthread_t threads[4];
    for (size_t i = 0; i < 4; i++) assert(!pthread_create(&threads[i], NULL, independent_worker, NULL));
    for (size_t i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
    assert(!collection_handles);
}
static void exhaustion(void) {
    collection_next_identity = UINTPTR_MAX;
    void *last = nl_hm_new(); assert(last && (uintptr_t)last == UINTPTR_MAX);
    assert(!collection_next_identity && !nl_hm_new() && !nl_set_new());
    nl_hm_put(last, "key", "last"); assert(nl_hm_size(last) == 1);
    nl_hm_free(last); assert(!collection_handles && !nl_hm_new());
}
int main(void) {
    gc_init(); lifetime(); contents(); allocation_refusals(); independent_threads(); exhaustion();
    assert(gc_get_stats().num_objects == 0); gc_shutdown();
    puts("I pass collection kind, alias, stale, repeated-free, contents, snapshot, allocation, exhaustion and independent-thread controls.");
    return 0;
}
