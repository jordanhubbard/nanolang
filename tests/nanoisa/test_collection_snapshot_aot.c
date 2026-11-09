#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <assert.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static jmp_buf refusal;
static int deny = -1, allocations, live_allocations, injected, release_calls;
static bool refuse_release;
static _Noreturn void probe_abort(void) { longjmp(refusal, 1); }
static void *probe_calloc(size_t count, size_t size) {
    if (allocations++ == deny) { injected = 1; return NULL; }
    void *value = calloc(count, size);
    if (value) ++live_allocations;
    return value;
}
static void *probe_malloc(size_t size) {
    if (allocations++ == deny) { injected = 1; return NULL; }
    void *value = malloc(size);
    if (value) ++live_allocations;
    return value;
}
static void probe_free(void *value) {
    if (value) { assert(live_allocations > 0); --live_allocations; free(value); }
}

/* I include the actual emitted program. The test runner changes only this
 * helper's allocator bindings and the invariant termination observation. */
#define main collection_generated_main
#include COLLECTION_AOT_PROGRAM
#undef main
#include "../../modules/std/collections/collections.h"

static bool release_snapshot(void *value) {
    ++release_calls;
    return !refuse_release && nl_hm_keys__nano_array_release_v1(value);
}

static DynArray *snapshot(bool empty) {
    void *map = nl_hm_new(); assert(map);
    if (!empty) { nl_hm_put(map, "d", "1"); nl_hm_put(map, "aa", "2"); }
    DynArray *value = nl_hm_keys(map); assert(value);
    nl_hm_free(map);
    assert(gc_get_stats().num_objects == 1);
    return value;
}

static void expect_failure(DynArray *value) {
    size_t debt = nagg_allocation_debt;
    release_calls = allocations = injected = 0;
    assert(!live_allocations && !nsarr_owners && !nsarr_strings && !nagg_live_bytes);
    if (!setjmp(refusal)) {
        (void)nhost_collection_snapshot(value, release_snapshot);
        assert(!"I require the original invariant refusal");
    }
    assert(release_calls == 1 && !live_allocations);
    assert(!nsarr_owners && !nsarr_strings && !nagg_live_bytes);
    assert(nagg_allocation_debt == debt);
}

int main(void) {
    (void)collection_generated_main;
    gc_init();
    DynArray *source = snapshot(false);
    nsarr_t copied = nhost_collection_snapshot(source, release_snapshot);
    assert(release_calls == 1 && gc_get_stats().num_objects == 0 && copied->len == 2);
    assert((!strcmp(copied->data[0], "d") && !strcmp(copied->data[1], "aa")) ||
           (!strcmp(copied->data[0], "aa") && !strcmp(copied->data[1], "d")));
    int allocation_count = allocations;
    assert(allocation_count == 7 && live_allocations == allocation_count);
    live_allocations = 0;
    nsarr_release_owned(); assert(!nagg_live_bytes);
    for (deny = 0; deny < allocation_count; ++deny) {
        expect_failure(snapshot(false));
        assert(injected && gc_get_stats().num_objects == 0);
    }
    deny = -1;
    expect_failure(NULL);
    source = snapshot(false);
    gc_retain(source); expect_failure(source);
    assert(gc_get_stats().num_objects == 1);
    gc_release(source); assert(nl_hm_keys__nano_array_release_v1(source));

    source = snapshot(false);
    source->length = -1; expect_failure(source);
    source->length = 2; assert(nl_hm_keys__nano_array_release_v1(source));
    source = snapshot(false);
    source->elem_type = ELEM_INT; expect_failure(source);
    source->elem_type = ELEM_STRING; assert(nl_hm_keys__nano_array_release_v1(source));
    source = snapshot(false);
    source->elem_size = 1; expect_failure(source);
    source->elem_size = sizeof(char *); assert(nl_hm_keys__nano_array_release_v1(source));
    source = snapshot(false);
    int64_t capacity = source->capacity;
    source->capacity = 1; expect_failure(source);
    source->capacity = capacity; assert(nl_hm_keys__nano_array_release_v1(source));
    source = snapshot(false);
    capacity = source->capacity;
    source->length = source->capacity = INT64_MAX; expect_failure(source);
    source->length = 2; source->capacity = capacity;
    assert(nl_hm_keys__nano_array_release_v1(source));
    source = snapshot(false);
    void *storage = source->data; source->data = NULL; expect_failure(source);
    source->data = storage; assert(nl_hm_keys__nano_array_release_v1(source));

    source = snapshot(false);
    char **data = source->data; free(data[0]); data[0] = NULL;
    expect_failure(source); assert(gc_get_stats().num_objects == 0);
    source = snapshot(false);
    data = source->data; data[0][0] = (char)0xff;
    expect_failure(source); assert(gc_get_stats().num_objects == 0);

    char *borrowed_data[] = {"borrowed"};
    DynArray borrowed = {.length=1, .capacity=1, .elem_type=ELEM_STRING,
                         .elem_size=sizeof(char *), .data=borrowed_data};
    expect_failure(&borrowed);
    source = snapshot(false);
    refuse_release = true; expect_failure(source); refuse_release = false;
    assert(gc_get_stats().num_objects == 1 && nl_hm_keys__nano_array_release_v1(source));

    source = snapshot(true);
    release_calls = allocations = 0;
    copied = nhost_collection_snapshot(source, release_snapshot);
    assert(release_calls == 1 && copied->len == 0 && gc_get_stats().num_objects == 0);
    assert(live_allocations == 2); live_allocations = 0;
    nsarr_release_owned(); assert(!nagg_live_bytes);
    gc_shutdown();
    puts("I pass actual emitted snapshot copies, seven allocation denials, shared/borrowed/metadata/UTF-8/NULL refusals and exactly-once cleanup without partial publication.");
    return 0;
}
