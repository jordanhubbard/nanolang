/* I exercise actual registry allocation failures without adopting payloads. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(test) do { if (!(test)) { \
    fprintf(stderr, "I fail carrier allocation check %s at line %d.\n", #test, __LINE__); \
    exit(1); } } while (0)

static size_t remaining = SIZE_MAX, metadata_live;
static void *registry_malloc(size_t size) {
    if (remaining == 0) return NULL;
    if (remaining != SIZE_MAX) remaining--;
    void *p = malloc(size); if (p) metadata_live++; return p;
}
static void *registry_calloc(size_t count, size_t size) {
    if (remaining == 0) return NULL;
    if (remaining != SIZE_MAX) remaining--;
    void *p = calloc(count, size); if (p) metadata_live++; return p;
}
static void registry_free(void *p) {
    if (p) { CHECK(metadata_live > 0); metadata_live--; }
    free(p);
}

#define malloc registry_malloc
#define calloc registry_calloc
#define free registry_free
#include "../src/eval/carrier_heap.c"
#undef malloc
#undef calloc
#undef free

static size_t payloads_destroyed;
static void destroy_payload(void *p, void *data) {
    CHECK(data == &payloads_destroyed); payloads_destroyed++; free(p);
}

int main(void) {
    remaining = 0; CHECK(!nl_carrier_heap_new()); CHECK(metadata_live == 0);
    remaining = 1; CHECK(!nl_carrier_heap_new()); CHECK(metadata_live == 0);
    remaining = SIZE_MAX;
    NlCarrierHeap *heap = nl_carrier_heap_new(); CHECK(heap);
    int *payload = malloc(sizeof *payload); CHECK(payload); *payload = 47;
    size_t baseline = metadata_live;
    remaining = 0;
    CHECK(!nl_carrier_heap_track(heap, payload, NULL, destroy_payload, &payloads_destroyed));
    CHECK(metadata_live == baseline); CHECK(*payload == 47); CHECK(payloads_destroyed == 0);
    remaining = SIZE_MAX;
    CHECK(nl_carrier_heap_track(heap, payload, NULL, destroy_payload, &payloads_destroyed));
    baseline = metadata_live;
    const void *roots[] = {payload};
    remaining = 0;
    CHECK(!nl_carrier_heap_collect(heap, roots, 1));
    CHECK(metadata_live == baseline); CHECK(*payload == 47); CHECK(payloads_destroyed == 0);
    CHECK(nl_carrier_heap_contains(heap, payload));
    remaining = SIZE_MAX; CHECK(nl_carrier_heap_collect(heap, roots, 1));
    CHECK(metadata_live == baseline); CHECK(*payload == 47);

    /* I fill the original table to its admission threshold, then fail both
     * before and after successful growth without transferring the new value. */
    for (size_t i = 0; i < 23; ++i) {
        int *p = malloc(sizeof *p); CHECK(p); *p = (int)i;
        CHECK(nl_carrier_heap_track(heap, p, NULL, destroy_payload, &payloads_destroyed));
    }
    int *extra = malloc(sizeof *extra); CHECK(extra); *extra = 91;
    baseline = metadata_live;
    remaining = 0;
    CHECK(!nl_carrier_heap_track(heap, extra, NULL, destroy_payload, &payloads_destroyed));
    CHECK(metadata_live == baseline); CHECK(*extra == 91);
    remaining = 1;
    CHECK(!nl_carrier_heap_track(heap, extra, NULL, destroy_payload, &payloads_destroyed));
    CHECK(metadata_live == baseline); CHECK(*extra == 91);
    NlCarrierHeapStats stats = nl_carrier_heap_stats(heap);
    CHECK(stats.registered == 24 && stats.live == 24 && stats.destroyed == 0);
    CHECK(!nl_carrier_heap_contains(heap, extra)); free(extra);
    remaining = 0;
    CHECK(nl_carrier_heap_clear(heap)); CHECK(payloads_destroyed == 24);
    stats = nl_carrier_heap_stats(heap);
    CHECK(stats.registered == 24 && stats.destroyed == 24 && stats.live == 0);
    CHECK(nl_carrier_heap_collect(heap, NULL, 0));
    CHECK(nl_carrier_heap_free(heap)); CHECK(metadata_live == 0);
    puts("I preserve caller ownership through six allocation failures and retire all registry metadata.");
    return 0;
}
