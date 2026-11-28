/* I exercise live registry identity while budgeting the bucket working set. */
#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/runtime/gc.c"

typedef struct { uint64_t index, marker; } Payload;
static size_t payload_size(size_t i) { return sizeof(Payload) + i % 37; }
static unsigned char pattern(size_t i, size_t j) { return (unsigned char)(i * 131 + j * 17); }
static void verify(void *owner, size_t i) {
    assert(gc_is_managed(owner));
    assert(!gc_is_managed((unsigned char *)owner + 1));
    assert(!gc_is_managed((unsigned char *)owner - sizeof(GCHeader)));
    Payload *p = owner;
    assert(p->index == i && p->marker == (UINT64_C(0xa61b70ecf193d428) ^ i));
    assert(gc_get_header(owner)->size == sizeof(GCHeader) + payload_size(i));
    for (size_t j = sizeof *p; j < payload_size(i); ++j)
        assert(((unsigned char *)owner)[j] == pattern(i, j));
}
static bool epoch(size_t count) {
    void *owners[4096]; assert(count <= sizeof owners / sizeof *owners);
    size_t bytes = 0, maximum = 0;
    gc_init();
    int stack_value = 3;
    void *foreign = malloc(31); assert(foreign);
    assert(!gc_is_managed(NULL) && !gc_is_managed(&stack_value) && !gc_is_managed(foreign));
    for (size_t i = 0; i < count; ++i) {
        owners[i] = gc_alloc(payload_size(i), GC_TYPE_OPAQUE); assert(owners[i]);
        Payload *p = owners[i]; p->index = i; p->marker = UINT64_C(0xa61b70ecf193d428) ^ i;
        for (size_t j = sizeof *p; j < payload_size(i); ++j)
            ((unsigned char *)p)[j] = pattern(i, j);
        bytes += sizeof(GCHeader) + payload_size(i);
        for (size_t j = 0; j <= i; ++j) verify(owners[j], j);
    }
    GCStats full = gc_get_stats();
    assert(full.num_objects == count && full.current_usage == bytes);
    for (size_t i = 0; i < count; ++i) {
        size_t bucket = gc_hash_ptr(owners[i]); if (bucket > maximum) maximum = bucket;
        gc_retain(owners[i]); gc_release(owners[i]); verify(owners[i], i);
    }
    bool compact = maximum < count * 8;
    printf("{\"owners\":%zu,\"maximum_bucket\":%zu,\"bucket_budget\":%zu,\"within_budget\":%s}\n",
           count, maximum, count * 8, compact ? "true" : "false");
    for (size_t i = 0; i < count; i += 2) {
        gc_release(owners[i]); assert(!gc_is_managed(owners[i]));
        for (size_t j = 1; j < count; j += 2) verify(owners[j], j);
    }
    for (size_t i = 1; i < count; i += 2) { verify(owners[i], i); gc_release(owners[i]); }
    GCStats empty = gc_get_stats();
    assert(empty.num_objects == 0 && empty.current_usage == 0 && empty.total_allocated == empty.total_freed);
    assert(!gc_is_managed(foreign)); free(foreign);
    gc_shutdown(); gc_shutdown();
    return compact;
}
int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const size_t counts[] = {16, 64, 256, 1024, 4096, 16};
    bool compact = true;
    for (size_t i = 0; i < sizeof counts / sizeof *counts; ++i)
        if (!epoch(counts[i])) compact = false;
    if (!compact) { fputs("I exceeded my registry working-set budget after semantic controls passed.\n", stderr); return 1; }
    puts("I retained registry membership, payloads, logical accounting, release and reinitialization within my working-set budget.");
    return 0;
}
