/* I retain every original allocation assertion and add growth-boundary refusals. */
#define main original_gc_allocation_main
#include "test_gc_allocation_failure.c"
#undef main

static void growth_refusals(size_t count) {
    assert(count <= 31);
    for (int kind = 0; kind < 3; ++kind) {
        for (size_t failure = 1; failure <= 2; ++failure) {
            void *owners[31]; size_t buckets[31];
            gc_init(); finalizer_count = 0; expected_managed = 0;
            for (size_t i = 0; i < count; ++i) {
                owners[i] = gc_alloc(sizeof(int), GC_TYPE_OPAQUE); CHECK(owners[i]);
                *(int *)owners[i] = (int)i + 100;
            }
            for (size_t i = 0; i < count; ++i) buckets[i] = gc_hash_ptr(owners[i]);
            GCStats before = gc_get_stats(); GCHeader *head = gc_state.all_objects;
            size_t allocations = live_count, bytes = live_bytes;
            size_t active_buckets = gc_state.hash_bucket_count;
            attempts = 0; refuse = failure;
            void *refused = allocate_kind(kind); refuse = 0;
            CHECK(!refused && attempts == failure);
            CHECK(live_count == allocations && live_bytes == bytes && gc_state.all_objects == head);
            CHECK(gc_state.hash_bucket_count == active_buckets);
            same_stats(before, gc_get_stats());
            for (size_t i = 0; i < count; ++i) {
                CHECK(gc_is_managed(owners[i]) && *(int *)owners[i] == (int)i + 100);
                CHECK(gc_hash_ptr(owners[i]) == buckets[i]);
            }
            void *retry = allocate_kind(kind); CHECK(retry && gc_is_managed(retry));
            if (kind == 2) *(int *)retry = 77;
            if (kind == 1) CHECK(((char *)retry)[7] == '\0');
            gc_release(retry);
            if (kind == 2) CHECK(finalizer_count == 1 && finalized[0] == 77);
            CHECK(live_count == allocations && live_bytes == bytes);
            for (size_t i = 0; i < count; ++i)
                CHECK(gc_is_managed(owners[i]) && *(int *)owners[i] == (int)i + 100);
            for (size_t i = 0; i < count; ++i) gc_release(owners[i]);
            gc_shutdown(); CHECK(live_count == 0 && live_bytes == 0);
        }
    }
    printf("I preserved both essential allocation failures around a %zu-owner boundary.\n", count);
}
int main(void) {
    CHECK(original_gc_allocation_main() == 0);
    const size_t counts[] = {15, 16, 17, 31};
    for (size_t i = 0; i < sizeof counts / sizeof *counts; ++i) growth_refusals(counts[i]);
    return 0;
}
