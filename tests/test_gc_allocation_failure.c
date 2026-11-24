/* I fail actual GC allocations without replacing neighboring runtime allocators. */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

static size_t attempts, refuse, live_count, live_bytes;
static int expected_managed, cleaning_failure;
static void *tracked[64];
static size_t sizes[64];

static void *controlled_malloc(size_t size) {
    attempts++;
    if (attempts == refuse) return NULL;
    void *ptr = malloc(size);
    if (!ptr) return NULL;
    for (size_t i = 0; i < 64; i++) {
        if (!tracked[i]) {
            tracked[i] = ptr;
            sizes[i] = size;
            live_count++;
            live_bytes += size;
            return ptr;
        }
    }
    abort();
}

static void controlled_free(void *ptr) {
    if (!ptr) return;
    for (size_t i = 0; i < 64; i++) {
        if (tracked[i] == ptr) {
            live_count--;
            live_bytes -= sizes[i];
            tracked[i] = NULL;
            free(ptr);
            return;
        }
    }
    fprintf(stderr, "I attempted an untracked or repeated GC free.\n");
    abort();
}

#define malloc controlled_malloc
#define free controlled_free
#include "../src/runtime/gc.c"
#undef malloc
#undef free

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "I failed allocation control at line %d: %s\n", __LINE__, #condition); \
    refuse = 0; expected_managed = 1; \
    if (!cleaning_failure) { cleaning_failure = 1; gc_shutdown(); } \
    exit(1); \
} } while (0)

static int finalized[16];
static size_t finalizer_count;

static void observe_finalizer(void *ptr) {
    CHECK(gc_is_managed(ptr) == (expected_managed != 0));
    CHECK(finalizer_count < 16);
    finalized[finalizer_count++] = *(int *)ptr;
}

static void same_stats(GCStats a, GCStats b) {
    CHECK(a.total_allocated == b.total_allocated);
    CHECK(a.total_freed == b.total_freed);
    CHECK(a.current_usage == b.current_usage);
    CHECK(a.num_objects == b.num_objects);
    CHECK(a.num_collections == b.num_collections);
}

static void *allocate_kind(int kind) {
    if (kind == 1) return gc_alloc_string(7);
    if (kind == 2) return gc_alloc_opaque(sizeof(int), observe_finalizer);
    return gc_alloc(sizeof(int), GC_TYPE_OPAQUE);
}

int main(void) {
    gc_init();
    int *first = gc_alloc_opaque(sizeof(int), observe_finalizer);
    int *second = gc_alloc_opaque(sizeof(int), observe_finalizer);
    CHECK(first && second);
    *first = 11;
    *second = 22;
    CHECK(live_count == 4);
    GCHeader *head = gc_state.all_objects;
    CHECK(head == gc_get_header(second));
    CHECK(head->next == gc_get_header(first));
    for (int kind = 0; kind < 3; kind++) {
        for (size_t failure = 1; failure <= 2; failure++) {
            GCStats before = gc_get_stats();
            size_t count = live_count, bytes = live_bytes;
            attempts = 0;
            refuse = failure;
            void *owner = allocate_kind(kind);
            refuse = 0;
            CHECK(owner == NULL);
            CHECK(attempts == failure);
            CHECK(live_count == count && live_bytes == bytes);
            same_stats(before, gc_get_stats());
            CHECK(gc_state.all_objects == head && head->prev == NULL);
            CHECK(head->next == gc_get_header(first));
            CHECK(gc_get_header(first)->prev == head);
            CHECK(gc_is_managed(first) && gc_is_managed(second));
            CHECK(*first == 11 && *second == 22 && finalizer_count == 0);
            void *retry = allocate_kind(kind);
            CHECK(retry && gc_is_managed(retry));
            if (kind == 2) *(int *)retry = 33;
            if (kind == 1) CHECK(((char *)retry)[7] == '\0');
            gc_retain(retry);
            gc_release(retry);
            CHECK(gc_is_managed(retry));
            gc_release(retry);
            CHECK(live_count == count && live_bytes == bytes);
            CHECK(gc_state.all_objects == head && head->prev == NULL);
            /* I keep cumulative allocation/free totals after successful retries. */
            GCStats completed = gc_get_stats();
            CHECK(completed.current_usage == before.current_usage);
            CHECK(completed.num_objects == before.num_objects);
            CHECK(completed.total_allocated - before.total_allocated ==
                  completed.total_freed - before.total_freed);
            if (kind == 2) {
                CHECK(finalizer_count == 1 && finalized[0] == 33);
                finalizer_count = 0;
            }
        }
    }
    gc_release(second);
    CHECK(finalizer_count == 1 && finalized[0] == 22);
    CHECK(gc_is_managed(first));
    expected_managed = 1;
    gc_shutdown();
    CHECK(finalizer_count == 2 && finalized[1] == 11);
    CHECK(live_count == 0 && live_bytes == 0);
    GCStats stopped = gc_get_stats();
    CHECK(stopped.current_usage == 0 && stopped.num_objects == 0);
    CHECK(stopped.total_allocated == stopped.total_freed);
    gc_shutdown();
    same_stats(stopped, gc_get_stats());
    gc_init();
    attempts = 0;
    refuse = 2;
    CHECK(gc_alloc(0, GC_TYPE_OPAQUE) == NULL);
    refuse = 0;
    CHECK(live_count == 0 && live_bytes == 0);
    GCStats empty = gc_get_stats();
    CHECK(empty.num_objects == 0 && empty.total_allocated == 0);
    void *last = gc_alloc(0, GC_TYPE_OPAQUE);
    CHECK(last && gc_is_managed(last));
    gc_release(last);
    gc_shutdown();
    CHECK(live_count == 0 && live_bytes == 0);
    puts("I passed header/registry refusal, three constructor families, retries, finalizers and reinitialization.");
    return 0;
}
