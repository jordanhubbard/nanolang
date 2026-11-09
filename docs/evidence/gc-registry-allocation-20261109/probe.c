#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
static size_t attempts;
static size_t refuse;
static void *controlled_malloc(size_t size) {
    attempts++;
    if (attempts == refuse) return NULL;
    return malloc(size);
}
#define malloc controlled_malloc
#include "../../../src/runtime/gc.c"
#undef malloc
int main(void) {
    gc_init();
    GCStats before = gc_get_stats();
    attempts = 0;
    refuse = 2;
    void *owner = gc_alloc(sizeof(int), GC_TYPE_OPAQUE);
    int returned = owner != NULL;
    int managed = owner ? gc_is_managed(owner) : 0;
    GCStats after = gc_get_stats();
    gc_release(owner);
    GCStats released = gc_get_stats();
    refuse = 0;
    gc_shutdown();
    printf("I received an owner: %d; managed: %d; allocation attempts: %zu; objects before/after/release: %zu/%zu/%zu.\n",
           returned, managed, attempts, before.num_objects, after.num_objects, released.num_objects);
    return returned && !managed && released.num_objects > before.num_objects ? 2 : 0;
}
