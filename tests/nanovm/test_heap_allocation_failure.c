/* I inject failure only into the heap implementation compiled in this test.
 * Production allocation APIs and unrelated allocations remain unchanged. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

static bool reject_calloc;
static unsigned frees;
static void *heap_test_calloc(size_t count, size_t size) {
    return reject_calloc ? NULL : calloc(count, size);
}
static void heap_test_free(void *pointer) {
    if (pointer) frees++;
    free(pointer);
}
#define calloc heap_test_calloc
#define free heap_test_free
#include "../../src/nanovm/heap.c"
#undef calloc
#undef free

int main(void) {
    VmHeap heap;
    vm_heap_init(&heap);
    uint64_t allocated = heap.stats.allocated;
    uint64_t objects = heap.stats.num_objects;
    uint64_t calls = heap.stats.allocation_calls;
    unsigned before = frees;
    reject_calloc = true;
    assert(vm_struct_new(&heap, 0, 2) == NULL);
    assert(frees == before + 1);
    assert(vm_union_new(&heap, 0, 0, 2) == NULL);
    assert(frees == before + 2);
    assert(heap.stats.allocated == allocated);
    assert(heap.stats.num_objects == objects);
    assert(heap.stats.allocation_calls == calls);
    /* A null zero-sized field allocation is legal, not an allocation error. */
    VmStruct *record = vm_struct_new(&heap, 0, 0);
    VmUnion *variant = vm_union_new(&heap, 0, 0, 0);
    assert(record && variant);
    reject_calloc = false;
    vm_release(&heap, val_struct(record));
    vm_release(&heap, val_union(variant));
    record = vm_struct_new(&heap, 0, 2);
    variant = vm_union_new(&heap, 0, 0, 2);
    assert(record && record->fields && variant && variant->fields);
    vm_release(&heap, val_struct(record));
    vm_release(&heap, val_union(variant));
    vm_heap_destroy(&heap);
    puts("I passed struct/union field-allocation failure and recovery checks.");
    return 0;
}
