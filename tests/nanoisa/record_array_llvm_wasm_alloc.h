/* I observe actual core allocation requests without allocating observer storage. */
#ifndef RECORD_ARRAY_LLVM_WASM_ALLOC_H
#define RECORD_ARRAY_LLVM_WASM_ALLOC_H
#include <stddef.h>
#include <stdint.h>
static struct { void *pointer; size_t bytes; } ra_slots[8192];
static size_t ra_live,ra_bytes,ra_peak,ra_calls,ra_created,ra_fail=SIZE_MAX;
static int ra_persistent;
static void ra_invariant(int condition) { if(!condition)__builtin_trap(); }
int nms_test_allocation_permitted(uint64_t bytes) {
    ra_invariant(bytes<=SIZE_MAX&&ra_calls<SIZE_MAX);
    size_t index=ra_calls++;
    return !(index==ra_fail||(ra_persistent&&index>ra_fail));
}
void nms_test_allocation_created(void *pointer,uint64_t bytes) {
    ra_invariant(pointer&&bytes<=SIZE_MAX&&bytes<=SIZE_MAX-ra_bytes);
    size_t empty=8192;
    for(size_t i=0;i<8192;i++) {
        ra_invariant(ra_slots[i].pointer!=pointer);
        if(!ra_slots[i].pointer&&empty==8192)empty=i;
    }
    ra_invariant(empty<8192&&ra_created<SIZE_MAX);
    ra_slots[empty].pointer=pointer;ra_slots[empty].bytes=(size_t)bytes;
    ra_live++;ra_created++;ra_bytes+=(size_t)bytes;
    if(ra_bytes>ra_peak)ra_peak=ra_bytes;
}
void nms_test_allocation_destroyed(void *pointer) {
    for(size_t i=0;i<8192;i++)if(ra_slots[i].pointer==pointer&&pointer) {
        ra_invariant(ra_live&&ra_bytes>=ra_slots[i].bytes);
        ra_live--;ra_bytes-=ra_slots[i].bytes;
        ra_slots[i].pointer=NULL;ra_slots[i].bytes=0;return;
    }
    ra_invariant(0);
}
#endif
