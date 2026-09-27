/* I observe actual core allocation requests without allocating observer storage. */
#ifndef RECORD_ARRAY_LLVM_WASM_ALLOC_H
#define RECORD_ARRAY_LLVM_WASM_ALLOC_H
#include <stddef.h>
#include <stdint.h>
static struct { void *pointer; size_t bytes; } ra_slots[8192];
static size_t ra_live,ra_bytes,ra_peak,ra_calls,ra_created,ra_fail=SIZE_MAX;
static int ra_persistent;
static size_t ra_bucket(void *p) { return (((uintptr_t)p>>4)*UINT32_C(2654435761))&8191u; }
static void ra_invariant(int condition) { if(!condition)__builtin_trap(); }
int nms_test_allocation_permitted(uint64_t bytes) {
    ra_invariant(bytes<=SIZE_MAX&&ra_calls<SIZE_MAX);
    size_t index=ra_calls++;
    return !(index==ra_fail||(ra_persistent&&index>ra_fail));
}
void nms_test_allocation_created(void *pointer,uint64_t bytes) {
    ra_invariant(pointer&&bytes<=SIZE_MAX&&bytes<=SIZE_MAX-ra_bytes);
    ra_invariant(pointer!=(void *)(uintptr_t)1);
    size_t empty=8192,start=ra_bucket(pointer);
    for(size_t step=0;step<8192;step++) {
        size_t i=(start+step)&8191u;
        ra_invariant(ra_slots[i].pointer!=pointer);
        if(ra_slots[i].pointer==(void *)(uintptr_t)1&&empty==8192)empty=i;
        if(!ra_slots[i].pointer){if(empty==8192)empty=i;break;}
    }
    ra_invariant(empty<8192&&ra_created<SIZE_MAX);
    ra_slots[empty].pointer=pointer;ra_slots[empty].bytes=(size_t)bytes;
    ra_live++;ra_created++;ra_bytes+=(size_t)bytes;
    if(ra_bytes>ra_peak)ra_peak=ra_bytes;
}
void nms_test_allocation_destroyed(void *pointer) {
    size_t start=ra_bucket(pointer);
    for(size_t step=0;step<8192;step++) {
        size_t i=(start+step)&8191u;
        if(!ra_slots[i].pointer)break;
        if(ra_slots[i].pointer!=pointer)continue;
        ra_invariant(pointer&&pointer!=(void *)(uintptr_t)1&&ra_live&&ra_bytes>=ra_slots[i].bytes);
        ra_live--;ra_bytes-=ra_slots[i].bytes;
        ra_slots[i].pointer=(void *)(uintptr_t)1;ra_slots[i].bytes=0;return;
    }
    ra_invariant(0);
}
#endif
