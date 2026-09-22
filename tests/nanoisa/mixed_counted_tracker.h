/* My fixed observer cannot allocate recursively. Bytes exclude backend headers. */
#ifndef MIXED_COUNTED_TRACKER_H
#define MIXED_COUNTED_TRACKER_H
#include <stdint.h>
#include <stddef.h>
#define MC_TRACKED 8192u
static struct {void *pointer;uint64_t bytes;} mc_allocations[MC_TRACKED];
static uint64_t mc_attempts,mc_successes,mc_frees,mc_live,mc_bytes,mc_peak;
static uint64_t mc_fail_at,mc_failures;
static unsigned mc_persistent,mc_tracker_error;
static void mc_tracker_reset(void) {
    if(mc_live||mc_bytes){mc_tracker_error=1;return;}
    for(unsigned i=0;i<MC_TRACKED;i++)if(mc_allocations[i].pointer){mc_tracker_error=2;return;}
    mc_attempts=mc_successes=mc_frees=mc_peak=mc_fail_at=mc_failures=0;
    mc_persistent=mc_tracker_error=0;
}
int nms_test_allocation_permitted(uint64_t bytes) {
    if(mc_attempts==UINT64_MAX){mc_tracker_error=3;return 0;}
    mc_attempts++;
    if(mc_fail_at&&(mc_attempts==mc_fail_at||(mc_persistent&&mc_attempts>mc_fail_at))){mc_failures++;return 0;}
    if(mc_tracker_error||mc_live==MC_TRACKED||bytes>UINT64_MAX-mc_bytes){mc_tracker_error=4;return 0;}
    return 1;
}
void nms_test_allocation_created(void *pointer,uint64_t bytes) {
    if(!pointer||bytes>UINT64_MAX-mc_bytes){mc_tracker_error=5;return;}
    unsigned free_slot=MC_TRACKED;
    for(unsigned i=0;i<MC_TRACKED;i++){
        if(mc_allocations[i].pointer==pointer){mc_tracker_error=6;return;}
        if(!mc_allocations[i].pointer&&free_slot==MC_TRACKED)free_slot=i;
    }
    if(free_slot==MC_TRACKED){mc_tracker_error=7;return;}
    mc_allocations[free_slot].pointer=pointer;mc_allocations[free_slot].bytes=bytes;
    mc_successes++;mc_live++;mc_bytes+=bytes;if(mc_bytes>mc_peak)mc_peak=mc_bytes;
}
void nms_test_allocation_destroyed(void *pointer) {
    for(unsigned i=0;i<MC_TRACKED;i++)if(mc_allocations[i].pointer==pointer&&pointer){
        if(!mc_live||mc_allocations[i].bytes>mc_bytes){mc_tracker_error=8;return;}
        mc_bytes-=mc_allocations[i].bytes;mc_allocations[i].pointer=NULL;mc_allocations[i].bytes=0;mc_live--;mc_frees++;return;
    }
    mc_tracker_error=9;
}
#endif
