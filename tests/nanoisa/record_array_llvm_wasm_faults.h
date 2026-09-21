/* I run numeric workers against the same captured sequence and exact predicates. */
static size_t wasm_calls,wasm_peak,wasm_recoveries,wasm_successes;
static void wasm_reset(void) {
    ra_invariant(!ra_live&&!ra_bytes&&!nms_test_live_allocations());
    ra_calls=0;ra_created=0;ra_peak=0;ra_fail=SIZE_MAX;ra_persistent=0;
}
int nano_fault_baseline(void) {
    CHECK(!exercise());wasm_reset();CHECK(!sequence(0));
    CHECK(ra_calls>0&&ra_created==ra_calls);
    wasm_calls=ra_calls;wasm_peak=ra_peak;wasm_successes=ra_created;
    wasm_recoveries=0;return 0;
}
int nano_fault_range(uint32_t begin,uint32_t end,uint32_t wanted) {
    CHECK(!nano_fault_baseline());
    CHECK(wanted==wasm_calls&&begin<end&&end<=wasm_calls&&end-begin<=16);
    for(int mode=0;mode<2;mode++)for(size_t i=begin;i<end;i++) {
        wasm_reset();ra_fail=i;ra_persistent=mode;CHECK(!sequence(1));
        CHECK(ra_calls>i&&ra_created<ra_calls&&!ra_live&&!ra_bytes&&!nms_test_live_allocations());
        wasm_reset();CHECK(!sequence(0));
        CHECK(ra_calls==wasm_calls&&ra_created==wasm_successes&&ra_peak==wasm_peak);
        wasm_recoveries++;
    }
    CHECK(wasm_recoveries==2*(end-begin));return 0;
}
uint32_t nano_fault_calls(void) { return (uint32_t)wasm_calls; }
uint64_t nano_fault_peak(void) { return (uint64_t)wasm_peak; }
uint32_t nano_fault_recoveries(void) { return (uint32_t)wasm_recoveries; }
uint32_t nano_fault_backend_successes(void) { return (uint32_t)wasm_successes; }
