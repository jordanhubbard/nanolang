/* I reject every ABI disagreement before allocation or runtime creation. */
#include "record_array_generated_private.h"
#include "record_array_llvm_wasm_alloc.h"
NrgStatus nrg_generated_create(NrgInstance **);
uint32_t nrg_layout_field_actual(uint32_t);
NrgStatus nrg_create_actual(const NrgProgram *,NrgInstance **);
static uint32_t wrong=UINT32_MAX,queries,creates;
uint32_t nrg_layout_field(uint32_t field) {
    queries++;
    uint32_t value=nrg_layout_field_actual(field);
    return field==wrong?value^1u:value;
}
NrgStatus nrg_create(const NrgProgram *program,NrgInstance **out) {
    creates++;return nrg_create_actual(program,out);
}
#define REQUIRE(x) do { if(!(x))return __LINE__; } while(0)
int nano_main(void) {
    REQUIRE(NRG_LAYOUT_COUNT==50);
    REQUIRE(nrg_layout_field_actual(UINT32_MAX)==UINT32_MAX);
#ifdef NRG_EXPECT_TABLE_REFUSAL
    NrgInstance *output=(void *)(uintptr_t)1;
    REQUIRE(nrg_generated_create(&output)==NRG_STATE);
    REQUIRE(output==(void *)(uintptr_t)1&&queries==51&&!creates&&!ra_calls);
#else
    for(wrong=0;wrong<=NRG_LAYOUT_COUNT;wrong++) {
        NrgInstance *output=(void *)(uintptr_t)1;queries=0;
        REQUIRE(nrg_generated_create(&output)==NRG_STATE);
        REQUIRE(output==(void *)(uintptr_t)1&&queries==wrong+1&&!creates&&!ra_calls);
    }
    wrong=UINT32_MAX;queries=0;NrgInstance *output=(void *)(uintptr_t)1;
    REQUIRE(nrg_generated_create(&output)==NRG_OK&&output!=(void *)(uintptr_t)1);
    REQUIRE(queries==51&&creates==1&&ra_calls>0);nrg_destroy(output);
#endif
    REQUIRE(!ra_live&&!ra_bytes&&!nms_test_live_allocations());return 0;
}
#ifndef __wasm32__
int main(void) { return nano_main()!=0; }
#endif
