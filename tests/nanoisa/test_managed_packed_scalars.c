#include "../../src/nanoisa/managed_strings.h"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
typedef struct {uint32_t kind,tag;uint64_t bits;uint32_t result_tag;uint64_t result_bits;} Vector;
static const Vector vectors[]={
#include "packed_vectors.h"
};
int nms_packed_values(void) {
    NmsRuntime r;nms_init(&r,0,0);
    for(unsigned i=0;i<sizeof vectors/sizeof vectors[0];i++) {
        const Vector *v=&vectors[i];NmsHandle array;NmsValue out={999,999};uint32_t length=999;
        CHECK(nms_packed_array_create(&r,v->kind,&array)==NMS_OK);
        CHECK(nms_retain(&r,array)==NMS_OK);
        for(unsigned j=0;j<9;j++)CHECK(nms_value_array_append(&r,array,(NmsValue){v->bits,v->tag})==NMS_OK);
        CHECK(r.slots[(uint32_t)array].element_tag==v->kind);
        CHECK(r.live_bytes==(uint64_t)r.slots[(uint32_t)array].capacity*(v->kind==1||v->kind==3?8:1));
        CHECK(nms_value_array_set(&r,array,0,(NmsValue){v->bits,v->tag})==NMS_OK);
        CHECK(nms_value_array_get(&r,array,0,&out)==NMS_OK && out.tag==v->result_tag && out.payload==v->result_bits);
        CHECK(nms_value_array_pop(&r,array,&out)==NMS_OK && out.tag==v->result_tag && out.payload==v->result_bits);
        CHECK(nms_value_array_length(&r,array,&length)==NMS_OK && length==8);
        CHECK(nms_value_array_get(&r,array,UINT64_MAX,&out)==NMS_OK && !out.tag && !out.payload);
        CHECK(nms_release(&r,array)==NMS_OK && nms_release(&r,array)==NMS_OK);
        CHECK(!r.live_objects && !r.live_bytes);
    }
    CHECK(nms_dispose(&r)==NMS_OK);return 0;
}
int nms_packed_lifecycle(void) {
    NmsRuntime r,other;nms_init(&r,0,0);nms_init(&other,0,0);
    NmsHandle held[20],array=999,child;
    CHECK(nms_packed_array_create(&r,5,&array)==NMS_TYPE && array==999);
    for(unsigned i=0;i<20;i++) {
        CHECK(nms_packed_array_create(&r,1+i%4,&held[i])==NMS_OK);
        CHECK(nms_value_array_append(&r,held[i],(NmsValue){1,1+i%4})==NMS_OK);
    }
    for(unsigned i=0;i<20;i++) {
        NmsValue out={999,999};uint32_t length=999;
        CHECK(nms_value_array_get(&r,held[i],0,&out)==NMS_OK && out.tag==1+i%4 && out.payload==1);
        CHECK(nms_string_array_length(&r,held[i],&length)==NMS_TYPE && length==999);
    }
    CHECK(nms_create(&r,(const unsigned char*)"owned",5,&child)==NMS_OK);
    CHECK(nms_value_array_append(&r,held[0],(NmsValue){child,5})==NMS_TYPE);
    CHECK(r.slots[(uint32_t)child].references==1);
    CHECK(nms_value_array_set(&r,held[0],0,(NmsValue){0,3})==NMS_TYPE);
    CHECK(nms_value_array_set(&r,held[0],UINT64_MAX,(NmsValue){7,1})==NMS_STATE);
    CHECK(nms_value_array_append(&r,held[1],(NmsValue){256,2})==NMS_TYPE);
    CHECK(nms_value_array_append(&r,held[3],(NmsValue){2,4})==NMS_TYPE);
    CHECK(nms_release(&r,child)==NMS_OK);
    NmsHandle alias=held[0];CHECK(nms_retain(&r,alias)==NMS_OK);
    CHECK(nms_value_array_set(&r,held[0],0,(NmsValue){255,2})==NMS_OK);
    NmsValue changed;CHECK(nms_value_array_get(&r,alias,0,&changed)==NMS_OK && changed.tag==1 && changed.payload==255);
    CHECK(nms_release(&r,alias)==NMS_OK);
    CHECK(nms_packed_array_create(&other,1,&array)==NMS_OK);
    NmsValue out={999,999};CHECK(nms_value_array_pop(&other,array,&out)==NMS_OK && !out.tag && !out.payload);
    CHECK(nms_value_array_append(&other,array,(NmsValue){UINT64_MAX,1})==NMS_OK);
    CHECK(nms_dispose(&other)==NMS_OK && !other.live_bytes && !other.live_objects);
    CHECK(nms_value_array_get(&other,array,0,&out)==NMS_DISPOSED);
    CHECK(nms_dispose(&r)==NMS_OK && !r.live_bytes && !r.live_objects);return 0;
}
#ifdef NMS_TESTING
int nms_packed_failures(void) {
    NmsRuntime r;nms_init(&r,0,0);NmsHandle array=999;
    nms_test_fail_after(&r,0);CHECK(nms_packed_array_create(&r,1,&array)==NMS_MEMORY && array==999);
    nms_test_fail_after(&r,UINT64_MAX);CHECK(nms_packed_array_create(&r,1,&array)==NMS_OK);
    uint64_t allocations=nms_test_live_allocations();
    nms_test_fail_after(&r,0);CHECK(nms_value_array_append(&r,array,(NmsValue){7,1})==NMS_MEMORY);
    CHECK(!r.slots[(uint32_t)array].length && !r.live_bytes && allocations==nms_test_live_allocations());
    nms_test_fail_after(&r,UINT64_MAX);
    for(unsigned i=0;i<4;i++)CHECK(nms_value_array_append(&r,array,(NmsValue){i,1})==NMS_OK);
    unsigned char *old=r.slots[(uint32_t)array].data;uint64_t bytes=r.live_bytes;
    nms_test_fail_after(&r,0);CHECK(nms_value_array_append(&r,array,(NmsValue){4,1})==NMS_MEMORY);
    CHECK(r.slots[(uint32_t)array].data==old && r.slots[(uint32_t)array].length==4 && r.live_bytes==bytes);
    NmsValue out;CHECK(nms_value_array_get(&r,array,3,&out)==NMS_OK && out.payload==3 && out.tag==1);
    CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());return 0;
}
int nms_packed_reuse(void) {
    uint64_t pages=0;
    for(unsigned round=0;round<100;round++) {
        NmsRuntime r;nms_init(&r,0,0);NmsHandle array;
        CHECK(nms_packed_array_create(&r,3,&array)==NMS_OK);
        for(unsigned i=0;i<1024;i++)CHECK(nms_value_array_append(&r,array,(NmsValue){i,1})==NMS_OK);
        CHECK(nms_release(&r,array)==NMS_OK && !r.live_bytes && !r.live_objects);
        CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
        if(!round)pages=nms_test_memory_pages();
        CHECK(pages==nms_test_memory_pages());
    }
#ifdef __wasm32__
    NmsRuntime r;nms_init(&r,0,0);NmsHandle array;CHECK(nms_packed_array_create(&r,1,&array)==NMS_OK);
    unsigned count=0;NmsStatus status=NMS_OK;
    while(count<131072){status=nms_value_array_append(&r,array,(NmsValue){count,1});if(status)break;count++;}
    CHECK(status==NMS_MEMORY && count>1024 && r.slots[(uint32_t)array].length==count);
    NmsValue out;CHECK(nms_value_array_get(&r,array,count-1,&out)==NMS_OK && out.tag==1 && out.payload==count-1);
    CHECK(nms_release(&r,array)==NMS_OK && nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
#endif
    return 0;
}
#endif
#ifndef __wasm32__
int main(void){int result=nms_packed_values();if(!result)result=nms_packed_lifecycle();
#ifdef NMS_TESTING
if(!result)result=nms_packed_failures();if(!result)result=nms_packed_reuse();
#endif
return result?1:0;}
#endif
