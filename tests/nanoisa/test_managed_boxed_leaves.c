#include "../../src/nanoisa/managed_strings.h"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char bytes[] = {'x', 0, 255};
static const NmsView literals[] = {{bytes, 3}};
static const NmsValue scalars[] = {
    {0,0}, {UINT64_MAX,1}, {255,2}, {UINT64_C(0x8000000000000000),3},
    {UINT64_C(0x7ff8000000001234),3}, {1,4}, {65535,9}
};
static int equal(NmsValue a, NmsValue b) { return a.payload == b.payload && a.tag == b.tag; }
int nms_leaf_tests(void) {
    NmsRuntime r, other; nms_init(&r,literals,1); nms_init(&other,literals,1);
    NmsHandle array, child, extra[20];
    CHECK(nms_value_array_create(&r,&array)==NMS_OK);
    CHECK(nms_retain(&r,array)==NMS_OK);
    CHECK(nms_create(&r,bytes,3,&child)==NMS_OK);
    NmsValue string={child,5}, value={999,999};
    for(unsigned i=0;i<sizeof scalars/sizeof scalars[0];i++) {
        CHECK(nms_value_retain(&r,scalars[i])==NMS_OK);
        CHECK(nms_value_array_append(&r,array,scalars[i])==NMS_OK);
        CHECK(nms_value_release(&r,scalars[i])==NMS_OK);
        CHECK(nms_value_array_get(&r,array,i,&value)==NMS_OK && equal(value,scalars[i]));
    }
    CHECK(nms_value_array_append(&r,array,string)==NMS_OK);
    CHECK(nms_value_array_append(&r,array,string)==NMS_OK);
    CHECK(nms_value_array_append(&r,array,(NmsValue){1,5})==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==3);
    CHECK(nms_value_array_set(&r,array,7,string)==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==3);
    CHECK(nms_value_array_set(&r,array,UINT64_MAX,string)==NMS_STATE);
    CHECK(nms_value_array_set(&r,array,0,(NmsValue){array,6})==NMS_TYPE);
    CHECK(nms_value_array_append(&r,array,(NmsValue){array,6})==NMS_TYPE);
    CHECK(nms_value_release(&r,(NmsValue){array,5})==NMS_TYPE);
    CHECK(r.slots[(uint32_t)array].references==2);
    for(unsigned i=0;i<20;i++) CHECK(nms_create(&r,bytes,3,&extra[i])==NMS_OK);
    CHECK(nms_value_array_get(&r,array,8,&value)==NMS_OK && equal(value,string));
    CHECK(nms_release(&r,child)==NMS_OK);
    CHECK(nms_value_array_set(&r,array,7,scalars[1])==NMS_OK);
    CHECK(nms_release(&r,array)==NMS_OK);
    CHECK(nms_release(&r,array)==NMS_OK);
    NmsView view;
    CHECK(nms_view(&r,value.payload,&view)==NMS_OK && view.length==3 && view.data[2]==255);
    CHECK(nms_value_release(&r,value)==NMS_OK);
    for(unsigned i=0;i<20;i++) CHECK(nms_release(&r,extra[i])==NMS_OK);
    CHECK(!r.live_objects && !r.live_bytes);
    /* Promotion preserves exact handles and transfers old edges, not copies. */
    CHECK(nms_string_array_create(&r,&array)==NMS_OK);
    CHECK(nms_create(&r,bytes,3,&child)==NMS_OK); string.payload=child;
    CHECK(nms_string_array_append(&r,array,child)==NMS_OK);
    CHECK(nms_string_array_append(&r,array,child)==NMS_OK);
    CHECK(nms_retain(&r,array)==NMS_OK);
    CHECK(nms_value_array_set(&r,array,0,scalars[4])==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==2);
    CHECK(nms_value_array_get(&r,array,0,&value)==NMS_OK && equal(value,scalars[4]));
    uint32_t length=777; NmsHandle handle=888;
    CHECK(nms_string_array_get(&r,array,0,&handle)==NMS_TYPE && handle==888);
    CHECK(nms_string_array_length(&r,array,&length)==NMS_TYPE && length==777);
    CHECK(nms_string_array_append(&r,array,child)==NMS_TYPE);
    CHECK(nms_value_array_length(&r,array,&length)==NMS_OK && length==2);
    CHECK(nms_value_array_pop(&r,array,&value)==NMS_OK && equal(value,string));
    CHECK(r.slots[(uint32_t)child].references==2);
    CHECK(nms_value_release(&r,value)==NMS_OK);
    CHECK(nms_value_array_pop(&r,array,&value)==NMS_OK && equal(value,scalars[4]));
    CHECK(nms_value_array_pop(&r,array,&value)==NMS_OK && value.tag==0 && value.payload==0);
    CHECK(nms_value_array_get(&r,array,UINT64_MAX,&value)==NMS_OK && value.tag==0 && value.payload==0);
    CHECK(nms_value_array_append(&r,array,string)==NMS_OK);
    CHECK(nms_release(&r,array)==NMS_OK && nms_release(&r,array)==NMS_OK);
    CHECK(nms_release(&r,child)==NMS_OK && !r.live_objects && !r.live_bytes);
    /* Generic pop also transfers an unpromoted string-array child. */
    CHECK(nms_string_array_create(&r,&array)==NMS_OK);
    CHECK(nms_create(&r,bytes,3,&child)==NMS_OK);string.payload=child;
    CHECK(nms_string_array_append(&r,array,child)==NMS_OK);
    CHECK(nms_release(&r,child)==NMS_OK);
    CHECK(nms_value_array_pop(&r,array,&value)==NMS_OK && equal(value,string));
    CHECK(nms_release(&r,array)==NMS_OK && nms_value_release(&r,value)==NMS_OK);
    CHECK(nms_value_array_create(&other,&array)==NMS_OK);
    CHECK(nms_create(&other,bytes,3,&child)==NMS_OK);
    CHECK(nms_value_array_append(&other,array,(NmsValue){child,5})==NMS_OK);
    CHECK(nms_value_array_append(&other,array,scalars[1])==NMS_OK);
    CHECK(nms_dispose(&other)==NMS_OK && !other.live_objects && !other.live_bytes);
    value=(NmsValue){888,999};
    CHECK(nms_value_array_get(&other,array,0,&value)==NMS_DISPOSED && value.payload==888 && value.tag==999);
    CHECK(nms_dispose(&r)==NMS_OK);
    return 0;
}
#ifdef NMS_TESTING
int nms_leaf_failures(void) {
    NmsRuntime r; nms_init(&r,literals,1);
    NmsHandle array=999,child,fill[7]; NmsValue value={888,999};
    nms_test_fail_after(&r,0);
    CHECK(nms_value_array_create(&r,&array)==NMS_MEMORY && array==999 && !r.live_objects);
    nms_test_fail_after(&r,UINT64_MAX);
    CHECK(nms_string_array_create(&r,&array)==NMS_OK);
    CHECK(nms_create(&r,bytes,3,&child)==NMS_OK);
    CHECK(nms_string_array_append(&r,array,child)==NMS_OK);
    NmsValue string={child,5};
    unsigned char *old=r.slots[(uint32_t)array].data;
    uint64_t live=r.live_bytes, allocations=nms_test_live_allocations();
    for(unsigned mode=0;mode<2;mode++) {
        nms_test_fail_after(&r,0);
        CHECK((mode?nms_value_array_set(&r,array,0,string):nms_value_array_append(&r,array,string))==NMS_MEMORY);
        CHECK(r.slots[(uint32_t)array].data==old && r.slots[(uint32_t)array].kind==NMS_SLOT_STRING_ARRAY);
        CHECK(r.slots[(uint32_t)array].length==1 && r.slots[(uint32_t)child].references==2);
        CHECK(r.live_bytes==live && nms_test_live_allocations()==allocations);
    }
    nms_test_fail_after(&r,UINT64_MAX);
    CHECK(nms_value_array_append(&r,array,string)==NMS_OK);
    CHECK(nms_value_array_append(&r,array,scalars[1])==NMS_OK);
    CHECK(nms_value_array_append(&r,array,scalars[2])==NMS_OK);
    old=r.slots[(uint32_t)array].data;live=r.live_bytes;allocations=nms_test_live_allocations();
    nms_test_fail_after(&r,0);
    CHECK(nms_value_array_append(&r,array,string)==NMS_MEMORY);
    CHECK(r.slots[(uint32_t)array].data==old && r.slots[(uint32_t)array].length==4);
    CHECK(r.slots[(uint32_t)child].references==3 && r.live_bytes==live && nms_test_live_allocations()==allocations);
    nms_test_fail_after(&r,UINT64_MAX);
    for(unsigned i=0;i<6;i++)CHECK(nms_value_array_create(&r,&fill[i])==NMS_OK);
    nms_test_fail_after(&r,0);NmsHandle out=999;
    CHECK(nms_value_array_create(&r,&out)==NMS_MEMORY && out==999 && r.capacity==8);
    CHECK(nms_value_array_get(&r,child,0,&value)==NMS_TYPE && value.payload==888 && value.tag==999);
    CHECK(nms_value_array_pop(&r,array,0)==NMS_STATE && r.slots[(uint32_t)array].length==4);
    CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
    return 0;
}
int nms_leaf_reuse(void) {
    uint64_t pages=0;
    for(unsigned round=0;round<100;round++) {
        NmsRuntime r;nms_init(&r,literals,1);NmsHandle array,child;
        CHECK(nms_string_array_create(&r,&array)==NMS_OK);
        CHECK(nms_create(&r,bytes,3,&child)==NMS_OK);
        CHECK(nms_string_array_append(&r,array,child)==NMS_OK);
        for(unsigned i=0;i<1024;i++)CHECK(nms_value_array_append(&r,array,i&1?(NmsValue){child,5}:scalars[4])==NMS_OK);
        CHECK(nms_release(&r,child)==NMS_OK && nms_release(&r,array)==NMS_OK);
        CHECK(!r.live_objects && !r.live_bytes && nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
        if(!round)pages=nms_test_memory_pages();
        CHECK(pages==nms_test_memory_pages());
    }
#ifdef __wasm32__
    NmsRuntime r;nms_init(&r,literals,1);NmsHandle array,child;
    CHECK(nms_value_array_create(&r,&array)==NMS_OK && nms_create(&r,bytes,3,&child)==NMS_OK);
    unsigned count=0;NmsStatus status=NMS_OK;
    while(count<131072) {status=nms_value_array_append(&r,array,(NmsValue){child,5});if(status)break;count++;}
    CHECK(status==NMS_MEMORY && count>1024 && r.slots[(uint32_t)array].length==count);
    CHECK(r.slots[(uint32_t)child].references==(uint64_t)count+1);
    CHECK(nms_release(&r,array)==NMS_OK && nms_release(&r,child)==NMS_OK);
    CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
#endif
    return 0;
}
#endif
