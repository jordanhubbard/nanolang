/* I test the private adapter checkpoint; emitted instruction ownership is later. */
#include "../../src/nanoisa/managed_module.c"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char source_bytes[]={'a',0,255,'b'}, empty_bytes[]={0};
static const NmsView literals[]={{source_bytes,4},{empty_bytes,0}};
int nms_mutable_core(void) {
    const uint32_t tags[]={0,1,2,3,4,5,9};
    NmsRuntime r;nms_init(&r,literals,2);
    NmsHandle held[20];
    for(unsigned i=0;i<20;i++) {
        uint32_t tag=tags[i%7];
        CHECK(nms_vm_array_create(&r,tag,&held[i])==NMS_OK);
        NmsSlot *slot=&r.slots[(uint32_t)held[i]];
        CHECK(slot->capacity==8 && slot->vm_array_policy==1 && slot->element_tag==tag);
        CHECK(slot->kind==(packed_width(tag)?NMS_SLOT_PACKED_SCALAR_ARRAY:NMS_SLOT_BOXED_LEAF_ARRAY));
    }
    for(unsigned i=0;i<20;i++) {
        uint32_t tag=tags[i%7];NmsValue value={tag?1:0,tag},out={999,999};
        CHECK(r.slots[(uint32_t)held[i]].element_tag==tag && r.slots[(uint32_t)held[i]].vm_array_policy==1);
        for(unsigned count=0;count<33;count++) {
            CHECK(nms_value_array_append(&r,held[i],value)==NMS_OK);
            uint32_t expected=count<8?8:count<16?16:count<32?32:64;
            CHECK(r.slots[(uint32_t)held[i]].capacity==expected);
        }
#ifdef NMS_TESTING
        nms_test_fail_after(&r,0);
#endif
        CHECK(nms_value_array_set(&r,held[i],0,value)==NMS_OK);
#ifdef NMS_TESTING
        nms_test_fail_after(&r,UINT64_MAX);
#endif
        CHECK(nms_value_array_get(&r,held[i],0,&out)==NMS_OK && out.tag==value.tag && out.payload==value.payload);
        CHECK(nms_release(&r,held[i])==NMS_OK);
    }
    NmsHandle array,child;CHECK(nms_split_values_owned(&r,1,2,&array)==NMS_OK);
    CHECK(r.slots[(uint32_t)array].kind==NMS_SLOT_BOXED_LEAF_ARRAY && r.slots[(uint32_t)array].capacity==8);
    NmsValue out;CHECK(nms_value_array_get(&r,array,1,&out)==NMS_OK && out.tag==5);
    NmsView view;CHECK(nms_view(&r,out.payload,&view)==NMS_OK && view.length==1 && view.data[0]==0);
    CHECK(nms_value_release(&r,out)==NMS_OK);
    CHECK(nms_create(&r,(const unsigned char*)"new",3,&child)==NMS_OK);
#ifdef NMS_TESTING
    nms_test_fail_after(&r,0);
#endif
    CHECK(nms_value_array_set(&r,array,1,(NmsValue){child,5})==NMS_OK);
    CHECK(nms_value_array_set(&r,array,1,(NmsValue){child,5})==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==2);
#ifdef NMS_TESTING
    nms_test_fail_after(&r,UINT64_MAX);
#endif
    CHECK(nms_release(&r,child)==NMS_OK && nms_release(&r,array)==NMS_OK);
    CHECK(nms_vm_array_create(&r,5,&array)==NMS_OK);
    CHECK(nms_create(&r,(const unsigned char*)"last",4,&child)==NMS_OK);
    CHECK(nms_value_array_append(&r,array,(NmsValue){child,5})==NMS_OK && nms_release(&r,child)==NMS_OK);
    CHECK(nms_value_array_pop(&r,array,&out)==NMS_OK && nms_release(&r,array)==NMS_OK);
    CHECK(out.payload==child && out.tag==5 && nms_view(&r,out.payload,&view)==NMS_OK && view.length==4);
    CHECK(nms_value_release(&r,out)==NMS_OK && !r.live_objects && !r.live_bytes);
    CHECK(nms_value_array_create(&r,&array)==NMS_OK && !r.slots[(uint32_t)array].vm_array_policy && !r.slots[(uint32_t)array].capacity);
    CHECK(nms_release(&r,array)==NMS_OK);
    uint32_t capacity=999;
    CHECK(array_next_capacity(UINT32_C(1)<<27,16,1,&capacity)==NMS_MEMORY && capacity==999);
    CHECK(array_next_capacity(UINT32_C(1)<<26,16,1,&capacity)==NMS_OK && capacity==(UINT32_C(1)<<27));
    capacity=999;CHECK(array_next_capacity(UINT32_C(1)<<28,8,1,&capacity)==NMS_MEMORY && capacity==999);
    CHECK(array_next_capacity(UINT32_C(1)<<27,8,1,&capacity)==NMS_OK && capacity==(UINT32_C(1)<<28));
    capacity=999;CHECK(array_next_capacity(UINT32_C(1)<<31,1,1,&capacity)==NMS_MEMORY && capacity==999);
    CHECK(array_next_capacity(UINT32_C(1)<<30,1,1,&capacity)==NMS_OK && capacity==(UINT32_C(1)<<31));
    CHECK(nms_dispose(&r)==NMS_OK);return 0;
}
int nms_mutable_abi(void) {
    CHECK(nms_module_begin(literals,2)==NMS_OK);
    uint64_t array=nms_module_array_create(5),child=nms_module_concat(1,2),bits=999;uint32_t tag=999;
    CHECK(array && child && !nms_module_status());
    CHECK(nms_module_array_append_value(array,child,5)==NMS_OK);
    nms_module_release(child,5);
    CHECK(nms_module_array_get_value(array,0,&bits,&tag)==NMS_OK && bits==child && tag==5);
    nms_module_release(bits,tag);
    CHECK(nms_module_array_pop_value(array,&bits,&tag)==NMS_OK && bits==child && tag==5);
    nms_module_release(array,7);
    CHECK(nms_module_length(bits)==4);nms_module_release(bits,tag);
    array=nms_module_array_create(1);bits=999;tag=999;
    CHECK(nms_module_array_get_value(array,UINT64_MAX,&bits,&tag)==NMS_OK && !bits && !tag);
    CHECK(nms_module_array_pop_value(array,&bits,&tag)==NMS_OK && !bits && !tag);
    CHECK(nms_module_array_set_value(array,UINT64_MAX,7,1)==NMS_BOUNDS);
    nms_module_fail(NMS_TYPE);CHECK(nms_module_status()==NMS_BOUNDS);
    CHECK(nms_module_finish(0)==((uint64_t)NMS_BOUNDS<<32));
    CHECK(nms_module_begin(literals,2)==NMS_OK && !nms_module_status());
    CHECK(nms_module_array_append_value(array,3,1)==NMS_OK && nms_module_array_value_length(array)==1);
    nms_module_release(array,7);CHECK(!nms_module_live_objects() && !nms_module_live_bytes());
    CHECK(nms_module_finish(0)==0 && nms_module_dispose()==NMS_OK);
    CHECK(nms_module_begin(literals,2)==NMS_DISPOSED);return 0;
}
#ifdef NMS_TESTING
int nms_mutable_failures(void) {
    for(unsigned budget=0;budget<2;budget++) {
        NmsRuntime r;nms_init(&r,0,0);NmsHandle out=999;nms_test_fail_after(&r,budget);
        CHECK(nms_vm_array_create(&r,1,&out)==NMS_MEMORY && out==999);
        CHECK(!r.live_objects && !r.live_bytes && !nms_test_live_allocations());
        CHECK(nms_dispose(&r)==NMS_OK);
    }
    unsigned failed=0,succeeded=0;
    static const unsigned char longer[]="abcdefghijklmnopqrst";
    for(unsigned budget=0;budget<32;budget++) {
        NmsRuntime r;nms_init(&r,literals,2);NmsHandle source=0,out=999;
        CHECK(nms_create(&r,longer,20,&source)==NMS_OK && nms_retain(&r,source)==NMS_OK);
        nms_test_fail_after(&r,budget);
        NmsStatus status=nms_split_values_owned(&r,source,2,&out);
        CHECK(status==NMS_OK || status==NMS_MEMORY);
        CHECK(r.slots[(uint32_t)source].references==1);
        if(status==NMS_OK){succeeded++;CHECK(nms_release(&r,out)==NMS_OK);}
        else {failed++;CHECK(out==999);}
        CHECK(nms_release(&r,source)==NMS_OK && !r.live_objects && !r.live_bytes);
        CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
    }
    CHECK(failed && succeeded);
    NmsRuntime r;nms_init(&r,0,0);NmsHandle array;
    CHECK(nms_vm_array_create(&r,1,&array)==NMS_OK && nms_retain(&r,array)==NMS_OK);
    for(unsigned i=0;i<8;i++)CHECK(nms_value_array_append(&r,array,(NmsValue){i,1})==NMS_OK);
    unsigned char *old=r.slots[(uint32_t)array].data;uint64_t bytes=r.live_bytes;
    nms_test_fail_after(&r,0);CHECK(nms_value_array_append(&r,array,(NmsValue){8,1})==NMS_MEMORY);
    CHECK(r.slots[(uint32_t)array].data==old && r.slots[(uint32_t)array].length==8 && r.live_bytes==bytes);
    CHECK(nms_release(&r,array)==NMS_OK);
    NmsValue out;CHECK(nms_value_array_get(&r,array,7,&out)==NMS_OK && out.tag==1 && out.payload==7);
    CHECK(nms_release(&r,array)==NMS_OK && nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
    return 0;
}
int nms_mutable_pressure(void) {
    uint64_t pages=0;
    for(unsigned round=0;round<100;round++) {
        NmsRuntime r;nms_init(&r,0,0);NmsHandle array;
        CHECK(nms_vm_array_create(&r,1,&array)==NMS_OK);
        for(unsigned i=0;i<257;i++)CHECK(nms_value_array_append(&r,array,(NmsValue){i,1})==NMS_OK);
        CHECK(nms_release(&r,array)==NMS_OK && nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
        if(!round)pages=nms_test_memory_pages();
        CHECK(pages==nms_test_memory_pages());
    }
#ifdef __wasm32__
    NmsRuntime r;nms_init(&r,0,0);NmsHandle array;CHECK(nms_vm_array_create(&r,1,&array)==NMS_OK);
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
int main(void) {
    int result=nms_mutable_core();
#ifdef NMS_TESTING
    if(!result)result=nms_mutable_failures();
    if(!result)result=nms_mutable_pressure();
#endif
    if(!result)result=nms_mutable_abi();
    return result?1:0;
}
#endif
