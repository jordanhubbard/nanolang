/* I qualify private preparation only; no literal/slice opcode is admitted. */
#include "../../src/nanoisa/managed_module.c"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char bytes[]={'a',0,255};
static const NmsView literals[]={{bytes,3}};
int nms_copy_core(void) {
    NmsRuntime r;nms_init(&r,literals,1);
    const uint32_t declarations[]={0,1,2,3,4,5,9};
    const uint32_t counts[]={0,1,7,8,9,16,17};
    uint64_t values[17];uint32_t tags[17];
    for(unsigned d=0;d<7;d++)for(unsigned n=0;n<7;n++) {
        uint32_t tag=declarations[d],count=counts[n];
        for(uint32_t i=0;i<count;i++) {
            tags[i]=packed_width(tag)?tag:(i%7==6?9:i%7);
            values[i]=tags[i]==5?1:tags[i]==4?i%2:tags[i]==3?UINT64_C(0x8000000000000000):i;
        }
        NmsHandle array=0;CHECK(nms_vm_array_literal(&r,tag,values,tags,count,&array)==NMS_OK);
        NmsSlot *slot=&r.slots[(uint32_t)array];
        CHECK(slot->length==count && slot->capacity==(count<8?8:count));
        CHECK(slot->element_tag==tag && slot->vm_array_policy);
        for(uint32_t i=0;i<count;i++) {
            NmsValue v;CHECK(nms_value_array_get(&r,array,i,&v)==NMS_OK && v.tag==tags[i] && v.payload==values[i]);
            CHECK(nms_value_release(&r,v)==NMS_OK);
        }
        NmsHandle copy=0;CHECK(nms_vm_array_slice(&r,array,1,UINT32_MAX,&copy)==NMS_OK && copy!=array);
        uint32_t expected=count?count-1:0;slot=&r.slots[(uint32_t)copy];
        CHECK(slot->length==expected && slot->capacity==(expected<8?8:expected));
        CHECK(slot->element_tag==tag && slot->vm_array_policy);
        for(uint32_t i=0;i<expected;i++) {
            NmsValue v;CHECK(nms_value_array_get(&r,copy,i,&v)==NMS_OK && v.tag==tags[i+1] && v.payload==values[i+1]);
            CHECK(nms_value_release(&r,v)==NMS_OK);
        }
        CHECK(nms_release(&r,array)==NMS_OK && nms_release(&r,copy)==NMS_OK);
    }
    const struct {uint32_t declared,input;uint64_t bits,expected;} pairs[]={
        {1,2,255,255},{2,1,UINT64_MAX,255},{3,1,UINT64_C(9007199254740993),UINT64_C(0x4340000000000000)}};
    for(unsigned i=0;i<3;i++) {
        NmsHandle a;CHECK(nms_vm_array_literal(&r,pairs[i].declared,&pairs[i].bits,&pairs[i].input,1,&a)==NMS_OK);
        NmsValue v;CHECK(nms_value_array_get(&r,a,0,&v)==NMS_OK && v.tag==pairs[i].declared && v.payload==pairs[i].expected);
        CHECK(nms_release(&r,a)==NMS_OK);
    }
    const uint64_t float_bits[]={1,UINT64_C(0x7fefffffffffffff),UINT64_C(0x7ff8000000000042),UINT64_C(0x8000000000000000)};
    const uint32_t float_tags[]={3,3,3,3};NmsHandle floating,copied;
    CHECK(nms_vm_array_literal(&r,3,float_bits,float_tags,4,&floating)==NMS_OK);
    const uint32_t bounds[][3]={{0,4,4},{3,1,0},{9,10,0},{0,UINT32_MAX,4},{UINT32_MAX,4,0}};
    for(unsigned i=0;i<5;i++) {
        CHECK(nms_vm_array_slice(&r,floating,bounds[i][0],bounds[i][1],&copied)==NMS_OK);
        CHECK(r.slots[(uint32_t)copied].length==bounds[i][2]);
        for(uint32_t j=0;j<bounds[i][2];j++) {NmsValue v;CHECK(nms_value_array_get(&r,copied,j,&v)==NMS_OK && v.tag==3 && v.payload==float_bits[j]);}
        CHECK(nms_release(&r,copied)==NMS_OK);
    }
    CHECK(nms_release(&r,floating)==NMS_OK);
    /* Private zero-buffer packed sources still produce a valid fresh empty copy. */
    NmsHandle empty,copy;CHECK(nms_packed_array_create(&r,3,&empty)==NMS_OK);
    CHECK(nms_vm_array_slice(&r,empty,0,0,&copy)==NMS_OK);
    CHECK(nms_release(&r,empty)==NMS_OK && nms_release(&r,copy)==NMS_OK);
    CHECK(!r.live_objects && !r.live_bytes && nms_dispose(&r)==NMS_OK);
    return 0;
}
int nms_copy_aliases(void) {
    NmsRuntime r;nms_init(&r,literals,1);NmsHandle child,source,copy;
    CHECK(nms_create(&r,bytes,3,&child)==NMS_OK);
    uint64_t values[]={child,child,1};uint32_t tags[]={5,5,5};
    CHECK(nms_vm_array_literal(&r,5,values,tags,3,&source)==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==3);
    NmsHandle held[64];unsigned count=0;
    while(r.free_head) {CHECK(count<64);CHECK(nms_vm_array_create(&r,1,&held[count++])==NMS_OK);}
    uint32_t old_capacity=r.capacity;
    CHECK(nms_vm_array_slice(&r,source,0,2,&copy)==NMS_OK && r.capacity>old_capacity);
    CHECK(r.slots[(uint32_t)child].references==5);
    CHECK(nms_value_array_set(&r,source,0,(NmsValue){42,1})==NMS_OK);
    NmsValue v;CHECK(nms_value_array_get(&r,copy,0,&v)==NMS_OK && v.tag==5 && v.payload==child);
    CHECK(nms_value_release(&r,v)==NMS_OK);
    CHECK(nms_value_array_set(&r,copy,1,(NmsValue){1,4})==NMS_OK);
    CHECK(nms_value_array_get(&r,source,1,&v)==NMS_OK && v.tag==5 && v.payload==child);
    CHECK(nms_value_release(&r,v)==NMS_OK);
    CHECK(nms_release(&r,source)==NMS_OK && nms_release(&r,child)==NMS_OK);
    CHECK(nms_value_array_get(&r,copy,0,&v)==NMS_OK);
    NmsView view;CHECK(nms_view(&r,v.payload,&view)==NMS_OK && view.length==3 && view.data[1]==0 && view.data[2]==255);
    CHECK(nms_release(&r,copy)==NMS_OK);
    CHECK(nms_view(&r,v.payload,&view)==NMS_OK && nms_value_release(&r,v)==NMS_OK);
    for(unsigned i=0;i<count;i++)CHECK(nms_release(&r,held[i])==NMS_OK);
    /* Legacy string-array copying prepares boxed storage without changing source. */
    CHECK(nms_string_array_create(&r,&source)==NMS_OK && nms_string_array_append(&r,source,1)==NMS_OK);
    CHECK(nms_vm_array_slice(&r,source,0,1,&copy)==NMS_OK);
    CHECK(r.slots[(uint32_t)source].kind==NMS_SLOT_STRING_ARRAY);
    CHECK(r.slots[(uint32_t)copy].kind==NMS_SLOT_BOXED_LEAF_ARRAY && r.slots[(uint32_t)copy].element_tag==5);
    CHECK(nms_release(&r,source)==NMS_OK && nms_release(&r,copy)==NMS_OK);
    CHECK(!r.live_objects && !r.live_bytes && nms_dispose(&r)==NMS_OK);
    return 0;
}
int nms_copy_abi(void) {
    uint64_t values[]={10,20,30};uint32_t tags[]={1,1,1};
    CHECK(nms_module_begin(literals,1)==NMS_OK);
    uint64_t array=nms_module_array_literal(1,3,values,tags);CHECK(array);
    uint64_t copy=nms_module_array_slice(array,UINT64_C(4294967297),1,UINT64_MAX,1);CHECK(copy && copy!=array);
    uint64_t bits=0;uint32_t tag=0;
    CHECK(nms_module_array_get_value(copy,0,&bits,&tag)==NMS_OK && bits==20 && tag==1);
    nms_module_release(copy,7);
    copy=nms_module_array_slice(array,1,5,1,5);CHECK(copy && nms_module_array_value_length(copy)==3);
    nms_module_release(array,7);nms_module_release(copy,7);
    CHECK(nms_module_finish(0)==0 && nms_module_live_objects()==0);
    CHECK(nms_module_dispose()==NMS_OK);
    return 0;
}
#ifdef NMS_TESTING
int nms_copy_failures(void) {
    for(unsigned budget=0;budget<2;budget++) {
        NmsRuntime r;nms_init(&r,literals,1);NmsHandle out=999;
        uint64_t bits[]={1,1};uint32_t tags[]={5,5};nms_test_fail_after(&r,budget);
        CHECK(nms_vm_array_literal(&r,5,bits,tags,2,&out)==NMS_MEMORY && out==999);
        CHECK(!r.live_objects && !r.live_bytes && !nms_test_live_allocations());
        CHECK(nms_dispose(&r)==NMS_OK);
    }
    NmsRuntime r;nms_init(&r,literals,1);NmsHandle child,source,held[64];unsigned count=0;
    CHECK(nms_create(&r,bytes,3,&child)==NMS_OK);
    uint64_t bits[]={child,child};uint32_t tags[]={5,5};
    CHECK(nms_vm_array_literal(&r,5,bits,tags,2,&source)==NMS_OK);
    while(r.free_head){CHECK(count<64);CHECK(nms_vm_array_create(&r,1,&held[count++])==NMS_OK);}
    uint64_t objects=r.live_objects,bytes_before=r.live_bytes,owners=r.slots[(uint32_t)child].references;
    for(unsigned budget=0;budget<2;budget++) {
        NmsHandle out=999;nms_test_fail_after(&r,budget);
        CHECK(nms_vm_array_slice(&r,source,0,2,&out)==NMS_MEMORY && out==999);
        CHECK(r.live_objects==objects && r.live_bytes==bytes_before && r.slots[(uint32_t)child].references==owners);
    }
    nms_test_fail_after(&r,UINT64_MAX);
    NmsHandle out=999;uint32_t badtags[]={5,7};
    CHECK(nms_vm_array_literal(&r,5,bits,badtags,2,&out)==NMS_TYPE && out==999);
    CHECK(nms_vm_array_literal(&r,1,bits,tags,2,&out)==NMS_TYPE && out==999);
    CHECK(nms_vm_array_literal(&r,1,0,0,65536,&out)==NMS_STATE && out==999);
    CHECK(nms_vm_array_literal(&r,1,0,0,1,&out)==NMS_STATE && out==999);
    CHECK(nms_vm_array_slice(&r,child,0,1,&out)==NMS_TYPE && out==999);
    CHECK(nms_vm_array_slice(&r,source,0,1,0)==NMS_STATE);
    /* I exercise the existing checked retain limit: one child edge prepares,
     * the next refuses, and rollback restores the original count. */
    r.slots[(uint32_t)child].references=UINT64_MAX-1;
    CHECK(nms_vm_array_literal(&r,5,bits,tags,2,&out)==NMS_MEMORY && out==999);
    CHECK(r.slots[(uint32_t)child].references==UINT64_MAX-1 && r.live_objects==objects && r.live_bytes==bytes_before);
    CHECK(nms_vm_array_slice(&r,source,0,2,&out)==NMS_MEMORY && out==999);
    CHECK(r.slots[(uint32_t)child].references==UINT64_MAX-1 && r.live_objects==objects && r.live_bytes==bytes_before);
    r.slots[(uint32_t)child].references=owners;
    CHECK(nms_release(&r,source)==NMS_OK && nms_release(&r,child)==NMS_OK);
    for(unsigned i=0;i<count;i++)CHECK(nms_release(&r,held[i])==NMS_OK);
    CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
    return 0;
}
int nms_copy_pressure(void) {
    NmsRuntime r;nms_init(&r,0,0);NmsHandle source,held[20];unsigned count=0;
    CHECK(nms_vm_array_create(&r,1,&source)==NMS_OK);
    for(uint32_t i=0;i<32768;i++)CHECK(nms_value_array_append(&r,source,(NmsValue){i,1})==NMS_OK);
    for(;count<20;count++) {
        NmsHandle out=999;NmsStatus status=nms_vm_array_slice(&r,source,0,32768,&out);
        if(status!=NMS_OK) {
            CHECK(status==NMS_MEMORY && out==999);
            NmsValue v;CHECK(nms_value_array_get(&r,source,32767,&v)==NMS_OK && v.payload==32767);
            CHECK(nms_vm_array_slice(&r,source,0,32768,&out)==NMS_MEMORY && out==999);
            break;
        }
        held[count]=out;
    }
#ifdef __wasm32__
    CHECK(count>0 && count<20);
#endif
    for(unsigned i=0;i<count;i++)CHECK(nms_release(&r,held[i])==NMS_OK);
    CHECK(nms_release(&r,source)==NMS_OK && !r.live_objects && !r.live_bytes);
    CHECK(nms_dispose(&r)==NMS_OK && !nms_test_live_allocations());
    return 0;
}
#endif
#ifndef __wasm32__
int main(void) {
    int result=nms_copy_core();if(result)return result;
    result=nms_copy_aliases();if(result)return result;
#ifdef NMS_TESTING
    result=nms_copy_failures();if(result)return result;
    result=nms_copy_pressure();if(result)return result;
#endif
    return nms_copy_abi();
}
#endif
