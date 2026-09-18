/* I test private nested edges; explicit cycle collection is the next checkpoint. */
#include "../../src/nanoisa/managed_module.c"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char bytes[]={'a',0,255};
static const NmsView literals[]={{bytes,3}};
int nms_graph_edges(void) {
    NmsRuntime r;nms_init(&r,literals,1);
    NmsHandle child,parent,copy,text;
    CHECK(nms_create(&r,bytes,3,&text)==NMS_OK);
    CHECK(nms_vm_array_create(&r,5,&child)==NMS_OK);
    CHECK(nms_value_array_append(&r,child,(NmsValue){text,5})==NMS_OK);
    CHECK(nms_release(&r,text)==NMS_OK);
    uint64_t payloads[]={child,child};uint32_t tags[]={7,7};
    CHECK(nms_vm_array_literal(&r,7,payloads,tags,2,&parent)==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==3);
    CHECK(nms_vm_array_slice(&r,parent,0,2,&copy)==NMS_OK);
    CHECK(parent!=copy && r.slots[(uint32_t)child].references==5);
    CHECK(nms_value_array_append(&r,child,(NmsValue){42,1})==NMS_OK);
    NmsValue got={0};uint32_t length;
    CHECK(nms_value_array_get(&r,copy,0,&got)==NMS_OK && got.tag==7 && got.payload==child);
    CHECK(nms_value_array_length(&r,got.payload,&length)==NMS_OK && length==2);
    CHECK(nms_value_release(&r,got)==NMS_OK);
    CHECK(nms_value_array_set(&r,parent,0,(NmsValue){child,7})==NMS_OK);
    CHECK(r.slots[(uint32_t)child].references==5);
    CHECK(nms_release(&r,child)==NMS_OK);
    CHECK(nms_release(&r,parent)==NMS_OK);
    CHECK(nms_value_array_pop(&r,copy,&got)==NMS_OK && got.payload==child);
    CHECK(nms_release(&r,copy)==NMS_OK && r.live_objects==2);
    CHECK(nms_value_release(&r,got)==NMS_OK && !r.live_objects && !r.live_bytes);
    CHECK(nms_dispose(&r)==NMS_OK);
    return 0;
}
int nms_graph_chain(void) {
    NmsRuntime r;nms_init(&r,literals,1);NmsHandle root,next;
    CHECK(nms_vm_array_create(&r,1,&root)==NMS_OK);
    CHECK(nms_value_array_append(&r,root,(NmsValue){123,1})==NMS_OK);
    for(unsigned i=0;i<4096;i++) {
        CHECK(nms_value_array_create(&r,&next)==NMS_OK);
        CHECK(nms_value_array_append(&r,next,(NmsValue){root,7})==NMS_OK);
        CHECK(nms_release(&r,root)==NMS_OK);root=next;
    }
#ifdef NMS_TESTING
    nms_test_fail_after(&r,0);
#endif
    CHECK(nms_release(&r,root)==NMS_OK && !r.live_objects && !r.live_bytes);
#ifdef NMS_TESTING
    nms_test_fail_after(&r,UINT64_MAX);
#endif
    CHECK(nms_vm_array_create(&r,7,&root)==NMS_OK);
    CHECK(nms_release(&r,root)==NMS_OK);
    CHECK(nms_dispose(&r)==NMS_OK);
    return 0;
}
int nms_graph_cycles(void) {
    NmsRuntime r;nms_init(&r,literals,1);NmsHandle a,b;
    CHECK(nms_value_array_create(&r,&a)==NMS_OK);
    CHECK(nms_value_array_create(&r,&b)==NMS_OK);
    CHECK(nms_value_array_append(&r,a,(NmsValue){a,7})==NMS_OK);
    CHECK(nms_value_array_set(&r,a,0,(NmsValue){b,7})==NMS_OK);
    CHECK(nms_value_array_append(&r,b,(NmsValue){a,7})==NMS_OK);
    CHECK(nms_release(&r,b)==NMS_OK);
    /* Breaking the cycle publishes VOID before iterative release drops b's edge. */
    CHECK(nms_value_array_set(&r,a,0,(NmsValue){0,0})==NMS_OK);
    CHECK(r.live_objects==1 && r.slots[(uint32_t)a].references==1);
    CHECK(nms_value_array_append(&r,a,(NmsValue){a,7})==NMS_OK);
    CHECK(nms_release(&r,a)==NMS_OK && r.live_objects==1);
    /* Unrooted cycles await the separately implemented collector; disposal is terminal. */
    CHECK(nms_dispose(&r)==NMS_OK && !r.live_objects && !r.live_bytes);
    return 0;
}
int nms_graph_promotion(void) {
    NmsRuntime r;nms_init(&r,literals,1);NmsHandle a,child;
    CHECK(nms_string_array_create(&r,&a)==NMS_OK);
    CHECK(nms_string_array_append(&r,a,1)==NMS_OK);
    CHECK(nms_vm_array_create(&r,1,&child)==NMS_OK);
    CHECK(nms_value_array_append(&r,a,(NmsValue){child,7})==NMS_OK);
    CHECK(r.slots[(uint32_t)a].kind==NMS_SLOT_BOXED_ARRAY);
    CHECK(nms_release(&r,child)==NMS_OK);
    NmsValue out={999,999};
    CHECK(nms_value_array_get(&r,a,1,&out)==NMS_OK && out.tag==7);
    CHECK(nms_value_array_append(&r,out.payload,(NmsValue){1,1})==NMS_OK);
    CHECK(nms_value_array_append(&r,out.payload,(NmsValue){a,7})==NMS_TYPE);
    CHECK(nms_value_release(&r,out)==NMS_OK);
    CHECK(nms_release(&r,a)==NMS_OK && !r.live_objects && !r.live_bytes);
    CHECK(nms_dispose(&r)==NMS_OK);return 0;
}
#ifdef NMS_TESTING
int nms_graph_failures(void) {
    for(uint64_t budget=0;budget<2;budget++) {
        NmsRuntime r;nms_init(&r,literals,1);NmsHandle a,child,out=999;
        CHECK(nms_string_array_create(&r,&a)==NMS_OK);
        CHECK(nms_string_array_append(&r,a,1)==NMS_OK);
        CHECK(nms_vm_array_create(&r,7,&child)==NMS_OK);
        uint64_t before=r.live_bytes,refs=r.slots[(uint32_t)child].references;
        nms_test_fail_after(&r,0);
        CHECK(nms_value_array_append(&r,a,(NmsValue){child,7})==NMS_MEMORY);
        CHECK(r.slots[(uint32_t)a].kind==NMS_SLOT_STRING_ARRAY && r.live_bytes==before);
        CHECK(r.slots[(uint32_t)child].references==refs);
        nms_test_fail_after(&r,UINT64_MAX);
        while(r.free_head) {NmsHandle fill;CHECK(nms_value_array_create(&r,&fill)==NMS_OK);}
        uint64_t objects=r.live_objects;before=r.live_bytes;
        uint64_t payloads[]={child,child};uint32_t tags[]={7,7};
        nms_test_fail_after(&r,budget);
        CHECK(nms_vm_array_literal(&r,7,payloads,tags,2,&out)==NMS_MEMORY && out==999);
        CHECK(r.live_objects==objects && r.live_bytes==before && r.slots[(uint32_t)child].references==refs);
        nms_test_fail_after(&r,UINT64_MAX);
        CHECK(nms_dispose(&r)==NMS_OK);
    }
    CHECK(!nms_test_live_allocations());return 0;
}
#endif
#ifndef __wasm32__
int main(void) {
    int r=nms_graph_edges();if(r)return r;
    r=nms_graph_chain();if(r)return r;
    r=nms_graph_cycles();if(r)return r;
    r=nms_graph_promotion();if(r)return r;
#ifdef NMS_TESTING
    r=nms_graph_failures();if(r)return r;
#endif
    return 0;
}
#endif
