/* I qualify private complete-owner collection boundaries, not emitted admission. */
#include "../../src/nanoisa/managed_module.c"
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char text[] = {'x',0,'y'};
static const NmsView literals[] = {{text,3}};
int nms_prepared_roots(void) {
    uint64_t marks,queue,size;
    CHECK(collection_layout(0,&marks,&queue,&size) && marks==8 && queue==12 && size==16);
    CHECK(collection_layout(8,&marks,&queue,&size) && marks==72 && queue==84 && size==120);
#ifdef __wasm32__
    CHECK(!collection_layout(UINT32_MAX,&marks,&queue,&size));
#else
    CHECK(collection_layout(UINT32_MAX,&marks,&queue,&size));
    CHECK(size==UINT64_C(55834574848) && !(queue&3));
#endif
    NmsRuntime r; nms_init(&r,literals,1);
    CHECK(nms_collect_prepared(&r)==NMS_STATE);
    CHECK(nms_prepare_collection(&r)==NMS_OK);
    CHECK(r.collection_workspace && r.collection_capacity==0);
    CHECK(nms_collect_prepared(&r)==NMS_OK);
    NmsHandle a,b,keep;
    CHECK(nms_create(&r,text,3,&keep)==NMS_OK);
    for(unsigned i=0;i<2000;i++) {
        CHECK(nms_vm_array_create(&r,7,&a)==NMS_OK);
        CHECK(nms_vm_array_create(&r,7,&b)==NMS_OK);
        CHECK(nms_value_array_append(&r,a,(NmsValue){b,7})==NMS_OK);
        CHECK(nms_value_array_append(&r,b,(NmsValue){a,7})==NMS_OK);
        CHECK(nms_value_array_append(&r,b,(NmsValue){keep,5})==NMS_OK);
        CHECK(nms_release(&r,b)==NMS_OK);
        /* A returned temporary is a real retained owner outside any slot. */
        NmsValue returned;
        CHECK(nms_value_array_get(&r,a,0,&returned)==NMS_OK);
        CHECK(nms_release(&r,a)==NMS_OK);
#ifdef NMS_TESTING
        nms_test_fail_after(&r,0);
#endif
        CHECK(nms_collect_prepared(&r)==NMS_OK && r.live_objects==3);
        CHECK(nms_value_release(&r,returned)==NMS_OK);
        CHECK(nms_collect_prepared(&r)==NMS_OK && r.live_objects==1);
        CHECK(r.slots[(uint32_t)keep].references==1);
#ifdef NMS_TESTING
        nms_test_fail_after(&r,UINT64_MAX);
#endif
    }
    CHECK(nms_release(&r,keep)==NMS_OK);
    CHECK(nms_dispose(&r)==NMS_OK && !r.collection_workspace);
    CHECK(nms_prepare_collection(&r)==NMS_DISPOSED);
    return 0;
}
static void reset_module(void) {
    /* Each test owns and disposes this private instance before fresh storage. */
    nms_init(&nms_module_instance,literals,1);
    nms_module_ready=1; nms_module_error=NMS_OK;
}
int nms_prepared_module(void) {
    reset_module();
    CHECK(nms_module_graph_collect()==NMS_STATE);
    CHECK(nms_module_graph_begin(literals,1)==NMS_OK);
    NmsHandle global;
    CHECK(nms_vm_array_create(&nms_module_instance,7,&global)==NMS_OK);
    CHECK(nms_value_array_append(&nms_module_instance,global,(NmsValue){global,7})==NMS_OK);
    CHECK(nms_module_graph_finish(17)==17);
    for(unsigned i=0;i<500;i++) {
        CHECK(nms_module_graph_begin(literals,1)==NMS_OK);
        NmsHandle dead;
        CHECK(nms_vm_array_create(&nms_module_instance,7,&dead)==NMS_OK);
        CHECK(nms_value_array_append(&nms_module_instance,dead,(NmsValue){dead,7})==NMS_OK);
        CHECK(nms_release(&nms_module_instance,dead)==NMS_OK);
        nms_module_fail(NMS_ASSERT);
        void *workspace=nms_module_instance.collection_workspace;
        CHECK(nms_module_graph_begin(NULL,0)==NMS_BUSY);
        CHECK(nms_module_instance.active && nms_module_error==NMS_ASSERT);
        CHECK(nms_module_instance.collection_workspace==workspace);
        CHECK(nms_module_dispose()==NMS_BUSY);
        CHECK(nms_module_graph_collect()==NMS_ASSERT);
        CHECK(nms_module_instance.live_objects==1);
        CHECK(nms_module_graph_finish(88)==((uint64_t)NMS_ASSERT<<32));
        CHECK(!nms_module_instance.active);
        CHECK(nms_module_instance.slots[(uint32_t)global].references==2);
    }
    CHECK(nms_module_graph_begin(literals,1)==NMS_OK);
    CHECK(nms_release(&nms_module_instance,global)==NMS_OK);
    CHECK(nms_module_graph_finish(9)==9 && !nms_module_instance.live_objects);
    CHECK(nms_module_dispose()==NMS_OK);
    CHECK(nms_module_graph_begin(literals,1)==NMS_DISPOSED);
    return 0;
}
#ifdef __wasm32__
int nms_prepared_pressure(void) {
    NmsRuntime r;nms_init(&r,literals,1);
    CHECK(nms_prepare_collection(&r)==NMS_OK);
    NmsHandle cycle;
    CHECK(nms_vm_array_create(&r,7,&cycle)==NMS_OK);
    CHECK(nms_value_array_append(&r,cycle,(NmsValue){cycle,7})==NMS_OK);
    CHECK(nms_release(&r,cycle)==NMS_OK);
    void *held[1024];unsigned count=0;
    while(count<1024) {
        void *block=allocate(&r,4096);
        if(!block)break;
        held[count++]=block;
    }
    CHECK(count && count<1024);
    CHECK(nms_collect_prepared(&r)==NMS_OK && !r.live_objects);
    for(unsigned i=0;i<count;i++)deallocate(held[i]);
    CHECK(nms_vm_array_create(&r,7,&cycle)==NMS_OK);
    CHECK(nms_release(&r,cycle)==NMS_OK);
    CHECK(nms_dispose(&r)==NMS_OK);
    return 0;
}
#endif
#ifdef NMS_TESTING
int nms_prepared_failures(void) {
    NmsRuntime r; nms_init(&r,literals,1);
    nms_test_fail_after(&r,0);
    CHECK(nms_prepare_collection(&r)==NMS_MEMORY);
    CHECK(!r.collection_prepared && !r.collection_workspace && !r.slots);
    nms_test_fail_after(&r,UINT64_MAX);
    CHECK(nms_prepare_collection(&r)==NMS_OK);
    for(unsigned i=0;i<8;i++) {NmsHandle a;CHECK(nms_value_array_create(&r,&a)==NMS_OK);}
    CHECK(!r.free_head && r.capacity==8);
    NmsSlot *old=r.slots;void *workspace=r.collection_workspace;
    uint64_t allocations=nms_test_live_allocations();
    for(unsigned budget=0;budget<2;budget++) {
        NmsHandle out=999;
        nms_test_fail_after(&r,budget);
        CHECK(nms_value_array_create(&r,&out)==NMS_MEMORY && out==999);
        CHECK(r.slots==old && r.collection_workspace==workspace && r.capacity==8);
        CHECK(r.live_objects==8 && !r.free_head && r.collection_capacity==8);
        CHECK(nms_test_live_allocations()==allocations);
        CHECK(nms_collect_prepared(&r)==NMS_OK);
    }
    nms_test_fail_after(&r,UINT64_MAX);
    NmsHandle out;CHECK(nms_value_array_create(&r,&out)==NMS_OK);
    CHECK(r.capacity==16 && r.collection_capacity==16);
    CHECK(nms_dispose(&r)==NMS_OK);
    reset_module();nms_test_fail_after(&nms_module_instance,0);
    CHECK(nms_module_graph_begin(literals,1)==NMS_MEMORY);
    CHECK(nms_module_instance.active && nms_module_status()==NMS_MEMORY);
    CHECK(nms_module_graph_begin(literals,1)==NMS_BUSY);
    CHECK(nms_module_graph_finish(7)==((uint64_t)NMS_MEMORY<<32));
    nms_test_fail_after(&nms_module_instance,UINT64_MAX);
    CHECK(nms_module_graph_begin(literals,1)==NMS_OK);
    CHECK(nms_module_graph_finish(7)==7);
    CHECK(nms_module_dispose()==NMS_OK && !nms_test_live_allocations());
    return 0;
}
#endif
#ifndef __wasm32__
int main(void) {
    int result=nms_prepared_roots();if(result)return result;
    result=nms_prepared_module();if(result)return result;
#ifdef NMS_TESTING
    result=nms_prepared_failures();if(result)return result;
#endif
    return 0;
}
#endif
