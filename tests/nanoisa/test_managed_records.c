/* I qualify private identities and complete child owners, not profile authority. */
#ifdef NMS_RECORD_LINKED
#include "../../src/nanoisa/managed_strings.h"
#else
#include "../../src/nanoisa/managed_strings.c"
#endif
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
static const unsigned char bytes[] = {'a',0,'b'};
static const NmsView literals[] = {{bytes,3}};
static const NmsRecordDescriptor definitions[] = {{0,0},{3,3},{5,1},{7,1}};
int record_values(void) {
    NmsRuntime r; nms_init(&r,literals,1);
    const NmsRecordDescriptor duplicate[] = {{1,0},{1,0}};
    CHECK(nms_bind_records(&r,duplicate,2)==NMS_TYPE && !r.records_bound);
    CHECK(nms_bind_records(&r,NULL,1)==NMS_STATE && !r.records_bound);
    CHECK(nms_bind_records(&r,definitions,4)==NMS_OK);
    CHECK(nms_bind_records(&r,definitions,4)==NMS_STATE);
    NmsHandle empty,a,b,text,array;
    CHECK(nms_record_create(&r,0,NULL,0,&empty)==NMS_OK);
    CHECK(nms_create(&r,bytes,3,&text)==NMS_OK);
    CHECK(nms_value_array_create(&r,&array)==NMS_OK);
    const uint64_t bits=UINT64_C(0x7ff0000000000042);
    NmsValue fields[]={{bits,3},{text,5},{array,7}};
    CHECK(nms_record_create(&r,1,fields,3,&a)==NMS_OK);
    NmsValue v={0,0};
    CHECK(nms_record_get(&r,a,0,&v)==NMS_OK && v.tag==3 && v.payload==bits);
    CHECK(nms_record_get(&r,a,1,&v)==NMS_OK && v.tag==5 && v.payload==text);
    CHECK(nms_release(&r,text)==NMS_OK); /* My GET owner outlives replacement. */
    CHECK(nms_record_set(&r,a,1,(NmsValue){1,5})==NMS_OK);
    NmsView view; CHECK(nms_view(&r,v.payload,&view)==NMS_OK && view.length==3 && view.data[1]==0);
    CHECK(nms_value_release(&r,v)==NMS_OK);
    CHECK(nms_record_set(&r,a,2,(NmsValue){array,7})==NMS_OK); /* Same child. */
    NmsValue field={empty,8};
    CHECK(nms_record_create(&r,2,&field,1,&b)==NMS_OK);
    uint32_t ordinal=99,global=99;
    CHECK(nms_record_identity(&r,b,&ordinal,&global)==NMS_OK && ordinal==2 && global==5);
    NmsHandle other;
    CHECK(nms_record_create(&r,3,&field,1,&other)==NMS_OK);
    CHECK(nms_record_identity(&r,other,&ordinal,&global)==NMS_OK && ordinal==3 && global==7);
    CHECK(nms_value_array_append(&r,array,(NmsValue){other,8})==NMS_OK);
    CHECK(nms_value_array_get(&r,array,0,&v)==NMS_OK && v.tag==8 && v.payload==other);
    CHECK(nms_value_release(&r,v)==NMS_OK);
    v=(NmsValue){123,1};
    CHECK(nms_record_get(&r,a,3,&v)==NMS_BOUNDS && v.payload==123 && v.tag==1);
    CHECK(nms_record_get(&r,array,0,&v)==NMS_TYPE && v.payload==123);
    CHECK(nms_value_array_get(&r,a,0,&v)==NMS_TYPE && v.payload==123);
    CHECK(nms_record_set(&r,a,0,(NmsValue){0,6})==NMS_TYPE);
    CHECK(nms_record_get(&r,a,0,&v)==NMS_OK && v.tag==3 && v.payload==bits);
    CHECK(nms_record_set(&r,a,UINT64_MAX,(NmsValue){1,1})==NMS_BOUNDS);
    NmsHandle untouched=123;
    CHECK(nms_record_create(&r,1,fields,2,&untouched)==NMS_TYPE && untouched==123);
    CHECK(nms_record_create(&r,4,NULL,0,&untouched)==NMS_TYPE && untouched==123);
    CHECK(nms_record_create(&r,0,NULL,0,NULL)==NMS_STATE);
    CHECK(nms_record_get(&r,a,0,NULL)==NMS_STATE);
    CHECK(nms_record_identity(&r,a,NULL,&global)==NMS_STATE);
    CHECK(nms_release(&r,empty)==NMS_OK && nms_release(&r,b)==NMS_OK);
    CHECK(nms_release(&r,other)==NMS_OK && nms_release(&r,array)==NMS_OK);
    CHECK(nms_release(&r,a)==NMS_OK && !r.live_objects && !r.live_bytes);
    CHECK(nms_dispose(&r)==NMS_OK && !r.record_descriptors);
    CHECK(nms_bind_records(&r,definitions,4)==NMS_DISPOSED);
    nms_init(&r,literals,1);
    CHECK(nms_value_array_create(&r,&array)==NMS_OK);
    CHECK(nms_bind_records(&r,definitions,4)==NMS_STATE && !r.records_bound);
    CHECK(nms_release(&r,array)==NMS_OK);
    CHECK(nms_bind_records(&r,definitions,4)==NMS_STATE); /* Allocated table persists. */
    CHECK(nms_dispose(&r)==NMS_OK);
    nms_init(&r,literals,1);
    CHECK(nms_bind_records(&r,NULL,0)==NMS_OK);
    CHECK(nms_record_create(&r,0,NULL,0,&untouched)==NMS_TYPE);
    CHECK(nms_dispose(&r)==NMS_OK);
    return 0;
}
int record_graphs(void) {
    for(unsigned prepared=0;prepared<2;prepared++) {
        NmsRuntime r; nms_init(&r,literals,1);
        CHECK(nms_bind_records(&r,definitions,4)==NMS_OK);
        if(prepared) CHECK(nms_prepare_collection(&r)==NMS_OK);
        NmsHandle keep; CHECK(nms_create(&r,bytes,3,&keep)==NMS_OK);
        for(unsigned iteration=0;iteration<1000;iteration++) {
            NmsHandle array,record;
            CHECK(nms_value_array_create(&r,&array)==NMS_OK);
            NmsValue fields[]={{array,7},{keep,5},{0,0}};
            CHECK(nms_record_create(&r,1,fields,3,&record)==NMS_OK);
            CHECK(nms_record_set(&r,record,2,(NmsValue){record,8})==NMS_OK);
            CHECK(nms_value_array_append(&r,array,(NmsValue){record,8})==NMS_OK);
            CHECK(nms_value_array_append(&r,array,(NmsValue){record,8})==NMS_OK);
            NmsValue returned;
            CHECK(nms_value_array_get(&r,array,0,&returned)==NMS_OK);
            CHECK(nms_release(&r,array)==NMS_OK && nms_release(&r,record)==NMS_OK);
#ifdef NMS_TESTING
            if(prepared)nms_test_fail_after(&r,0);
#endif
            CHECK((prepared?nms_collect_prepared(&r):nms_collect(&r))==NMS_OK && r.live_objects==3);
            CHECK(nms_value_release(&r,returned)==NMS_OK);
            CHECK((prepared?nms_collect_prepared(&r):nms_collect(&r))==NMS_OK && r.live_objects==1);
            CHECK(r.slots[(uint32_t)keep].references==1); /* Each dead-to-live edge once. */
#ifdef NMS_TESTING
            nms_test_fail_after(&r,UINT64_MAX);
#endif
        }
        CHECK(nms_release(&r,keep)==NMS_OK && !r.live_objects && !r.live_bytes);
        CHECK(nms_dispose(&r)==NMS_OK);
    }
    NmsRuntime r; nms_init(&r,literals,1);
    CHECK(nms_bind_records(&r,definitions,4)==NMS_OK);
    NmsHandle previous;CHECK(nms_record_create(&r,0,NULL,0,&previous)==NMS_OK);
    for(unsigned i=0;i<4000;i++) {
        NmsValue field={previous,8};NmsHandle next;
        CHECK(nms_record_create(&r,2,&field,1,&next)==NMS_OK);
        CHECK(nms_release(&r,previous)==NMS_OK);previous=next;
    }
    uint32_t ordinal,global;
    CHECK(nms_record_identity(&r,previous,&ordinal,&global)==NMS_OK && ordinal==2 && global==5);
    CHECK(nms_release(&r,previous)==NMS_OK && !r.live_objects && !r.live_bytes);
    CHECK(nms_dispose(&r)==NMS_OK);
    return 0;
}
#ifdef NMS_TESTING
int record_failures(void) {
    {
        NmsRuntime r; nms_init(&r,literals,1);
        CHECK(nms_bind_records(&r,definitions,4)==NMS_OK);
        NmsHandle child,record,output=123;
        CHECK(nms_record_create(&r,0,NULL,0,&child)==NMS_OK);
        NmsValue initial={1,1};
        CHECK(nms_record_create(&r,2,&initial,1,&record)==NMS_OK);
        NmsValue fields[]={{child,8},{child,8},{1,5}};
        /* I exercise checked counter saturation, then restore the test owner. */
        r.slots[(uint32_t)child].references=UINT64_MAX-1;
        CHECK(nms_record_create(&r,1,fields,3,&output)==NMS_MEMORY && output==123);
        CHECK(r.slots[(uint32_t)child].references==UINT64_MAX-1 && r.live_objects==2);
        r.slots[(uint32_t)child].references=UINT64_MAX;
        CHECK(nms_record_set(&r,record,0,(NmsValue){child,8})==NMS_MEMORY);
        NmsValue value={99,1};
        CHECK(nms_record_get(&r,record,0,&value)==NMS_OK && value.tag==1 && value.payload==1);
        r.slots[(uint32_t)child].references=1;
        CHECK(nms_record_set(&r,record,0,(NmsValue){child,8})==NMS_OK);
        r.slots[(uint32_t)child].references=UINT64_MAX;
        value=(NmsValue){99,1};
        CHECK(nms_record_get(&r,record,0,&value)==NMS_MEMORY && value.tag==1 && value.payload==99);
        r.slots[(uint32_t)child].references=2;
        CHECK(nms_release(&r,record)==NMS_OK && nms_release(&r,child)==NMS_OK);
        CHECK(!r.live_objects && !r.live_bytes && nms_dispose(&r)==NMS_OK);
    }
    for(uint64_t budget=0;budget<4;budget++) {
        NmsRuntime r; nms_init(&r,literals,1);
        CHECK(nms_bind_records(&r,definitions,4)==NMS_OK);
        CHECK(nms_prepare_collection(&r)==NMS_OK);
        NmsHandle owners[8];
        for(unsigned i=0;i<8;i++)CHECK(nms_record_create(&r,0,NULL,0,&owners[i])==NMS_OK);
        NmsSlot *table=r.slots;void *workspace=r.collection_workspace;
        uint64_t live=r.live_bytes;NmsHandle output=123;
        NmsValue fields[]={{owners[0],8},{owners[0],8},{1,5}};
        nms_test_fail_after(&r,budget);
        NmsStatus status=nms_record_create(&r,1,fields,3,&output);
        nms_test_fail_after(&r,UINT64_MAX);
        if(budget<3) {
            CHECK(status==NMS_MEMORY && output==123);
            CHECK(r.slots==table && r.collection_workspace==workspace && r.capacity==8);
            CHECK(r.live_objects==8 && r.live_bytes==live && r.slots[(uint32_t)owners[0]].references==1);
        } else {
            CHECK(status==NMS_OK && r.capacity==16);
            CHECK(r.slots[(uint32_t)owners[0]].references==3);
            CHECK(nms_release(&r,output)==NMS_OK);
            CHECK(r.live_objects==8 && r.live_bytes==live && r.slots[(uint32_t)owners[0]].references==1);
        }
        CHECK(nms_record_create(&r,1,fields,3,&output)==NMS_OK);
        CHECK(nms_release(&r,output)==NMS_OK);
        for(unsigned i=0;i<8;i++)CHECK(nms_release(&r,owners[i])==NMS_OK);
        CHECK(!r.live_objects && !r.live_bytes && nms_dispose(&r)==NMS_OK);
    }
    return 0;
}
#endif
#ifndef __wasm32__
int main(void) {
    int result=record_values();if(result)return result;
    result=record_graphs();if(result)return result;
#ifdef NMS_TESTING
    result=record_failures();if(result)return result;
#endif
    return 0;
}
#endif
