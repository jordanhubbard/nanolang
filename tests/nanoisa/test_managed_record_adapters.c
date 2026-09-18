/* I qualify private scalar ABI/root adapters; no bytecode admission follows. */
#ifdef NMS_RECORD_ADAPTER_LINKED
#include "../../src/nanoisa/managed_strings.h"
extern uint64_t nms_module_record_begin(const NmsView *,uint32_t,const NmsRecordDescriptor *,uint32_t);
extern uint64_t nms_module_graph_finish(int32_t);
extern uint32_t nms_module_graph_collect(void),nms_module_status(void),nms_module_dispose(void);
extern uint32_t nms_module_retain(uint64_t,uint32_t);
extern void nms_module_release(uint64_t,uint32_t),nms_module_fail(uint32_t);
extern uint64_t nms_module_record_literal(uint32_t,uint32_t,const uint64_t *,const uint32_t *);
extern uint32_t nms_module_record_get_value(uint64_t,uint32_t,uint32_t,uint32_t,uint64_t *,uint32_t *);
extern uint32_t nms_module_record_set_value(uint64_t,uint32_t,uint32_t,uint32_t,uint64_t,uint32_t);
extern uint64_t nms_module_concat(uint64_t,uint64_t),nms_module_length(uint64_t);
extern uint64_t nms_module_live_objects(void),nms_module_live_bytes(void);
#else
#include "../../src/nanoisa/managed_module.c"
#endif
#include "../../src/nanoisa/isa.h"
_Static_assert(NMS_RECORD_TAG==TAG_STRUCT,"I retain the ISA record tag");
#define CHECK(x) do {if(!(x))return __LINE__;}while(0)
#define ACQUIRED (UINT64_C(1)<<32)
static const unsigned char bytes[]={'a',0,'b'};
static const NmsView literals[]={{bytes,3}},other_literals[]={{bytes,3}};
static const NmsRecordDescriptor definitions[]={{0,0},{2,5},{5,256},{7,1},{9,3}};
static const NmsRecordDescriptor other_definitions[]={{0,0},{2,5},{5,256},{7,1},{9,3}};
static uint64_t begin(void){return nms_module_record_begin(literals,1,definitions,5);}
#ifndef NMS_RECORD_ADAPTER_LINKED
static void reset_module(void) {
    if(nms_module_ready)nms_dispose(&nms_module_instance);
    volatile unsigned char *bytes=(volatile unsigned char *)&nms_module_instance;
    for(size_t i=0;i<sizeof nms_module_instance;i++)bytes[i]=0;
    nms_module_ready=0;nms_module_error=NMS_OK;
}
#endif
int record_adapter_values(void) {
#ifndef NMS_RECORD_ADAPTER_LINKED
    reset_module();
#endif
    CHECK(begin()==ACQUIRED);
    uint64_t child=nms_module_record_literal(0,0,NULL,NULL);CHECK(child);
    uint64_t text=nms_module_concat(1,1);CHECK(text);
    uint64_t fields[5]={child,text,UINT64_C(0x7ff0000000000042),123,1};
    uint32_t tags[5]={8,5,3,1,4};
    uint64_t record=nms_module_record_literal(1,5,fields,tags);CHECK(record);
    nms_module_release(child,8);nms_module_release(text,5);
    CHECK(nms_module_retain(record,8)==NMS_OK);
    uint64_t alias=record,got=99;uint32_t tag=99;
    CHECK(nms_module_record_get_value(alias,8,2,0,&got,&tag)==NMS_OK && tag==3 && got==fields[2]);
    CHECK(nms_module_record_get_value(alias,8,1,1,&got,&tag)==NMS_OK && tag==5 && got==text);
    CHECK(nms_module_record_set_value(record,8,1,0,1,5)==NMS_OK);
    CHECK(nms_module_length(got)==6);nms_module_release(got,tag); /* GET outlives replaced edge. */
    uint64_t replacement=nms_module_record_literal(0,0,NULL,NULL);CHECK(replacement);
    CHECK(nms_module_record_set_value(record,8,0,1,replacement,8)==NMS_OK);
    CHECK(nms_module_record_set_value(record,8,0,0,replacement,8)==NMS_OK); /* Same child. */
    nms_module_release(replacement,8);
    CHECK(nms_module_record_get_value(alias,8,0,0,&got,&tag)==NMS_OK && tag==8 && got==replacement);
    nms_module_release(record,8);nms_module_release(alias,8);
    CHECK(nms_module_live_objects()==1); /* Retained GET is the last root. */
    nms_module_release(got,tag);CHECK(!nms_module_live_objects() && !nms_module_live_bytes());
    uint64_t payloads[256];uint32_t kinds[256];
    for(unsigned i=0;i<256;i++){payloads[i]=i;kinds[i]=1;}
    record=nms_module_record_literal(2,256,payloads,kinds);CHECK(record);
    CHECK(nms_module_record_get_value(record,8,255,0,&got,&tag)==NMS_OK && got==255 && tag==1);
    nms_module_release(record,8);
    CHECK(nms_module_graph_finish(17)==17);
    CHECK(nms_module_record_begin(literals,1,other_definitions,5)==NMS_STATE);
    CHECK(nms_module_record_begin(other_literals,1,definitions,5)==NMS_STATE);
    CHECK(nms_module_record_begin(literals,1,definitions,4)==NMS_STATE);
    CHECK(nms_module_record_begin(literals,0,definitions,5)==NMS_STATE);
    /* Persist a record root across successful and failed entries. */
    CHECK(begin()==ACQUIRED);record=nms_module_record_literal(0,0,NULL,NULL);CHECK(record);
    CHECK(nms_module_graph_finish(0)==0);
    for(unsigned i=0;i<10;i++) {
        CHECK(begin()==ACQUIRED);nms_module_fail(NMS_ASSERT);
        CHECK(nms_module_record_begin(NULL,0,NULL,0)==NMS_BUSY && nms_module_status()==NMS_ASSERT);
        CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_ASSERT<<32));
        CHECK(nms_module_live_objects()==1);
    }
    CHECK(begin()==ACQUIRED);nms_module_release(record,8);CHECK(nms_module_graph_finish(0)==0);
    for(unsigned aggregate=0;aggregate<2;aggregate++) {
        CHECK(begin()==ACQUIRED);got=123;tag=4;
        CHECK(nms_module_record_get_value(0,1,0,aggregate,&got,&tag)==(aggregate?NMS_BOUNDS:NMS_TYPE));
        CHECK(got==123 && tag==4);
        CHECK(nms_module_record_set_value(0,1,0,aggregate,9,1)==(aggregate?NMS_BOUNDS:NMS_TYPE));
        CHECK(nms_module_graph_finish(0)==((uint64_t)(aggregate?NMS_BOUNDS:NMS_TYPE)<<32));
        CHECK(begin()==ACQUIRED);record=nms_module_record_literal(0,0,NULL,NULL);CHECK(record);
        CHECK(nms_module_record_get_value(record,8,0,aggregate,&got,&tag)==NMS_BOUNDS);
        CHECK(got==123 && tag==4);
        CHECK(nms_module_record_set_value(record,8,0,aggregate,9,1)==NMS_BOUNDS);
        nms_module_release(record,8);CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_BOUNDS<<32));
    }
    CHECK(begin()==ACQUIRED);CHECK(!nms_module_record_literal(2,257,payloads,kinds));
    CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_TYPE<<32));
    CHECK(begin()==ACQUIRED);CHECK(!nms_module_record_literal(3,1,NULL,NULL));
    CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_STATE<<32));
    CHECK(begin()==ACQUIRED);CHECK(!nms_module_record_literal(1,4,payloads,kinds));
    CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_TYPE<<32));
    CHECK(begin()==ACQUIRED);got=77;tag=4;
    CHECK(nms_module_record_get_value(0,8,0,2,&got,&tag)==NMS_STATE && got==77 && tag==4);
    CHECK(nms_module_record_get_value(0,8,0,0,NULL,&tag)==NMS_STATE && tag==4);
    CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_STATE<<32));
    CHECK(!nms_module_live_objects() && !nms_module_live_bytes());
    CHECK(nms_module_dispose()==NMS_OK);
    CHECK(begin()==NMS_DISPOSED);
    return 0;
}
#ifndef NMS_RECORD_ADAPTER_LINKED
int record_adapter_lifecycle(void) {
    reset_module();CHECK(nms_module_dispose()==NMS_OK);CHECK(begin()==NMS_DISPOSED);
    reset_module();CHECK(nms_module_record_begin(literals,1,NULL,1)==NMS_STATE);
    CHECK(!nms_module_instance.active && !nms_module_instance.records_bound);
    CHECK(begin()==ACQUIRED);CHECK(nms_module_graph_finish(0)==0);CHECK(nms_module_dispose()==NMS_OK);
    return 0;
}
#ifdef NMS_TESTING
int record_adapter_failures(void) {
    reset_module();nms_init(&nms_module_instance,literals,1);nms_module_ready=1;
    nms_test_fail_after(&nms_module_instance,0);
    CHECK(begin()==(ACQUIRED|NMS_MEMORY));
    CHECK(nms_module_instance.active && nms_module_status()==NMS_MEMORY);
    CHECK(begin()==NMS_BUSY && nms_module_status()==NMS_MEMORY);
    CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_MEMORY<<32));
    CHECK(!nms_module_instance.active);
    nms_test_fail_after(&nms_module_instance,UINT64_MAX);
    CHECK(begin()==ACQUIRED);CHECK(nms_module_graph_finish(0)==0);CHECK(nms_module_dispose()==NMS_OK);
    for(uint64_t budget=0;budget<4;budget++) {
        reset_module();CHECK(begin()==ACQUIRED);
        uint64_t owners[8];for(unsigned i=0;i<8;i++){owners[i]=nms_module_record_literal(0,0,NULL,NULL);CHECK(owners[i]);}
        NmsSlot *slots=nms_module_instance.slots;void *workspace=nms_module_instance.collection_workspace;
        uint64_t fields[3]={owners[0],owners[0],1};uint32_t tags[3]={8,8,5};
        uint64_t before=nms_module_live_bytes();
        nms_test_fail_after(&nms_module_instance,budget);
        uint64_t result=nms_module_record_literal(4,3,fields,tags);
        nms_test_fail_after(&nms_module_instance,UINT64_MAX);
        if(budget<3) {
            CHECK(!result && nms_module_status()==NMS_MEMORY);
            CHECK(nms_module_instance.slots==slots && nms_module_instance.collection_workspace==workspace);
            CHECK(nms_module_live_objects()==8 && nms_module_live_bytes()==before);
            CHECK(nms_module_instance.slots[(uint32_t)owners[0]].references==1);
        } else {CHECK(result);nms_module_release(result,8);}
        for(unsigned i=0;i<8;i++)nms_module_release(owners[i],8);
        CHECK(nms_module_graph_finish(0)==(budget<3?((uint64_t)NMS_MEMORY<<32):0));
        CHECK(!nms_module_live_objects() && !nms_module_live_bytes());
        CHECK(begin()==ACQUIRED);CHECK(nms_module_graph_finish(0)==0);CHECK(nms_module_dispose()==NMS_OK);
    }
    reset_module();CHECK(begin()==ACQUIRED);
    uint64_t child=nms_module_record_literal(0,0,NULL,NULL);CHECK(child);
    uint64_t fields[3]={child,child,1};uint32_t tags[3]={8,8,5};
    nms_module_instance.slots[(uint32_t)child].references=UINT64_MAX-1;
    CHECK(!nms_module_record_literal(4,3,fields,tags));
    CHECK(nms_module_instance.slots[(uint32_t)child].references==UINT64_MAX-1 && nms_module_live_objects()==1);
    nms_module_instance.slots[(uint32_t)child].references=1;nms_module_release(child,8);
    CHECK(nms_module_graph_finish(0)==((uint64_t)NMS_MEMORY<<32));
    CHECK(nms_module_dispose()==NMS_OK && !live_allocations);
    return 0;
}
#endif
#endif
#ifndef __wasm32__
int main(void) {
    int result=record_adapter_values();if(result)return result;
#ifndef NMS_RECORD_ADAPTER_LINKED
    result=record_adapter_lifecycle();if(result)return result;
#ifdef NMS_TESTING
    result=record_adapter_failures();if(result)return result;
#endif
#endif
    return 0;
}
#endif
