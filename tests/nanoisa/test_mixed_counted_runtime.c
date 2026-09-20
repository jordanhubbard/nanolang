/* I exercise counted storage/adapters, not bytecode or public admission. */
#include "mixed_counted_catalog.h"
#include "mixed_counted_wasm_memory.h"
#ifdef NMS_TEST_ALLOC_HOOKS
#include "mixed_counted_tracker.h"
#endif
#ifndef MC_LINKED
#include "../../src/nanoisa/managed_module.c"
#else
extern uint64_t nms_module_record_begin(const NmsView*,uint32_t,const NmsRecordDescriptor*,uint32_t);
extern uint64_t nms_module_graph_finish(int32_t);
extern uint32_t nms_module_status(void),nms_module_dispose(void),nms_module_graph_collect(void);
extern void nms_module_fail(uint32_t),nms_module_release(uint64_t,uint32_t);
extern uint32_t nms_module_retain(uint64_t,uint32_t);
extern uint64_t nms_module_record_literal(uint32_t,uint32_t,const uint64_t*,const uint32_t*);
extern uint32_t nms_module_record_get_value(uint64_t,uint32_t,uint32_t,uint32_t,uint64_t*,uint32_t*);
extern uint32_t nms_module_record_set_value(uint64_t,uint32_t,uint32_t,uint32_t,uint64_t,uint32_t);
extern uint64_t nms_module_array_create(uint32_t),nms_module_array_slice(uint64_t,uint64_t,uint32_t,uint64_t,uint32_t);
extern uint32_t nms_module_array_append_value(uint64_t,uint64_t,uint32_t),nms_module_array_set_value(uint64_t,uint64_t,uint64_t,uint32_t);
extern uint32_t nms_module_array_get_value(uint64_t,uint64_t,uint64_t*,uint32_t*),nms_module_array_pop_value(uint64_t,uint64_t*,uint32_t*);
extern uint32_t nms_module_array_value_length(uint64_t);
extern uint64_t nms_module_live_objects(void),nms_module_live_bytes(void),nms_module_concat(uint64_t,uint64_t);
#endif
#ifndef __wasm32__
#include <stdio.h>
#endif
static uint64_t mc_checks;
static unsigned mc_failure_line;
#ifdef NMS_TEST_ALLOC_HOOKS
static uint64_t mc_fault_cases,mc_fault_peak;
#endif
#define REQUIRE(x) do{mc_checks++;if(!(x)){if(!mc_failure_line)mc_failure_line=__LINE__;return __LINE__;}}while(0)
#define ACQUIRED (UINT64_C(1)<<32)
static const uint64_t mc_float_bits[]={0,UINT64_C(0x8000000000000000),UINT64_C(0x3ff8000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000042),UINT64_C(0x7ff0000000000042)};
static NmsValue mc_scalar(uint32_t tag,unsigned i,NmsHandle text){
    uint64_t bits=0;
    if(tag==1)bits=i%2?UINT64_C(0x8000000000000000):UINT64_C(0x7fffffffffffffff);
    if(tag==2)bits=i%2?255:0;
    if(tag==3)bits=mc_float_bits[i%7];
    if(tag==4)bits=i%2;
    if(tag==5)bits=text;
    return (NmsValue){bits,tag};
}
static int mc_equal(NmsValue a,NmsValue b){return a.tag==b.tag&&a.payload==b.payload;}
static int mc_text(NmsRuntime *r,NmsHandle text){
    NmsView v;REQUIRE(nms_view(r,text,&v)==NMS_OK&&v.length==4);
    for(unsigned i=0;i<4;i++)REQUIRE(v.data[i]==mc_literal_bytes[i]);
    return 0;
}
static NmsStatus mc_relay(NmsRuntime *r,NmsHandle borrowed,NmsHandle *out){
    NmsStatus s=nms_retain(r,borrowed);if(s==NMS_OK)*out=borrowed;return s;
}
int mixed_counted_core(void){
    for(uint32_t tag=1;tag<=5;tag++){
        NmsRuntime r;nms_init(&r,mc_literals,1);REQUIRE(nms_bind_records(&r,mc_records,3)==NMS_OK);
        REQUIRE(nms_begin(&r)==NMS_OK&&nms_prepare_collection(&r)==NMS_OK);
        NmsHandle text=0,other=0,a=0,b=0,record=0,equal=0,nested=0,copy=0;
        REQUIRE(nms_create(&r,mc_literal_bytes,4,&text)==NMS_OK);
        static const unsigned char alternate[]={0x7a,0,0x71,0xff};
        REQUIRE(nms_create(&r,alternate,4,&other)==NMS_OK);
        REQUIRE(nms_vm_array_create(&r,tag,&a)==NMS_OK&&nms_vm_array_create(&r,tag,&b)==NMS_OK);
        for(unsigned i=0;i<17;i++)REQUIRE(nms_value_array_append(&r,a,mc_scalar(tag,i,text))==NMS_OK);
        REQUIRE(nms_value_array_append(&r,b,mc_scalar(tag,1,other))==NMS_OK);
        NmsValue field={a,7};REQUIRE(nms_record_create(&r,0,&field,1,&record)==NMS_OK);
        REQUIRE(nms_record_create(&r,1,&field,1,&equal)==NMS_OK);
        uint32_t ordinal=99,layout=99;REQUIRE(nms_record_identity(&r,record,&ordinal,&layout)==NMS_OK&&ordinal==0&&layout==1);
        REQUIRE(nms_record_identity(&r,equal,&ordinal,&layout)==NMS_OK&&ordinal==1&&layout==2);
        NmsValue held={99,99};REQUIRE(nms_record_get(&r,record,0,&held)==NMS_OK&&held.tag==7&&held.payload==a);
        REQUIRE(r.slots[(uint32_t)a].references==4); /* Local, two fields, GET. */
        REQUIRE(nms_value_array_set(&r,held.payload,0,mc_scalar(tag,1,other))==NMS_OK);
        NmsValue got;REQUIRE(nms_value_array_get(&r,a,0,&got)==NMS_OK&&mc_equal(got,mc_scalar(tag,1,other)));
        REQUIRE(nms_value_release(&r,got)==NMS_OK);
        field=(NmsValue){record,8};REQUIRE(nms_record_create(&r,2,&field,1,&nested)==NMS_OK);
        NmsHandle returned=0;REQUIRE(mc_relay(&r,nested,&returned)==NMS_OK&&returned==nested);
        REQUIRE(nms_release(&r,nested)==NMS_OK);nested=0;
        REQUIRE(nms_record_set(&r,record,0,(NmsValue){b,7})==NMS_OK);
        REQUIRE(nms_record_set(&r,record,0,(NmsValue){b,7})==NMS_OK);
        REQUIRE(r.slots[(uint32_t)a].references==3&&r.slots[(uint32_t)b].references==2);
        for(unsigned i=17;i<65;i++)REQUIRE(nms_value_array_append(&r,a,mc_scalar(tag,i,text))==NMS_OK);
        for(unsigned i=1;i<33;i++)REQUIRE(nms_value_array_append(&r,b,mc_scalar(tag,i,other))==NMS_OK);
        uint32_t len;REQUIRE(nms_value_array_length(&r,a,&len)==NMS_OK&&len==65);
        REQUIRE(nms_value_array_length(&r,b,&len)==NMS_OK&&len==33);
        REQUIRE(nms_vm_array_slice(&r,a,0,UINT32_MAX,&copy)==NMS_OK&&copy!=a&&copy!=b);
        for(unsigned i=0;i<65;i++){
            REQUIRE(nms_value_array_get(&r,copy,i,&got)==NMS_OK);
            REQUIRE(mc_equal(got,mc_scalar(tag,i==0?1:i,i==0?other:text)));
            REQUIRE(nms_value_release(&r,got)==NMS_OK);
        }
        REQUIRE(nms_value_array_set(&r,copy,1,mc_scalar(tag,0,other))==NMS_OK);
        REQUIRE(nms_value_array_get(&r,a,1,&got)==NMS_OK&&mc_equal(got,mc_scalar(tag,1,text)));
        REQUIRE(nms_value_release(&r,got)==NMS_OK);
        REQUIRE(nms_value_array_append(&r,copy,mc_scalar(tag,0,text))==NMS_OK);
        REQUIRE(nms_value_array_length(&r,a,&len)==NMS_OK&&len==65);
        REQUIRE(nms_value_array_get(&r,a,UINT64_MAX,&got)==NMS_OK&&got.tag==0);
        REQUIRE(nms_value_array_set(&r,a,UINT64_MAX,mc_scalar(tag,0,text))==NMS_STATE);
        got=(NmsValue){99,99};REQUIRE(nms_record_get(&r,record,1,&got)==NMS_BOUNDS&&got.payload==99&&got.tag==99);
        REQUIRE(nms_record_get(&r,record,0,NULL)==NMS_STATE);
        REQUIRE(nms_record_set(&r,record,1,(NmsValue){a,7})==NMS_BOUNDS);
        REQUIRE(nms_release(&r,text)==NMS_OK&&nms_release(&r,other)==NMS_OK); /* STRING roots now belong to arrays. */
        if(tag==5)REQUIRE(mc_text(&r,text)==0);
        REQUIRE(nms_release(&r,record)==NMS_OK&&nms_release(&r,returned)==NMS_OK&&nms_release(&r,equal)==NMS_OK);
        REQUIRE(nms_release(&r,a)==NMS_OK&&nms_release(&r,b)==NMS_OK&&nms_release(&r,copy)==NMS_OK);
        REQUIRE(nms_value_array_length(&r,held.payload,&len)==NMS_OK&&len==65);
#ifdef NMS_TEST_ALLOC_HOOKS
        uint64_t requests=mc_attempts;mc_fail_at=requests+1;mc_persistent=1;
#endif
        REQUIRE(nms_collect_prepared(&r)==NMS_OK);
#ifdef NMS_TEST_ALLOC_HOOKS
        REQUIRE(mc_attempts==requests);mc_fail_at=0;mc_persistent=0;
#endif
        REQUIRE(nms_value_release(&r,held)==NMS_OK);
        REQUIRE(!r.live_objects&&!r.live_bytes);
        REQUIRE(nms_finish(&r,NMS_OK,17)==17&&nms_dispose(&r)==NMS_OK);
    }
    return 0;
}
/* I keep adapter-width/repeated-edge boundaries separate from the source catalog. */
static int mc_core_boundaries(void){
    static const NmsRecordDescriptor bounds[]={{0,2},{1,256},{2,0}};
    NmsRuntime r;nms_init(&r,mc_literals,1);REQUIRE(nms_bind_records(&r,bounds,3)==NMS_OK);
    REQUIRE(nms_begin(&r)==NMS_OK&&nms_prepare_collection(&r)==NMS_OK);
    NmsHandle a=0,pair=0,empty=0,wide=0;REQUIRE(nms_vm_array_create(&r,1,&a)==NMS_OK);
    NmsValue fields[256];for(unsigned i=0;i<256;i++)fields[i]=(NmsValue){i,1};
    REQUIRE(nms_record_create(&r,1,fields,256,&wide)==NMS_OK);
    NmsValue last;REQUIRE(nms_record_get(&r,wide,255,&last)==NMS_OK&&last.tag==1&&last.payload==255);
    REQUIRE(nms_record_create(&r,2,NULL,0,&empty)==NMS_OK);
    NmsValue repeated[2]={{a,7},{a,7}};
    REQUIRE(nms_record_create(&r,0,repeated,2,&pair)==NMS_OK&&r.slots[(uint32_t)a].references==3);
    REQUIRE(nms_collect_prepared(&r)==NMS_OK&&r.slots[(uint32_t)a].references==3);
#ifdef NMS_TEST_ALLOC_HOOKS
    uint64_t old=r.slots[(uint32_t)a].references,objects=r.live_objects,bytes=r.live_bytes;
    r.slots[(uint32_t)a].references=UINT64_MAX-1;NmsHandle sentinel=123;
    REQUIRE(nms_record_create(&r,0,repeated,2,&sentinel)==NMS_MEMORY&&sentinel==123);
    REQUIRE(r.slots[(uint32_t)a].references==UINT64_MAX-1&&r.live_objects==objects&&r.live_bytes==bytes);
    r.slots[(uint32_t)a].references=old;
#endif
    REQUIRE(nms_release(&r,pair)==NMS_OK&&nms_release(&r,wide)==NMS_OK&&nms_release(&r,empty)==NMS_OK&&nms_release(&r,a)==NMS_OK);
    REQUIRE(!r.live_objects&&!r.live_bytes&&nms_finish(&r,NMS_OK,0)==0&&nms_dispose(&r)==NMS_OK);
    return 0;
}
/* My module lifetime is one target instance; its global persists across entries. */
int mixed_counted_module(void){
    NmsHandle global=0;
    uint64_t warm_pages=0;
    for(unsigned iteration=0;iteration<1024;iteration++){
        REQUIRE(nms_module_record_begin(mc_literals,1,mc_records,3)==ACQUIRED);
        if(global){
            uint64_t bits=0;uint32_t kind=0;
            REQUIRE(nms_module_record_get_value(global,8,0,0,&bits,&kind)==NMS_OK&&kind==7);
            REQUIRE(nms_module_array_value_length(bits)==65);nms_module_release(bits,kind);
        }
        uint32_t element=iteration%5+1;
        uint64_t text=element==5?nms_module_concat(1,1):1,other=element==5?nms_module_concat(1,1):1;
        REQUIRE(text&&other);if(element==5)REQUIRE(text!=other);
        uint64_t a=nms_module_array_create(element),b=nms_module_array_create(element);REQUIRE(a&&b&&a!=b);
        for(unsigned i=0;i<65;i++){NmsValue v=mc_scalar(element,i,text);REQUIRE(nms_module_array_append_value(a,v.payload,v.tag)==NMS_OK);}
        NmsValue first=mc_scalar(element,0,text),replacement=mc_scalar(element,1,other);
        REQUIRE(nms_module_array_append_value(b,first.payload,first.tag)==NMS_OK);
        uint32_t array_tag=7;uint64_t record=nms_module_record_literal(0,1,&a,&array_tag);REQUIRE(record);
        uint64_t held=0;uint32_t tag=0;
        REQUIRE(nms_module_record_get_value(record,8,0,1,&held,&tag)==NMS_OK&&held==a&&tag==7);
        REQUIRE(nms_module_record_set_value(record,8,0,0,b,7)==NMS_OK);
        REQUIRE(nms_module_array_set_value(held,0,replacement.payload,replacement.tag)==NMS_OK);
        uint64_t bits=0;uint32_t kind=0;
        REQUIRE(nms_module_array_get_value(b,0,&bits,&kind)==NMS_OK&&bits==first.payload&&kind==first.tag);nms_module_release(bits,kind);
        uint64_t copy=nms_module_array_slice(a,0,1,65,1);REQUIRE(copy&&copy!=a);
        REQUIRE(nms_module_array_set_value(copy,0,first.payload,first.tag)==NMS_OK);
        REQUIRE(nms_module_array_get_value(a,0,&bits,&kind)==NMS_OK&&bits==replacement.payload&&kind==replacement.tag);nms_module_release(bits,kind);
        NmsValue last=mc_scalar(element,64,text);
        REQUIRE(nms_module_array_pop_value(copy,&bits,&kind)==NMS_OK&&bits==last.payload&&kind==last.tag);nms_module_release(bits,kind);
        nms_module_release(copy,7);nms_module_release(record,8);nms_module_release(b,7);
        REQUIRE(nms_module_array_value_length(held)==65);
        uint64_t next=nms_module_record_literal(1,1,&a,&array_tag);REQUIRE(next);
        if(global)nms_module_release(global,8);
        global=next;nms_module_release(a,7);nms_module_release(held,7);
        if(element==5){nms_module_release(text,5);nms_module_release(other,5);}
        if(iteration%3==1){
            REQUIRE(nms_module_array_set_value(a,UINT64_C(0x8000000000000000),0,1)==NMS_BOUNDS);
            REQUIRE(nms_module_array_get_value(a,0,&bits,&kind)==NMS_OK&&bits==replacement.payload&&kind==replacement.tag);nms_module_release(bits,kind);
            nms_module_fail(NMS_ASSERT);
            REQUIRE(nms_module_record_begin(NULL,UINT32_MAX,NULL,UINT32_MAX)==NMS_BUSY&&nms_module_status()==NMS_BOUNDS);
            REQUIRE(nms_module_graph_finish(19)==((uint64_t)NMS_BOUNDS<<32));
        }else{
            if(iteration%3==2)nms_module_fail(NMS_ASSERT);
            REQUIRE(nms_module_graph_finish(19)==(iteration%3==2?((uint64_t)NMS_ASSERT<<32):19));
        }
#ifdef __wasm32__
        uint64_t pages=__builtin_wasm_memory_size(0);
        REQUIRE(pages<=64);if(iteration==63)warm_pages=pages;if(iteration>63)REQUIRE(pages==warm_pages);
#else
        (void)warm_pages;
#endif
    }
    REQUIRE(nms_module_record_begin(mc_literals,1,mc_records,3)==ACQUIRED);
    nms_module_release(global,8);REQUIRE(nms_module_graph_collect()==NMS_OK);
    REQUIRE(!nms_module_live_objects()&&!nms_module_live_bytes());
    REQUIRE(nms_module_graph_finish(0)==0&&nms_module_dispose()==NMS_OK);
    REQUIRE(nms_module_record_begin(mc_literals,1,mc_records,3)==NMS_DISPOSED);
    return 0;
}
#ifdef NMS_TEST_ALLOC_HOOKS
/* I measure each allocating transaction after constructing stable input roots. */
static int mc_fault_transaction(unsigned operation,uint64_t failure,unsigned persistent,uint64_t *sites){
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error);mc_tracker_reset();
    NmsRuntime r;nms_init(&r,mc_literals,1);REQUIRE(nms_bind_records(&r,mc_records,3)==NMS_OK);
    REQUIRE(nms_begin(&r)==NMS_OK);
    NmsHandle a=0,record=0,result=123,text=0,extra[5]={0};
    REQUIRE(nms_create(&r,mc_literal_bytes,4,&text)==NMS_OK);
    REQUIRE(nms_vm_array_create(&r,5,&a)==NMS_OK);
    for(unsigned i=0;i<8;i++)REQUIRE(nms_value_array_append(&r,a,(NmsValue){text,5})==NMS_OK);
    REQUIRE(nms_record_create(&r,0,&(NmsValue){a,7},1,&record)==NMS_OK);
    if(operation==5){
        REQUIRE(nms_prepare_collection(&r)==NMS_OK);
        for(unsigned i=0;i<5;i++)REQUIRE(nms_record_create(&r,0,&(NmsValue){a,7},1,&extra[i])==NMS_OK);
        REQUIRE(r.live_objects==8&&r.capacity==8);
    }
    uint64_t objects=r.live_objects,bytes=r.live_bytes,refs=r.slots[(uint32_t)a].references;
    uint64_t text_refs=r.slots[(uint32_t)text].references,start=mc_attempts;
    mc_fail_at=failure?start+failure:0;mc_persistent=persistent;
    NmsStatus status;
    if(operation==0)status=nms_vm_array_create(&r,5,&result);
    else if(operation==1)status=nms_value_array_append(&r,a,(NmsValue){text,5});
    else if(operation==2)status=nms_vm_array_slice(&r,a,0,8,&result);
    else if(operation==3)status=nms_record_create(&r,0,&(NmsValue){a,7},1,&result);
    else if(operation==4)status=nms_prepare_collection(&r);
    else if(operation==5)status=nms_record_create(&r,0,&(NmsValue){a,7},1,&result);
    else {uint64_t bits[2]={text,text};uint32_t tags[2]={5,5};status=nms_vm_array_literal(&r,5,bits,tags,2,&result);}
    *sites=mc_attempts-start;
    if(failure){
        REQUIRE(status==NMS_MEMORY&&mc_failures>=1);
        REQUIRE(result==123&&r.live_objects==objects&&r.live_bytes==bytes);
        REQUIRE(r.slots[(uint32_t)a].references==refs&&r.slots[(uint32_t)text].references==text_refs);
        uint32_t length=0;REQUIRE(nms_value_array_length(&r,a,&length)==NMS_OK&&length==8);
        NmsValue value;REQUIRE(nms_record_get(&r,record,0,&value)==NMS_OK&&value.payload==a&&value.tag==7);
        REQUIRE(nms_value_release(&r,value)==NMS_OK);
        REQUIRE(mc_text(&r,text)==0);
    }else{
        REQUIRE(status==NMS_OK);if(operation==0||operation==2||operation==3||operation>=5)REQUIRE(nms_release(&r,result)==NMS_OK);
    }
    /* Cleanup remains under persistent exhaustion; releasing DAGs allocates nothing. */
    for(unsigned i=0;i<5;i++)if(extra[i])REQUIRE(nms_release(&r,extra[i])==NMS_OK);
    REQUIRE(nms_release(&r,record)==NMS_OK&&nms_release(&r,a)==NMS_OK&&nms_release(&r,text)==NMS_OK);
    REQUIRE(!r.live_objects&&!r.live_bytes);
    REQUIRE(nms_finish(&r,status,0)==((uint64_t)status<<32)&&nms_dispose(&r)==NMS_OK);
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error&&mc_successes==mc_frees);
    mc_fault_cases++;if(mc_peak>mc_fault_peak)mc_fault_peak=mc_peak;
#ifndef __wasm32__
    printf("I measured operation %u position %llu mode %u: requests %llu successes %llu frees %llu peak %llu status %u.\n",operation,(unsigned long long)failure,persistent,(unsigned long long)*sites,(unsigned long long)mc_successes,(unsigned long long)mc_frees,(unsigned long long)mc_peak,(unsigned)status);
#endif
    mc_fail_at=0;mc_persistent=0;return 0;
}
static int mc_tracker_controls(void){
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error);mc_tracker_reset();
    unsigned char token[2];
    REQUIRE(nms_test_allocation_permitted(0));nms_test_allocation_created(&token[0],0);
    REQUIRE(mc_attempts==1&&mc_successes==1&&mc_live==1&&!mc_bytes);
    nms_test_allocation_destroyed(&token[0]);REQUIRE(mc_frees==1&&!mc_live&&!mc_tracker_error);
    mc_tracker_reset();REQUIRE(nms_test_allocation_permitted(17));nms_test_allocation_created(&token[0],17);
    REQUIRE(mc_bytes==17&&mc_peak==17);nms_test_allocation_created(&token[0],17);REQUIRE(mc_tracker_error==6&&mc_live==1&&mc_bytes==17);
    nms_test_allocation_destroyed(&token[0]);mc_tracker_reset();
    nms_test_allocation_destroyed(&token[1]);REQUIRE(mc_tracker_error==9);mc_tracker_reset();
    mc_bytes=UINT64_MAX;REQUIRE(!nms_test_allocation_permitted(1)&&mc_tracker_error==4);mc_bytes=0;mc_tracker_reset();
    mc_live=MC_TRACKED;REQUIRE(!nms_test_allocation_permitted(0)&&mc_tracker_error==4);mc_live=0;mc_tracker_reset();
    mc_fail_at=1;REQUIRE(!nms_test_allocation_permitted(1));REQUIRE(nms_test_allocation_permitted(1));REQUIRE(mc_failures==1&&mc_attempts==2);mc_tracker_reset();
    mc_fail_at=1;mc_persistent=1;REQUIRE(!nms_test_allocation_permitted(1)&&!nms_test_allocation_permitted(1));REQUIRE(mc_failures==2);mc_tracker_reset();
    return 0;
}
int mixed_counted_faults(void){
    REQUIRE(mc_tracker_controls()==0);
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error);
    for(unsigned operation=0;operation<7;operation++){
        uint64_t sites=0;REQUIRE(mc_fault_transaction(operation,0,0,&sites)==0&&sites>0);
        for(unsigned mode=0;mode<2;mode++)for(uint64_t failure=1;failure<=sites;failure++){
            uint64_t reached=0;REQUIRE(mc_fault_transaction(operation,failure,mode,&reached)==0&&reached>=failure);
            uint64_t recovered=0;REQUIRE(mc_fault_transaction(operation,0,0,&recovered)==0&&recovered==sites);
        }
    }
    mc_tracker_reset();NmsRuntime r;nms_init(&r,mc_literals,1);REQUIRE(nms_bind_records(&r,mc_records,3)==NMS_OK);
    NmsHandle a=0,b=0,record=0;REQUIRE(nms_vm_array_create(&r,1,&a)==NMS_OK&&nms_vm_array_create(&r,1,&b)==NMS_OK);
    REQUIRE(nms_record_create(&r,0,&(NmsValue){a,7},1,&record)==NMS_OK);
    r.slots[(uint32_t)b].references=UINT64_MAX;
    REQUIRE(nms_record_set(&r,record,0,(NmsValue){b,7})==NMS_MEMORY);
    REQUIRE(r.slots[(uint32_t)a].references==2&&r.slots[(uint32_t)b].references==UINT64_MAX);
    r.slots[(uint32_t)b].references=1;NmsValue held;
    REQUIRE(nms_record_get(&r,record,0,&held)==NMS_OK&&held.payload==a);
    REQUIRE(nms_value_release(&r,held)==NMS_OK&&nms_release(&r,record)==NMS_OK&&nms_release(&r,a)==NMS_OK&&nms_release(&r,b)==NMS_OK);
    REQUIRE(nms_dispose(&r)==NMS_OK&&!mc_live&&!mc_bytes&&!mc_tracker_error);
    return 0;
}
#endif
uint64_t mixed_counted_checks(void){return mc_checks;}
uint64_t mixed_counted_peak(void){
#ifdef NMS_TEST_ALLOC_HOOKS
    return mc_fault_peak;
#else
    return 0;
#endif
}
uint64_t mixed_counted_fault_cases(void){
#ifdef NMS_TEST_ALLOC_HOOKS
    return mc_fault_cases;
#else
    return 0;
#endif
}
int mixed_counted_all(void){
    int result=mixed_counted_core();if(result)return result;
    result=mc_core_boundaries();if(result)return result;
#ifdef NMS_TEST_ALLOC_HOOKS
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error);
    result=mixed_counted_faults();if(result)return result;
    /* I observe failed preparation after acquisition, then recover before reset. */
    mc_tracker_reset();mc_fail_at=1;
    REQUIRE(nms_module_record_begin(mc_literals,1,mc_records,3)==(ACQUIRED|NMS_MEMORY));
    REQUIRE(nms_module_instance.active&&nms_module_status()==NMS_MEMORY);
    REQUIRE(nms_module_record_begin(NULL,UINT32_MAX,NULL,UINT32_MAX)==NMS_BUSY&&nms_module_status()==NMS_MEMORY);
    REQUIRE(nms_module_graph_finish(123)==((uint64_t)NMS_MEMORY<<32)&&!nms_module_instance.active);
    mc_fail_at=0;REQUIRE(nms_module_record_begin(mc_literals,1,mc_records,3)==ACQUIRED);
    REQUIRE(nms_module_graph_finish(0)==0&&nms_module_dispose()==NMS_OK);
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error);
    nms_module_ready=0;nms_module_error=NMS_OK;
#endif
    result=mixed_counted_module();if(result)return result;
#ifdef NMS_TEST_ALLOC_HOOKS
    REQUIRE(!mc_live&&!mc_bytes&&!mc_tracker_error&&mc_successes==mc_frees);
#endif
    return 0;
}
#ifndef __wasm32__
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);puts("I begin counted mixed core, boundaries, faults and module entries.");
    int result=mixed_counted_all();
    printf("I checked %llu counted mixed assertions; status %d; no admission.\n",(unsigned long long)mc_checks,result);
    if(result)printf("I first failed fixture line %u.\n",mc_failure_line);
#ifdef NMS_TEST_ALLOC_HOOKS
    printf("I measured %llu requests, %llu backend successes, %llu frees, %llu live requested bytes, %llu peak requested bytes.\n",(unsigned long long)mc_attempts,(unsigned long long)mc_successes,(unsigned long long)mc_frees,(unsigned long long)mc_bytes,(unsigned long long)mc_peak);
#endif
    return result?1:0;
}
#endif
