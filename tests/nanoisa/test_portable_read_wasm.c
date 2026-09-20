/* I export private fixture controls, not a NanoISA producer or host ABI. */
#include "portable_read_wasm.h"
#define EXPORT(name) __attribute__((export_name(name)))
#define CHECK(x) do { if (!(x)) __builtin_trap(); } while (0)
__attribute__((import_module("nanolang_host_v1"), import_name("read_text")))
extern int32_t npr_wasm_host_read_text(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
static uint8_t input[8192], raw[8192];
static NmsRuntime runtime;
static NmsHandle roots[8];
static NprManagedResult results[2];
static uint32_t root_count, input_length;
static uint64_t baseline_bytes, baseline_objects;
static unsigned live;

static uint32_t hash(const uint8_t *p,uint32_t n) {
    uint32_t h=2166136261u;for(uint32_t i=0;i<n;i++){h^=p[i];h*=16777619u;}return h;
}
EXPORT("put") void put(uint32_t index,uint32_t byte) {CHECK(index<sizeof input && byte<256);input[index]=(uint8_t)byte;}
EXPORT("init") void init(uint32_t length,uint32_t fill) {
    CHECK(!live && length<=sizeof input && (fill==1||fill==8));
    nms_init(&runtime,NULL,0);CHECK(nms_bind_records(&runtime,NULL,0)==NMS_OK);CHECK(nms_begin(&runtime)==NMS_OK);
    for(uint32_t i=0;i<fill;i++)CHECK(nms_create(&runtime,input,length,&roots[i])==NMS_OK);
    root_count=fill;input_length=length;CHECK(nms_retain(&runtime,roots[0])==NMS_OK);CHECK(nms_retain(&runtime,roots[0])==NMS_OK);
    CHECK(nms_prepare_collection(&runtime)==NMS_OK);
    baseline_bytes=runtime.live_bytes;baseline_objects=runtime.live_objects;
    results[0]=(NprManagedResult){0};results[1]=(NprManagedResult){0};live=1;
}
EXPORT("roots_ok") uint32_t roots_ok(void) {
    CHECK(live);NmsView v;CHECK(nms_view(&runtime,roots[0],&v)==NMS_OK && v.length==input_length);
    for(uint32_t i=0;i<input_length;i++)CHECK(v.data[i]==input[i]);
    CHECK(runtime.slots[(uint32_t)(roots[0]&~NMS_DYNAMIC)].references==3);
    uint64_t bytes=baseline_bytes,objects=baseline_objects;
    for(unsigned i=0;i<2;i++)if(results[i].value){CHECK(nms_view(&runtime,results[i].value,&v)==NMS_OK);bytes+=v.length;objects++;}
    CHECK(runtime.live_bytes==bytes && runtime.live_objects==objects);return 1;
}
EXPORT("read") uint32_t read_result(uint32_t slot,uint32_t mode) {
    CHECK(live && slot<2 && !results[slot].value);
#ifdef NMS_TESTING
    uint64_t before_allocations=nms_test_live_allocations();
#endif
    NmsRuntime *selected=&runtime;NmsHandle argument=roots[0],array=0;
    if(mode==1)selected=NULL;
    else if(mode==2)runtime.active=0;
    else if(mode==3)runtime.disposed=1;
    else if(mode==4)argument=0;
    else if(mode==5){CHECK(nms_string_array_create(&runtime,&array)==NMS_OK);argument=array;}
    else CHECK(mode==0);
    results[slot]=npr_wasm_read_managed(selected,argument);
    runtime.active=1;runtime.disposed=0;
    if(array)CHECK(nms_release(&runtime,array)==NMS_OK);
    CHECK((results[slot].host_status==NPR_OK && results[slot].managed_status==NMS_OK)==!!results[slot].value);
    roots_ok();
#ifdef NMS_TESTING
    if(!results[slot].value)CHECK(nms_test_live_allocations()==before_allocations);
#endif
    return ((uint32_t)results[slot].host_status<<8)|(uint32_t)results[slot].managed_status;
}
EXPORT("length") uint32_t result_length(uint32_t slot) {CHECK(slot<2 && results[slot].value);NmsView v;CHECK(nms_view(&runtime,results[slot].value,&v)==NMS_OK);return v.length;}
EXPORT("hash") uint32_t result_hash(uint32_t slot) {CHECK(slot<2 && results[slot].value);NmsView v;CHECK(nms_view(&runtime,results[slot].value,&v)==NMS_OK);return hash(v.data,v.length);}
EXPORT("release") void release(uint32_t slot) {CHECK(slot<2);if(results[slot].value)CHECK(nms_release(&runtime,results[slot].value)==NMS_OK);results[slot]=(NprManagedResult){0};roots_ok();}
EXPORT("testing") uint32_t testing(void) {
#ifdef NMS_TESTING
 return 1;
#else
 return 0;
#endif
}
EXPORT("budget") void budget(uint32_t count) {
#ifdef NMS_TESTING
 nms_test_fail_after(&runtime,count==UINT32_MAX?UINT64_MAX:count);
#else
 CHECK(count==UINT32_MAX);
#endif
}
EXPORT("remaining") uint32_t remaining(void) {
#ifdef NMS_TESTING
 return (uint32_t)runtime.fail_after;
#else
 return UINT32_MAX;
#endif
}
EXPORT("finish") uint32_t finish(void) {
    CHECK(live);release(0);release(1);budget(UINT32_MAX);
    CHECK(nms_release(&runtime,roots[0])==NMS_OK);CHECK(nms_release(&runtime,roots[0])==NMS_OK);
    for(uint32_t i=0;i<root_count;i++)CHECK(nms_release(&runtime,roots[i])==NMS_OK);
    CHECK(!runtime.live_bytes && !runtime.live_objects);CHECK(nms_finish(&runtime,NMS_OK,0)==0);CHECK(!runtime.active);
    CHECK(nms_dispose(&runtime)==NMS_OK);
#ifdef NMS_TESTING
 CHECK(!nms_test_live_allocations());
#endif
    live=0;return 1;
}
EXPORT("pages") uint32_t pages(void) {return (uint32_t)__builtin_wasm_memory_size(0);}
EXPORT("grow") uint32_t grow(uint32_t n) {return (uint32_t)__builtin_wasm_memory_grow(0,n);}
EXPORT("trap") void trap(void) {__builtin_trap();}
EXPORT("raw_base") uint32_t raw_base(void) {return (uint32_t)(uintptr_t)raw;}
EXPORT("raw_put") void raw_put(uint32_t offset,uint32_t b) {CHECK(offset<sizeof raw && b<256);raw[offset]=(uint8_t)b;}
EXPORT("raw_get") uint32_t raw_get(uint32_t offset) {CHECK(offset<sizeof raw);return raw[offset];}
EXPORT("raw_call") int32_t raw_call(uint32_t p,uint32_t n,uint32_t d,uint32_t cap,uint32_t out) {return npr_wasm_host_read_text(p,n,d,cap,out);}
