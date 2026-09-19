/* I link separately against the production target IR, with counted ABI roots. */
#include <stdint.h>
#include "../../src/nanoisa/managed_strings.h"
extern uint32_t nms_module_graph_begin(const NmsView *,uint32_t);
extern uint32_t nms_module_graph_collect(void);
extern uint64_t nms_module_graph_finish(int32_t);
extern uint32_t nms_module_status(void);
extern void nms_module_fail(uint32_t);
extern uint32_t nms_module_dispose(void);
extern uint64_t nms_module_array_create(uint32_t);
extern uint32_t nms_module_array_append_value(uint64_t,uint64_t,uint32_t);
extern uint32_t nms_module_array_get_value(uint64_t,uint64_t,uint64_t *,uint32_t *);
extern uint32_t nms_module_array_value_length(uint64_t);
extern uint32_t nms_module_retain(uint64_t,uint32_t);
extern void nms_module_release(uint64_t,uint32_t);
#define CHECK(x) do { if (!(x)) return __LINE__; } while(0)
static uint64_t global;
static uint32_t calls;
/* I receive a transferred argument and return an owned temporary. */
static uint64_t borrowed_result(uint64_t argument) {
    uint64_t result=0;uint32_t tag=0;
    nms_module_array_get_value(argument,0,&result,&tag);
    nms_module_release(argument,7);
    if(tag!=7)nms_module_fail(NMS_TYPE);
    return result;
}
int graph_entry(void) {
    CHECK(nms_module_graph_begin(0,0)==NMS_OK);
    if(!global)global=nms_module_array_create(7);
    CHECK(global && nms_module_status()==NMS_OK);
    CHECK(nms_module_array_append_value(global,calls,1)==NMS_OK);
    calls++;
    for(unsigned i=0;i<30000;i++) {
        CHECK(nms_module_graph_collect()==NMS_OK);
        uint64_t a=nms_module_array_create(7),b=nms_module_array_create(7);
        CHECK(a && b);
        CHECK(nms_module_array_append_value(a,b,7)==NMS_OK);
        CHECK(nms_module_array_append_value(b,a,7)==NMS_OK);
        nms_module_release(b,7);
        uint64_t temporary=borrowed_result(a); /* a moves into the call. */
        CHECK(temporary && nms_module_graph_collect()==NMS_OK);
        CHECK(nms_module_array_value_length(temporary)==1);
        nms_module_release(temporary,7);
    }
    CHECK(nms_module_array_value_length(global)==calls);
    nms_module_fail(NMS_ASSERT);
    CHECK(nms_module_graph_begin(0,0)==NMS_BUSY);
    CHECK(nms_module_graph_finish(12)==((uint64_t)NMS_ASSERT<<32));
    CHECK(nms_module_graph_begin(0,0)==NMS_OK);
    CHECK(nms_module_array_value_length(global)==calls);
    CHECK(nms_module_graph_finish((int32_t)calls)==calls);
    return 0;
}
int graph_dispose(void) {
    CHECK(nms_module_graph_begin(0,0)==NMS_OK);
    nms_module_release(global,7);global=0;
    CHECK(nms_module_graph_finish(0)==0);
    CHECK(nms_module_dispose()==NMS_OK);
    CHECK(nms_module_graph_begin(0,0)==NMS_DISPOSED);
    return 0;
}
#ifndef __wasm32__
int main(void) {
    for(unsigned i=0;i<3;i++){int result=graph_entry();if(result)return result;}
    return graph_dispose();
}
#endif
