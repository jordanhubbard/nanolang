/* I emit ordinary, declared VM packed-array values as target reference vectors. */
#include "../../src/nanovm/heap.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
static uint64_t bits(NanoValue value) {
    uint64_t result = 0;
    switch (value.tag) {
        case TAG_INT: memcpy(&result,&value.as.i64,8); break;
        case TAG_FLOAT: memcpy(&result,&value.as.f64,8); break;
        case TAG_U8: result=value.as.u8; break;
        case TAG_BOOL: result=value.as.boolean; break;
        default: assert(0);
    }
    return result;
}
static void emit(VmHeap *heap, uint8_t kind, NanoValue input) {
    VmArray *array=vm_array_new(heap,kind,8); assert(array);
    assert(vm_array_push(heap,array,input));
    NanoValue output=vm_array_get(array,0), popped=vm_array_pop(array);
    assert(output.tag==kind && popped.tag==kind && bits(output)==bits(popped));
    printf("{%u,%u,UINT64_C(%" PRIu64 "),%u,UINT64_C(%" PRIu64 ")},\n",
           kind,input.tag,bits(input),output.tag,bits(output));
    vm_release(heap,val_array(array));
}
int main(void) {
    VmHeap heap; vm_heap_init(&heap);
    const int64_t integers[]={0,1,-1,255,256,-256,INT64_MIN,INT64_MAX,
        INT64_C(9007199254740991),INT64_C(9007199254740992),INT64_C(9007199254740993),
        INT64_C(9007199254740995),-INT64_C(9007199254740993),-INT64_C(9007199254740995)};
    for(unsigned i=0;i<sizeof integers/sizeof integers[0];i++) {
        emit(&heap,TAG_INT,val_int(integers[i]));
        emit(&heap,TAG_U8,val_int(integers[i]));
        emit(&heap,TAG_FLOAT,val_int(integers[i]));
    }
    for(unsigned i=0;i<256;i++) {emit(&heap,TAG_U8,val_u8((uint8_t)i));emit(&heap,TAG_INT,val_u8((uint8_t)i));}
    const uint64_t floats[]={0,UINT64_C(0x8000000000000000),1,UINT64_C(0x3ff0000000000000),
        UINT64_C(0x7fefffffffffffff),UINT64_C(0x7ff8000000001234),UINT64_C(0xfff8000000004321)};
    for(unsigned i=0;i<sizeof floats/sizeof floats[0];i++) {
        double value;memcpy(&value,&floats[i],8);emit(&heap,TAG_FLOAT,val_float(value));
    }
    emit(&heap,TAG_BOOL,val_bool(false));emit(&heap,TAG_BOOL,val_bool(true));
    assert(!heap.stats.num_objects);vm_heap_destroy(&heap);return 0;
}
