#include "dyn_array.h"
#include "gc.h"
#include <assert.h>
#include <stdint.h>
#include <string.h>
/* GENERATED_HELPER */
int main(void) {
    gc_init();
    DynArray *a= dyn_array_new(ELEM_INT);
    for(int i=0;i<4;i++) dyn_array_push_int(a,10+i);
    DynArray *copy=nl_array_slice(a,1,INT64_MAX);
    assert(copy!=a && dyn_array_length(copy)==3 && dyn_array_get_int(copy,0)==11);
    dyn_array_set_int(copy,0,99);assert(dyn_array_get_int(a,1)==11);
    assert(dyn_array_length(nl_array_slice(a,INT64_MAX,INT64_MAX))==0);
    assert(dyn_array_length(nl_array_slice(a,INT64_MIN,INT64_MIN))==0);
    assert(dyn_array_length(nl_array_slice(a,INT64_MIN,INT64_MAX))==4);
    assert(dyn_array_length(nl_array_slice(NULL,1,2))==0);
    DynArray *u=dyn_array_new(ELEM_U8);dyn_array_push_u8(u,255);
    assert(dyn_array_get_u8(nl_array_slice(u,0,1),0)==255);
    DynArray *b=dyn_array_new(ELEM_BOOL);dyn_array_push_bool(b,true);
    assert(dyn_array_get_bool(nl_array_slice(b,0,1),0));
    DynArray *s=dyn_array_new(ELEM_STRING);dyn_array_push_string(s,"retained");
    assert(strcmp(dyn_array_get_string(nl_array_slice(s,0,1),0),"retained")==0);
    const uint64_t bits[]={UINT64_C(0x8000000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001),UINT64_C(1)};
    DynArray *f=dyn_array_new(ELEM_FLOAT);
    for(unsigned i=0;i<4;i++){double value;memcpy(&value,&bits[i],8);dyn_array_push_float(f,value);}
    DynArray *fs=nl_array_slice(f,0,4);
    for(unsigned i=0;i<4;i++){double value=dyn_array_get_float(fs,i);uint64_t got;memcpy(&got,&value,8);assert(got==bits[i]);}
    DynArray *nested=dyn_array_new(ELEM_ARRAY);dyn_array_push_array(nested,a);
    assert(dyn_array_get_array(nl_array_slice(nested,0,1),0)==a);
    struct Pair {int64_t number; DynArray *child;} pair={42,a};
    DynArray *records=dyn_array_new(ELEM_STRUCT);dyn_array_push_struct(records,&pair,sizeof pair);
    DynArray *rs=nl_array_slice(records,0,1);
    struct Pair *got=dyn_array_get_struct(rs,0);
    assert(got->number==42 && got->child==a && got!=dyn_array_get_struct(records,0));
    gc_shutdown();
    return 0;
}
