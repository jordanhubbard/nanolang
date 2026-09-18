/* I exercise actual emitted helpers with a fixed bit oracle. */
#include "emitted_runtime.h"
#include "array_cases.h"
static double from_bits(uint64_t u) { double d; memcpy(&d,&u,8); return d; }
static uint64_t bits(double d) { uint64_t u; memcpy(&u,&d,8); return u; }
static DynArray *input(uint64_t u) {
    return dyn_array_push_float(dyn_array_new(ELEM_FLOAT),from_bits(u));
}
static void observe(DynArray *a, int length, uint64_t u) {
    assert(a && dyn_array_get_elem_type(a)==ELEM_FLOAT);
    assert(dyn_array_length(a)==length);
    if(length) assert(bits(dyn_array_get_float(a,0))==u);
}
static DynArray *(*const pair[])(DynArray*,DynArray*)={nl_array_add,nl_array_sub,nl_array_mul,nl_array_div};
static DynArray *(*const right[])(DynArray*,double)={nl_array_add_scalar_float,nl_array_sub_scalar_float,nl_array_mul_scalar_float,nl_array_div_scalar_float};
static DynArray *(*const left[])(double,DynArray*)={nl_array_radd_scalar_float,nl_array_rsub_scalar_float,nl_array_rmul_scalar_float,nl_array_rdiv_scalar_float};
int main(void) {
    retain_emitted_helpers(); gc_init();
    size_t count=sizeof(cases)/sizeof(cases[0]); assert(count==68);
    for(size_t i=0;i<count;i++) {
        ArrayCase c=cases[i]; DynArray *a=input(c.left),*b=input(c.right);
        DynArray *r=pair[c.operation](a,b); observe(r,1,c.expected); gc_release(r);
        r=right[c.operation](a,from_bits(c.right)); observe(r,1,c.expected); gc_release(r);
        r=left[c.operation](from_bits(c.left),b); observe(r,1,c.expected); gc_release(r);
        DynArray *aa=dyn_array_push_array(dyn_array_new(ELEM_ARRAY),a);
        DynArray *bb=dyn_array_push_array(dyn_array_new(ELEM_ARRAY),b);
        r=pair[c.operation](aa,bb);
        assert(dyn_array_get_elem_type(r)==ELEM_ARRAY && dyn_array_length(r)==1);
        DynArray *leaf=dyn_array_get_array(r,0); observe(leaf,1,c.expected);
        assert(leaf!=a && leaf!=b); gc_release(leaf); gc_release(r);
        observe(a,1,c.left); observe(b,1,c.right);
        gc_release(aa);gc_release(bb);gc_release(a);gc_release(b);
    }
    for(unsigned op=0;op<4;op++) {
        DynArray *a=dyn_array_new(ELEM_FLOAT),*b=dyn_array_new(ELEM_FLOAT);
        DynArray *r=pair[op](a,b);observe(r,0,0);gc_release(r);
        r=right[op](a,from_bits(UINT64_C(0x8000000000000000)));observe(r,0,0);gc_release(r);
        r=left[op](from_bits(UINT64_C(0x7ff0000000000001)),b);observe(r,0,0);gc_release(r);
        observe(a,0,0);observe(b,0,0);gc_release(a);gc_release(b);
    }
    int64_t expected[]={12,6,27,3};
    for(unsigned op=0;op<4;op++) {
        DynArray *a=dyn_array_push_int(dyn_array_new(ELEM_INT),9);
        DynArray *b=dyn_array_push_int(dyn_array_new(ELEM_INT),3);
        DynArray *r=pair[op](a,b);
        assert(dyn_array_get_int(r,0)==expected[op]);
        assert(dyn_array_get_int(a,0)==9 && dyn_array_get_int(b,0)==3);
        gc_release(r);gc_release(a);gc_release(b);
    }
    DynArray *a=dyn_array_push_string(dyn_array_new(ELEM_STRING),"left");
    DynArray *b=dyn_array_push_string(dyn_array_new(ELEM_STRING),"right");
    DynArray *out[]={nl_array_add(a,b),nl_array_add_scalar_string(a,"right"),nl_array_radd_scalar_string("left",b)};
    for(unsigned i=0;i<3;i++) {
        char *s=dyn_array_get_string(out[i],0);assert(strcmp(s,"leftright")==0);
        gc_release(s);gc_release(out[i]);
    }
    assert(strcmp(dyn_array_get_string(a,0),"left")==0);
    assert(strcmp(dyn_array_get_string(b,0),"right")==0);
    gc_release(a);gc_release(b);gc_shutdown();
    puts("PASS: 272 fixed-bit results, 12 empty outputs, 4 integer and 3 string controls; inputs unchanged");
    return 0;
}
