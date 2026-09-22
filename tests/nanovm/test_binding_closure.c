/* I qualify corrected atomic construction before any VM execution admission. */
#include "../../src/nanovm/binding_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks, attempts, fail_at;
static bool injecting, persistent;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr,"I failed closure line %d: %s\n",__LINE__,#x);exit(1); \
} } while(0)
static void *closure_calloc(size_t n,size_t size) {
    if(injecting) {
        ++attempts;
        if(fail_at && (attempts==fail_at || (persistent && attempts>=fail_at)))return NULL;
    }
    return calloc(n,size);
}
#define calloc closure_calloc
#include "../../src/nanovm/heap.c"
#include "../../src/nanovm/binding_state.c"
#undef calloc

typedef struct {
    VmHeap heap;
    VmBindingState *state;
    NanoValue locals[4];
    VmClosure *sibling;
    VmBindingSource sources[6];
    uint8_t modes[6];
    size_t live,objects;
    uint32_t cell_refs;
    void **cycle_buf;
    uint32_t cycle_count,cycle_capacity;
} Fixture;
static void setup(Fixture *f) {
    memset(f,0,sizeof *f);vm_heap_init(&f->heap);
    const uint8_t modes[]={1,1,1,0};
    CHECK(vm_binding_state_new(&f->heap,modes,4,4,SIZE_MAX,&f->state)==VM_BINDING_OK);
    for(unsigned i=0;i<4;i++) {
        char text[2]={(char)('a'+i),0};VmString *s=vm_string_new(&f->heap,text,1);CHECK(s);
        f->locals[i]=val_string(s);
    }
    VmBindingSource first={.state=f->state,.locals=f->locals,.slot=0,.mode=1};
    const uint8_t shared=1;
    CHECK(vm_binding_closure(&f->heap,3,7,&shared,&first,1,SIZE_MAX,1,&f->sibling)==VM_BINDING_OK);
    const uint16_t slots[]={1,1,2,0,0,3};
    for(unsigned i=0;i<6;i++) {
        f->modes[i]=i==5?0:1;
        f->sources[i]=(VmBindingSource){.state=f->state,.locals=f->locals,
            .slot=slots[i],.mode=f->modes[i]};
    }
    f->sources[4]=(VmBindingSource){.value=f->sibling->captures[0],.mode=1};
    f->live=f->heap.stats.allocated-f->heap.stats.freed;f->objects=f->heap.stats.num_objects;
    f->cell_refs=f->state->slots[0].cell->header.ref_count;
    f->cycle_buf=f->heap.cycle_buf;f->cycle_count=f->heap.cycle_count;
    f->cycle_capacity=f->heap.cycle_capacity;
}
static void unchanged(Fixture *f) {
    CHECK(f->locals[0].tag==TAG_VOID&&f->state->slots[0].cell==f->sibling->captures[0].as.tuple);
    CHECK(f->state->slots[0].cell->header.ref_count==f->cell_refs);
    for(unsigned i=1;i<4;i++) {
        CHECK(!f->state->slots[i].cell&&f->locals[i].tag==TAG_STRING);
        CHECK(f->locals[i].as.string->length==1&&f->locals[i].as.string->data[0]==(char)('a'+i));
    }
    CHECK(f->heap.stats.num_objects==f->objects);
    CHECK(f->heap.stats.allocated-f->heap.stats.freed==f->live);
    CHECK(f->heap.cycle_buf==f->cycle_buf&&f->heap.cycle_count==f->cycle_count&&
          f->heap.cycle_capacity==f->cycle_capacity);
}
static void destroy(Fixture *f,VmClosure *closure) {
    if(closure)vm_release(&f->heap,val_closure(closure));
    vm_release(&f->heap,val_closure(f->sibling));
    vm_binding_state_destroy(f->state,f->locals);vm_gc_collect_cycles(&f->heap);
    CHECK(!f->heap.stats.num_objects&&f->heap.stats.allocated==f->heap.stats.freed);
    vm_heap_destroy(&f->heap);
}
static VmBindingResult construct(Fixture *f,size_t limit,size_t work,VmClosure **out) {
    return vm_binding_closure(&f->heap,3,19,f->modes,f->sources,6,limit,work,out);
}
static void published(Fixture *f,VmClosure *closure) {
    CHECK(closure->fn_idx==19&&closure->callable_module==3&&closure->capture_count==6);
    CHECK(closure->captures[0].as.tuple==closure->captures[1].as.tuple);
    CHECK(closure->captures[3].as.tuple==closure->captures[4].as.tuple);
    CHECK(closure->captures[3].as.tuple==f->sibling->captures[0].as.tuple);
    CHECK(closure->captures[0].as.tuple!=closure->captures[2].as.tuple);
    CHECK(closure->captures[5].as.string==f->locals[3].as.string);
    CHECK(f->locals[3].as.string->header.ref_count==2);
    for(unsigned i=0;i<3;i++)CHECK(f->locals[i].tag==TAG_VOID&&f->state->slots[i].cell);
    NanoValue incoming=val_int(42);
    CHECK(vm_binding_assign(f->state,f->locals,1,&incoming)==VM_BINDING_OK);
    CHECK(closure->captures[0].as.tuple->elements[0].as.i64==42);
    CHECK(closure->captures[1].as.tuple->elements[0].as.i64==42);
    incoming=val_int(51);
    CHECK(vm_binding_initialize(f->state,f->locals,1,&incoming)==VM_BINDING_OK);
    CHECK(closure->captures[0].as.tuple->elements[0].as.i64==42&&f->locals[1].as.i64==51);
    VmBindingSource fresh={.state=f->state,.locals=f->locals,.slot=1,.mode=1};
    const uint8_t shared=1;VmClosure *next=NULL;
    CHECK(vm_binding_closure(&f->heap,3,20,&shared,&fresh,1,SIZE_MAX,1,&next)==VM_BINDING_OK);
    CHECK(next->captures[0].as.tuple!=closure->captures[0].as.tuple);
    CHECK(next->captures[0].as.tuple->elements[0].as.i64==51);
    vm_release(&f->heap,val_closure(next));
}
static void allocation_controls(void) {
    for(unsigned mode=0;mode<2;mode++)for(unsigned failure=1;failure<=4;failure++) {
        Fixture f;setup(&f);VmClosure *out=f.sibling;
        injecting=true;attempts=0;fail_at=failure;persistent=mode!=0;
        CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_MEMORY);
        injecting=false;CHECK(attempts==failure&&out==f.sibling);unchanged(&f);
        attempts=0;fail_at=0;injecting=true;
        CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_OK);
        injecting=false;CHECK(attempts==4);published(&f,out);destroy(&f,out);
    }
}
static void refusal_controls(void) {
    Fixture f;setup(&f);VmClosure *out=f.sibling;
    size_t bytes=6*sizeof(BindingCaptureStage)+sizeof(VmClosure)+6*sizeof(NanoValue)+
        2*(sizeof(VmTuple)+sizeof(NanoValue));
    CHECK(construct(&f,f.live+bytes-1,100,&out)==VM_BINDING_LIMIT);unchanged(&f);
    CHECK(construct(&f,SIZE_MAX,16,&out)==VM_BINDING_LIMIT);unchanged(&f);
    f.modes[0]=0;CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_INVALID);
    f.modes[0]=1;unchanged(&f);
    NanoValue moved[4];memcpy(moved,f.locals,sizeof moved);f.sources[1].locals=moved;
    CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_INVALID);f.sources[1].locals=f.locals;unchanged(&f);
    f.sources[4].value=val_int(17);
    CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_INVALID);
    f.sources[4].value=f.sibling->captures[0];unchanged(&f);
    /* Only the new checked constructor sees this representable retain limit. */
    f.locals[3].as.string->header.ref_count=UINT32_MAX;
    CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_LIMIT);
    CHECK(f.locals[3].as.string->header.ref_count==UINT32_MAX);
    f.locals[3].as.string->header.ref_count=1;unchanged(&f);
    uint64_t releases=f.heap.stats.release_calls;
    f.heap.stats.release_calls=UINT64_MAX;
    CHECK(construct(&f,SIZE_MAX,100,&out)==VM_BINDING_LIMIT);
    CHECK(f.heap.stats.release_calls==UINT64_MAX);f.heap.stats.release_calls=releases;unchanged(&f);
    CHECK(out==f.sibling);
    CHECK(construct(&f,f.live+bytes,17,&out)==VM_BINDING_OK);
    published(&f,out);destroy(&f,out);
}
static void empty_controls(void) {
    VmHeap heap;vm_heap_init(&heap);VmClosure *out=NULL;
    CHECK(vm_binding_closure(&heap,1,0,NULL,NULL,0,sizeof(VmClosure)-1,0,&out)==VM_BINDING_LIMIT);
    CHECK(!out&&!heap.stats.num_objects);
    CHECK(vm_binding_closure(&heap,1,0,NULL,NULL,0,sizeof(VmClosure),0,&out)==VM_BINDING_OK);
    CHECK(!out->capture_count&&out->callable_module==1);
    vm_release(&heap,val_closure(out));CHECK(heap.stats.allocated==heap.stats.freed);
    vm_heap_destroy(&heap);
}
static void environment_controls(void) {
    Fixture f;setup(&f);VmClosure *closure=NULL;
    CHECK(construct(&f,SIZE_MAX,100,&closure)==VM_BINDING_OK);
    CHECK(vm_binding_environment(closure,3,19,f.modes,6)==VM_BINDING_OK);
    CHECK(vm_binding_environment(closure,2,19,f.modes,6)==VM_BINDING_INVALID);
    CHECK(vm_binding_environment(closure,3,20,f.modes,6)==VM_BINDING_INVALID);
    CHECK(vm_binding_environment(closure,3,19,f.modes,5)==VM_BINDING_INVALID);
    CHECK(vm_binding_environment(NULL,3,19,f.modes,6)==VM_BINDING_INVALID);
    CHECK(vm_binding_environment(NULL,3,19,NULL,0)==VM_BINDING_OK);
    CHECK(vm_binding_environment(NULL,0,19,NULL,0)==VM_BINDING_INVALID);
    uint8_t bad[6];memcpy(bad,f.modes,sizeof bad);bad[0]=2;
    CHECK(vm_binding_environment(closure,3,19,bad,6)==VM_BINDING_INVALID);
    NanoValue out=val_int(777),incoming=val_int(66);
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,20,f.modes,6,0,&out)==VM_BINDING_INVALID);
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,19,f.modes,6,6,&out)==VM_BINDING_INVALID);
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,19,bad,6,0,&out)==VM_BINDING_INVALID);
    CHECK(out.tag==TAG_INT&&out.as.i64==777);
    CHECK(vm_binding_upvalue_assign(&f.heap,closure,3,19,f.modes,6,5,&incoming)==VM_BINDING_INVALID);
    CHECK(incoming.as.i64==66);
    VmString *original=closure->captures[0].as.tuple->elements[0].as.string;
    CHECK(original->header.ref_count==1);
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,19,f.modes,6,0,&out)==VM_BINDING_OK);
    CHECK(out.tag==TAG_STRING&&out.as.string==original&&original->header.ref_count==2);
    CHECK(vm_binding_upvalue_assign(&f.heap,closure,3,19,f.modes,6,1,&out)==VM_BINDING_OK);
    CHECK(out.tag==TAG_VOID&&original->header.ref_count==1);
    CHECK(vm_binding_upvalue_assign(&f.heap,closure,3,19,f.modes,6,0,&incoming)==VM_BINDING_OK);
    CHECK(incoming.tag==TAG_VOID&&f.state->slots[1].cell->elements[0].as.i64==66);
    incoming=val_int(99);
    CHECK(vm_binding_upvalue_assign(&f.heap,closure,3,19,f.modes,6,4,&incoming)==VM_BINDING_OK);
    CHECK(f.sibling->captures[0].as.tuple->elements[0].as.i64==99);
    CHECK(vm_binding_read(f.state,f.locals,0,&out)==VM_BINDING_OK&&out.as.i64==99);
    out=val_int(777);
    VmString *immutable=f.locals[3].as.string;CHECK(immutable->header.ref_count==2);
    immutable->header.ref_count=UINT32_MAX;
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,19,f.modes,6,5,&out)==VM_BINDING_LIMIT);
    CHECK(out.as.i64==777&&immutable->header.ref_count==UINT32_MAX);
    immutable->header.ref_count=2;
    uint64_t retained=f.heap.stats.retain_calls;f.heap.stats.retain_calls=UINT64_MAX;
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,19,f.modes,6,5,&out)==VM_BINDING_LIMIT);
    CHECK(out.as.i64==777&&immutable->header.ref_count==2);f.heap.stats.retain_calls=retained;
    CHECK(vm_binding_upvalue_read(&f.heap,closure,3,19,f.modes,6,5,&out)==VM_BINDING_OK);
    CHECK(out.as.string==immutable&&immutable->header.ref_count==3);
    vm_release(&f.heap,out);destroy(&f,closure);

    VmHeap heap;vm_heap_init(&heap);VmTuple *tuple=vm_tuple_new(&heap,2);CHECK(tuple);
    tuple->elements[0]=val_int(5);tuple->elements[1]=val_int(6);
    const uint8_t copied=0,shared=1;
    VmBindingSource source={.value=val_tuple(tuple),.mode=0};
    CHECK(vm_binding_closure(&heap,4,6,&copied,&source,1,SIZE_MAX,1,&closure)==VM_BINDING_OK);
    CHECK(vm_binding_environment(closure,4,6,&copied,1)==VM_BINDING_OK);
    CHECK(vm_binding_environment(closure,4,6,&shared,1)==VM_BINDING_INVALID);
    out=val_void();
    CHECK(vm_binding_upvalue_read(&heap,closure,4,6,&copied,1,0,&out)==VM_BINDING_OK);
    CHECK(out.tag==TAG_TUPLE&&out.as.tuple==tuple&&out.as.tuple->count==2);
    CHECK(out.as.tuple->elements[0].as.i64==5&&out.as.tuple->elements[1].as.i64==6);
    vm_release(&heap,out);vm_release(&heap,val_closure(closure));vm_release(&heap,val_tuple(tuple));
    vm_gc_collect_cycles(&heap);CHECK(!heap.stats.num_objects&&heap.stats.allocated==heap.stats.freed);
    vm_heap_destroy(&heap);
}
int main(void) {
    allocation_controls();refusal_controls();empty_controls();environment_controls();
    printf("I passed %u atomic closure checks with complete staged-allocation recovery.\n",checks);
    return 0;
}
