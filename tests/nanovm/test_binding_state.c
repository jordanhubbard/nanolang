/* I exercise actual owned local/cell storage before any VM opcode admission. */
#include "../../src/nanovm/binding_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks, state_live;
static bool fail_state;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr,"I failed binding line %d: %s\n",__LINE__,#x);exit(1); \
} } while(0)
static void *state_calloc(size_t count,size_t size) {
    if(fail_state)return NULL;
    void *p=calloc(count,size);if(p)++state_live;return p;
}
static void state_free(void *p) {
    if(p){CHECK(state_live);--state_live;}free(p);
}
#define calloc state_calloc
#define free state_free
#include "../../src/nanovm/binding_state.c"
#undef calloc
#undef free

static void quiescent(VmHeap *heap) {
    vm_gc_collect_cycles(heap);
    CHECK(!state_live && !heap->stats.num_objects);
    CHECK(heap->stats.allocated==heap->stats.freed);
}
/* I establish the already specified internal shape for storage tests only.
 * Whole-environment construction/rollback is a separate pending operation. */
static VmTuple *fixture_cell(VmHeap *heap,VmBindingState *state,NanoValue *locals,uint16_t slot) {
    CHECK(state->slots[slot].initialized&&state->slots[slot].shared&&!state->slots[slot].cell);
    VmTuple *cell=vm_tuple_new(heap,1);CHECK(cell);
    cell->elements[0]=locals[slot];locals[slot]=val_void();state->slots[slot].cell=cell;
    return cell;
}
static void allocation_controls(VmHeap *heap) {
    const uint8_t modes[]={1,0,1};
    VmBindingState *sentinel=(void *)heap,*out=sentinel;
    size_t bytes=sizeof(VmBindingState)+3*sizeof(VmBindingSlot);
    VmHeapStats before;memcpy(&before,&heap->stats,sizeof before);
    CHECK(vm_binding_state_new(heap,modes,3,1,bytes-1,&out)==VM_BINDING_LIMIT);
    CHECK(out==sentinel&&!memcmp(&before,&heap->stats,sizeof before));
    fail_state=true;
    CHECK(vm_binding_state_new(heap,modes,3,1,bytes,&out)==VM_BINDING_MEMORY);
    CHECK(out==sentinel&&!memcmp(&before,&heap->stats,sizeof before));
    fail_state=false;
    CHECK(vm_binding_state_new(heap,modes,3,1,bytes,&out)==VM_BINDING_OK);
    CHECK(out->count==3&&out->slots[0].initialized&&!out->slots[1].initialized);
    CHECK(heap->stats.allocated-heap->stats.freed==bytes);
    VmBindingState *second=sentinel;
    CHECK(vm_binding_state_new(heap,modes,3,0,2*bytes-1,&second)==VM_BINDING_LIMIT);
    CHECK(second==sentinel&&state_live==1);
    NanoValue locals[]={val_int(7),val_void(),val_void()};
    vm_binding_state_destroy(out,locals);quiescent(heap);
    out=sentinel;
    const uint8_t bad[]={2};
    CHECK(vm_binding_state_new(heap,bad,1,0,SIZE_MAX,&out)==VM_BINDING_INVALID);
    CHECK(vm_binding_state_new(heap,NULL,1,0,SIZE_MAX,&out)==VM_BINDING_INVALID);
    CHECK(vm_binding_state_new(heap,modes,1,2,SIZE_MAX,&out)==VM_BINDING_INVALID);
    CHECK(out==sentinel);
    size_t saved=heap->stats.allocated;
    heap->stats.allocated=SIZE_MAX;
    CHECK(vm_binding_state_new(heap,NULL,0,0,SIZE_MAX,&out)==VM_BINDING_LIMIT);
    CHECK(out==sentinel);heap->stats.allocated=saved;
    CHECK(vm_binding_state_new(heap,NULL,0,0,SIZE_MAX,&out)==VM_BINDING_OK);
    vm_binding_state_destroy(out,NULL);quiescent(heap);
}
static void local_controls(VmHeap *heap) {
    const uint8_t modes[]={1,0};VmBindingState *state=NULL;
    CHECK(vm_binding_state_new(heap,modes,2,1,SIZE_MAX,&state)==VM_BINDING_OK);
    NanoValue locals[]={val_int(7),val_void()},out=val_int(999),incoming=val_int(21);
    CHECK(vm_binding_read(state,locals,1,&out)==VM_BINDING_INVALID&&out.as.i64==999);
    CHECK(vm_binding_assign(state,locals,1,&incoming)==VM_BINDING_INVALID&&incoming.as.i64==21);
    CHECK(vm_binding_initialize(state,locals,1,&incoming)==VM_BINDING_OK&&incoming.tag==TAG_VOID);
    incoming=val_int(22);
    CHECK(vm_binding_assign(state,locals,1,&incoming)==VM_BINDING_INVALID&&incoming.as.i64==22);
    CHECK(vm_binding_initialize(state,locals,1,&incoming)==VM_BINDING_OK);
    CHECK(vm_binding_read(state,locals,1,&out)==VM_BINDING_OK&&out.as.i64==22);
    CHECK(vm_binding_clear(state,locals,1)==VM_BINDING_OK);
    CHECK(vm_binding_clear(state,locals,1)==VM_BINDING_OK);
    CHECK(vm_binding_read(state,locals,2,&out)==VM_BINDING_INVALID);
    /* I move the owned locals to a new stack address without changing state. */
    NanoValue moved[2];memcpy(moved,locals,sizeof moved);
    locals[0]=locals[1]=val_void();incoming=val_int(31);
    CHECK(vm_binding_assign(state,moved,0,&incoming)==VM_BINDING_OK);
    CHECK(vm_binding_read(state,moved,0,&out)==VM_BINDING_OK&&out.as.i64==31);
    vm_binding_state_destroy(state,moved);CHECK(moved[0].tag==TAG_VOID);quiescent(heap);
}
static void activation_range_controls(VmHeap *heap) {
    const uint8_t modes[] = {1, 0, 1, 0, 1};
    VmBindingState *sentinel = (void *)heap, *state = sentinel;
    VmHeapStats before; memcpy(&before, &heap->stats, sizeof before);
    CHECK(vm_binding_state_new_range(heap,modes,5,6,0,SIZE_MAX,&state)==VM_BINDING_INVALID);
    CHECK(vm_binding_state_new_range(heap,modes,5,4,2,SIZE_MAX,&state)==VM_BINDING_INVALID);
    CHECK(vm_binding_state_new_range(heap,modes,5,1,UINT16_MAX,SIZE_MAX,&state)==VM_BINDING_INVALID);
    const uint8_t bad[] = {0, 2};
    CHECK(vm_binding_state_new_range(heap,bad,2,1,1,SIZE_MAX,&state)==VM_BINDING_INVALID);
    CHECK(state==sentinel&&!memcmp(&before,&heap->stats,sizeof before));
    fail_state=true;
    CHECK(vm_binding_state_new_range(heap,modes,5,2,2,SIZE_MAX,&state)==VM_BINDING_MEMORY);
    CHECK(state==sentinel&&!memcmp(&before,&heap->stats,sizeof before));
    fail_state=false;
    CHECK(vm_binding_state_new_range(heap,modes,5,2,2,SIZE_MAX,&state)==VM_BINDING_OK);
    for (unsigned i=0;i<5;++i) {
        CHECK(state->slots[i].initialized==(i==2||i==3));
        CHECK(state->slots[i].shared==(modes[i]!=0)&&!state->slots[i].cell);
    }
    VmString *parameter=vm_string_new(heap,"parameter",9);CHECK(parameter);
    NanoValue locals[]={val_void(),val_void(),val_string(parameter),val_int(17),val_void()};
    NanoValue out=val_int(999), incoming=val_int(21);
    CHECK(vm_binding_read(state,locals,0,&out)==VM_BINDING_INVALID&&out.as.i64==999);
    CHECK(vm_binding_assign(state,locals,4,&incoming)==VM_BINDING_INVALID&&incoming.as.i64==21);
    CHECK(vm_binding_read(state,locals,2,&out)==VM_BINDING_OK&&out.as.string==parameter);
    CHECK(parameter->header.ref_count==2);
    vm_binding_state_destroy(state,locals);
    for (unsigned i=0;i<5;++i) CHECK(locals[i].tag==TAG_VOID);
    CHECK(out.as.string->length==9&&!memcmp(out.as.string->data,"parameter",9));
    vm_release(heap,out);quiescent(heap);
    CHECK(vm_binding_state_new_range(heap,modes,5,5,0,SIZE_MAX,&state)==VM_BINDING_OK);
    for (unsigned i=0;i<5;++i) CHECK(!state->slots[i].initialized);
    vm_binding_state_destroy(state,locals);quiescent(heap);
    CHECK(vm_binding_state_new_range(heap,NULL,0,0,0,SIZE_MAX,&state)==VM_BINDING_OK);
    vm_binding_state_destroy(state,NULL);quiescent(heap);
}
static void retained_value_controls(VmHeap *heap) {
    const uint8_t mode=1;VmBindingState *state=NULL;
    CHECK(vm_binding_state_new(heap,&mode,1,0,SIZE_MAX,&state)==VM_BINDING_OK);
    NanoValue local=val_void();VmString *string=vm_string_new(heap,"owned",5);CHECK(string);
    NanoValue incoming=val_string(string),out=val_int(999);
    CHECK(vm_binding_initialize(state,&local,0,&incoming)==VM_BINDING_OK);
    CHECK(vm_binding_read(state,&local,0,&out)==VM_BINDING_OK&&out.as.string==string);
    CHECK(string->header.ref_count==2);
    /* I test only the corrected checked read at its representable boundary. */
    string->header.ref_count=UINT32_MAX;NanoValue sentinel=val_int(999);
    uint64_t retained=heap->stats.retain_calls;
    CHECK(vm_binding_read(state,&local,0,&sentinel)==VM_BINDING_LIMIT);
    CHECK(sentinel.as.i64==999&&string->header.ref_count==UINT32_MAX&&heap->stats.retain_calls==retained);
    string->header.ref_count=2;
    incoming=val_int(42);CHECK(vm_binding_assign(state,&local,0,&incoming)==VM_BINDING_OK);
    CHECK(out.as.string->length==5&&!memcmp(out.as.string->data,"owned",5));
    vm_release(heap,out);vm_binding_state_destroy(state,&local);quiescent(heap);
}
static void shared_and_cycle_controls(VmHeap *heap) {
    const uint8_t mode=1;VmBindingState *state=NULL;
    CHECK(vm_binding_state_new(heap,&mode,1,1,SIZE_MAX,&state)==VM_BINDING_OK);
    NanoValue local=val_int(7);VmTuple *cell=fixture_cell(heap,state,&local,0);
    VmClosure *closure=vm_closure_new(heap,0,2);CHECK(closure);
    for(unsigned i=0;i<2;i++){closure->captures[i]=val_tuple(cell);vm_retain(heap,val_tuple(cell));}
    NanoValue incoming=val_int(17),out=val_void();
    CHECK(vm_binding_assign(state,&local,0,&incoming)==VM_BINDING_OK);
    CHECK(vm_binding_read(state,&local,0,&out)==VM_BINDING_OK&&out.as.i64==17);
    CHECK(local.tag==TAG_VOID&&closure->captures[0].as.tuple==closure->captures[1].as.tuple);
    incoming=val_int(19);CHECK(vm_binding_initialize(state,&local,0,&incoming)==VM_BINDING_OK);
    CHECK(cell->elements[0].as.i64==17&&local.as.i64==19&&!state->slots[0].cell);
    VmTuple *fresh=fixture_cell(heap,state,&local,0);CHECK(fresh!=cell);
    CHECK(vm_binding_clear(state,&local,0)==VM_BINDING_OK);
    CHECK(cell->elements[0].as.i64==17);
    vm_release(heap,val_closure(closure));vm_binding_state_destroy(state,&local);quiescent(heap);

    CHECK(vm_binding_state_new(heap,&mode,1,1,SIZE_MAX,&state)==VM_BINDING_OK);
    local=val_void();cell=fixture_cell(heap,state,&local,0);
    closure=vm_closure_new(heap,0,1);CHECK(closure);
    closure->captures[0]=val_tuple(cell);vm_retain(heap,val_tuple(cell));
    incoming=val_closure(closure);CHECK(vm_binding_assign(state,&local,0,&incoming)==VM_BINDING_OK);
    CHECK(vm_binding_clear(state,&local,0)==VM_BINDING_OK);
    uint64_t collected=vm_gc_collect_cycles(heap);CHECK(collected==2);
    vm_binding_state_destroy(state,&local);quiescent(heap);
}
int main(void) {
    VmHeap heap;vm_heap_init(&heap);
    allocation_controls(&heap);local_controls(&heap);
    activation_range_controls(&heap);
    retained_value_controls(&heap);shared_and_cycle_controls(&heap);
    vm_heap_destroy(&heap);
    printf("I passed %u binding storage checks, including shared cell lifetime and cycle reclamation.\n",checks);
    return 0;
}
