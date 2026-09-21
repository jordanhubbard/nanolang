/* I package this private runtime independently from the old singleton adapter.
 * Including the core keeps its native/Wasm allocator and testing hooks exact. */
#include "record_array_generated_private.h"
#ifdef NANO_RECORD_ARRAY_GENERATED_PRIVATE
#include "managed_strings.c"
#include <stddef.h>
#ifndef __wasm32__
#include <pthread.h>
#endif

typedef struct { uint32_t function, resume, stack; } NrgFrame;
struct NrgInstance {
    const NrgProgram *program;
    NmsRuntime heap;
    NmsValue *roots;
    NmsValue globals[256], result, completed;
    NrgFrame frames[NRG_FRAMES];
    uint64_t epoch, preparation_bytes;
    uint32_t depth, maximum_frames;
    NrgStatus status;
    bool busy, has_result, yielded;
#ifndef __wasm32__
    pthread_t owner;
#endif
};
_Static_assert(sizeof(NmsValue)==16 && offsetof(NmsValue,tag)==8,
               "I require my generated boxed value ABI");
_Static_assert(NRG_ROOTS==1024u*512u+1u,"I reserve every frame and initializer root");
static bool nrg_owner(const NrgInstance *p) {
#ifdef __wasm32__
    /* My private Wasm target has no shared-memory/thread imports. */
    return p!=NULL;
#else
    return p && pthread_equal(p->owner,pthread_self());
#endif
}
static NmsValue nrg_void(void) { NmsValue v={0,0}; return v; }
static bool nrg_leaf(uint32_t t) { return t>=1 && t<=5; }
static NrgStatus nrg_core_status(NmsStatus s) {
    switch(s) {
    case NMS_OK:return NRG_OK;
    case NMS_TYPE:return NRG_TYPE;
    case NMS_BOUNDS:return NRG_BOUNDS;
    case NMS_ASSERT:return NRG_ASSERT;
    case NMS_MEMORY:return NRG_MEMORY;
    case NMS_BUSY:return NRG_BUSY;
    default:return NRG_STATE;
    }
}
void nrg_fail(NrgInstance *p,NrgStatus status) {
    if(p && p->status==NRG_OK && status!=NRG_OK)p->status=status;
}
static bool nrg_core(NrgInstance *p,NmsStatus s) {
    nrg_fail(p,nrg_core_status(s));return s==NMS_OK;
}
NrgStatus nrg_status(const NrgInstance *p) { return p?p->status:NRG_STATE; }
static NrgFrame *nrg_frame(NrgInstance *p) {
    if(!nrg_owner(p) || !p->busy || !p->depth || p->depth>NRG_FRAMES || p->status!=NRG_OK)return NULL;
    return &p->frames[p->depth-1];
}
static NmsValue *nrg_slots(NrgInstance *p,uint32_t depth) { return p->roots+depth*512u; }
static void nrg_release(NrgInstance *p,NmsValue *slot) {
    NmsValue old=*slot;*slot=nrg_void();nrg_core(p,nms_value_release(&p->heap,old));
}
static bool nrg_valid(const NrgInstance *p,NmsValue v) {
    if(v.tag==0 || (v.tag>=1 && v.tag<=4))return true;
    if(v.tag==5) { NmsView view;return nms_view(&p->heap,v.payload,&view)==NMS_OK; }
    uint32_t index;
    if(slot_for(&p->heap,v.payload,&index)!=NMS_OK)return false;
    const NmsSlot *s=&p->heap.slots[index];
    if(v.tag==7)return s->vm_array_policy && nrg_leaf(s->element_tag) &&
        (s->kind==NMS_SLOT_PACKED_SCALAR_ARRAY || s->kind==NMS_SLOT_BOXED_ARRAY) &&
        s->length<=s->capacity && (!s->capacity || s->data);
    if(v.tag==8) {
        uint32_t ordinal,layout;
        return nms_record_identity(&p->heap,v.payload,&ordinal,&layout)==NMS_OK;
    }
    return false;
}
static bool nrg_accepts(NrgInstance *p,uint32_t ordinal,uint32_t field,NmsValue v) {
    if(ordinal>=p->program->record_count || field>=p->program->records[ordinal].field_count || !nrg_valid(p,v))return false;
    const NrgField *f=&p->program->fields[p->program->field_starts[ordinal]+field];
    if(v.tag!=f->tag)return false;
    if(v.tag==8) {
        uint32_t actual,layout;
        return nms_record_identity(&p->heap,v.payload,&actual,&layout)==NMS_OK && layout==f->nested_layout;
    }
    if(v.tag==7)return p->heap.slots[(uint32_t)v.payload].element_tag==f->element;
    return true;
}
/* Generated table validation is not a bytecode admission API. The emitter must
 * independently compare every table with its complete copied execution plan. */
static bool nrg_program_valid(const NrgProgram *p,uint64_t *bytes) {
    if(!p || p->abi!=NRG_ABI || p->value_size!=sizeof(NmsValue) ||
       p->value_tag_offset!=offsetof(NmsValue,tag) || p->frame_limit!=NRG_FRAMES ||
       !p->function_count || p->function_count>256 || !p->functions ||
       p->entry>=p->function_count || (p->initializer!=UINT32_MAX && p->initializer>=p->function_count) ||
       p->has_main>1 || p->global_count>256 || p->record_count>256 || p->field_count>65536 ||
       (p->literal_count && !p->literals) || (p->record_count && (!p->records || !p->field_starts)) ||
       (p->field_count && !p->fields))return false;
    uint64_t work=sizeof *p+(uint64_t)p->function_count*sizeof(NrgFunction)+
        (uint64_t)p->record_count*(sizeof(NmsRecordDescriptor)+sizeof(uint32_t))+
        (uint64_t)p->field_count*sizeof(NrgField)+(uint64_t)p->literal_count*sizeof(NmsView);
    if(work>NRG_EXTRA_STEPS)return false;
    for(uint32_t i=0;i<p->function_count;i++) {
        const NrgFunction *f=&p->functions[i];
        if(!f->body || f->locals>256 || f->arity>f->locals || f->maximum_stack>256 ||
           f->result_count>1 || (f->result_tag>5 && f->result_tag!=7 && f->result_tag!=8))return false;
        if(f->parameters)for(uint32_t j=0;j<f->arity;j++) {
            if(++work>NRG_EXTRA_STEPS || (f->parameters[j]>5 && f->parameters[j]!=7 && f->parameters[j]!=8))return false;
        }
    }
    if(p->functions[p->entry].arity || (p->initializer!=UINT32_MAX && p->functions[p->initializer].arity))return false;
    for(uint32_t i=0;i<p->literal_count;i++) {
        if(!p->literals[i].data)return false;
        if(p->literals[i].length>NRG_EXTRA_STEPS-work)return false;
        work+=p->literals[i].length;
    }
    uint32_t fields=0;
    for(uint32_t i=0;i<p->record_count;i++) {
        if(p->field_starts[i]!=fields ||
           p->records[i].field_count>p->field_count-fields)return false;
        if(i && p->records[i-1].global_layout_index>=p->records[i].global_layout_index)return false;
        fields+=p->records[i].field_count;
    }
    if(fields!=p->field_count)return false;
    /* I build a bounded adjacency bitset once, then compute finite DAG ranks.
     * Binary search uses the already checked sorted nominal identities. */
    uint64_t edges[256][4]={{0}};
    uint16_t ranks[256]={0};uint32_t done=0;
    work+=sizeof edges+sizeof ranks;
    for(uint32_t i=0;i<p->record_count;i++)for(uint32_t j=0;j<p->records[i].field_count;j++) {
        if(++work>NRG_EXTRA_STEPS)return false;
        const NrgField *f=&p->fields[p->field_starts[i]+j];
        if(nrg_leaf(f->tag))continue;
        if(f->tag==7) { if(!nrg_leaf(f->element))return false;continue; }
        if(f->tag!=8)return false;
        uint32_t lo=0,hi=p->record_count;
        while(lo<hi) {
            if(++work>NRG_EXTRA_STEPS)return false;
            uint32_t middle=lo+(hi-lo)/2;
            if(p->records[middle].global_layout_index<f->nested_layout)lo=middle+1;else hi=middle;
        }
        if(lo==p->record_count || p->records[lo].global_layout_index!=f->nested_layout)return false;
        edges[i][lo/64]|=UINT64_C(1)<<(lo%64);
    }
    for(uint32_t round=0;round<p->record_count && done<p->record_count;round++) {
        bool changed=false;
        for(uint32_t i=0;i<p->record_count;i++) {
            if(++work>NRG_EXTRA_STEPS)return false;
            if(ranks[i])continue;
            bool ready=true;uint16_t rank=1;
            for(uint32_t j=0;j<p->record_count;j++) {
                if(++work>NRG_EXTRA_STEPS)return false;
                if(!(edges[i][j/64]&(UINT64_C(1)<<(j%64))))continue;
                if(!ranks[j])ready=false;
                else if(rank<=ranks[j])rank=(uint16_t)(ranks[j]+1);
            }
            if(ready) { ranks[i]=rank;done++;changed=true; }
        }
        if(!changed)break;
    }
    if(done!=p->record_count)return false;
    *bytes=sizeof(NrgInstance)+(uint64_t)NRG_ROOTS*sizeof(NmsValue);
    /* The fixed validation scratch is automatic, separately charged at peak. */
    if(*bytes>NRG_EXTRA_BYTES-sizeof edges-sizeof ranks)return false;
    return *bytes<=NRG_EXTRA_BYTES && *bytes<=NRG_EXTRA_STEPS-work;
}
NrgStatus nrg_create(const NrgProgram *program,NrgInstance **out) {
    uint64_t bytes;
    if(!out || !nrg_program_valid(program,&bytes))return NRG_STATE;
    NmsRuntime allocator;nms_init(&allocator,NULL,0);
    NrgInstance *p=allocate(&allocator,sizeof *p);
    if(!p)return NRG_MEMORY;
    volatile unsigned char *zero=(volatile unsigned char *)p;
    for(size_t i=0;i<sizeof *p;i++)zero[i]=0;
    p->program=program;p->preparation_bytes=bytes;
#ifndef __wasm32__
    p->owner=pthread_self();
#endif
    nms_init(&p->heap,program->literals,program->literal_count);
    p->roots=allocate(&p->heap,(uint64_t)NRG_ROOTS*sizeof *p->roots);
    if(!p->roots) { deallocate(p);return NRG_MEMORY; }
    for(uint32_t i=0;i<NRG_ROOTS;i++)p->roots[i]=nrg_void();
    NmsStatus status=nms_bind_records(&p->heap,program->records,program->record_count);
    if(status!=NMS_OK) { deallocate(p->roots);deallocate(p);return nrg_core_status(status); }
    *out=p;return NRG_OK;
}
bool nrg_peek(NrgInstance *p,uint32_t distance,NmsValue *out) {
    NrgFrame *f=nrg_frame(p);
    if(!f || !out)return false;
    if(distance>=f->stack) { nrg_fail(p,NRG_STATE);return false; }
    uint32_t locals=p->program->functions[f->function].locals;
    NmsValue value=nrg_slots(p,p->depth-1)[locals+f->stack-1-distance];
    if(!nrg_valid(p,value)) { nrg_fail(p,NRG_TYPE);return false; }
    *out=value;return true;
}
bool nrg_push_move(NrgInstance *p,NmsValue *value) {
    NrgFrame *f=nrg_frame(p);
    if(!f || !value)return false;
    if(f->stack==256 || !nrg_valid(p,*value)) { nrg_fail(p,NRG_TYPE);return false; }
    uint32_t locals=p->program->functions[f->function].locals;
    nrg_slots(p,p->depth-1)[locals+f->stack++]=*value;*value=nrg_void();return true;
}
bool nrg_replace(NrgInstance *p,uint32_t count,NmsValue *value) {
    NrgFrame *f=nrg_frame(p);
    if(!p || !value)return false;
    if(!f) { nrg_release(p,value);return false; }
    if(count>f->stack || f->stack-count>=256 || !nrg_valid(p,*value)) {
        nrg_fail(p,NRG_TYPE);nrg_release(p,value);return false;
    }
    uint32_t locals=p->program->functions[f->function].locals;
    NmsValue *slots=nrg_slots(p,p->depth-1);
    while(count--)nrg_release(p,&slots[locals+--f->stack]);
    /* A checked release cannot fail for live roots; preserve an owned result
     * nevertheless if a defensive core failure occurs during cleanup. */
    if(p->status!=NRG_OK) { nrg_release(p,value);return false; }
    return nrg_push_move(p,value);
}
static NmsValue *nrg_variable(NrgInstance *p,bool global,uint32_t index) {
    NrgFrame *f=nrg_frame(p);
    if(!f)return NULL;
    uint32_t count=global?p->program->global_count:p->program->functions[f->function].locals;
    if(index>=count) { nrg_fail(p,NRG_BOUNDS);return NULL; }
    return global?&p->globals[index]:&nrg_slots(p,p->depth-1)[index];
}
bool nrg_load(NrgInstance *p,bool global,uint32_t index) {
    NmsValue *slot=nrg_variable(p,global,index);
    if(!slot)return false;
    NmsValue value=*slot;
    if(!nrg_core(p,nms_value_retain(&p->heap,value)))return false;
    if(nrg_push_move(p,&value))return true;
    nrg_release(p,&value);return false;
}
bool nrg_store(NrgInstance *p,bool global,uint32_t index) {
    NmsValue value,*slot=nrg_variable(p,global,index);
    if(!slot || !nrg_peek(p,0,&value))return false;
    NrgFrame *f=nrg_frame(p);uint32_t locals=p->program->functions[f->function].locals;
    NmsValue previous=*slot;*slot=value;
    nrg_slots(p,p->depth-1)[locals+--f->stack]=nrg_void();
    nrg_release(p,&previous);return p->status==NRG_OK;
}
bool nrg_drop(NrgInstance *p) {
    NrgFrame *f=nrg_frame(p);
    if(!f)return false;
    if(!f->stack) { nrg_fail(p,NRG_STATE);return false; }
    nrg_release(p,&nrg_slots(p,p->depth-1)[p->program->functions[f->function].locals+--f->stack]);
    return p->status==NRG_OK;
}
bool nrg_dup(NrgInstance *p) {
    NmsValue value;
    if(!nrg_peek(p,0,&value) || !nrg_core(p,nms_value_retain(&p->heap,value)))return false;
    if(nrg_push_move(p,&value))return true;
    nrg_release(p,&value);return false;
}
bool nrg_swap(NrgInstance *p) {
    NmsValue a,b;if(!nrg_peek(p,0,&a) || !nrg_peek(p,1,&b))return false;
    NrgFrame *f=nrg_frame(p);uint32_t end=p->program->functions[f->function].locals+f->stack;
    NmsValue *slots=nrg_slots(p,p->depth-1);slots[end-1]=b;slots[end-2]=a;return true;
}
bool nrg_safe_point(NrgInstance *p) {
    if(!nrg_frame(p))return false;
    return nrg_core(p,nms_prepare_collection(&p->heap)) && nrg_core(p,nms_collect_prepared(&p->heap));
}
uint32_t nrg_resume(const NrgInstance *p) {
    return p && p->depth?p->frames[p->depth-1].resume:UINT32_MAX;
}
void nrg_call(NrgInstance *p,uint32_t function,uint32_t continuation) {
    NrgFrame *caller=nrg_frame(p);
    if(!caller)return;
    if(function>=p->program->function_count) { nrg_fail(p,NRG_STATE);return; }
    if(p->depth==NRG_FRAMES) { nrg_fail(p,NRG_FRAMES_EXHAUSTED);return; }
    const NrgFunction *callee=&p->program->functions[function];
    if(callee->arity>caller->stack) { nrg_fail(p,NRG_TYPE);return; }
    uint32_t locals=p->program->functions[caller->function].locals;
    NmsValue *source=nrg_slots(p,p->depth-1)+locals+caller->stack-callee->arity;
    for(uint32_t i=0;i<callee->arity;i++)
        if(!nrg_valid(p,source[i]) || (callee->parameters && callee->parameters[i] && callee->parameters[i]!=source[i].tag)) {
            nrg_fail(p,NRG_TYPE);return;
        }
    NmsValue *target=nrg_slots(p,p->depth);
    for(uint32_t i=0;i<callee->arity;i++) { target[i]=source[i];source[i]=nrg_void(); }
    caller->stack-=callee->arity;caller->resume=continuation;
    p->frames[p->depth++]=(NrgFrame){function,0,0};
    if(p->maximum_frames<p->depth)p->maximum_frames=p->depth;
    p->yielded=true;
}
static void nrg_clear_frame(NrgInstance *p) {
    NrgFrame *f=&p->frames[p->depth-1];
    uint32_t n=p->program->functions[f->function].locals+f->stack;
    NmsValue *slots=nrg_slots(p,p->depth-1);
    while(n)nrg_release(p,&slots[--n]);
    *f=(NrgFrame){0,0,0};p->depth--;
}
void nrg_return(NrgInstance *p) {
    NrgFrame *frame=nrg_frame(p);if(!frame)return;
    const NrgFunction *f=&p->program->functions[frame->function];
    NmsValue result=nrg_void();
    if(frame->stack!=f->result_count) { nrg_fail(p,NRG_TYPE);return; }
    if(f->result_count) {
        NmsValue *slot=&nrg_slots(p,p->depth-1)[f->locals];
        if(!nrg_valid(p,*slot) || (f->result_tag && f->result_tag!=slot->tag)) { nrg_fail(p,NRG_TYPE);return; }
        result=*slot;*slot=nrg_void();frame->stack=0;
    }
    nrg_clear_frame(p);p->yielded=true;
    if(p->status!=NRG_OK) { nrg_release(p,&result);return; }
    if(!p->depth)p->completed=result;
    else if(f->result_count && !nrg_push_move(p,&result))nrg_release(p,&result);
}
static void nrg_execute_root(NrgInstance *p,uint32_t function) {
    p->frames[0]=(NrgFrame){function,0,0};p->depth=1;
    if(!p->maximum_frames)p->maximum_frames=1;
    while(p->depth && p->status==NRG_OK) {
        p->yielded=false;
        p->program->functions[p->frames[p->depth-1].function].body(p);
        if(!p->yielded && p->status==NRG_OK)nrg_fail(p,NRG_STATE);
    }
}
NrgStatus nrg_run(NrgInstance *p) {
    if(!p)return NRG_STATE;
    if(!nrg_owner(p))return NRG_TYPE;
    if(p->busy)return NRG_BUSY;
    if(p->epoch==UINT64_MAX)return NRG_STATE;
    NmsStatus begin=nms_begin(&p->heap);
    if(begin!=NMS_OK)return nrg_core_status(begin);
    p->busy=true;p->status=NRG_OK;p->epoch++;
    if(!p->program->has_main)nrg_fail(p,NRG_UNDEFINED_FUNCTION);
    if(p->status==NRG_OK && nrg_core(p,nms_prepare_collection(&p->heap))) {
        if(p->program->initializer!=UINT32_MAX) {
            nrg_execute_root(p,p->program->initializer);
            if(p->status==NRG_OK) { p->roots[NRG_ROOTS-1]=p->completed;p->completed=nrg_void(); }
        }
        if(p->status==NRG_OK)nrg_execute_root(p,p->program->entry);
    }
    while(p->depth)nrg_clear_frame(p);
    nrg_release(p,&p->roots[NRG_ROOTS-1]);
    if(p->status==NRG_OK) {
        NmsValue previous=p->result;p->result=p->completed;p->completed=nrg_void();
        p->has_result=true;nrg_release(p,&previous);
    } else nrg_release(p,&p->completed);
    /* The private status is separate from the core's historical packed result. */
    NmsStatus cleanup=nms_finish(&p->heap,NMS_OK,0)>>32;
    nrg_core(p,cleanup);p->busy=false;return p->status;
}
void nrg_destroy(NrgInstance *p) {
    if(!nrg_owner(p) || p->busy)return;
    nrg_release(p,&p->result);nrg_release(p,&p->completed);
    for(uint32_t i=0;i<p->program->global_count;i++)nrg_release(p,&p->globals[i]);
    nms_dispose(&p->heap);deallocate(p->roots);deallocate(p);
}
bool nrg_stats(const NrgInstance *p,NrgStats *out) {
    if(!nrg_owner(p) || !out || p->busy)return false;
    NrgStats s={p->epoch,p->preparation_bytes,p->heap.live_bytes,p->heap.live_objects,
        p->depth,p->maximum_frames,p->status,p->has_result};*out=s;return true;
}

static bool nrg_receiver_result(NrgInstance *p,uint32_t count,uint32_t distance) {
    NmsValue value;
    if(!nrg_peek(p,distance,&value))return false;
    NrgFrame *f=nrg_frame(p);uint32_t locals=p->program->functions[f->function].locals;
    nrg_slots(p,p->depth-1)[locals+f->stack-1-distance]=nrg_void();
    return nrg_replace(p,count,&value);
}
static bool nrg_array(NrgInstance *p,NmsValue value,uint32_t *length,uint32_t *element) {
    if(value.tag!=7 || !nrg_valid(p,value)) { nrg_fail(p,NRG_TYPE);return false; }
    const NmsSlot *slot=&p->heap.slots[(uint32_t)value.payload];
    if(length)*length=slot->length;
    if(element)*element=slot->element_tag;
    return true;
}
bool nrg_record_new(NrgInstance *p,uint32_t ordinal) {
    if(!nrg_frame(p))return false;
    if(ordinal>=p->program->record_count) { nrg_fail(p,NRG_TYPE);return false; }
    uint32_t count=p->program->records[ordinal].field_count;
    NrgFrame *frame=nrg_frame(p);
    if(count>frame->stack) { nrg_fail(p,NRG_TYPE);return false; }
    NmsValue *values=nrg_slots(p,p->depth-1)+p->program->functions[frame->function].locals+frame->stack-count;
    for(uint32_t i=0;i<count;i++)if(!nrg_accepts(p,ordinal,i,values[i])) { nrg_fail(p,NRG_TYPE);return false; }
    if(!nrg_safe_point(p))return false;
    NmsValue result={0,8};
    if(!nrg_core(p,nms_record_create(&p->heap,ordinal,values,count,&result.payload)))return false;
    return nrg_replace(p,count,&result);
}
static bool nrg_record(NrgInstance *p,NmsValue value,uint32_t field,bool aggregate,uint32_t *ordinal) {
    if(value.tag!=8 || !nrg_valid(p,value)) { nrg_fail(p,aggregate?NRG_BOUNDS:NRG_TYPE);return false; }
    uint32_t layout;
    if(!nrg_core(p,nms_record_identity(&p->heap,value.payload,ordinal,&layout)))return false;
    if(field>=p->program->records[*ordinal].field_count) { nrg_fail(p,NRG_BOUNDS);return false; }
    return true;
}
bool nrg_record_get(NrgInstance *p,uint32_t field,bool aggregate) {
    NmsValue receiver,result=nrg_void();uint32_t ordinal;
    if(!nrg_peek(p,0,&receiver) || !nrg_record(p,receiver,field,aggregate,&ordinal))return false;
    if(!nrg_core(p,nms_record_get(&p->heap,receiver.payload,field,&result)))return false;
    return nrg_replace(p,1,&result);
}
bool nrg_record_set(NrgInstance *p,uint32_t field,bool aggregate) {
    NmsValue receiver,value;uint32_t ordinal;
    if(!nrg_peek(p,1,&receiver) || !nrg_peek(p,0,&value) || !nrg_record(p,receiver,field,aggregate,&ordinal))return false;
    if(!nrg_accepts(p,ordinal,field,value)) { nrg_fail(p,NRG_TYPE);return false; }
    if(!nrg_core(p,nms_record_set(&p->heap,receiver.payload,field,value)))return false;
    return nrg_receiver_result(p,2,1);
}
bool nrg_array_new(NrgInstance *p,uint32_t tag,uint32_t count,bool literal) {
    NrgFrame *frame=nrg_frame(p);if(!frame)return false;
    if(!nrg_leaf(tag) || count>frame->stack || count>256 || (!literal && count)) { nrg_fail(p,NRG_TYPE);return false; }
    /* Fixed constructor scratch is bounded by the already admitted stack. */
    uint64_t bits[256];uint32_t tags[256];
    for(uint32_t i=0;i<count;i++) {
        NmsValue v;if(!nrg_peek(p,count-1-i,&v))return false;
        if(v.tag!=tag) { nrg_fail(p,NRG_TYPE);return false; }
        bits[i]=v.payload;tags[i]=v.tag;
    }
    if(!nrg_safe_point(p))return false;
    NmsValue result={0,7};
    NmsStatus status=literal?nms_vm_array_literal(&p->heap,tag,bits,tags,count,&result.payload):
        nms_vm_array_create(&p->heap,tag,&result.payload);
    if(!nrg_core(p,status))return false;
    return nrg_replace(p,count,&result);
}
bool nrg_array_get(NrgInstance *p) {
    NmsValue array,index,result=nrg_void();uint32_t length;
    if(!nrg_peek(p,1,&array) || !nrg_peek(p,0,&index) || !nrg_array(p,array,&length,NULL))return false;
    /* Only a missing integer index produces VOID; a wrong tag is TYPE. */
    if(index.tag!=1) { nrg_fail(p,NRG_TYPE);return false; }
    if(index.payload<length &&
       !nrg_core(p,nms_value_array_get(&p->heap,array.payload,index.payload,&result)))return false;
    return nrg_replace(p,2,&result);
}
bool nrg_array_set(NrgInstance *p) {
    NmsValue array,index,value;uint32_t length,tag;
    if(!nrg_peek(p,2,&array) || !nrg_peek(p,1,&index) || !nrg_peek(p,0,&value) ||
       !nrg_array(p,array,&length,&tag))return false;
    if(index.tag!=1 || value.tag!=tag) { nrg_fail(p,NRG_TYPE);return false; }
    if(index.payload>=length) { nrg_fail(p,NRG_BOUNDS);return false; }
    if(!nrg_safe_point(p) || !nrg_core(p,nms_value_array_set(&p->heap,array.payload,index.payload,value)))return false;
    return nrg_receiver_result(p,3,2);
}
bool nrg_array_push(NrgInstance *p) {
    NmsValue array,value;uint32_t tag;
    if(!nrg_peek(p,1,&array) || !nrg_peek(p,0,&value) || !nrg_array(p,array,NULL,&tag))return false;
    if(value.tag!=tag) { nrg_fail(p,NRG_TYPE);return false; }
    if(!nrg_safe_point(p) || !nrg_core(p,nms_value_array_append(&p->heap,array.payload,value)))return false;
    return nrg_receiver_result(p,2,1);
}
bool nrg_array_pop(NrgInstance *p) {
    NmsValue array,result=nrg_void();
    if(!nrg_peek(p,0,&array) || !nrg_array(p,array,NULL,NULL))return false;
    if(!nrg_core(p,nms_value_array_pop(&p->heap,array.payload,&result)))return false;
    return nrg_replace(p,1,&result);
}
bool nrg_array_slice(NrgInstance *p) {
    NmsValue array,start,end,result={0,7};uint32_t length;
    if(!nrg_peek(p,2,&array) || !nrg_peek(p,1,&start) || !nrg_peek(p,0,&end) ||
       !nrg_array(p,array,&length,NULL))return false;
    uint32_t first=start.tag==1?(uint32_t)start.payload:0;
    uint32_t last=end.tag==1?(uint32_t)end.payload:length;
    if(!nrg_safe_point(p) || !nrg_core(p,nms_vm_array_slice(&p->heap,array.payload,first,last,&result.payload)))return false;
    return nrg_replace(p,3,&result);
}

static double nrg_double(uint64_t bits) {
    double value;copy_bytes((unsigned char *)&value,(const unsigned char *)&bits,8);return value;
}
static uint64_t nrg_double_bits(double value) {
    uint64_t bits;copy_bytes((unsigned char *)&bits,(const unsigned char *)&value,8);return bits;
}
bool nrg_truth(NrgInstance *p,const NmsValue *input) {
    if(!p || !input)return false;
    NmsValue v=*input;
    if(!nrg_valid(p,v)) { nrg_fail(p,NRG_TYPE);return false; }
    switch(v.tag) {
    case 0:return false;
    case 1:return v.payload!=0;
    case 2:return (uint8_t)v.payload!=0;
    case 3:return nrg_double(v.payload)!=0.0;
    case 4:return v.payload!=0;
    default:return true; /* Valid empty strings remain non-null heap objects. */
    }
}
static int nrg_string_order(NrgInstance *p,NmsValue a,NmsValue b) {
    NmsView x,y;
    if(!nrg_core(p,nms_view(&p->heap,a.payload,&x)) || !nrg_core(p,nms_view(&p->heap,b.payload,&y)))return 0;
    uint32_t n=x.length<y.length?x.length:y.length;
    for(uint32_t i=0;i<n;i++)if(x.data[i]!=y.data[i])return (int)x.data[i]-(int)y.data[i];
    return x.length<y.length?-1:x.length>y.length?1:0;
}
bool nrg_equal(NrgInstance *p,const NmsValue *left,const NmsValue *right) {
    if(!p || !left || !right)return false;
    NmsValue a=*left,b=*right;
    if(!nrg_valid(p,a) || !nrg_valid(p,b)) { nrg_fail(p,NRG_TYPE);return false; }
    if(a.tag==1 && b.tag==3)return (double)(int64_t)a.payload==nrg_double(b.payload);
    if(a.tag==3 && b.tag==1)return nrg_double(a.payload)==(double)(int64_t)b.payload;
    if(a.tag!=b.tag)return false;
    switch(a.tag) {
    case 0:return true;
    case 2:return (uint8_t)a.payload==(uint8_t)b.payload;
    case 3:return nrg_double(a.payload)==nrg_double(b.payload);
    case 4:return (a.payload!=0)==(b.payload!=0);
    case 5:return nrg_string_order(p,a,b)==0;
    default:return a.payload==b.payload;
    }
}
int nrg_order(NrgInstance *p,const NmsValue *left,const NmsValue *right) {
    if(!p || !left || !right)return 0;
    NmsValue a=*left,b=*right;
    if(!nrg_valid(p,a) || !nrg_valid(p,b)) { nrg_fail(p,NRG_TYPE);return 0; }
    if((a.tag==1 && b.tag==3) || (a.tag==3 && b.tag==1) || (a.tag==3 && b.tag==3)) {
        double x=a.tag==3?nrg_double(a.payload):(double)(int64_t)a.payload;
        double y=b.tag==3?nrg_double(b.payload):(double)(int64_t)b.payload;
        return x<y?-1:x>y?1:0;
    }
    if(a.tag!=b.tag)return (int)a.tag-(int)b.tag;
    switch(a.tag) {
    case 1:return (int64_t)a.payload<(int64_t)b.payload?-1:(int64_t)a.payload>(int64_t)b.payload?1:0;
    case 2:return (int)(uint8_t)a.payload-(int)(uint8_t)b.payload;
    case 4:return (int)(a.payload!=0)-(int)(b.payload!=0);
    case 5:return nrg_string_order(p,a,b);
    default:return 0;
    }
}
bool nrg_cast_int(NrgInstance *p) {
    NmsValue v,r={0,1};if(!nrg_peek(p,0,&v))return false;
    switch(v.tag) {
    case 1:r.payload=v.payload;break;
    case 2:r.payload=(uint8_t)v.payload;break;
    case 3: {
        double value=nrg_double(v.payload);
        if(!(value>=-0x1p63 && value<0x1p63)) { nrg_fail(p,NRG_TYPE);return false; }
        r.payload=(uint64_t)(int64_t)value;break;
    }
    case 4:r.payload=v.payload!=0;break;
    case 5: {
        int64_t value;
        if(!nrg_core(p,nms_parse_i64(&p->heap,v.payload,&value)))return false;
        r.payload=(uint64_t)value;break;
    }
    default:break;
    }
    return nrg_replace(p,1,&r);
}
bool nrg_cast_float(NrgInstance *p) {
    NmsValue v,r={0,3};if(!nrg_peek(p,0,&v))return false;
    switch(v.tag) {
    case 1:r.payload=nrg_double_bits((double)(int64_t)v.payload);break;
    case 2:r.payload=nrg_double_bits((double)(uint8_t)v.payload);break;
    case 3:r.payload=v.payload;break;
    case 4:r.payload=nrg_double_bits(v.payload?1.0:0.0);break;
    case 5: {
        NmsView view;
        if(!nrg_core(p,nms_view(&p->heap,v.payload,&view)))return false;
        if(!nbp_parse(view.data,view.length,&r.payload)) { nrg_fail(p,NRG_TYPE);return false; }
        break;
    }
    default:break;
    }
    return nrg_replace(p,1,&r);
}
bool nrg_format(NrgInstance *p,uint32_t tag) {
    NmsValue v,r={0,5};if(!nrg_peek(p,0,&v) || !nrg_safe_point(p))return false;
    if(tag!=1 && tag!=3) { nrg_fail(p,NRG_STATE);return false; }
    uint64_t bits=v.tag==tag?v.payload:0;
    if(!nrg_core(p,nms_format_scalar(&p->heap,bits,tag,&r.payload)))return false;
    return nrg_replace(p,1,&r);
}
bool nrg_cast_string(NrgInstance *p) {
    NmsValue v,r={0,5};if(!nrg_peek(p,0,&v))return false;
    if(v.tag==5)return true;
    if(!nrg_safe_point(p))return false;
    NmsStatus status=nrg_leaf(v.tag)?nms_format_scalar(&p->heap,v.payload,v.tag,&r.payload):
        nms_create(&p->heap,NULL,0,&r.payload);
    if(!nrg_core(p,status))return false;
    return nrg_replace(p,1,&r);
}
/* Move one root into a core helper with a documented consume-on-every-path
 * contract. The other operand slots remain visible to collection/unwind. */
static void nrg_take(NrgInstance *p,uint32_t distance) {
    NrgFrame *f=nrg_frame(p);
    nrg_slots(p,p->depth-1)[p->program->functions[f->function].locals+f->stack-1-distance]=nrg_void();
}
static bool nrg_string_input(NrgInstance *p,uint32_t distance,NmsValue *v) {
    if(!nrg_peek(p,distance,v))return false;
    if(v->tag!=5) { nrg_fail(p,NRG_TYPE);return false; }
    return true;
}
bool nrg_concat(NrgInstance *p) {
    NmsValue a,b,r={0,5};
    if(!nrg_string_input(p,1,&a) || !nrg_string_input(p,0,&b) || !nrg_safe_point(p))return false;
    nrg_take(p,1);nrg_take(p,0);
    if(!nrg_core(p,nms_concat_owned(&p->heap,a.payload,b.payload,&r.payload)))return false;
    return nrg_replace(p,2,&r);
}
bool nrg_trim(NrgInstance *p) {
    NmsValue v,r={0,5};if(!nrg_string_input(p,0,&v) || !nrg_safe_point(p))return false;
    nrg_take(p,0);
    if(!nrg_core(p,nms_trim_owned(&p->heap,v.payload,&r.payload)))return false;
    return nrg_replace(p,1,&r);
}
bool nrg_case(NrgInstance *p,bool upper) {
    NmsValue v,r={0,5};if(!nrg_string_input(p,0,&v) || !nrg_safe_point(p))return false;
    nrg_take(p,0);
    if(!nrg_core(p,nms_case_owned(&p->heap,v.payload,upper,&r.payload)))return false;
    return nrg_replace(p,1,&r);
}
bool nrg_substring(NrgInstance *p) {
    NmsValue v,first,length,r={0,5};
    if(!nrg_string_input(p,2,&v) || !nrg_peek(p,1,&first) || !nrg_peek(p,0,&length) || !nrg_safe_point(p))return false;
    nrg_take(p,2);
    if(!nrg_core(p,nms_substr_owned(&p->heap,v.payload,first.tag==1?(uint32_t)first.payload:0,
        length.tag==1?(uint32_t)length.payload:0,&r.payload)))return false;
    return nrg_replace(p,3,&r);
}
bool nrg_split(NrgInstance *p) {
    NmsValue a,b,r={0,7};
    if(!nrg_string_input(p,1,&a) || !nrg_string_input(p,0,&b) || !nrg_safe_point(p))return false;
    nrg_take(p,1);nrg_take(p,0);
    if(!nrg_core(p,nms_split_values_owned(&p->heap,a.payload,b.payload,&r.payload)))return false;
    return nrg_replace(p,2,&r);
}
bool nrg_string_replace(NrgInstance *p) {
    NmsValue a,b,c,r={0,5};
    if(!nrg_string_input(p,2,&a) || !nrg_string_input(p,1,&b) || !nrg_string_input(p,0,&c) || !nrg_safe_point(p))return false;
    nrg_take(p,2);nrg_take(p,1);nrg_take(p,0);
    if(!nrg_core(p,nms_replace_owned(&p->heap,a.payload,b.payload,c.payload,&r.payload)))return false;
    return nrg_replace(p,3,&r);
}
bool nrg_length(NrgInstance *p,bool array) {
    NmsValue v,r={0,1};if(!nrg_peek(p,0,&v))return false;
    if(array) {
        uint32_t length;if(!nrg_array(p,v,&length,NULL))return false;r.payload=length;
    } else {
        if(v.tag!=5) { nrg_fail(p,NRG_TYPE);return false; }
        NmsView view;if(!nrg_core(p,nms_view(&p->heap,v.payload,&view)))return false;r.payload=view.length;
    }
    return nrg_replace(p,1,&r);
}
bool nrg_character(NrgInstance *p) {
    NmsValue value,index,r={0,1};int64_t ch;
    if(!nrg_string_input(p,1,&value) || !nrg_peek(p,0,&index) ||
       !nrg_core(p,nms_char_at(&p->heap,value.payload,index.payload,index.tag,&ch)))return false;
    r.payload=(uint64_t)ch;return nrg_replace(p,2,&r);
}
bool nrg_predicate(NrgInstance *p,uint32_t predicate) {
    NmsValue a,b,r={0,4};uint32_t answer;
    if(!nrg_string_input(p,1,&a) || !nrg_string_input(p,0,&b) ||
       !nrg_core(p,nms_predicate(&p->heap,a.payload,b.payload,predicate,&answer)))return false;
    r.payload=answer;return nrg_replace(p,2,&r);
}

static bool nrg_observed_value(const NrgInstance *p,bool global,uint32_t index,
    const uint32_t *path,uint16_t count,NmsValue *out) {
    if(!nrg_owner(p) || p->busy || count>257 || (count && !path) ||
       (global?index>=p->program->global_count:(!p->has_result || index)))return false;
    NmsValue value=global?p->globals[index]:p->result;
    if(!nrg_valid(p,value))return false;
    for(uint16_t i=0;i<count;i++) {
        if(value.tag!=7 && value.tag!=8)return false;
        uint32_t slot;if(slot_for(&p->heap,value.payload,&slot)!=NMS_OK)return false;
        const NmsSlot *s=&p->heap.slots[slot];
        if(path[i]>=s->length)return false;
        value=slot_value(s,path[i]);
        if(!nrg_valid(p,value))return false;
    }
    *out=value;return true;
}
bool nrg_observe(const NrgInstance *p,bool global,uint32_t index,const uint32_t *path,
    uint16_t count,NrgObservation *out) {
    NmsValue v;if(!out || !nrg_observed_value(p,global,index,path,count,&v))return false;
    NrgObservation o={p->epoch,0,v.payload,v.tag,0,UINT32_MAX,0};
    if(v.tag==5) {
        NmsView view;if(nms_view(&p->heap,v.payload,&view)!=NMS_OK)return false;
        o.identity=v.payload;o.scalar_bits=0;o.length=view.length;
    } else if(v.tag==7 || v.tag==8) {
        uint32_t slot;if(slot_for(&p->heap,v.payload,&slot)!=NMS_OK)return false;
        const NmsSlot *s=&p->heap.slots[slot];
        o.identity=v.payload;o.scalar_bits=0;o.length=s->length;
        if(v.tag==7)o.element=s->element_tag;
        else o.layout=p->program->records[s->record_ordinal].global_layout_index;
    }
    *out=o;return true;
}
bool nrg_string(const NrgInstance *p,bool global,uint32_t index,const uint32_t *path,
    uint16_t count,uint32_t offset,uint32_t length,void *out) {
    NmsValue v;NmsView view;
    if((length && !out) || !nrg_observed_value(p,global,index,path,count,&v) || v.tag!=5 ||
       nms_view(&p->heap,v.payload,&view)!=NMS_OK || offset>view.length || length>view.length-offset)return false;
    if(length)copy_bytes(out,view.data+offset,length);
    return true;
}
#endif
