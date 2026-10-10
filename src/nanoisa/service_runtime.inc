#include "service_runtime_catalog.inc"
typedef struct {
    SERVICE_TYPE(RuntimeView) view;
    CoreValue owner;
    uint32_t reference;
    const SERVICE_TYPE(IndirectHostedPlan) *callable_plan;
    uint32_t callable_target;
} FileRuntimeValue;
typedef struct {
    bool live, formal, shared;
    uint32_t owner_root, origin, formal_root;
    uint64_t region;
    CoreBorrow borrow;
} FileRuntimeReference;
/* Reserved concrete adapter bookkeeping, not an asserted NanoValue/C ABI. */
typedef struct {
    uint32_t function, instruction, locals_base, stack_base, stack_count;
    uint32_t staging_base, reference_base, region_base;
    bool waiting, instruction_open;
    uint8_t variant;
    uint32_t selected_target;
} FileRuntimeFrame;
typedef enum { FR_ACYCLIC, FR_CYCLIC, FR_INDIRECT } FileRuntimeKind;
typedef struct { CoreValue owners[SERVICE_CONST(FLOW_OWNERS)]; uint32_t roots[SERVICE_CONST(FLOW_OWNERS)]; } FileRuntimeWitness;
typedef enum { FR_READY, FR_ACTIVE, FR_TERMINAL } FileRuntimePhase;
struct SERVICE_TYPE(Runtime) {
    SERVICE_TYPE(HostedPlan) *plan;
    SERVICE_TYPE(CyclicHostedPlan) *cyclic_plan;
    SERVICE_TYPE(IndirectHostedPlan) *indirect_plan;
    FileRuntimeKind kind;
    FileRuntimeWitness *witness;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted, transfer_active;
    SERVICE_TYPE(HostedStartup) startup;
    SERVICE_TYPE(RuntimeStorage) storage;
    SERVICE_TYPE(FlowDeclaration) types[RUNTIME_TYPE_CAPACITY];
    RUNTIME_CONTEXT_MEMBERS
    FileRuntimeValue *values;
    FileRuntimeReference *references;
    uint64_t *regions;
    FileRuntimeFrame *frames;
    CoreValues *files;
    SERVICE_TYPE(RuntimeReport) report;
    SERVICE_TYPE(RuntimeView) result;
    FileRuntimePhase phase;
    SERVICE_TYPE(RuntimeMode) mode;
    uint16_t frame_count;
    bool busy, acquired, complete;
    uint32_t current_root, region_count;
    uint64_t next_region;
};
static SERVICE_TYPE(FlowDeclaration) fr_scalar_type(uint8_t tag) {
    return (SERVICE_TYPE(FlowDeclaration)){tag,0,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX,SERVICE_CONST(CATEGORY_UNKNOWN)};
}
static bool fr_same(SERVICE_TYPE(FlowDeclaration) a,SERVICE_TYPE(FlowDeclaration) b) {
    return a.tag==b.tag && a.mode==b.mode && a.global_index==b.global_index &&
           a.catalog_ordinal==b.catalog_ordinal && a.category==b.category;
}
static SERVICE_TYPE(RuntimeStatus) fr_error(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeStatus) s,CoreStatus core) {
    if(c && c->phase!=FR_TERMINAL && c->report.status==SERVICE_CONST(RUNTIME_OK) && s!=SERVICE_CONST(RUNTIME_BUSY)) {
        c->report.status=s;c->report.core_status=core;
    }
    return s;
}
static SERVICE_TYPE(RuntimeStatus) fr_ready(SERVICE_TYPE(Runtime) *c) {
    if(!c)return SERVICE_CONST(RUNTIME_INVALID);
    if(c->busy)return SERVICE_CONST(RUNTIME_BUSY);
    if(c->phase!=FR_ACTIVE)return SERVICE_CONST(RUNTIME_STATE);
    if(c->report.status!=SERVICE_CONST(RUNTIME_OK))return c->report.status;
    if(c->complete)return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    return SERVICE_CONST(RUNTIME_OK);
}
static SERVICE_TYPE(RuntimeStatus) fr_effect(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(RuntimeStatus) status=fr_ready(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(c->kind!=FR_ACYCLIC && !c->transfer_active &&
       (!c->frame_count || c->frames[c->frame_count-1].waiting ||
        !c->frames[c->frame_count-1].instruction_open))
        return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    return SERVICE_CONST(RUNTIME_OK);
}
static SERVICE_TYPE(RuntimeStatus) fr_core(SERVICE_TYPE(Runtime) *c,CoreStatus s) {
    SERVICE_TYPE(RuntimeStatus) result;
    switch(s) {
    case CORE_STATUS(OK):return SERVICE_CONST(RUNTIME_OK);
    case CORE_STATUS(ARGUMENT):result=SERVICE_CONST(RUNTIME_INVALID);break;
    case CORE_STATUS(TYPE):result=SERVICE_CONST(RUNTIME_TYPE);break;
    case CORE_STATUS(STALE):result=SERVICE_CONST(RUNTIME_STALE);break;
    case CORE_STATUS(BORROWED):result=SERVICE_CONST(RUNTIME_BORROWED);break;
    case CORE_STATUS(LIMIT):result=SERVICE_CONST(RUNTIME_LIMIT);break;
    case CORE_STATUS(MEMORY):result=SERVICE_CONST(RUNTIME_MEMORY);break;
    default:result=SERVICE_CONST(RUNTIME_STATE);break;
    }
    return fr_error(c,result,s);
}
static SERVICE_TYPE(RuntimeStatus) fr_prepare(SERVICE_TYPE(FlowStatus) s) {
    switch(s) {
    case SERVICE_CONST(FLOW_OK):return SERVICE_CONST(RUNTIME_OK);
    case SERVICE_CONST(FLOW_INVALID):return SERVICE_CONST(RUNTIME_INVALID);
    case SERVICE_CONST(FLOW_LIMIT):return SERVICE_CONST(RUNTIME_LIMIT);
    case SERVICE_CONST(FLOW_MEMORY):return SERVICE_CONST(RUNTIME_MEMORY);
    default:return SERVICE_CONST(RUNTIME_UNRESOLVED);
    }
}
static bool fr_add(size_t *total,size_t count,size_t width) {
    if(count && width>SIZE_MAX/count)return false;
    size_t n=count*width;
    if(*total>SERVICE_CONST(RUNTIME_BYTES) || n>SERVICE_CONST(RUNTIME_BYTES)-*total)return false;
    *total+=n;return true;
}
static bool fr_empty(const SERVICE_TYPE(Runtime) *c,uint32_t root) {
    return root<c->storage.values && !c->values[root].view.initialized;
}
static bool fr_live(const SERVICE_TYPE(Runtime) *c,uint32_t root) {
    return root<c->storage.values && c->values[root].view.initialized;
}
static void fr_clear(SERVICE_TYPE(Runtime) *c,uint32_t root) { c->values[root]=(FileRuntimeValue){0}; }
static bool fr_member(const SERVICE_TYPE(Runtime) *c,uint32_t context,const char *id,SERVICE_TYPE(FlowDeclaration) *out) {
    if(!id){*out=fr_scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=fr_scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=fr_scalar_type(TAG_BOOL);return true;}
    (void)c;(void)context;
    uint32_t base=RUNTIME_TYPE_BASE(context);
    for(uint32_t i=base;i<base+RUNTIME_TYPE_COUNT(c,context);i++) {
        const NlServicePlanType *t=RUNTIME_CATALOG_TYPE(c,i);
        if(t && !strcmp(t->id,id)){*out=c->types[i];return true;}
    }
    return false;
}
static SERVICE_TYPE(RuntimeView) fr_view(SERVICE_TYPE(FlowDeclaration) type) {
    SERVICE_TYPE(RuntimeView) v={0};v.initialized=true;v.type=type;
    v.owning=type.category==NOMINAL_CATEGORY_OWNER || type.category==NOMINAL_CATEGORY_OWNER_RESULT;
    return v;
}
#include RUNTIME_VALUE_ADAPTER
#include "service_cyclic_runtime_facts.inc"
#include "service_indirect_runtime_create.inc"

SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_create)(const uint8_t *bytes,size_t size,SERVICE_TYPE(RuntimeMode) mode,SERVICE_TYPE(Runtime) **out) {
    if(!out || (mode!=SERVICE_CONST(RUNTIME_VM) && mode!=SERVICE_CONST(RUNTIME_NATIVE)))return SERVICE_CONST(RUNTIME_INVALID);
    SERVICE_TYPE(HostedPlan) *plan=NULL;SERVICE_TYPE(FlowStatus) checked=SERVICE_FN(hosted_prepare)(bytes,size,&plan);
    if(checked!=SERVICE_CONST(FLOW_OK))return fr_prepare(checked);
    SERVICE_TYPE(HostedStartup) startup;size_t core=0,total;
    if(!SERVICE_FN(hosted_startup)(plan,&startup) || !RUNTIME_BOUND(&core)) {
        SERVICE_FN(hosted_free)(plan);return SERVICE_CONST(RUNTIME_INVALID);
    }
    uint64_t values=mode==SERVICE_CONST(RUNTIME_VM)?startup.vm_value_slots:startup.native_value_slots;
    total=startup.allocation_bound;
    if(!values || values>UINT32_MAX || !startup.frames || !startup.reference_slots || !startup.region_slots ||
       !fr_add(&total,1,sizeof(SERVICE_TYPE(Runtime))) || !fr_add(&total,(size_t)values,sizeof(FileRuntimeValue)) ||
       !fr_add(&total,startup.reference_slots,sizeof(FileRuntimeReference)) ||
       !fr_add(&total,startup.region_slots,sizeof(uint64_t)) ||
       !fr_add(&total,startup.frames,sizeof(FileRuntimeFrame)) || !fr_add(&total,1,core)) {
        SERVICE_FN(hosted_free)(plan);return SERVICE_CONST(RUNTIME_LIMIT);
    }
    SERVICE_TYPE(Runtime) *c=calloc(1,sizeof *c);
    if(!c){SERVICE_FN(hosted_free)(plan);return SERVICE_CONST(RUNTIME_MEMORY);}
    c->plan=plan;c->startup=startup;c->next_region=1;c->mode=mode;
    c->storage=(SERVICE_TYPE(RuntimeStorage)){total,(uint32_t)values,startup.reference_slots,startup.region_slots,startup.frames};
    c->values=calloc(c->storage.values,sizeof *c->values);
    c->references=calloc(c->storage.references,sizeof *c->references);
    c->regions=calloc(c->storage.regions,sizeof *c->regions);
    c->frames=calloc(c->storage.frames,sizeof *c->frames);
    SERVICE_TYPE(RuntimeStatus) status=SERVICE_CONST(RUNTIME_MEMORY);
    if(!c->values || !c->references || !c->regions || !c->frames)goto fail;
    status=SERVICE_CONST(RUNTIME_INVALID);
    RUNTIME_LOAD_TYPES(c,SERVICE_FN(hosted_type),plan);
    c->report.function=c->report.instruction=NVM_V2_NO_INDEX;
    c->current_root=startup.initializer!=NVM_V2_NO_INDEX?startup.initializer:startup.entry;
    *out=c;return SERVICE_CONST(RUNTIME_OK);
fail:
    free(c->frames);free(c->regions);free(c->references);free(c->values);SERVICE_FN(hosted_free)(plan);free(c);return status;
}
bool SERVICE_FN(runtime_storage)(const SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeStorage) *out) {
    if(!c || !out)return false;
    *out=c->storage;return true;
}
const SERVICE_TYPE(HostedPlan) *SERVICE_FN(runtime_plan)(const SERVICE_TYPE(Runtime) *c) { return c?c->plan:NULL; }
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_begin)(SERVICE_TYPE(Runtime) *c) {
    if(!c)return SERVICE_CONST(RUNTIME_INVALID);
    if(c->busy || c->phase==FR_ACTIVE)return SERVICE_CONST(RUNTIME_BUSY);
    if(c->phase!=FR_READY)return SERVICE_CONST(RUNTIME_STATE);
    c->busy=true;CoreStatus core=RUNTIME_BEGIN(c);c->busy=false;c->phase=FR_ACTIVE;
    if(core!=CORE_STATUS(OK))return fr_core(c,core);
    c->acquired=true;c->report.acquired=true;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_site)(SERVICE_TYPE(Runtime) *c,uint32_t function,uint16_t instruction) {
    SERVICE_TYPE(RuntimeStatus) status=fr_ready(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    uint8_t variant=0;
    if(c->kind!=FR_ACYCLIC) {
        if(!c->frame_count || c->frames[c->frame_count-1].function!=function ||
           c->frames[c->frame_count-1].instruction!=instruction)
            return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
        variant=c->frames[c->frame_count-1].variant;
    }
    if(!fr_instruction(c,function,instruction,variant,&in,&fact) || !fact.reachable)
        return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
    c->report.function=function;c->report.instruction=in.byte_offset;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_fail)(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeStatus) status) {
    SERVICE_TYPE(RuntimeStatus) ready=fr_ready(c);if(ready!=SERVICE_CONST(RUNTIME_OK))return ready;
    if(status<=SERVICE_CONST(RUNTIME_OK) || status>SERVICE_CONST(RUNTIME_CLEANUP))return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
    return fr_error(c,status,CORE_STATUS(STATE));
}
bool SERVICE_FN(runtime_current_root)(const SERVICE_TYPE(Runtime) *c,uint32_t *out) {
    if(!c || !out || c->phase!=FR_ACTIVE || c->complete || c->busy)return false;
    *out=c->current_root;return true;
}
bool SERVICE_FN(runtime_view)(const SERVICE_TYPE(Runtime) *c,uint32_t root,SERVICE_TYPE(RuntimeView) *out) {
    if(!c || !out || root>=c->storage.values || c->busy)return false;
    *out=c->values[root].view;return true;
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_scalar)(SERVICE_TYPE(Runtime) *c,uint32_t dst,uint8_t tag,int64_t value) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_empty(c,dst) || (tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_VOID) ||
       (tag==TAG_BOOL && value!=0 && value!=1) || (tag==TAG_VOID && value))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    SERVICE_TYPE(RuntimeView) v=fr_view(fr_scalar_type(tag));v.fields=tag==TAG_VOID?0:1;v.values[0]=value;
    c->values[dst].view=v;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_copy)(SERVICE_TYPE(Runtime) *c,uint32_t src,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_live(c,src) || !fr_empty(c,dst) || c->values[src].view.owning || c->values[src].view.formal)
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    c->values[dst]=c->values[src];return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_move)(SERVICE_TYPE(Runtime) *c,uint32_t src,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_live(c,src) || !fr_empty(c,dst) || c->values[src].view.formal)
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(c->values[src].view.owning) {
        c->busy=true;CoreStatus core=CORE_FN(value_move)(c->files,&c->values[src].owner,&c->values[dst].owner);c->busy=false;
        if(core!=CORE_STATUS(OK))return fr_core(c,core);
    }
    c->values[dst].view=c->values[src].view;
    c->values[dst].callable_plan=c->values[src].callable_plan;
    c->values[dst].callable_target=c->values[src].callable_target;fr_clear(c,src);return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_drop)(SERVICE_TYPE(Runtime) *c,uint32_t root) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_live(c,root) || c->values[root].view.formal)return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(c->values[root].view.owning) {
        c->busy=true;CoreStatus core=CORE_FN(value_drop)(c->files,&c->values[root].owner);c->busy=false;
        if(core!=CORE_STATUS(OK))return fr_core(c,core);
    }
    fr_clear(c,root);return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_construct)(SERVICE_TYPE(Runtime) *c,uint32_t ordinal,uint16_t variant,const uint32_t *inputs,size_t count,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!RUNTIME_TYPE_VALID(c,ordinal) || dst>=c->storage.values || count>SERVICE_CONST(RUNTIME_FIELDS) || (count && !inputs))
        return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
    SERVICE_TYPE(FlowDeclaration) type=c->types[ordinal];bool record=type.category==SERVICE_CONST(CATEGORY_RECORD);
    const NlServicePlanType *catalog=RUNTIME_CATALOG_TYPE(c,ordinal);
    if(!catalog || (type.category!=SERVICE_CONST(CATEGORY_RECORD) && type.category!=SERVICE_CONST(CATEGORY_SCALAR_RESULT)) ||
       (record?variant!=0:variant>=catalog->member_count) || count!=(record?catalog->member_count:1))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    SERVICE_TYPE(RuntimeView) value=fr_view(type);bool replaces=false;
    if(!record)value.arm=variant?SERVICE_CONST(FLOW_ARM_ERROR):SERVICE_CONST(FLOW_ARM_OK);
    for(size_t i=0;i<count;i++) {
        for(size_t j=0;j<i;j++)if(inputs[j]==inputs[i])return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
        if(!fr_live(c,inputs[i]))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        SERVICE_TYPE(RuntimeView) input=c->values[inputs[i]].view;SERVICE_TYPE(FlowDeclaration) expected;
        if(input.owning || input.formal || !fr_member(c,ordinal,catalog->members[record?i:variant].type_id,&expected) || !fr_same(input.type,expected))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        replaces|=inputs[i]==dst;
        if(record) {
            if(input.fields!=1 || (input.type.tag!=TAG_INT && input.type.tag!=TAG_BOOL))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
            value.values[i]=input.values[0];value.fields=(uint8_t)count;
        } else { value.fields=input.fields;memcpy(value.values,input.values,sizeof value.values); }
    }
    if(!replaces && !fr_empty(c,dst))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    for(size_t i=0;i<count;i++)fr_clear(c,inputs[i]);
    c->values[dst].view=value;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_project)(SERVICE_TYPE(Runtime) *c,uint32_t src,uint16_t field,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_live(c,src) || (src!=dst && !fr_empty(c,dst)))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    SERVICE_TYPE(RuntimeView) input=c->values[src].view;uint32_t ordinal=input.type.catalog_ordinal;
    if(input.owning || input.formal || !RUNTIME_TYPE_VALID(c,ordinal))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    const NlServicePlanType *catalog=RUNTIME_CATALOG_TYPE(c,ordinal);SERVICE_TYPE(FlowDeclaration) type;
    SERVICE_TYPE(RuntimeView) result={0};
    if(input.type.category==SERVICE_CONST(CATEGORY_RECORD)) {
        if(!catalog || field>=catalog->member_count || field>=input.fields || !fr_member(c,ordinal,catalog->members[field].type_id,&type))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        result=fr_view(type);result.fields=1;result.values[0]=input.values[field];
    } else if(input.type.category==SERVICE_CONST(CATEGORY_SCALAR_RESULT) && !field &&
              (input.arm==SERVICE_CONST(FLOW_ARM_OK) || input.arm==SERVICE_CONST(FLOW_ARM_ERROR))) {
        if(!catalog || !fr_member(c,ordinal,catalog->members[input.arm==SERVICE_CONST(FLOW_ARM_ERROR)].type_id,&type))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        result=fr_view(type);result.fields=input.fields;memcpy(result.values,input.values,sizeof result.values);
    } else return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    fr_clear(c,src);c->values[dst].view=result;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_result_arm)(SERVICE_TYPE(Runtime) *c,uint32_t root,SERVICE_TYPE(FlowArm) *out) {
    SERVICE_TYPE(RuntimeStatus) status=fr_ready(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!out || !fr_live(c,root))return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
    SERVICE_TYPE(RuntimeView) *view=&c->values[root].view;SERVICE_TYPE(FlowArm) arm;
    if(view->type.category==NOMINAL_CATEGORY_OWNER_RESULT) {
        CoreOpenView actual;c->busy=true;CoreStatus core=CORE_FN(open_view)(c->files,&c->values[root].owner,&actual);c->busy=false;
        if(core!=CORE_STATUS(OK))return fr_core(c,core);
        arm=actual.ok?SERVICE_CONST(FLOW_ARM_OK):SERVICE_CONST(FLOW_ARM_ERROR);
    } else if(view->type.category==SERVICE_CONST(CATEGORY_SCALAR_RESULT))arm=view->arm;
    else return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(arm!=SERVICE_CONST(FLOW_ARM_OK) && arm!=SERVICE_CONST(FLOW_ARM_ERROR))return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    *out=arm;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_take)(SERVICE_TYPE(Runtime) *c,uint32_t src,SERVICE_TYPE(FlowArm) arm,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_live(c,src) || !fr_empty(c,dst) || (arm!=SERVICE_CONST(FLOW_ARM_OK) && arm!=SERVICE_CONST(FLOW_ARM_ERROR)))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    SERVICE_TYPE(FlowArm) actual;status=SERVICE_FN(runtime_result_arm)(c,src,&actual);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(actual!=arm)return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    SERVICE_TYPE(RuntimeView) input=c->values[src].view,value={0};
    uint32_t ordinal=input.type.catalog_ordinal,base=RUNTIME_TYPE_BASE(ordinal);
    if(input.type.category==NOMINAL_CATEGORY_OWNER_RESULT) {
        CoreStatus core;
        if(arm==SERVICE_CONST(FLOW_ARM_OK)) {
            value=fr_view(c->types[base]);c->busy=true;
            core=CORE_FN(open_take_ok)(c->files,&c->values[src].owner,&c->values[dst].owner);c->busy=false;
        } else {
            CoreOpenView observed;c->busy=true;core=CORE_FN(open_view)(c->files,&c->values[src].owner,&observed);c->busy=false;
            if(core!=CORE_STATUS(OK))return fr_core(c,core);
            value=fr_view(c->types[base+1]);value.fields=RUNTIME_ERROR_FIELDS(c,ordinal);
            if(!fr_error_payload(RUNTIME_ERROR(observed),value.values))return fr_error(c,SERVICE_CONST(RUNTIME_LIMIT),CORE_STATUS(LIMIT));
            CoreResult error;c->busy=true;core=CORE_FN(open_take_error)(c->files,&c->values[src].owner,&error);c->busy=false;
        }
        if(core!=CORE_STATUS(OK))return fr_core(c,core);
    } else {
        const NlServicePlanType *catalog=RUNTIME_CATALOG_TYPE(c,input.type.catalog_ordinal);SERVICE_TYPE(FlowDeclaration) type;
        if(!catalog || !fr_member(c,ordinal,catalog->members[arm==SERVICE_CONST(FLOW_ARM_ERROR)].type_id,&type))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
        value=fr_view(type);value.fields=input.fields;memcpy(value.values,input.values,sizeof value.values);
    }
    fr_clear(c,src);c->values[dst].view=value;return SERVICE_CONST(RUNTIME_OK);
}
static bool fr_reference(const SERVICE_TYPE(Runtime) *c,uint32_t index) {
    if(index>=c->storage.references)return false;
    const FileRuntimeReference *r=&c->references[index];
    if(!r->live || r->origin>=c->storage.references)return false;
    const FileRuntimeReference *origin=&c->references[r->origin];
    return origin->live && !origin->formal && origin->origin==r->origin &&
           origin->shared==r->shared && RUNTIME_BORROW_EPOCH(origin->borrow)==RUNTIME_BORROW_EPOCH(r->borrow) && origin->owner_root==r->owner_root &&
           fr_live(c,r->owner_root) && c->values[r->owner_root].view.owning &&
           c->values[r->owner_root].view.type.category==NOMINAL_CATEGORY_OWNER;
}
static bool fr_aliases(const SERVICE_TYPE(Runtime) *c,uint32_t origin) {
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && c->references[i].formal && c->references[i].origin==origin)return true;
    return false;
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_region_begin)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(c->region_count==c->storage.regions || c->next_region==UINT64_MAX)return fr_error(c,SERVICE_CONST(RUNTIME_LIMIT),CORE_STATUS(LIMIT));
    c->regions[c->region_count++]=c->next_region++;return SERVICE_CONST(RUNTIME_OK);
}
static SERVICE_TYPE(RuntimeStatus) file_runtime_borrow_mode(SERVICE_TYPE(Runtime) *c,uint32_t owner,uint32_t reference,bool shared) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_live(c,owner) || c->values[owner].view.type.category!=NOMINAL_CATEGORY_OWNER ||
       !c->values[owner].view.owning || !c->region_count || reference>=c->storage.references || c->references[reference].live)
        return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    FileRuntimeReference r={0};r.owner_root=owner;r.origin=reference;r.formal_root=SERVICE_CONST(RUNTIME_NO_SLOT);
    r.region=c->regions[c->region_count-1];r.shared=shared;
    if(shared)for(uint32_t i=0;i<c->storage.references;i++) {
        const FileRuntimeReference *prior=&c->references[i];
        if(!prior->live || prior->owner_root!=owner)continue;
        if(!prior->shared || !fr_reference(c,i))return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
        r.origin=prior->origin;r.borrow=prior->borrow;r.live=true;
        c->references[reference]=r;return SERVICE_CONST(RUNTIME_OK);
    }
    c->busy=true;
    CoreStatus core=CORE_FN(value_borrow)(c->files,&c->values[owner].owner,&r.borrow);c->busy=false;
    if(core!=CORE_STATUS(OK))return fr_core(c,core);
    r.live=true;c->references[reference]=r;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_borrow)(SERVICE_TYPE(Runtime) *c,uint32_t owner,uint32_t reference) {
    return file_runtime_borrow_mode(c,owner,reference,false);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_borrow_shared)(SERVICE_TYPE(Runtime) *c,uint32_t owner,uint32_t reference) {
    return file_runtime_borrow_mode(c,owner,reference,true);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_bind_formal)(SERVICE_TYPE(Runtime) *c,uint32_t source,uint32_t root,uint32_t reference) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_reference(c,source) || !fr_empty(c,root) || reference>=c->storage.references || c->references[reference].live)
        return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    FileRuntimeReference r=c->references[source];r.formal=true;r.formal_root=root;r.region=0;
    SERVICE_TYPE(RuntimeView) value=fr_view(c->values[r.owner_root].view.type);value.type.mode=r.shared?1:2;value.owning=false;value.formal=true;
    c->references[reference]=r;c->values[root].view=value;c->values[root].reference=reference;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_end_reference)(SERVICE_TYPE(Runtime) *c,uint32_t reference) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_reference(c,reference))return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    FileRuntimeReference *r=&c->references[reference];
    if(r->formal) {
        if(!fr_live(c,r->formal_root) || !c->values[r->formal_root].view.formal || c->values[r->formal_root].reference!=reference)
            return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
        fr_clear(c,r->formal_root);
    } else {
        uint32_t replacement=SERVICE_CONST(RUNTIME_NO_SLOT);
        if(r->shared)for(uint32_t i=0;i<c->storage.references;i++)
            if(i!=reference && c->references[i].live && !c->references[i].formal &&
               c->references[i].shared && c->references[i].origin==r->origin) {replacement=i;break;}
        if(replacement!=SERVICE_CONST(RUNTIME_NO_SLOT)) {
            if(r->origin==reference)for(uint32_t i=0;i<c->storage.references;i++)
                if(c->references[i].live && c->references[i].origin==reference)c->references[i].origin=replacement;
        } else {
            if(fr_aliases(c,r->origin))return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
            c->busy=true;CoreStatus core=CORE_FN(value_end_borrow)(c->files,&r->borrow);c->busy=false;
            if(core!=CORE_STATUS(OK))return fr_core(c,core);
        }
    }
    *r=(FileRuntimeReference){0};return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_region_end)(SERVICE_TYPE(Runtime) *c) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!c->region_count)return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    uint64_t region=c->regions[c->region_count-1];
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && !c->references[i].formal &&
       c->references[i].region==region && fr_aliases(c,i))return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && !c->references[i].formal && c->references[i].region==region) {
        status=SERVICE_FN(runtime_end_reference)(c,i);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    }
    c->regions[--c->region_count]=0;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_service)(SERVICE_TYPE(Runtime) *c,uint32_t import,uint32_t reference,uint32_t input,uint32_t output) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    uint32_t ordinal;
    if(!fr_empty(c,output) || !fr_import(c,import,&ordinal) || ordinal>=RUNTIME_IMPORTS(c))
        return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
    if(c->kind!=FR_ACYCLIC) {
        const FileRuntimeFrame *frame=&c->frames[c->frame_count-1];
        SERVICE_TYPE(CodeInstruction) instruction;SERVICE_TYPE(BodyInstruction) fact;
        if(!fr_instruction(c,frame->function,(uint16_t)frame->instruction,frame->variant,&instruction,&fact) ||
           instruction.decoded.opcode!=OP_FILE_SERVICE || instruction.decoded.operands[0].u32!=import ||
           instruction.catalog_ordinal!=ordinal)
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    }
    uint32_t method=RUNTIME_METHOD_LOCAL(ordinal),base=RUNTIME_METHOD_OWNER(ordinal);
    bool endpoint=RUNTIME_ENDPOINT(c,ordinal);
    bool borrows=method>=1 && method<=3;
    if((borrows && (!fr_reference(c,reference) || c->references[reference].shared)) || (!borrows && reference!=SERVICE_CONST(RUNTIME_NO_SLOT)))
        return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    if(borrows && !fr_same(c->values[c->references[reference].owner_root].view.type,c->types[base]))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(endpoint && method==0) {
        if(!fr_live(c,input) || !fr_same(c->values[input].view.type,c->types[RUNTIME_ENDPOINT_TYPE(ordinal)]) ||
           c->values[input].view.owning || c->values[input].view.formal || c->values[input].view.fields!=7)
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    } else if(method==1) {
        if(!fr_live(c,input) || !fr_same(c->values[input].view.type,fr_scalar_type(TAG_INT)))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    } else if(method==4) {
        if(!fr_live(c,input) || !c->values[input].view.owning || !fr_same(c->values[input].view.type,c->types[base]))
            return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    } else if(input!=SERVICE_CONST(RUNTIME_NO_SLOT))return fr_error(c,SERVICE_CONST(RUNTIME_INVALID),CORE_STATUS(ARGUMENT));
    /* Empty output is already part of the counted arena. Core publication and
     * the remaining metadata assignment cannot allocate. Busy blocks nested
     * cleanup while a fixture/host shim is inside the core call. */
    c->busy=true;CoreStatus core;CoreScalarResult scalar={0};
#ifdef RUNTIME_MIXED
    core=fr_mixed_service(c,ordinal,reference,input,output,&scalar);
#else
    switch(ordinal) {
    case 0:core=fr_acquire(c->files,RUNTIME_HAS_ENDPOINT?&c->values[input].view:NULL,&c->values[output].owner);break;
    case 1:core=CORE_FN(value_write_byte)(c->files,&c->references[reference].borrow,c->values[input].view.values[0],&scalar);break;
    case 2:core=CORE_FN(value_rewind)(c->files,&c->references[reference].borrow,&scalar);break;
    case 3:core=CORE_FN(value_read_byte)(c->files,&c->references[reference].borrow,&scalar);break;
    default:core=CORE_FN(value_close)(c->files,&c->values[input].owner,&scalar);break;
    }
#endif
    c->busy=false;
    if(core!=CORE_STATUS(OK))return fr_core(c,core);
    if(!method) {
        if(endpoint)fr_clear(c,input);
        c->values[output].view=fr_view(c->types[base+3]);return SERVICE_CONST(RUNTIME_OK);
    }
    /* Accepted close consumes on either Result arm. Never leave its old root
     * apparently live while checking the passive result representation. */
    if(method==4)fr_clear(c,input);
    SERVICE_TYPE(RuntimeView) result;
    if(!fr_scalar_result(c,ordinal,scalar,&result))return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    if(method==1)fr_clear(c,input);
    c->values[output].view=result;return SERVICE_CONST(RUNTIME_OK);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_complete_root)(SERVICE_TYPE(Runtime) *c,uint32_t root) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(c->kind!=FR_ACYCLIC) {
        SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
        FileRuntimeFrame *f=&c->frames[c->frame_count-1];
        if(c->frame_count!=1 || !fr_instruction(c,f->function,(uint16_t)f->instruction,f->variant,&in,&fact) ||
           in.decoded.opcode!=OP_RET || !fact.exit_checked)
            return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    }
    bool initializer=c->current_root==c->startup.initializer;
    SERVICE_TYPE(HostedFunction) fn;
    if(!fr_function(c,c->current_root,&fn))return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    if(initializer) {
        if(root!=SERVICE_CONST(RUNTIME_NO_SLOT) || fn.code.declaration.result_count)return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    } else if(!fr_live(c,root) || !fr_same(c->values[root].view.type,fn.code.declaration.result) ||
              c->values[root].view.owning || c->values[root].view.formal ||
              (c->values[root].view.type.tag!=TAG_INT && c->values[root].view.type.tag!=TAG_BOOL))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(c->region_count)return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live)return fr_error(c,SERVICE_CONST(RUNTIME_BORROWED),CORE_STATUS(BORROWED));
    for(uint32_t i=0;i<c->storage.values;i++)if(c->values[i].view.initialized &&
       (c->values[i].view.owning || c->values[i].view.formal))return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    if(initializer) {
        CoreFinish observed;
        if(!CORE_FN(values_report)(c->files,&observed))return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
        if(observed.execution!=CORE_STATUS(OK) || observed.cleanup_failures)
            return fr_error(c,SERVICE_CONST(RUNTIME_CLEANUP),observed.execution);
    }
    if(!initializer)c->result=c->values[root].view;
    memset(c->values,0,(size_t)c->storage.values*sizeof *c->values);
    memset(c->frames,0,(size_t)c->storage.frames*sizeof *c->frames);c->frame_count=0;
    if(initializer)c->current_root=c->startup.entry;
    else c->complete=true;
    return SERVICE_CONST(RUNTIME_OK);
}
static SERVICE_TYPE(RuntimeReport) fr_refused(SERVICE_TYPE(RuntimeStatus) status) {
    SERVICE_TYPE(RuntimeReport) report={0};report.status=status;
    report.function=report.instruction=NVM_V2_NO_INDEX;return report;
}
static SERVICE_TYPE(RuntimeReport) fr_finish(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeView) *out) {
    if(!c)return fr_refused(SERVICE_CONST(RUNTIME_INVALID));
    if(c->busy)return fr_refused(SERVICE_CONST(RUNTIME_BUSY));
    if(c->phase==FR_TERMINAL) {
        if(c->report.status==SERVICE_CONST(RUNTIME_OK) && out)*out=c->result;
        return c->report;
    }
    if(!c->complete && c->report.status==SERVICE_CONST(RUNTIME_OK))fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    c->busy=true;
    /* Formal references never release caller epochs. Clear aliases before the
     * originating borrows, then attempt every root before terminal disposal. */
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && c->references[i].formal) {
        uint32_t root=c->references[i].formal_root;
        if(root<c->storage.values && c->values[root].view.formal && c->values[root].reference==i)fr_clear(c,root);
        else fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
        c->references[i]=(FileRuntimeReference){0};
    }
    if(c->acquired) {
        for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live) {
            if(!c->references[i].shared || c->references[i].origin==i) {
                CoreStatus core=CORE_FN(value_end_borrow)(c->files,&c->references[i].borrow);
                if(core!=CORE_STATUS(OK))fr_core(c,core);
            }
            c->references[i]=(FileRuntimeReference){0};
        }
        for(uint32_t i=0;i<c->storage.values;i++) {
            if(c->values[i].view.initialized && c->values[i].view.owning) {
                CoreStatus core=CORE_FN(value_drop)(c->files,&c->values[i].owner);
                if(core!=CORE_STATUS(OK))fr_core(c,core);
            }
            fr_clear(c,i);
        }
        CoreStatus execution=c->report.status==SERVICE_CONST(RUNTIME_OK)?CORE_STATUS(OK):
            c->report.core_status==CORE_STATUS(OK)?CORE_STATUS(STATE):c->report.core_status;
        c->report.cleanup=CORE_FN(values_destroy)(c->files,execution);c->files=NULL;
        if(c->report.status==SERVICE_CONST(RUNTIME_OK) &&
           (c->report.cleanup.execution!=CORE_STATUS(OK) || c->report.cleanup.cleanup_failures))
            fr_error(c,SERVICE_CONST(RUNTIME_CLEANUP),c->report.cleanup.execution);
    } else c->report.cleanup.execution=c->report.core_status;
    memset(c->regions,0,(size_t)c->storage.regions*sizeof *c->regions);c->region_count=0;
    c->frame_count=0;c->busy=false;c->phase=FR_TERMINAL;
    if(c->report.status==SERVICE_CONST(RUNTIME_OK) && out)*out=c->result;
    return c->report;
}
SERVICE_TYPE(RuntimeReport) SERVICE_FN(runtime_finish)(SERVICE_TYPE(Runtime) *c,SERVICE_TYPE(RuntimeView) *out) {
    if(c && c->kind!=FR_ACYCLIC)return fr_refused(SERVICE_CONST(RUNTIME_STATE));
    return fr_finish(c,out);
}
SERVICE_TYPE(RuntimeReport) SERVICE_FN(runtime_destroy)(SERVICE_TYPE(Runtime) **address,SERVICE_TYPE(RuntimeView) *out) {
    if(!address || !*address)return fr_refused(SERVICE_CONST(RUNTIME_INVALID));
    SERVICE_TYPE(Runtime) *c=*address;
    if(c->kind!=FR_ACYCLIC)return fr_refused(SERVICE_CONST(RUNTIME_STATE));
    if(c->busy)return fr_refused(SERVICE_CONST(RUNTIME_BUSY));
    SERVICE_TYPE(RuntimeReport) report=SERVICE_FN(runtime_finish)(c,out);
    free(c->frames);free(c->regions);free(c->references);free(c->values);SERVICE_FN(hosted_free)(c->plan);free(c);*address=NULL;
    return report;
}

#include "service_runtime_frames.inc"
#include "service_cyclic_runtime.inc"
#include "service_indirect_runtime.inc"
