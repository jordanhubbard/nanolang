#include "file_runtime.h"
#include "file_runtime_frames.h"
#include "file_cyclic_runtime.h"
#include "../nsi_file_values_internal.h"
#include "nvm_v2_sections.h"
#include "../nsi_file_catalog.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    NvmFileRuntimeView view;
    NlFileValue owner;
    uint32_t reference;
} FileRuntimeValue;
typedef struct {
    bool live, formal;
    uint32_t owner_root, origin, formal_root;
    uint64_t region;
    NlFileValueBorrow borrow;
} FileRuntimeReference;
/* Reserved concrete adapter bookkeeping, not an asserted NanoValue/C ABI. */
typedef struct {
    uint32_t function, instruction, locals_base, stack_base, stack_count;
    uint32_t staging_base, reference_base, region_base;
    bool waiting, instruction_open;
    uint8_t variant;
} FileRuntimeFrame;
typedef enum { FR_ACYCLIC, FR_CYCLIC } FileRuntimeKind;
typedef struct { NlFileValue owners[NVM_FILE_FLOW_OWNERS]; uint32_t roots[NVM_FILE_FLOW_OWNERS]; } FileRuntimeWitness;
typedef enum { FR_READY, FR_ACTIVE, FR_TERMINAL } FileRuntimePhase;
struct NvmFileRuntime {
    NvmFileHostedPlan *plan;
    NvmFileCyclicHostedPlan *cyclic_plan;
    FileRuntimeKind kind;
    FileRuntimeWitness *witness;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted, transfer_active;
    NvmFileHostedStartup startup;
    NvmFileRuntimeStorage storage;
    NvmFileFlowDeclaration types[NVM_FILE_NOMINAL_TYPES];
    FileRuntimeValue *values;
    FileRuntimeReference *references;
    uint64_t *regions;
    FileRuntimeFrame *frames;
    NlFileValues *files;
    NvmFileRuntimeReport report;
    NvmFileRuntimeView result;
    FileRuntimePhase phase;
    NvmFileRuntimeMode mode;
    uint16_t frame_count;
    bool busy, acquired, complete;
    uint32_t current_root, region_count;
    uint64_t next_region;
};
static NvmFileFlowDeclaration fr_scalar_type(uint8_t tag) {
    return (NvmFileFlowDeclaration){tag,0,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX,NVM_FILE_CATEGORY_UNKNOWN};
}
static bool fr_same(NvmFileFlowDeclaration a,NvmFileFlowDeclaration b) {
    return a.tag==b.tag && a.mode==b.mode && a.global_index==b.global_index &&
           a.catalog_ordinal==b.catalog_ordinal && a.category==b.category;
}
static NvmFileRuntimeStatus fr_error(NvmFileRuntime *c,NvmFileRuntimeStatus s,NlFileValueStatus core) {
    if(c && c->phase!=FR_TERMINAL && c->report.status==NVM_FILE_RUNTIME_OK && s!=NVM_FILE_RUNTIME_BUSY) {
        c->report.status=s;c->report.core_status=core;
    }
    return s;
}
static NvmFileRuntimeStatus fr_ready(NvmFileRuntime *c) {
    if(!c)return NVM_FILE_RUNTIME_INVALID;
    if(c->busy)return NVM_FILE_RUNTIME_BUSY;
    if(c->phase!=FR_ACTIVE)return NVM_FILE_RUNTIME_STATE;
    if(c->report.status!=NVM_FILE_RUNTIME_OK)return c->report.status;
    if(c->complete)return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    return NVM_FILE_RUNTIME_OK;
}
static NvmFileRuntimeStatus fr_effect(NvmFileRuntime *c) {
    NvmFileRuntimeStatus status=fr_ready(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(c->kind==FR_CYCLIC && !c->transfer_active &&
       (!c->frame_count || c->frames[c->frame_count-1].waiting ||
        !c->frames[c->frame_count-1].instruction_open))
        return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    return NVM_FILE_RUNTIME_OK;
}
static NvmFileRuntimeStatus fr_core(NvmFileRuntime *c,NlFileValueStatus s) {
    NvmFileRuntimeStatus result;
    switch(s) {
    case NL_FILE_VALUE_OK:return NVM_FILE_RUNTIME_OK;
    case NL_FILE_VALUE_ARGUMENT:result=NVM_FILE_RUNTIME_INVALID;break;
    case NL_FILE_VALUE_TYPE:result=NVM_FILE_RUNTIME_TYPE;break;
    case NL_FILE_VALUE_STALE:result=NVM_FILE_RUNTIME_STALE;break;
    case NL_FILE_VALUE_BORROWED:result=NVM_FILE_RUNTIME_BORROWED;break;
    case NL_FILE_VALUE_LIMIT:result=NVM_FILE_RUNTIME_LIMIT;break;
    case NL_FILE_VALUE_MEMORY:result=NVM_FILE_RUNTIME_MEMORY;break;
    default:result=NVM_FILE_RUNTIME_STATE;break;
    }
    return fr_error(c,result,s);
}
static NvmFileRuntimeStatus fr_prepare(NvmFileFlowStatus s) {
    switch(s) {
    case NVM_FILE_FLOW_OK:return NVM_FILE_RUNTIME_OK;
    case NVM_FILE_FLOW_INVALID:return NVM_FILE_RUNTIME_INVALID;
    case NVM_FILE_FLOW_LIMIT:return NVM_FILE_RUNTIME_LIMIT;
    case NVM_FILE_FLOW_MEMORY:return NVM_FILE_RUNTIME_MEMORY;
    default:return NVM_FILE_RUNTIME_UNRESOLVED;
    }
}
static bool fr_add(size_t *total,size_t count,size_t width) {
    if(count && width>SIZE_MAX/count)return false;
    size_t n=count*width;
    if(*total>NVM_FILE_RUNTIME_BYTES || n>NVM_FILE_RUNTIME_BYTES-*total)return false;
    *total+=n;return true;
}
static bool fr_empty(const NvmFileRuntime *c,uint32_t root) {
    return root<c->storage.values && !c->values[root].view.initialized;
}
static bool fr_live(const NvmFileRuntime *c,uint32_t root) {
    return root<c->storage.values && c->values[root].view.initialized;
}
static void fr_clear(NvmFileRuntime *c,uint32_t root) { c->values[root]=(FileRuntimeValue){0}; }
static bool fr_member(const NvmFileRuntime *c,const char *id,NvmFileFlowDeclaration *out) {
    if(!id){*out=fr_scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=fr_scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=fr_scalar_type(TAG_BOOL);return true;}
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        const NlFilePlanType *t=nl_file_catalog_type(i);
        if(t && !strcmp(t->id,id)){*out=c->types[i];return true;}
    }
    return false;
}
static NvmFileRuntimeView fr_view(NvmFileFlowDeclaration type) {
    NvmFileRuntimeView v={0};v.initialized=true;v.type=type;
    v.owning=type.category==NVM_FILE_CATEGORY_FILE || type.category==NVM_FILE_CATEGORY_OPEN_RESULT;
    return v;
}
static bool fr_error_payload(NlFileResult r,int64_t out[NVM_FILE_RUNTIME_FIELDS]) {
    if(r.bytes>(uint64_t)INT64_MAX)return false;
    out[0]=(int64_t)r.status;out[1]=r.host_errno;out[2]=r.cleanup_errno;out[3]=(int64_t)r.bytes;
    out[4]=r.eof;out[5]=r.consumed;out[6]=r.cleanup_failed;return true;
}
#include "file_cyclic_runtime_facts.inc"

NvmFileRuntimeStatus nvm_file_runtime_create(const uint8_t *bytes,size_t size,NvmFileRuntimeMode mode,NvmFileRuntime **out) {
    if(!out || (mode!=NVM_FILE_RUNTIME_VM && mode!=NVM_FILE_RUNTIME_NATIVE))return NVM_FILE_RUNTIME_INVALID;
    NvmFileHostedPlan *plan=NULL;NvmFileFlowStatus checked=nvm_file_hosted_prepare(bytes,size,&plan);
    if(checked!=NVM_FILE_FLOW_OK)return fr_prepare(checked);
    NvmFileHostedStartup startup;size_t core=0,total;
    if(!nvm_file_hosted_startup(plan,&startup) || !nl_file_values_storage_bound(&core)) {
        nvm_file_hosted_free(plan);return NVM_FILE_RUNTIME_INVALID;
    }
    uint64_t values=mode==NVM_FILE_RUNTIME_VM?startup.vm_value_slots:startup.native_value_slots;
    total=startup.allocation_bound;
    if(!values || values>UINT32_MAX || !startup.frames || !startup.reference_slots || !startup.region_slots ||
       !fr_add(&total,1,sizeof(NvmFileRuntime)) || !fr_add(&total,(size_t)values,sizeof(FileRuntimeValue)) ||
       !fr_add(&total,startup.reference_slots,sizeof(FileRuntimeReference)) ||
       !fr_add(&total,startup.region_slots,sizeof(uint64_t)) ||
       !fr_add(&total,startup.frames,sizeof(FileRuntimeFrame)) || !fr_add(&total,1,core)) {
        nvm_file_hosted_free(plan);return NVM_FILE_RUNTIME_LIMIT;
    }
    NvmFileRuntime *c=calloc(1,sizeof *c);
    if(!c){nvm_file_hosted_free(plan);return NVM_FILE_RUNTIME_MEMORY;}
    c->plan=plan;c->startup=startup;c->next_region=1;c->mode=mode;
    c->storage=(NvmFileRuntimeStorage){total,(uint32_t)values,startup.reference_slots,startup.region_slots,startup.frames};
    c->values=calloc(c->storage.values,sizeof *c->values);
    c->references=calloc(c->storage.references,sizeof *c->references);
    c->regions=calloc(c->storage.regions,sizeof *c->regions);
    c->frames=calloc(c->storage.frames,sizeof *c->frames);
    NvmFileRuntimeStatus status=NVM_FILE_RUNTIME_MEMORY;
    if(!c->values || !c->references || !c->regions || !c->frames)goto fail;
    status=NVM_FILE_RUNTIME_INVALID;
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        NvmFileNominalLayout type;
        if(!nvm_file_hosted_type(plan,i,&type))goto fail;
        c->types[i]=(NvmFileFlowDeclaration){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
            0,type.global_index,type.catalog_ordinal,type.category};
    }
    c->report.function=c->report.instruction=NVM_V2_NO_INDEX;
    c->current_root=startup.initializer!=NVM_V2_NO_INDEX?startup.initializer:startup.entry;
    *out=c;return NVM_FILE_RUNTIME_OK;
fail:
    free(c->frames);free(c->regions);free(c->references);free(c->values);nvm_file_hosted_free(plan);free(c);return status;
}
bool nvm_file_runtime_storage(const NvmFileRuntime *c,NvmFileRuntimeStorage *out) {
    if(!c || !out)return false;
    *out=c->storage;return true;
}
const NvmFileHostedPlan *nvm_file_runtime_plan(const NvmFileRuntime *c) { return c?c->plan:NULL; }
NvmFileRuntimeStatus nvm_file_runtime_begin(NvmFileRuntime *c) {
    if(!c)return NVM_FILE_RUNTIME_INVALID;
    if(c->busy || c->phase==FR_ACTIVE)return NVM_FILE_RUNTIME_BUSY;
    if(c->phase!=FR_READY)return NVM_FILE_RUNTIME_STATE;
    c->busy=true;NlFileValueStatus core=nl_file_values_create(&c->files);c->busy=false;c->phase=FR_ACTIVE;
    if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    c->acquired=true;c->report.acquired=true;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_site(NvmFileRuntime *c,uint32_t function,uint16_t instruction) {
    NvmFileRuntimeStatus status=fr_ready(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
    uint8_t variant=0;
    if(c->kind==FR_CYCLIC) {
        if(!c->frame_count || c->frames[c->frame_count-1].function!=function ||
           c->frames[c->frame_count-1].instruction!=instruction)
            return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
        variant=c->frames[c->frame_count-1].variant;
    }
    if(!fr_instruction(c,function,instruction,variant,&in,&fact) || !fact.reachable)
        return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
    c->report.function=function;c->report.instruction=in.byte_offset;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_fail(NvmFileRuntime *c,NvmFileRuntimeStatus status) {
    NvmFileRuntimeStatus ready=fr_ready(c);if(ready!=NVM_FILE_RUNTIME_OK)return ready;
    if(status<=NVM_FILE_RUNTIME_OK || status>NVM_FILE_RUNTIME_CLEANUP)return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
    return fr_error(c,status,NL_FILE_VALUE_STATE);
}
bool nvm_file_runtime_current_root(const NvmFileRuntime *c,uint32_t *out) {
    if(!c || !out || c->phase!=FR_ACTIVE || c->complete || c->busy)return false;
    *out=c->current_root;return true;
}
bool nvm_file_runtime_view(const NvmFileRuntime *c,uint32_t root,NvmFileRuntimeView *out) {
    if(!c || !out || root>=c->storage.values || c->busy)return false;
    *out=c->values[root].view;return true;
}
NvmFileRuntimeStatus nvm_file_runtime_scalar(NvmFileRuntime *c,uint32_t dst,uint8_t tag,int64_t value) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_empty(c,dst) || (tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_VOID) ||
       (tag==TAG_BOOL && value!=0 && value!=1) || (tag==TAG_VOID && value))
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    NvmFileRuntimeView v=fr_view(fr_scalar_type(tag));v.fields=tag==TAG_VOID?0:1;v.values[0]=value;
    c->values[dst].view=v;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_copy(NvmFileRuntime *c,uint32_t src,uint32_t dst) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_live(c,src) || !fr_empty(c,dst) || c->values[src].view.owning || c->values[src].view.formal)
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    c->values[dst].view=c->values[src].view;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_move(NvmFileRuntime *c,uint32_t src,uint32_t dst) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_live(c,src) || !fr_empty(c,dst) || c->values[src].view.formal)
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    if(c->values[src].view.owning) {
        c->busy=true;NlFileValueStatus core=nl_file_value_move(c->files,&c->values[src].owner,&c->values[dst].owner);c->busy=false;
        if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    }
    c->values[dst].view=c->values[src].view;fr_clear(c,src);return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_drop(NvmFileRuntime *c,uint32_t root) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_live(c,root) || c->values[root].view.formal)return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    if(c->values[root].view.owning) {
        c->busy=true;NlFileValueStatus core=nl_file_value_drop(c->files,&c->values[root].owner);c->busy=false;
        if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    }
    fr_clear(c,root);return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_construct(NvmFileRuntime *c,uint32_t ordinal,uint16_t variant,const uint32_t *inputs,size_t count,uint32_t dst) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(ordinal>=NVM_FILE_NOMINAL_TYPES || dst>=c->storage.values || count>NVM_FILE_RUNTIME_FIELDS || (count && !inputs))
        return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
    NvmFileFlowDeclaration type=c->types[ordinal];bool record=type.category==NVM_FILE_CATEGORY_RECORD;
    const NlFilePlanType *catalog=nl_file_catalog_type(ordinal);
    if(!catalog || (type.category!=NVM_FILE_CATEGORY_RECORD && type.category!=NVM_FILE_CATEGORY_SCALAR_RESULT) ||
       (record?variant!=0:variant>=catalog->member_count) || count!=(record?catalog->member_count:1))
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    NvmFileRuntimeView value=fr_view(type);bool replaces=false;
    if(!record)value.arm=variant?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK;
    for(size_t i=0;i<count;i++) {
        for(size_t j=0;j<i;j++)if(inputs[j]==inputs[i])return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
        if(!fr_live(c,inputs[i]))return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
        NvmFileRuntimeView input=c->values[inputs[i]].view;NvmFileFlowDeclaration expected;
        if(input.owning || input.formal || !fr_member(c,catalog->members[record?i:variant].type_id,&expected) || !fr_same(input.type,expected))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
        replaces|=inputs[i]==dst;
        if(record) {
            if(input.fields!=1 || (input.type.tag!=TAG_INT && input.type.tag!=TAG_BOOL))return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
            value.values[i]=input.values[0];value.fields=(uint8_t)count;
        } else { value.fields=input.fields;memcpy(value.values,input.values,sizeof value.values); }
    }
    if(!replaces && !fr_empty(c,dst))return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    for(size_t i=0;i<count;i++)fr_clear(c,inputs[i]);
    c->values[dst].view=value;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_project(NvmFileRuntime *c,uint32_t src,uint16_t field,uint32_t dst) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_live(c,src) || (src!=dst && !fr_empty(c,dst)))return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    NvmFileRuntimeView input=c->values[src].view;uint32_t ordinal=input.type.catalog_ordinal;
    if(input.owning || input.formal || ordinal>=NVM_FILE_NOMINAL_TYPES)return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    const NlFilePlanType *catalog=nl_file_catalog_type(ordinal);NvmFileFlowDeclaration type;
    NvmFileRuntimeView result={0};
    if(input.type.category==NVM_FILE_CATEGORY_RECORD) {
        if(!catalog || field>=catalog->member_count || field>=input.fields || !fr_member(c,catalog->members[field].type_id,&type))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
        result=fr_view(type);result.fields=1;result.values[0]=input.values[field];
    } else if(input.type.category==NVM_FILE_CATEGORY_SCALAR_RESULT && !field &&
              (input.arm==NVM_FILE_FLOW_ARM_OK || input.arm==NVM_FILE_FLOW_ARM_ERROR)) {
        if(!catalog || !fr_member(c,catalog->members[input.arm==NVM_FILE_FLOW_ARM_ERROR].type_id,&type))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
        result=fr_view(type);result.fields=input.fields;memcpy(result.values,input.values,sizeof result.values);
    } else return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    fr_clear(c,src);c->values[dst].view=result;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_result_arm(NvmFileRuntime *c,uint32_t root,NvmFileFlowArm *out) {
    NvmFileRuntimeStatus status=fr_ready(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!out || !fr_live(c,root))return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
    NvmFileRuntimeView *view=&c->values[root].view;NvmFileFlowArm arm;
    if(view->type.category==NVM_FILE_CATEGORY_OPEN_RESULT) {
        NlFileOpenView actual;c->busy=true;NlFileValueStatus core=nl_file_open_view(c->files,&c->values[root].owner,&actual);c->busy=false;
        if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
        arm=actual.ok?NVM_FILE_FLOW_ARM_OK:NVM_FILE_FLOW_ARM_ERROR;
    } else if(view->type.category==NVM_FILE_CATEGORY_SCALAR_RESULT)arm=view->arm;
    else return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    if(arm!=NVM_FILE_FLOW_ARM_OK && arm!=NVM_FILE_FLOW_ARM_ERROR)return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    *out=arm;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_take(NvmFileRuntime *c,uint32_t src,NvmFileFlowArm arm,uint32_t dst) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_live(c,src) || !fr_empty(c,dst) || (arm!=NVM_FILE_FLOW_ARM_OK && arm!=NVM_FILE_FLOW_ARM_ERROR))
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    NvmFileFlowArm actual;status=nvm_file_runtime_result_arm(c,src,&actual);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(actual!=arm)return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    NvmFileRuntimeView input=c->values[src].view,value={0};
    if(input.type.category==NVM_FILE_CATEGORY_OPEN_RESULT) {
        NlFileValueStatus core;
        if(arm==NVM_FILE_FLOW_ARM_OK) {
            value=fr_view(c->types[0]);c->busy=true;
            core=nl_file_open_take_ok(c->files,&c->values[src].owner,&c->values[dst].owner);c->busy=false;
        } else {
            NlFileOpenView observed;c->busy=true;core=nl_file_open_view(c->files,&c->values[src].owner,&observed);c->busy=false;
            if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
            value=fr_view(c->types[1]);value.fields=NVM_FILE_RUNTIME_FIELDS;
            if(!fr_error_payload(observed.error,value.values))return fr_error(c,NVM_FILE_RUNTIME_LIMIT,NL_FILE_VALUE_LIMIT);
            NlFileResult error;c->busy=true;core=nl_file_open_take_error(c->files,&c->values[src].owner,&error);c->busy=false;
        }
        if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    } else {
        const NlFilePlanType *catalog=nl_file_catalog_type(input.type.catalog_ordinal);NvmFileFlowDeclaration type;
        if(!catalog || !fr_member(c,catalog->members[arm==NVM_FILE_FLOW_ARM_ERROR].type_id,&type))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
        value=fr_view(type);value.fields=input.fields;memcpy(value.values,input.values,sizeof value.values);
    }
    fr_clear(c,src);c->values[dst].view=value;return NVM_FILE_RUNTIME_OK;
}
static bool fr_reference(const NvmFileRuntime *c,uint32_t index) {
    if(index>=c->storage.references)return false;
    const FileRuntimeReference *r=&c->references[index];
    if(!r->live || r->origin>=c->storage.references)return false;
    const FileRuntimeReference *origin=&c->references[r->origin];
    return origin->live && !origin->formal && origin->origin==r->origin &&
           origin->borrow.epoch==r->borrow.epoch && origin->owner_root==r->owner_root &&
           fr_live(c,r->owner_root) && c->values[r->owner_root].view.owning &&
           c->values[r->owner_root].view.type.category==NVM_FILE_CATEGORY_FILE;
}
static bool fr_aliases(const NvmFileRuntime *c,uint32_t origin) {
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && c->references[i].formal && c->references[i].origin==origin)return true;
    return false;
}
NvmFileRuntimeStatus nvm_file_runtime_region_begin(NvmFileRuntime *c) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(c->region_count==c->storage.regions || c->next_region==UINT64_MAX)return fr_error(c,NVM_FILE_RUNTIME_LIMIT,NL_FILE_VALUE_LIMIT);
    c->regions[c->region_count++]=c->next_region++;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_borrow(NvmFileRuntime *c,uint32_t owner,uint32_t reference) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_live(c,owner) || c->values[owner].view.type.category!=NVM_FILE_CATEGORY_FILE ||
       !c->values[owner].view.owning || !c->region_count || reference>=c->storage.references || c->references[reference].live)
        return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    FileRuntimeReference r={0};r.owner_root=owner;r.origin=reference;r.formal_root=NVM_FILE_RUNTIME_NO_SLOT;
    r.region=c->regions[c->region_count-1];c->busy=true;
    NlFileValueStatus core=nl_file_value_borrow(c->files,&c->values[owner].owner,&r.borrow);c->busy=false;
    if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    r.live=true;c->references[reference]=r;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_bind_formal(NvmFileRuntime *c,uint32_t source,uint32_t root,uint32_t reference) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_reference(c,source) || !fr_empty(c,root) || reference>=c->storage.references || c->references[reference].live)
        return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    FileRuntimeReference r=c->references[source];r.formal=true;r.formal_root=root;r.region=0;
    NvmFileRuntimeView value=fr_view(c->types[0]);value.type.mode=2;value.owning=false;value.formal=true;
    c->references[reference]=r;c->values[root].view=value;c->values[root].reference=reference;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_end_reference(NvmFileRuntime *c,uint32_t reference) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!fr_reference(c,reference))return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    FileRuntimeReference *r=&c->references[reference];
    if(r->formal) {
        if(!fr_live(c,r->formal_root) || !c->values[r->formal_root].view.formal || c->values[r->formal_root].reference!=reference)
            return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
        fr_clear(c,r->formal_root);
    } else {
        if(fr_aliases(c,reference))return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
        c->busy=true;NlFileValueStatus core=nl_file_value_end_borrow(c->files,&r->borrow);c->busy=false;
        if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    }
    *r=(FileRuntimeReference){0};return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_region_end(NvmFileRuntime *c) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(!c->region_count)return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    uint64_t region=c->regions[c->region_count-1];
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && !c->references[i].formal &&
       c->references[i].region==region && fr_aliases(c,i))return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && !c->references[i].formal && c->references[i].region==region) {
        status=nvm_file_runtime_end_reference(c,i);if(status!=NVM_FILE_RUNTIME_OK)return status;
    }
    c->regions[--c->region_count]=0;return NVM_FILE_RUNTIME_OK;
}
static bool fr_scalar_result(NvmFileRuntime *c,uint32_t ordinal,NlFileScalarResult actual,NvmFileRuntimeView *out) {
    if(!ordinal || ordinal>=NVM_SERVICE_BINDING_COUNT || actual.kind!=(NlFileScalarKind)(ordinal-1) ||
       actual.ok!=(actual.detail.status==NL_FILE_OK))return false;
    NvmFileRuntimeView result=fr_view(c->types[ordinal+3]);
    result.arm=actual.ok?NVM_FILE_FLOW_ARM_OK:NVM_FILE_FLOW_ARM_ERROR;
    if(!actual.ok) {
        result.fields=NVM_FILE_RUNTIME_FIELDS;
        if(!fr_error_payload(actual.detail,result.values))return false;
    } else if(ordinal==1) {
        if(actual.value<0 || actual.value>1)return false;
        result.fields=1;result.values[0]=actual.value;
    } else if(ordinal==3) {
        if(actual.value<0 || actual.value>255 || (actual.eof && actual.value))return false;
        result.fields=2;result.values[0]=actual.value;result.values[1]=actual.eof;
    } else if(actual.value || actual.eof)return false;
    *out=result;return true;
}
NvmFileRuntimeStatus nvm_file_runtime_service(NvmFileRuntime *c,uint32_t import,uint32_t reference,uint32_t input,uint32_t output) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    uint32_t ordinal;
    if(!fr_empty(c,output) || !fr_import(c,import,&ordinal) || ordinal>=NVM_SERVICE_BINDING_COUNT)
        return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
    bool borrows=ordinal>=1 && ordinal<=3;
    if((borrows && !fr_reference(c,reference)) || (!borrows && reference!=NVM_FILE_RUNTIME_NO_SLOT))
        return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    if(ordinal==1) {
        if(!fr_live(c,input) || !fr_same(c->values[input].view.type,fr_scalar_type(TAG_INT)))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    } else if(ordinal==4) {
        if(!fr_live(c,input) || !c->values[input].view.owning || !fr_same(c->values[input].view.type,c->types[0]))
            return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    } else if(input!=NVM_FILE_RUNTIME_NO_SLOT)return fr_error(c,NVM_FILE_RUNTIME_INVALID,NL_FILE_VALUE_ARGUMENT);
    /* Empty output is already part of the counted arena. Core publication and
     * the remaining metadata assignment cannot allocate. Busy blocks nested
     * cleanup while a fixture/host shim is inside the core call. */
    c->busy=true;NlFileValueStatus core;NlFileScalarResult scalar={0};
    switch(ordinal) {
    case 0:core=nl_file_values_temp(c->files,&c->values[output].owner);break;
    case 1:core=nl_file_value_write_byte(c->files,&c->references[reference].borrow,c->values[input].view.values[0],&scalar);break;
    case 2:core=nl_file_value_rewind(c->files,&c->references[reference].borrow,&scalar);break;
    case 3:core=nl_file_value_read_byte(c->files,&c->references[reference].borrow,&scalar);break;
    default:core=nl_file_value_close(c->files,&c->values[input].owner,&scalar);break;
    }
    c->busy=false;
    if(core!=NL_FILE_VALUE_OK)return fr_core(c,core);
    if(!ordinal) {
        c->values[output].view=fr_view(c->types[3]);return NVM_FILE_RUNTIME_OK;
    }
    /* Accepted close consumes on either Result arm. Never leave its old root
     * apparently live while checking the passive result representation. */
    if(ordinal==4)fr_clear(c,input);
    NvmFileRuntimeView result;
    if(!fr_scalar_result(c,ordinal,scalar,&result))return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    if(ordinal==1)fr_clear(c,input);
    c->values[output].view=result;return NVM_FILE_RUNTIME_OK;
}
NvmFileRuntimeStatus nvm_file_runtime_complete_root(NvmFileRuntime *c,uint32_t root) {
    NvmFileRuntimeStatus status=fr_effect(c);if(status!=NVM_FILE_RUNTIME_OK)return status;
    if(c->kind==FR_CYCLIC) {
        NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
        FileRuntimeFrame *f=&c->frames[c->frame_count-1];
        if(c->frame_count!=1 || !fr_instruction(c,f->function,(uint16_t)f->instruction,f->variant,&in,&fact) ||
           in.decoded.opcode!=OP_RET || !fact.exit_checked)
            return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    }
    bool initializer=c->current_root==c->startup.initializer;
    NvmFileHostedFunction fn;
    if(!fr_function(c,c->current_root,&fn))return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    if(initializer) {
        if(root!=NVM_FILE_RUNTIME_NO_SLOT || fn.code.declaration.result_count)return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    } else if(!fr_live(c,root) || !fr_same(c->values[root].view.type,fn.code.declaration.result) ||
              c->values[root].view.owning || c->values[root].view.formal ||
              (c->values[root].view.type.tag!=TAG_INT && c->values[root].view.type.tag!=TAG_BOOL))
        return fr_error(c,NVM_FILE_RUNTIME_TYPE,NL_FILE_VALUE_TYPE);
    if(c->region_count)return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live)return fr_error(c,NVM_FILE_RUNTIME_BORROWED,NL_FILE_VALUE_BORROWED);
    for(uint32_t i=0;i<c->storage.values;i++)if(c->values[i].view.initialized &&
       (c->values[i].view.owning || c->values[i].view.formal))return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    if(initializer) {
        NlFileValuesFinish observed;
        if(!nl_file_values_report(c->files,&observed))return fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
        if(observed.execution!=NL_FILE_VALUE_OK || observed.cleanup_failures)
            return fr_error(c,NVM_FILE_RUNTIME_CLEANUP,observed.execution);
    }
    if(!initializer)c->result=c->values[root].view;
    memset(c->values,0,(size_t)c->storage.values*sizeof *c->values);
    memset(c->frames,0,(size_t)c->storage.frames*sizeof *c->frames);c->frame_count=0;
    if(initializer)c->current_root=c->startup.entry;
    else c->complete=true;
    return NVM_FILE_RUNTIME_OK;
}
static NvmFileRuntimeReport fr_refused(NvmFileRuntimeStatus status) {
    NvmFileRuntimeReport report={0};report.status=status;
    report.function=report.instruction=NVM_V2_NO_INDEX;return report;
}
static NvmFileRuntimeReport fr_finish(NvmFileRuntime *c,NvmFileRuntimeView *out) {
    if(!c)return fr_refused(NVM_FILE_RUNTIME_INVALID);
    if(c->busy)return fr_refused(NVM_FILE_RUNTIME_BUSY);
    if(c->phase==FR_TERMINAL) {
        if(c->report.status==NVM_FILE_RUNTIME_OK && out)*out=c->result;
        return c->report;
    }
    if(!c->complete && c->report.status==NVM_FILE_RUNTIME_OK)fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
    c->busy=true;
    /* Formal references never release caller epochs. Clear aliases before the
     * originating borrows, then attempt every root before terminal disposal. */
    for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live && c->references[i].formal) {
        uint32_t root=c->references[i].formal_root;
        if(root<c->storage.values && c->values[root].view.formal && c->values[root].reference==i)fr_clear(c,root);
        else fr_error(c,NVM_FILE_RUNTIME_STATE,NL_FILE_VALUE_STATE);
        c->references[i]=(FileRuntimeReference){0};
    }
    if(c->acquired) {
        for(uint32_t i=0;i<c->storage.references;i++)if(c->references[i].live) {
            NlFileValueStatus core=nl_file_value_end_borrow(c->files,&c->references[i].borrow);
            if(core!=NL_FILE_VALUE_OK)fr_core(c,core);
            c->references[i]=(FileRuntimeReference){0};
        }
        for(uint32_t i=0;i<c->storage.values;i++) {
            if(c->values[i].view.initialized && c->values[i].view.owning) {
                NlFileValueStatus core=nl_file_value_drop(c->files,&c->values[i].owner);
                if(core!=NL_FILE_VALUE_OK)fr_core(c,core);
            }
            fr_clear(c,i);
        }
        NlFileValueStatus execution=c->report.status==NVM_FILE_RUNTIME_OK?NL_FILE_VALUE_OK:
            c->report.core_status==NL_FILE_VALUE_OK?NL_FILE_VALUE_STATE:c->report.core_status;
        c->report.cleanup=nl_file_values_destroy(c->files,execution);c->files=NULL;
        if(c->report.status==NVM_FILE_RUNTIME_OK &&
           (c->report.cleanup.execution!=NL_FILE_VALUE_OK || c->report.cleanup.cleanup_failures))
            fr_error(c,NVM_FILE_RUNTIME_CLEANUP,c->report.cleanup.execution);
    } else c->report.cleanup.execution=c->report.core_status;
    memset(c->regions,0,(size_t)c->storage.regions*sizeof *c->regions);c->region_count=0;
    c->frame_count=0;c->busy=false;c->phase=FR_TERMINAL;
    if(c->report.status==NVM_FILE_RUNTIME_OK && out)*out=c->result;
    return c->report;
}
NvmFileRuntimeReport nvm_file_runtime_finish(NvmFileRuntime *c,NvmFileRuntimeView *out) {
    if(c && c->kind!=FR_ACYCLIC)return fr_refused(NVM_FILE_RUNTIME_STATE);
    return fr_finish(c,out);
}
NvmFileRuntimeReport nvm_file_runtime_destroy(NvmFileRuntime **address,NvmFileRuntimeView *out) {
    if(!address || !*address)return fr_refused(NVM_FILE_RUNTIME_INVALID);
    NvmFileRuntime *c=*address;
    if(c->kind!=FR_ACYCLIC)return fr_refused(NVM_FILE_RUNTIME_STATE);
    if(c->busy)return fr_refused(NVM_FILE_RUNTIME_BUSY);
    NvmFileRuntimeReport report=nvm_file_runtime_finish(c,out);
    free(c->frames);free(c->regions);free(c->references);free(c->values);nvm_file_hosted_free(c->plan);free(c);*address=NULL;
    return report;
}

#include "file_runtime_frames.inc"
#include "file_cyclic_runtime.inc"

#if defined(NVM_FILE_NATIVE_PRIVATE) || defined(NVM_FILE_PUBLIC_ENGINE)
#include "file_native_abi.h"
bool nvm_file_runtime_native_abi(uint32_t revision,size_t view,size_t frame,size_t report) {
    return revision==NVM_FILE_NATIVE_ABI && view==sizeof(NvmFileRuntimeView) &&
        frame==sizeof(NvmFileRuntimeFrameView) && report==sizeof(NvmFileRuntimeReport);
}
#endif
