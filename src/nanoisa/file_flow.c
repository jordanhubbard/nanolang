#include "file_flow.h"
#include "ownership_contracts.h"
#include "nvm_v2_sections.h"
#include "isa.h"
#include "../nsi_file_catalog.h"
#include <stdlib.h>
#include <string.h>

struct NvmFileFlowDeclarations {
    uint32_t references, count;
    uint64_t next_identity;
    NvmFileNominalPlan *nominal;
    NvmFileFlowFunction functions[NVM_FILE_FLOW_FUNCTIONS];
    uint32_t offsets[NVM_FILE_FLOW_FUNCTIONS];
    NvmFileFlowDeclaration locals[];
};
struct NvmFileFlowState {
    NvmFileFlowDeclarations *declarations;
    uint32_t function;
    uint16_t stack_count, region_count, obligation_count;
    uint64_t family, cleanup_obligations;
    bool reachable;
    NvmFileFlowObligation obligations[NVM_FILE_FLOW_OBLIGATIONS];
    NvmFileFlowValue locals[NVM_FILE_FLOW_LOCALS];
    NvmFileFlowValue stack[NVM_FILE_FLOW_STACK];
    NvmFileFlowReference references[NVM_FILE_FLOW_REFERENCES];
    uint64_t regions[NVM_FILE_FLOW_REFERENCES];
};
static bool scalar(uint8_t tag) { return tag==TAG_INT || tag==TAG_BOOL || tag==TAG_VOID; }
static bool owned(NvmFileFlowDeclaration d) {
    return !d.mode && (d.category==NVM_FILE_CATEGORY_FILE || d.category==NVM_FILE_CATEGORY_OPEN_RESULT);
}
static bool same(NvmFileFlowDeclaration a,NvmFileFlowDeclaration b) {
    return a.tag==b.tag && a.mode==b.mode && a.global_index==b.global_index &&
           a.catalog_ordinal==b.catalog_ordinal && a.category==b.category;
}
static NvmFileFlowDeclaration scalar_type(uint8_t tag) {
    return (NvmFileFlowDeclaration){tag,0,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX,NVM_FILE_CATEGORY_UNKNOWN};
}
static NvmFileFlowStatus read_declaration(NvmV2Cursor *c,const NvmFileNominalPlan *p,
                                         NvmFileFlowDeclaration *out) {
    uint8_t tag,mode;uint16_t pad;uint32_t index;
    if(nvm_v2_u8(c,&tag)!=NVM_V2_OK || nvm_v2_u8(c,&mode)!=NVM_V2_OK ||
       nvm_v2_u16(c,&pad)!=NVM_V2_OK || nvm_v2_u32(c,&index)!=NVM_V2_OK || pad)
        return NVM_FILE_FLOW_INVALID;
    NvmFileFlowDeclaration d=scalar_type(tag);d.mode=mode;d.global_index=index;
    if(index==NVM_V2_NO_INDEX) {
        if(mode || !scalar(tag))return NVM_FILE_FLOW_UNRESOLVED;
    } else {
        NvmFileNominalLayout layout;
        if(!nvm_file_nominal_layout(p,index,&layout))return NVM_FILE_FLOW_INVALID;
        if(layout.category==NVM_FILE_CATEGORY_UNKNOWN)return NVM_FILE_FLOW_UNRESOLVED;
        d.catalog_ordinal=layout.catalog_ordinal;d.category=layout.category;
        if((mode && (mode!=2 || d.category!=NVM_FILE_CATEGORY_FILE)) ||
           tag!=(layout.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION))
            return NVM_FILE_FLOW_INVALID;
    }
    *out=d;return NVM_FILE_FLOW_OK;
}
static NvmFileFlowStatus nominal_status(NvmFileNominalStatus s) {
    switch(s) {
    case NVM_FILE_NOMINAL_DESCRIBED:return NVM_FILE_FLOW_OK;
    case NVM_FILE_NOMINAL_LIMIT:return NVM_FILE_FLOW_LIMIT;
    case NVM_FILE_NOMINAL_MEMORY:return NVM_FILE_FLOW_MEMORY;
    default:return NVM_FILE_FLOW_INVALID;
    }
}
NvmFileFlowStatus nvm_file_flow_declarations(const NvmModule *m,NvmFileFlowDeclarations **out) {
    if(!m || !out || !m->function_count || !m->functions)return NVM_FILE_FLOW_INVALID;
    if(m->function_count>NVM_FILE_FLOW_FUNCTIONS)return NVM_FILE_FLOW_LIMIT;
    if(m->callback_contract_count || m->module_ref_count)return NVM_FILE_FLOW_UNRESOLVED;
    size_t count=0;
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *f=&m->functions[i];
        if(f->arity>f->local_count || f->result_count>1)return NVM_FILE_FLOW_INVALID;
        if(f->upvalue_count)return NVM_FILE_FLOW_UNRESOLVED;
        if(f->local_count>NVM_FILE_FLOW_LOCALS)return NVM_FILE_FLOW_LIMIT;
        if(count>SIZE_MAX-f->local_count)return NVM_FILE_FLOW_LIMIT;
        count+=f->local_count;
    }
    if(count>(SIZE_MAX-sizeof(NvmFileFlowDeclarations))/sizeof(NvmFileFlowDeclaration))
        return NVM_FILE_FLOW_LIMIT;
    size_t bytes=sizeof(NvmFileFlowDeclarations)+count*sizeof(NvmFileFlowDeclaration);
    /* I account for the nominal allocator's exact maximum extent before calling
     * it; no duplicated private struct size can drift from that allocation. */
    size_t nominal_bound;
    if(!nvm_file_nominal_storage_bound(NVM_FILE_NOMINAL_MAX_LAYOUTS,&nominal_bound) ||
       bytes>NVM_FILE_FLOW_BYTES || nominal_bound>NVM_FILE_FLOW_BYTES-bytes)return NVM_FILE_FLOW_LIMIT;
    NvmFileNominalPlan *nominal=NULL;
    NvmFileFlowStatus status=nominal_status(nvm_file_nominal_plan(m,&nominal));
    if(status!=NVM_FILE_FLOW_OK)return status;
    NvmFileFlowDeclarations *d=calloc(1,bytes);
    if(!d){nvm_file_nominal_plan_free(nominal);return NVM_FILE_FLOW_MEMORY;}
    d->references=1;d->count=m->function_count;d->nominal=nominal;d->next_identity=1;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    uint32_t version,layouts,functions;const uint8_t *flags;
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || version!=NVM_OWNERSHIP_VERSION ||
       nvm_v2_u32(&c,&layouts)!=NVM_V2_OK || layouts!=nvm_file_nominal_layout_count(nominal) ||
       nvm_v2_take(&c,layouts,&flags)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&functions)!=NVM_V2_OK || functions!=d->count) {
        status=NVM_FILE_FLOW_INVALID;goto fail;
    }
    size_t offset=0;
    for(uint32_t i=0;i<d->count;i++) {
        NvmFileFlowFunction *f=&d->functions[i];const NvmFunctionEntry *actual=&m->functions[i];
        if(nvm_v2_u16(&c,&f->locals)!=NVM_V2_OK || nvm_v2_u16(&c,&f->parameters)!=NVM_V2_OK ||
           f->locals!=actual->local_count || f->parameters!=actual->arity) {
            status=NVM_FILE_FLOW_INVALID;goto fail;
        }
        d->offsets[i]=(uint32_t)offset;f->result_count=actual->result_count;
        status=read_declaration(&c,nominal,&f->result);if(status!=NVM_FILE_FLOW_OK)goto fail;
        if(f->result.mode || f->result.tag!=(f->result_count?actual->result_tag:TAG_VOID)) {
            status=NVM_FILE_FLOW_INVALID;goto fail;
        }
        for(uint16_t j=0;j<f->locals;j++) {
            status=read_declaration(&c,nominal,&d->locals[offset+j]);if(status!=NVM_FILE_FLOW_OK)goto fail;
            if(j>=f->parameters && d->locals[offset+j].mode) {status=NVM_FILE_FLOW_INVALID;goto fail;}
        }
        offset+=f->locals;
    }
    if(offset!=count || c.pos!=c.size){status=NVM_FILE_FLOW_INVALID;goto fail;}
    *out=d;return NVM_FILE_FLOW_OK;
fail:
    nvm_file_flow_declarations_free(d);return status;
}
void nvm_file_flow_declarations_free(NvmFileFlowDeclarations *d) {
    if(d && --d->references==0){nvm_file_nominal_plan_free(d->nominal);free(d);}
}
bool nvm_file_flow_function(const NvmFileFlowDeclarations *d,uint32_t f,NvmFileFlowFunction *out) {
    if(!d || !out || f>=d->count)return false;
    *out=d->functions[f];return true;
}
bool nvm_file_flow_declaration(const NvmFileFlowDeclarations *d,uint32_t f,uint16_t local,
                              NvmFileFlowDeclaration *out) {
    if(!d || !out || f>=d->count || local>=d->functions[f].locals)return false;
    *out=d->locals[d->offsets[f]+local];return true;
}
static NvmFileFlowValue initial_value(NvmFileFlowDeclaration type,bool initialized,uint64_t owner) {
    NvmFileFlowArm arm=type.category==NVM_FILE_CATEGORY_OPEN_RESULT ||
                       type.category==NVM_FILE_CATEGORY_SCALAR_RESULT ? NVM_FILE_FLOW_ARM_UNKNOWN:NVM_FILE_FLOW_ARM_NONE;
    return (NvmFileFlowValue){type,initialized,owner,arm};
}
NvmFileFlowStatus nvm_file_flow_state(NvmFileFlowDeclarations *d,uint32_t function,NvmFileFlowState **out) {
    if(!d || !out || function>=d->count)return NVM_FILE_FLOW_INVALID;
    if(d->references==UINT32_MAX || sizeof(NvmFileFlowState)>NVM_FILE_FLOW_BYTES)return NVM_FILE_FLOW_LIMIT;
    uint32_t owners=0,identities=1;
    for(uint16_t i=0;i<d->functions[function].parameters;i++) {
        NvmFileFlowDeclaration type=d->locals[d->offsets[function]+i];
        owners+=owned(type);identities+=owned(type) || type.mode;
    }
    if(owners>NVM_FILE_FLOW_OWNERS || identities>UINT64_MAX-d->next_identity)return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowState *s=calloc(1,sizeof *s);if(!s)return NVM_FILE_FLOW_MEMORY;
    s->declarations=d;s->function=function;s->family=d->next_identity++;s->reachable=true;
    const NvmFileFlowFunction *f=&d->functions[function];
    for(uint16_t i=0;i<f->locals;i++) {
        NvmFileFlowDeclaration type=d->locals[d->offsets[function]+i];bool parameter=i<f->parameters;
        s->locals[i]=initial_value(type,parameter,parameter && owned(type)?d->next_identity++:0);
        if(parameter && type.mode) {
            uint64_t identity=d->next_identity++;
            s->references[i]=(NvmFileFlowReference){true,true,i,identity,identity,0};
        }
    }
    d->references++;*out=s;return NVM_FILE_FLOW_OK;
}
void nvm_file_flow_state_free(NvmFileFlowState *s) {
    if(s){nvm_file_flow_declarations_free(s->declarations);free(s);}
}
static bool local_valid(const NvmFileFlowState *s,uint16_t local) {
    return s && local<s->declarations->functions[s->function].locals;
}
bool nvm_file_flow_local(const NvmFileFlowState *s,uint16_t local,NvmFileFlowValue *out) {
    if(!out || !local_valid(s,local))return false;
    *out=s->locals[local];return true;
}
bool nvm_file_flow_stack(const NvmFileFlowState *s,uint16_t index,NvmFileFlowValue *out) {
    if(!s || !out || index>=s->stack_count)return false;
    *out=s->stack[index];return true;
}
bool nvm_file_flow_reference(const NvmFileFlowState *s,uint16_t reference,NvmFileFlowReference *out) {
    if(!s || !out || reference>=NVM_FILE_FLOW_REFERENCES)return false;
    *out=s->references[reference];return true;
}
bool nvm_file_flow_counts(const NvmFileFlowState *s,NvmFileFlowCounts *out) {
    if(!s || !out)return false;
    NvmFileFlowCounts c={s->stack_count,s->region_count,0,0,s->cleanup_obligations,s->obligation_count,s->reachable};
    for(uint16_t i=0;i<s->declarations->functions[s->function].locals;i++)c.owners+=s->locals[i].owner!=0;
    for(uint16_t i=0;i<s->stack_count;i++)c.owners+=s->stack[i].owner!=0;
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)c.references+=s->references[i].live;
    *out=c;return true;
}
static bool held(const NvmFileFlowState *s,uint64_t owner) {
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)
        if(s->references[i].live && s->references[i].owner==owner)return true;
    return false;
}
static bool live_owner(const NvmFileFlowState *s,uint16_t local) {
    return local_valid(s,local) && s->locals[local].initialized && s->locals[local].owner &&
           owned(s->locals[local].type) && !held(s,s->locals[local].owner);
}
static void clear_local(NvmFileFlowState *s,uint16_t local) {
    s->locals[local]=initial_value(s->locals[local].type,false,0);
}
static void pop(NvmFileFlowState *s) { s->stack[--s->stack_count]=(NvmFileFlowValue){0}; }
NvmFileFlowStatus nvm_file_flow_push_scalar(NvmFileFlowState *s,uint8_t tag) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s)return NVM_FILE_FLOW_INVALID;
    if(!scalar(tag))return NVM_FILE_FLOW_UNRESOLVED;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count++]=initial_value(scalar_type(tag),true,0);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_load(NvmFileFlowState *s,uint16_t local) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!local_valid(s,local) || !s->locals[local].initialized || s->locals[local].type.mode ||
       owned(s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count++]=s->locals[local];return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_store(NvmFileFlowState *s,uint16_t local) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!local_valid(s,local) || !s->stack_count)return NVM_FILE_FLOW_INVALID;
    NvmFileFlowValue value=s->stack[s->stack_count-1];
    if(!value.initialized || value.owner || s->locals[local].type.mode || owned(s->locals[local].type) ||
       !same(value.type,s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    s->locals[local]=value;pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_dup(NvmFileFlowState *s) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s || !s->stack_count || s->stack[s->stack_count-1].owner)return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count]=s->stack[s->stack_count-1];s->stack_count++;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_pop(NvmFileFlowState *s) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s || !s->stack_count || s->stack[s->stack_count-1].owner)return NVM_FILE_FLOW_INVALID;
    pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_move(NvmFileFlowState *s,uint16_t from,uint16_t to) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!live_owner(s,from) || !local_valid(s,to) || s->locals[to].initialized ||
       !same(s->locals[from].type,s->locals[to].type))return NVM_FILE_FLOW_INVALID;
    s->locals[to]=s->locals[from];
    if(s->locals[to].type.category==NVM_FILE_CATEGORY_OPEN_RESULT)s->locals[to].arm=NVM_FILE_FLOW_ARM_UNKNOWN;
    clear_local(s,from);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_take(NvmFileFlowState *s,uint16_t local) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!live_owner(s,local))return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count]=s->locals[local];
    if(s->stack[s->stack_count].type.category==NVM_FILE_CATEGORY_OPEN_RESULT)s->stack[s->stack_count].arm=NVM_FILE_FLOW_ARM_UNKNOWN;
    s->stack_count++;clear_local(s,local);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_put(NvmFileFlowState *s,uint16_t local) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!local_valid(s,local) || s->locals[local].initialized || !s->stack_count)return NVM_FILE_FLOW_INVALID;
    NvmFileFlowValue value=s->stack[s->stack_count-1];
    if(!value.owner || !owned(value.type) || !same(value.type,s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    if(value.type.category==NVM_FILE_CATEGORY_OPEN_RESULT)value.arm=NVM_FILE_FLOW_ARM_UNKNOWN;
    s->locals[local]=value;pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_drop_local(NvmFileFlowState *s,uint16_t local) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!live_owner(s,local))return NVM_FILE_FLOW_INVALID;
    if(s->cleanup_obligations==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->cleanup_obligations++;clear_local(s,local);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_drop_stack(NvmFileFlowState *s) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s || !s->stack_count || !s->stack[s->stack_count-1].owner)return NVM_FILE_FLOW_INVALID;
    if(s->cleanup_obligations==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->cleanup_obligations++;pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_region_begin(NvmFileFlowState *s) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s)return NVM_FILE_FLOW_INVALID;
    if(s->region_count==NVM_FILE_FLOW_REFERENCES || s->declarations->next_identity==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->regions[s->region_count++]=s->declarations->next_identity++;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_region_end(NvmFileFlowState *s) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s || !s->region_count)return NVM_FILE_FLOW_INVALID;
    uint64_t region=s->regions[s->region_count-1];
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)
        if(s->references[i].live && !s->references[i].formal && s->references[i].region==region)
            s->references[i]=(NvmFileFlowReference){0};
    s->regions[--s->region_count]=0;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_borrow(NvmFileFlowState *s,uint16_t local,uint16_t reference) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!live_owner(s,local) || s->locals[local].type.category!=NVM_FILE_CATEGORY_FILE ||
       reference>=NVM_FILE_FLOW_REFERENCES || s->references[reference].live || !s->region_count)
        return NVM_FILE_FLOW_INVALID;
    if(s->declarations->next_identity==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->references[reference]=(NvmFileFlowReference){true,false,local,s->locals[local].owner,
        s->declarations->next_identity++,s->regions[s->region_count-1]};return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_end_borrow(NvmFileFlowState *s,uint16_t reference) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s || reference>=NVM_FILE_FLOW_REFERENCES || !s->references[reference].live ||
       s->references[reference].formal)return NVM_FILE_FLOW_INVALID;
    s->references[reference]=(NvmFileFlowReference){0};return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_can_exit(const NvmFileFlowState *s) {
    if(s && !s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(!s)return NVM_FILE_FLOW_INVALID;
    const NvmFileFlowFunction *f=&s->declarations->functions[s->function];
    if(s->region_count || s->stack_count!=f->result_count)return NVM_FILE_FLOW_INVALID;
    if(f->result_count) {
        NvmFileFlowValue value=s->stack[0];
        if(!value.initialized || !same(value.type,f->result) ||
           (owned(value.type)!=(value.owner!=0)))return NVM_FILE_FLOW_INVALID;
    }
    for(uint16_t i=0;i<f->locals;i++)if(s->locals[i].owner)return NVM_FILE_FLOW_INVALID;
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)
        if(s->references[i].live && !s->references[i].formal)return NVM_FILE_FLOW_INVALID;
    return NVM_FILE_FLOW_OK;
}

static bool result_type(NvmFileFlowDeclaration type) {
    return type.category==NVM_FILE_CATEGORY_OPEN_RESULT || type.category==NVM_FILE_CATEGORY_SCALAR_RESULT;
}
static bool catalog_type(const NvmFileFlowState *s,uint32_t ordinal,NvmFileFlowDeclaration *out) {
    NvmFileNominalLayout l;
    if(!nvm_file_nominal_type(s->declarations->nominal,ordinal,&l))return false;
    *out=(NvmFileFlowDeclaration){l.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
        0,l.global_index,l.catalog_ordinal,l.category};return true;
}
static bool member_type(const NvmFileFlowState *s,const char *id,NvmFileFlowDeclaration *out) {
    if(!id){*out=scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=scalar_type(TAG_BOOL);return true;}
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        const NlFilePlanType *t=nl_file_catalog_type(i);
        if(t && !strcmp(t->id,id))return catalog_type(s,i,out);
    }
    return false;
}
static bool value_same(NvmFileFlowValue a,NvmFileFlowValue b) {
    return same(a.type,b.type) && a.initialized==b.initialized && a.owner==b.owner;
}
static bool reference_same(NvmFileFlowReference a,NvmFileFlowReference b) {
    return a.live==b.live && (!a.live || (a.formal==b.formal && a.local==b.local &&
        a.owner==b.owner && a.identity==b.identity && a.region==b.region));
}
static bool obligation_same(NvmFileFlowObligation a,NvmFileFlowObligation b) {
    return a.kind==b.kind && a.site==b.site && a.target==b.target && a.checks==b.checks &&
        a.required_rights==b.required_rights && a.acquired_rights==b.acquired_rights &&
        a.parameters==b.parameters && a.owned_inputs==b.owned_inputs && a.borrowed_inputs==b.borrowed_inputs &&
        a.result_count==b.result_count && same(a.result,b.result) &&
        a.outcomes[0]==b.outcomes[0] && a.outcomes[1]==b.outcomes[1];
}
static NvmFileFlowStatus obligation_slot(const NvmFileFlowState *s,NvmFileFlowObligation item,uint16_t *slot) {
    if(item.site==UINT32_MAX)return NVM_FILE_FLOW_INVALID;
    for(uint16_t i=0;i<s->obligation_count;i++)if(s->obligations[i].site==item.site) {
        if(!obligation_same(s->obligations[i],item))return NVM_FILE_FLOW_UNRESOLVED;
        *slot=i;return NVM_FILE_FLOW_OK;
    }
    if(s->obligation_count==NVM_FILE_FLOW_OBLIGATIONS)return NVM_FILE_FLOW_LIMIT;
    *slot=s->obligation_count;return NVM_FILE_FLOW_OK;
}
static void commit_obligation(NvmFileFlowState *s,NvmFileFlowObligation item,uint16_t slot) {
    if(slot==s->obligation_count)s->obligations[s->obligation_count++]=item;
}
bool nvm_file_flow_obligation(const NvmFileFlowState *s,uint16_t index,NvmFileFlowObligation *out) {
    if(!s || !out || index>=s->obligation_count)return false;
    *out=s->obligations[index];return true;
}
NvmFileFlowStatus nvm_file_flow_clone(const NvmFileFlowState *s,NvmFileFlowState **out) {
    if(!s || !out)return NVM_FILE_FLOW_INVALID;
    if(s->declarations->references==UINT32_MAX)return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowState *copy=malloc(sizeof *copy);if(!copy)return NVM_FILE_FLOW_MEMORY;
    *copy=*s;s->declarations->references++;*out=copy;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_refine(const NvmFileFlowState *s,uint16_t local,
                                      NvmFileFlowState **ok,NvmFileFlowState **error) {
    if(!s || !ok || !error || ok==error || !local_valid(s,local))return NVM_FILE_FLOW_INVALID;
    if(!s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    NvmFileFlowValue value=s->locals[local];
    if(!value.initialized || !result_type(value.type) || value.type.mode)return NVM_FILE_FLOW_INVALID;
    if(s->declarations->references>UINT32_MAX-2)return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowState *a=NULL,*b=NULL;
    NvmFileFlowStatus status=nvm_file_flow_clone(s,&a);if(status!=NVM_FILE_FLOW_OK)return status;
    status=nvm_file_flow_clone(s,&b);
    if(status!=NVM_FILE_FLOW_OK){nvm_file_flow_state_free(a);return status;}
    a->reachable=value.arm!=NVM_FILE_FLOW_ARM_ERROR;b->reachable=value.arm!=NVM_FILE_FLOW_ARM_OK;
    a->locals[local].arm=NVM_FILE_FLOW_ARM_OK;b->locals[local].arm=NVM_FILE_FLOW_ARM_ERROR;
    *ok=a;*error=b;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_join(NvmFileFlowState *s,const NvmFileFlowState *incoming,bool *changed) {
    if(!s || !incoming || !changed)return NVM_FILE_FLOW_INVALID;
    if(s->declarations!=incoming->declarations || s->function!=incoming->function || s->family!=incoming->family)
        return NVM_FILE_FLOW_UNRESOLVED;
    if(!incoming->reachable){*changed=false;return NVM_FILE_FLOW_OK;}
    if(!s->reachable){*s=*incoming;*changed=true;return NVM_FILE_FLOW_OK;}
    if(s->stack_count!=incoming->stack_count || s->region_count!=incoming->region_count ||
       s->cleanup_obligations!=incoming->cleanup_obligations)return NVM_FILE_FLOW_UNRESOLVED;
    bool differs=false;
    uint16_t locals=s->declarations->functions[s->function].locals;
    for(uint16_t i=0;i<locals;i++) {
        if(!value_same(s->locals[i],incoming->locals[i]))return NVM_FILE_FLOW_UNRESOLVED;
        if(s->locals[i].initialized && s->locals[i].arm!=incoming->locals[i].arm) {
            if(!result_type(s->locals[i].type))return NVM_FILE_FLOW_UNRESOLVED;
            differs|=s->locals[i].arm!=NVM_FILE_FLOW_ARM_UNKNOWN;
        }
    }
    for(uint16_t i=0;i<s->stack_count;i++) {
        if(!value_same(s->stack[i],incoming->stack[i]))return NVM_FILE_FLOW_UNRESOLVED;
        if(s->stack[i].arm!=incoming->stack[i].arm) {
            if(!result_type(s->stack[i].type))return NVM_FILE_FLOW_UNRESOLVED;
            differs|=s->stack[i].arm!=NVM_FILE_FLOW_ARM_UNKNOWN;
        }
    }
    for(uint16_t i=0;i<s->region_count;i++)if(s->regions[i]!=incoming->regions[i])return NVM_FILE_FLOW_UNRESOLVED;
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)
        if(!reference_same(s->references[i],incoming->references[i]))return NVM_FILE_FLOW_UNRESOLVED;
    uint16_t add=0;
    for(uint16_t i=0;i<incoming->obligation_count;i++) {
        bool found=false;
        for(uint16_t j=0;j<s->obligation_count;j++)if(s->obligations[j].site==incoming->obligations[i].site) {
            if(!obligation_same(s->obligations[j],incoming->obligations[i]))return NVM_FILE_FLOW_UNRESOLVED;
            found=true;break;
        }
        add+=!found;
    }
    if((uint32_t)s->obligation_count+add>NVM_FILE_FLOW_OBLIGATIONS)return NVM_FILE_FLOW_LIMIT;
    /* All comparisons and capacity checks precede this allocation-free commit. */
    for(uint16_t i=0;i<locals;i++)if(s->locals[i].initialized && s->locals[i].arm!=incoming->locals[i].arm)
        s->locals[i].arm=NVM_FILE_FLOW_ARM_UNKNOWN;
    for(uint16_t i=0;i<s->stack_count;i++)if(s->stack[i].arm!=incoming->stack[i].arm)
        s->stack[i].arm=NVM_FILE_FLOW_ARM_UNKNOWN;
    for(uint16_t i=0;i<incoming->obligation_count;i++) {
        bool found=false;
        for(uint16_t j=0;j<s->obligation_count;j++)if(s->obligations[j].site==incoming->obligations[i].site){found=true;break;}
        if(!found)s->obligations[s->obligation_count++]=incoming->obligations[i];
    }
    *changed=differs || add!=0;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_clear_copy(NvmFileFlowState *s,uint16_t local) {
    if(!local_valid(s,local))return NVM_FILE_FLOW_INVALID;
    if(!s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    if(s->locals[local].type.mode || owned(s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    clear_local(s,local);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_construct(NvmFileFlowState *s,uint32_t ordinal,uint16_t variant) {
    if(!s)return NVM_FILE_FLOW_INVALID;
    if(!s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    NvmFileFlowDeclaration result;
    if(!catalog_type(s,ordinal,&result))return NVM_FILE_FLOW_INVALID;
    if(owned(result))return NVM_FILE_FLOW_INVALID;
    const NlFilePlanType *type=nl_file_catalog_type(ordinal);
    bool record=result.category==NVM_FILE_CATEGORY_RECORD;
    if(!type || (record?variant!=0:variant>=type->member_count))return NVM_FILE_FLOW_INVALID;
    size_t count=record?type->member_count:1;
    if(count>s->stack_count)return NVM_FILE_FLOW_INVALID;
    if((size_t)s->stack_count-count+1>NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    uint16_t start=(uint16_t)(s->stack_count-count);
    for(size_t i=0;i<count;i++) {
        NvmFileFlowDeclaration expected;const NlFilePlanMember *member=&type->members[record?i:variant];
        if(!member_type(s,member->type_id,&expected) || owned(expected))return NVM_FILE_FLOW_INVALID;
        NvmFileFlowValue value=s->stack[start+i];
        if(!value.initialized || value.owner || !same(expected,value.type))return NVM_FILE_FLOW_INVALID;
    }
    while(s->stack_count>start)pop(s);
    NvmFileFlowValue value=initial_value(result,true,0);
    if(!record)value.arm=variant?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK;
    s->stack[s->stack_count++]=value;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_field(NvmFileFlowState *s,uint16_t field) {
    if(!s || !s->stack_count)return NVM_FILE_FLOW_INVALID;
    if(!s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    NvmFileFlowValue value=s->stack[s->stack_count-1];
    if(value.owner || value.type.category!=NVM_FILE_CATEGORY_RECORD)return NVM_FILE_FLOW_INVALID;
    const NlFilePlanType *type=nl_file_catalog_type(value.type.catalog_ordinal);NvmFileFlowDeclaration result;
    if(!type || field>=type->member_count || !member_type(s,type->members[field].type_id,&result) ||
       !scalar(result.tag) || result.global_index!=NVM_V2_NO_INDEX)return NVM_FILE_FLOW_INVALID;
    s->stack[s->stack_count-1]=initial_value(result,true,0);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_take_result(NvmFileFlowState *s,uint16_t local,NvmFileFlowArm arm) {
    if(!local_valid(s,local))return NVM_FILE_FLOW_INVALID;
    if(!s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    NvmFileFlowValue value=s->locals[local];
    if(!value.initialized || !result_type(value.type) || value.arm!=arm ||
       (arm!=NVM_FILE_FLOW_ARM_OK && arm!=NVM_FILE_FLOW_ARM_ERROR))return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    const NlFilePlanType *type=nl_file_catalog_type(value.type.catalog_ordinal);NvmFileFlowDeclaration payload;
    if(!type || type->member_count!=2 || !member_type(s,type->members[arm==NVM_FILE_FLOW_ARM_ERROR].type_id,&payload))
        return NVM_FILE_FLOW_INVALID;
    if(owned(payload) && (!value.owner || value.type.category!=NVM_FILE_CATEGORY_OPEN_RESULT))return NVM_FILE_FLOW_INVALID;
    s->stack[s->stack_count++]=initial_value(payload,true,owned(payload)?value.owner:0);
    clear_local(s,local);return NVM_FILE_FLOW_OK;
}
static bool live_reference(const NvmFileFlowState *s,uint16_t index,NvmFileFlowDeclaration *out) {
    if(index>=NVM_FILE_FLOW_REFERENCES)return false;
    NvmFileFlowReference r=s->references[index];
    if(!r.live || !local_valid(s,r.local))return false;
    NvmFileFlowValue value=s->locals[r.local];
    if(!value.initialized || value.type.category!=NVM_FILE_CATEGORY_FILE)return false;
    if(r.formal) {
        if(index!=r.local || value.type.mode!=2 || value.owner)return false;
    } else {
        if(value.type.mode || value.owner!=r.owner)return false;
        bool found=false;
        for(uint16_t i=0;i<s->region_count;i++)found|=s->regions[i]==r.region;
        if(!found)return false;
    }
    value.type.mode=2;*out=value.type;return true;
}
static NvmFileFlowInputState input_state(NlFileOwnerState state) {
    return state==NL_FILE_OWNER_PRESERVED?NVM_FILE_FLOW_INPUT_PRESERVED:
           state==NL_FILE_OWNER_CONSUMED?NVM_FILE_FLOW_INPUT_CONSUMED:NVM_FILE_FLOW_INPUT_NONE;
}
NvmFileFlowStatus nvm_file_flow_service(NvmFileFlowState *s,uint32_t site,uint32_t import,uint16_t reference) {
    if(!s)return NVM_FILE_FLOW_INVALID;
    if(!s->reachable)return NVM_FILE_FLOW_UNRESOLVED;
    uint32_t ordinal=NVM_SERVICE_BINDING_COUNT;
    for(uint32_t i=0;i<NVM_SERVICE_BINDING_COUNT;i++) {
        uint32_t index;
        if(nvm_file_nominal_import(s->declarations->nominal,i,&index) && index==import){ordinal=i;break;}
    }
    const NlFilePlanMethod *method=nl_file_catalog_method(ordinal);
    if(!method || !method->param_count)return NVM_FILE_FLOW_INVALID;
    NvmFileFlowDeclaration result,file;
    if(!member_type(s,method->params[method->param_count-1].type_id,&result) || !catalog_type(s,0,&file))
        return NVM_FILE_FLOW_INVALID;
    uint16_t consumed=0;
    NvmFileFlowObligation obligation={0};obligation.kind=NVM_FILE_FLOW_SERVICE;obligation.site=site;
    obligation.target=import;obligation.parameters=(uint16_t)(method->param_count-1);
    obligation.result_count=1;obligation.result=result;
    obligation.required_rights=method->required_rights;obligation.acquired_rights=method->acquired_rights;
    obligation.checks=NVM_FILE_FLOW_CHECK_BINDING|NVM_FILE_FLOW_CHECK_INVOCATION|NVM_FILE_FLOW_CHECK_LIVENESS|
        NVM_FILE_FLOW_CHECK_RIGHTS|NVM_FILE_FLOW_CHECK_RESULT|NVM_FILE_FLOW_CHECK_CLEANUP;
    obligation.outcomes[0]=input_state(method->outcomes[0].input_state);
    obligation.outcomes[1]=input_state(method->outcomes[1].input_state);
    if(method->input_mode==NL_FILE_INPUT_EXCLUSIVE) {
        NvmFileFlowDeclaration ref_type;
        if(!live_reference(s,reference,&ref_type))return NVM_FILE_FLOW_INVALID;
        file.mode=2;if(!same(file,ref_type))return NVM_FILE_FLOW_INVALID;
        obligation.borrowed_inputs=1;obligation.checks|=NVM_FILE_FLOW_CHECK_BORROW;
        if(ordinal==1) {
            if(!s->stack_count || !s->stack[s->stack_count-1].initialized ||
               !same(s->stack[s->stack_count-1].type,scalar_type(TAG_INT)))return NVM_FILE_FLOW_INVALID;
            consumed=1;obligation.checks|=NVM_FILE_FLOW_CHECK_BYTE;
        }
    } else {
        if(reference!=NVM_FILE_FLOW_NO_REFERENCE)return NVM_FILE_FLOW_INVALID;
        if(method->input_mode==NL_FILE_INPUT_CONSUME) {
            if(!s->stack_count)return NVM_FILE_FLOW_INVALID;
            NvmFileFlowValue value=s->stack[s->stack_count-1];
            if(!value.initialized || !value.owner || !same(value.type,file) || held(s,value.owner))return NVM_FILE_FLOW_INVALID;
            consumed=1;obligation.owned_inputs=1;
        }
    }
    if((uint32_t)s->stack_count-consumed+1>NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    if(owned(result)) {
        NvmFileFlowCounts counts;nvm_file_flow_counts(s,&counts);
        if(counts.owners==NVM_FILE_FLOW_OWNERS || s->declarations->next_identity==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    }
    uint16_t slot;NvmFileFlowStatus status=obligation_slot(s,obligation,&slot);if(status!=NVM_FILE_FLOW_OK)return status;
    uint64_t owner=owned(result)?s->declarations->next_identity++:0;
    if(consumed)pop(s);
    s->stack[s->stack_count++]=initial_value(result,true,owner);commit_obligation(s,obligation,slot);
    return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_call(NvmFileFlowState *s,uint32_t site,uint32_t function,
                                    const uint16_t *references,uint16_t count) {
    if(!s || function>=s->declarations->count)return NVM_FILE_FLOW_INVALID;
    if(!s->reachable || function==s->function)return NVM_FILE_FLOW_UNRESOLVED;
    const NvmFileFlowFunction *f=&s->declarations->functions[function];
    if(count!=f->parameters || (count && !references))return NVM_FILE_FLOW_INVALID;
    uint16_t values=0,owners=0,borrows=0;
    const NvmFileFlowDeclaration *params=s->declarations->locals+s->declarations->offsets[function];
    for(uint16_t i=0;i<count;i++){values+=!params[i].mode;owners+=owned(params[i]);borrows+=params[i].mode!=0;}
    if(values>s->stack_count)return NVM_FILE_FLOW_INVALID;
    uint16_t start=(uint16_t)(s->stack_count-values),value_index=start;
    for(uint16_t i=0;i<count;i++) {
        if(params[i].mode) {
            NvmFileFlowDeclaration actual;
            if(!live_reference(s,references[i],&actual) || !same(params[i],actual))return NVM_FILE_FLOW_INVALID;
            for(uint16_t j=0;j<i;j++)if(params[j].mode &&
                s->references[references[j]].owner==s->references[references[i]].owner)return NVM_FILE_FLOW_INVALID;
        } else {
            if(references[i]!=NVM_FILE_FLOW_NO_REFERENCE)return NVM_FILE_FLOW_INVALID;
            NvmFileFlowValue actual=s->stack[value_index++];
            if(!actual.initialized || !same(actual.type,params[i]) || owned(params[i])!=(actual.owner!=0) ||
               (actual.owner && held(s,actual.owner)))return NVM_FILE_FLOW_INVALID;
        }
    }
    if((uint32_t)start+f->result_count>NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    bool owner_result=f->result_count && owned(f->result);
    NvmFileFlowCounts counts;nvm_file_flow_counts(s,&counts);
    if(counts.owners<owners)return NVM_FILE_FLOW_INVALID;
    if((uint32_t)counts.owners-owners+owner_result>NVM_FILE_FLOW_OWNERS ||
       (owner_result && s->declarations->next_identity==UINT64_MAX))return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowObligation obligation={0};obligation.kind=NVM_FILE_FLOW_CALL;obligation.site=site;
    obligation.target=function;obligation.parameters=count;obligation.owned_inputs=owners;obligation.borrowed_inputs=borrows;
    obligation.result_count=f->result_count;obligation.result=f->result;
    obligation.checks=NVM_FILE_FLOW_CHECK_CALLEE|NVM_FILE_FLOW_CHECK_RESULT|NVM_FILE_FLOW_CHECK_CLEANUP;
    uint16_t slot;NvmFileFlowStatus status=obligation_slot(s,obligation,&slot);if(status!=NVM_FILE_FLOW_OK)return status;
    uint64_t owner=owner_result?s->declarations->next_identity++:0;
    while(s->stack_count>start)pop(s);
    if(f->result_count)s->stack[s->stack_count++]=initial_value(f->result,true,owner);
    commit_obligation(s,obligation,slot);return NVM_FILE_FLOW_OK;
}

#include "file_code.inc"

#include "file_body.inc"

#include "file_hosted.inc"
