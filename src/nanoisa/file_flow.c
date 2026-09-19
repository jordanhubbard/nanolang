#include "file_flow.h"
#include "ownership_contracts.h"
#include "nvm_v2_sections.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>

struct NvmFileFlowDeclarations {
    uint32_t references, count;
    NvmFileNominalPlan *nominal;
    NvmFileFlowFunction functions[NVM_FILE_FLOW_FUNCTIONS];
    uint32_t offsets[NVM_FILE_FLOW_FUNCTIONS];
    NvmFileFlowDeclaration locals[];
};
struct NvmFileFlowState {
    NvmFileFlowDeclarations *declarations;
    uint32_t function;
    uint16_t stack_count, region_count;
    uint64_t next_identity, cleanup_obligations;
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
    d->references=1;d->count=m->function_count;d->nominal=nominal;
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
    uint32_t owners=0;
    for(uint16_t i=0;i<d->functions[function].parameters;i++)
        owners+=owned(d->locals[d->offsets[function]+i]);
    if(owners>NVM_FILE_FLOW_OWNERS)return NVM_FILE_FLOW_LIMIT;
    NvmFileFlowState *s=calloc(1,sizeof *s);if(!s)return NVM_FILE_FLOW_MEMORY;
    /* Only owned formals introduce identities in this checkpoint. The256
     * parameter bound also bounds all live owners; moves create none. */
    s->declarations=d;s->function=function;s->next_identity=1;
    const NvmFileFlowFunction *f=&d->functions[function];
    for(uint16_t i=0;i<f->locals;i++) {
        NvmFileFlowDeclaration type=d->locals[d->offsets[function]+i];bool parameter=i<f->parameters;
        s->locals[i]=initial_value(type,parameter,parameter && owned(type)?s->next_identity++:0);
        if(parameter && type.mode) {
            uint64_t identity=s->next_identity++;
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
    NvmFileFlowCounts c={s->stack_count,s->region_count,0,0,s->cleanup_obligations};
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
    if(!s)return NVM_FILE_FLOW_INVALID;
    if(!scalar(tag))return NVM_FILE_FLOW_UNRESOLVED;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count++]=initial_value(scalar_type(tag),true,0);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_load(NvmFileFlowState *s,uint16_t local) {
    if(!local_valid(s,local) || !s->locals[local].initialized || s->locals[local].type.mode ||
       owned(s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count++]=s->locals[local];return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_store(NvmFileFlowState *s,uint16_t local) {
    if(!local_valid(s,local) || !s->stack_count)return NVM_FILE_FLOW_INVALID;
    NvmFileFlowValue value=s->stack[s->stack_count-1];
    if(!value.initialized || value.owner || s->locals[local].type.mode || owned(s->locals[local].type) ||
       !same(value.type,s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    s->locals[local]=value;pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_dup(NvmFileFlowState *s) {
    if(!s || !s->stack_count || s->stack[s->stack_count-1].owner)return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count]=s->stack[s->stack_count-1];s->stack_count++;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_pop(NvmFileFlowState *s) {
    if(!s || !s->stack_count || s->stack[s->stack_count-1].owner)return NVM_FILE_FLOW_INVALID;
    pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_move(NvmFileFlowState *s,uint16_t from,uint16_t to) {
    if(!live_owner(s,from) || !local_valid(s,to) || s->locals[to].initialized ||
       !same(s->locals[from].type,s->locals[to].type))return NVM_FILE_FLOW_INVALID;
    s->locals[to]=s->locals[from];clear_local(s,from);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_take(NvmFileFlowState *s,uint16_t local) {
    if(!live_owner(s,local))return NVM_FILE_FLOW_INVALID;
    if(s->stack_count==NVM_FILE_FLOW_STACK)return NVM_FILE_FLOW_LIMIT;
    s->stack[s->stack_count++]=s->locals[local];clear_local(s,local);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_put(NvmFileFlowState *s,uint16_t local) {
    if(!local_valid(s,local) || s->locals[local].initialized || !s->stack_count)return NVM_FILE_FLOW_INVALID;
    NvmFileFlowValue value=s->stack[s->stack_count-1];
    if(!value.owner || !owned(value.type) || !same(value.type,s->locals[local].type))return NVM_FILE_FLOW_INVALID;
    s->locals[local]=value;pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_drop_local(NvmFileFlowState *s,uint16_t local) {
    if(!live_owner(s,local))return NVM_FILE_FLOW_INVALID;
    if(s->cleanup_obligations==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->cleanup_obligations++;clear_local(s,local);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_drop_stack(NvmFileFlowState *s) {
    if(!s || !s->stack_count || !s->stack[s->stack_count-1].owner)return NVM_FILE_FLOW_INVALID;
    if(s->cleanup_obligations==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->cleanup_obligations++;pop(s);return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_region_begin(NvmFileFlowState *s) {
    if(!s)return NVM_FILE_FLOW_INVALID;
    if(s->region_count==NVM_FILE_FLOW_REFERENCES || s->next_identity==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->regions[s->region_count++]=s->next_identity++;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_region_end(NvmFileFlowState *s) {
    if(!s || !s->region_count)return NVM_FILE_FLOW_INVALID;
    uint64_t region=s->regions[s->region_count-1];
    for(uint16_t i=0;i<NVM_FILE_FLOW_REFERENCES;i++)
        if(s->references[i].live && !s->references[i].formal && s->references[i].region==region)
            s->references[i]=(NvmFileFlowReference){0};
    s->regions[--s->region_count]=0;return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_borrow(NvmFileFlowState *s,uint16_t local,uint16_t reference) {
    if(!live_owner(s,local) || s->locals[local].type.category!=NVM_FILE_CATEGORY_FILE ||
       reference>=NVM_FILE_FLOW_REFERENCES || s->references[reference].live || !s->region_count)
        return NVM_FILE_FLOW_INVALID;
    if(s->next_identity==UINT64_MAX)return NVM_FILE_FLOW_LIMIT;
    s->references[reference]=(NvmFileFlowReference){true,false,local,s->locals[local].owner,
        s->next_identity++,s->regions[s->region_count-1]};return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_end_borrow(NvmFileFlowState *s,uint16_t reference) {
    if(!s || reference>=NVM_FILE_FLOW_REFERENCES || !s->references[reference].live ||
       s->references[reference].formal)return NVM_FILE_FLOW_INVALID;
    s->references[reference]=(NvmFileFlowReference){0};return NVM_FILE_FLOW_OK;
}
NvmFileFlowStatus nvm_file_flow_can_exit(const NvmFileFlowState *s) {
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
