/* I share transactional ownership states across separately checked catalogs. */
#include "ownership_contracts.h"
#include "service_flow_catalog.inc"
#include "nvm_v2_sections.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>

struct FLOW_TYPE(Declarations) {
    uint32_t references, count;
    uint64_t next_identity;
    NominalPlan *nominal;
    FLOW_TYPE(Function) functions[FLOW_CONST(FUNCTIONS)];
    uint32_t offsets[FLOW_CONST(FUNCTIONS)];
    FLOW_TYPE(Declaration) locals[];
};
struct FLOW_TYPE(State) {
    FLOW_TYPE(Declarations) *declarations;
    uint32_t function;
    uint16_t stack_count, region_count, obligation_count;
    uint64_t family, cleanup_obligations;
    bool reachable;
    FLOW_TYPE(Obligation) obligations[FLOW_CONST(OBLIGATIONS)];
    FLOW_TYPE(Value) locals[FLOW_CONST(LOCALS)];
    FLOW_TYPE(Value) stack[FLOW_CONST(STACK)];
    FLOW_TYPE(Reference) references[FLOW_CONST(REFERENCES)];
    uint64_t regions[FLOW_CONST(REFERENCES)];
};
static bool scalar(uint8_t tag) { return tag==TAG_INT || tag==TAG_BOOL || tag==TAG_VOID; }
static bool owned(FLOW_TYPE(Declaration) d) {
    return !d.mode && (d.category==NOMINAL_CATEGORY_OWNER || d.category==NOMINAL_CATEGORY_OWNER_RESULT);
}
static bool same(FLOW_TYPE(Declaration) a,FLOW_TYPE(Declaration) b) {
    return a.tag==b.tag && a.mode==b.mode && a.global_index==b.global_index &&
           a.catalog_ordinal==b.catalog_ordinal && a.category==b.category;
}
static FLOW_TYPE(Declaration) scalar_type(uint8_t tag) {
    return (FLOW_TYPE(Declaration)){tag,0,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX,NOMINAL_CATEGORY_UNKNOWN};
}
static FLOW_TYPE(Status) read_declaration(NvmV2Cursor *c,const NominalPlan *p,
                                         FLOW_TYPE(Declaration) *out,bool callables) {
    uint8_t tag,mode;uint16_t pad;uint32_t index;
    if(nvm_v2_u8(c,&tag)!=NVM_V2_OK || nvm_v2_u8(c,&mode)!=NVM_V2_OK ||
       nvm_v2_u16(c,&pad)!=NVM_V2_OK || nvm_v2_u32(c,&index)!=NVM_V2_OK || pad)
        return FLOW_CONST(INVALID);
    FLOW_TYPE(Declaration) d=scalar_type(tag);d.mode=mode;d.global_index=index;
    if(index==NVM_V2_NO_INDEX) {
        if(mode || (!scalar(tag) && !(callables && tag==TAG_FUNCTION)))return FLOW_CONST(UNRESOLVED);
    } else {
        NominalLayout layout;
        if(!NOMINAL_FN(layout)(p,index,&layout))return FLOW_CONST(INVALID);
        if(layout.category==NOMINAL_CATEGORY_UNKNOWN)return FLOW_CONST(UNRESOLVED);
        d.catalog_ordinal=layout.catalog_ordinal;d.category=layout.category;
        if((mode && ((mode!=1 && mode!=2) || d.category!=NOMINAL_CATEGORY_OWNER)) ||
           tag!=(layout.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION))
            return FLOW_CONST(INVALID);
    }
    *out=d;return FLOW_CONST(OK);
}
static FLOW_TYPE(Status) nominal_status(NominalStatus s) {
    switch(s) {
    case NOMINAL_DESCRIBED:return FLOW_CONST(OK);
    case NOMINAL_LIMIT:return FLOW_CONST(LIMIT);
    case NOMINAL_MEMORY:return FLOW_CONST(MEMORY);
    default:return FLOW_CONST(INVALID);
    }
}
static FLOW_TYPE(Status) file_flow_declarations_mode(const NvmModule *m,FLOW_TYPE(Declarations) **out,bool callables) {
    if(!m || !out || !m->function_count || !m->functions)return FLOW_CONST(INVALID);
    if(m->function_count>FLOW_CONST(FUNCTIONS))return FLOW_CONST(LIMIT);
    if(m->callback_contract_count || m->module_ref_count)return FLOW_CONST(UNRESOLVED);
    size_t count=0;
    for(uint32_t i=0;i<m->function_count;i++) {
        const NvmFunctionEntry *f=&m->functions[i];
        if(f->arity>f->local_count || f->result_count>1)return FLOW_CONST(INVALID);
        if(f->upvalue_count)return FLOW_CONST(UNRESOLVED);
        if(f->local_count>FLOW_CONST(LOCALS))return FLOW_CONST(LIMIT);
        if(count>SIZE_MAX-f->local_count)return FLOW_CONST(LIMIT);
        count+=f->local_count;
    }
    if(count>(SIZE_MAX-sizeof(FLOW_TYPE(Declarations)))/sizeof(FLOW_TYPE(Declaration)))
        return FLOW_CONST(LIMIT);
    size_t bytes=sizeof(FLOW_TYPE(Declarations))+count*sizeof(FLOW_TYPE(Declaration));
    /* I account for the nominal allocator's exact maximum extent before calling
     * it; no duplicated private struct size can drift from that allocation. */
    size_t nominal_bound;
    if(!NOMINAL_FN(storage_bound)(NOMINAL_MAX_LAYOUTS,&nominal_bound) ||
       bytes>FLOW_CONST(BYTES) || nominal_bound>FLOW_CONST(BYTES)-bytes)return FLOW_CONST(LIMIT);
    NominalPlan *nominal=NULL;
    FLOW_TYPE(Status) status=nominal_status(NOMINAL_FN(plan)(m,&nominal));
    if(status!=FLOW_CONST(OK))return status;
    FLOW_TYPE(Declarations) *d=calloc(1,bytes);
    if(!d){NOMINAL_FN(plan_free)(nominal);return FLOW_CONST(MEMORY);}
    d->references=1;d->count=m->function_count;d->nominal=nominal;d->next_identity=1;
    NvmV2Cursor c;nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    uint32_t version,layouts,functions;const uint8_t *flags;
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || version!=NVM_OWNERSHIP_VERSION ||
       nvm_v2_u32(&c,&layouts)!=NVM_V2_OK || layouts!=NOMINAL_FN(layout_count)(nominal) ||
       nvm_v2_take(&c,layouts,&flags)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&functions)!=NVM_V2_OK || functions!=d->count) {
        status=FLOW_CONST(INVALID);goto fail;
    }
    size_t offset=0;
    for(uint32_t i=0;i<d->count;i++) {
        FLOW_TYPE(Function) *f=&d->functions[i];const NvmFunctionEntry *actual=&m->functions[i];
        if(nvm_v2_u16(&c,&f->locals)!=NVM_V2_OK || nvm_v2_u16(&c,&f->parameters)!=NVM_V2_OK ||
           f->locals!=actual->local_count || f->parameters!=actual->arity) {
            status=FLOW_CONST(INVALID);goto fail;
        }
        d->offsets[i]=(uint32_t)offset;f->result_count=actual->result_count;
        status=read_declaration(&c,nominal,&f->result,callables);if(status!=FLOW_CONST(OK))goto fail;
        if(f->result.mode || f->result.tag!=(f->result_count?actual->result_tag:TAG_VOID)) {
            status=FLOW_CONST(INVALID);goto fail;
        }
        for(uint16_t j=0;j<f->locals;j++) {
            status=read_declaration(&c,nominal,&d->locals[offset+j],callables);if(status!=FLOW_CONST(OK))goto fail;
            if(j>=f->parameters && d->locals[offset+j].mode) {status=FLOW_CONST(INVALID);goto fail;}
        }
        offset+=f->locals;
    }
    if(offset!=count || c.pos!=c.size){status=FLOW_CONST(INVALID);goto fail;}
    *out=d;return FLOW_CONST(OK);
fail:
    FLOW_FN(declarations_free)(d);return status;
}
FLOW_TYPE(Status) FLOW_FN(declarations)(const NvmModule *m,FLOW_TYPE(Declarations) **out) {
    return file_flow_declarations_mode(m,out,false);
}
void FLOW_FN(declarations_free)(FLOW_TYPE(Declarations) *d) {
    if(d && --d->references==0){NOMINAL_FN(plan_free)(d->nominal);free(d);}
}
bool FLOW_FN(function)(const FLOW_TYPE(Declarations) *d,uint32_t f,FLOW_TYPE(Function) *out) {
    if(!d || !out || f>=d->count)return false;
    *out=d->functions[f];return true;
}
bool FLOW_FN(declaration)(const FLOW_TYPE(Declarations) *d,uint32_t f,uint16_t local,
                              FLOW_TYPE(Declaration) *out) {
    if(!d || !out || f>=d->count || local>=d->functions[f].locals)return false;
    *out=d->locals[d->offsets[f]+local];return true;
}
static FLOW_TYPE(Value) initial_value(FLOW_TYPE(Declaration) type,bool initialized,uint64_t owner) {
    FLOW_TYPE(Arm) arm=type.category==NOMINAL_CATEGORY_OWNER_RESULT ||
                       type.category==NOMINAL_CATEGORY_SCALAR_RESULT ? FLOW_CONST(ARM_UNKNOWN):FLOW_CONST(ARM_NONE);
    return (FLOW_TYPE(Value)){type,initialized,owner,arm};
}
FLOW_TYPE(Status) FLOW_FN(state)(FLOW_TYPE(Declarations) *d,uint32_t function,FLOW_TYPE(State) **out) {
    if(!d || !out || function>=d->count)return FLOW_CONST(INVALID);
    if(d->references==UINT32_MAX || sizeof(FLOW_TYPE(State))>FLOW_CONST(BYTES))return FLOW_CONST(LIMIT);
    uint32_t owners=0,identities=1;
    for(uint16_t i=0;i<d->functions[function].parameters;i++) {
        FLOW_TYPE(Declaration) type=d->locals[d->offsets[function]+i];
        owners+=owned(type);identities+=owned(type) || type.mode;
    }
    if(owners>FLOW_CONST(OWNERS) || identities>UINT64_MAX-d->next_identity)return FLOW_CONST(LIMIT);
    FLOW_TYPE(State) *s=calloc(1,sizeof *s);if(!s)return FLOW_CONST(MEMORY);
    s->declarations=d;s->function=function;s->family=d->next_identity++;s->reachable=true;
    const FLOW_TYPE(Function) *f=&d->functions[function];
    for(uint16_t i=0;i<f->locals;i++) {
        FLOW_TYPE(Declaration) type=d->locals[d->offsets[function]+i];bool parameter=i<f->parameters;
        s->locals[i]=initial_value(type,parameter,parameter && owned(type)?d->next_identity++:0);
        if(parameter && type.mode) {
            uint64_t identity=d->next_identity++;
            s->references[i]=(FLOW_TYPE(Reference)){true,true,i,identity,identity,0,type.mode==1};
        }
    }
    d->references++;*out=s;return FLOW_CONST(OK);
}
void FLOW_FN(state_free)(FLOW_TYPE(State) *s) {
    if(s){FLOW_FN(declarations_free)(s->declarations);free(s);}
}
static bool local_valid(const FLOW_TYPE(State) *s,uint16_t local) {
    return s && local<s->declarations->functions[s->function].locals;
}
bool FLOW_FN(local)(const FLOW_TYPE(State) *s,uint16_t local,FLOW_TYPE(Value) *out) {
    if(!out || !local_valid(s,local))return false;
    *out=s->locals[local];return true;
}
bool FLOW_FN(stack)(const FLOW_TYPE(State) *s,uint16_t index,FLOW_TYPE(Value) *out) {
    if(!s || !out || index>=s->stack_count)return false;
    *out=s->stack[index];return true;
}
bool FLOW_FN(reference)(const FLOW_TYPE(State) *s,uint16_t reference,FLOW_TYPE(Reference) *out) {
    if(!s || !out || reference>=FLOW_CONST(REFERENCES))return false;
    *out=s->references[reference];return true;
}
bool FLOW_FN(counts)(const FLOW_TYPE(State) *s,FLOW_TYPE(Counts) *out) {
    if(!s || !out)return false;
    FLOW_TYPE(Counts) c={s->stack_count,s->region_count,0,0,s->cleanup_obligations,s->obligation_count,s->reachable};
    for(uint16_t i=0;i<s->declarations->functions[s->function].locals;i++)c.owners+=s->locals[i].owner!=0;
    for(uint16_t i=0;i<s->stack_count;i++)c.owners+=s->stack[i].owner!=0;
    for(uint16_t i=0;i<FLOW_CONST(REFERENCES);i++)c.references+=s->references[i].live;
    *out=c;return true;
}
static bool held(const FLOW_TYPE(State) *s,uint64_t owner) {
    for(uint16_t i=0;i<FLOW_CONST(REFERENCES);i++)
        if(s->references[i].live && s->references[i].owner==owner)return true;
    return false;
}
static bool live_owner(const FLOW_TYPE(State) *s,uint16_t local) {
    return local_valid(s,local) && s->locals[local].initialized && s->locals[local].owner &&
           owned(s->locals[local].type) && !held(s,s->locals[local].owner);
}
static void clear_local(FLOW_TYPE(State) *s,uint16_t local) {
    s->locals[local]=initial_value(s->locals[local].type,false,0);
}
static void pop(FLOW_TYPE(State) *s) { s->stack[--s->stack_count]=(FLOW_TYPE(Value)){0}; }
FLOW_TYPE(Status) FLOW_FN(push_scalar)(FLOW_TYPE(State) *s,uint8_t tag) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s)return FLOW_CONST(INVALID);
    if(!scalar(tag))return FLOW_CONST(UNRESOLVED);
    if(s->stack_count==FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    s->stack[s->stack_count++]=initial_value(scalar_type(tag),true,0);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(load)(FLOW_TYPE(State) *s,uint16_t local) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!local_valid(s,local) || !s->locals[local].initialized || s->locals[local].type.mode ||
       owned(s->locals[local].type))return FLOW_CONST(INVALID);
    if(s->stack_count==FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    s->stack[s->stack_count++]=s->locals[local];return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(store)(FLOW_TYPE(State) *s,uint16_t local) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!local_valid(s,local) || !s->stack_count)return FLOW_CONST(INVALID);
    FLOW_TYPE(Value) value=s->stack[s->stack_count-1];
    /* I use PUSH_VOID/STORE_LOCAL to end a copy local's lifetime. This
     * cannot erase an affine owner or a formal loan. */
    if(value.initialized && !value.owner && value.type.tag==TAG_VOID &&
       !value.type.mode && value.type.global_index==NVM_V2_NO_INDEX &&
       s->locals[local].type.tag!=TAG_VOID) {
        FLOW_TYPE(Status) status=FLOW_FN(clear_copy)(s,local);
        if(status==FLOW_CONST(OK))pop(s);
        return status;
    }
    if(!value.initialized || value.owner || s->locals[local].type.mode || owned(s->locals[local].type) ||
       !same(value.type,s->locals[local].type))return FLOW_CONST(INVALID);
    s->locals[local]=value;pop(s);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(dup)(FLOW_TYPE(State) *s) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s || !s->stack_count || s->stack[s->stack_count-1].owner)return FLOW_CONST(INVALID);
    if(s->stack_count==FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    s->stack[s->stack_count]=s->stack[s->stack_count-1];s->stack_count++;return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(pop)(FLOW_TYPE(State) *s) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s || !s->stack_count || s->stack[s->stack_count-1].owner)return FLOW_CONST(INVALID);
    pop(s);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(move)(FLOW_TYPE(State) *s,uint16_t from,uint16_t to) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!live_owner(s,from) || !local_valid(s,to) || s->locals[to].initialized ||
       !same(s->locals[from].type,s->locals[to].type))return FLOW_CONST(INVALID);
    s->locals[to]=s->locals[from];
    if(s->locals[to].type.category==NOMINAL_CATEGORY_OWNER_RESULT)s->locals[to].arm=FLOW_CONST(ARM_UNKNOWN);
    clear_local(s,from);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(take)(FLOW_TYPE(State) *s,uint16_t local) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!live_owner(s,local))return FLOW_CONST(INVALID);
    if(s->stack_count==FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    s->stack[s->stack_count]=s->locals[local];
    if(s->stack[s->stack_count].type.category==NOMINAL_CATEGORY_OWNER_RESULT)s->stack[s->stack_count].arm=FLOW_CONST(ARM_UNKNOWN);
    s->stack_count++;clear_local(s,local);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(put)(FLOW_TYPE(State) *s,uint16_t local) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!local_valid(s,local) || s->locals[local].initialized || !s->stack_count)return FLOW_CONST(INVALID);
    FLOW_TYPE(Value) value=s->stack[s->stack_count-1];
    if(!value.owner || !owned(value.type) || !same(value.type,s->locals[local].type))return FLOW_CONST(INVALID);
    if(value.type.category==NOMINAL_CATEGORY_OWNER_RESULT)value.arm=FLOW_CONST(ARM_UNKNOWN);
    s->locals[local]=value;pop(s);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(drop_local)(FLOW_TYPE(State) *s,uint16_t local) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!live_owner(s,local))return FLOW_CONST(INVALID);
    if(s->cleanup_obligations==UINT64_MAX)return FLOW_CONST(LIMIT);
    s->cleanup_obligations++;clear_local(s,local);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(drop_stack)(FLOW_TYPE(State) *s) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s || !s->stack_count || !s->stack[s->stack_count-1].owner)return FLOW_CONST(INVALID);
    if(s->cleanup_obligations==UINT64_MAX)return FLOW_CONST(LIMIT);
    s->cleanup_obligations++;pop(s);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(region_begin)(FLOW_TYPE(State) *s) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s)return FLOW_CONST(INVALID);
    if(s->region_count==FLOW_CONST(REFERENCES) || s->declarations->next_identity==UINT64_MAX)return FLOW_CONST(LIMIT);
    s->regions[s->region_count++]=s->declarations->next_identity++;return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(region_end)(FLOW_TYPE(State) *s) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s || !s->region_count)return FLOW_CONST(INVALID);
    uint64_t region=s->regions[s->region_count-1];
    for(uint16_t i=0;i<FLOW_CONST(REFERENCES);i++)
        if(s->references[i].live && !s->references[i].formal && s->references[i].region==region)
            s->references[i]=(FLOW_TYPE(Reference)){0};
    s->regions[--s->region_count]=0;return FLOW_CONST(OK);
}
static FLOW_TYPE(Status) file_flow_borrow_mode(FLOW_TYPE(State) *s,uint16_t local,uint16_t reference,bool shared) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!local_valid(s,local) || !s->locals[local].initialized || !s->locals[local].owner ||
       s->locals[local].type.mode || s->locals[local].type.category!=NOMINAL_CATEGORY_OWNER ||
       reference>=FLOW_CONST(REFERENCES) || s->references[reference].live || !s->region_count)
        return FLOW_CONST(INVALID);
    for(uint16_t i=0;i<FLOW_CONST(REFERENCES);i++)
        if(s->references[i].live && s->references[i].owner==s->locals[local].owner &&
           (!shared || !s->references[i].shared))return FLOW_CONST(INVALID);
    if(s->declarations->next_identity==UINT64_MAX)return FLOW_CONST(LIMIT);
    s->references[reference]=(FLOW_TYPE(Reference)){true,false,local,s->locals[local].owner,
        s->declarations->next_identity++,s->regions[s->region_count-1],shared};return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(borrow)(FLOW_TYPE(State) *s,uint16_t local,uint16_t reference) {
    return file_flow_borrow_mode(s,local,reference,false);
}
FLOW_TYPE(Status) FLOW_FN(borrow_shared)(FLOW_TYPE(State) *s,uint16_t local,uint16_t reference) {
    return file_flow_borrow_mode(s,local,reference,true);
}
FLOW_TYPE(Status) FLOW_FN(end_borrow)(FLOW_TYPE(State) *s,uint16_t reference) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s || reference>=FLOW_CONST(REFERENCES) || !s->references[reference].live ||
       s->references[reference].formal)return FLOW_CONST(INVALID);
    s->references[reference]=(FLOW_TYPE(Reference)){0};return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(can_exit)(const FLOW_TYPE(State) *s) {
    if(s && !s->reachable)return FLOW_CONST(UNRESOLVED);
    if(!s)return FLOW_CONST(INVALID);
    const FLOW_TYPE(Function) *f=&s->declarations->functions[s->function];
    if(s->region_count || s->stack_count!=f->result_count)return FLOW_CONST(INVALID);
    if(f->result_count) {
        FLOW_TYPE(Value) value=s->stack[0];
        if(!value.initialized || !same(value.type,f->result) ||
           (owned(value.type)!=(value.owner!=0)))return FLOW_CONST(INVALID);
    }
    for(uint16_t i=0;i<f->locals;i++)if(s->locals[i].owner)return FLOW_CONST(INVALID);
    for(uint16_t i=0;i<FLOW_CONST(REFERENCES);i++)
        if(s->references[i].live && !s->references[i].formal)return FLOW_CONST(INVALID);
    return FLOW_CONST(OK);
}

static bool result_type(FLOW_TYPE(Declaration) type) {
    return type.category==NOMINAL_CATEGORY_OWNER_RESULT || type.category==NOMINAL_CATEGORY_SCALAR_RESULT;
}
static bool catalog_type(const FLOW_TYPE(State) *s,uint32_t ordinal,FLOW_TYPE(Declaration) *out) {
    NominalLayout l;
    if(!NOMINAL_FN(type)(s->declarations->nominal,ordinal,&l))return false;
    *out=(FLOW_TYPE(Declaration)){l.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION,
        0,l.global_index,l.catalog_ordinal,l.category};return true;
}
static bool member_type(const FLOW_TYPE(State) *s,uint32_t context,const char *id,FLOW_TYPE(Declaration) *out) {
    if(!id){*out=scalar_type(TAG_VOID);return true;}
    if(!strcmp(id,"nsi:core/int")){*out=scalar_type(TAG_INT);return true;}
    if(!strcmp(id,"nsi:core/bool")){*out=scalar_type(TAG_BOOL);return true;}
    for(uint32_t i=FLOW_TYPE_BASE(context);i<FLOW_TYPE_BASE(context)+FLOW_TYPE_COUNT(s->declarations->nominal,context);i++) {
        const NlServicePlanType *t=FLOW_CATALOG_TYPE(s->declarations->nominal,i);
        if(t && !strcmp(t->id,id))return catalog_type(s,i,out);
    }
    return false;
}
static bool value_same(FLOW_TYPE(Value) a,FLOW_TYPE(Value) b) {
    return same(a.type,b.type) && a.initialized==b.initialized && a.owner==b.owner;
}
static bool reference_same(FLOW_TYPE(Reference) a,FLOW_TYPE(Reference) b) {
    return a.live==b.live && (!a.live || (a.formal==b.formal && a.local==b.local &&
        a.owner==b.owner && a.identity==b.identity && a.region==b.region && a.shared==b.shared));
}
static bool obligation_same(FLOW_TYPE(Obligation) a,FLOW_TYPE(Obligation) b) {
    return a.kind==b.kind && a.site==b.site && a.target==b.target && a.checks==b.checks &&
        a.required_rights==b.required_rights && a.acquired_rights==b.acquired_rights &&
        a.parameters==b.parameters && a.owned_inputs==b.owned_inputs && a.borrowed_inputs==b.borrowed_inputs &&
        a.result_count==b.result_count && same(a.result,b.result) &&
        a.outcomes[0]==b.outcomes[0] && a.outcomes[1]==b.outcomes[1];
}
static FLOW_TYPE(Status) obligation_slot(const FLOW_TYPE(State) *s,FLOW_TYPE(Obligation) item,uint16_t *slot) {
    if(item.site==UINT32_MAX)return FLOW_CONST(INVALID);
    for(uint16_t i=0;i<s->obligation_count;i++)if(s->obligations[i].site==item.site) {
        if(!obligation_same(s->obligations[i],item))return FLOW_CONST(UNRESOLVED);
        *slot=i;return FLOW_CONST(OK);
    }
    if(s->obligation_count==FLOW_CONST(OBLIGATIONS))return FLOW_CONST(LIMIT);
    *slot=s->obligation_count;return FLOW_CONST(OK);
}
static void commit_obligation(FLOW_TYPE(State) *s,FLOW_TYPE(Obligation) item,uint16_t slot) {
    if(slot==s->obligation_count)s->obligations[s->obligation_count++]=item;
}
bool FLOW_FN(obligation)(const FLOW_TYPE(State) *s,uint16_t index,FLOW_TYPE(Obligation) *out) {
    if(!s || !out || index>=s->obligation_count)return false;
    *out=s->obligations[index];return true;
}
FLOW_TYPE(Status) FLOW_FN(clone)(const FLOW_TYPE(State) *s,FLOW_TYPE(State) **out) {
    if(!s || !out)return FLOW_CONST(INVALID);
    if(s->declarations->references==UINT32_MAX)return FLOW_CONST(LIMIT);
    FLOW_TYPE(State) *copy=malloc(sizeof *copy);if(!copy)return FLOW_CONST(MEMORY);
    *copy=*s;s->declarations->references++;*out=copy;return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(refine)(const FLOW_TYPE(State) *s,uint16_t local,
                                      FLOW_TYPE(State) **ok,FLOW_TYPE(State) **error) {
    if(!s || !ok || !error || ok==error || !local_valid(s,local))return FLOW_CONST(INVALID);
    if(!s->reachable)return FLOW_CONST(UNRESOLVED);
    FLOW_TYPE(Value) value=s->locals[local];
    if(!value.initialized || !result_type(value.type) || value.type.mode)return FLOW_CONST(INVALID);
    if(s->declarations->references>UINT32_MAX-2)return FLOW_CONST(LIMIT);
    FLOW_TYPE(State) *a=NULL,*b=NULL;
    FLOW_TYPE(Status) status=FLOW_FN(clone)(s,&a);if(status!=FLOW_CONST(OK))return status;
    status=FLOW_FN(clone)(s,&b);
    if(status!=FLOW_CONST(OK)){FLOW_FN(state_free)(a);return status;}
    a->reachable=value.arm!=FLOW_CONST(ARM_ERROR);b->reachable=value.arm!=FLOW_CONST(ARM_OK);
    a->locals[local].arm=FLOW_CONST(ARM_OK);b->locals[local].arm=FLOW_CONST(ARM_ERROR);
    *ok=a;*error=b;return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(join)(FLOW_TYPE(State) *s,const FLOW_TYPE(State) *incoming,bool *changed) {
    if(!s || !incoming || !changed)return FLOW_CONST(INVALID);
    if(s->declarations!=incoming->declarations || s->function!=incoming->function || s->family!=incoming->family)
        return FLOW_CONST(UNRESOLVED);
    if(!incoming->reachable){*changed=false;return FLOW_CONST(OK);}
    if(!s->reachable){*s=*incoming;*changed=true;return FLOW_CONST(OK);}
    if(s->stack_count!=incoming->stack_count || s->region_count!=incoming->region_count ||
       s->cleanup_obligations!=incoming->cleanup_obligations)return FLOW_CONST(UNRESOLVED);
    bool differs=false;
    uint16_t locals=s->declarations->functions[s->function].locals;
    for(uint16_t i=0;i<locals;i++) {
        if(!value_same(s->locals[i],incoming->locals[i]))return FLOW_CONST(UNRESOLVED);
        if(s->locals[i].initialized && s->locals[i].arm!=incoming->locals[i].arm) {
            if(!result_type(s->locals[i].type))return FLOW_CONST(UNRESOLVED);
            differs|=s->locals[i].arm!=FLOW_CONST(ARM_UNKNOWN);
        }
    }
    for(uint16_t i=0;i<s->stack_count;i++) {
        if(!value_same(s->stack[i],incoming->stack[i]))return FLOW_CONST(UNRESOLVED);
        if(s->stack[i].arm!=incoming->stack[i].arm) {
            if(!result_type(s->stack[i].type))return FLOW_CONST(UNRESOLVED);
            differs|=s->stack[i].arm!=FLOW_CONST(ARM_UNKNOWN);
        }
    }
    for(uint16_t i=0;i<s->region_count;i++)if(s->regions[i]!=incoming->regions[i])return FLOW_CONST(UNRESOLVED);
    for(uint16_t i=0;i<FLOW_CONST(REFERENCES);i++)
        if(!reference_same(s->references[i],incoming->references[i]))return FLOW_CONST(UNRESOLVED);
    uint16_t add=0;
    for(uint16_t i=0;i<incoming->obligation_count;i++) {
        bool found=false;
        for(uint16_t j=0;j<s->obligation_count;j++)if(s->obligations[j].site==incoming->obligations[i].site) {
            if(!obligation_same(s->obligations[j],incoming->obligations[i]))return FLOW_CONST(UNRESOLVED);
            found=true;break;
        }
        add+=!found;
    }
    if((uint32_t)s->obligation_count+add>FLOW_CONST(OBLIGATIONS))return FLOW_CONST(LIMIT);
    /* All comparisons and capacity checks precede this allocation-free commit. */
    for(uint16_t i=0;i<locals;i++)if(s->locals[i].initialized && s->locals[i].arm!=incoming->locals[i].arm)
        s->locals[i].arm=FLOW_CONST(ARM_UNKNOWN);
    for(uint16_t i=0;i<s->stack_count;i++)if(s->stack[i].arm!=incoming->stack[i].arm)
        s->stack[i].arm=FLOW_CONST(ARM_UNKNOWN);
    for(uint16_t i=0;i<incoming->obligation_count;i++) {
        bool found=false;
        for(uint16_t j=0;j<s->obligation_count;j++)if(s->obligations[j].site==incoming->obligations[i].site){found=true;break;}
        if(!found)s->obligations[s->obligation_count++]=incoming->obligations[i];
    }
    *changed=differs || add!=0;return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(clear_copy)(FLOW_TYPE(State) *s,uint16_t local) {
    if(!local_valid(s,local))return FLOW_CONST(INVALID);
    if(!s->reachable)return FLOW_CONST(UNRESOLVED);
    if(s->locals[local].type.mode || owned(s->locals[local].type))return FLOW_CONST(INVALID);
    clear_local(s,local);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(construct)(FLOW_TYPE(State) *s,uint32_t ordinal,uint16_t variant) {
    if(!s)return FLOW_CONST(INVALID);
    if(!s->reachable)return FLOW_CONST(UNRESOLVED);
    FLOW_TYPE(Declaration) result;
    if(!catalog_type(s,ordinal,&result))return FLOW_CONST(INVALID);
    if(owned(result))return FLOW_CONST(INVALID);
    const NlServicePlanType *type=FLOW_CATALOG_TYPE(s->declarations->nominal,ordinal);
    bool record=result.category==NOMINAL_CATEGORY_RECORD;
    if(!type || (record?variant!=0:variant>=type->member_count))return FLOW_CONST(INVALID);
    size_t count=record?type->member_count:1;
    if(count>s->stack_count)return FLOW_CONST(INVALID);
    if((size_t)s->stack_count-count+1>FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    uint16_t start=(uint16_t)(s->stack_count-count);
    for(size_t i=0;i<count;i++) {
        FLOW_TYPE(Declaration) expected;const NlServicePlanMember *member=&type->members[record?i:variant];
        if(!member_type(s,ordinal,member->type_id,&expected) || owned(expected))return FLOW_CONST(INVALID);
        FLOW_TYPE(Value) value=s->stack[start+i];
        if(!value.initialized || value.owner || !same(expected,value.type))return FLOW_CONST(INVALID);
    }
    while(s->stack_count>start)pop(s);
    FLOW_TYPE(Value) value=initial_value(result,true,0);
    if(!record)value.arm=variant?FLOW_CONST(ARM_ERROR):FLOW_CONST(ARM_OK);
    s->stack[s->stack_count++]=value;return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(field)(FLOW_TYPE(State) *s,uint16_t field) {
    if(!s || !s->stack_count)return FLOW_CONST(INVALID);
    if(!s->reachable)return FLOW_CONST(UNRESOLVED);
    FLOW_TYPE(Value) value=s->stack[s->stack_count-1];
    if(value.owner || value.type.category!=NOMINAL_CATEGORY_RECORD)return FLOW_CONST(INVALID);
    const NlServicePlanType *type=FLOW_CATALOG_TYPE(s->declarations->nominal,value.type.catalog_ordinal);FLOW_TYPE(Declaration) result;
    if(!type || field>=type->member_count || !member_type(s,value.type.catalog_ordinal,type->members[field].type_id,&result) ||
       !scalar(result.tag) || result.global_index!=NVM_V2_NO_INDEX)return FLOW_CONST(INVALID);
    s->stack[s->stack_count-1]=initial_value(result,true,0);return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(take_result)(FLOW_TYPE(State) *s,uint16_t local,FLOW_TYPE(Arm) arm) {
    if(!local_valid(s,local))return FLOW_CONST(INVALID);
    if(!s->reachable)return FLOW_CONST(UNRESOLVED);
    FLOW_TYPE(Value) value=s->locals[local];
    if(!value.initialized || !result_type(value.type) || value.arm!=arm ||
       (arm!=FLOW_CONST(ARM_OK) && arm!=FLOW_CONST(ARM_ERROR)))return FLOW_CONST(INVALID);
    if(s->stack_count==FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    const NlServicePlanType *type=FLOW_CATALOG_TYPE(s->declarations->nominal,value.type.catalog_ordinal);FLOW_TYPE(Declaration) payload;
    if(!type || type->member_count!=2 || !member_type(s,value.type.catalog_ordinal,type->members[arm==FLOW_CONST(ARM_ERROR)].type_id,&payload))
        return FLOW_CONST(INVALID);
    if(owned(payload) && (!value.owner || value.type.category!=NOMINAL_CATEGORY_OWNER_RESULT))return FLOW_CONST(INVALID);
    s->stack[s->stack_count++]=initial_value(payload,true,owned(payload)?value.owner:0);
    clear_local(s,local);return FLOW_CONST(OK);
}
static bool live_reference(const FLOW_TYPE(State) *s,uint16_t index,FLOW_TYPE(Declaration) *out) {
    if(index>=FLOW_CONST(REFERENCES))return false;
    FLOW_TYPE(Reference) r=s->references[index];
    if(!r.live || !local_valid(s,r.local))return false;
    FLOW_TYPE(Value) value=s->locals[r.local];
    if(!value.initialized || value.type.category!=NOMINAL_CATEGORY_OWNER)return false;
    if(r.formal) {
        if(index!=r.local || value.type.mode!=(r.shared?1:2) || value.owner)return false;
    } else {
        if(value.type.mode || value.owner!=r.owner)return false;
        bool found=false;
        for(uint16_t i=0;i<s->region_count;i++)found|=s->regions[i]==r.region;
        if(!found)return false;
    }
    value.type.mode=r.shared?1:2;*out=value.type;return true;
}
static FLOW_TYPE(InputState) input_state(NlServiceOwnerState state) {
    return state==NL_SERVICE_OWNER_PRESERVED?FLOW_CONST(INPUT_PRESERVED):
           state==NL_SERVICE_OWNER_CONSUMED?FLOW_CONST(INPUT_CONSUMED):FLOW_CONST(INPUT_NONE);
}
FLOW_TYPE(Status) FLOW_FN(service)(FLOW_TYPE(State) *s,uint32_t site,uint32_t import,uint16_t reference) {
    if(!s)return FLOW_CONST(INVALID);
    if(!s->reachable)return FLOW_CONST(UNRESOLVED);
    uint32_t ordinal=FLOW_IMPORT_COUNT(s->declarations->nominal);
    for(uint32_t i=0;i<FLOW_IMPORT_COUNT(s->declarations->nominal);i++) {
        uint32_t index;
        if(NOMINAL_FN(import)(s->declarations->nominal,i,&index) && index==import){ordinal=i;break;}
    }
    const NlServicePlanMethod *method=FLOW_CATALOG_METHOD(s->declarations->nominal,ordinal);
    if(!method || !method->param_count)return FLOW_CONST(INVALID);
    FLOW_TYPE(Declaration) result,file;
    if(!member_type(s,FLOW_METHOD_OWNER(ordinal),method->params[method->param_count-1].type_id,&result) || !catalog_type(s,FLOW_METHOD_OWNER(ordinal),&file))
        return FLOW_CONST(INVALID);
    uint16_t consumed=0;
    FLOW_TYPE(Obligation) obligation={0};obligation.kind=FLOW_CONST(SERVICE);obligation.site=site;
    obligation.target=import;obligation.parameters=(uint16_t)(method->param_count-1);
    obligation.result_count=1;obligation.result=result;
    obligation.required_rights=method->required_rights;obligation.acquired_rights=method->acquired_rights;
    obligation.checks=FLOW_CONST(CHECK_BINDING)|FLOW_CONST(CHECK_INVOCATION)|FLOW_CONST(CHECK_LIVENESS)|
        FLOW_CONST(CHECK_RIGHTS)|FLOW_CONST(CHECK_RESULT)|FLOW_CONST(CHECK_CLEANUP);
    obligation.outcomes[0]=input_state(method->outcomes[0].input_state);
    obligation.outcomes[1]=input_state(method->outcomes[1].input_state);
    if(method->input_mode==NL_SERVICE_INPUT_EXCLUSIVE) {
        FLOW_TYPE(Declaration) ref_type;
        if(!live_reference(s,reference,&ref_type))return FLOW_CONST(INVALID);
        file.mode=2;if(!same(file,ref_type))return FLOW_CONST(INVALID);
        obligation.borrowed_inputs=1;obligation.checks|=FLOW_CONST(CHECK_BORROW);
        if(FLOW_METHOD_LOCAL(ordinal)==1) {
            if(!s->stack_count || !s->stack[s->stack_count-1].initialized ||
               !same(s->stack[s->stack_count-1].type,scalar_type(TAG_INT)))return FLOW_CONST(INVALID);
            consumed=1;obligation.checks|=FLOW_CONST(CHECK_BYTE);
        }
    } else {
        if(reference!=FLOW_CONST(NO_REFERENCE))return FLOW_CONST(INVALID);
        if(method->input_mode==NL_SERVICE_INPUT_CONSUME) {
            if(!s->stack_count)return FLOW_CONST(INVALID);
            FLOW_TYPE(Value) value=s->stack[s->stack_count-1];
            if(!value.initialized || !value.owner || !same(value.type,file) || held(s,value.owner))return FLOW_CONST(INVALID);
            consumed=1;obligation.owned_inputs=1;
        } else {
            /* I consume exact copy inputs, including TCP's Endpoint, before
             * creating a new owner. Field domains remain runtime obligations. */
            size_t inputs=method->param_count-1;
            if(inputs>s->stack_count)return FLOW_CONST(INVALID);
            for(size_t i=0;i<inputs;i++) {
                FLOW_TYPE(Declaration) expected;
                FLOW_TYPE(Value) value=s->stack[s->stack_count-inputs+i];
                if(!member_type(s,FLOW_METHOD_OWNER(ordinal),method->params[i].type_id,&expected) || owned(expected) ||
                   !value.initialized || value.owner || !same(expected,value.type))return FLOW_CONST(INVALID);
                if(FLOW_IS_ENDPOINT(s->declarations->nominal,expected.catalog_ordinal) && expected.category==NOMINAL_CATEGORY_RECORD)
                    obligation.checks|=FLOW_ENDPOINT_CHECK;
            }
            consumed=(uint16_t)inputs;
        }
    }
    if((uint32_t)s->stack_count-consumed+1>FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    if(owned(result)) {
        FLOW_TYPE(Counts) counts;FLOW_FN(counts)(s,&counts);
        if(counts.owners==FLOW_CONST(OWNERS) || s->declarations->next_identity==UINT64_MAX)return FLOW_CONST(LIMIT);
    }
    uint16_t slot;FLOW_TYPE(Status) status=obligation_slot(s,obligation,&slot);if(status!=FLOW_CONST(OK))return status;
    uint64_t owner=owned(result)?s->declarations->next_identity++:0;
    while(consumed){pop(s);consumed--;}
    s->stack[s->stack_count++]=initial_value(result,true,owner);commit_obligation(s,obligation,slot);
    return FLOW_CONST(OK);
}
FLOW_TYPE(Status) FLOW_FN(call)(FLOW_TYPE(State) *s,uint32_t site,uint32_t function,
                                    const uint16_t *references,uint16_t count) {
    if(!s || function>=s->declarations->count)return FLOW_CONST(INVALID);
    if(!s->reachable || function==s->function)return FLOW_CONST(UNRESOLVED);
    const FLOW_TYPE(Function) *f=&s->declarations->functions[function];
    if(count!=f->parameters || (count && !references))return FLOW_CONST(INVALID);
    uint16_t values=0,owners=0,borrows=0;
    const FLOW_TYPE(Declaration) *params=s->declarations->locals+s->declarations->offsets[function];
    for(uint16_t i=0;i<count;i++){values+=!params[i].mode;owners+=owned(params[i]);borrows+=params[i].mode!=0;}
    if(values>s->stack_count)return FLOW_CONST(INVALID);
    uint16_t start=(uint16_t)(s->stack_count-values),value_index=start;
    for(uint16_t i=0;i<count;i++) {
        if(params[i].mode) {
            FLOW_TYPE(Declaration) actual;
            if(!live_reference(s,references[i],&actual) || !same(params[i],actual))return FLOW_CONST(INVALID);
            for(uint16_t j=0;j<i;j++)if(params[j].mode &&
                s->references[references[j]].owner==s->references[references[i]].owner &&
                (params[j].mode!=1 || params[i].mode!=1))return FLOW_CONST(INVALID);
        } else {
            if(references[i]!=FLOW_CONST(NO_REFERENCE))return FLOW_CONST(INVALID);
            FLOW_TYPE(Value) actual=s->stack[value_index++];
            if(!actual.initialized || !same(actual.type,params[i]) || owned(params[i])!=(actual.owner!=0) ||
               (actual.owner && held(s,actual.owner)))return FLOW_CONST(INVALID);
        }
    }
    if((uint32_t)start+f->result_count>FLOW_CONST(STACK))return FLOW_CONST(LIMIT);
    bool owner_result=f->result_count && owned(f->result);
    FLOW_TYPE(Counts) counts;FLOW_FN(counts)(s,&counts);
    if(counts.owners<owners)return FLOW_CONST(INVALID);
    if((uint32_t)counts.owners-owners+owner_result>FLOW_CONST(OWNERS) ||
       (owner_result && s->declarations->next_identity==UINT64_MAX))return FLOW_CONST(LIMIT);
    FLOW_TYPE(Obligation) obligation={0};obligation.kind=FLOW_CONST(CALL);obligation.site=site;
    obligation.target=function;obligation.parameters=count;obligation.owned_inputs=owners;obligation.borrowed_inputs=borrows;
    obligation.result_count=f->result_count;obligation.result=f->result;
    obligation.checks=FLOW_CONST(CHECK_CALLEE)|FLOW_CONST(CHECK_RESULT)|FLOW_CONST(CHECK_CLEANUP);
    uint16_t slot;FLOW_TYPE(Status) status=obligation_slot(s,obligation,&slot);if(status!=FLOW_CONST(OK))return status;
    uint64_t owner=owner_result?s->declarations->next_identity++:0;
    while(s->stack_count>start)pop(s);
    if(f->result_count)s->stack[s->stack_count++]=initial_value(f->result,true,owner);
    commit_obligation(s,obligation,slot);return FLOW_CONST(OK);
}

