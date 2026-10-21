#include "record_map_cost_private.h"
#include "record_map_snapshot_private.h"
/* I count requested retained payloads only. Temporaries have already died. */
static NvmMapCostStatus rme_frozen_retained_cost(const NvmRecordMapFrozenContext *p,
                                              NvmMapCopyWork *w,uint64_t *out) {
    if(!nvm_map_copy_charge(w,16))return NVM_MAP_COST_LIMIT;
    if(!p || !out || !p->base || !p->base->origins || !p->base->declarations || !p->declarations ||
       p->counts.origins>64 || p->counts.map_origins>p->counts.origins || p->counts.fields>FIELD_CELLS ||
       p->base->origins->origin_count!=p->counts.origins ||
       p->base->origins->field_value_count!=p->counts.fields)return NVM_MAP_COST_INVALID;
    uint64_t bytes=0,part;NvmMapCostStatus status;
#define MRC_TERM(ptr,count) do {status=nvm_map_copy_term(w,(ptr),(count),sizeof *(ptr),&bytes);if(status!=NVM_MAP_COST_OK)return status;}while(0)
    MRC_TERM(p,1);MRC_TERM(p->rows,p->counts.map_origins);MRC_TERM(p->base,1);
    MRC_TERM(p->base->origins,1);MRC_TERM(p->base->origins->fields,p->counts.fields);
#undef MRC_TERM
    status=nvm_ownership_declarations_retained_cost(p->base->declarations,w,&part);
    if(status!=NVM_MAP_COST_OK)return status;
    if(nvm_map_copy_sum(&bytes,part)!=NVM_MAP_COST_OK)return NVM_MAP_COST_LIMIT;
    status=nvm_map_declarations_retained_cost(p->declarations,w,&part);
    if(status!=NVM_MAP_COST_OK)return status;
    if(nvm_map_copy_sum(&bytes,part)!=NVM_MAP_COST_OK)return NVM_MAP_COST_LIMIT;
    *out=bytes;return NVM_MAP_COST_OK;
}
static NvmMapCostStatus rme_plan_retained_cost(const NvmRecordMapExecutionPlan *p,
                                            NvmMapCopyWork *w,uint64_t *out) {
    if(!nvm_map_copy_charge(w,16))return NVM_MAP_COST_LIMIT;
    if(!p || !out || !p->carrier || !p->facts || p->carrier->origins)return NVM_MAP_COST_INVALID;
    const NvmRecordArrayExecutionPlan *c=p->carrier;const NvmModule *m=&c->snapshot;
    if(!m->function_count || m->function_count>256 || m->string_count>NVM_MAX_STRINGS ||
       c->counts.functions!=m->function_count || c->counts.strings!=m->string_count ||
       c->counts.metadata!=m->metadata_count || c->counts.debug!=m->debug_count ||
       c->counts.sections!=m->section_count || m->section_count>NVM_MAX_SECTIONS ||
       c->counts.instructions>65536 || m->code_size>16u*1024u*1024u)return NVM_MAP_COST_INVALID;
    uint64_t bytes=0,part;NvmMapCostStatus status;
#define MRC_TERM(ptr,count) do {status=nvm_map_copy_term(w,(ptr),(count),sizeof *(ptr),&bytes);if(status!=NVM_MAP_COST_OK)return status;}while(0)
    MRC_TERM(p,1);MRC_TERM(c,1);MRC_TERM(c->instructions,c->counts.instructions);MRC_TERM(p->instructions,c->counts.instructions);
    MRC_TERM(m->functions,m->function_count);MRC_TERM(m->code,m->code_size);
    MRC_TERM(m->layout_data,m->layout_size);MRC_TERM(m->ownership_data,m->ownership_size);
    MRC_TERM(m->metadata,m->metadata_count);MRC_TERM(m->debug_entries,m->debug_count);
    MRC_TERM(m->string_lengths,m->string_count);MRC_TERM(m->strings,m->string_count);
    MRC_TERM(m->function_param_types,m->function_param_types?m->function_count:0);
    uint32_t instructions=0;
    for(uint32_t i=0;i<m->function_count;i++) {
        if(!nvm_map_copy_charge(w,8))return NVM_MAP_COST_LIMIT;
        const NvmFunctionEntry *f=&m->functions[i];
        const NvmRecordArrayExecutionFunction *copy=&c->functions[i];
        if(f->arity>256 || f->local_count>256 || copy->signature.arity!=f->arity ||
           copy->signature.local_count!=f->local_count || copy->instruction_start!=instructions ||
           copy->instruction_count>c->counts.instructions-instructions)return NVM_MAP_COST_INVALID;
        instructions+=copy->instruction_count;
        const uint8_t *row=m->function_param_types?m->function_param_types[i]:NULL;
        MRC_TERM(row,row?(f->arity?f->arity:1):0);
    }
    if(instructions!=c->counts.instructions)return NVM_MAP_COST_INVALID;
    for(uint32_t i=0;i<m->string_count;i++) {
        if(!nvm_map_copy_charge(w,8))return NVM_MAP_COST_LIMIT;
        uint64_t length=(uint64_t)m->string_lengths[i]+1;
        if(length>SIZE_MAX)return NVM_MAP_COST_LIMIT;
        MRC_TERM(m->strings[i],length);
    }
#undef MRC_TERM
    status=rme_frozen_retained_cost(p->facts,w,&part);
    if(status!=NVM_MAP_COST_OK)return status;
    if(nvm_map_copy_sum(&bytes,part)!=NVM_MAP_COST_OK)return NVM_MAP_COST_LIMIT;
    *out=bytes;return NVM_MAP_COST_OK;
}
struct NvmRecordMapImage { NvmRecordMapExecutionPlan *plan;NvmRecordMapImageCost cost; };
void nvm_record_map_image_free(NvmRecordMapImage *p) {
    if(!p)return;
    nvm_record_map_execution_free(p->plan);free(p);
}
static NvmArrayEligibilityResult rme_image_copy_reserved(const NvmRecordMapExecutionPlan *source,
        NvmRecordMapImage **out,uint64_t extra_bytes,uint64_t extra_work) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require a copied map plan and image output."};
    if(!source || !out)return result;
    NvmMapCopyWork w={0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};uint64_t retained;
    NvmMapCostStatus status=rme_plan_retained_cost(source,&w,&retained);
    if(status!=NVM_MAP_COST_OK){result.status=status==NVM_MAP_COST_LIMIT?NVM_ARRAY_LIMIT:NVM_ARRAY_INVALID;return result;}
    uint64_t bytes=retained,work=w.used;
    /* This conservative preflight includes the destination constructor's fixed
     * initial reservation, so impossible extras refuse before image allocation. */
    uint64_t initial_bytes=UINT64_C(64)*1024*1024+65536u+sizeof(NvmRecordMapExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    uint64_t initial_work=UINT64_C(16777216)+65536u+sizeof(NvmRecordMapExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    if(nvm_map_copy_sum(&bytes,sizeof(NvmRecordMapImage))!=NVM_MAP_COST_OK ||
       nvm_map_copy_sum(&work,sizeof(NvmRecordMapImage))!=NVM_MAP_COST_OK ||
       nvm_map_copy_sum(&bytes,extra_bytes)!=NVM_MAP_COST_OK || nvm_map_copy_sum(&work,extra_work)!=NVM_MAP_COST_OK ||
       initial_bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-initial_bytes ||
       initial_work>NVM_RECORD_ARRAY_EXECUTION_STEPS || work>NVM_RECORD_ARRAY_EXECUTION_STEPS-initial_work) {
        result.status=NVM_ARRAY_LIMIT;return result;
    }
    NvmRecordMapImage *image=calloc(1,sizeof *image);
    if(!image){result.status=NVM_ARRAY_MEMORY;return result;}
    /* Fresh constructor revalidates the owned bytes; source facts are never
     * borrowed as acceptance authority. The two reports have independent owners. */
    result=rme_prepare_reserved(&source->carrier->snapshot,&image->plan,bytes,work);
    if(result.status!=NVM_ARRAY_ELIGIBLE){nvm_record_map_image_free(image);return result;}
    image->cost=(NvmRecordMapImageCost){retained,image->plan->carrier->counts.peak_bytes_reserved,
                                              image->plan->carrier->counts.work_reserved};
    *out=image;snprintf(result.message,sizeof result.message,"I independently copied map image facts without executable admission.");return result;
}
NvmArrayEligibilityResult nvm_record_map_image_copy(const NvmRecordMapExecutionPlan *p,NvmRecordMapImage **out) {
    return rme_image_copy_reserved(p,out,0,0);
}
bool nvm_record_map_image_cost(const NvmRecordMapImage *p,NvmRecordMapImageCost *out) {
    if(!p || !out)return false;
    *out=p->cost;return true;
}
const NvmModule *nvm_record_map_image_borrow(const NvmRecordMapImage *p) {
    return p?&p->plan->carrier->snapshot:NULL;
}
bool nvm_record_map_image_counts(const NvmRecordMapImage *p,NvmRecordMapExecutionCounts *out) {
    return p && nvm_record_map_execution_counts(p->plan,out);
}
bool nvm_record_map_image_instruction(const NvmRecordMapImage *p,uint32_t i,NvmRecordMapExecutionInstruction *out) {
    return p && nvm_record_map_execution_instruction(p->plan,i,out);
}
bool nvm_record_map_image_field(const NvmRecordMapImage *p,uint32_t i,uint16_t field,uint8_t *key,uint8_t *value) {
    return p && nvm_record_map_execution_map_field(p->plan,i,field,key,value);
}
bool nvm_record_map_image_origin(const NvmRecordMapImage *p,uint32_t i,uint8_t *key,uint8_t *value) {
    return p && nvm_record_map_execution_map_origin(p->plan,i,key,value);
}

NvmArrayEligibilityResult nvm_record_map_image_copy_reserved(const NvmRecordMapExecutionPlan *p,
        uint64_t bytes,uint64_t work,NvmRecordMapImage **out) {
    return rme_image_copy_reserved(p,out,bytes,work);
}
const NvmRecordMapExecutionPlan *nvm_record_map_image_plan(const NvmRecordMapImage *p) {
    return p?p->plan:NULL;
}
