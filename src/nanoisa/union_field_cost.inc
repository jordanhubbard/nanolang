#include "union_field_cost_private.h"
/* No allocation or semantic proof here: count exact owned capacities, with
 * redundant identities checked before scans. Embedded carrier arrays occur
 * only in sizeof(carrier). Original reservation counters are never consulted. */
static NvmMapCostStatus ufe_proof_retained_cost(const NvmUnionFieldProof *p,
                                              NvmMapCopyWork *w,uint64_t *out) {
    if(!ufc_outputs_disjoint(w,out))return NVM_MAP_COST_INVALID;
    if(!nvm_map_copy_charge(w,16))return NVM_MAP_COST_LIMIT;
    if(!p || !p->declarations || p->counts.origins>64 || p->counts.globals>256 ||
       !p->counts.functions || p->counts.functions>256)return NVM_MAP_COST_INVALID;
    uint64_t bytes=0,part;NvmMapCostStatus status;
#define UFC_TERM(ptr,count) do {status=nvm_map_copy_term(w,(ptr),(count),sizeof *(ptr),&bytes);if(status!=NVM_MAP_COST_OK)return status;}while(0)
    UFC_TERM(p,1);UFC_TERM(p->origins,p->counts.origins);UFC_TERM(p->globals,p->counts.globals);
    UFC_TERM(p->results,p->counts.functions);UFC_TERM(p->stacks,p->counts.functions);
#undef UFC_TERM
    status=nvm_union_declarations_retained_cost(p->declarations,w,&part);
    if(status!=NVM_MAP_COST_OK)return status;
    if(nvm_map_copy_sum(&bytes,part)!=NVM_MAP_COST_OK)return NVM_MAP_COST_LIMIT;
    *out=bytes;return NVM_MAP_COST_OK;
}
static NvmMapCostStatus ufe_plan_retained_cost(const NvmUnionFieldExecutionPlan *p,
                                             NvmMapCopyWork *w,uint64_t *out) {
    if(!ufc_outputs_disjoint(w,out))return NVM_MAP_COST_INVALID;
    if(!nvm_map_copy_charge(w,32))return NVM_MAP_COST_LIMIT;
    if(!p || !p->carrier || !p->proof || p->reserved || p->carrier->origins)return NVM_MAP_COST_INVALID;
    const NvmRecordArrayExecutionPlan *c=p->carrier;const NvmModule *m=&c->snapshot;
    if(!m->function_count || m->function_count>256 || m->string_count>NVM_MAX_STRINGS ||
       m->section_count>NVM_MAX_SECTIONS || m->header.section_count>NVM_MAX_SECTIONS ||
       m->code_size>16u*1024u*1024u || m->layout_size>NVM_RECORD_ARRAY_EXECUTION_BYTES ||
       m->ownership_size>NVM_RECORD_ARRAY_EXECUTION_BYTES ||
       m->metadata_count>NVM_RECORD_ARRAY_EXECUTION_BYTES/sizeof *m->metadata ||
       m->debug_count>NVM_RECORD_ARRAY_EXECUTION_BYTES/sizeof *m->debug_entries ||
       c->counts.functions!=m->function_count || c->counts.strings!=m->string_count ||
       c->counts.sections!=m->section_count || c->counts.metadata!=m->metadata_count ||
       c->counts.debug!=m->debug_count || c->counts.instructions>65536 ||
       p->site_count!=c->counts.instructions || p->proof->counts.functions!=m->function_count ||
       p->proof->counts.globals!=c->counts.globals || m->struct_count!=1 || m->union_count!=1 ||
       m->enum_count || c->counts.records!=1 || m->service_data || m->service_size ||
       m->passive_data || m->passive_size || m->import_count || m->module_ref_count ||
       m->callback_contract_count || m->call_descriptors || m->call_descriptor_count)
        return NVM_MAP_COST_INVALID;
    uint64_t bytes=0,part;NvmMapCostStatus status;
#define UFC_TERM(ptr,count) do {status=nvm_map_copy_term(w,(ptr),(count),sizeof *(ptr),&bytes);if(status!=NVM_MAP_COST_OK)return status;}while(0)
    UFC_TERM(p,1);UFC_TERM(c,1);UFC_TERM(c->instructions,c->counts.instructions);UFC_TERM(p->sites,p->site_count);
    UFC_TERM(m->functions,m->function_count);UFC_TERM(m->code,m->code_size);
    UFC_TERM(m->layout_data,m->layout_size);UFC_TERM(m->ownership_data,m->ownership_size);
    UFC_TERM(m->metadata,m->metadata_count);UFC_TERM(m->debug_entries,m->debug_count);
    UFC_TERM(m->string_lengths,m->string_count);UFC_TERM(m->strings,m->string_count);
    UFC_TERM(m->function_param_types,m->function_param_types?m->function_count:0);
    uint32_t instructions=0;
    for(uint32_t i=0;i<m->function_count;i++) {
        if(!nvm_map_copy_charge(w,16))return NVM_MAP_COST_LIMIT;
        const NvmFunctionEntry *f=&m->functions[i];
        const NvmRecordArrayExecutionFunction *copy=&c->functions[i];
        const uint8_t *row=m->function_param_types?m->function_param_types[i]:NULL;
        if(f->arity>256 || f->local_count>256 || copy->signature.arity!=f->arity ||
           copy->signature.local_count!=f->local_count || copy->parameter_tags_present!=!!row ||
           copy->instruction_start!=instructions ||
           copy->instruction_count>c->counts.instructions-instructions)return NVM_MAP_COST_INVALID;
        instructions+=copy->instruction_count;
        UFC_TERM(row,row?(f->arity?f->arity:1):0);
    }
    if(instructions!=c->counts.instructions)return NVM_MAP_COST_INVALID;
    for(uint32_t i=0;i<m->string_count;i++) {
        if(!nvm_map_copy_charge(w,8))return NVM_MAP_COST_LIMIT;
        uint64_t length=(uint64_t)m->string_lengths[i]+1;
        if(length>SIZE_MAX)return NVM_MAP_COST_LIMIT;
        UFC_TERM(m->strings[i],length);
    }
#undef UFC_TERM
    status=ufe_proof_retained_cost(p->proof,w,&part);
    if(status!=NVM_MAP_COST_OK)return status;
    if(nvm_map_copy_sum(&bytes,part)!=NVM_MAP_COST_OK)return NVM_MAP_COST_LIMIT;
    *out=bytes;return NVM_MAP_COST_OK;
}
