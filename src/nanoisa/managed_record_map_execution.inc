#include "managed_record_map_execution.h"
_Static_assert(NVM_RM_INVALID_BASE_RECIPE>NVM_RA_ROOT_TAIL,"map recipe cannot alias an old root recipe");
struct NvmRecordMapExecutionPlan {
    NvmRecordArrayExecutionPlan *carrier;
    NvmRecordMapFrozenContext *facts;
    NvmRecordMapExecutionInstruction *instructions;
};
void nvm_record_map_execution_free(NvmRecordMapExecutionPlan *p) {
    if(!p)return;
    /* carrier->origins is always NULL; only facts owns its base report. */
    nvm_record_array_execution_free(p->carrier);
    nvm_record_map_frozen_free(p->facts);free(p->instructions);free(p);
}
static int rme_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    if(!nvm_record_map_opcode(op))return rae_old_recipe(op,r);
    r->recipe=NVM_RM_INVALID_BASE_RECIPE;
    r->obligations=NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS|NVM_RA_STAGE_OPERANDS;
    if(op==OP_HM_NEW || op==OP_HM_SET)r->obligations|=NVM_RA_SAFEPOINT_BEFORE;
    if(op==OP_HM_SET || op==OP_HM_GET)r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;
    return 1;
}
static void rme_tags(NvmRecordMapExecutionInstruction *r) {
    if(r->base.recipe!=NVM_RM_INVALID_BASE_RECIPE)return;
    r->key_tags=(uint16_t)(1u<<TAG_STRING);r->value_tags=(uint16_t)(1u<<TAG_INT);
    switch(r->base.opcode) {
    case OP_HM_NEW:r->map_operation=NVM_RM_NEW;r->result_tags=(uint16_t)(1u<<TAG_HASHMAP);break;
    case OP_HM_SET:r->map_operation=NVM_RM_SET;r->result_tags=(uint16_t)(1u<<TAG_HASHMAP);break;
    case OP_HM_GET:r->map_operation=NVM_RM_GET;r->result_tags=(uint16_t)((1u<<TAG_INT)|(1u<<TAG_VOID));break;
    case OP_HM_HAS:r->map_operation=NVM_RM_HAS;r->result_tags=(uint16_t)(1u<<TAG_BOOL);break;
    case OP_HM_LEN:r->map_operation=NVM_RM_LEN;r->result_tags=(uint16_t)(1u<<TAG_INT);break;
    }
}
/* Conservative positive pre-reservation is internal, never a refund or authority token. */
static NvmArrayEligibilityResult rme_prepare_reserved(const NvmModule *m,NvmRecordMapExecutionPlan **out,
                                                     uint64_t extra_bytes,uint64_t extra_work) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require an original module and copied map plan output."};
    if(!m || !out)return result;
    uint64_t bytes=UINT64_C(64)*1024*1024+65536u+sizeof(NvmRecordMapExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    uint64_t work=UINT64_C(16777216)+65536u+sizeof(NvmRecordMapExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    if(bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || work>NVM_RECORD_ARRAY_EXECUTION_STEPS ||
       extra_bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-bytes || extra_work>NVM_RECORD_ARRAY_EXECUTION_STEPS-work) {
        result.status=NVM_ARRAY_LIMIT;return result;
    }
    bytes+=extra_bytes;work+=extra_work;
    NvmRecordMapExecutionPlan *p=calloc(1,sizeof *p);
    if(!p){result.status=NVM_ARRAY_MEMORY;return result;}
    p->carrier=calloc(1,sizeof *p->carrier);
    if(!p->carrier){result.status=NVM_ARRAY_MEMORY;nvm_record_map_execution_free(p);return result;}
    NvmRecordArrayExecutionPlan *c=p->carrier;
    c->bytes=bytes;c->work=work;c->failure=NVM_ARRAY_INVALID;c->counts.initializer=UINT32_MAX;
    if(!rae_snapshot(c,m))goto fail;
    for(uint32_t fi=0;fi<c->snapshot.function_count;fi++) {
        uint32_t name=c->snapshot.functions[fi].name_idx;
        if(name>=c->snapshot.string_count)goto fail;
        uint32_t length=c->snapshot.string_lengths[name];
        if(!rae_charge(c,0,(uint64_t)length+1))goto fail;
        if(memchr(c->snapshot.strings[name],0,length)){c->failure=NVM_ARRAY_UNRESOLVED;goto fail;}
    }
    /* One analysis publishes facts and stacks only after complete success. */
    result=analyze(&c->snapshot,NULL,NULL,NULL,NULL,c->maximum_stacks,0,&p->facts);
    if(result.status!=NVM_ARRAY_ELIGIBLE){nvm_record_map_execution_free(p);return result;}
    if(!rae_charge(c,0,256))goto fail;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction row={0};
        int supported=nvm_record_array_opcode_supported((uint8_t)op)||nvm_record_map_opcode((uint8_t)op);
        if(!!supported!=!!rme_recipe((uint8_t)op,&row)){c->failure=NVM_ARRAY_UNRESOLVED;goto fail;}
    }
    NvmDeclarationCounts declarations;
    if(!nvm_record_array_declaration_counts(p->facts->base,&declarations) || declarations.layouts>256)goto fail;
    if(!rae_charge(c,0,(uint64_t)declarations.layouts*4u))goto fail;
    for(uint32_t i=0;i<declarations.layouts;i++) {
        NvmDeclarationLayout row;
        if(!nvm_record_array_declaration_layout(p->facts->base,i,&row))goto fail;
        if(row.kind==NVM_V2_LAYOUT_STRUCT) {
            uint32_t ordinal=c->counts.records++;
            c->descriptors[ordinal]=(NvmRecordArrayExecutionDescriptor){ordinal,i,row.fields};
        }
    }
    if(c->counts.records!=c->snapshot.struct_count || !rae_instructions_using(c,rme_recipe))goto fail;
    p->instructions=rae_zero(c,c->counts.instructions,sizeof *p->instructions);
    if(c->counts.instructions && !p->instructions)goto fail;
    if(!rae_charge(c,0,(uint64_t)c->counts.instructions*4u))goto fail;
    for(uint32_t i=0;i<c->counts.instructions;i++) {
        p->instructions[i].base=c->instructions[i];rme_tags(&p->instructions[i]);
    }
    if(!rae_same(c,m))goto fail;
    c->counts.strings=c->snapshot.string_count;c->counts.sections=c->snapshot.section_count;
    c->counts.metadata=c->snapshot.metadata_count;c->counts.debug=c->snapshot.debug_count;
    c->counts.functions=c->snapshot.function_count;c->counts.entry=c->snapshot.header.entry_point;
    c->counts.frame_limit=NVM_RECORD_ARRAY_EXECUTION_FRAMES;
    c->counts.peak_bytes_reserved=c->bytes;c->counts.work_reserved=c->work;
    *out=p;result.status=NVM_ARRAY_ELIGIBLE;
    snprintf(result.message,sizeof result.message,"I copied exact map execution recipes without executable admission.");return result;
fail:
    result.status=c->failure;
    snprintf(result.message,sizeof result.message,"I could not copy bounded exact map execution recipes.");
    nvm_record_map_execution_free(p);return result;
}
NvmArrayEligibilityResult nvm_prepare_record_map_execution(const NvmModule *m,NvmRecordMapExecutionPlan **out) {
    return rme_prepare_reserved(m,out,0,0);
}
bool nvm_record_map_execution_counts(const NvmRecordMapExecutionPlan *p,NvmRecordMapExecutionCounts *out) {
    if(!p || !out)return false;
    NvmRecordMapExecutionCounts value;
    if(!nvm_record_array_execution_counts(p->carrier,&value.carrier) || !nvm_record_map_frozen_counts(p->facts,&value.maps))return false;
    *out=value;return true;
}
bool nvm_record_map_execution_instruction(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordMapExecutionInstruction *out) {
    if(!p || !out || i>=p->carrier->counts.instructions)return false;
    *out=p->instructions[i];return true;
}
bool nvm_record_map_execution_map_field(const NvmRecordMapExecutionPlan *p,uint32_t layout,uint16_t field,uint8_t *key,uint8_t *value) {
    return p && nvm_record_map_frozen_field(p->facts,layout,field,key,value);
}
bool nvm_record_map_execution_map_origin(const NvmRecordMapExecutionPlan *p,uint32_t origin,uint8_t *key,uint8_t *value) {
    return p && nvm_record_map_frozen_origin(p->facts,origin,key,value);
}
bool nvm_record_map_execution_function(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionFunction *out) {
    return p && nvm_record_array_execution_function(p->carrier,i,out);
}
bool nvm_record_map_execution_parameter(const NvmRecordMapExecutionPlan *p,uint32_t fi,uint16_t i,uint8_t *out) {
    return p && nvm_record_array_execution_parameter(p->carrier,fi,i,out);
}
bool nvm_record_map_execution_bytes(const NvmRecordMapExecutionPlan *p,NvmRecordArraySnapshotKind kind,uint32_t index,uint32_t offset,uint32_t count,void *out) {
    return p && nvm_record_array_execution_bytes(p->carrier,kind,index,offset,count,out);
}
bool nvm_record_map_execution_size(const NvmRecordMapExecutionPlan *p,NvmRecordArraySnapshotKind kind,uint32_t index,uint32_t *out) {
    return p && nvm_record_array_execution_size(p->carrier,kind,index,out);
}
bool nvm_record_map_execution_descriptor(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionDescriptor *out) {
    return p && nvm_record_array_execution_descriptor(p->carrier,i,out);
}
bool nvm_record_map_execution_header(const NvmRecordMapExecutionPlan *p,NvmHeader *out) {
    return p && nvm_record_array_execution_header(p->carrier,out);
}
bool nvm_record_map_execution_section(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmSectionEntry *out) {
    return p && nvm_record_array_execution_section(p->carrier,i,out);
}
bool nvm_record_map_execution_metadata(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmMetadataEntry *out) {
    return p && nvm_record_array_execution_metadata(p->carrier,i,out);
}
bool nvm_record_map_execution_debug(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmDebugEntry *out) {
    return p && nvm_record_array_execution_debug(p->carrier,i,out);
}
bool nvm_record_map_execution_origin_counts(const NvmRecordMapExecutionPlan *p,NvmRecordArrayOriginCounts *out) {
    return p && nvm_record_array_origins_counts(p->facts->base,out);
}
bool nvm_record_map_execution_origin(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordHeapOrigin *out) {
    return p && nvm_record_array_origin(p->facts->base,i,out);
}
bool nvm_record_map_execution_field_value(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordValueOrigins *out) {
    return p && nvm_record_array_field_value(p->facts->base,i,out);
}
bool nvm_record_map_execution_required_elements(const NvmRecordMapExecutionPlan *p,uint32_t i,uint16_t *out) {
    return p && nvm_record_array_required_elements(p->facts->base,i,out);
}
bool nvm_record_map_execution_declaration_counts(const NvmRecordMapExecutionPlan *p,NvmDeclarationCounts *out) {
    return p && nvm_record_array_declaration_counts(p->facts->base,out);
}
bool nvm_record_map_execution_layout(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmDeclarationLayout *out) {
    return p && nvm_record_array_declaration_layout(p->facts->base,i,out);
}
bool nvm_record_map_execution_field(const NvmRecordMapExecutionPlan *p,uint32_t i,uint16_t field,NvmV2LayoutField *out) {
    return p && nvm_record_array_declaration_field(p->facts->base,i,field,out);
}
bool nvm_record_map_execution_type(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmOrdinaryArrayType *out) {
    return p && nvm_record_array_declaration_type(p->facts->base,i,out);
}
bool nvm_record_map_execution_binding(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmOrdinaryArrayBinding *out) {
    return p && nvm_record_array_declaration_binding(p->facts->base,i,out);
}
bool nvm_record_map_execution_variant(const NvmRecordMapExecutionPlan *p,uint32_t i,uint16_t variant,NvmUnionVariantFact *out) {
    return p && nvm_record_array_declaration_variant(p->facts->base,i,variant,out);
}
