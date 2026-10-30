#include "union_field_snapshot_private.h"
#include "union_field_vm_image_private.h"
struct NvmUnionFieldImage { NvmUnionFieldExecutionPlan *plan;NvmUnionFieldImageCost cost; };
/* Called only after retained-cost validation established all source count and
 * pointer bounds. The per-function cost pass already prepaid these arity reads.
 * This computes work, it neither traverses recipe rows nor grants authority. */
static NvmMapCostStatus ufi_comparison_work(const NvmUnionFieldExecutionPlan *p,uint64_t *out) {
    uint64_t work=32+sizeof(NvmRecordArrayExecutionCounts)+sizeof(NvmUnionFieldProofCounts)+sizeof(NvmHeader);
    const NvmRecordArrayExecutionCounts *c=&p->carrier->counts;
#define UFI_ADD(n,width) do {uint64_t count=(n),size=(width);if(size && count>UINT64_MAX/size)return NVM_MAP_COST_LIMIT; if(nvm_map_copy_sum(&work,count*size)!=NVM_MAP_COST_OK)return NVM_MAP_COST_LIMIT;}while(0)
    UFI_ADD(c->functions,16+sizeof(NvmRecordArrayExecutionFunction));
    for(uint32_t i=0;i<c->functions;i++)UFI_ADD(p->carrier->snapshot.functions[i].arity,9);
    UFI_ADD(c->instructions,24+sizeof(NvmRecordArrayExecutionInstruction)+sizeof(NvmUnionFieldSite));
    UFI_ADD(p->proof->counts.origins,8+sizeof(NvmUnionFieldOrigin));
    UFI_ADD((uint64_t)p->proof->counts.globals+p->proof->counts.functions,8+sizeof(NvmRecordValueOrigins));
    UFI_ADD(c->functions,10);
    UFI_ADD(2,8+sizeof(uint32_t));UFI_ADD(2,8+sizeof(NvmDeclarationLayout));
    UFI_ADD(2,8+sizeof(NvmUnionVariantFact));UFI_ADD(2,8+sizeof(NvmV2LayoutField));
#undef UFI_ADD
    *out=work;return NVM_MAP_COST_OK;
}
static bool ufi_value_same(NvmRecordValueOrigins a,NvmRecordValueOrigins b) {
    return a.tags==b.tags && a.origins==b.origins && a.unknown==b.unknown;
}
static bool ufi_rows_same(const NvmUnionFieldExecutionInstruction *x,const NvmUnionFieldExecutionInstruction *y) {
    const NvmRecordArrayExecutionInstruction *a=&x->base,*b=&y->base;
#define UFI_EQ(field) do {if(a->field!=b->field)return false;}while(0)
    UFI_EQ(function);UFI_EQ(pc);UFI_EQ(next_pc);UFI_EQ(callee);UFI_EQ(pops);UFI_EQ(pushes);UFI_EQ(obligations);
    UFI_EQ(opcode);UFI_EQ(recipe);UFI_EQ(successor_count);UFI_EQ(operand_count);
    for(unsigned i=0;i<2;i++)UFI_EQ(successors[i]);
    for(unsigned i=0;i<MAX_OPERANDS;i++){UFI_EQ(operand_types[i]);UFI_EQ(operand_bits[i]);}
#undef UFI_EQ
    const NvmUnionFieldSite *s=&x->site,*t=&y->site;
#define UFI_EQ(field) do {if(s->field!=t->field)return false;}while(0)
    UFI_EQ(function);UFI_EQ(pc);UFI_EQ(nominal);UFI_EQ(layout);UFI_EQ(member);UFI_EQ(variant);UFI_EQ(operation);UFI_EQ(state);
#undef UFI_EQ
    for(unsigned i=0;i<3;i++)if(s->reserved[i] || t->reserved[i])return false;
    return ufi_value_same(s->receiver,t->receiver)&&ufi_value_same(s->input,t->input)&&ufi_value_same(s->result,t->result);
}
/* Comparison is fully prepaid before destination allocation. No hidden budget
 * mutation or source repair; no padding comparisons. All loop bounds originate
 * in the validated source, with destination counts checked first. */
static bool ufi_same(const NvmUnionFieldExecutionPlan *source,const NvmUnionFieldExecutionPlan *copy) {
    NvmRecordArrayExecutionCounts a,b;NvmUnionFieldProofCounts pa,pb;NvmHeader ha,hb;
    if(!nvm_union_field_execution_counts(source,&a)||!nvm_union_field_execution_counts(copy,&b)||
       !nvm_union_field_execution_proof_counts(source,&pa)||!nvm_union_field_execution_proof_counts(copy,&pb)||
       !nvm_union_field_execution_header(source,&ha)||!nvm_union_field_execution_header(copy,&hb))return false;
#define UFI_EQ(field) do {if(a.field!=b.field)return false;}while(0)
    UFI_EQ(functions);UFI_EQ(instructions);UFI_EQ(globals);UFI_EQ(initializer);UFI_EQ(entry);UFI_EQ(frame_limit);
    UFI_EQ(strings);UFI_EQ(sections);UFI_EQ(metadata);UFI_EQ(debug);UFI_EQ(records);
#undef UFI_EQ
    if(pa.origins!=pb.origins||pa.globals!=pb.globals||pa.functions!=pb.functions)return false;
#define UFI_EQ(field) do {if(ha.field!=hb.field)return false;}while(0)
    UFI_EQ(format_version);UFI_EQ(flags);UFI_EQ(entry_point);UFI_EQ(section_count);
    UFI_EQ(string_pool_offset);UFI_EQ(string_pool_length);UFI_EQ(checksum);
    for(unsigned i=0;i<4;i++)UFI_EQ(magic[i]);
#undef UFI_EQ
    for(uint32_t i=0;i<a.functions;i++) {
        NvmRecordArrayExecutionFunction x,y;
        if(!nvm_union_field_execution_function(source,i,&x)||!nvm_union_field_execution_function(copy,i,&y))return false;
#define UFI_EQ(field) do {if(x.field!=y.field)return false;}while(0)
        UFI_EQ(signature.name_idx);UFI_EQ(signature.arity);UFI_EQ(signature.code_offset);UFI_EQ(signature.code_length);
        UFI_EQ(signature.local_count);UFI_EQ(signature.upvalue_count);UFI_EQ(signature.result_tag);UFI_EQ(signature.result_count);
        UFI_EQ(instruction_start);UFI_EQ(instruction_count);UFI_EQ(maximum_stack);UFI_EQ(parameter_tags_present);
#undef UFI_EQ
        if(x.parameter_tags_present)for(uint16_t j=0;j<x.signature.arity;j++) {
            uint8_t u,v;
            if(!nvm_union_field_execution_parameter(source,i,j,&u)||!nvm_union_field_execution_parameter(copy,i,j,&v)||u!=v)return false;
        }
        uint16_t u,v;
        if(!nvm_union_field_proof_stack(source->proof,i,&u)||!nvm_union_field_proof_stack(copy->proof,i,&v)||u!=v||
           source->carrier->maximum_stacks[i]!=u)return false;
    }
    for(uint32_t i=0;i<a.instructions;i++) {
        NvmUnionFieldExecutionInstruction x,y;
        if(!nvm_union_field_execution_instruction(source,i,&x)||!nvm_union_field_execution_instruction(copy,i,&y)||!ufi_rows_same(&x,&y))return false;
    }
    for(uint32_t i=0;i<pa.origins;i++) {
        NvmUnionFieldOrigin x,y;
        if(!nvm_union_field_execution_origin(source,i,&x)||!nvm_union_field_execution_origin(copy,i,&y))return false;
#define UFI_EQ(field) do {if(x.field!=y.field)return false;}while(0)
        UFI_EQ(function);UFI_EQ(pc);UFI_EQ(nominal_ordinal);UFI_EQ(layout_index);UFI_EQ(kind);UFI_EQ(variant);UFI_EQ(member_count);
#undef UFI_EQ
        if(!ufi_value_same(x.member,y.member))return false;
    }
    for(uint32_t i=0;i<pa.globals;i++) {
        NvmRecordValueOrigins x,y;
        if(!nvm_union_field_execution_global(source,i,&x)||!nvm_union_field_execution_global(copy,i,&y)||!ufi_value_same(x,y))return false;
    }
    for(uint32_t i=0;i<pa.functions;i++) {
        NvmRecordValueOrigins x,y;
        if(!nvm_union_field_execution_result(source,i,&x)||!nvm_union_field_execution_result(copy,i,&y)||!ufi_value_same(x,y))return false;
    }
    for(uint32_t i=0;i<2;i++) {
        uint32_t u,v;uint8_t tag=i?TAG_UNION:TAG_STRUCT;
        if(!nvm_union_field_execution_nominal(source,tag,0,&u)||!nvm_union_field_execution_nominal(copy,tag,0,&v)||u!=v)return false;
        NvmDeclarationLayout x,y;
        if(!nvm_union_field_execution_layout(source,i,&x)||!nvm_union_field_execution_layout(copy,i,&y))return false;
#define UFI_EQ(field) do {if(x.field!=y.field)return false;}while(0)
        UFI_EQ(kind);UFI_EQ(flags);UFI_EQ(fields);UFI_EQ(name);
#undef UFI_EQ
        NvmUnionVariantFact va,vb;
        if(!nvm_union_field_execution_variant(source,0,(uint16_t)i,&va)||!nvm_union_field_execution_variant(copy,0,(uint16_t)i,&vb)||
           va.layout!=vb.layout||va.name_idx!=vb.name_idx||va.field_offset!=vb.field_offset||va.field_count!=vb.field_count)return false;
        NvmV2LayoutField fa,fb;
        if(!nvm_union_field_execution_field(source,i,0,&fa)||!nvm_union_field_execution_field(copy,i,0,&fb)||
           fa.type_tag!=fb.type_tag||fa.nested_idx!=fb.nested_idx||fa.name_idx!=fb.name_idx)return false;
    }
    /* Fixed active descriptor is embedded, not another owning allocation. */
    return source->carrier->descriptors[0].ordinal==copy->carrier->descriptors[0].ordinal&&
           source->carrier->descriptors[0].layout==copy->carrier->descriptors[0].layout&&
           source->carrier->descriptors[0].fields==copy->carrier->descriptors[0].fields;
}
void nvm_union_field_image_free(NvmUnionFieldImage *p) {
    if(!p)return;
    nvm_union_field_execution_free(p->plan);free(p);
}
NvmArrayEligibilityResult nvm_union_field_image_copy_reserved(const NvmUnionFieldExecutionPlan *source,
        uint64_t extra_bytes,uint64_t extra_work,NvmUnionFieldImage **out) {
    NvmArrayEligibilityResult result={NVM_ARRAY_INVALID,0,0,"I require an immutable union plan and image output."};
    if(!source || !out)return result;
    NvmMapCopyWork cost={0,NVM_RECORD_ARRAY_EXECUTION_STEPS,false};uint64_t retained,comparison;
    NvmMapCostStatus status=ufe_plan_retained_cost(source,&cost,&retained);
    if(status==NVM_MAP_COST_OK)status=ufi_comparison_work(source,&comparison);
    if(status!=NVM_MAP_COST_OK){result.status=status==NVM_MAP_COST_LIMIT?NVM_ARRAY_LIMIT:NVM_ARRAY_INVALID;return result;}
    uint64_t bytes=retained,work=cost.used;
    uint64_t initial_bytes=UINT64_C(64)*1024*1024+65536u+sizeof(NvmUnionFieldExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    uint64_t initial_work=UINT64_C(16777216)+65536u+sizeof(NvmUnionFieldExecutionPlan)+sizeof(NvmRecordArrayExecutionPlan);
    if(nvm_map_copy_sum(&bytes,extra_bytes)!=NVM_MAP_COST_OK||
       nvm_map_copy_sum(&work,extra_work)!=NVM_MAP_COST_OK||
       nvm_map_copy_sum(&bytes,sizeof(NvmUnionFieldImage))!=NVM_MAP_COST_OK||
       nvm_map_copy_sum(&work,sizeof(NvmUnionFieldImage))!=NVM_MAP_COST_OK||
       nvm_map_copy_sum(&work,comparison)!=NVM_MAP_COST_OK||
       initial_bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES||bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-initial_bytes||
       initial_work>NVM_RECORD_ARRAY_EXECUTION_STEPS||work>NVM_RECORD_ARRAY_EXECUTION_STEPS-initial_work) {
        result.status=NVM_ARRAY_LIMIT;return result;
    }
    NvmUnionFieldImage *image=calloc(1,sizeof *image);
    if(!image){result.status=NVM_ARRAY_MEMORY;return result;}
    NvmUnionFieldExecutionPlan *staged=NULL;
    result=ufe_prepare_reserved(&source->carrier->snapshot,&staged,bytes,work);
    if(result.status!=NVM_ARRAY_ELIGIBLE){free(image);return result;}
    if(!ufi_same(source,staged)) {
        nvm_union_field_execution_free(staged);free(image);result.status=NVM_ARRAY_INVALID;
        snprintf(result.message,sizeof result.message,"I require source union facts to match independent reconstruction.");return result;
    }
    image->cost=(NvmUnionFieldImageCost){retained,staged->carrier->counts.peak_bytes_reserved,staged->carrier->counts.work_reserved};
    image->plan=staged;staged=NULL;*out=image;
    snprintf(result.message,sizeof result.message,"I independently copied descriptive union facts without execution authority.");return result;
}
/* Old image construction keeps its exact allocation and charge path. */
NvmArrayEligibilityResult nvm_union_field_image_copy(const NvmUnionFieldExecutionPlan *source,NvmUnionFieldImage **out) {
    return nvm_union_field_image_copy_reserved(source,0,0,out);
}
const NvmModule *nvm_union_field_image_module_internal(const NvmUnionFieldImage *image) {
    return image && image->plan && image->plan->carrier?&image->plan->carrier->snapshot:NULL;
}
bool nvm_union_field_image_cost(const NvmUnionFieldImage *p,NvmUnionFieldImageCost *out) {
    if(!p || !out)return false;
    *out=p->cost;return true;
}
#define UFI_QUERY(name,type) \
bool nvm_union_field_image_##name(const NvmUnionFieldImage *p,type *out) { \
    return p && nvm_union_field_execution_##name(p->plan,out); }
UFI_QUERY(counts,NvmRecordArrayExecutionCounts)
UFI_QUERY(header,NvmHeader)
UFI_QUERY(proof_counts,NvmUnionFieldProofCounts)
#undef UFI_QUERY
#define UFI_QUERY(name,type) \
bool nvm_union_field_image_##name(const NvmUnionFieldImage *p,uint32_t i,type *out) { \
    return p && nvm_union_field_execution_##name(p->plan,i,out); }
UFI_QUERY(instruction,NvmUnionFieldExecutionInstruction)
UFI_QUERY(function,NvmRecordArrayExecutionFunction)
UFI_QUERY(origin,NvmUnionFieldOrigin)
UFI_QUERY(global,NvmRecordValueOrigins)
UFI_QUERY(result,NvmRecordValueOrigins)
UFI_QUERY(layout,NvmDeclarationLayout)
#undef UFI_QUERY
bool nvm_union_field_image_parameter(const NvmUnionFieldImage *p,uint32_t i,uint16_t j,uint8_t *out) {
    return p && nvm_union_field_execution_parameter(p->plan,i,j,out);
}
bool nvm_union_field_image_size(const NvmUnionFieldImage *p,NvmRecordArraySnapshotKind k,uint32_t i,uint32_t *out) {
    return p && nvm_union_field_execution_size(p->plan,k,i,out);
}
bool nvm_union_field_image_bytes(const NvmUnionFieldImage *p,NvmRecordArraySnapshotKind k,uint32_t i,uint32_t offset,uint32_t count,void *out) {
    return p && nvm_union_field_execution_bytes(p->plan,k,i,offset,count,out);
}
bool nvm_union_field_image_field(const NvmUnionFieldImage *p,uint32_t i,uint16_t j,NvmV2LayoutField *out) {
    return p && nvm_union_field_execution_field(p->plan,i,j,out);
}
bool nvm_union_field_image_variant(const NvmUnionFieldImage *p,uint32_t i,uint16_t j,NvmUnionVariantFact *out) {
    return p && nvm_union_field_execution_variant(p->plan,i,j,out);
}
bool nvm_union_field_image_nominal(const NvmUnionFieldImage *p,uint8_t tag,uint32_t i,uint32_t *out) {
    return p && nvm_union_field_execution_nominal(p->plan,tag,i,out);
}
