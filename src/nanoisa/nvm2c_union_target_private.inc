/* Owning-TU, non-emitting target preparation. No execution authority escapes.
 * All correspondence helpers below run only after their bounded work charge.
 * Old constructor helpers remain literal; this independent target is compared
 * across all256 opcode values and copied immutable rows. */
#include "union_field_vm_image_private.h"
#include "record_array_structure_private.h"
#define UFN_BIT(tag) ((uint16_t)(1u<<(tag)))
#define UFN_WORK_LIMIT UINT64_C(33554432)
#define UFN_SCRATCH_LIMIT UINT64_C(65536)
typedef struct {
    const NvmUnionFieldImage *image;
    const NvmModule *module;
    uint32_t record_ordinal,record_layout,union_ordinal,union_layout;
    uint8_t payload_variant,empty_variant,reserved[6];
} Nvm2cUnionTargetContext;
typedef struct {
    NvmArrayEligibilityResult eligibility;
    uint64_t work,scratch_reserved;
} Nvm2cUnionTargetResult;
static int ufn_charge(Nvm2cUnionTargetResult *r,uint64_t allowance,uint64_t amount) {
    if(r->work>allowance || amount>allowance-r->work) {
        r->eligibility.status=NVM_ARRAY_LIMIT;return 0;
    }
    r->work+=amount;return 1;
}
static int ufn_base_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    r->obligations=NVM_RA_FIRST_ERROR_CLEANUP|NVM_RA_CHECK_TAGS;
    switch(op) {
    case OP_PUSH_I64: case OP_PUSH_U8: case OP_PUSH_F64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
        r->recipe=NVM_RA_ROOT_CONSTANT;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;break;
    case OP_PUSH_STR:
        r->recipe=NVM_RA_ROOT_CONSTANT;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_NOP:r->recipe=NVM_RA_ROOT_SCALAR;r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;break;
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG: case OP_F64_TO_BITS: case OP_F64_FROM_BITS:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
    case OP_CAST_INT: case OP_CAST_FLOAT: case OP_CAST_BOOL: case OP_CAST_U8: case OP_TYPE_CHECK:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT: case OP_AND: case OP_OR: case OP_NOT:
    case OP_STR_LEN: case OP_STR_CHAR_AT: case OP_ARR_LEN:
    case OP_STR_EQ: case OP_STR_CONTAINS: case OP_STR_STARTS_WITH: case OP_STR_ENDS_WITH:
        r->recipe=NVM_RA_ROOT_SCALAR;break;
    case OP_CAST_STRING: case OP_STR_CONCAT: case OP_STR_SUBSTR: case OP_STR_TRIM:
    case OP_STR_TO_LOWER: case OP_STR_TO_UPPER: case OP_STR_REPLACE:
    case OP_STR_FROM_INT: case OP_STR_FROM_FLOAT: case OP_STR_SPLIT:
        r->recipe=NVM_RA_ROOT_STRING;
        r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_LOAD_LOCAL: case OP_LOAD_GLOBAL:r->recipe=NVM_RA_ROOT_LOAD;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_STORE_LOCAL: case OP_STORE_GLOBAL:r->recipe=NVM_RA_ROOT_STORE;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_DUP:r->recipe=NVM_RA_ROOT_DUP;r->obligations|=NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_SWAP:r->recipe=NVM_RA_ROOT_SWAP;break;
    case OP_POP: case OP_ASSERT:r->recipe=NVM_RA_ROOT_DROP;break;
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:r->recipe=NVM_RA_ROOT_BRANCH;break;
    case OP_CALL:r->recipe=NVM_RA_ROOT_CALL;r->obligations|=NVM_RA_STAGE_OPERANDS;break;
    case OP_TAIL_CALL:r->recipe=NVM_RA_ROOT_TAIL;r->obligations|=NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_RET:r->recipe=NVM_RA_ROOT_RETURN;r->obligations|=NVM_RA_STAGE_OPERANDS;break;
    case OP_STRUCT_NEW: case OP_STRUCT_LITERAL: case OP_AGG_PACK:
        r->recipe=NVM_RA_ROOT_CONSTRUCT;
        r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_ARR_NEW: case OP_ARR_LITERAL:
        r->recipe=NVM_RA_ROOT_CONSTRUCT;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;break;
    case OP_STRUCT_GET: case OP_AGG_GET:
        r->recipe=NVM_RA_ROOT_GET;r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_GET:r->recipe=NVM_RA_ROOT_GET;r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_STRUCT_SET: case OP_AGG_SET:
        r->recipe=NVM_RA_ROOT_SET;r->obligations|=NVM_RA_CHECK_NOMINAL|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_SET:r->recipe=NVM_RA_ROOT_SET;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_PUSH:r->recipe=NVM_RA_ROOT_ARRAY_PUSH;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_POP:r->recipe=NVM_RA_ROOT_ARRAY_POP;r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;break;
    case OP_ARR_SLICE:r->recipe=NVM_RA_ROOT_ARRAY_COPY;r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS|NVM_RA_CHECK_BOUNDS;break;
    default:return 0;
    }
    return 1;
}
static int ufn_sensitive(uint8_t op) { return (op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_STRUCT_GET || op==OP_STRUCT_SET || op==OP_AGG_PACK || op==OP_AGG_GET || op==OP_AGG_SET) || op==OP_AGG_TAG || op==OP_HALT; }
static uint8_t ufn_site_operation(const DecodedInstruction *in) {
    switch(in->opcode) {
    case OP_STRUCT_NEW:return UFS_RECORD_NEW;
    case OP_STRUCT_LITERAL:return UFS_RECORD_PACK;
    case OP_AGG_PACK:return in->operands[0].u8==AGG_VARIANT?UFS_VARIANT_PACK:UFS_RECORD_PACK;
    case OP_STRUCT_GET:return UFS_RECORD_GET;
    case OP_STRUCT_SET:case OP_AGG_SET:return UFS_RECORD_SET;
    case OP_AGG_TAG:return UFS_TAG;
    case OP_HALT:return UFS_HALT;
    default:return UFS_NONE;
    }
}
static int ufn_recipe(uint8_t op,NvmRecordArrayExecutionInstruction *r) {
    if(!ufn_sensitive(op))return (nvm_record_array_opcode_supported(op) && ufn_base_recipe(op,r));
    r->recipe=NVM_UF_INVALID_BASE_RECIPE;
    r->obligations=NVM_RA_FIRST_ERROR_CLEANUP;
    if(op==OP_HALT)return 1;
    r->obligations|=NVM_RA_CHECK_TAGS|NVM_RA_CHECK_NOMINAL;
    if(op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_AGG_PACK)
        r->obligations|=NVM_RA_SAFEPOINT_BEFORE|NVM_RA_STAGE_OPERANDS;
    else if(op!=OP_AGG_TAG)
        r->obligations|=NVM_RA_CHECK_BOUNDS|NVM_RA_RETAIN_BEFORE_RELEASE;
    return 1;
}
static int ufn_empty(NvmRecordValueOrigins v) { return !v.tags && !v.origins && !v.unknown; }
typedef struct {
    NvmUnionFieldOrigin rows[64];
    uint64_t records,variants[2];
    uint32_t count;
} UfnOrigins;
static int ufn_origins_value(const UfnOrigins *o,NvmRecordValueOrigins v) {
    uint64_t unions=o->variants[0]|o->variants[1];
    if(v.unknown || (v.tags&~(UFN_BIT(TAG_VOID)|UFN_BIT(TAG_INT)|UFN_BIT(TAG_U8)|UFN_BIT(TAG_FLOAT)|
        UFN_BIT(TAG_BOOL)|UFN_BIT(TAG_STRING)|UFN_BIT(TAG_STRUCT)|UFN_BIT(TAG_UNION))) ||
       (v.origins&~(o->records|unions)))return 0;
    if(!!(v.tags&UFN_BIT(TAG_STRUCT))!=!!(v.origins&o->records) ||
       !!(v.tags&UFN_BIT(TAG_UNION))!=!!(v.origins&unions))return 0;
    return 1;
}
static int ufn_origin_correspondence(const UfnOrigins *o,const NvmUnionFieldSite *s) {
    if(!ufn_origins_value(o,s->receiver) || !ufn_origins_value(o,s->input) || !ufn_origins_value(o,s->result))return 0;
    if(s->state!=UFS_PROVED)return 1;
    if(s->operation==UFS_VARIANT_PACK || s->operation==UFS_RECORD_PACK || s->operation==UFS_RECORD_NEW) {
        uint64_t site=0;
        for(uint32_t i=0;i<o->count;i++) {
            const NvmUnionFieldOrigin *r=&o->rows[i];
            if(r->function==s->function && r->pc==s->pc) {
                if(site || r->nominal_ordinal!=s->nominal || r->layout_index!=s->layout ||
                   r->member_count!=s->member || r->variant!=s->variant ||
                   r->kind!=(s->operation==UFS_VARIANT_PACK?NVM_UNION_FIELD_ORIGIN:NVM_HEAP_ORIGIN_RECORD) ||
                   (s->input.tags&~r->member.tags) || (s->input.origins&~r->member.origins))return 0;
                site=UINT64_C(1)<<i;
            }
        }
        return site && site==s->result.origins;
    }
    if(s->operation==UFS_PAYLOAD_GET)
        return s->variant<2 && s->receiver.origins && !(s->receiver.origins&~o->variants[s->variant]);
    if(s->operation==UFS_RECORD_GET || s->operation==UFS_RECORD_SET) {
        uint64_t children=0;uint16_t tags=0;
        for(uint32_t i=0;i<o->count;i++)if(s->receiver.origins&(UINT64_C(1)<<i)) {
            const NvmUnionFieldOrigin *r=&o->rows[i];
            if(r->kind!=NVM_HEAP_ORIGIN_RECORD || r->nominal_ordinal!=s->nominal || r->layout_index!=s->layout)return 0;
            if(s->operation==UFS_RECORD_SET && ((s->input.origins&~r->member.origins) || (s->input.tags&~r->member.tags)))return 0;
            children|=r->member.origins;tags|=r->member.tags;
        }
        if(s->operation==UFS_RECORD_GET)return children==s->result.origins && tags==s->result.tags;
    }
    return 1;
}
/* Internal immutable rows came from this constructor's successful analysis.
 * This checks decoded correspondence, not an externally supplied authority. */
static int ufn_corresponds(const NvmRecordArrayExecutionInstruction *r,const NvmUnionFieldSite *s,
                          uint32_t record_layout,uint32_t union_layout,const NvmUnionVariantFact variants[2],const UfnOrigins *origins) {
    if(!ufn_origin_correspondence(origins,s))return 0;
    NvmRecordArrayExecutionInstruction expected={0};
    if(!ufn_recipe(r->opcode,&expected) || expected.recipe!=r->recipe || expected.obligations!=r->obligations)return 0;
    if(r->function!=s->function || r->pc!=s->pc || s->reserved[0] || s->reserved[1] || s->reserved[2] ||
       s->receiver.unknown || s->input.unknown || s->result.unknown)return 0;
    if(!ufn_sensitive(r->opcode))return s->state==UFS_NOT_REQUIRED && s->operation==UFS_NONE &&
        ufn_empty(s->receiver) && ufn_empty(s->input) && ufn_empty(s->result) &&
        s->nominal==UINT32_MAX && s->layout==UINT32_MAX && s->member==UINT16_MAX && s->variant==UINT8_MAX;
    if(r->recipe!=NVM_UF_INVALID_BASE_RECIPE)return 0;
    DecodedInstruction in={0};in.opcode=r->opcode;in.operands[0].u8=(uint8_t)r->operand_bits[0];
    uint8_t operation=ufn_site_operation(&in);
    if(s->state==UFS_NOT_ESTABLISHED)return s->operation==operation &&
        ufn_empty(s->receiver) && ufn_empty(s->input) && ufn_empty(s->result) &&
        s->nominal==UINT32_MAX && s->layout==UINT32_MAX && s->member==UINT16_MAX && s->variant==UINT8_MAX;
    if(s->state!=UFS_PROVED)return 0;
    if(r->opcode==OP_HALT)return s->operation==UFS_HALT && !r->successor_count && !r->pops && !r->pushes &&
        ufn_empty(s->receiver) && ufn_empty(s->input) && ufn_empty(s->result) &&
        s->nominal==UINT32_MAX && s->layout==UINT32_MAX && s->member==UINT16_MAX && s->variant==UINT8_MAX;
    if(s->nominal!=0)return 0;
    switch(s->operation) {
    case UFS_RECORD_NEW:case UFS_RECORD_PACK:case UFS_VARIANT_PACK: {
        int variant=s->operation==UFS_VARIANT_PACK;
        if(s->operation!=operation || !ufn_empty(s->receiver) || s->layout!=(variant?union_layout:record_layout) ||
           s->result.tags!=UFN_BIT(variant?TAG_UNION:TAG_STRUCT) || !s->result.origins)return 0;
        if(r->opcode==OP_AGG_PACK) {
            if(r->operand_bits[0]!=(uint64_t)(variant?AGG_VARIANT:AGG_RECORD) || r->operand_bits[1]!=s->nominal ||
               r->operand_bits[2]!=(variant?s->variant:0) || r->operand_bits[3]!=s->member)return 0;
        } else if(r->operand_bits[0]!=s->nominal || (r->opcode==OP_STRUCT_LITERAL && r->operand_bits[1]!=s->member))return 0;
        uint16_t count=1;
        if(variant) {
            if(s->variant>=2)return 0;
            count=variants[s->variant].field_count;
        } else if(s->variant!=UINT8_MAX)return 0;
        return r->pushes==1 && r->pops==(r->opcode==OP_STRUCT_NEW?0:count) && s->member==count && (count?(s->input.tags==UFN_BIT(variant?TAG_INT:TAG_UNION) &&
             (variant?!s->input.origins:!!s->input.origins)):ufn_empty(s->input));
    }
    case UFS_RECORD_GET:
        return r->pops==1 && r->pushes==1 && (r->opcode==OP_STRUCT_GET || r->opcode==OP_AGG_GET) && !r->operand_bits[0] && !s->member &&
            s->layout==record_layout && s->variant==UINT8_MAX && s->receiver.tags==UFN_BIT(TAG_STRUCT) &&
            s->receiver.origins && ufn_empty(s->input) && s->result.tags==UFN_BIT(TAG_UNION) && s->result.origins;
    case UFS_PAYLOAD_GET:
        return r->pops==1 && r->pushes==1 && r->opcode==OP_AGG_GET && !r->operand_bits[0] && !s->member && s->layout==union_layout &&
            s->variant<2 && variants[s->variant].field_count==1 && s->receiver.tags==UFN_BIT(TAG_UNION) &&
            s->receiver.origins && ufn_empty(s->input) && s->result.tags==UFN_BIT(TAG_INT) && !s->result.origins;
    case UFS_RECORD_SET:
        return r->pops==2 && r->pushes==1 && (r->opcode==OP_STRUCT_SET || r->opcode==OP_AGG_SET) && !r->operand_bits[0] && !s->member &&
            s->layout==record_layout && s->variant==UINT8_MAX && s->receiver.tags==UFN_BIT(TAG_STRUCT) && s->receiver.origins &&
            s->input.tags==UFN_BIT(TAG_UNION) && s->input.origins && s->result.tags==s->receiver.tags && s->result.origins==s->receiver.origins;
    case UFS_TAG:
        return r->pops==1 && r->pushes==1 && r->opcode==OP_AGG_TAG && s->layout==union_layout && s->variant==UINT8_MAX && s->member==UINT16_MAX &&
            s->receiver.tags==UFN_BIT(TAG_UNION) && s->receiver.origins && ufn_empty(s->input) && s->result.tags==UFN_BIT(TAG_INT) && !s->result.origins;
    default:return 0;
    }
}
typedef struct {
    UfnOrigins origins;
    NvmUnionFieldExecutionInstruction instruction;
    NvmRecordArrayExecutionFunction function;
    NvmHeader header;
    NvmRecordArrayExecutionCounts counts;
    NvmUnionFieldProofCounts proof;
    NvmRecordValueOrigins value;
    NvmDeclarationLayout layouts[2];
    NvmUnionVariantFact variants[2];
    NvmV2LayoutField fields[2];
    Nvm2cUnionTargetContext staged;
} Nvm2cUnionTargetScratch;
_Static_assert(sizeof(Nvm2cUnionTargetScratch)<65536,"I bound native union target scratch.");
_Static_assert(MAX_OPERANDS==4,"I audit four native operands.");
_Static_assert(ISA_MAX_INSTRUCTION_SIZE<=33,"I audit bounded native instruction bytes.");
/* No scan or allocation: numeric snapshot correspondence only. */
static int ufn_signature(const NvmFunctionEntry *a,const NvmFunctionEntry *b) {
    return a->name_idx==b->name_idx && a->arity==b->arity &&
        a->code_offset==b->code_offset && a->code_length==b->code_length &&
        a->local_count==b->local_count && a->upvalue_count==b->upvalue_count &&
        a->result_tag==b->result_tag && a->result_count==b->result_count;
}
static int ufn_boundary(const NvmUnionFieldImage *image,const NvmRecordArrayExecutionFunction *f,
                        uint32_t pc,Nvm2cUnionTargetResult *result,uint64_t allowance) {
    if(!ufn_charge(result,allowance,8))return 0;
    if(pc==f->signature.code_length)return 1;
    uint32_t lo=0,hi=f->instruction_count;
    while(lo<hi) {
        NvmUnionFieldExecutionInstruction row;
        if(!ufn_charge(result,allowance,4+sizeof row+16))return 0;
        uint32_t mid=lo+(hi-lo)/2;
        if(!nvm_union_field_image_instruction(image,f->instruction_start+mid,&row))return 0;
        if(row.base.pc==pc)return 1;
        if(row.base.pc<pc)lo=mid+1;else hi=mid;
    }
    return 0;
}
/* Allowance is an internal whitebox limit, never caller execution authority.
 * Header/getter/copy charges are explicit. Fixed-block allowances below cover
 * bounded scalar comparisons/recipe switches; origin correspondence reserves
 * 4096 (64 visits with <=48 scalar checks plus fixed row checks), before use.
 * This conservatively groups the logical operations; it is not timing/fuel. */
static Nvm2cUnionTargetResult nvm2c_union_target(const NvmUnionFieldImage *image,
        uint64_t allowance,Nvm2cUnionTargetContext *out) {
    Nvm2cUnionTargetResult result={{NVM_ARRAY_INVALID,0,0,{0}},0,0};
    if(allowance>UFN_WORK_LIMIT)allowance=UFN_WORK_LIMIT;
#define UFN_CHARGE(n) do {if(!ufn_charge(&result,allowance,(uint64_t)(n)))goto done;}while(0)
#define UFN_GET(call,row) do {UFN_CHARGE(4+sizeof(row));if(!(call))goto done;}while(0)
#define UFN_REQUIRE(expr) do {if(!(expr))goto done;}while(0)
    /* Reserve ceiling before scratch initialization. No heap allocation occurs. */
    result.scratch_reserved=UFN_SCRATCH_LIMIT;
    UFN_CHARGE(sizeof(Nvm2cUnionTargetScratch)+128);
    Nvm2cUnionTargetScratch s={0};
    const char *unsupported=NULL;
    UFN_REQUIRE(image && out);
    const NvmModule *m=nvm_union_field_image_module_internal(image);
    UFN_REQUIRE(m);
    UFN_GET(nvm_union_field_image_header(image,&s.header),s.header);
    UFN_GET(nvm_union_field_image_counts(image,&s.counts),s.counts);
    UFN_GET(nvm_union_field_image_proof_counts(image,&s.proof),s.proof);
    UFN_CHARGE(128);
    UFN_REQUIRE(s.counts.functions && s.counts.functions<=256 && s.counts.instructions<=65536 &&
        s.counts.globals<=256 && s.proof.origins<=64 && s.proof.globals<=256 &&
        s.proof.functions==s.counts.functions && s.proof.globals==s.counts.globals &&
        s.counts.functions==m->function_count && m->functions &&
        s.counts.strings<=NVM_MAX_STRINGS && s.counts.sections<=NVM_MAX_SECTIONS &&
        m->code_size<=16u*1024u*1024u &&
        !memcmp(&s.header,&m->header,sizeof s.header) &&
        s.counts.entry==s.header.entry_point && s.counts.entry<m->function_count &&
        s.counts.records==1 && m->struct_count==1 && m->union_count==1 &&
        s.counts.frame_limit==NVM_RECORD_ARRAY_EXECUTION_FRAMES &&
        s.counts.strings==m->string_count && s.counts.sections==m->section_count &&
        s.counts.metadata==m->metadata_count && s.counts.debug==m->debug_count &&
        (!m->code_size || m->code) && (!m->string_count || (m->strings && m->string_lengths)));
    if(m->import_count || m->module_ref_count || m->callback_contract_count || m->service_size || m->passive_size)
        unsupported="I require a closed native union module.";
    s.staged.image=image;s.staged.module=m;
    UFN_GET(nvm_union_field_image_nominal(image,TAG_STRUCT,0,&s.staged.record_layout),s.staged.record_layout);
    UFN_GET(nvm_union_field_image_nominal(image,TAG_UNION,0,&s.staged.union_layout),s.staged.union_layout);
    UFN_REQUIRE(s.staged.record_layout==1 && s.staged.union_layout==0);
    for(uint32_t i=0;i<2;i++) {
        UFN_CHARGE(16);
        UFN_GET(nvm_union_field_image_layout(image,i,&s.layouts[i]),s.layouts[i]);
        UFN_GET(nvm_union_field_image_variant(image,0,(uint16_t)i,&s.variants[i]),s.variants[i]);
        UFN_GET(nvm_union_field_image_field(image,i,0,&s.fields[i]),s.fields[i]);
    }
    UFN_CHARGE(64);
    UFN_REQUIRE(s.layouts[0].kind==NVM_V2_LAYOUT_UNION && s.layouts[1].kind==NVM_V2_LAYOUT_STRUCT &&
        s.layouts[0].fields==1 && s.layouts[1].fields==1 &&
        s.layouts[0].flags==NVM_LAYOUT_COMPLETE && s.layouts[1].flags==NVM_LAYOUT_COMPLETE &&
        s.fields[0].type_tag==TAG_INT && s.fields[0].nested_idx==NVM_V2_NO_INDEX &&
        s.fields[1].type_tag==TAG_UNION && s.fields[1].nested_idx==0 &&
        s.layouts[0].name<m->string_count && s.layouts[1].name<m->string_count &&
        s.fields[0].name_idx<m->string_count && s.fields[1].name_idx<m->string_count &&
        s.variants[0].name_idx<m->string_count && s.variants[1].name_idx<m->string_count &&
        s.variants[0].layout==0 && s.variants[1].layout==0 &&
        s.variants[0].field_count<=1 && s.variants[1].field_count<=1 &&
        s.variants[0].field_count+s.variants[1].field_count==1 &&
        s.variants[0].field_offset==0 && s.variants[1].field_offset==s.variants[0].field_count);
    s.staged.payload_variant=s.variants[0].field_count?0:1;
    s.staged.empty_variant=(uint8_t)(1-s.staged.payload_variant);
    s.origins.count=s.proof.origins;
    for(uint32_t i=0;i<s.proof.origins;i++) {
        UFN_CHARGE(96);
        NvmUnionFieldOrigin *r=&s.origins.rows[i];
        UFN_GET(nvm_union_field_image_origin(image,i,r),*r);
        UFN_REQUIRE(r->function<m->function_count && r->pc<m->functions[r->function].code_length &&
            !r->nominal_ordinal && !r->member.unknown);
        if(r->kind==NVM_HEAP_ORIGIN_RECORD) {
            UFN_REQUIRE(r->layout_index==1 && r->variant==UINT8_MAX && r->member_count==1 && r->member.tags==UFN_BIT(TAG_UNION));
            s.origins.records|=UINT64_C(1)<<i;
        } else {
            UFN_REQUIRE(r->kind==NVM_UNION_FIELD_ORIGIN && r->layout_index==0 && r->variant<2 &&
                r->member_count==s.variants[r->variant].field_count &&
                (r->member_count?(r->member.tags==UFN_BIT(TAG_INT) && !r->member.origins):ufn_empty(r->member)));
            s.origins.variants[r->variant]|=UINT64_C(1)<<i;
        }
    }
    for(uint32_t i=0;i<s.proof.origins;i++) {
        UFN_CHARGE(32);UFN_REQUIRE(ufn_origins_value(&s.origins,s.origins.rows[i].member));
    }
    for(uint32_t i=0;i<s.proof.globals;i++) {
        UFN_CHARGE(32);UFN_GET(nvm_union_field_image_global(image,i,&s.value),s.value);
        UFN_REQUIRE(ufn_origins_value(&s.origins,s.value));
    }
    for(uint32_t op=0;op<256;op++) {
        UFN_CHARGE(128+sizeof(NvmRecordArrayExecutionInstruction));
        NvmRecordArrayExecutionInstruction expected={0};
        UFN_REQUIRE(!!ufn_recipe((uint8_t)op,&expected)==
            !!(nvm_record_array_opcode_supported((uint8_t)op) || op==OP_AGG_TAG || op==OP_HALT));
    }
    uint32_t total=0,initializer=UINT32_MAX,globals=0;
    uint64_t matched_origins=0;
    for(uint32_t fi=0;fi<s.counts.functions;fi++) {
        UFN_CHARGE(128);
        UFN_GET(nvm_union_field_image_function(image,fi,&s.function),s.function);
        UFN_GET(nvm_union_field_image_result(image,fi,&s.value),s.value);
        const NvmFunctionEntry *f=&m->functions[fi];
        UFN_REQUIRE(ufn_signature(f,&s.function.signature) && ufn_origins_value(&s.origins,s.value) &&
            f->result_count<=1 && f->arity<=256 && f->local_count<=256 && f->arity<=f->local_count && s.function.maximum_stack<=256 &&
            s.function.instruction_start==total && total<=s.counts.instructions &&
            s.function.instruction_count<=s.counts.instructions-total &&
            f->code_offset<=m->code_size && f->code_length<=m->code_size-f->code_offset &&
            f->name_idx<m->string_count && m->strings[f->name_idx] &&
            s.function.parameter_tags_present==!!(m->function_param_types && m->function_param_types[fi]));
        if(f->upvalue_count)unsupported="I require capture-free native union functions.";
        uint64_t name_bytes=(uint64_t)m->string_lengths[f->name_idx]+1;
        UFN_REQUIRE(name_bytes<=SIZE_MAX && name_bytes<=UINT64_MAX-16);
        UFN_CHARGE(name_bytes+16);
        UFN_REQUIRE(!memchr(m->strings[f->name_idx],0,(size_t)(name_bytes-1)) &&
            m->strings[f->name_idx][name_bytes-1]==0);
        if(initializer==UINT32_MAX && name_bytes==9 && !memcmp(m->strings[f->name_idx],"__init__",8))initializer=fi;
        for(uint32_t j=0;j<f->arity;j++) {
            UFN_CHARGE(16);
            if(s.function.parameter_tags_present) {
                uint8_t tag=0;UFN_GET(nvm_union_field_image_parameter(image,fi,(uint16_t)j,&tag),tag);
                UFN_REQUIRE(tag==m->function_param_types[fi][j]);
            }
        }
        uint32_t pc=0;
        for(uint32_t j=0;j<s.function.instruction_count;j++) {
            UFN_CHARGE(1024+sizeof(DecodedInstruction)+sizeof(NvmRecordArrayExecutionInstruction));
            UFN_GET(nvm_union_field_image_instruction(image,total+j,&s.instruction),s.instruction);
            NvmRecordArrayExecutionInstruction *r=&s.instruction.base;
            UFN_REQUIRE(pc<f->code_length);
            DecodedInstruction in={0};
            uint32_t width=isa_decode(m->code+f->code_offset+pc,f->code_length-pc,&in);
            const InstructionInfo *info=width?isa_get_info(in.opcode):NULL;
            UFN_REQUIRE(info && width<=ISA_MAX_INSTRUCTION_SIZE && width<=f->code_length-pc &&
                info->operand_count<=MAX_OPERANDS && r->function==fi && r->pc==pc && r->next_pc==pc+width &&
                r->opcode==in.opcode && r->operand_count==info->operand_count);
            uint32_t offset=1;
            for(uint32_t k=0;k<MAX_OPERANDS;k++) {
                if(k<info->operand_count) {
                    uint32_t bytes=isa_operand_size(info->operands[k]);uint64_t bits=0;
                    UFN_REQUIRE(bytes<=8 && offset<=width && bytes<=width-offset && r->operand_types[k]==info->operands[k]);
                    for(uint32_t b=0;b<bytes;b++)bits|=(uint64_t)m->code[f->code_offset+pc+offset+b]<<(8*b);
                    UFN_REQUIRE(bits==r->operand_bits[k]);offset+=bytes;
                } else UFN_REQUIRE(!r->operand_types[k] && !r->operand_bits[k]);
            }
            UFN_REQUIRE(offset==width);
            int pops=info->pop_count,pushes=info->push_count;
            uint32_t callee=UINT32_MAX,successors[2]={0,0};uint8_t nsuccessors=0;
            if(in.opcode==OP_ARR_LITERAL || in.opcode==OP_STRUCT_LITERAL){pops=in.operands[1].u16;pushes=1;}
            if(in.opcode==OP_AGG_PACK){pops=in.operands[3].u16;pushes=1;}
            if(in.opcode==OP_CALL || in.opcode==OP_TAIL_CALL) {
                callee=in.operands[0].u32;UFN_REQUIRE(callee<m->function_count);
                pops=m->functions[callee].arity;pushes=in.opcode==OP_CALL?m->functions[callee].result_count:0;
            }
            if(in.opcode==OP_RET){pops=f->result_count;pushes=0;}
            UFN_REQUIRE(pops>=0 && pops<=256 && pushes>=0 && pushes<=256 &&
                r->pops==pops && r->pushes==pushes && r->callee==callee);
            if(in.opcode==OP_JMP || in.opcode==OP_JMP_TRUE || in.opcode==OP_JMP_FALSE) {
                int64_t target=(int64_t)pc+in.operands[0].i32;
                UFN_REQUIRE(target>=0 && (uint64_t)target<=f->code_length);
                successors[nsuccessors++]=(uint32_t)target;
            }
            if(in.opcode!=OP_JMP && in.opcode!=OP_RET && in.opcode!=OP_TAIL_CALL && in.opcode!=OP_HALT)
                successors[nsuccessors++]=pc+width;
            UFN_REQUIRE(r->successor_count==nsuccessors && r->successors[0]==successors[0] && r->successors[1]==successors[1]);
            for(uint8_t k=0;k<nsuccessors;k++)
                UFN_REQUIRE(ufn_boundary(image,&s.function,successors[k],&result,allowance));
            UFN_CHARGE(4096);
            UFN_REQUIRE(ufn_corresponds(r,&s.instruction.site,1,0,s.variants,&s.origins));
            /* Reverse correspondence includes dormant construction rows. A
             * scalar/mid-instruction or duplicate origin cannot disappear. */
            UFN_CHARGE(4096);
            uint64_t at_site=0;
            for(uint32_t oi=0;oi<s.origins.count;oi++) {
                const NvmUnionFieldOrigin *o=&s.origins.rows[oi];
                if(o->function!=fi || o->pc!=pc)continue;
                uint64_t bit=UINT64_C(1)<<oi;
                UFN_REQUIRE(!at_site && !(matched_origins&bit));
                at_site=bit;
                if(in.opcode==OP_AGG_PACK) {
                    bool variant=in.operands[0].u8==AGG_VARIANT;
                    UFN_REQUIRE((variant || in.operands[0].u8==AGG_RECORD) &&
                        o->kind==(variant?NVM_UNION_FIELD_ORIGIN:NVM_HEAP_ORIGIN_RECORD) &&
                        o->nominal_ordinal==in.operands[1].u32 &&
                        o->variant==(variant?in.operands[2].u16:UINT8_MAX) &&
                        o->member_count==in.operands[3].u16);
                } else {
                    UFN_REQUIRE((in.opcode==OP_STRUCT_LITERAL || in.opcode==OP_STRUCT_NEW) &&
                        o->kind==NVM_HEAP_ORIGIN_RECORD && o->variant==UINT8_MAX &&
                        o->nominal_ordinal==in.operands[0].u32 &&
                        (in.opcode==OP_STRUCT_NEW || o->member_count==in.operands[1].u16));
                }
                matched_origins|=bit;
            }

            if(in.opcode==OP_LOAD_GLOBAL || in.opcode==OP_STORE_GLOBAL) {
                UFN_REQUIRE(in.operands[0].u32<256);
                uint32_t end=in.operands[0].u32+1;if(end>globals)globals=end;
            }
            if(in.opcode==OP_PUSH_STR) {
                uint32_t at=in.operands[0].u32;
                UFN_REQUIRE(at<m->string_count && m->strings[at]);
                uint64_t bytes=(uint64_t)m->string_lengths[at]+1;
                UFN_REQUIRE(bytes<=SIZE_MAX);
                UFN_CHARGE(bytes);
                UFN_REQUIRE(m->strings[at][bytes-1]==0);
                if(memchr(m->strings[at],0,(size_t)(bytes-1)))unsupported="I require NUL-free executable native union literals.";
            }
            if(s.instruction.site.state==UFS_NOT_ESTABLISHED)
                unsupported="I have not qualified dormant native union blocks.";
            if(in.opcode==OP_STRUCT_NEW || (r->recipe==NVM_RA_ROOT_STRING && in.opcode!=OP_STR_CONCAT) ||
                in.opcode==OP_ARR_NEW || in.opcode==OP_ARR_LITERAL || in.opcode==OP_ARR_GET || in.opcode==OP_ARR_SET ||
                in.opcode==OP_ARR_PUSH || in.opcode==OP_ARR_POP || in.opcode==OP_ARR_SLICE || in.opcode==OP_ARR_LEN)
                unsupported="I have not qualified this native union producer or array composition.";
            pc+=width;
        }
        UFN_REQUIRE(pc==f->code_length);total+=s.function.instruction_count;
    }
    UFN_CHARGE(64+sizeof(Nvm2cUnionTargetContext));
    UFN_REQUIRE(total==s.counts.instructions && initializer==s.counts.initializer && globals==s.counts.globals &&
        matched_origins==(s.origins.count==64?UINT64_MAX:((UINT64_C(1)<<s.origins.count)-1)));
    const NvmFunctionEntry *entry=&m->functions[s.counts.entry];
    if(entry->arity || entry->upvalue_count || entry->result_count!=1 || entry->result_tag!=TAG_INT)
        unsupported="I require a zero-argument INT native union entry.";
    if(initializer!=UINT32_MAX) {
        const NvmFunctionEntry *init=&m->functions[initializer];
        if(init->arity || init->upvalue_count || init->result_count>1 ||
            (init->result_count?init->result_tag!=TAG_INT:init->result_tag!=TAG_VOID))
            unsupported="I require a zero-argument VOID/INT native union initializer.";
    }
    if(unsupported) {
        result.eligibility.status=NVM_ARRAY_UNRESOLVED;
        snprintf(result.eligibility.message,sizeof result.eligibility.message,"%s",unsupported);
    } else {
        result.eligibility.status=NVM_ARRAY_ELIGIBLE;
        *out=s.staged;
    }
done:
    if(!result.eligibility.message[0])snprintf(result.eligibility.message,sizeof result.eligibility.message,
        "%s",result.eligibility.status==NVM_ARRAY_ELIGIBLE?
        "I prepared a private native union target; I emit no code.":"I could not prepare exact bounded native union target facts.");
#undef UFN_CHARGE
#undef UFN_GET
#undef UFN_REQUIRE
    return result;
}
/* Test/feature owning-TU seam only. No output context or borrowed image escapes.
 * This query success is target preparation, not public/execution eligibility. */
static Nvm2cUnionTargetResult nvm2c_prepare_union_target_source(const NvmModule *source) {
    Nvm2cUnionTargetResult result={{NVM_ARRAY_INVALID,0,0,{0}},0,0};
    NvmUnionFieldExecutionPlan *plan=NULL;NvmUnionFieldImage *image=NULL;
    result.eligibility=nvm_prepare_union_field_execution(source,&plan);
    if(result.eligibility.status!=NVM_ARRAY_ELIGIBLE)return result;
    result.eligibility=nvm_union_field_image_copy(plan,&image);
    nvm_union_field_execution_free(plan);
    if(result.eligibility.status==NVM_ARRAY_ELIGIBLE) {
        Nvm2cUnionTargetContext context;
        result=nvm2c_union_target(image,UFN_WORK_LIMIT,&context);
    }
    nvm_union_field_image_free(image);
    return result;
}
#undef UFN_BIT
