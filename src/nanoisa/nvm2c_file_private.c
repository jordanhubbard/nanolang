#include "nvm2c_file_private.h"
#ifdef NVM_FILE_NATIVE_PRIVATE
#include "nvm_v2_sections.h"
#include "../nsi_file_catalog.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Generation only. No generated function depends on this emitter or VM code. */
typedef struct { char *text; size_t used,capacity; NvmFileRuntimeStatus status; } Fne;
static void fn_text(Fne *b,const char *format,...) {
    if(b->status!=NVM_FILE_RUNTIME_OK)return;
    va_list ap,copy;va_start(ap,format);va_copy(copy,ap);
    int n=vsnprintf(NULL,0,format,copy);va_end(copy);
    if(n<0){b->status=NVM_FILE_RUNTIME_INVALID;va_end(ap);return;}
    size_t size=(size_t)n;
    if(size>=NVM_FILE_NATIVE_OUTPUT_BYTES || b->used>NVM_FILE_NATIVE_OUTPUT_BYTES-size-1u) {
        b->status=NVM_FILE_RUNTIME_LIMIT;va_end(ap);return;
    }
    size_t need=b->used+size+1;
    if(need>b->capacity) {
        size_t capacity=b->capacity?b->capacity:4096;
        while(capacity<need) {
            if(capacity>NVM_FILE_NATIVE_OUTPUT_BYTES/2u){capacity=NVM_FILE_NATIVE_OUTPUT_BYTES;break;}
            capacity*=2;
        }
        char *text=realloc(b->text,capacity);
        if(!text){b->status=NVM_FILE_RUNTIME_MEMORY;va_end(ap);return;}
        b->text=text;b->capacity=capacity;
    }
    int written=vsnprintf(b->text+b->used,b->capacity-b->used,format,ap);va_end(ap);
    if(written!=n){b->status=NVM_FILE_RUNTIME_INVALID;return;}
    b->used+=size;
}
static void fn_expect(Fne *b,const char *field,uint64_t value) {
    fn_text(b,"if((uint64_t)(%s)!=UINT64_C(%" PRIu64 "))return false;\n",field,value);
}
#define FN_FIELD(b,p,v,member) fn_expect((b),p #member,(uint64_t)((v).member))
static void fn_declaration(Fne *b,const char *prefix,NvmFileFlowDeclaration d) {
    char field[96];
#define DECL(member) do { (void)snprintf(field,sizeof field,"%s%s",prefix,#member);fn_expect(b,field,d.member); } while(0)
    DECL(tag);DECL(mode);DECL(global_index);DECL(catalog_ordinal);DECL(category);
#undef DECL
}
static bool fne_supported(uint8_t op) {
    switch(op) {
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
    case OP_DUP: case OP_POP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
    case OP_REGION_BEGIN: case OP_REGION_END: case OP_BORROW_LOCAL_EXCLUSIVE:
    case OP_FILE_SERVICE: case OP_FILE_RESULT_BRANCH: case OP_FILE_RESULT_TAKE:
    case OP_FILE_DROP_LOCAL: case OP_FILE_DROP_STACK: case OP_FILE_END_BORROW:
    case OP_CALL: case OP_CALL_REF: case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE:
    case OP_RET: case OP_ASSERT:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S:
    case OP_I64_REM_S: case OP_I64_NEG:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S:
    case OP_I64_GT_S: case OP_I64_GE_S: case OP_AND: case OP_OR: case OP_NOT:
    case OP_AGG_PACK: case OP_UNION_CONSTRUCT: case OP_AGG_GET: case OP_UNION_FIELD:
    case OP_AGG_TAG: case OP_UNION_TAG:return true;
    default:return false;
    }
}
static bool fne_fact(const NvmFileHostedPlan *p,const NvmFileCodeInstruction *in,
                     const NvmFileBodyInstruction *fact) {
    uint8_t op=in->decoded.opcode;
    if(!fne_supported(op))return false;
    if(!fact->reachable)return true; /* Opcode coverage still includes dead code. */
    NvmFileBodyCleanup cleanup=NVM_FILE_BODY_CLEANUP_NORMAL;
    switch(op) {
    case OP_FILE_DROP_LOCAL:cleanup=NVM_FILE_BODY_CLEANUP_DROP_LOCAL;break;
    case OP_FILE_DROP_STACK:cleanup=NVM_FILE_BODY_CLEANUP_DROP_STACK;break;
    case OP_ASSERT:cleanup=NVM_FILE_BODY_CLEANUP_ASSERT;break;
    case OP_RET:cleanup=NVM_FILE_BODY_CLEANUP_RETURN;break;
    case OP_CALL:case OP_CALL_REF:cleanup=NVM_FILE_BODY_CLEANUP_CALL;break;
    case OP_FILE_SERVICE:cleanup=NVM_FILE_BODY_CLEANUP_SERVICE;break;
    default:break;
    }
    if(fact->cleanup!=cleanup || fact->refinement!=(op==OP_FILE_RESULT_BRANCH) ||
       fact->exit_checked!=(op==OP_RET))return false;
    uint32_t checks=NVM_FILE_FLOW_CHECK_CLEANUP,discharged=0;
    bool service=op==OP_FILE_SERVICE,call=op==OP_CALL || op==OP_CALL_REF;
    if(fact->has_obligation!=(service || call))return false;
    if(service || call) {
        const NvmFileFlowObligation *o=&fact->obligation;
        if(o->site!=in->byte_offset || o->target!=in->decoded.operands[0].u32 ||
           o->kind!=(service?NVM_FILE_FLOW_SERVICE:NVM_FILE_FLOW_CALL))return false;
        checks|=NVM_FILE_FLOW_CHECK_RESULT;
        if(service) {
            uint32_t ordinal;
            if(!nvm_file_hosted_import(p,o->target,&ordinal) || ordinal!=in->catalog_ordinal)return false;
            const NlFilePlanMethod *method=nl_file_catalog_method(ordinal);
            if(!method || o->required_rights!=method->required_rights ||
               o->acquired_rights!=method->acquired_rights)return false;
            checks|=NVM_FILE_FLOW_CHECK_BINDING|NVM_FILE_FLOW_CHECK_INVOCATION|
                    NVM_FILE_FLOW_CHECK_LIVENESS|NVM_FILE_FLOW_CHECK_RIGHTS;
            if(ordinal>=1 && ordinal<=3)checks|=NVM_FILE_FLOW_CHECK_BORROW;
            if(ordinal==1)checks|=NVM_FILE_FLOW_CHECK_BYTE;
        } else { checks|=NVM_FILE_FLOW_CHECK_CALLEE;discharged=NVM_FILE_FLOW_CHECK_CALLEE; }
        if(o->checks!=checks)return false;
    }
    return fact->discharged_checks==discharged && fact->pending_checks==(checks&~discharged);
}
static bool fne_coverage(const NvmFileHostedPlan *p) {
    NvmFileHostedStartup startup;
    if(!nvm_file_hosted_startup(p,&startup))return false;
    for(uint32_t f=0;f<startup.functions;f++) {
        NvmFileHostedFunction fn;if(!nvm_file_hosted_function(p,f,&fn))return false;
        for(uint16_t i=0;i<fn.code.instruction_count;i++) {
            NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
            if(!nvm_file_hosted_instruction(p,f,i,&in,&fact) || !fne_fact(p,&in,&fact))return false;
        }
    }
    return true;
}

/* I compare semantic fields, not padding, hashes or pointer identities. */
static void fn_agreement(Fne *b,const NvmFileHostedPlan *p,NvmFileHostedStartup start) {
    fn_text(b,"static bool nf_agrees(const NvmFileHostedPlan *p){\n"
        "NvmFileHostedStartup s; NvmFileHostedFunction f; NvmFileFlowDeclaration l;\n"
        "NvmFileCodeInstruction in; NvmFileBodyInstruction fact; NvmFileNominalLayout t;\n"
        "uint32_t ordinal; (void)l;(void)ordinal;\n"
        "if(!nvm_file_hosted_startup(p,&s))return false;\n");
#define START(m) FN_FIELD(b,"s.",start,m)
    START(entry);START(initializer);START(functions);START(features);START(vm_value_slots);
    START(native_value_slots);START(frames);START(reference_slots);START(region_slots);START(allocation_bound);
#undef START
    for(uint32_t i=0;i<NVM_FILE_NOMINAL_TYPES;i++) {
        NvmFileNominalLayout t;
        if(!nvm_file_hosted_type(p,i,&t)){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        fn_text(b,"if(!nvm_file_hosted_type(p,%u,&t))return false;\n",i);
#define TYPE(m) FN_FIELD(b,"t.",t,m)
        TYPE(global_index);TYPE(catalog_ordinal);TYPE(source_ordinal);TYPE(layout_kind);TYPE(ownership_flags);TYPE(category);
#undef TYPE
    }
    for(uint32_t index=0;index<start.functions;index++) {
        NvmFileHostedFunction f;
        if(!nvm_file_hosted_function(p,index,&f)){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        fn_text(b,"if(!nvm_file_hosted_function(p,%u,&f))return false;\n",index);
#define FUNC(m) FN_FIELD(b,"f.",f,m)
        FUNC(code.code_offset);FUNC(code.code_length);FUNC(code.instruction_count);
        FUNC(code.declaration.parameters);FUNC(code.declaration.locals);FUNC(code.declaration.result_count);
        fn_declaration(b,"f.code.declaration.result.",f.code.declaration.result);
        FUNC(declared_stack);FUNC(operand_peak);FUNC(locals);FUNC(staging_slots);FUNC(frames);
        FUNC(reference_slots);FUNC(region_slots);FUNC(vm_value_slots);FUNC(native_value_slots);
#undef FUNC
        for(uint16_t j=0;j<f.locals;j++) {
            NvmFileFlowDeclaration l;
            if(!nvm_file_hosted_local(p,index,j,&l)){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
            fn_text(b,"if(!nvm_file_hosted_local(p,%u,%u,&l))return false;\n",index,j);
            fn_declaration(b,"l.",l);
        }
        for(uint16_t j=0;j<f.code.instruction_count;j++) {
            NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
            if(!nvm_file_hosted_instruction(p,index,j,&in,&fact)){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
            fn_text(b,"if(!nvm_file_hosted_instruction(p,%u,%u,&in,&fact))return false;\n",index,j);
#define IN(m) FN_FIELD(b,"in.",in,m)
            IN(byte_offset);IN(decoded.opcode);IN(decoded.operand_count);IN(decoded.byte_length);
            IN(successor_count);IN(catalog_ordinal);
            for(uint8_t k=0;k<in.successor_count;k++)
                fn_text(b,"if(in.successors[%u]!=%u)return false;\n",k,in.successors[k]);
            for(uint8_t k=0;k<in.decoded.operand_count;k++) {
                const char *member=NULL;uint64_t value=0;
                switch(in.decoded.operand_types[k]) {
                case OPERAND_U8:member="u8";value=in.decoded.operands[k].u8;break;
                case OPERAND_U16:member="u16";value=in.decoded.operands[k].u16;break;
                case OPERAND_U32:member="u32";value=in.decoded.operands[k].u32;break;
                case OPERAND_I32:member="i32";value=(uint64_t)(int64_t)in.decoded.operands[k].i32;break;
                case OPERAND_I64:member="i64";value=(uint64_t)in.decoded.operands[k].i64;break;
                default:b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;
                }
                fn_text(b,"if(in.decoded.operand_types[%u]!=%u || (uint64_t)in.decoded.operands[%u].%s!=UINT64_C(%" PRIu64 "))return false;\n",k,in.decoded.operand_types[k],k,member,value);
            }
#undef IN
#define FACT(m) FN_FIELD(b,"fact.",fact,m)
            FACT(reachable);FACT(exit_checked);FACT(refinement);FACT(has_obligation);
            FACT(input_stack);FACT(output_stack);FACT(cleanup_local);FACT(cleanup);
            FACT(discharged_checks);FACT(pending_checks);
            if(fact.has_obligation) {
                FACT(obligation.kind);FACT(obligation.site);FACT(obligation.target);FACT(obligation.checks);
                FACT(obligation.required_rights);FACT(obligation.acquired_rights);
                FACT(obligation.parameters);FACT(obligation.owned_inputs);FACT(obligation.borrowed_inputs);
                FACT(obligation.result_count);FACT(obligation.outcomes[0]);FACT(obligation.outcomes[1]);
                fn_declaration(b,"fact.obligation.result.",fact.obligation.result);
            }
#undef FACT
            if(in.decoded.opcode==OP_FILE_SERVICE) {
                fn_text(b,"if(!nvm_file_hosted_import(p,%u,&ordinal) || ordinal!=%u)return false;\n",in.decoded.operands[0].u32,in.catalog_ordinal);
            }
        }
    }
    fn_text(b,"return true;}\n");
}

static void fn_preamble(Fne *b) {
    fn_text(b,"%s","#include \"src/nanoisa/nvm2c_file_private.h\"\n");
    fn_text(b,"%s","#include \"src/nanoisa/nvm_v2_sections.h\"\n");
    fn_text(b,"%s","#include <limits.h>\n");
    fn_text(b,"%s","#include <string.h>\n");
    fn_text(b,"%s","#ifndef NVM_FILE_NATIVE_PRIVATE\n");
    fn_text(b,"%s","#error I require the private File native provider configuration\n");
    fn_text(b,"%s","#endif\n");
    fn_text(b,"%s","#if NVM_FILE_NATIVE_ABI != 1u\n");
    fn_text(b,"%s","#error I require the qualified File native semantic ABI revision\n");
    fn_text(b,"%s","#endif\n");
    fn_text(b,"%s","#define NF_TRY(expr) do { NvmFileRuntimeStatus nf_status_=(expr); if(nf_status_!=NVM_FILE_RUNTIME_OK)return nf_status_; } while(0)\n");
    fn_text(b,"%s","static NvmFileRuntimeStatus nf_bad(NvmFileRuntime *c){return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_STATE);}\n");
    fn_text(b,"%s","static int64_t nf_bits(uint64_t x){int64_t y;memcpy(&y,&x,sizeof y);return y;}\n");
    fn_text(b,"%s","static bool nf_scalar(const NvmFileRuntimeView *v,uint8_t tag){\n");
    fn_text(b,"%s"," return v->initialized && !v->owning && !v->formal && !v->type.mode &&\n");
    fn_text(b,"%s"," v->type.category==NVM_FILE_CATEGORY_UNKNOWN && v->type.tag==tag &&\n");
    fn_text(b,"%s"," v->type.global_index==NVM_V2_NO_INDEX && v->type.catalog_ordinal==NVM_V2_NO_INDEX &&\n");
    fn_text(b,"%s"," v->fields==(tag==TAG_VOID?0:1) && (tag!=TAG_BOOL || v->values[0]==0 || v->values[0]==1);\n");
    fn_text(b,"%s","}\n");
    fn_text(b,"%s","static NvmFileRuntimeReport nf_refused(NvmFileRuntimeStatus status){\n");
    fn_text(b,"%s"," NvmFileRuntimeReport r={0};r.status=status;r.function=r.instruction=NVM_V2_NO_INDEX;return r;\n");
    fn_text(b,"%s","}\n");
}

static void fn_numeric(Fne *b,uint8_t op) {
    bool unary=op==OP_NEG || op==OP_I64_NEG || op==OP_NOT;
    bool logic=op==OP_AND || op==OP_OR || op==OP_NOT;
    uint16_t count=unary?1:2;
    const char *expression=NULL;uint8_t result=TAG_INT;
    switch(op) {
    case OP_ADD:case OP_I64_ADD:expression="nf_bits((uint64_t)a+(uint64_t)z)";break;
    case OP_SUB:case OP_I64_SUB:expression="nf_bits((uint64_t)a-(uint64_t)z)";break;
    case OP_MUL:case OP_I64_MUL:expression="nf_bits((uint64_t)a*(uint64_t)z)";break;
    case OP_DIV:case OP_I64_DIV_S:expression="!z?0:a==INT64_MIN&&z==-1?INT64_MIN:a/z";break;
    case OP_MOD:case OP_I64_REM_S:expression="!z||(a==INT64_MIN&&z==-1)?0:a%z";break;
    case OP_NEG:case OP_I64_NEG:expression="nf_bits(UINT64_C(0)-(uint64_t)a)";break;
    case OP_EQ:case OP_I64_EQ:expression="a==z";result=TAG_BOOL;break;
    case OP_NE:case OP_I64_NE:expression="a!=z";result=TAG_BOOL;break;
    case OP_LT:case OP_I64_LT_S:expression="a<z";result=TAG_BOOL;break;
    case OP_LE:case OP_I64_LE_S:expression="a<=z";result=TAG_BOOL;break;
    case OP_GT:case OP_I64_GT_S:expression="a>z";result=TAG_BOOL;break;
    case OP_GE:case OP_I64_GE_S:expression="a>=z";result=TAG_BOOL;break;
    case OP_AND:expression="a&&z";result=TAG_BOOL;break;
    case OP_OR:expression="a||z";result=TAG_BOOL;break;
    case OP_NOT:expression="!a";result=TAG_BOOL;break;
    default:b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;
    }
    fn_text(b,"if(f.stack_count<%u)return nf_bad(c);\n",count);
    for(uint16_t k=0;k<count;k++)fn_text(b,
        "if(!nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-%u),&roots[%u]) || !nvm_file_runtime_view(c,roots[%u],&values[%u]))return nf_bad(c);\n",count-k,k,k,k);
    fn_text(b,"tag=%u;\n",logic?TAG_BOOL:TAG_INT);
    if(op==OP_EQ || op==OP_NE)fn_text(b,"if(values[0].type.tag==TAG_BOOL)tag=TAG_BOOL;\n");
    for(uint16_t k=0;k<count;k++)fn_text(b,"if(!nf_scalar(&values[%u],tag))return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);\n",k);
    fn_text(b,"a=values[0].values[0];z=%s;r=(%s);\n",unary?"0":"values[1].values[0]",expression);
    for(uint16_t k=0;k<count;k++)fn_text(b,"NF_TRY(nvm_file_runtime_drop(c,roots[%u]));\n",k);
    fn_text(b,"NF_TRY(nvm_file_runtime_scalar(c,roots[0],%u,r));\n",result);
}
static void fn_instruction(Fne *b,uint32_t function,const NvmFileCodeInstruction *in) {
    const DecodedInstruction *d=&in->decoded;
    uint8_t op=d->opcode;bool branch=false;
    switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_PUSH_I64:case OP_PUSH_BOOL:case OP_PUSH_VOID:
        fn_text(b,"if(!nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(nvm_file_runtime_scalar(c,dst,%u,nf_bits(UINT64_C(%" PRIu64 "))));\n",
            op==OP_PUSH_I64?TAG_INT:op==OP_PUSH_BOOL?TAG_BOOL:TAG_VOID,
            op==OP_PUSH_I64?(uint64_t)d->operands[0].i64:op==OP_PUSH_BOOL?(uint64_t)d->operands[0].u8:UINT64_C(0));break;
    case OP_DUP:
        fn_text(b,"if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) || !nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(nvm_file_runtime_copy(c,src,dst));\n");break;
    case OP_POP:case OP_FILE_DROP_STACK:
        fn_text(b,"if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) || !nvm_file_runtime_view(c,src,&value))return nf_bad(c);\n");
        if(op==OP_POP)fn_text(b,"if(value.owning || value.formal)return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);\n");
        fn_text(b,"NF_TRY(nvm_file_runtime_drop(c,src));\n");break;
    case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
        fn_text(b,"if(!nvm_file_runtime_frame_local(c,%u,&src) || !nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst) || !nvm_file_runtime_view(c,src,&value))return nf_bad(c);\n"
            "if(!value.initialized || value.formal || value.owning!=%u)return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);\n"
            "NF_TRY(nvm_file_runtime_%s(c,src,dst));\n",d->operands[0].u16,op==OP_OWN_MOVE_LOCAL,op==OP_OWN_MOVE_LOCAL?"move":"copy");break;
    case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:fn_text(b,"NF_TRY(nvm_file_runtime_frame_store(c));\n");break;
    case OP_REGION_BEGIN:fn_text(b,"NF_TRY(nvm_file_runtime_frame_region_begin(c));\n");break;
    case OP_REGION_END:fn_text(b,"NF_TRY(nvm_file_runtime_frame_region_end(c));\n");break;
    case OP_BORROW_LOCAL_EXCLUSIVE:fn_text(b,"NF_TRY(nvm_file_runtime_frame_borrow(c));\n");break;
    case OP_FILE_END_BORROW:fn_text(b,"NF_TRY(nvm_file_runtime_frame_end_reference(c));\n");break;
    case OP_FILE_DROP_LOCAL:
        fn_text(b,"if(!nvm_file_runtime_frame_local(c,%u,&src))return nf_bad(c);\nNF_TRY(nvm_file_runtime_drop(c,src));\n",d->operands[0].u16);break;
    case OP_FILE_RESULT_BRANCH:
        fn_text(b,"if(!nvm_file_runtime_frame_local(c,%u,&src))return nf_bad(c);\nNF_TRY(nvm_file_runtime_result_arm(c,src,&arm));\nedge=arm==NVM_FILE_FLOW_ARM_ERROR;\n",d->operands[0].u16);branch=true;break;
    case OP_FILE_RESULT_TAKE:
        fn_text(b,"if(!nvm_file_runtime_frame_local(c,%u,&src) || !nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(nvm_file_runtime_take(c,src,%u,dst));\n",d->operands[0].u16,d->operands[1].u8?NVM_FILE_FLOW_ARM_ERROR:NVM_FILE_FLOW_ARM_OK);break;
    case OP_FILE_SERVICE: {
        uint32_t ordinal=in->catalog_ordinal;bool consumes=ordinal==1 || ordinal==4;
        fn_text(b,"src=reference=NVM_FILE_RUNTIME_NO_SLOT;\nif(!nvm_file_runtime_frame_scratch(c,&dst))return nf_bad(c);\n");
        if(consumes)fn_text(b,"if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src))return nf_bad(c);\n");
        if(ordinal>=1 && ordinal<=3)fn_text(b,"if(!nvm_file_runtime_frame_reference(c,%u,&reference))return nf_bad(c);\n",d->operands[1].u16);
        fn_text(b,"NF_TRY(nvm_file_runtime_service(c,%u,reference,src,dst));\n"
            "if(!nvm_file_runtime_frame_reserve(c,(uint16_t)(f.stack_count-%u),&output))return nf_bad(c);\n"
            "NF_TRY(nvm_file_runtime_move(c,dst,output));\n",d->operands[0].u32,consumes?1u:0u);break;
    }
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        fn_text(b,"if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) || !nvm_file_runtime_view(c,src,&value))return nf_bad(c);\n"
            "if(!nf_scalar(&value,TAG_BOOL))return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);\n"
            "NF_TRY(nvm_file_runtime_drop(c,src));\n");
        if(op==OP_ASSERT)fn_text(b,"if(!value.values[0])return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_ASSERT);\n");
        else {fn_text(b,"edge=(value.values[0]!=0)==%u;\n",op==OP_JMP_TRUE);branch=true;}
        break;
    case OP_AGG_PACK:case OP_UNION_CONSTRUCT: {
        uint16_t variant=d->operands[op==OP_AGG_PACK?2:1].u16,count=d->operands[op==OP_AGG_PACK?3:2].u16;
        if(count>NVM_FILE_RUNTIME_FIELDS){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        if(count)fn_text(b,"if(f.stack_count<%u)return nf_bad(c);\n",count);
        for(uint16_t k=0;k<count;k++)fn_text(b,"if(!nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-%u),&inputs[%u]))return nf_bad(c);\n",count-k,k);
        if(count)fn_text(b,"dst=inputs[0];\n");else fn_text(b,"if(!nvm_file_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n");
        fn_text(b,"NF_TRY(nvm_file_runtime_construct(c,%u,%u,inputs,%u,dst));\n",in->catalog_ordinal,variant,count);break;
    }
    case OP_AGG_GET:case OP_UNION_FIELD:case OP_AGG_TAG:case OP_UNION_TAG:
        fn_text(b,"if(!f.stack_count || !nvm_file_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src))return nf_bad(c);\n");
        if(op==OP_AGG_GET || op==OP_UNION_FIELD)fn_text(b,"NF_TRY(nvm_file_runtime_project(c,src,%u,src));\n",d->operands[0].u16);
        else fn_text(b,"if(!nvm_file_runtime_view(c,src,&value))return nf_bad(c);\n"
            "if(value.owning || value.formal || value.type.category!=NVM_FILE_CATEGORY_SCALAR_RESULT)return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_TYPE);\n"
            "NF_TRY(nvm_file_runtime_result_arm(c,src,&arm));\nNF_TRY(nvm_file_runtime_drop(c,src));\n"
            "NF_TRY(nvm_file_runtime_scalar(c,src,TAG_INT,arm==NVM_FILE_FLOW_ARM_ERROR));\n");break;
    case OP_CALL:case OP_CALL_REF:
        if(in->successor_count!=1){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        fn_text(b,"NF_TRY(nvm_file_runtime_frame_call(c));\nNF_TRY(nf_function_%u(c,steps));\ngoto nf_label_%u;\n",d->operands[0].u32,in->successors[0]);return;
    case OP_RET:fn_text(b,"return nvm_file_runtime_frame_return(c);\n");return;
    default:fn_numeric(b,op);break;
    }
    (void)function;
    if(branch) {
        if(in->successor_count!=2){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        fn_text(b,"NF_TRY(nvm_file_runtime_frame_next(c,edge));\nif(edge)goto nf_label_%u;\ngoto nf_label_%u;\n",in->successors[1],in->successors[0]);
    } else {
        if(in->successor_count!=1){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        fn_text(b,"NF_TRY(nvm_file_runtime_frame_next(c,0));\ngoto nf_label_%u;\n",in->successors[0]);
    }
}
static void fn_function(Fne *b,const NvmFileHostedPlan *p,uint32_t index) {
    NvmFileHostedFunction f;
    if(!nvm_file_hosted_function(p,index,&f)){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
    fn_text(b,"static NvmFileRuntimeStatus nf_function_%u(NvmFileRuntime *c,uint64_t *steps){\n"
        "NvmFileRuntimeFrameView f; NvmFileRuntimeView value,values[2]; NvmFileFlowArm arm;\n"
        "uint32_t src=0,dst=0,output=0,reference=0,roots[2],inputs[NVM_FILE_RUNTIME_FIELDS];\n"
        "uint8_t edge=0,tag=0;int64_t a=0,z=0,r=0;\n"
        "(void)value;(void)values;(void)arm;(void)src;(void)dst;(void)output;(void)reference;\n"
        "(void)roots;(void)inputs;(void)edge;(void)tag;(void)a;(void)z;(void)r;\n",index);
    /* Mark dead labels referenced without executing them. They remain explicit
     * refusal bodies, never an inferred successful instruction implementation. */
    for(uint16_t j=0;j<f.code.instruction_count;j++)fn_text(b,"if(0)goto nf_label_%u;\n",j);
    fn_text(b,"goto nf_label_0;\n");
    for(uint16_t j=0;j<f.code.instruction_count;j++) {
        NvmFileCodeInstruction in;NvmFileBodyInstruction fact;
        if(!nvm_file_hosted_instruction(p,index,j,&in,&fact)){b->status=NVM_FILE_RUNTIME_UNRESOLVED;return;}
        fn_text(b,"nf_label_%u:\n",j);
        if(!fact.reachable){fn_text(b,"return nf_bad(c);\n");continue;}
        fn_text(b,"if(!nvm_file_runtime_frame_view(c,&f) || f.mode!=NVM_FILE_RUNTIME_NATIVE || f.function!=%u || f.instruction!=%u || f.stack_count!=%u)return nf_bad(c);\n"
            "if(*steps==UINT64_MAX)return nvm_file_runtime_fail(c,NVM_FILE_RUNTIME_LIMIT);\n++*steps;\n",index,j,fact.input_stack);
        fn_instruction(b,index,&in);
    }
    fn_text(b,"}\n");
}
NvmFileRuntimeStatus nvm2c_file_private_emit(const uint8_t *bytes,size_t size,char **out,char *err,size_t err_size) {
    if(!out || !bytes || !size || (err_size && !err))return NVM_FILE_RUNTIME_INVALID;
    NvmFileHostedPlan *plan=NULL;
    NvmFileFlowStatus status=nvm_file_hosted_prepare(bytes,size,&plan);
    NvmFileRuntimeStatus result=NVM_FILE_RUNTIME_OK;
    if(status!=NVM_FILE_FLOW_OK) {
        result=status==NVM_FILE_FLOW_MEMORY?NVM_FILE_RUNTIME_MEMORY:status==NVM_FILE_FLOW_LIMIT?NVM_FILE_RUNTIME_LIMIT:
            status==NVM_FILE_FLOW_UNRESOLVED?NVM_FILE_RUNTIME_UNRESOLVED:NVM_FILE_RUNTIME_INVALID;
        if(err_size)(void)snprintf(err,err_size,"I refuse private File native preparation (%u)",(unsigned)result);
        return result;
    }
    NvmFileHostedStartup startup;
    if(!nvm_file_hosted_startup(plan,&startup) || !fne_coverage(plan)) {
        nvm_file_hosted_free(plan);
        if(err_size)(void)snprintf(err,err_size,"I lack complete private File native instruction coverage");
        return NVM_FILE_RUNTIME_UNRESOLVED;
    }
    Fne b={0};fn_preamble(&b);
    fn_text(&b,"static const uint8_t nf_module[]={\n");
    for(size_t i=0;i<size;i++)fn_text(&b,"%u,%s",bytes[i],i%24==23?"\n":"");
    fn_text(&b,"\n};\n");fn_agreement(&b,plan,startup);
    for(uint32_t f=0;f<startup.functions;f++)fn_text(&b,"static NvmFileRuntimeStatus nf_function_%u(NvmFileRuntime *,uint64_t *);\n",f);
    for(uint32_t f=0;f<startup.functions;f++)fn_function(&b,plan,f);
    fn_text(&b,"NvmFileRuntimeReport nvm_file_native_execute(NvmFileRuntimeView *out){\n"
        "if(!out)return nf_refused(NVM_FILE_RUNTIME_INVALID);\n"
        "if(!nvm_file_runtime_native_abi(1u,sizeof(NvmFileRuntimeView),sizeof(NvmFileRuntimeFrameView),sizeof(NvmFileRuntimeReport)))return nf_refused(NVM_FILE_RUNTIME_UNRESOLVED);\n"
        "NvmFileRuntime *c=NULL;uint64_t steps=0;uint32_t root;\n"
        "NvmFileRuntimeStatus status=nvm_file_runtime_create(nf_module,sizeof nf_module,NVM_FILE_RUNTIME_NATIVE,&c);\n"
        "if(status!=NVM_FILE_RUNTIME_OK)return nf_refused(status);\n"
        "if(!nf_agrees(nvm_file_runtime_plan(c))){(void)nvm_file_runtime_destroy(&c,NULL);return nf_refused(NVM_FILE_RUNTIME_UNRESOLVED);}\n"
        "status=nvm_file_runtime_begin(c);\n"
        "(void)nf_bits;(void)nf_scalar;\n");
    for(uint32_t f=0;f<startup.functions;f++)fn_text(&b,"(void)nf_function_%u;\n",f);
    if(startup.initializer!=NVM_V2_NO_INDEX) {
        fn_text(&b,"if(status==NVM_FILE_RUNTIME_OK)status=nvm_file_runtime_frame_start(c);\n"
            "if(status==NVM_FILE_RUNTIME_OK)status=nf_function_%u(c,&steps);\n",startup.initializer);
    }
    fn_text(&b,"if(status==NVM_FILE_RUNTIME_OK && (!nvm_file_runtime_current_root(c,&root) || root!=%u))status=nf_bad(c);\n"
        "if(status==NVM_FILE_RUNTIME_OK)status=nvm_file_runtime_frame_start(c);\n"
        "if(status==NVM_FILE_RUNTIME_OK)status=nf_function_%u(c,&steps);\n"
        "if(status==NVM_FILE_RUNTIME_OK && nvm_file_runtime_current_root(c,&root))status=nf_bad(c);\n"
        "if(status!=NVM_FILE_RUNTIME_OK)(void)nvm_file_runtime_fail(c,status);\n"
        "return nvm_file_runtime_destroy(&c,out);\n}\n#undef NF_TRY\n",startup.entry,startup.entry);
    nvm_file_hosted_free(plan);
    result=b.status;
    if(result!=NVM_FILE_RUNTIME_OK) {
        free(b.text);if(err_size)(void)snprintf(err,err_size,"I refuse private File native emission (%u)",(unsigned)result);return result;
    }
    *out=b.text;if(err_size)err[0]='\0';return NVM_FILE_RUNTIME_OK;
}
#undef FN_FIELD
#endif
