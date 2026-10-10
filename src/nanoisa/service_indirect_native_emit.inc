#include "nvm_v2_sections.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef enum { FNE_PRIVATE, FNE_PUBLIC } FileIndirectNativeSurface;

/* Generation only. No generated function depends on this emitter or VM code. */
typedef struct { char *text; size_t used,capacity; SERVICE_TYPE(RuntimeStatus) status; } Fne;
static void fn_text(Fne *b,const char *format,...) {
    if(b->status!=SERVICE_CONST(RUNTIME_OK))return;
    va_list ap,copy;va_start(ap,format);va_copy(copy,ap);
    int n=vsnprintf(NULL,0,format,copy);va_end(copy);
    if(n<0){b->status=SERVICE_CONST(RUNTIME_INVALID);va_end(ap);return;}
    size_t size=(size_t)n;
    if(size>=SERVICE_CONST(INDIRECT_NATIVE_OUTPUT_BYTES) || b->used>SERVICE_CONST(INDIRECT_NATIVE_OUTPUT_BYTES)-size-1u) {
        b->status=SERVICE_CONST(RUNTIME_LIMIT);va_end(ap);return;
    }
    size_t need=b->used+size+1;
    if(need>b->capacity) {
        size_t capacity=b->capacity?b->capacity:4096;
        while(capacity<need) {
            if(capacity>SERVICE_CONST(INDIRECT_NATIVE_OUTPUT_BYTES)/2u){capacity=SERVICE_CONST(INDIRECT_NATIVE_OUTPUT_BYTES);break;}
            capacity*=2;
        }
        char *text=realloc(b->text,capacity);
        if(!text){b->status=SERVICE_CONST(RUNTIME_MEMORY);va_end(ap);return;}
        b->text=text;b->capacity=capacity;
    }
    int written=vsnprintf(b->text+b->used,b->capacity-b->used,format,ap);va_end(ap);
    if(written!=n){b->status=SERVICE_CONST(RUNTIME_INVALID);return;}
    b->used+=size;
}
static void fn_expect(Fne *b,const char *field,uint64_t value) {
    fn_text(b,"if((uint64_t)(%s)!=UINT64_C(%" PRIu64 "))return false;\n",field,value);
}
#define FN_FIELD(b,p,v,member) fn_expect((b),p #member,(uint64_t)((v).member))
static void fn_declaration(Fne *b,const char *prefix,SERVICE_TYPE(FlowDeclaration) d) {
    char field[96];
#define DECL(member) do { (void)snprintf(field,sizeof field,"%s%s",prefix,#member);fn_expect(b,field,d.member); } while(0)
    DECL(tag);DECL(mode);DECL(global_index);DECL(catalog_ordinal);DECL(category);
#undef DECL
}
#include "service_indirect_dispatch.inc"

static void fn_value(Fne *b,SERVICE_TYPE(FlowValue) value) {
    fn_declaration(b,"value.type.",value.type);
    FN_FIELD(b,"value.",value,initialized);FN_FIELD(b,"value.",value,owner);
    FN_FIELD(b,"value.",value,arm);
}
static void fn_variant(Fne *b,const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t f,uint16_t i,uint8_t v) {
    SERVICE_TYPE(CyclicVariant) variant;
    if(!SERVICE_FN(indirect_hosted_variant)(p,f,i,v,&variant)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
    fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_variant(p,%u,%u,%u,&variant))return false;\n",f,i,v);
#define V(m) FN_FIELD(b,"variant.",variant,m)
    V(input.locals);V(input.stack);V(input.regions);V(input.owners);V(input.references);
    V(output.locals);V(output.stack);V(output.regions);V(output.owners);V(output.references);
    V(edge_mask);V(edge_variants[0]);V(edge_variants[1]);
    V(body.reachable);V(body.exit_checked);V(body.refinement);V(body.has_obligation);
    V(body.input_stack);V(body.output_stack);V(body.cleanup_local);V(body.cleanup);
    V(body.discharged_checks);V(body.pending_checks);
    if(variant.body.has_obligation) {
        V(body.obligation.kind);V(body.obligation.site);V(body.obligation.target);V(body.obligation.checks);
        V(body.obligation.required_rights);V(body.obligation.acquired_rights);
        V(body.obligation.parameters);V(body.obligation.owned_inputs);V(body.obligation.borrowed_inputs);
        V(body.obligation.result_count);V(body.obligation.outcomes[0]);V(body.obligation.outcomes[1]);
        fn_declaration(b,"variant.body.obligation.result.",variant.body.obligation.result);
    }
#undef V
    SERVICE_TYPE(CodeInstruction) instruction;
    if(!SERVICE_FN(indirect_hosted_instruction)(p,f,i,&instruction)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
    if(isa_is_file_indirect_call(instruction.decoded.opcode)) {
        SERVICE_TYPE(IndirectFlowCall) call;
        if(!SERVICE_FN(indirect_hosted_call)(p,f,i,v,&call)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"{" SERVICE_TYPE_TEXT "IndirectFlowCall call;if(!" SERVICE_FN_TEXT "indirect_hosted_call(p,%u,%u,%u,&call))return false;\n",f,i,v);
        FN_FIELD(b,"call.",call,function);FN_FIELD(b,"call.",call,pc);
        FN_FIELD(b,"call.",call,candidates);FN_FIELD(b,"call.",call,checked_candidates);
#define CALL(m) FN_FIELD(b,"call.common.",call.common,m)
        CALL(kind);CALL(site);CALL(target);CALL(checks);CALL(required_rights);CALL(acquired_rights);
        CALL(parameters);CALL(owned_inputs);CALL(borrowed_inputs);CALL(result_count);
        CALL(outcomes[0]);CALL(outcomes[1]);
#undef CALL
        fn_declaration(b,"call.common.result.",call.common.result);
        fn_text(b,"}\n");
    }
    for(uint16_t n=0;n<variant.input.locals;n++) {
        SERVICE_TYPE(FlowValue) value;
        if(!SERVICE_FN(indirect_hosted_input_local)(p,f,i,v,n,&value)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_input_local(p,%u,%u,%u,%u,&value))return false;\n",f,i,v,n);
        fn_value(b,value);
    }
    for(uint16_t n=0;n<variant.input.stack;n++) {
        SERVICE_TYPE(FlowValue) value;
        if(!SERVICE_FN(indirect_hosted_input_stack)(p,f,i,v,n,&value)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_input_stack(p,%u,%u,%u,%u,&value))return false;\n",f,i,v,n);
        fn_value(b,value);
    }
    /* I compare every reference slot. Empty canonical slots share an explicit
     * zero-field predicate, keeping output bounded without omitting relations. */
    fn_text(b,"for(uint16_t n=0;n<" SERVICE_CONST_TEXT "FLOW_REFERENCES;n++){\n"
        "if(!" SERVICE_FN_TEXT "indirect_hosted_input_reference(p,%u,%u,%u,n,&reference))return false;\nswitch(n){\n",f,i,v);
    for(uint16_t n=0;n<SERVICE_CONST(FLOW_REFERENCES);n++) {
        SERVICE_TYPE(FlowReference) reference;
        if(!SERVICE_FN(indirect_hosted_input_reference)(p,f,i,v,n,&reference)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        if(!reference.live) {
            if(reference.formal || reference.local || reference.owner || reference.identity || reference.region || reference.shared) {
                b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;
            }
            continue;
        }
        fn_text(b,"case %u:\n",n);
#define REF(m) FN_FIELD(b,"reference.",reference,m)
        REF(live);REF(formal);REF(local);REF(owner);REF(identity);REF(region);REF(shared);
#undef REF
        fn_text(b,"break;\n");
    }
    fn_text(b,"default:if(reference.live || reference.formal || reference.local || reference.owner || reference.identity || reference.region || reference.shared)return false;break;}}\n");
    for(uint16_t n=0;n<variant.input.regions;n++) {
        uint64_t region;
        if(!SERVICE_FN(indirect_hosted_input_region)(p,f,i,v,n,&region)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_input_region(p,%u,%u,%u,%u,&region) || region!=UINT64_C(%" PRIu64 "))return false;\n",f,i,v,n,region);
    }
}
static void fn_instruction_agreement(Fne *b,const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t index,uint16_t j) {
    fn_text(b,"static bool nf_agrees_%u_%u(const " SERVICE_TYPE_TEXT "IndirectHostedPlan *p){\n"
        "" SERVICE_TYPE_TEXT "CodeInstruction in;" SERVICE_TYPE_TEXT "CyclicVariant variant;\n"
        "" SERVICE_TYPE_TEXT "FlowValue value;" SERVICE_TYPE_TEXT "FlowReference reference;\n"
        "uint32_t ordinal;uint16_t component;uint8_t count;uint64_t region;\n"
        "(void)variant;(void)value;(void)reference;(void)ordinal;(void)region;\n",index,j);
    SERVICE_TYPE(CodeInstruction) in;uint8_t count;uint16_t component;
    if(!SERVICE_FN(indirect_hosted_instruction)(p,index,j,&in) ||
       !SERVICE_FN(indirect_hosted_variant_count)(p,index,j,&count) ||
       !SERVICE_FN(indirect_hosted_component)(p,index,j,&component)) {b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
    fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_instruction(p,%u,%u,&in) || !" SERVICE_FN_TEXT "indirect_hosted_variant_count(p,%u,%u,&count) || count!=%u || !" SERVICE_FN_TEXT "indirect_hosted_component(p,%u,%u,&component) || component!=%u)return false;\n",index,j,index,j,count,index,j,component);
#define IN(m) FN_FIELD(b,"in.",in,m)
    IN(byte_offset);IN(decoded.opcode);IN(decoded.operand_count);IN(decoded.byte_length);
    IN(successor_count);IN(catalog_ordinal);
#undef IN
    for(uint8_t k=0;k<in.successor_count;k++)fn_text(b,"if(in.successors[%u]!=%u)return false;\n",k,in.successors[k]);
    for(uint8_t k=0;k<in.decoded.operand_count;k++) {
        const char *member=NULL;uint64_t value=0;
        switch(in.decoded.operand_types[k]) {
        case OPERAND_U8:member="u8";value=in.decoded.operands[k].u8;break;
        case OPERAND_U16:member="u16";value=in.decoded.operands[k].u16;break;
        case OPERAND_U32:member="u32";value=in.decoded.operands[k].u32;break;
        case OPERAND_I32:member="i32";value=(uint64_t)(int64_t)in.decoded.operands[k].i32;break;
        case OPERAND_I64:member="i64";value=(uint64_t)in.decoded.operands[k].i64;break;
        default:b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;
        }
        fn_text(b,"if(in.decoded.operand_types[%u]!=%u || (uint64_t)in.decoded.operands[%u].%s!=UINT64_C(%" PRIu64 "))return false;\n",k,in.decoded.operand_types[k],k,member,value);
    }
    if(in.decoded.opcode==OP_CALL || in.decoded.opcode==OP_CALL_REF || in.decoded.opcode==OP_FILE_CALL_REFS) {
        SERVICE_TYPE(IndirectHostedFunction) callee;
        if(!SERVICE_FN(indirect_hosted_function)(p,in.decoded.operands[0].u32,&callee)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        for(uint16_t k=0;k<callee.code.declaration.parameters;k++)
            fn_text(b,"if(in.call_references[%u]!=%u)return false;\n",k,in.call_references[k]);
    }
    if(in.decoded.opcode==OP_FILE_CALL_INDIRECT_REFS)
        for(uint16_t k=0;k<in.decoded.operands[0].u16;k++)
            fn_text(b,"if(in.call_references[%u]!=%u)return false;\n",k,in.call_references[k]);
    for(uint8_t v=0;v<count;v++)fn_variant(b,p,index,j,v);
    if(in.decoded.opcode==OP_FILE_SERVICE) {
        uint32_t original;
        if(!SERVICE_FN(indirect_hosted_import)(p,in.catalog_ordinal,&original)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_import(p,%u,&ordinal) || ordinal!=%u)return false;\n",in.catalog_ordinal,original);
    }
    fn_text(b,"return true;}\n");
}
static void fn_agreement(Fne *b,const SERVICE_TYPE(IndirectHostedPlan) *p,SERVICE_TYPE(IndirectHostedStartup) start) {
    for(uint32_t f=0;f<start.functions;f++) {
        SERVICE_TYPE(IndirectHostedFunction) function;
        if(!SERVICE_FN(indirect_hosted_function)(p,f,&function)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        for(uint16_t i=0;i<function.code.instruction_count;i++)fn_instruction_agreement(b,p,f,i);
    }
    fn_text(b,"static bool nf_agrees(const " SERVICE_TYPE_TEXT "IndirectHostedPlan *p){\n"
        "" SERVICE_TYPE_TEXT "IndirectHostedStartup s;" SERVICE_TYPE_TEXT "IndirectHostedFunction f;" SERVICE_TYPE_TEXT "FlowDeclaration l;\n"
        "" SERVICE_TYPE_TEXT "NominalLayout t;uint32_t ordinal;uint8_t byte;\n"
        "(void)l;\n"
        "if(!" SERVICE_FN_TEXT "indirect_hosted_startup(p,&s))return false;\n");
#define START(m) FN_FIELD(b,"s.",start,m)
    START(revision);START(entry);START(initializer);START(functions);START(features);
    START(vm_value_slots);START(native_value_slots);START(frames);START(reference_slots);START(region_slots);
    START(input_bytes);START(allocation_bound);START(query_storage_peak);START(retained_bound);START(runtime_admitted);
#undef START
    fn_text(b,"if(s.input_bytes!=sizeof nf_module)return false;\n"
        "for(size_t n=0;n<sizeof nf_module;n++)if(!" SERVICE_FN_TEXT "indirect_hosted_bytes(p,n,&byte,1) || byte!=nf_module[n])return false;\n");
    for(uint32_t i=0;i<DISPATCH_TYPE_CAPACITY(p);i++) {
        SERVICE_TYPE(NominalLayout) t;
        if(!SERVICE_FN(indirect_hosted_type)(p,i,&t)) {
            if(!DISPATCH_TYPE_OPTIONAL(i)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
            fn_text(b,"if(" SERVICE_FN_TEXT "indirect_hosted_type(p,%u,&t))return false;\n",i);continue;
        }
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_type(p,%u,&t))return false;\n",i);
#define TYPE(m) FN_FIELD(b,"t.",t,m)
        TYPE(global_index);TYPE(catalog_ordinal);TYPE(source_ordinal);TYPE(layout_kind);TYPE(ownership_flags);TYPE(category);
#undef TYPE
    }
    for(uint32_t method=0;method<DISPATCH_IMPORT_COUNT(p);method++) {
        uint32_t original;
        if(!SERVICE_FN(indirect_hosted_import)(p,method,&original)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_import(p,%u,&ordinal) || ordinal!=%u)return false;\n",method,original);
    }
    for(uint32_t index=0;index<start.functions;index++) {
        SERVICE_TYPE(IndirectHostedFunction) f;uint32_t order;
        if(!SERVICE_FN(indirect_hosted_function)(p,index,&f) || !SERVICE_FN(indirect_hosted_function_order)(p,index,&order)) {
            b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;
        }
        fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_function_order(p,%u,&ordinal) || ordinal!=%u || !" SERVICE_FN_TEXT "indirect_hosted_function(p,%u,&f))return false;\n",index,order,index);
#define FUNC(m) FN_FIELD(b,"f.",f,m)
        FUNC(code.code_offset);FUNC(code.code_length);FUNC(code.instruction_count);
        FUNC(code.declaration.parameters);FUNC(code.declaration.locals);FUNC(code.declaration.result_count);
        fn_declaration(b,"f.code.declaration.result.",f.code.declaration.result);
        FUNC(declared_stack);FUNC(operand_peak);FUNC(locals);FUNC(staging_slots);FUNC(frames);
        FUNC(frame_owner_peak);FUNC(frame_reference_peak);FUNC(frame_region_peak);
        FUNC(reference_slots);FUNC(region_slots);FUNC(vm_value_slots);FUNC(native_value_slots);FUNC(entry_variant);
#undef FUNC
        for(uint16_t j=0;j<f.locals;j++) {
            SERVICE_TYPE(FlowDeclaration) l;
            if(!SERVICE_FN(indirect_hosted_local)(p,index,j,&l)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
            fn_text(b,"if(!" SERVICE_FN_TEXT "indirect_hosted_local(p,%u,%u,&l))return false;\n",index,j);
            fn_declaration(b,"l.",l);
        }
        for(uint16_t j=0;j<f.code.instruction_count;j++)
            fn_text(b,"if(!nf_agrees_%u_%u(p))return false;\n",index,j);
    }
    fn_text(b,"return true;}\n");
}

static void fn_preamble(Fne *b,FileIndirectNativeSurface surface) {
    if(surface==FNE_PUBLIC) {
        fn_text(b,"%s","#include <nanolang/" SERVICE_INSTALL_TEXT "/nanoisa/" SERVICE_PUBLIC_HEADER_TEXT ">\n#include <limits.h>\n#include <string.h>\n");
        fn_text(b,"%s","#if " SERVICE_CONST_TEXT "INDIRECT_PUBLIC_ABI != 1u || " SERVICE_CONST_TEXT "INDIRECT_NATIVE_ABI != 1u\n#error I require indirect public and native ABI revision 1\n#endif\n");
    } else {
    fn_text(b,"%s","#include \"src/nanoisa/" SERVICE_EMITTER_TEXT "indirect_private.h\"\n#include \"src/nanoisa/nvm_v2_sections.h\"\n#include <limits.h>\n#include <string.h>\n");
    fn_text(b,"%s","#ifndef " SERVICE_CONST_TEXT "INDIRECT_NATIVE_PRIVATE\n#error I require private indirect " SERVICE_NAME_TEXT " providers\n#endif\n#if " SERVICE_CONST_TEXT "INDIRECT_NATIVE_ABI != 1u\n#error I require indirect native ABI revision 1\n#endif\n");
    }
    fn_text(b,"%s","#define NF_TRY(expr) do { " SERVICE_TYPE_TEXT "RuntimeStatus nf_status_=(expr); if(nf_status_!=" SERVICE_CONST_TEXT "RUNTIME_OK)return nf_status_; } while(0)\n");
    fn_text(b,"%s","static " SERVICE_TYPE_TEXT "RuntimeStatus nf_bad(" SERVICE_TYPE_TEXT "Runtime *c){return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_STATE);}\n");
    fn_text(b,"%s","static int64_t nf_bits(uint64_t x){int64_t y;memcpy(&y,&x,sizeof y);return y;}\n");
    fn_text(b,"%s","static bool nf_scalar(const " SERVICE_TYPE_TEXT "RuntimeView *v,uint8_t tag){\n");
    fn_text(b,"%s"," return v->initialized && !v->owning && !v->formal && !v->type.mode &&\n");
    fn_text(b,"%s"," v->type.category==" SERVICE_CONST_TEXT "CATEGORY_UNKNOWN && v->type.tag==tag &&\n");
    fn_text(b,"%s"," v->type.global_index==NVM_V2_NO_INDEX && v->type.catalog_ordinal==NVM_V2_NO_INDEX &&\n");
    fn_text(b,"%s"," v->fields==(tag==TAG_VOID?0:1) && (tag!=TAG_BOOL || v->values[0]==0 || v->values[0]==1);\n");
    fn_text(b,"%s","}\n");
    fn_text(b,"%s","static " SERVICE_TYPE_TEXT "IndirectExecutionReport nf_refused(" SERVICE_TYPE_TEXT "RuntimeStatus status,const " SERVICE_TYPE_TEXT "IndirectOptions *options){\n"
        "" SERVICE_TYPE_TEXT "IndirectExecutionReport r={0};r.revision=1u;r.instruction_limit=options?options->instruction_limit:0;\n"
        "r.runtime.status=status;r.runtime.function=r.runtime.instruction=NVM_V2_NO_INDEX;return r;}\n");
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
    default:b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;
    }
    fn_text(b,"if(f.stack_count<%u)return nf_bad(c);\n",count);
    for(uint16_t k=0;k<count;k++)fn_text(b,
        "if(!" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-%u),&roots[%u]) || !" SERVICE_FN_TEXT "runtime_view(c,roots[%u],&values[%u]))return nf_bad(c);\n",count-k,k,k,k);
    fn_text(b,"tag=%u;\n",logic?TAG_BOOL:TAG_INT);
    if(op==OP_EQ || op==OP_NE)fn_text(b,"if(values[0].type.tag==TAG_BOOL)tag=TAG_BOOL;\n");
    for(uint16_t k=0;k<count;k++)fn_text(b,"if(!nf_scalar(&values[%u],tag))return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_TYPE);\n",k);
    fn_text(b,"a=values[0].values[0];z=%s;r=(%s);\n",unary?"0":"values[1].values[0]",expression);
    for(uint16_t k=0;k<count;k++)fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_drop(c,roots[%u]));\n",k);
    fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_scalar(c,roots[0],%u,r));\n",result);
}
static void fn_instruction(Fne *b,const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t function,uint16_t index,const SERVICE_TYPE(CodeInstruction) *in) {
    const DecodedInstruction *d=&in->decoded;
    uint8_t op=d->opcode;bool branch=false;
    switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_PUSH_I64:case OP_PUSH_BOOL:case OP_PUSH_VOID:
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_scalar(c,dst,%u,nf_bits(UINT64_C(%" PRIu64 "))));\n",
            op==OP_PUSH_I64?TAG_INT:op==OP_PUSH_BOOL?TAG_BOOL:TAG_VOID,
            op==OP_PUSH_I64?(uint64_t)d->operands[0].i64:op==OP_PUSH_BOOL?(uint64_t)d->operands[0].u8:UINT64_C(0));break;
    case OP_FUNCREF:
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_indirect_funcref(c,dst));\n");break;
    case OP_DUP:
        fn_text(b,"if(!f.stack_count || !" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) || !" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_copy(c,src,dst));\n");break;
    case OP_POP:case OP_FILE_DROP_STACK:
        fn_text(b,"if(!f.stack_count || !" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) || !" SERVICE_FN_TEXT "runtime_view(c,src,&value))return nf_bad(c);\n");
        if(op==OP_POP)fn_text(b,"if(value.owning || value.formal)return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_TYPE);\n");
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_drop(c,src));\n");break;
    case OP_LOAD_LOCAL:case OP_OWN_MOVE_LOCAL:
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_local(c,%u,&src) || !" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst) || !" SERVICE_FN_TEXT "runtime_view(c,src,&value))return nf_bad(c);\n"
            "if(!value.initialized || value.formal || value.owning!=%u)return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_TYPE);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_%s(c,src,dst));\n",d->operands[0].u16,op==OP_OWN_MOVE_LOCAL,op==OP_OWN_MOVE_LOCAL?"move":"copy");break;
    case OP_STORE_LOCAL:case OP_OWN_STORE_LOCAL:fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_store(c));\n");break;
    case OP_REGION_BEGIN:fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_region_begin(c));\n");break;
    case OP_REGION_END:fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_region_end(c));\n");break;
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE:fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_borrow(c));\n");break;
    case OP_FILE_END_BORROW:fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_end_reference(c));\n");break;
    case OP_FILE_DROP_LOCAL:
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_local(c,%u,&src))return nf_bad(c);\nNF_TRY(" SERVICE_FN_TEXT "runtime_drop(c,src));\n",d->operands[0].u16);break;
    case OP_FILE_RESULT_BRANCH:
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_local(c,%u,&src))return nf_bad(c);\nNF_TRY(" SERVICE_FN_TEXT "runtime_result_arm(c,src,&arm));\nedge=arm==" SERVICE_CONST_TEXT "FLOW_ARM_ERROR;\n",d->operands[0].u16);branch=true;break;
    case OP_FILE_RESULT_TAKE:
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_local(c,%u,&src) || !" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_take(c,src,%u,dst));\n",d->operands[0].u16,d->operands[1].u8?SERVICE_CONST(FLOW_ARM_ERROR):SERVICE_CONST(FLOW_ARM_OK));break;
    case OP_FILE_SERVICE: {
        uint32_t ordinal=in->catalog_ordinal,method=DISPATCH_METHOD_LOCAL(ordinal);bool consumes=method==1 || method==4 || (DISPATCH_ENDPOINT(p,ordinal) && method==0);
        fn_text(b,"src=reference=" SERVICE_CONST_TEXT "RUNTIME_NO_SLOT;\nif(!" SERVICE_FN_TEXT "runtime_frame_scratch(c,&dst))return nf_bad(c);\n");
        if(consumes)fn_text(b,"if(!f.stack_count || !" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src))return nf_bad(c);\n");
        if(method>=1 && method<=3)fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_reference(c,%u,&reference))return nf_bad(c);\n",d->operands[1].u16);
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_service(c,%u,reference,src,dst));\n"
            "if(!" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)(f.stack_count-%u),&output))return nf_bad(c);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_move(c,dst,output));\n",d->operands[0].u32,consumes?1u:0u);break;
    }
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        fn_text(b,"if(!f.stack_count || !" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src) || !" SERVICE_FN_TEXT "runtime_view(c,src,&value))return nf_bad(c);\n"
            "if(!nf_scalar(&value,TAG_BOOL))return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_TYPE);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_drop(c,src));\n");
        if(op==OP_ASSERT)fn_text(b,"if(!value.values[0])return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_ASSERT);\n");
        else {fn_text(b,"edge=(value.values[0]!=0)==%u;\n",op==OP_JMP_TRUE);branch=true;}
        break;
    case OP_AGG_PACK:case OP_UNION_CONSTRUCT: {
        uint16_t variant=d->operands[op==OP_AGG_PACK?2:1].u16,count=d->operands[op==OP_AGG_PACK?3:2].u16;
        if(count>SERVICE_CONST(RUNTIME_FIELDS)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        if(count)fn_text(b,"if(f.stack_count<%u)return nf_bad(c);\n",count);
        for(uint16_t k=0;k<count;k++)fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-%u),&inputs[%u]))return nf_bad(c);\n",count-k,k);
        if(count)fn_text(b,"dst=inputs[0];\n");else fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst))return nf_bad(c);\n");
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_construct(c,%u,%u,inputs,%u,dst));\n",in->catalog_ordinal,variant,count);break;
    }
    case OP_AGG_GET:case OP_UNION_FIELD:case OP_AGG_TAG:case OP_UNION_TAG:
        fn_text(b,"if(!f.stack_count || !" SERVICE_FN_TEXT "runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src))return nf_bad(c);\n");
        if(op==OP_AGG_GET || op==OP_UNION_FIELD)fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_project(c,src,%u,src));\n",d->operands[0].u16);
        else fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_view(c,src,&value))return nf_bad(c);\n"
            "if(value.owning || value.formal || value.type.category!=" SERVICE_CONST_TEXT "CATEGORY_SCALAR_RESULT)return " SERVICE_FN_TEXT "runtime_fail(c," SERVICE_CONST_TEXT "RUNTIME_TYPE);\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_result_arm(c,src,&arm));\nNF_TRY(" SERVICE_FN_TEXT "runtime_drop(c,src));\n"
            "NF_TRY(" SERVICE_FN_TEXT "runtime_scalar(c,src,TAG_INT,arm==" SERVICE_CONST_TEXT "FLOW_ARM_ERROR));\n");
        break;
    case OP_FILE_CALL_INDIRECT_REFS:case OP_CALL_INDIRECT: {
        uint8_t variants;uint64_t candidates=0;
        if(in->successor_count!=1 || !SERVICE_FN(indirect_hosted_variant_count)(p,function,index,&variants)) {
            b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;
        }
        for(uint8_t v=0;v<variants;v++) {
            SERVICE_TYPE(IndirectFlowCall) call;
            if(!SERVICE_FN(indirect_hosted_call)(p,function,index,v,&call) || call.candidates!=call.checked_candidates) {
                b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;
            }
            candidates|=call.candidates;
        }
        if(!candidates){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_call(c));\n"
            "if(!" SERVICE_FN_TEXT "runtime_indirect_frame_view(c,&current))return nf_bad(c);\n"
            "switch(current.frame.function){\n");
        for(uint32_t target=0;target<SERVICE_CONST(FLOW_FUNCTIONS);target++)if(candidates&(UINT64_C(1)<<target))
            fn_text(b,"case %u:NF_TRY(nf_function_%u(c));break;\n",target,target);
        fn_text(b,"default:return nf_bad(c);\n}\ngoto nf_label_%u;\n",in->successors[0]);return;
    }
    case OP_CALL:case OP_CALL_REF:case OP_FILE_CALL_REFS:
        if(in->successor_count!=1){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_call(c));\nNF_TRY(nf_function_%u(c));\ngoto nf_label_%u;\n",d->operands[0].u32,in->successors[0]);return;
    case OP_RET:fn_text(b,"return " SERVICE_FN_TEXT "runtime_frame_return(c);\n");return;
    default:fn_numeric(b,op);break;
    }
    (void)function;
    if(branch) {
        if(in->successor_count!=2){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_next(c,edge));\nif(edge)goto nf_label_%u;\ngoto nf_label_%u;\n",in->successors[1],in->successors[0]);
    } else {
        if(in->successor_count!=1){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"NF_TRY(" SERVICE_FN_TEXT "runtime_frame_next(c,0));\ngoto nf_label_%u;\n",in->successors[0]);
    }
}
static void fn_function(Fne *b,const SERVICE_TYPE(IndirectHostedPlan) *p,uint32_t index) {
    SERVICE_TYPE(IndirectHostedFunction) function;
    if(!SERVICE_FN(indirect_hosted_function)(p,index,&function)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
    fn_text(b,"static " SERVICE_TYPE_TEXT "RuntimeStatus nf_function_%u(" SERVICE_TYPE_TEXT "Runtime *c){\n"
        "" SERVICE_TYPE_TEXT "IndirectFrameView current;" SERVICE_TYPE_TEXT "CyclicVariant variant;" SERVICE_TYPE_TEXT "RuntimeFrameView f;\n"
        "" SERVICE_TYPE_TEXT "RuntimeView value,values[2];" SERVICE_TYPE_TEXT "FlowArm arm;\n"
        "uint32_t src=0,dst=0,output=0,reference=0,roots[2],inputs[" SERVICE_CONST_TEXT "RUNTIME_FIELDS];\n"
        "uint8_t edge=0,tag=0;int64_t a=0,z=0,r=0;\n"
        "(void)value;(void)values;(void)arm;(void)src;(void)dst;(void)output;(void)reference;\n"
        "(void)roots;(void)inputs;(void)edge;(void)tag;(void)a;(void)z;(void)r;\n",index);
    for(uint16_t i=0;i<function.code.instruction_count;i++)fn_text(b,"if(0)goto nf_label_%u;\n",i);
    fn_text(b,"goto nf_label_0;\n");
    for(uint16_t i=0;i<function.code.instruction_count;i++) {
        SERVICE_TYPE(CodeInstruction) in;uint8_t count;
        if(!SERVICE_FN(indirect_hosted_instruction)(p,index,i,&in) ||
           !SERVICE_FN(indirect_hosted_variant_count)(p,index,i,&count)){b->status=SERVICE_CONST(RUNTIME_UNRESOLVED);return;}
        fn_text(b,"nf_label_%u:\n",i);
        if(!count){fn_text(b,"return nf_bad(c);\n");continue;}
        fn_text(b,"if(!" SERVICE_FN_TEXT "runtime_indirect_frame_view(c,&current) || current.instruction_open || current.frame.mode!=" SERVICE_CONST_TEXT "RUNTIME_NATIVE || current.frame.function!=%u || current.frame.instruction!=%u || !" SERVICE_FN_TEXT "indirect_hosted_variant(" SERVICE_FN_TEXT "runtime_indirect_plan(c),%u,%u,current.variant,&variant) || !variant.body.reachable || variant.input.stack!=current.frame.stack_count)return nf_bad(c);\n"
            "f=current.frame;(void)f;NF_TRY(" SERVICE_FN_TEXT "runtime_indirect_enter(c));\n",index,i,index,i);
        fn_instruction(b,p,index,i,&in);
    }
    fn_text(b,"}\n");
}
static SERVICE_TYPE(RuntimeStatus) file_indirect_native_emit_serialized(const uint8_t *bytes,size_t size,
    FileIndirectNativeSurface surface,const char *identifier,char **out,char *err,size_t err_size) {
    if(surface!=FNE_PRIVATE && surface!=FNE_PUBLIC)return SERVICE_CONST(RUNTIME_INVALID);
    if(surface==FNE_PUBLIC && !SERVICE_PUBLIC_SUPPORTED)return SERVICE_CONST(RUNTIME_UNRESOLVED);
    if(surface==FNE_PUBLIC && !identifier)return SERVICE_CONST(RUNTIME_INVALID);
    if(!out || !bytes || !size || (err_size && !err))return SERVICE_CONST(RUNTIME_INVALID);
    SERVICE_TYPE(IndirectHostedPlan) *plan=NULL;
    SERVICE_TYPE(FlowStatus) prepared=SERVICE_FN(indirect_hosted_prepare)(bytes,size,&plan);
    SERVICE_TYPE(RuntimeStatus) result=SERVICE_CONST(RUNTIME_OK);
    if(prepared!=SERVICE_CONST(FLOW_OK)) {
        result=prepared==SERVICE_CONST(FLOW_MEMORY)?SERVICE_CONST(RUNTIME_MEMORY):prepared==SERVICE_CONST(FLOW_LIMIT)?SERVICE_CONST(RUNTIME_LIMIT):
            prepared==SERVICE_CONST(FLOW_UNRESOLVED)?SERVICE_CONST(RUNTIME_UNRESOLVED):SERVICE_CONST(RUNTIME_INVALID);
        if(err_size)(void)snprintf(err,err_size,"I refuse %sindirect " SERVICE_NAME_TEXT " preparation (%u)",surface==FNE_PRIVATE?"private ":"",(unsigned)result);
        return result;
    }
    SERVICE_TYPE(IndirectHostedStartup) startup;
    if(!SERVICE_FN(indirect_hosted_startup)(plan,&startup) || !fcd_coverage(plan)) {
        SERVICE_FN(indirect_hosted_free)(plan);
        if(err_size)(void)snprintf(err,err_size,"I lack complete %sindirect " SERVICE_NAME_TEXT " instruction coverage",surface==FNE_PRIVATE?"private ":"");
        return SERVICE_CONST(RUNTIME_UNRESOLVED);
    }
    Fne b={0};fn_preamble(&b,surface);
    fn_text(&b,"static const uint8_t nf_module[]={\n");
    for(size_t i=0;i<size;i++)fn_text(&b,"%u,%s",bytes[i],i%24==23?"\n":"");
    fn_text(&b,"\n};\n");fn_agreement(&b,plan,startup);
    for(uint32_t f=0;f<startup.functions;f++)fn_text(&b,"static " SERVICE_TYPE_TEXT "RuntimeStatus nf_function_%u(" SERVICE_TYPE_TEXT "Runtime *);\n",f);
    for(uint32_t f=0;f<startup.functions;f++)fn_function(&b,plan,f);
    fn_text(&b,"%s",surface==FNE_PUBLIC?
        "static " SERVICE_TYPE_TEXT "IndirectExecutionReport nf_execute(const " SERVICE_TYPE_TEXT "IndirectOptions *options," SERVICE_TYPE_TEXT "RuntimeView *out){\n":
        "" SERVICE_TYPE_TEXT "IndirectExecutionReport " SERVICE_FN_TEXT "native_indirect_execute(const " SERVICE_TYPE_TEXT "IndirectOptions *options," SERVICE_TYPE_TEXT "RuntimeView *out){\n");
    fn_text(&b,
        "if(!out || !options || options->revision!=" SERVICE_CONST_TEXT "INDIRECT_RUNTIME_REVISION || options->instruction_limit>" SERVICE_CONST_TEXT "INDIRECT_FUEL_MAX)return nf_refused(" SERVICE_CONST_TEXT "RUNTIME_INVALID,options);\n"
        "if(!" SERVICE_FN_TEXT "runtime_indirect_native_abi(1u,sizeof(" SERVICE_TYPE_TEXT "IndirectOptions),sizeof(" SERVICE_TYPE_TEXT "IndirectExecutionReport),sizeof(" SERVICE_TYPE_TEXT "IndirectFrameView),sizeof(" SERVICE_TYPE_TEXT "RuntimeFrameView),sizeof(" SERVICE_TYPE_TEXT "RuntimeView)))return nf_refused(" SERVICE_CONST_TEXT "RUNTIME_UNRESOLVED,options);\n"
        "" SERVICE_TYPE_TEXT "Runtime *c=NULL;uint32_t root;\n"
        "" SERVICE_TYPE_TEXT "RuntimeStatus status=" SERVICE_FN_TEXT "runtime_indirect_create(nf_module,sizeof nf_module," SERVICE_CONST_TEXT "RUNTIME_NATIVE,options,&c);\n"
        "if(status!=" SERVICE_CONST_TEXT "RUNTIME_OK)return nf_refused(status,options);\n"
        "if(!nf_agrees(" SERVICE_FN_TEXT "runtime_indirect_plan(c))){(void)" SERVICE_FN_TEXT "runtime_indirect_destroy(&c,NULL);return nf_refused(" SERVICE_CONST_TEXT "RUNTIME_UNRESOLVED,options);}\n"
        "status=" SERVICE_FN_TEXT "runtime_begin(c);\n(void)nf_bits;(void)nf_scalar;\n");
    for(uint32_t f=0;f<startup.functions;f++)fn_text(&b,"(void)nf_function_%u;\n",f);
    if(startup.initializer!=NVM_V2_NO_INDEX)fn_text(&b,
        "if(status==" SERVICE_CONST_TEXT "RUNTIME_OK)status=" SERVICE_FN_TEXT "runtime_frame_start(c);\n"
        "if(status==" SERVICE_CONST_TEXT "RUNTIME_OK)status=nf_function_%u(c);\n",startup.initializer);
    fn_text(&b,"if(status==" SERVICE_CONST_TEXT "RUNTIME_OK && (!" SERVICE_FN_TEXT "runtime_current_root(c,&root) || root!=%u))status=nf_bad(c);\n"
        "if(status==" SERVICE_CONST_TEXT "RUNTIME_OK)status=" SERVICE_FN_TEXT "runtime_frame_start(c);\n"
        "if(status==" SERVICE_CONST_TEXT "RUNTIME_OK)status=nf_function_%u(c);\n"
        "if(status==" SERVICE_CONST_TEXT "RUNTIME_OK && " SERVICE_FN_TEXT "runtime_current_root(c,&root))status=nf_bad(c);\n"
        "if(status!=" SERVICE_CONST_TEXT "RUNTIME_OK)(void)" SERVICE_FN_TEXT "runtime_fail(c,status);\n"
        "return " SERVICE_FN_TEXT "runtime_indirect_destroy(&c,out);\n}\n#undef NF_TRY\n",startup.entry,startup.entry);
    if(surface==FNE_PUBLIC)fn_text(&b,
        "" SERVICE_TYPE_TEXT "IndirectExecutionReport " SERVICE_FN_TEXT "indirect_program_%s(" SERVICE_TYPE_TEXT "HostGrant *grant,const " SERVICE_TYPE_TEXT "IndirectOptions *options," SERVICE_TYPE_TEXT "Scalar *out){\n"
        "" SERVICE_TYPE_TEXT "HostStatus entered=" SERVICE_FN_TEXT "host_enter(grant," SERVICE_CONST_TEXT "HOST_ABI," SERVICE_CONST_TEXT "HOST_CATALOG);\n"
        "if(entered!=" SERVICE_CONST_TEXT "HOST_OK)return " SERVICE_FN_TEXT "indirect_public_refused(" SERVICE_FN_TEXT "public_grant_status(entered),entered==" SERVICE_CONST_TEXT "HOST_BUSY?NULL:options);\n"
        "" SERVICE_TYPE_TEXT "IndirectExecutionReport report;\n"
        "if(!" SERVICE_FN_TEXT "indirect_public_options(options) || !out)report=" SERVICE_FN_TEXT "indirect_public_refused(" SERVICE_CONST_TEXT "RUNTIME_INVALID,options);\n"
        "else {" SERVICE_TYPE_TEXT "IndirectOptions copied=*options;" SERVICE_TYPE_TEXT "RuntimeView view={0};\n"
        "if(!" SERVICE_FN_TEXT "runtime_indirect_public_abi(1u,sizeof copied,sizeof report,sizeof *out))report=" SERVICE_FN_TEXT "indirect_public_refused(" SERVICE_CONST_TEXT "RUNTIME_UNRESOLVED,&copied);\n"
        "else {report=nf_execute(&copied,&view);report.runtime=" SERVICE_FN_TEXT "public_publish(report.runtime,&view,out);}}\n"
        "" SERVICE_FN_TEXT "host_leave();return report;}\n",identifier);
    SERVICE_FN(indirect_hosted_free)(plan);
    result=b.status;
    if(result!=SERVICE_CONST(RUNTIME_OK)) {
        free(b.text);
        if(err_size)(void)snprintf(err,err_size,"I refuse %sindirect " SERVICE_NAME_TEXT " emission (%u)",surface==FNE_PRIVATE?"private ":"",(unsigned)result);
        return result;
    }
    *out=b.text;if(err_size)err[0]='\0';return SERVICE_CONST(RUNTIME_OK);
}
#undef FN_FIELD
