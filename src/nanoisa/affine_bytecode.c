#include "affine_bytecode.h"
#include "affine_state.h"
#include "../nanovm/vm_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t tag;
    bool observation;
    uint16_t root;
    bool owned;
    uint32_t layout;
    uint16_t variant;
} Value;
typedef struct {
    NvmAffineState *locals;
    Value *stack;
    uint16_t count;
    bool visited;
} Frame;
static void frame_free(Frame *frame) {
    if (!frame) return;
    nvm_affine_state_free(frame->locals);free(frame->stack);free(frame);
}
static Frame *frame_clone(const Frame *from) {
    Frame *out=calloc(1,sizeof(*out));
    if (!out) return NULL;
    out->locals=nvm_affine_state_clone(from->locals);
    if (!out->locals) {frame_free(out);return NULL;}
    out->count=from->count;
    if (out->count) {
        out->stack=malloc(out->count*sizeof(*out->stack));
        if (!out->stack) {frame_free(out);return NULL;}
        memcpy(out->stack,from->stack,out->count*sizeof(*out->stack));
    }
    return out;
}
static bool stack_meet(Frame *destination,const Frame *incoming,bool *changed) {
    if (!changed) return false;
    *changed=false;
    if (destination->count!=incoming->count) return false;
    for (uint16_t i=0;i<destination->count;i++) {
        Value x=destination->stack[i],y=incoming->stack[i];
        if (x.tag!=y.tag || x.observation!=y.observation || x.owned!=y.owned ||
            (x.owned && x.layout!=y.layout) ||
            (x.observation && x.root!=y.root)) return false;
        if (x.tag==TAG_UNION) {
            if (x.layout!=y.layout) return false;
            if (x.variant!=y.variant && x.variant!=NVM_AFFINE_UNKNOWN_VARIANT) {
                destination->stack[i].variant=NVM_AFFINE_UNKNOWN_VARIANT;*changed=true;
            }
            if (x.root!=y.root && x.root!=UINT16_MAX) {
                destination->stack[i].root=UINT16_MAX;*changed=true;
            }
        }
    }
    return true;
}
static bool scalar(uint8_t tag) {
    return tag==TAG_INT || tag==TAG_U8 || tag==TAG_BOOL || tag==TAG_FLOAT;
}
static bool union_fact_for_layout(const NvmModule *module,uint32_t layout,
                                  uint16_t variant,NvmUnionVariantFact *out) {
    if (!module || !out) return false;
    for (uint32_t ordinal=0;ordinal<module->union_count;ordinal++) {
        NvmUnionVariantFact fact;
        if (nvm_ownership_union_variant(module,ordinal,variant,&fact)!=NVM_V2_OK)
            continue;
        if (fact.layout==layout) {*out=fact;return true;}
    }
    return false;
}
static bool push(Frame *f,Value value) {
    if (f->count==NVM_AFFINE_MAX_STACK) return false;
    Value *next=realloc(f->stack,(f->count+1)*sizeof(*next));
    if (!next) return false;
    f->stack=next;f->stack[f->count++]=value;return true;
}
static bool pop_scalar(Frame *f,uint8_t tag) {
    if (!f->count || (f->stack[f->count-1].observation || f->stack[f->count-1].owned) ||
        f->stack[f->count-1].tag!=tag) return false;
    f->count--;return true;
}
typedef struct {
    bool value_graph;
    uint8_t status[NVM_OWNED_MAX_FUNCTIONS];
    NvmAffineAnalysis results[NVM_OWNED_MAX_FUNCTIONS];
} AnalysisCalls;

bool nvm_affine_value_call_graph(const NvmModule *m) {
    if (!m || !m->functions || !m->function_count ||
        m->function_count>NVM_OWNED_MAX_FUNCTIONS || m->header.entry_point!=0) return false;
    bool edges[NVM_OWNED_MAX_FUNCTIONS][NVM_OWNED_MAX_FUNCTIONS]={{false}};
    for (uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        if ((!f && (fn->arity || fn->result_count!=1 ||
            (fn->result_tag!=TAG_INT && fn->result_tag!=TAG_BOOL && fn->result_tag!=TAG_U8))) || fn->upvalue_count ||
            fn->local_count>NVM_AFFINE_MAX_LOCALS ||
            fn->code_length>NVM_AFFINE_MAX_INSTRUCTIONS*ISA_MAX_INSTRUCTION_SIZE) return false;
        NvmAffineState *state=nvm_affine_state_create(m,f,fn->local_count);
        NvmAffineType parameters[NVM_AFFINE_MAX_PARAMETERS],result;uint16_t count=0,fields=0;
        bool valid=nvm_affine_value_parameters(state,parameters,NVM_AFFINE_MAX_PARAMETERS,&count) && count==fn->arity &&
            nvm_affine_value_result(state,&result,&fields);
        nvm_affine_state_free(state);
        if (!valid) return false;
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
        if (!vm_decode_function(m,f,&decoded,error)) return false;
        if (!decoded.instruction_count || decoded.instruction_count>NVM_AFFINE_MAX_INSTRUCTIONS) valid=false;
        for (uint32_t i=0;i<decoded.instruction_count && valid;i++) {
            const DecodedInstruction *in=&decoded.instructions[i].instruction;
            if (in->opcode==OP_CALL) {
                uint32_t target=in->operands[0].u32;
                if (!target || target>=m->function_count) valid=false;
                else edges[f][target]=true;
            } else if (in->opcode==OP_CALL_REF || in->opcode==OP_TAIL_CALL ||
                       in->opcode==OP_CALL_INDIRECT || in->opcode==OP_CALL_MODULE ||
                       in->opcode==OP_CALL_EXTERN) valid=false;
        }
        vm_decoded_function_free(&decoded);
        if (!valid) return false;
    }
    for (uint32_t k=0;k<m->function_count;k++)
        for (uint32_t i=0;i<m->function_count;i++)
            for (uint32_t j=0;j<m->function_count;j++)
                edges[i][j]=edges[i][j] || (edges[i][k] && edges[k][j]);
    for (uint32_t i=0;i<m->function_count;i++) if (edges[i][i]) return false;
    return true;
}
static NvmAffineAnalysis analyze(const NvmModule *m,uint32_t function,
                                   const NvmAffineState *caller,uint32_t reference,AnalysisCalls *calls);
static bool supported(uint8_t op,bool value_graph) {
    switch(op) {
    case OP_HALT: case OP_CALL: case OP_CALL_REF: case OP_MATCH_TAG:
    case OP_BORROW_PATH_SHARED: case OP_BORROW_PATH_EXCLUSIVE:
    case OP_REBORROW_SHARED: case OP_REBORROW_EXCLUSIVE:
    case OP_REGION_BEGIN: case OP_REGION_END:
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE: case OP_REF_GET: case OP_REF_SET:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL: case OP_OWN_PACK: case OP_OWN_UNPACK_LOCAL: case OP_OWN_UNPACK_VARIANT:
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_U8: case OP_PUSH_F64: case OP_PUSH_BOOL:
    case OP_DUP: case OP_POP: case OP_SWAP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_AGG_PACK: case OP_AGG_GET: case OP_AGG_TAG: case OP_STRUCT_GET:
    case OP_ADD: case OP_SUB: case OP_MUL:
    case OP_DIV: case OP_MOD: case OP_NEG: case OP_EQ: case OP_NE: case OP_LT:
    case OP_LE: case OP_GT: case OP_GE: case OP_AND: case OP_OR: case OP_NOT:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: case OP_RET: case OP_ASSERT:
        return true;
    case OP_PUSH_STR: case OP_PRINT: case OP_PRINTLN:
        return value_graph;
    default:return false;
    }
}
static bool observed_local(const Frame *f,uint16_t local) {
    for (uint16_t i=0;i<f->count;i++)
        if (f->stack[i].observation && f->stack[i].root==local) return true;
    return false;
}
static const char *step(Frame *f,const DecodedInstruction *in,uint16_t locals,const NvmModule *module,uint32_t function,AnalysisCalls *calls) {
    uint8_t op=in->opcode,tag=TAG_VOID,mode;
    uint16_t local;
    switch(op) {
    case OP_NOP: case OP_JMP: case OP_HALT: return NULL;
    case OP_CALL: {
        uint32_t target=in->operands[0].u32;
        if (!calls->value_graph || !target || target>=module->function_count)
            return "I require a checked acyclic owned value call";
        NvmAffineState *callee=nvm_affine_state_create(module,target,module->functions[target].local_count);
        NvmAffineType parameters[NVM_AFFINE_MAX_PARAMETERS],result;uint16_t count=0,fields=0;
        bool valid=nvm_affine_value_parameters(callee,parameters,NVM_AFFINE_MAX_PARAMETERS,&count) &&
            nvm_affine_value_result(callee,&result,&fields);
        nvm_affine_state_free(callee);
        if (!valid || count!=module->functions[target].arity || f->count<count)
            return "I require a complete consuming argument list";
        for (uint16_t p=0;p<count;p++) {
            Value argument=f->stack[f->count-count+p];
            NvmAffineType parameter=parameters[p];
            if (argument.observation || argument.tag!=parameter.tag ||
                argument.owned!=nvm_affine_type_requires_move(f->locals,parameter) ||
                ((argument.owned || argument.tag==TAG_UNION) &&
                 argument.layout!=parameter.layout))
                return "I require exact positional consuming argument types";
        }
        NvmAffineAnalysis call=analyze(module,target,NULL,0,calls);
        if (!call.ok) return "I require complete consuming-helper owner resolution";
        f->count-=count;
        if (result.tag==TAG_VOID) return NULL;
        return push(f,(Value){.tag=result.tag,.root=UINT16_MAX,
                              .owned=nvm_affine_type_requires_move(f->locals,result),.layout=result.layout,
                              .variant=NVM_AFFINE_UNKNOWN_VARIANT})
            ? NULL:"I cannot retain a checked value-call result";
    }
    case OP_CALL_REF: {
        if (function!=0 || module->function_count!=2 || in->operands[0].u32!=1)
            return "I require entry-to-helper reference calls without recursion";
        NvmAffineAnalysis call=analyze(module,1,f->locals,in->operands[1].u16,calls);
        if (!call.ok) return "I require checked caller authority and a non-escaping helper";
        tag=module->functions[1].result_tag;
        if (module->functions[1].result_count!=1 ||
            (tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_U8))
            return "I require a single scalar reference-call result";
        break;
    }
    case OP_PUSH_I64:tag=TAG_INT;break;
    case OP_PUSH_U8:tag=TAG_U8;break;
    case OP_PUSH_F64:tag=TAG_FLOAT;break;
    case OP_PUSH_BOOL:
        if (in->operands[0].u8>1) return "I require a Boolean literal";
        tag=TAG_BOOL;break;
    case OP_PUSH_STR: {
        if (!calls->value_graph)
            return "I require string literals inside an owned value-call graph";
        uint32_t index=in->operands[0].u32;
        if (index>=module->string_count || !module->strings || !module->string_lengths ||
            !module->strings[index] ||
            memchr(module->strings[index],'\0',module->string_lengths[index]))
            return "I require an instantiated string literal without embedded NUL";
        tag=TAG_STRING;break;
    }
    case OP_REGION_BEGIN:
        return nvm_affine_region_begin(f->locals)?NULL:"I cannot begin another reference region";
    case OP_REGION_END:
        return nvm_affine_region_end(f->locals)?NULL:"I need a live reference region to end";
    case OP_REBORROW_SHARED: case OP_REBORROW_EXCLUSIVE:
        return nvm_affine_reborrow(f->locals,in->operands[0].u16,in->operands[1].u16,
            op==OP_REBORROW_SHARED?NVM_REFERENCE_SHARED:NVM_REFERENCE_EXCLUSIVE)?NULL:
            "I require permitted parent authority and a fresh reborrow in a deeper region";
    case OP_BORROW_PATH_SHARED: case OP_BORROW_PATH_EXCLUSIVE:
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE: {
        bool exclusive=op==OP_BORROW_LOCAL_EXCLUSIVE || op==OP_BORROW_PATH_EXCLUSIVE;
        uint16_t fields[NVM_OWNERSHIP_MAX_PATH_DEPTH],count=0;
        if ((op==OP_BORROW_PATH_SHARED || op==OP_BORROW_PATH_EXCLUSIVE) &&
            nvm_ownership_path(module,in->operands[2].u32,fields,NVM_OWNERSHIP_MAX_PATH_DEPTH,&count)!=NVM_V2_OK)
            return "I require a retained bounded ownership path";
        /* An existing stack observation must not outlive an exclusive hold. */
        for (uint16_t i=0;i<f->count;i++)
            if (f->stack[i].observation && f->stack[i].root==in->operands[1].u16 &&
                exclusive)
                return "I cannot borrow exclusively while an owner observation is on my stack";
        return nvm_affine_borrow(f->locals,in->operands[0].u16,in->operands[1].u16,
            fields,count,exclusive?NVM_REFERENCE_EXCLUSIVE:NVM_REFERENCE_SHARED)?NULL:
            "I require an available scalar-leaf resource owner and reference slot in a live region";
    }
    case OP_REF_GET: case OP_REF_SET:
        if (!nvm_affine_reference_field(f->locals,in->operands[0].u16,
                in->operands[1].u16,op==OP_REF_SET,&tag))
            return "I require a live permitted scalar reference field";
        if (op==OP_REF_SET)
            return pop_scalar(f,tag)?NULL:"I require the exact scalar reference field type";
        break;
    case OP_OWN_MOVE_LOCAL: {
        if (observed_local(f,in->operands[0].u16)) return "I cannot move an observed owner";
        NvmAffineType type;
        if (!nvm_affine_take_local(f->locals,in->operands[0].u16,&type))
            return "I require a live unheld owner for an explicit move";
        return push(f,(Value){.tag=type.tag,.root=UINT16_MAX,.owned=true,
                              .layout=type.layout,.variant=NVM_AFFINE_UNKNOWN_VARIANT})?NULL:
            "I cannot extend my owned analysis stack";
    }
    case OP_OWN_STORE_LOCAL: {
        if (observed_local(f,in->operands[0].u16)) return "I cannot replace an observed owner";
        if (!f->count || !f->stack[f->count-1].owned)
            return "I require an owned token for an explicit store";
        Value value=f->stack[f->count-1];
        if (!nvm_affine_put_local(f->locals,in->operands[0].u16,(NvmAffineType){value.tag,value.layout}))
            return "I require an exact available owner destination";
        f->count--;return NULL;
    }
    case OP_OWN_PACK: {
        NvmAffineType fields[NVM_AFFINE_MAX_STACK];uint16_t count;
        uint32_t layout=in->operands[0].u32;
        if (!nvm_affine_record_fields(f->locals,layout,fields,NVM_AFFINE_MAX_STACK,&count) ||
            count>f->count) return "I require every declared field for owned construction";
        for (uint16_t i=0;i<count;i++) {
            Value value=f->stack[f->count-count+i];NvmAffineType field=fields[i];
            if (value.observation || value.tag!=field.tag ||
                (value.owned!=nvm_affine_type_requires_move(f->locals,field)) ||
                ((field.tag==TAG_STRUCT || field.tag==TAG_UNION) && value.layout!=field.layout))
                return "I require exact scalar or owned fields in declaration order";
        }
        f->count-=count;
        return push(f,(Value){.tag=TAG_STRUCT,.root=UINT16_MAX,.owned=true,
                              .layout=layout,.variant=NVM_AFFINE_UNKNOWN_VARIANT})?NULL:
            "I cannot extend my owned analysis stack";
    }
    case OP_OWN_UNPACK_LOCAL: {
        NvmAffineType type,fields[NVM_AFFINE_MAX_STACK];uint16_t count;
        uint16_t source=in->operands[0].u16;
        if (observed_local(f,source)) return "I cannot unpack an observed owner";
        if (!nvm_affine_local_type(f->locals,source,&type) ||
            !nvm_affine_record_fields(f->locals,type.layout,fields,NVM_AFFINE_MAX_STACK,&count) ||
            f->count+count>NVM_AFFINE_MAX_STACK || !nvm_affine_take_local(f->locals,source,&type))
            return "I require an intact owner and space for all unpacked fields";
        for (uint16_t i=0;i<count;i++) if (!push(f,(Value){.tag=fields[i].tag,
                                                    .root=UINT16_MAX,
                                                    .owned=nvm_affine_type_requires_move(f->locals,fields[i]),
                                                    .layout=fields[i].layout,
                                                    .variant=NVM_AFFINE_UNKNOWN_VARIANT}))
            return "I cannot extend my unpacked analysis stack";
        return NULL;
    }
    case OP_OWN_UNPACK_VARIANT: {
        uint16_t source=in->operands[0].u16,variant=in->operands[1].u16;
        uint16_t wanted=in->operands[2].u16,count=0;
        NvmAffineType type,fields[NVM_AFFINE_MAX_STACK];
        if (observed_local(f,source) || !nvm_affine_local_type(f->locals,source,&type) ||
            type.tag!=TAG_UNION || wanted>NVM_AFFINE_MAX_STACK-f->count ||
            !nvm_affine_union_fields(f->locals,type.layout,variant,fields,NVM_AFFINE_MAX_STACK,&count) ||
            count!=wanted || !nvm_affine_take_union_payload(f->locals,source,variant,
                fields,NVM_AFFINE_MAX_STACK,&count))
            return "I require an unobserved selected union and its exact payload count";
        for (uint16_t i=0;i<count;i++)
            if (!push(f,(Value){.tag=fields[i].tag,.root=UINT16_MAX,
                .owned=nvm_affine_type_requires_move(f->locals,fields[i]),.layout=fields[i].layout,
                .variant=NVM_AFFINE_UNKNOWN_VARIANT})) return "I cannot retain selected payload fields";
        return NULL;
    }
    case OP_AGG_PACK: {
        if (in->operands[0].u8!=AGG_VARIANT)
            return "I require explicit owned construction for records";
        NvmUnionVariantFact fact;
        uint16_t variant=in->operands[2].u16,count=in->operands[3].u16;
        if (nvm_ownership_union_variant(module,in->operands[1].u32,variant,&fact)!=NVM_V2_OK ||
            count!=fact.field_count || count>f->count)
            return "I require an exact retained union constructor shape";
        NvmAffineType fields[NVM_AFFINE_MAX_STACK];uint16_t available=0;
        if (!nvm_affine_union_fields(f->locals,fact.layout,variant,fields,
                                     NVM_AFFINE_MAX_STACK,&available) || available!=count)
            return "I require exact union payload facts";
        for (uint16_t i=0;i<count;i++) {
            Value value=f->stack[f->count-count+i];
            if (value.observation || value.tag!=fields[i].tag ||
                value.owned!=nvm_affine_type_requires_move(f->locals,fields[i]) ||
                ((fields[i].tag==TAG_STRUCT || fields[i].tag==TAG_UNION) && value.layout!=fields[i].layout))
                return "I require exact union fields in declaration order";
        }
        f->count-=count;
        return push(f,(Value){.tag=TAG_UNION,.root=UINT16_MAX,.layout=fact.layout,
                              .variant=variant,.owned=nvm_affine_type_requires_move(f->locals,(NvmAffineType){TAG_UNION,fact.layout})})?NULL:
            "I cannot retain a checked union value";
    }
    case OP_LOAD_LOCAL: {
        local=in->operands[0].u16;
        if (!nvm_affine_local_info(f->locals,local,&tag,&mode))
            return "I require a live checked local";
        uint32_t layout=NVM_V2_NO_INDEX;uint16_t variant=NVM_AFFINE_UNKNOWN_VARIANT;
        if (tag==TAG_UNION) {
            NvmAffineType type;
            if (!nvm_affine_local_type(f->locals,local,&type))
                return "I require an exact union local";
            layout=type.layout;
            (void)nvm_affine_union_variant(f->locals,local,&variant);
        }
        if (!push(f,(Value){.tag=tag,.observation=tag==TAG_STRUCT || nvm_affine_type_requires_move(f->locals,(NvmAffineType){tag,layout}),.root=local,
                            .layout=layout,.variant=variant}))
            return "I cannot extend my analysis stack";
        return NULL;
    }
    case OP_STORE_LOCAL: {
        local=in->operands[0].u16;
        if (local>=locals || !f->count || (f->stack[f->count-1].observation || f->stack[f->count-1].owned))
            return "I refuse an observation escape or missing scalar store";
        Value value=f->stack[f->count-1];
        /* Defining a scalar consults its exact declaration. The subsequent
         * lookup makes type disagreement a refusal, never a widening. */
        bool defined=scalar(value.tag) ? nvm_affine_scalar_define(f->locals,local) :
            calls->value_graph && value.tag==TAG_STRING ? nvm_affine_string_define(f->locals,local) :
            calls->value_graph && value.tag==TAG_UNION ?
                nvm_affine_union_define(f->locals,local,value.layout,value.variant) : false;
        if (!defined ||
            !nvm_affine_local_info(f->locals,local,&tag,&mode) || tag!=value.tag)
            return "I require the exact scalar local type";
        f->count--;return NULL;
    }
    case OP_MATCH_TAG: {
        if (!f->count || f->stack[f->count-1].tag!=TAG_UNION ||
            f->stack[f->count-1].owned)
            return "I require an exact union match source";
        NvmUnionVariantFact fact;
        if (!union_fact_for_layout(module,f->stack[f->count-1].layout,
                                   in->operands[0].u16,&fact))
            return "I require a match variant within the concrete union";
        /* The branch handler validates the variant against the value's layout. */
        return NULL;
    }
    case OP_AGG_TAG:
        if (!f->count || f->stack[f->count-1].tag!=TAG_UNION ||
            f->stack[f->count-1].owned)
            return "I require an exact union tag source";
        f->count--;tag=TAG_INT;break;
    case OP_AGG_GET:
        if (f->count && f->stack[f->count-1].tag==TAG_UNION &&
            !f->stack[f->count-1].owned) {
            Value value=f->stack[f->count-1];
            if (value.variant==NVM_AFFINE_UNKNOWN_VARIANT)
                return "I require a proven scalar-union variant before projection";
            NvmAffineType fields[NVM_AFFINE_MAX_STACK];uint16_t count=0;
            if (!nvm_affine_union_fields(f->locals,value.layout,value.variant,fields,
                                         NVM_AFFINE_MAX_STACK,&count) ||
                in->operands[0].u16>=count)
                return "I require an exact scalar-union payload field";
            NvmAffineType field=fields[in->operands[0].u16];
            if (!scalar(field.tag) && field.tag!=TAG_STRING)
                return "I require scalar or STRING union observations, not partial owner moves";
            tag=field.tag;f->count--;break;
        }
        /* Resource-record observations retain their established path. */
        /* fall through */
    case OP_STRUCT_GET:
        if (!f->count || !f->stack[f->count-1].observation || f->stack[f->count-1].owned ||
            !(nvm_affine_scalar_field(f->locals,f->stack[f->count-1].root,
                                     in->operands[0].u16,&tag) ||
              (calls->value_graph && nvm_affine_string_field(f->locals,f->stack[f->count-1].root,
                                     in->operands[0].u16,&tag))))
            return "I require a checked scalar field observation";
        f->count--;break;
    case OP_DUP:
        if (!f->count || (f->stack[f->count-1].observation || f->stack[f->count-1].owned))
            return "I refuse to duplicate reference authority";
        if (!calls->value_graph && f->stack[f->count-1].tag==TAG_STRING)
            return "I refuse unsupported string duplication";
        if (!push(f,f->stack[f->count-1])) return "I cannot extend my analysis stack";
        return NULL;
    case OP_POP:
        if (f->count && f->stack[f->count-1].tag==TAG_UNION &&
            f->stack[f->count-1].observation && !f->stack[f->count-1].owned) {
            f->count--;return NULL;
        }
        if (!f->count || (f->stack[f->count-1].observation || f->stack[f->count-1].owned))
            return "I require a scalar discard; an observation is not an owned consume";
        if (!calls->value_graph && f->stack[f->count-1].tag==TAG_STRING)
            return "I require PRINT or PRINTLN to consume a string";
        f->count--;return NULL;
    case OP_SWAP:
        if (f->count<2 || (f->stack[f->count-1].observation || f->stack[f->count-1].owned) || (f->stack[f->count-2].observation || f->stack[f->count-2].owned))
            return "I require scalar stack permutation";
        if (!calls->value_graph && (f->stack[f->count-1].tag==TAG_STRING || f->stack[f->count-2].tag==TAG_STRING))
            return "I refuse unsupported string permutation";
        {Value value=f->stack[f->count-1];f->stack[f->count-1]=f->stack[f->count-2];f->stack[f->count-2]=value;}
        return NULL;
    case OP_PRINT: case OP_PRINTLN:
        if (!calls->value_graph)
            return "I require string output inside an owned value-call graph";
        return pop_scalar(f,TAG_STRING)?NULL:
            "I require one exact immutable string print operand";
    case OP_ASSERT:
        return pop_scalar(f,TAG_BOOL)?NULL:"I require an exact Boolean assertion condition";
    case OP_JMP_TRUE: case OP_JMP_FALSE:
        return pop_scalar(f,TAG_BOOL)?NULL:"I require an exact Boolean branch condition";
    case OP_NOT: case OP_AND: case OP_OR:
        if (!pop_scalar(f,TAG_BOOL) || (op!=OP_NOT && !pop_scalar(f,TAG_BOOL)))
            return "I require Boolean operands";
        tag=TAG_BOOL;break;
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
        if (!pop_scalar(f,TAG_INT) || (op!=OP_NEG && !pop_scalar(f,TAG_INT)))
            return "I require integer arithmetic operands";
        tag=TAG_INT;break;
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
        if (!pop_scalar(f,TAG_FLOAT) || (op!=OP_F64_NEG && !pop_scalar(f,TAG_FLOAT)))
            return "I require float arithmetic operands";
        tag=TAG_FLOAT;break;
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
        if (!pop_scalar(f,TAG_FLOAT) || !pop_scalar(f,TAG_FLOAT))
            return "I require float comparison operands";
        tag=TAG_BOOL;break;
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
        if (calls->value_graph && (op==OP_EQ || op==OP_NE) && f->count &&
            f->stack[f->count-1].tag==TAG_STRING) {
            if (!pop_scalar(f,TAG_STRING) || !pop_scalar(f,TAG_STRING))
                return "I require exact STRING equality operands";
            tag=TAG_BOOL;break;
        }
        if (!f->count || (f->stack[f->count-1].observation || f->stack[f->count-1].owned) || !scalar(f->stack[f->count-1].tag))
            return "I require scalar comparison operands";
        tag=f->stack[f->count-1].tag;
        if (!pop_scalar(f,tag) || !pop_scalar(f,tag)) return "I require matching comparison tags";
        tag=TAG_BOOL;break;
    default:return "I require a connected affine instruction contract";
    }
    return push(f,(Value){.tag=tag,.root=UINT16_MAX,.layout=NVM_V2_NO_INDEX,
                          .variant=NVM_AFFINE_UNKNOWN_VARIANT})?NULL:
        "I cannot extend my analysis stack";
}
typedef struct {
    uint32_t *items;
    bool *queued;
    uint32_t capacity,head,tail,pending;
} Worklist;
static bool enqueue(Worklist *work,uint32_t index) {
    if (work->queued[index]) return true;
    if (work->pending==work->capacity) return false;
    work->items[work->tail]=index;
    work->tail=(work->tail+1)%work->capacity;work->pending++;
    work->queued[index]=true;return true;
}
static bool propagate(Frame **frames,uint32_t target,const Frame *state,Worklist *work,
                       const char **error) {
    bool changed=true;
    if (frames[target]) {
        bool stack_changed=false,locals_changed=false;
        if (!stack_meet(frames[target],state,&stack_changed) ||
            !nvm_affine_state_meet_initialization(frames[target]->locals,state->locals,&locals_changed)) {
            *error="I require exact ownership and stack provenance at every join";return false;
        }
        changed=stack_changed || locals_changed;
    } else {
        frames[target]=frame_clone(state);
        if (!frames[target]) {*error="I cannot allocate an analysis branch";return false;}
    }
    if (changed && !enqueue(work,target)) {
        *error="I exceeded my bounded affine worklist";return false;
    }
    return true;
}
static bool match_refine(Frame *frame,uint16_t variant) {
    if (!frame || !frame->count) return false;
    Value *value=&frame->stack[frame->count-1];
    if (value->tag!=TAG_UNION || value->owned ||
        (value->variant!=NVM_AFFINE_UNKNOWN_VARIANT && value->variant!=variant))
        return false;
    if (value->observation && !nvm_affine_union_refine(frame->locals,value->root,variant)) return false;
    value->variant=variant;
    /* A rooted owner observation cannot survive a move or replacement. Other
     * MATCH_TAG values prove only this tested value on this successful edge.  I do
     * not turn its source local into a wider fact: a later load is a different
     * receiver and must retain or establish its own exact proof. */
    return true;
}
#ifdef NVM_AFFINE_TEST_VISIT_LIMIT
extern uint32_t NVM_AFFINE_TEST_VISIT_LIMIT(uint32_t limit);
#endif
static NvmAffineAnalysis analyze(const NvmModule *m,uint32_t function,
                                   const NvmAffineState *caller,uint32_t reference,AnalysisCalls *calls) {
    NvmAffineAnalysis result={0};
    const char *error="I require checked ownership declarations";
    VmDecodedFunction decoded={0};Frame **frames=NULL;Worklist work={0};
    Frame *current=NULL;
    if (!m || !m->functions || function>=m->function_count) goto done;
    if (calls->value_graph && !caller) {
        if (calls->status[function]==2) return calls->results[function];
        if (calls->status[function]==1) {error="I refuse recursive owned value analysis";goto done;}
        calls->status[function]=1;
    }
    const NvmFunctionEntry *entry=&m->functions[function];
    if (entry->local_count>NVM_AFFINE_MAX_LOCALS ||
        entry->code_length>NVM_AFFINE_MAX_INSTRUCTIONS*ISA_MAX_INSTRUCTION_SIZE) {
        error="I exceeded my bounded affine analysis size";goto done;
    }
    NvmAffineState *initial=nvm_affine_state_create(m,function,entry->local_count);
    if (!initial) goto done;
    if (caller && !nvm_affine_bind_caller(initial,caller,reference)) {
        nvm_affine_state_free(initial);error="I require exact caller-origin parameter authority";goto done;
    }
    char detail[VM_DECODE_ERROR_SIZE];
    if (!vm_decode_function(m,function,&decoded,detail)) {
        nvm_affine_state_free(initial);error="I require decodable function control flow";goto done;
    }
    uint32_t count=decoded.instruction_count;
    if (!count || count>NVM_AFFINE_MAX_INSTRUCTIONS) {
        nvm_affine_state_free(initial);error="I exceeded my bounded affine instruction count";goto done;
    }
    for (uint32_t i=0;i<count;i++) if (!supported(decoded.instructions[i].instruction.opcode,
                                                  calls->value_graph)) {
        result.byte_offset=decoded.instructions[i].byte_offset;
        uint8_t op=decoded.instructions[i].instruction.opcode;
        nvm_affine_state_free(initial);
        error=op==OP_PUSH_STR
            ? "I require string literals inside an owned value-call graph"
            : (op==OP_PRINT || op==OP_PRINTLN)
                ? "I require string output inside an owned value-call graph"
                : "I require a connected affine instruction contract";
        goto done;
    }
    frames=calloc(count,sizeof(*frames));work.capacity=count;
    work.items=malloc(count*sizeof(*work.items));work.queued=calloc(count,sizeof(*work.queued));
    if (!frames || !work.items || !work.queued) {nvm_affine_state_free(initial);error="I cannot allocate analysis state";goto done;}
    frames[0]=calloc(1,sizeof(*frames[0]));
    if (!frames[0]) {nvm_affine_state_free(initial);error="I cannot allocate entry state";goto done;}
    frames[0]->locals=initial;
    /* Each instruction is first visited once, then only after one or more
     * of its at most local_count initialized scalar bits decrease. */
    uint32_t visit_limit=count*((uint32_t)entry->local_count+1u);
#ifdef NVM_AFFINE_TEST_VISIT_LIMIT
    visit_limit=NVM_AFFINE_TEST_VISIT_LIMIT(visit_limit);
#endif
    if (!enqueue(&work,0)) {error="I cannot queue my entry state";goto done;}
    while (work.pending) {
        if (result.visits>=visit_limit) {error="I exceeded my bounded affine visit count";goto done;}
        result.visits++;
        uint32_t index=work.items[work.head];
        work.head=(work.head+1)%work.capacity;work.pending--;work.queued[index]=false;
        const VmDecodedInstruction *instruction=&decoded.instructions[index];
        result.byte_offset=instruction->byte_offset;
        if (!frames[index]->visited) {frames[index]->visited=true;result.reachable++;}
        current=frame_clone(frames[index]);
        if (!current) {error="I cannot allocate an instruction state";goto done;}
        uint8_t op=instruction->instruction.opcode;
        bool fallthrough=true;
        if (op==OP_RET) {
            uint8_t tag=TAG_VOID;
            if (current->count==1 && !current->stack[0].observation) tag=current->stack[0].tag;
            else if (current->count) {error="I refuse an observation escape or extra return operands";goto done;}
            bool exact_value=current->count==1 &&
                (current->stack[0].owned || current->stack[0].tag==TAG_UNION);
            bool exit_ok=exact_value
                ? nvm_affine_can_exit_type(current->locals,(NvmAffineType){tag,current->stack[0].layout})
                : nvm_affine_can_exit_scalar(current->locals,tag);
            if (!exit_ok) {
                error="I require an exact declared result and no live owned obligations";goto done;
            }
        } else if (op==OP_HALT) {
            if (current->count) {error="I require an empty stack at my terminal invariant";goto done;}
        } else {
            error=step(current,&instruction->instruction,entry->local_count,m,function,calls);
            if (error) goto done;
            if (op==OP_MATCH_TAG) {
                uint32_t relative=instruction->resolved_target-entry->code_offset;
                const VmDecodedInstruction *target=vm_decoded_function_at(&decoded,relative);
                uint16_t wanted=instruction->instruction.operands[0].u16;
                uint16_t known=current->stack[current->count-1].variant;
                if (!target) {error="I require an instruction at my match target";goto done;}
                if (known==NVM_AFFINE_UNKNOWN_VARIANT || known==wanted) {
                    Frame *matched=frame_clone(current);
                    if (!matched || !match_refine(matched,wanted) ||
                        !propagate(frames,(uint32_t)(target-decoded.instructions),matched,&work,&error)) {
                        frame_free(matched);
                        if (!error) error="I require a checked scalar-union match edge";
                        goto done;
                    }
                    frame_free(matched);
                }
                /* A constructed or previously refined exact variant makes one
                 * edge unreachable.  I do not reject the dead source-order arm. */
                if (known==wanted) fallthrough=false;
            }
            if (op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
                uint32_t relative=instruction->resolved_target-entry->code_offset;
                const VmDecodedInstruction *target=vm_decoded_function_at(&decoded,relative);
                if (!target) {error="I require an instruction at my branch target";goto done;}
                if (!propagate(frames,(uint32_t)(target-decoded.instructions),current,&work,&error)) goto done;
            }
            if (op!=OP_JMP && fallthrough) {
                if (index+1==count) {error="I require an explicit return rather than fallthrough";goto done;}
                if (!propagate(frames,index+1,current,&work,&error)) goto done;
            }
        }
        frame_free(current);current=NULL;
    }
    result.ok=true;result.byte_offset=0;error=NULL;
done:
    frame_free(current);
    if (frames) for (uint32_t i=0;i<decoded.instruction_count;i++) frame_free(frames[i]);
    free(frames);free(work.items);free(work.queued);vm_decoded_function_free(&decoded);
    if (error) snprintf(result.message,sizeof(result.message),"%s",error);
    if (calls->value_graph && !caller && function<NVM_OWNED_MAX_FUNCTIONS) {
        calls->results[function]=result;calls->status[function]=2;
    }
    return result;
}

NvmAffineAnalysis nvm_affine_analyze_function(const NvmModule *m,uint32_t function) {
    AnalysisCalls calls={0};calls.value_graph=nvm_affine_value_call_graph(m);
    return analyze(m,function,NULL,0,&calls);
}
