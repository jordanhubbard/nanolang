/* I conservatively collect portable leaf-array facts; I do not admit execution. */
#include "managed_array_shapes.h"
#include "verifier.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define FUNCTIONS 256u
#define SLOTS 256u
#define ORIGINS 64u
#define INSTRUCTIONS 65536u
#define CELLS 1048576u
#define BIT(tag) ((uint16_t)(1u << (tag)))
#define LEAVES (BIT(TAG_VOID)|BIT(TAG_INT)|BIT(TAG_U8)|BIT(TAG_FLOAT)|BIT(TAG_BOOL)|BIT(TAG_STRING)|BIT(TAG_ENUM))
typedef struct { uint64_t origins; uint16_t tags; uint8_t unknown; } Value;
typedef struct {
    VmDecodedFunction decoded;
    Value *states, result;
    uint16_t *depths, stack, locals;
    uint32_t *queue, head, tail, queued_count, stride;
    uint8_t *seen, *queued;
    int16_t *origins;
} Function;
typedef struct {
    const NvmModule *module;
    Function functions[FUNCTIONS];
    Value globals[SLOTS];
    NvmArrayEligibilityReport report;
    NvmArrayEligibilityResult result;
    uint32_t global_count;
    int changed;
} Analysis;
#ifdef NMA_TESTING
static uint64_t allocation_budget = UINT64_MAX;
void nvm_array_analysis_fail_after(uint64_t n) { allocation_budget = n; }
#endif
static void *allocate(size_t n, size_t width) {
    if (width && n > SIZE_MAX / width) return NULL;
#ifdef NMA_TESTING
    if (!allocation_budget) return NULL;
    if (allocation_budget != UINT64_MAX) allocation_budget--;
#endif
    return calloc(n, width);
}
void nvm_array_eligibility_free(NvmArrayEligibilityReport *report) { free(report); }
static Value tag(uint8_t t) { Value v = {0, BIT(t), 0}; return v; }
static int merge(Value *to, Value from) {
    Value old = *to;
    to->tags |= from.tags; to->origins |= from.origins; to->unknown |= from.unknown;
    return old.tags != to->tags || old.origins != to->origins || old.unknown != to->unknown;
}
static int stop(Analysis *a, NvmArrayEligibilityStatus status, uint32_t f, uint32_t pc, const char *message) {
    a->result.status = status; a->result.function = f; a->result.pc = pc;
    snprintf(a->result.message, sizeof a->result.message, "%.*s", (int)sizeof a->result.message - 1, message);
    return 0;
}
static int verified(Analysis *a, NvmVerifyResult result) {
    if (result.ok) return 1;
    return stop(a, strstr(result.error_msg, "allocat") || strstr(result.error_msg, "memory") ?
                NVM_ARRAY_MEMORY : NVM_ARRAY_INVALID, 0, 0, result.error_msg);
}
static void enqueue(Function *f, uint32_t pc) {
    if (f->queued[pc]) return;
    uint32_t capacity = f->decoded.instruction_count + 1;
    f->queue[f->tail] = pc; f->tail = (f->tail + 1) % capacity;
    f->queued[pc] = 1; f->queued_count++;
}
static int join(Analysis *a, uint32_t fi, uint32_t pc, Value *state, uint16_t depth) {
    Function *f = &a->functions[fi];
    if (pc > f->decoded.instruction_count || depth > f->stack)
        return stop(a,NVM_ARRAY_INVALID,fi,pc,"I require bounded abstract stack successors.");
    int changed = !f->seen[pc];
    if (f->seen[pc] && f->depths[pc] != depth)
        return stop(a,NVM_ARRAY_INVALID,fi,pc,"I require equal stack heights at joins.");
    f->seen[pc] = 1; f->depths[pc] = depth;
    Value *target = f->states + (size_t)pc * f->stride;
    for (uint32_t i=0;i<(uint32_t)f->locals+depth;i++) changed |= merge(&target[i],state[i]);
    if (changed) { a->changed = 1; enqueue(f,pc); }
    return 1;
}
static int seed(Analysis *a,uint32_t fi,const Value *args) {
    Function *f=&a->functions[fi]; Value state[SLOTS*2]={{0}};
    for(uint16_t i=0;i<f->locals;i++) state[i]=tag(TAG_VOID);
    for(uint16_t i=0;i<a->module->functions[fi].arity;i++) state[i]=args[i];
    return join(a,fi,0,state,0);
}
/* An explicit producer list: unsupported transfers are unresolved, never guessed. */
static int fixed_result(uint8_t op) {
    switch(op) {
    case OP_PUSH_I64: case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL:
    case OP_F64_TO_BITS: case OP_I64_DIV_S: case OP_I64_REM_S: case OP_I64_NEG: case OP_CAST_INT:
    case OP_STR_LEN: case OP_STR_CHAR_AT: case OP_ARR_LEN: return TAG_INT;
    case OP_PUSH_U8: return TAG_U8;
    case OP_PUSH_F64: case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL:
    case OP_F64_FROM_BITS: case OP_F64_DIV: case OP_F64_NEG: case OP_CAST_FLOAT: return TAG_FLOAT;
    case OP_PUSH_BOOL: case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT: case OP_AND: case OP_OR: case OP_NOT:
    case OP_CAST_BOOL: case OP_TYPE_CHECK: case OP_STR_EQ: case OP_STR_CONTAINS:
    case OP_STR_STARTS_WITH: case OP_STR_ENDS_WITH: return TAG_BOOL;
    case OP_PUSH_STR: case OP_CAST_STRING: case OP_STR_CONCAT: case OP_STR_SUBSTR:
    case OP_STR_TRIM: case OP_STR_TO_LOWER: case OP_STR_TO_UPPER: case OP_STR_REPLACE:
    case OP_STR_FROM_INT: case OP_STR_FROM_FLOAT: return TAG_STRING;
    case OP_ENUM_VAL: return TAG_ENUM;
    case OP_PUSH_VOID: return TAG_VOID;
    default: return -1;
    }
}
static int supported(uint8_t op) {
    if(fixed_result(op)>=0)return 1;
    switch(op) {
    case OP_NOP: case OP_DUP: case OP_POP: case OP_SWAP:
    case OP_LOAD_LOCAL: case OP_STORE_LOCAL: case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL:
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: case OP_CALL: case OP_RET: case OP_ASSERT:
    case OP_ARR_LITERAL: case OP_ARR_SLICE: case OP_ARR_NEW: case OP_ARR_PUSH: case OP_ARR_SET: case OP_ARR_GET: case OP_ARR_POP: case OP_STR_SPLIT:
        return 1;
    default:return 0;
    }
}
static int packed(uint8_t t) { return t==TAG_INT || t==TAG_U8 || t==TAG_FLOAT || t==TAG_BOOL; }
static uint16_t writes(uint8_t t) {
    return BIT(t) | (t==TAG_INT?BIT(TAG_U8):t==TAG_U8 || t==TAG_FLOAT?BIT(TAG_INT):0);
}
static Value read_array(Analysis *a,Value receiver) {
    Value result=tag(TAG_VOID);
    result.unknown=receiver.unknown || ((receiver.tags&BIT(TAG_ARRAY)) && !receiver.origins);
    for(uint32_t i=0;i<a->report.origin_count;i++) if(receiver.origins&(UINT64_C(1)<<i)) {
        NvmArrayOrigin *o=&a->report.origins[i];
        result.tags |= o->packed?BIT(o->declared_tag):o->child_tags;
    }
    return result;
}
static void write_array(Analysis *a,Value receiver,Value value) {
    for(uint32_t i=0;i<a->report.origin_count;i++) if(receiver.origins&(UINT64_C(1)<<i)) {
        NvmArrayOrigin *o=&a->report.origins[i];
        if(!o->packed) {
            uint16_t old=o->child_tags;o->child_tags|=value.tags;
            a->changed |= old!=o->child_tags;
        }
    }
}
/* I keep copies distinct from sources, but weakly merge repeated copies at one site. */
static int slice_origins(Analysis *a,uint32_t fi,uint32_t pc,Value receiver,Value *result) {
    *result=tag(TAG_ARRAY);
    result->unknown=receiver.unknown || ((receiver.tags&BIT(TAG_ARRAY)) && !receiver.origins);
    for(uint32_t i=0;i<a->report.origin_count;i++)if(receiver.origins&(UINT64_C(1)<<i)) {
        NvmArrayOrigin source=a->report.origins[i];
        uint32_t target=0;
        while(target<a->report.origin_count) {
            NvmArrayOrigin *o=&a->report.origins[target];
            if(o->function==fi && o->pc==pc && o->declared_tag==source.declared_tag)break;
            target++;
        }
        if(target==a->report.origin_count) {
            if(target==ORIGINS)return stop(a,NVM_ARRAY_LIMIT,fi,pc,"I reached my derived array origin limit.");
            a->report.origin_count++;
            a->report.origins[target]=(NvmArrayOrigin){fi,pc,0,source.declared_tag,source.packed};
            a->changed=1;
        }
        NvmArrayOrigin *o=&a->report.origins[target];
        uint16_t old=o->child_tags;o->child_tags|=source.child_tags;
        a->changed|=old!=o->child_tags;
        result->origins|=UINT64_C(1)<<target;
    }
    return 1;
}
static int check_write(Analysis *a,uint32_t fi,uint32_t pc,Value receiver,Value value) {
    a->report.checked_writes++;
    if(value.unknown || !value.tags || (value.tags&~LEAVES))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require proved scalar/string leaf writes.");
    for(uint32_t i=0;i<a->report.origin_count;i++)if(receiver.origins&(UINT64_C(1)<<i)) {
        NvmArrayOrigin *o=&a->report.origins[i];
        if(o->packed && (value.tags&~writes(o->declared_tag)))
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I cannot prove a portable packed write for every possible tag.");
    }
    return 1;
}
static int walk(Analysis *a,uint32_t fi) {
    Function *f=&a->functions[fi];const NvmFunctionEntry *entry=&a->module->functions[fi];
    for(uint32_t i=0;i<=f->decoded.instruction_count;i++)if(f->seen[i])enqueue(f,i);
    while(f->queued_count) {
        uint32_t index=f->queue[f->head];f->head=(f->head+1)%(f->decoded.instruction_count+1);
        f->queued_count--;f->queued[index]=0;
        Value state[SLOTS*2]={{0}};uint16_t depth=f->depths[index];
        memcpy(state,f->states+(size_t)index*f->stride,((size_t)f->locals+depth)*sizeof(Value));
        Value *stack=state+f->locals;
        if(index==f->decoded.instruction_count) {
            if(entry->result_count)a->changed|=merge(&f->result,stack[depth-1]);
            continue;
        }
        VmDecodedInstruction *d=&f->decoded.instructions[index];DecodedInstruction *in=&d->instruction;
        uint8_t op=in->opcode;const InstructionInfo *info=isa_get_info(op);
        int pops=info->pop_count,pushes=info->push_count;
        if(op==OP_ARR_LITERAL){pops=in->operands[1].u16;pushes=1;}
        if(op==OP_CALL){pops=a->module->functions[in->operands[0].u32].arity;pushes=a->module->functions[in->operands[0].u32].result_count;}
        if(op==OP_RET){if(entry->result_count)a->changed|=merge(&f->result,stack[depth-1]);continue;}
        if(pops<0 || pushes<0 || depth<pops || depth-pops+pushes>f->stack)
            return stop(a,NVM_ARRAY_INVALID,fi,d->byte_offset,"I require a verified bounded instruction stack effect.");
        Value result={0};int exact=fixed_result(op);if(exact>=0)result=tag((uint8_t)exact);
        uint32_t base=depth-(uint16_t)pops;
        switch(op) {
        case OP_DUP: result=stack[depth-1];stack[depth]=result;depth++;goto successors;
        case OP_SWAP: result=stack[depth-1];stack[depth-1]=stack[depth-2];stack[depth-2]=result;goto successors;
        case OP_LOAD_LOCAL: result=state[in->operands[0].u16];break;
        case OP_STORE_LOCAL: state[in->operands[0].u16]=stack[base];break;
        case OP_LOAD_GLOBAL: result=a->globals[in->operands[0].u32];break;
        case OP_STORE_GLOBAL: a->changed|=merge(&a->globals[in->operands[0].u32],stack[base]);break;
        case OP_ARR_NEW: case OP_STR_SPLIT: case OP_ARR_LITERAL:
            result=tag(TAG_ARRAY);result.origins=UINT64_C(1)<<f->origins[index];
            if(op==OP_ARR_LITERAL)for(uint32_t i=base;i<depth;i++)write_array(a,result,stack[i]);
            break;
        case OP_ARR_SLICE:
            if(!slice_origins(a,fi,d->byte_offset,stack[base],&result))return 0;
            break;
        case OP_ARR_PUSH: case OP_ARR_SET:
            write_array(a,stack[base],stack[depth-1]);result=stack[base];break;
        case OP_ARR_GET: case OP_ARR_POP: result=read_array(a,stack[base]);break;
        case OP_CALL: {
            uint32_t callee=in->operands[0].u32;
            if(!seed(a,callee,stack+base))return 0;
            result=a->functions[callee].result;break;
        }
        default:break;
        }
        depth=(uint16_t)(base+pushes);if(pushes)stack[base]=result;
    successors:
        if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
            uint32_t relative=d->resolved_target-entry->code_offset;
            uint32_t target=relative==f->decoded.code_size?f->decoded.instruction_count:
                f->decoded.instruction_indices[relative]-1;
            if(!join(a,fi,target,state,depth))return 0;
            if(op==OP_JMP)continue;
        }
        if(!join(a,fi,index+1,state,depth))return 0;
    }
    return 1;
}
static int final_check(Analysis *a) {
    for(uint32_t i=0;i<a->global_count;i++)if(a->globals[i].unknown)
        return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I cannot resolve an escaped global value.");
    for(uint32_t fi=0;fi<a->module->function_count;fi++) {
        Function *f=&a->functions[fi];
        if(a->module->functions[fi].result_count && (f->result.unknown || !f->result.tags))
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,0,"I cannot resolve this function result or return path.");
        for(uint32_t pc=0;pc<f->decoded.instruction_count;pc++) {
            if(!f->seen[pc])continue;
            VmDecodedInstruction *d=&f->decoded.instructions[pc];uint8_t op=d->instruction.opcode;
            if(op==OP_ARR_LITERAL) {
                uint16_t count=d->instruction.operands[1].u16;
                Value receiver=tag(TAG_ARRAY);receiver.origins=UINT64_C(1)<<f->origins[pc];
                Value *stack=f->states+(size_t)pc*f->stride+f->locals;
                for(uint32_t i=f->depths[pc]-count;i<f->depths[pc];i++)
                    if(!check_write(a,fi,d->byte_offset,receiver,stack[i]))return 0;
                continue;
            }
            if(op!=OP_ARR_PUSH && op!=OP_ARR_SET && op!=OP_ARR_GET && op!=OP_ARR_POP && op!=OP_ARR_LEN && op!=OP_ARR_SLICE)continue;
            uint16_t depth=f->depths[pc];int pops=isa_get_info(op)->pop_count;
            Value *stack=f->states+(size_t)pc*f->stride+f->locals;
            Value receiver=stack[depth-pops];
            if(receiver.unknown || !receiver.tags || ((receiver.tags&BIT(TAG_ARRAY)) && !receiver.origins))
                return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I require authoritative array origins.");
            if(receiver.tags!=BIT(TAG_ARRAY))a->report.runtime_tag_checks++;
            if(op==OP_ARR_GET || op==OP_ARR_SET) {
                Value index=stack[depth-pops+1];
                if(index.unknown || !index.tags)return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I cannot resolve the array index tags.");
                if(index.tags!=BIT(TAG_INT))a->report.runtime_tag_checks++;
            }
            if(op!=OP_ARR_PUSH && op!=OP_ARR_SET)continue;
            if(!check_write(a,fi,d->byte_offset,receiver,stack[depth-1]))return 0;
        }
    }
    return 1;
}
static void destroy(Analysis *a) {
    for(uint32_t i=0;i<FUNCTIONS;i++) {
        Function *f=&a->functions[i];vm_decoded_function_free(&f->decoded);
        free(f->states);free(f->depths);free(f->queue);free(f->seen);free(f->queued);free(f->origins);
    }
    free(a);
}
NvmArrayEligibilityResult nvm_analyze_managed_arrays(const NvmModule *m,NvmArrayEligibilityReport **out) {
    NvmArrayEligibilityResult early={NVM_ARRAY_INVALID,0,0,"I require a module and report output."};
    if(!m || !out)return early;
    Analysis *a=allocate(1,sizeof *a);
    if(!a){early.status=NVM_ARRAY_MEMORY;snprintf(early.message,sizeof early.message,"I could not allocate array analysis state.");return early;}
    a->module=m;
    if(!verified(a,nvm_verify(m)))goto done;
    if(m->function_count>FUNCTIONS){stop(a,NVM_ARRAY_LIMIT,0,0,"I reached my array analysis function limit.");goto done;}
    if(!(m->header.flags&NVM_FLAG_HAS_MAIN) || m->functions[m->header.entry_point].arity ||
       m->import_count || m->module_ref_count || m->struct_count || m->union_count || m->ownership_size || m->passive_size || m->layout_size) {
        stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require a closed zero-argument entry without nominal, ownership or host contracts.");goto done;
    }
    uint64_t cells=0;uint32_t instructions=0;int initializer=-1;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        Function *f=&a->functions[fi];const NvmFunctionEntry *e=&m->functions[fi];
        if(e->local_count>SLOTS){stop(a,NVM_ARRAY_LIMIT,fi,0,"I reached my array analysis local limit.");goto done;}
        if(e->upvalue_count || e->result_count>1){stop(a,NVM_ARRAY_UNRESOLVED,fi,0,"I require no captures and at most one result.");goto done;}
        const char *name=nvm_get_string(m,e->name_idx);
        if(initializer<0 && name && !strcmp(name,"__init__")) {
            if(e->arity){stop(a,NVM_ARRAY_UNRESOLVED,fi,0,"I require a zero-argument initializer.");goto done;}initializer=(int)fi;
        }
        for(uint32_t pc=0;pc<e->code_length;) {
            DecodedInstruction in={0};uint32_t width=isa_decode(m->code+e->code_offset+pc,e->code_length-pc,&in);
            if(!width){stop(a,NVM_ARRAY_INVALID,fi,pc,"I require decodable instructions.");goto done;}
            if(++instructions>INSTRUCTIONS){stop(a,NVM_ARRAY_LIMIT,fi,pc,"I reached my decoded instruction limit.");goto done;}
            if(!supported(in.opcode)){stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I have no proved transfer for this instruction.");goto done;}
            if(in.opcode==OP_LOAD_GLOBAL || in.opcode==OP_STORE_GLOBAL) {
                uint32_t global=in.operands[0].u32;
                if(global>=SLOTS){stop(a,NVM_ARRAY_LIMIT,fi,pc,"I reached my array analysis global limit.");goto done;}
                if(global>=a->global_count)a->global_count=global+1;
            }
            pc+=width;
        }
        if(!verified(a,nvm_verify_function_max_stack(m,fi,&f->stack)))goto done;
        if(f->stack>SLOTS){stop(a,NVM_ARRAY_LIMIT,fi,0,"I reached my array analysis stack limit.");goto done;}
        char error[VM_DECODE_ERROR_SIZE];
        if(!vm_decode_function(m,fi,&f->decoded,error)){stop(a,NVM_ARRAY_MEMORY,fi,0,error);goto done;}
        f->locals=e->local_count;f->stride=(uint32_t)f->locals+f->stack;if(!f->stride)f->stride=1;
        uint32_t count=f->decoded.instruction_count+1;
        cells+=(uint64_t)count*f->stride;
        if(cells>CELLS){stop(a,NVM_ARRAY_LIMIT,fi,0,"I reached my stored abstract-state cell limit.");goto done;}
        f->states=allocate((size_t)count*f->stride,sizeof(Value));f->depths=allocate(count,sizeof(uint16_t));
        f->queue=allocate(count,sizeof(uint32_t));f->seen=allocate(count,1);f->queued=allocate(count,1);f->origins=allocate(count,sizeof(int16_t));
        if(!f->states || !f->depths || !f->queue || !f->seen || !f->queued || !f->origins){stop(a,NVM_ARRAY_MEMORY,fi,0,"I could not allocate my bounded function analysis.");goto done;}
        for(uint32_t i=0;i<f->decoded.instruction_count;i++) {
            VmDecodedInstruction *d=&f->decoded.instructions[i];uint8_t op=d->instruction.opcode;
            f->origins[i]=-1;
            if(op!=OP_ARR_NEW && op!=OP_STR_SPLIT && op!=OP_ARR_LITERAL)continue;
            uint8_t declared=op==OP_STR_SPLIT?TAG_STRING:d->instruction.operands[0].u8;
            if(!(LEAVES&BIT(declared))){stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I have not qualified this declared array shape.");goto done;}
            if(a->report.origin_count==ORIGINS){stop(a,NVM_ARRAY_LIMIT,fi,d->byte_offset,"I reached my allocation-site origin limit.");goto done;}
            uint32_t origin=a->report.origin_count++;f->origins[i]=(int16_t)origin;
            a->report.origins[origin]=(NvmArrayOrigin){fi,d->byte_offset,op==OP_STR_SPLIT?BIT(TAG_STRING):0,declared,(uint8_t)packed(declared)};
        }
    }
    for(uint32_t i=0;i<a->global_count;i++)a->globals[i]=tag(TAG_VOID);
    if(initializer>=0 && !seed(a,(uint32_t)initializer,NULL))goto done;
    if(!seed(a,m->header.entry_point,NULL))goto done;
    int seeded_unused=0;
    do {
        a->changed=0;
        for(uint32_t fi=0;fi<m->function_count;fi++)if(!walk(a,fi))goto done;
        if(!a->changed && !seeded_unused) {
            seeded_unused=1;
            for(uint32_t fi=0;fi<m->function_count;fi++)if(!a->functions[fi].seen[0]) {
                Value args[SLOTS]={{0}};
                for(uint16_t i=0;i<m->functions[fi].arity;i++)args[i].unknown=1;
                if(!seed(a,fi,args))goto done;
            }
        }
    } while(a->changed);
    if(!final_check(a))goto done;
    NvmArrayEligibilityReport *report=allocate(1,sizeof *report);
    if(!report){stop(a,NVM_ARRAY_MEMORY,0,0,"I could not publish my array analysis report.");goto done;}
    *report=a->report;*out=report;a->result.status=NVM_ARRAY_ELIGIBLE;
    snprintf(a->result.message,sizeof a->result.message,"I established only the bounded portable array-shape obligations.");
 done:;
    NvmArrayEligibilityResult result=a->result;destroy(a);return result;
}
