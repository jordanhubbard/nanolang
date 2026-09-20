/* I conservatively collect portable leaf-array facts; I do not admit execution. */
#include "managed_array_shapes.h"
#include "managed_record_shapes.h"
#include "managed_record_array_origins.h"
#include "record_array_structure_private.h"
#include "managed_record_plan.h"
#include "ownership_contracts.h"
#include "verifier.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define FUNCTIONS 256u
#define SLOTS 256u
#define ORIGINS 64u
#define INSTRUCTIONS 65536u
#define CELLS 1048576u
#define FIELD_CELLS 65536u
#define BIT(tag) ((uint16_t)(1u << (tag)))
#define LEAVES (BIT(TAG_VOID)|BIT(TAG_INT)|BIT(TAG_U8)|BIT(TAG_FLOAT)|BIT(TAG_BOOL)|BIT(TAG_STRING)|BIT(TAG_ENUM))
#define RECORD_ARRAY_LEAVES (BIT(TAG_INT)|BIT(TAG_U8)|BIT(TAG_FLOAT)|BIT(TAG_BOOL)|BIT(TAG_STRING))
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
    Value children[ORIGINS];
    int graph, records;
    NvmRecordPlan *plan;
    Value *fields;
    uint32_t field_count, checked_field_writes;
    uint64_t state_cells;
    uint8_t origin_kind[ORIGINS];
    uint32_t record_ordinal[ORIGINS], field_start[ORIGINS];
    uint16_t record_fields[ORIGINS];
    NvmArrayEligibilityReport report;
    NvmArrayEligibilityResult result;
    uint32_t global_count;
    int changed;
    NvmRecordArrayStructure *structure;
    NvmRecordArrayBudget budget;
    uint32_t layout_fields[256];
    uint8_t *field_elements;
    uint16_t required_elements[ORIGINS];
    Value all_writes[ORIGINS];
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
static void *analysis_alloc(Analysis *a,size_t n,size_t width) {
    if(width && n>SIZE_MAX/width){if(a->structure)a->budget.limited=true;return NULL;}
    if(a->structure && (!nvm_ra_bytes(&a->budget,(uint64_t)n*width) || !nvm_ra_steps(&a->budget,n)))return NULL;
    void *p=allocate(n,width);if(!p && a->structure)a->budget.memory=true;return p;
}
void nvm_record_eligibility_free(NvmRecordEligibilityReport *report) {
    if (report) { free(report->fields); free(report); }
}
void nvm_array_eligibility_free(NvmArrayEligibilityReport *report) { free(report); }
void nvm_array_graph_eligibility_free(NvmArrayGraphEligibilityReport *report) { free(report); }
static Value tag(uint8_t t) { Value v = {0, BIT(t), 0}; return v; }
static int merge(Value *to, Value from) {
    Value old = *to;
    to->tags |= from.tags; to->origins |= from.origins; to->unknown |= from.unknown;
    return old.tags != to->tags || old.origins != to->origins || old.unknown != to->unknown;
}
static int stop(Analysis *a, NvmArrayEligibilityStatus status, uint32_t f, uint32_t pc, const char *message) {
    if(a->structure && a->budget.limited)status=NVM_ARRAY_LIMIT;
    a->result.status = status; a->result.function = f; a->result.pc = pc;
    snprintf(a->result.message, sizeof a->result.message, "%.*s", (int)sizeof a->result.message - 1, message);
    return 0;
}
static int analysis_work(Analysis *a,uint64_t n) {
    if(!a->structure || nvm_ra_steps(&a->budget,n))return 1;
    return stop(a,NVM_ARRAY_LIMIT,0,0,"I reached my record-array work bound.");
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
    if(!analysis_work(a,(uint64_t)f->locals+depth+1))return 0;
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
static int record_operation(uint8_t op) {
    return op == OP_STRUCT_NEW || op == OP_STRUCT_LITERAL || op == OP_STRUCT_GET ||
           op == OP_STRUCT_SET || op == OP_AGG_PACK || op == OP_AGG_GET || op == OP_AGG_SET;
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
bool nvm_record_array_opcode_supported(uint8_t op) {
    return op!=OP_ENUM_VAL && (supported(op) || record_operation(op));
}
#include "record_array_origins.inc"
static int packed(uint8_t t) { return t==TAG_INT || t==TAG_U8 || t==TAG_FLOAT || t==TAG_BOOL; }
static uint16_t writes(uint8_t t) {
    return BIT(t) | (t==TAG_INT?BIT(TAG_U8):t==TAG_U8 || t==TAG_FLOAT?BIT(TAG_INT):0);
}
static Value read_array(Analysis *a,Value receiver) {
    Value result=tag(TAG_VOID);
    if(!analysis_work(a,ORIGINS*4u))return result;
    result.unknown=receiver.unknown || ((receiver.tags&BIT(TAG_ARRAY)) && !receiver.origins);
    for(uint32_t i=0;i<a->report.origin_count;i++) if((receiver.origins&(UINT64_C(1)<<i)) && a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        NvmArrayOrigin *o=&a->report.origins[i];
        result.tags |= o->packed?BIT(o->declared_tag):o->child_tags;
        if(a->graph && !o->packed)merge(&result,a->children[i]);
    }
    return result;
}
static void write_array(Analysis *a,Value receiver,Value value) {
    if(!analysis_work(a,ORIGINS*4u))return;
    for(uint32_t i=0;i<a->report.origin_count;i++) if((receiver.origins&(UINT64_C(1)<<i)) && a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        NvmArrayOrigin *o=&a->report.origins[i];
        if(a->structure)a->changed|=merge(&a->all_writes[i],value);
        if(!o->packed) {
            uint16_t old=o->child_tags;o->child_tags|=value.tags;
            a->changed |= old!=o->child_tags;
            if(a->graph)a->changed |= merge(&a->children[i],value);
        }
    }
}
/* I keep copies distinct from sources, but weakly merge repeated copies at one site. */
static int slice_origins(Analysis *a,uint32_t fi,uint32_t pc,Value receiver,Value *result) {
    if(!analysis_work(a,ORIGINS*(ORIGINS+8u)))return 0;
    *result=tag(TAG_ARRAY);
    result->unknown=receiver.unknown || ((receiver.tags&BIT(TAG_ARRAY)) && !receiver.origins);
    /* A graph slice cannot succeed on a currently non-array receiver. Keep
     * this transfer at bottom until a possible source origin arrives; a
     * tag-only fabricated ARRAY would poison later joins with unknown. */
    if(a->graph && !receiver.origins && !receiver.unknown && !(receiver.tags&BIT(TAG_ARRAY)))
        *result=(Value){0};
    for(uint32_t i=0;i<a->report.origin_count;i++)if((receiver.origins&(UINT64_C(1)<<i)) && a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        NvmArrayOrigin source=a->report.origins[i];
        uint32_t target=0;
        while(target<a->report.origin_count) {
            NvmArrayOrigin *o=&a->report.origins[target];
            if(a->origin_kind[target]==NVM_HEAP_ORIGIN_ARRAY && o->function==fi && o->pc==pc && o->declared_tag==source.declared_tag)break;
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
        if(a->graph && !o->packed)a->changed|=merge(&a->children[target],a->children[i]);
        result->origins|=UINT64_C(1)<<target;
    }
    return 1;
}
static int check_write(Analysis *a,uint32_t fi,uint32_t pc,Value receiver,Value value) {
    if(!analysis_work(a,ORIGINS*4u))return 0;
    a->report.checked_writes++;
    uint16_t allowed=a->structure?RECORD_ARRAY_LEAVES:LEAVES|(a->graph?BIT(TAG_ARRAY):0);
    if(a->structure && value.origins)
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require flat scalar/string array content without heap children.");
    if(value.unknown || !value.tags || (value.tags&~allowed))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,a->graph && !a->structure?
            "I require proved scalar/string/array graph writes.":"I require proved scalar/string leaf writes.");
    if(a->graph && (value.tags&BIT(TAG_ARRAY)) && !value.origins)
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require authoritative child array origins.");
    for(uint32_t i=0;i<a->report.origin_count;i++)if((receiver.origins&(UINT64_C(1)<<i)) && a->origin_kind[i]==NVM_HEAP_ORIGIN_ARRAY) {
        NvmArrayOrigin *o=&a->report.origins[i];
        if(o->packed && (value.tags&~writes(o->declared_tag)))
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I cannot prove a portable packed write for every possible tag.");
    }
    return 1;
}
/* Record facts never enter the existing leaf/graph API. These identities are
 * per-allocation-site, not strong facts about one particular live instance. */
static int prepare_records(Analysis *a) {
    if(a->structure)return ra_prepare_records(a);
    NvmRecordPlanResult result=nvm_describe_managed_records(a->module,&a->plan);
    if(result.status!=NVM_RECORD_DESCRIBED) {
        NvmArrayEligibilityStatus status=result.status==NVM_RECORD_MEMORY?NVM_ARRAY_MEMORY:
            result.status==NVM_RECORD_LIMIT?NVM_ARRAY_LIMIT:
            result.status==NVM_RECORD_INVALID?NVM_ARRAY_INVALID:NVM_ARRAY_UNRESOLVED;
        return stop(a,status,0,0,result.message);
    }
    if(a->plan->authority!=NVM_RECORD_AUTHORITY_ORDINARY)
        return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require explicit ordinary record authority.");
    bool requires_verifier=false;
    if(nvm_ownership_contracts_validate(a->module,&requires_verifier)!=NVM_V2_OK || requires_verifier)
        return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require ordinary declarations without affine/reference execution.");
    for(uint32_t i=0;i<a->plan->record_count;i++) {
        const NvmV2Layout *layout=&a->plan->layouts.items[a->plan->record_to_layout[i]];
        for(uint16_t j=0;j<layout->field_count;j++) {
            const NvmV2LayoutField *field=&layout->fields[j];
            if(field->type_tag!=TAG_STRUCT && !(LEAVES&BIT(field->type_tag)))
                return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require the bounded scalar/string/acyclic-record field schema.");
        }
    }
    return 1;
}
static int record_site(Analysis *a,uint32_t fi,uint32_t pc) {
    Function *f=&a->functions[fi];VmDecodedInstruction *d=&f->decoded.instructions[pc];
    const DecodedInstruction *in=&d->instruction;uint8_t op=in->opcode;
    if(op==OP_AGG_PACK && (in->operands[0].u8!=AGG_RECORD || in->operands[2].u16))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I require neutral-variant record construction.");
    uint32_t record=in->operands[op==OP_AGG_PACK?1:0].u32;
    if(record>=a->plan->record_count)
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I require an authoritative record ordinal.");
    const NvmV2Layout *layout=&a->plan->layouts.items[a->plan->record_to_layout[record]];
    uint16_t count=op==OP_STRUCT_NEW?0:in->operands[op==OP_AGG_PACK?3:1].u16;
    if(count!=layout->field_count)
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I require the actual constructed field count to match its record.");
    if(a->report.origin_count==ORIGINS || count>FIELD_CELLS-a->field_count ||
       a->state_cells+(uint64_t)a->field_count+count>CELLS)
        return stop(a,NVM_ARRAY_LIMIT,fi,d->byte_offset,"I reached my record origin or field-summary limit.");
    uint32_t origin=a->report.origin_count++;f->origins[pc]=(int16_t)origin;
    a->origin_kind[origin]=NVM_HEAP_ORIGIN_RECORD;
    a->record_ordinal[origin]=record;a->record_fields[origin]=count;
    a->field_start[origin]=a->field_count;a->field_count+=count;
    a->report.origins[origin]=(NvmArrayOrigin){fi,d->byte_offset,0,TAG_STRUCT,0};
    return 1;
}
static uint64_t origins_of_kind(const Analysis *a,Value value,uint8_t kind) {
    uint64_t found=0;
    for(uint32_t i=0;i<a->report.origin_count;i++)
        if((value.origins&(UINT64_C(1)<<i)) && a->origin_kind[i]==kind)found|=UINT64_C(1)<<i;
    return found;
}
static void write_record(Analysis *a,Value receiver,uint16_t field,Value value) {
    if(!analysis_work(a,ORIGINS*4u))return;
    uint64_t origins=origins_of_kind(a,receiver,NVM_HEAP_ORIGIN_RECORD);
    for(uint32_t i=0;i<a->report.origin_count;i++)if(origins&(UINT64_C(1)<<i)) {
        /* Unsupported bounds are diagnosed after convergence, never indexed. */
        if(field<a->record_fields[i])a->changed|=merge(&a->fields[a->field_start[i]+field],value);
    }
}
static Value read_record(Analysis *a,Value receiver,uint16_t field) {
    Value result={0};
    if(!analysis_work(a,ORIGINS*4u))return result;
    uint64_t origins=origins_of_kind(a,receiver,NVM_HEAP_ORIGIN_RECORD);
    result.unknown=receiver.unknown || ((receiver.tags&BIT(TAG_STRUCT)) && !origins);
    for(uint32_t i=0;i<a->report.origin_count;i++)if(origins&(UINT64_C(1)<<i))
        if(field<a->record_fields[i])merge(&result,a->fields[a->field_start[i]+field]);
    return result;
}
static int record_value_matches(Analysis *a,uint32_t fi,uint32_t pc,
                                const NvmV2LayoutField *field,Value value,uint8_t element) {
    if(!analysis_work(a,ORIGINS*4u))return 0;
    a->checked_field_writes++;
    if(value.unknown || value.tags!=BIT(field->type_tag))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require exact known field-write tags.");
    if(a->structure && field->type_tag==TAG_ARRAY) {
        uint64_t origins=origins_of_kind(a,value,NVM_HEAP_ORIGIN_ARRAY);
        if(!element || !origins || origins!=value.origins)
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require complete field array origins and element identity.");
        for(uint32_t i=0;i<a->report.origin_count;i++)if(origins&(UINT64_C(1)<<i)) {
            NvmArrayOrigin *origin=&a->report.origins[i];
            if(origin->declared_tag!=element || a->children[i].unknown ||
               a->children[i].origins || (origin->child_tags&~BIT(element)))
                return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require the exact flat field element contract.");
            a->required_elements[i]|=BIT(element);
            if(a->required_elements[i]!=BIT(element))
                return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I refuse conflicting field element constraints.");
        }
    }
    if(field->type_tag==TAG_STRUCT) {
        uint64_t origins=origins_of_kind(a,value,NVM_HEAP_ORIGIN_RECORD);
        if(!origins || origins!=value.origins)
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require authoritative nested record origins.");
        for(uint32_t i=0;i<a->report.origin_count;i++)if(origins&(UINT64_C(1)<<i))
            if(a->plan->record_to_layout[a->record_ordinal[i]]!=field->nested_idx)
                return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require the exact nested nominal identity for every field write.");
    }
    return 1;
}
static int check_record_access(Analysis *a,uint32_t fi,uint32_t pc,
                               Value receiver,uint16_t field,const Value *write) {
    if(!analysis_work(a,ORIGINS*4u))return 0;
    uint64_t origins=origins_of_kind(a,receiver,NVM_HEAP_ORIGIN_RECORD);
    if(receiver.unknown || !receiver.tags || ((receiver.tags&BIT(TAG_STRUCT)) && !origins))
        return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require authoritative record receiver origins.");
    if(receiver.tags!=BIT(TAG_STRUCT))a->report.runtime_tag_checks++;
    for(uint32_t i=0;i<a->report.origin_count;i++)if(origins&(UINT64_C(1)<<i)) {
        if(field>=a->record_fields[i])
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I require a valid field in every possible record receiver.");
        const NvmV2Layout *layout=&a->plan->layouts.items[a->plan->record_to_layout[a->record_ordinal[i]]];
        if(write && !record_value_matches(a,fi,pc,&layout->fields[field],*write,
            a->structure?a->field_elements[a->layout_fields[a->plan->record_to_layout[a->record_ordinal[i]]]+field]:0))return 0;
    }
    return 1;
}
static int publish_records(Analysis *a,NvmRecordEligibilityReport **out) {
    NvmRecordEligibilityReport *report=analysis_alloc(a,1,sizeof *report);
    if(!report)return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not publish my record analysis report.");
    if(a->field_count) {
        report->fields=analysis_alloc(a,a->field_count,sizeof *report->fields);
        if(!report->fields) {
            nvm_record_eligibility_free(report);
            return stop(a,NVM_ARRAY_MEMORY,0,0,"I could not publish my record field summaries.");
        }
    }
    report->origin_count=a->report.origin_count;report->field_value_count=a->field_count;
    report->checked_field_writes=a->checked_field_writes;
    report->checked_array_writes=a->report.checked_writes;
    report->runtime_tag_checks=a->report.runtime_tag_checks;
    for(uint32_t i=0;i<a->report.origin_count;i++) {
        NvmArrayOrigin *source=&a->report.origins[i];NvmRecordHeapOrigin *origin=&report->origins[i];
        origin->function=source->function;origin->pc=source->pc;origin->kind=a->origin_kind[i];
        origin->record_ordinal=origin->layout_index=NVM_V2_NO_INDEX;
        origin->declared_tag=source->declared_tag;origin->packed=source->packed;
        if(origin->kind==NVM_HEAP_ORIGIN_RECORD) {
            origin->record_ordinal=a->record_ordinal[i];
            origin->layout_index=a->plan->record_to_layout[origin->record_ordinal];
            origin->field_start=a->field_start[i];origin->field_count=a->record_fields[i];
        } else {
            origin->children=(NvmRecordValueOrigins){a->children[i].origins,source->child_tags,a->children[i].unknown};
        }
    }
    for(uint32_t i=0;i<a->field_count;i++)
        report->fields[i]=(NvmRecordValueOrigins){a->fields[i].origins,a->fields[i].tags,a->fields[i].unknown};
    *out=report;return 1;
}

static int walk(Analysis *a,uint32_t fi) {
    Function *f=&a->functions[fi];const NvmFunctionEntry *entry=&a->module->functions[fi];
    if(!analysis_work(a,(uint64_t)f->decoded.instruction_count+1))return 0;
    for(uint32_t i=0;i<=f->decoded.instruction_count;i++)if(f->seen[i])enqueue(f,i);
    while(f->queued_count) {
        /* Fixed stack/seed scratch and copies; heap scans and joins charge
         * separately, including every constructor/literal operand. */
        if(!analysis_work(a,2048))return 0;
        uint32_t index=f->queue[f->head];f->head=(f->head+1)%(f->decoded.instruction_count+1);
        f->queued_count--;f->queued[index]=0;
        Value state[SLOTS*2]={{0}};uint16_t depth=f->depths[index];
        memcpy(state,f->states+(size_t)index*f->stride,((size_t)f->locals+depth)*sizeof(Value));
        Value *stack=state+f->locals;
        if(index==f->decoded.instruction_count) {
            if(a->structure && depth!=entry->result_count)
                return stop(a,NVM_ARRAY_INVALID,fi,entry->code_length,"I require the declared result depth at an implicit return.");
            if(entry->result_count)a->changed|=merge(&f->result,stack[depth-1]);
            continue;
        }
        VmDecodedInstruction *d=&f->decoded.instructions[index];DecodedInstruction *in=&d->instruction;
        uint8_t op=in->opcode;const InstructionInfo *info=isa_get_info(op);
        int pops=info->pop_count,pushes=info->push_count;
        if(op==OP_ARR_LITERAL || op==OP_STRUCT_LITERAL){pops=in->operands[1].u16;pushes=1;}
        if(op==OP_AGG_PACK){pops=in->operands[3].u16;pushes=1;}
        if(op==OP_CALL){pops=a->module->functions[in->operands[0].u32].arity;pushes=a->module->functions[in->operands[0].u32].result_count;}
        if(op==OP_RET) {
            if(a->structure && depth!=entry->result_count)
                return stop(a,NVM_ARRAY_INVALID,fi,d->byte_offset,"I require the declared result depth at an explicit return.");
            if(entry->result_count)a->changed|=merge(&f->result,stack[depth-1]);
            continue;
        }
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
            if(op==OP_ARR_LITERAL)for(uint32_t i=base;i<depth;i++) {
                write_array(a,result,stack[i]);
                if(a->structure && a->budget.limited)return 0;
            }
            break;
        case OP_ARR_SLICE:
            if(!slice_origins(a,fi,d->byte_offset,stack[base],&result))return 0;
            break;
        case OP_ARR_PUSH: case OP_ARR_SET:
            write_array(a,stack[base],stack[depth-1]);result=stack[base];break;
        case OP_ARR_GET: case OP_ARR_POP: result=read_array(a,stack[base]);break;
        case OP_STRUCT_NEW: case OP_STRUCT_LITERAL: case OP_AGG_PACK: {
            int ready=1;
            for(uint32_t i=base;i<depth;i++)if(!stack[i].tags && !stack[i].unknown)ready=0;
            if(ready) {
                result=tag(TAG_STRUCT);result.origins=UINT64_C(1)<<f->origins[index];
                for(uint32_t i=base;i<depth;i++) {
                    write_record(a,result,(uint16_t)(i-base),stack[i]);
                    if(a->structure && a->budget.limited)return 0;
                }
            }
            break;
        }
        case OP_STRUCT_GET: case OP_AGG_GET:
            result=read_record(a,stack[base],in->operands[0].u16);break;
        case OP_STRUCT_SET: case OP_AGG_SET:
            write_record(a,stack[base],in->operands[0].u16,stack[depth-1]);result=stack[base];break;
        case OP_CALL: {
            uint32_t callee=in->operands[0].u32;
            if(!seed(a,callee,stack+base))return 0;
            result=a->functions[callee].result;break;
        }
        default:break;
        }
        if(a->structure && a->budget.limited)return 0;
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
    if(!analysis_work(a,SLOTS+FUNCTIONS))return 0;
    for(uint32_t i=0;i<a->global_count;i++)if(a->globals[i].unknown)
        return stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I cannot resolve an escaped global value.");
    for(uint32_t fi=0;fi<a->module->function_count;fi++) {
        Function *f=&a->functions[fi];
        if(a->structure && a->module->functions[fi].result_count &&
           f->result.tags!=BIT(a->module->functions[fi].result_tag))
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,0,"I require exact ordinary result tags.");
        if(a->module->functions[fi].result_count && (f->result.unknown || !f->result.tags))
            return stop(a,NVM_ARRAY_UNRESOLVED,fi,0,"I cannot resolve this function result or return path.");
        for(uint32_t pc=0;pc<f->decoded.instruction_count;pc++) {
            if(!analysis_work(a,512))return 0;
            if(!f->seen[pc])continue;
            VmDecodedInstruction *d=&f->decoded.instructions[pc];uint8_t op=d->instruction.opcode;
            if(a->structure && op==OP_CALL) {
                uint32_t target=d->instruction.operands[0].u32;
                const NvmFunctionEntry *callee=&a->module->functions[target];
                Value *stack=f->states+(size_t)pc*f->stride+f->locals;
                for(uint16_t i=0;i<callee->arity;i++) {
                    uint8_t tag=a->module->function_param_types && a->module->function_param_types[target]?
                        a->module->function_param_types[target][i]:TAG_VOID;
                    Value argument=stack[f->depths[pc]-callee->arity+i];
                    if(tag!=TAG_VOID && (argument.unknown || argument.tags!=BIT(tag)))
                        return stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I require exact known direct-call parameter tags.");
                }
            }
            if(a->records && record_operation(op)) {
                Value *stack=f->states+(size_t)pc*f->stride+f->locals;
                uint16_t depth=f->depths[pc];
                if(op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_AGG_PACK) {
                    uint32_t origin=(uint32_t)f->origins[pc];
                    uint16_t count=a->record_fields[origin];
                    const NvmV2Layout *layout=&a->plan->layouts.items[a->plan->record_to_layout[a->record_ordinal[origin]]];
                    for(uint16_t i=0;i<count;i++)
                        if(!record_value_matches(a,fi,d->byte_offset,&layout->fields[i],stack[depth-count+i],
                            a->structure?a->field_elements[a->layout_fields[a->plan->record_to_layout[a->record_ordinal[origin]]]+i]:0))return 0;
                } else {
                    int writes=op==OP_STRUCT_SET || op==OP_AGG_SET;
                    Value receiver=stack[depth-(writes?2:1)];
                    if(!check_record_access(a,fi,d->byte_offset,receiver,d->instruction.operands[0].u16,
                                            writes?&stack[depth-1]:NULL))return 0;
                }
                continue;
            }
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
    nvm_record_array_structure_free(a->structure);
    free(a->field_elements);
    nvm_record_plan_free(a->plan);
    free(a->fields);
    free(a);
}
static NvmArrayEligibilityResult analyze(const NvmModule *m,NvmArrayEligibilityReport **out,
                                          NvmArrayGraphEligibilityReport **graph_out,
                                          NvmRecordEligibilityReport **record_out,NvmRecordArrayOrigins **mixed_out,uint16_t *prepared_stacks) {
    NvmArrayEligibilityResult early={NVM_ARRAY_INVALID,0,0,"I require a module and report output."};
    if(!m || (!out && !graph_out && !record_out && !mixed_out))return early;
    NvmRecordArrayBudget budget={0};NvmRecordArrayStructure *structure=NULL;
    if(mixed_out) {
        early=nvm_record_array_structure_prepare(m,&budget,&structure);
        if(early.status!=NVM_ARRAY_ELIGIBLE)return early;
        if(!nvm_ra_bytes(&budget,sizeof(Analysis))) {
            nvm_record_array_structure_free(structure);early.status=NVM_ARRAY_LIMIT;return early;
        }
    }
    Analysis *a=allocate(1,sizeof *a);
    if(!a){nvm_record_array_structure_free(structure);early.status=NVM_ARRAY_MEMORY;snprintf(early.message,sizeof early.message,"I could not allocate array analysis state.");return early;}
    a->module=m;a->structure=structure;a->budget=budget;
    a->records=record_out!=NULL || mixed_out!=NULL;a->graph=graph_out!=NULL || a->records;
    if(!a->structure && !verified(a,nvm_verify(m)))goto done;
    if(m->function_count>FUNCTIONS){stop(a,NVM_ARRAY_LIMIT,0,0,"I reached my array analysis function limit.");goto done;}
    if(!(m->header.flags&NVM_FLAG_HAS_MAIN) || m->functions[m->header.entry_point].arity ||
       m->import_count || m->module_ref_count || (!a->structure && m->union_count) || m->passive_size ||
       (!a->records && (m->struct_count || m->ownership_size || m->layout_size))) {
        stop(a,NVM_ARRAY_UNRESOLVED,0,0,"I require a closed zero-argument entry without nominal, ownership or host contracts.");goto done;
    }
    if(a->records && !prepare_records(a))goto done;
    uint32_t instructions=0;int initializer=-1;
    for(uint32_t fi=0;fi<m->function_count;fi++) {
        if(!analysis_work(a,(uint64_t)m->functions[fi].code_length*2u+1024u))goto done;
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
            if(!supported(in.opcode) && !(a->records && record_operation(in.opcode))){stop(a,NVM_ARRAY_UNRESOLVED,fi,pc,"I have no proved transfer for this instruction.");goto done;}
            if(in.opcode==OP_LOAD_GLOBAL || in.opcode==OP_STORE_GLOBAL) {
                uint32_t global=in.operands[0].u32;
                if(global>=SLOTS){stop(a,NVM_ARRAY_LIMIT,fi,pc,"I reached my array analysis global limit.");goto done;}
                if(global>=a->global_count)a->global_count=global+1;
            }
            pc+=width;
        }
        if(a->structure)f->stack=a->structure->stacks[fi];
        else if(!verified(a,nvm_verify_function_max_stack(m,fi,&f->stack)))goto done;
        if(f->stack>SLOTS){stop(a,NVM_ARRAY_LIMIT,fi,0,"I reached my array analysis stack limit.");goto done;}
        char error[VM_DECODE_ERROR_SIZE];
        if(a->structure) {
            f->decoded=a->structure->decoded[fi];memset(&a->structure->decoded[fi],0,sizeof f->decoded);
        } else if(!vm_decode_function(m,fi,&f->decoded,error)){stop(a,NVM_ARRAY_MEMORY,fi,0,error);goto done;}
        f->locals=e->local_count;f->stride=(uint32_t)f->locals+f->stack;if(!f->stride)f->stride=1;
        uint32_t count=f->decoded.instruction_count+1;
        a->state_cells+=(uint64_t)count*f->stride;
        if(a->state_cells+a->field_count>CELLS){stop(a,NVM_ARRAY_LIMIT,fi,0,"I reached my stored abstract-state cell limit.");goto done;}
        f->states=analysis_alloc(a,(size_t)count*f->stride,sizeof(Value));f->depths=analysis_alloc(a,count,sizeof(uint16_t));
        f->queue=analysis_alloc(a,count,sizeof(uint32_t));f->seen=analysis_alloc(a,count,1);f->queued=analysis_alloc(a,count,1);f->origins=analysis_alloc(a,count,sizeof(int16_t));
        if(!f->states || !f->depths || !f->queue || !f->seen || !f->queued || !f->origins){stop(a,NVM_ARRAY_MEMORY,fi,0,"I could not allocate my bounded function analysis.");goto done;}
        for(uint32_t i=0;i<f->decoded.instruction_count;i++) {
            VmDecodedInstruction *d=&f->decoded.instructions[i];uint8_t op=d->instruction.opcode;
            f->origins[i]=-1;
            if(a->records && (op==OP_STRUCT_NEW || op==OP_STRUCT_LITERAL || op==OP_AGG_PACK)) {
                if(!record_site(a,fi,i))goto done;
                continue;
            }
            if(op!=OP_ARR_NEW && op!=OP_STR_SPLIT && op!=OP_ARR_LITERAL)continue;
            uint8_t declared=op==OP_STR_SPLIT?TAG_STRING:d->instruction.operands[0].u8;
            uint16_t allowed=a->structure?RECORD_ARRAY_LEAVES:LEAVES|(a->graph?BIT(TAG_ARRAY):0);
            if(!(allowed&BIT(declared))){stop(a,NVM_ARRAY_UNRESOLVED,fi,d->byte_offset,"I have not qualified this declared array shape.");goto done;}
            if(a->report.origin_count==ORIGINS){stop(a,NVM_ARRAY_LIMIT,fi,d->byte_offset,"I reached my allocation-site origin limit.");goto done;}
            uint32_t origin=a->report.origin_count++;f->origins[i]=(int16_t)origin;
            a->report.origins[origin]=(NvmArrayOrigin){fi,d->byte_offset,op==OP_STR_SPLIT?BIT(TAG_STRING):0,declared,(uint8_t)packed(declared)};
            if(op==OP_STR_SPLIT)a->children[origin]=tag(TAG_STRING);
        }
    }
    if(a->field_count) {
        a->fields=analysis_alloc(a,a->field_count,sizeof *a->fields);
        if(!a->fields){stop(a,NVM_ARRAY_MEMORY,0,0,"I could not allocate my record field summaries.");goto done;}
    }
    for(uint32_t i=0;i<a->global_count;i++)a->globals[i]=tag(TAG_VOID);
    if(initializer>=0 && !seed(a,(uint32_t)initializer,NULL))goto done;
    if(!seed(a,m->header.entry_point,NULL))goto done;
    int seeded_unused=0;
    do {
        if(!analysis_work(a,(uint64_t)m->function_count*512u+256u))goto done;
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
    if(mixed_out) {
        if(prepared_stacks && !analysis_work(a,256))goto done;
        if(!ra_check_constraints(a) || !ra_publish(a,mixed_out))goto done;
        if(prepared_stacks)memcpy(prepared_stacks,a->structure->stacks,sizeof a->structure->stacks);
    } else if(record_out) {
        if(!publish_records(a,record_out))goto done;
    } else if(graph_out) {
        NvmArrayGraphEligibilityReport *report=analysis_alloc(a,1,sizeof *report);
        if(!report){stop(a,NVM_ARRAY_MEMORY,0,0,"I could not publish my graph analysis report.");goto done;}
        report->arrays=a->report;
        for(uint32_t i=0;i<a->report.origin_count;i++) {
            report->child_origins[i]=a->children[i].origins;
            report->child_unknown[i]=a->children[i].unknown;
        }
        *graph_out=report;
    } else {
        NvmArrayEligibilityReport *report=analysis_alloc(a,1,sizeof *report);
        if(!report){stop(a,NVM_ARRAY_MEMORY,0,0,"I could not publish my array analysis report.");goto done;}
        *report=a->report;*out=report;
    }
    a->result.status=NVM_ARRAY_ELIGIBLE;
    snprintf(a->result.message,sizeof a->result.message,"I established only the bounded portable array-shape obligations.");
 done:;
    NvmArrayEligibilityResult result=a->result;destroy(a);return result;
}

NvmArrayEligibilityResult nvm_analyze_managed_arrays(const NvmModule *m,NvmArrayEligibilityReport **out) {
    return analyze(m,out,NULL,NULL,NULL,NULL);
}
NvmArrayEligibilityResult nvm_analyze_managed_array_graphs(const NvmModule *m,NvmArrayGraphEligibilityReport **out) {
    return analyze(m,NULL,out,NULL,NULL,NULL);
}

NvmArrayEligibilityResult nvm_analyze_managed_records(const NvmModule *m,NvmRecordEligibilityReport **out) {
    return analyze(m,NULL,NULL,out,NULL,NULL);
}

NvmArrayEligibilityResult nvm_analyze_record_array_origins(const NvmModule *m,NvmRecordArrayOrigins **out) {
    return analyze(m,NULL,NULL,NULL,out,NULL);
}

NvmArrayEligibilityResult nvm_select_managed_array_mode(const NvmModule *m,int *graph_required) {
    if(!graph_required) {
        NvmArrayEligibilityResult result={.status=NVM_ARRAY_INVALID};
        snprintf(result.message,sizeof result.message,"I require an array-mode output.");
        return result;
    }
    NvmArrayEligibilityReport *leaf=NULL;
    NvmArrayEligibilityResult result=nvm_analyze_managed_arrays(m,&leaf);
    nvm_array_eligibility_free(leaf);
    if(result.status==NVM_ARRAY_ELIGIBLE){*graph_required=0;return result;}
    if(result.status!=NVM_ARRAY_UNRESOLVED)return result;
    NvmArrayGraphEligibilityReport *graph=NULL;
    result=nvm_analyze_managed_array_graphs(m,&graph);
    nvm_array_graph_eligibility_free(graph);
    if(result.status==NVM_ARRAY_ELIGIBLE)*graph_required=1;
    return result;
}

void nvm_managed_heap_plan_free(NvmManagedHeapPlan *plan) {
    if(!plan)return;
    nvm_record_plan_free(plan->records);
    nvm_record_eligibility_free(plan->fields);
    free(plan);
}
NvmArrayEligibilityResult nvm_select_managed_heap(const NvmModule *m,int mutable_arrays,
                                                 NvmManagedHeapPlan **out) {
    NvmArrayEligibilityResult result={.status=NVM_ARRAY_INVALID};
    snprintf(result.message,sizeof result.message,"I require a module and heap-plan output.");
    if(!m || !out)return result;
    NvmVerifyResult verified=nvm_verify(m);
    if(!verified.ok)return result;
    NvmManagedHeapPlan staged={0};
    if(m->struct_count || m->layout_size || m->ownership_size) {
        NvmRecordPlanResult described=nvm_describe_managed_records(m,&staged.records);
        if(described.status!=NVM_RECORD_DESCRIBED) {
            result.status=described.status==NVM_RECORD_MEMORY?NVM_ARRAY_MEMORY:
                described.status==NVM_RECORD_LIMIT?NVM_ARRAY_LIMIT:
                described.status==NVM_RECORD_INVALID?NVM_ARRAY_INVALID:NVM_ARRAY_UNRESOLVED;
            snprintf(result.message,sizeof result.message,"%s",described.message);
            return result;
        }
        if(staged.records->authority!=NVM_RECORD_AUTHORITY_ORDINARY) {
            result.status=NVM_ARRAY_UNRESOLVED;
            snprintf(result.message,sizeof result.message,"I require checked ordinary record authority.");
        } else result=nvm_analyze_managed_records(m,&staged.fields);
        staged.mode=NVM_MANAGED_RECORD;
    } else if(mutable_arrays) {
        int graph=0;
        result=nvm_select_managed_array_mode(m,&graph);
        staged.mode=graph?NVM_MANAGED_ARRAY_GRAPH:NVM_MANAGED_LEAF;
    } else result.status=NVM_ARRAY_ELIGIBLE;
    if(result.status==NVM_ARRAY_ELIGIBLE) {
        NvmManagedHeapPlan *plan=allocate(1,sizeof *plan);
        if(plan){*plan=staged;*out=plan;return result;}
        result.status=NVM_ARRAY_MEMORY;
        snprintf(result.message,sizeof result.message,"I could not publish my managed heap plan.");
    }
    nvm_record_plan_free(staged.records);
    nvm_record_eligibility_free(staged.fields);
    return result;
}

#include "managed_record_array_execution.inc"
