/* I prove closed managed shape without granting scalar or affine authority. */
#include "mixed_float_proof.h"
#include "ownership_contracts.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>
#define MF_FUNCS 8u
#define MF_SLOTS 256u
#define MF_CELLS 1048576u
#define MF_FIELDS 65536u
#define MF_VISITS 262144u
#define MF_BIT(t) ((uint16_t)(1u << (t)))
#define MF_SCALARS (MF_BIT(TAG_VOID)|MF_BIT(TAG_INT)|MF_BIT(TAG_U8)|MF_BIT(TAG_FLOAT)|MF_BIT(TAG_BOOL))
typedef struct { uint16_t tags; uint64_t origins; uint32_t nominal; uint16_t root; uint8_t kind, unknown; } MFValue;
/* kind: 0 ordinary/scalar, 1 unique owner, 2 nonescaping owner observation. */
typedef struct { uint8_t tag; uint32_t layout; } MFType;
typedef struct {
    DecodedInstruction in;
    uint32_t pc, next, target, function;
    int16_t origin;
    int16_t depth;
    MFValue *state;
    bool queued;
} MFInstruction;
typedef struct {
    uint32_t start, end;
    MFType result, locals[MF_SLOTS];
    uint8_t calls, color;
} MFFunction;
typedef struct {
    const NvmModule *module;
    NvmMixedFloatProof *proof;
    MFFunction functions[MF_FUNCS];
    MFInstruction instructions[NVM_MIXED_FLOAT_INSTRUCTIONS];
    uint32_t count, cells, visits;
    bool checking;
    uint32_t queue[NVM_MIXED_FLOAT_INSTRUCTIONS], head, tail, queued;
    NvmMixedShapeResult error;
} MFAnalysis;
static bool mf_stop(MFAnalysis *a,NvmMixedShapeStatus s,uint32_t f,uint32_t pc,const char *message) {
    a->error=(NvmMixedShapeResult){s,f,pc,message}; return false;
}
static MFValue mf_tag(uint8_t tag) { MFValue v={0};v.tags=MF_BIT(tag);return v; }
static MFValue mf_owner(uint32_t nominal) { MFValue v=mf_tag(TAG_STRUCT);v.kind=1;v.nominal=nominal;return v; }
static bool mf_exact(MFValue v,uint16_t tags) { return !v.unknown && !v.kind && !v.origins && v.tags==tags; }
static bool mf_merge(MFValue *to,MFValue from) {
    uint16_t tags=to->tags;uint64_t origins=to->origins;uint8_t unknown=to->unknown;
    to->tags|=from.tags;to->origins|=from.origins;to->unknown|=from.unknown;
    return tags!=to->tags || origins!=to->origins || unknown!=to->unknown;
}
static void mf_enqueue(MFAnalysis *a,uint32_t index) {
    if(a->instructions[index].queued)return;
    a->instructions[index].queued=true;a->queue[a->tail]=index;
    a->tail=(a->tail+1)%NVM_MIXED_FLOAT_INSTRUCTIONS;a->queued++;
}
static bool mf_join(MFAnalysis *a,uint32_t index,const MFValue *state,uint16_t depth) {
    MFInstruction *i=&a->instructions[index];uint32_t locals=a->module->functions[i->function].local_count;
    if(depth>MF_SLOTS)return mf_stop(a,NVM_MIXED_SHAPE_LIMIT,i->function,i->pc,"I reached my shape stack limit.");
    if(i->depth>=0 && i->depth!=depth)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,i->function,i->pc,"I require equal stack heights at shape joins.");
    bool changed=false;
    if(!i->state) {
        uint32_t cells=locals+MF_SLOTS;
        if(cells>MF_CELLS-a->cells)return mf_stop(a,NVM_MIXED_SHAPE_LIMIT,i->function,i->pc,"I reached my shape state budget.");
        i->state=calloc(cells,sizeof *i->state);
        if(!i->state)return mf_stop(a,NVM_MIXED_SHAPE_MEMORY,i->function,i->pc,"I could not allocate shape state.");
        a->cells+=cells;memcpy(i->state,state,(locals+depth)*sizeof *state);i->depth=(int16_t)depth;changed=true;
    } else for(uint32_t n=0;n<locals+depth;n++) {
        MFValue *old=&i->state[n],v=state[n];
        if((old->kind || v.kind) && (old->kind!=v.kind || old->nominal!=v.nominal || old->root!=v.root || old->tags!=v.tags))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,i->function,i->pc,"I retain exact opaque owner states at joins.");
        changed|=mf_merge(old,v);
    }
    if(changed)mf_enqueue(a,index);
    return true;
}
static bool mf_scalar_layout(const NvmMixedLayoutView *v,uint32_t layout) {
    if(layout>=v->layouts.count || v->classes[layout]!=NVM_MIXED_RESOURCE)return false;
    const NvmV2Layout *l=&v->layouts.items[layout];
    for(uint16_t i=0;i<l->field_count;i++) {
        uint8_t tag=l->fields[i].type_tag;
        if((tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_U8) || l->fields[i].nested_idx!=NVM_V2_NO_INDEX)return false;
    }
    return true;
}
static bool mf_signature_type(MFAnalysis *a,MFType t) {
    if(t.tag==TAG_STRUCT)return mf_scalar_layout(a->proof->view,t.layout);
    return t.layout==NVM_V2_NO_INDEX && (t.tag==TAG_VOID || t.tag==TAG_INT || t.tag==TAG_BOOL || t.tag==TAG_U8);
}
static bool mf_descriptors(MFAnalysis *a) {
    NvmV2Cursor c;NvmMixedLayoutView *v=a->proof->view;uint32_t n,version;const uint8_t *unused;
    nvm_v2_cursor_init(&c,v->ownership,v->ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || nvm_v2_u32(&c,&n)!=NVM_V2_OK ||
       nvm_v2_take(&c,n,&unused)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&n)!=NVM_V2_OK)return false;
    for(uint32_t f=0;f<n;f++) {
        uint16_t locals,params;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK)return false;
        for(uint32_t j=0;j<=locals;j++) {
            MFType t;uint8_t mode;uint16_t reserved;
            if(nvm_v2_u8(&c,&t.tag)!=NVM_V2_OK || nvm_v2_u8(&c,&mode)!=NVM_V2_OK ||
               nvm_v2_u16(&c,&reserved)!=NVM_V2_OK || nvm_v2_u32(&c,&t.layout)!=NVM_V2_OK)return false;
            if(mode)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I keep reference calls outside my shape proof.");
            if(j>0 && j<=params && t.tag==TAG_VOID)
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I require concrete parameter tags.");
            if(j==0 || j<=params) {
                if(!mf_signature_type(a,t))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I require existing scalar or scalar-leaf owner signatures.");
            }
            if(!j && !f && t.tag!=TAG_INT && t.tag!=TAG_BOOL && t.tag!=TAG_U8)
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I retain scalar entry results.");
            if(!j)a->functions[f].result=t;else a->functions[f].locals[j-1]=t;
        }
    }
    return true;
}
/* Result and operand requirements are explicit; no default producer inference. */
static int mf_scalar_op(uint8_t op,uint16_t *required) {
    *required=0;
    switch(op) {
    case OP_PUSH_I64:return TAG_INT;case OP_PUSH_F64:return TAG_FLOAT;
    case OP_PUSH_U8:return TAG_U8;case OP_PUSH_BOOL:return TAG_BOOL;case OP_PUSH_VOID:return TAG_VOID;
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:case OP_I64_DIV_S:case OP_I64_REM_S:case OP_I64_NEG:
        *required=MF_BIT(TAG_INT);return TAG_INT;
    case OP_F64_ADD:case OP_F64_SUB:case OP_F64_MUL:case OP_F64_DIV:case OP_F64_NEG:
        *required=MF_BIT(TAG_FLOAT);return TAG_FLOAT;
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
        *required=MF_BIT(TAG_INT);return TAG_BOOL;
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:
        *required=MF_BIT(TAG_FLOAT);return TAG_BOOL;
    case OP_BOOL_AND:case OP_BOOL_OR:case OP_BOOL_NOT:
        *required=MF_BIT(TAG_BOOL);return TAG_BOOL;
    case OP_EQ:case OP_NE:case OP_LT:case OP_LE:case OP_GT:case OP_GE:return TAG_BOOL;
    default:return -1;
    }
}
static bool mf_supported(uint8_t op) {
    uint16_t required;if(mf_scalar_op(op,&required)>=0)return true;
    switch(op) {
    case OP_NOP:case OP_DUP:case OP_POP:case OP_SWAP:case OP_ROT3:
    case OP_LOAD_LOCAL:case OP_STORE_LOCAL:case OP_OWN_MOVE_LOCAL:case OP_OWN_STORE_LOCAL:
    case OP_OWN_PACK:case OP_OWN_UNPACK_LOCAL:case OP_AGG_PACK:case OP_AGG_GET:
    case OP_ARR_NEW:case OP_ARR_LITERAL:case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN:
    case OP_JMP:case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_CALL:case OP_RET:case OP_ASSERT:return true;
    default:return false;
    }
}
static bool mf_graph(MFAnalysis *a,uint32_t f) {
    MFFunction *fn=&a->functions[f];if(fn->color==2)return true;
    if(fn->color==1)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I require an acyclic complete call graph.");
    fn->color=1;
    for(uint32_t c=0;c<a->module->function_count;c++)if((fn->calls&(1u<<c)) && !mf_graph(a,c))return false;
    fn->color=2;return true;
}
static bool mf_preflight(MFAnalysis *a) {
    const NvmModule *m=a->module;NvmMixedLayoutView *v=a->proof->view;
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];MFFunction *af=&a->functions[f];af->start=a->count;
        if(fn->name_idx>=m->string_count || !m->strings || !m->strings[fn->name_idx])
            return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,0,"I require exact function name indices.");
        if(!strcmp(m->strings[fn->name_idx],"__init__"))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I keep initializer execution outside this proof.");
        if(fn->code_offset>m->code_size || fn->code_length>m->code_size-fn->code_offset || !fn->code_length)
            return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,0,"I require complete nonempty code ranges.");
        for(uint32_t earlier=0;earlier<f;earlier++) {
            const NvmFunctionEntry *other=&m->functions[earlier];
            if(fn->code_offset<(uint64_t)other->code_offset+other->code_length &&
               other->code_offset<(uint64_t)fn->code_offset+fn->code_length)
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I require distinct function code ranges.");
        }
        for(uint32_t pc=0;pc<fn->code_length;) {
            if(a->count==NVM_MIXED_FLOAT_INSTRUCTIONS)return mf_stop(a,NVM_MIXED_SHAPE_LIMIT,f,pc,"I reached my decoded instruction limit.");
            MFInstruction *i=&a->instructions[a->count++];i->pc=pc;i->function=f;i->origin=-1;i->depth=-1;
            uint32_t width=isa_decode(m->code+fn->code_offset+pc,fn->code_length-pc,&i->in);
            if(!width)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require complete decoded instructions.");
            uint8_t op=i->in.opcode;
            if(!mf_supported(op))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I have no closed transfer for this opcode.");
            if(op==OP_PUSH_BOOL && i->in.operands[0].u8>1)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require canonical boolean operands.");
            if(op==OP_LOAD_LOCAL || op==OP_STORE_LOCAL || op==OP_OWN_MOVE_LOCAL || op==OP_OWN_STORE_LOCAL || op==OP_OWN_UNPACK_LOCAL) {
                if(i->in.operands[0].u16>=fn->local_count)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require an existing local index.");
            }
            if(op==OP_CALL) {
                uint32_t target=i->in.operands[0].u32;
                if(target>=m->function_count)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require an existing direct callee.");
                if(!target)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I do not call my entry function.");
                af->calls|=(uint8_t)(1u<<target);
            }
            if(op==OP_OWN_PACK || op==OP_OWN_UNPACK_LOCAL) {
                uint32_t layout=op==OP_OWN_PACK?i->in.operands[0].u32:af->locals[i->in.operands[0].u16].layout;
                if(!mf_scalar_layout(v,layout))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I require exact scalar-leaf owner effects.");
            }
            if(op==OP_ARR_NEW || op==OP_ARR_LITERAL || op==OP_AGG_PACK) {
                uint32_t layout=NVM_V2_NO_INDEX,source=NVM_V2_NO_INDEX,fields=0;
                if(op==OP_AGG_PACK) {
                    source=i->in.operands[1].u32;
                    if(i->in.operands[0].u8 || i->in.operands[2].u16 || source>=v->record_count)
                        return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I require an exact ordinary struct constructor.");
                    layout=v->source_to_global[source];fields=v->layouts.items[layout].field_count;
                    if(fields!=i->in.operands[3].u16)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require the declared constructor field count.");
                    if(v->classes[layout]!=NVM_MIXED_ORDINARY_STRUCTURAL && v->classes[layout]!=NVM_MIXED_PENDING_ARRAY_PROOF)
                        return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I require a known nonresource constructor declaration.");
                } else if(i->in.operands[0].u8!=TAG_FLOAT)
                    return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I prove only declared packed FLOAT arrays.");
                if(a->proof->origin_count==NVM_MIXED_FLOAT_ORIGINS || fields>MF_FIELDS-a->proof->field_count)
                    return mf_stop(a,NVM_MIXED_SHAPE_LIMIT,f,pc,"I reached my allocation-origin or field-summary budget.");
                i->origin=(int16_t)a->proof->origin_count++;
                a->proof->origins[i->origin]=(NvmMixedFloatOrigin){f,pc,layout,source,a->proof->field_count,(uint16_t)fields,op==OP_AGG_PACK?TAG_STRUCT:TAG_ARRAY};
                a->proof->field_count+=fields;
            }
            pc+=width;i->next=a->count;i->target=UINT32_MAX;
        }
        af->end=a->count;
    }
    for(uint32_t f=0;f<m->function_count;f++) {
        MFFunction *fn=&a->functions[f];
        for(uint32_t n=fn->start;n<fn->end;n++) {
            MFInstruction *i=&a->instructions[n];uint8_t op=i->in.opcode;
            if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
                int64_t target=(int64_t)i->pc+i->in.byte_length+i->in.operands[0].i32;
                for(uint32_t j=fn->start;j<fn->end;j++)if(target==a->instructions[j].pc)i->target=j;
                if(target==m->functions[f].code_length)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require explicit terminal instructions, not end-of-code branches.");
                if(i->target==UINT32_MAX)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,i->pc,"I require an instruction-boundary branch target.");
            }
            if(i->next==fn->end && op!=OP_JMP && op!=OP_RET)
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require explicit terminal control flow.");
        }
        if(!mf_graph(a,f))return false;
    }
    return true;
}
static bool mf_array(MFAnalysis *a,MFValue v) {
    if(v.kind || v.unknown || v.tags!=MF_BIT(TAG_ARRAY) || !v.origins)return false;
    for(uint32_t i=0;i<a->proof->origin_count;i++)if((v.origins&(UINT64_C(1)<<i)) && a->proof->origins[i].tag!=TAG_ARRAY)return false;
    return true;
}
static bool mf_field_matches(MFAnalysis *a,MFValue v,const NvmV2LayoutField *field) {
    if(v.kind || v.unknown || v.tags!=MF_BIT(field->type_tag))return false;
    if(field->type_tag==TAG_ARRAY)return mf_array(a,v);
    if(field->type_tag!=TAG_STRUCT)return !v.origins && (v.tags&MF_SCALARS)!=0;
    if(!v.origins)return false;
    for(uint32_t i=0;i<a->proof->origin_count;i++)if(v.origins&(UINT64_C(1)<<i)) {
        NvmMixedFloatOrigin *o=&a->proof->origins[i];
        if(o->tag!=TAG_STRUCT || o->global_layout!=field->nested_idx)return false;
    }
    return true;
}
static bool mf_type_matches(MFValue v,MFType t) {
    if(t.tag==TAG_STRUCT)return v.kind==1 && !v.unknown && !v.origins && v.tags==MF_BIT(TAG_STRUCT) && v.nominal==t.layout;
    return mf_exact(v,MF_BIT(t.tag));
}
static void mf_field_union(MFAnalysis *a,uint32_t origin,uint16_t field,MFValue value) {
    NvmMixedFieldFact *fact=&a->proof->fields[a->proof->origins[origin].field_start+field];
    uint16_t old_tags=fact->tags;uint64_t old_origins=fact->origins;
    fact->tags|=value.tags;fact->origins|=value.origins;
    if(old_tags!=fact->tags || old_origins!=fact->origins)
        for(uint32_t i=0;i<a->count;i++)if(a->instructions[i].state)mf_enqueue(a,i);
}
static bool mf_step(MFAnalysis *a,uint32_t index) {
    MFInstruction *i=&a->instructions[index];uint32_t f=i->function;
    const NvmFunctionEntry *fn=&a->module->functions[f];MFFunction *af=&a->functions[f];
    MFValue state[MF_SLOTS*2];memcpy(state,i->state,(fn->local_count+(uint16_t)i->depth)*sizeof *state);
    MFValue *stack=state+fn->local_count;uint16_t depth=(uint16_t)i->depth;
    DecodedInstruction *in=&i->in;uint8_t op=in->opcode;const InstructionInfo *info=isa_get_info(op);
    int pops=info->pop_count,pushes=info->push_count;
    if(op==OP_ARR_LITERAL) {pops=in->operands[1].u16;pushes=1;}
    if(op==OP_AGG_PACK) {pops=in->operands[3].u16;pushes=1;}
    if(op==OP_OWN_PACK)pops=a->proof->view->layouts.items[in->operands[0].u32].field_count;
    if(op==OP_OWN_UNPACK_LOCAL) {pops=0;pushes=a->proof->view->layouts.items[af->locals[in->operands[0].u16].layout].field_count;}
    if(op==OP_CALL) {const NvmFunctionEntry *callee=&a->module->functions[in->operands[0].u32];pops=callee->arity;pushes=callee->result_count;}
    if(op==OP_RET) {
        if(depth!=fn->result_count)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,i->pc,"I require the exact declared return count.");
        if(a->checking && depth && !mf_type_matches(stack[0],af->result))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact scalar or owned return identity.");
        return true; /* Exit consumption remains a separate affine obligation. */
    }
    if(pops<0 || pushes<0 || depth<pops)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,i->pc,"I require complete operand stacks.");
    if((uint32_t)(depth-pops+pushes)>MF_SLOTS)return mf_stop(a,NVM_MIXED_SHAPE_LIMIT,f,i->pc,"I reached my operand stack budget.");
    if(op==OP_OWN_MOVE_LOCAL || op==OP_OWN_UNPACK_LOCAL) {
        uint16_t local=in->operands[0].u16;
        for(uint16_t n=0;n<depth;n++)if(stack[n].kind==2 && stack[n].root==local)
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I cannot consume an observed owner.");
    }
    uint16_t base=(uint16_t)(depth-pops);MFValue result={0};uint16_t required;
    int scalar=mf_scalar_op(op,&required);
    NvmMixedScalarObligation *ob=&a->proof->obligations[index];ob->function=f;ob->pc=i->pc;
    if(scalar>=0) {
        result=mf_tag((uint8_t)scalar);
        for(uint16_t n=base;n<depth;n++) {
            if(stack[n].kind || stack[n].origins || stack[n].unknown || (stack[n].tags&~MF_SCALARS))
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I do not feed managed or opaque values to scalar instructions.");
            if(a->checking && !stack[n].tags)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require known scalar input alternatives.");
            if(required && stack[n].tags!=required) {ob->required_tags|=required;ob->actual_tags|=stack[n].tags;}
        }
        if(!required && pops>0) { /* Generic comparisons retain a scalar runtime obligation. */
            ob->actual_tags|=stack[base].tags;
            if(pops>1)ob->actual_tags|=stack[base+1].tags;
        }
    } else switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_DUP:
        if(stack[depth-1].kind)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I cannot duplicate an opaque owner or observation.");
        stack[depth]=stack[depth-1];depth++;goto successors;
    case OP_SWAP:result=stack[depth-1];stack[depth-1]=stack[depth-2];stack[depth-2]=result;goto successors;
    case OP_ROT3:result=stack[depth-1];stack[depth-1]=stack[depth-2];stack[depth-2]=stack[depth-3];stack[depth-3]=result;goto successors;
    case OP_POP:
        if(stack[base].kind)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require explicit opaque owner consumption.");
        break;
    case OP_LOAD_LOCAL:
        result=state[in->operands[0].u16];
        if(a->checking && (af->locals[in->operands[0].u16].tag==TAG_ARRAY ||
                          af->locals[in->operands[0].u16].tag==TAG_STRUCT) &&
           (result.tags&MF_BIT(TAG_VOID)))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require initialized managed or owned local alternatives.");
        if(result.kind==1) {result.kind=2;result.root=in->operands[0].u16;}
        break;
    case OP_STORE_LOCAL: {
        uint16_t local=in->operands[0].u16;MFType t=af->locals[local];result=stack[base];
        if(result.kind || state[local].kind)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I keep opaque ownership out of ordinary stores.");
        if(a->checking) {
            NvmV2LayoutField field={t.tag,t.layout,NVM_V2_NO_INDEX};
            if(!mf_field_matches(a,result,&field))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact initialized local shape.");
        }
        state[local]=result;break;
    }
    case OP_OWN_MOVE_LOCAL: {
        uint16_t local=in->operands[0].u16;result=state[local];
        if(!mf_type_matches(result,af->locals[local]))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an intact exact owner move.");
        if(!result.kind)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an owner for an owner move.");
        state[local]=mf_tag(TAG_VOID);break;
    }
    case OP_OWN_STORE_LOCAL: {
        uint16_t local=in->operands[0].u16;
        if(state[local].kind || !mf_exact(state[local],MF_BIT(TAG_VOID)) || !mf_type_matches(stack[base],af->locals[local]) || stack[base].kind!=1)
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an empty exact owner destination.");
        state[local]=stack[base];break;
    }
    case OP_OWN_PACK: {
        uint32_t layout=in->operands[0].u32;const NvmV2Layout *l=&a->proof->view->layouts.items[layout];
        for(uint16_t n=0;n<l->field_count;n++)if(a->checking && !mf_exact(stack[base+n],MF_BIT(l->fields[n].type_tag)))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact scalar owner fields.");
        result=mf_owner(layout);break;
    }
    case OP_OWN_UNPACK_LOCAL: {
        uint16_t local=in->operands[0].u16;MFType t=af->locals[local];
        if(!mf_type_matches(state[local],t))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an intact exact unpack source.");
        const NvmV2Layout *l=&a->proof->view->layouts.items[t.layout];state[local]=mf_tag(TAG_VOID);
        for(uint16_t n=0;n<l->field_count;n++)stack[depth++]=mf_tag(l->fields[n].type_tag);
        goto successors;
    }
    case OP_AGG_PACK: {
        NvmMixedFloatOrigin *o=&a->proof->origins[i->origin];const NvmV2Layout *l=&a->proof->view->layouts.items[o->global_layout];
        for(uint16_t n=0;n<l->field_count;n++) {
            MFValue value=stack[base+n];
            if(value.kind || value.unknown)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I refuse owner or unknown ordinary fields.");
            if(a->checking && !mf_field_matches(a,value,&l->fields[n]))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require every exact nominal field alternative.");
            if(!a->checking)mf_field_union(a,(uint32_t)i->origin,n,value);
        }
        result=mf_tag(TAG_STRUCT);result.origins=UINT64_C(1)<<i->origin;break;
    }
    case OP_AGG_GET: {
        MFValue receiver=stack[base];uint16_t field=in->operands[0].u16;
        if(receiver.kind) {
            if(receiver.kind!=2 || receiver.nominal>=a->proof->view->layouts.count)
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an exact owner observation.");
            const NvmV2Layout *l=&a->proof->view->layouts.items[receiver.nominal];
            if(field>=l->field_count)return mf_stop(a,NVM_MIXED_SHAPE_INVALID,f,i->pc,"I require an existing owner field.");
            if(receiver.root>=fn->local_count || state[receiver.root].kind!=1 || state[receiver.root].nominal!=receiver.nominal)
                return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require the observed owner to remain live.");
            result=mf_tag(l->fields[field].type_tag);break;
        }
        if(a->checking && (receiver.unknown || receiver.tags!=MF_BIT(TAG_STRUCT) || !receiver.origins))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact ordinary record receiver origins.");
        for(uint32_t n=0;n<a->proof->origin_count;n++)if(receiver.origins&(UINT64_C(1)<<n)) {
            NvmMixedFloatOrigin *o=&a->proof->origins[n];
            if(o->tag!=TAG_STRUCT || field>=o->field_count)return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require a field on every receiver alternative.");
            NvmMixedFieldFact *fact=&a->proof->fields[o->field_start+field];result.tags|=fact->tags;result.origins|=fact->origins;
        }
        result.unknown=receiver.unknown;break;
    }
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        for(uint16_t n=base;n<depth;n++)if(a->checking && !mf_exact(stack[n],MF_BIT(TAG_FLOAT)))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact FLOAT literal elements.");
        result=mf_tag(TAG_ARRAY);result.origins=UINT64_C(1)<<i->origin;break;
    case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN: {
        MFValue receiver=stack[base];
        if(a->checking && !mf_array(a,receiver))return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require positive closed FLOAT-array origins.");
        if((op==OP_ARR_SET || op==OP_ARR_GET) && a->checking && !mf_exact(stack[base+1],MF_BIT(TAG_INT)))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact integer array indices.");
        if((op==OP_ARR_PUSH || op==OP_ARR_SET) && a->checking && !mf_exact(stack[depth-1],MF_BIT(TAG_FLOAT)))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact FLOAT writes on every array alias.");
        if(op==OP_ARR_GET) {result=mf_tag(TAG_FLOAT);result.tags|=MF_BIT(TAG_VOID);ob->read_tags=result.tags;}
        else if(op==OP_ARR_LEN)result=mf_tag(TAG_INT);
        else {result=receiver;if(a->checking)a->proof->checked_writes++;}
        break;
    }
    case OP_CALL: {
        uint32_t target=in->operands[0].u32;const NvmFunctionEntry *callee=&a->module->functions[target];
        for(uint16_t n=0;n<callee->arity;n++)if(a->checking && !mf_type_matches(stack[base+n],a->functions[target].locals[n]))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require every positional call identity before transfer.");
        MFType t=a->functions[target].result;result=t.tag==TAG_STRUCT?mf_owner(t.layout):mf_tag(t.tag);break;
    }
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        if(stack[base].kind || stack[base].origins || stack[base].unknown || (stack[base].tags&~MF_SCALARS))
            return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require scalar control operands.");
        if(stack[base].tags!=MF_BIT(TAG_BOOL)) {ob->required_tags|=MF_BIT(TAG_BOOL);ob->actual_tags|=stack[base].tags;}
        break;
    default:return mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an explicit shape transfer.");
    }
    depth=(uint16_t)(base+pushes);if(pushes)stack[base]=result;
successors:
    if(a->checking)return true;
    if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
        if(!mf_join(a,i->target,state,depth))return false;
        if(op==OP_JMP)return true;
    }
    return mf_join(a,i->next,state,depth);
}
void nvm_mixed_float_proof_free(NvmMixedFloatProof *proof) {
    if(!proof)return;
    nvm_mixed_layout_view_free(proof->view);free(proof->fields);free(proof);
}
NvmMixedShapeResult nvm_analyze_mixed_float_origins(const NvmModule *m,NvmMixedFloatProof **out) {
    NvmMixedShapeResult invalid={NVM_MIXED_SHAPE_INVALID,0,0,"I require a complete immutable module and output."};
    if(!m || !out || !m->functions || !m->code || !m->function_count)return invalid;
    if(m->function_count>MF_FUNCS || m->header.entry_point || m->import_count || m->module_ref_count || m->callback_contract_count || m->passive_size)
        return (NvmMixedShapeResult){NVM_MIXED_SHAPE_UNRESOLVED,0,0,"I require my standalone eight-function closed value graph."};
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        if(fn->local_count>MF_SLOTS || fn->arity>8)return (NvmMixedShapeResult){NVM_MIXED_SHAPE_LIMIT,f,0,"I reached my function slot or argument budget."};
        if(fn->arity>fn->local_count || fn->result_count>1 || fn->upvalue_count || (!f && fn->arity))return invalid;
    }
    NvmMixedLayoutView *view=NULL;NvmRecordPlanResult described=nvm_describe_mixed_layouts(m,&view);
    if(described.status!=NVM_RECORD_DESCRIBED) {
        NvmMixedShapeStatus status=described.status==NVM_RECORD_MEMORY?NVM_MIXED_SHAPE_MEMORY:
            described.status==NVM_RECORD_LIMIT?NVM_MIXED_SHAPE_LIMIT:
            described.status==NVM_RECORD_INVALID?NVM_MIXED_SHAPE_INVALID:NVM_MIXED_SHAPE_UNRESOLVED;
        return (NvmMixedShapeResult){status,0,0,described.message};
    }
    if(!view->ownership_size) {nvm_mixed_layout_view_free(view);return (NvmMixedShapeResult){NVM_MIXED_SHAPE_UNRESOLVED,0,0,"I require explicit mixed declarations."};}
    MFAnalysis *a=calloc(1,sizeof *a);NvmMixedFloatProof *proof=calloc(1,sizeof *proof);
    if(!a || !proof) {free(a);free(proof);nvm_mixed_layout_view_free(view);return (NvmMixedShapeResult){NVM_MIXED_SHAPE_MEMORY,0,0,"I could not allocate my bounded proof."};}
    a->module=m;a->proof=proof;proof->view=view;proof->requires_affine_verification=true;a->error=invalid;
    if(!mf_descriptors(a) || !mf_preflight(a))goto done;
    a->cells=proof->field_count;
    if(proof->field_count) {
        proof->fields=calloc(proof->field_count,sizeof *proof->fields);
        if(!proof->fields) {mf_stop(a,NVM_MIXED_SHAPE_MEMORY,0,0,"I could not allocate field summaries.");goto done;}
    }
    for(uint32_t f=0;f<m->function_count;f++) {
        MFValue seed[MF_SLOTS*2]={0};const NvmFunctionEntry *fn=&m->functions[f];
        for(uint16_t n=0;n<fn->local_count;n++)seed[n]=mf_tag(TAG_VOID);
        for(uint16_t n=0;n<fn->arity;n++) {MFType t=a->functions[f].locals[n];seed[n]=t.tag==TAG_STRUCT?mf_owner(t.layout):mf_tag(t.tag);}
        if(!mf_join(a,a->functions[f].start,seed,0))goto done;
    }
    while(a->queued) {
        if(a->visits++==MF_VISITS) {mf_stop(a,NVM_MIXED_SHAPE_LIMIT,0,0,"I reached my shape fixed-point work budget.");goto done;}
        uint32_t index=a->queue[a->head];a->head=(a->head+1)%NVM_MIXED_FLOAT_INSTRUCTIONS;a->queued--;
        a->instructions[index].queued=false;if(!mf_step(a,index))goto done;
    }
    a->checking=true;
    for(uint32_t i=0;i<a->count;i++) {
        if(!a->instructions[i].state) {mf_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,a->instructions[i].function,a->instructions[i].pc,"I require checked state for every declared instruction.");goto done;}
        if(!mf_step(a,i))goto done;
    }
    proof->obligation_count=a->count;
    a->error=(NvmMixedShapeResult){NVM_MIXED_SHAPE_PROVED,0,0,"I proved closed managed shape only; scalar and affine admission remain separate."};
done:
    for(uint32_t i=0;i<a->count;i++)free(a->instructions[i].state);
    NvmMixedShapeResult result=a->error;free(a);
    if(result.status==NVM_MIXED_SHAPE_PROVED)*out=proof;else nvm_mixed_float_proof_free(proof);
    return result;
}
