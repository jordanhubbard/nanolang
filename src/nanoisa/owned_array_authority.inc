/* I live inside affine_state.c; my checked Facts constructor never escapes. */
#include "owned_array_authority_internal.h"
#define LA_FUNCS 8u
#define LA_SLOTS 256u
#define LA_INSTRUCTIONS 4096u
#define LA_CELLS 1048576u
#define LA_VISITS 262144u
#define LA_WORK 4194304u
#define LA_BIT(t) ((uint16_t)(1u<<(t)))
#define LA_SCALARS (LA_BIT(TAG_VOID)|LA_BIT(TAG_INT)|LA_BIT(TAG_U8)|LA_BIT(TAG_FLOAT)|LA_BIT(TAG_BOOL))
typedef struct { uint16_t tags, root; uint32_t layout; uint8_t kind, unknown; } LAValue;
typedef struct { NvmAffineState *owners; LAValue *values; uint16_t depth; } LAFrame;
typedef struct {
    DecodedInstruction in;
    uint32_t function, pc, next, target;
    LAFrame *frame;
    uint16_t operand_actual[2], operand_required[2];
    uint8_t operand_count, runtime_checks;
    bool queued;
} LAInstruction;
typedef struct { uint32_t start, end; NvmAffineState *seed; } LAFunction;
typedef struct {
    const NvmModule *module;
    NvmOwnerLifetimeFacts *proof;
    LAFunction functions[LA_FUNCS];
    LAInstruction instructions[LA_INSTRUCTIONS];
    uint32_t count,cells,visits,work;
    uint32_t queue[LA_INSTRUCTIONS],head,tail,pending;
    NvmOwnerAuthorityResult error;
} LAAnalysis;
static bool la_stop(LAAnalysis *a,NvmOwnerAuthorityStatus s,uint32_t f,uint32_t pc,const char *message) {
    a->error=(NvmOwnerAuthorityResult){s,f,pc,message};return false;
}
static bool la_work(LAAnalysis *a,uint32_t amount) {
    if(amount>LA_WORK-a->work)return la_stop(a,NVM_OWNER_AUTH_LIMIT,0,0,"I reached my independent owner work budget.");
    a->work+=amount;return true;
}
static LAValue la_tag(uint8_t tag) {LAValue v={0};v.tags=LA_BIT(tag);v.layout=NVM_V2_NO_INDEX;return v;}
static LAValue la_owner(uint32_t layout) {LAValue v=la_tag(TAG_STRUCT);v.kind=1;v.layout=layout;return v;}
static bool la_exact(LAValue v,uint8_t tag) {return !v.kind && !v.unknown && v.tags==LA_BIT(tag);}
static bool la_owner_layout(const Facts *facts,uint32_t layout) {
    return layout<facts->layouts.count &&
        (facts->flags[layout]&(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE))==(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE) &&
        facts->layouts.items[layout].kind==NVM_V2_LAYOUT_STRUCT;
}
static bool la_type(const Facts *facts,LAValue v,Slot type) {
    if(type.mode)return false;
    if(type.tag==TAG_STRUCT)return la_owner_layout(facts,type.layout) && !v.unknown &&
        v.kind==1 && v.tags==LA_BIT(TAG_STRUCT) && v.layout==type.layout;
    return type.layout==NVM_V2_NO_INDEX && la_exact(v,type.tag);
}
static NvmOwnerDeclaration la_declaration(const Facts *facts,Slot slot) {
    return (NvmOwnerDeclaration){slot.tag,slot.mode,slot.layout,slot.tag==TAG_STRUCT && la_owner_layout(facts,slot.layout)};
}
static bool la_signature(const Facts *facts,Slot slot,bool result) {
    if(slot.mode)return false;
    if(slot.tag==TAG_STRUCT)return la_owner_layout(facts,slot.layout);
    return slot.layout==NVM_V2_NO_INDEX && (slot.tag==TAG_INT || slot.tag==TAG_BOOL || slot.tag==TAG_U8 ||
        (result?slot.tag==TAG_VOID:slot.tag==TAG_STRING));
}
/* Both queries have just succeeded on these same immutable bytes. I preserve
 * all real tags, flags and indices; no ordinary local is rewritten to VOID. */
static NvmAffineState *la_checked_state(LAAnalysis *a,uint32_t function) {
    NvmAffineState *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    s->facts=calloc(1,sizeof *s->facts);
    if(!s->facts){free(s);return NULL;}
    Facts *facts=s->facts;facts->users=1;facts->module=a->module;facts->function=function;s->invocation=1;
    NvmOwnedArrayTransport transport;nvm_owned_array_transport(a->proof->layouts,&transport);
    if(nvm_v2_layouts_decode(transport.layouts,transport.layout_size,&facts->layouts)!=NVM_V2_OK)goto fail;
    NvmV2Cursor c;uint32_t count,ignored;const uint8_t *flags;
    nvm_v2_cursor_init(&c,transport.ownership,transport.ownership_size);
    if(nvm_v2_u32(&c,&ignored)!=NVM_V2_OK || nvm_v2_u32(&c,&count)!=NVM_V2_OK ||
       count!=facts->layouts.count || nvm_v2_take(&c,count,&flags)!=NVM_V2_OK)goto fail;
    facts->flags=malloc(count?count:1);if(!facts->flags)goto fail;memcpy(facts->flags,flags,count);
    if(nvm_v2_align4(&c)!=NVM_V2_OK || nvm_v2_u32(&c,&ignored)!=NVM_V2_OK)goto fail;
    for(uint32_t f=0;f<=function;f++) {
        uint16_t locals,params;Slot result;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK || !slot_read(&c,&result))goto fail;
        if(f==function) {
            facts->count=locals;facts->params=params;facts->result=result;
            facts->locals=calloc(locals?locals:1,sizeof *facts->locals);
            s->live=calloc(locals?locals:1,sizeof *s->live);s->refs=calloc(1,sizeof *s->refs);
            s->variants=malloc((locals?locals:1)*sizeof *s->variants);
            if(!facts->locals || !s->live || !s->variants || !s->refs)goto fail;
            for(uint16_t local=0;local<locals;local++)s->variants[local]=NVM_AFFINE_UNKNOWN_VARIANT;
        }
        for(uint16_t n=0;n<locals;n++) {
            Slot slot;if(!slot_read(&c,&slot))goto fail;
            if(f==function){facts->locals[n]=slot;s->live[n]=n<params;}
        }
    }
    return s;
fail:nvm_affine_state_free(s);return NULL;
}
static bool la_facts(LAAnalysis *a,uint32_t f) {
    const NvmFunctionEntry *fn=&a->module->functions[f];
    NvmAffineState *s=la_checked_state(a,f);
    if(!s)return la_stop(a,NVM_OWNER_AUTH_MEMORY,f,0,"I could not materialize checked owner Facts.");
    a->functions[f].seed=s;Facts *facts=s->facts;
    NvmOwnerSignature *signature=&a->proof->signatures[f];
    if(facts->count!=fn->local_count || facts->params!=fn->arity ||
       !la_signature(facts,facts->result,true) || facts->result.tag!=fn->result_tag ||
       fn->result_count!=(facts->result.tag==TAG_VOID?0:1))
        return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,0,"I retain exact existing owner value signatures.");
    for(uint32_t l=0;l<facts->layouts.count;l++)if(la_owner_layout(facts,l)) {
        const NvmV2Layout *layout=&facts->layouts.items[l];
        NvmOwnedArrayLayoutFact shape;
        if(!nvm_owned_array_layout_fact(a->proof->layouts,l,&shape))
            return la_stop(a,NVM_OWNER_AUTH_INVALID,f,0,"I require the exact independently retained layout row.");
        /* Descriptive depth counts edges; existing value-result depth counts
         * record levels, including the root. I preserve that stricter bound. */
        if(shape.owner_depth>=NVM_AFFINE_MAX_RESULT_DEPTH || layout->field_count>LA_SLOTS)
            return la_stop(a,NVM_OWNER_AUTH_LIMIT,f,0,"I reached my owner depth or direct-field limit.");
        if(!la_work(a,layout->field_count))return false;
        for(uint16_t n=0;n<layout->field_count;n++) {
            NvmV2LayoutField field=layout->fields[n];
            if(field.type_tag==TAG_STRUCT) {
                if(field.nested_idx>=l || !la_owner_layout(facts,field.nested_idx))
                    return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,0,"I require prior exact nested owners.");
            } else if(field.nested_idx!=NVM_V2_NO_INDEX || (field.type_tag!=TAG_INT && field.type_tag!=TAG_BOOL &&
                      field.type_tag!=TAG_U8 && field.type_tag!=TAG_STRING && field.type_tag!=TAG_ARRAY))
                return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,0,"I keep unsupported owner fields outside authority.");
        }
    }
    signature->local_count=facts->count;signature->parameter_count=facts->params;
    signature->result=la_declaration(facts,facts->result);
    if(signature->result.owner)signature->result_fields=facts->layouts.items[facts->result.layout].field_count;
    for(uint16_t n=0;n<facts->count;n++) {
        Slot slot=facts->locals[n];bool valid=n<facts->params?la_signature(facts,slot,false):
            !slot.mode && (slot.tag==TAG_STRUCT?la_owner_layout(facts,slot.layout):
                slot.layout==NVM_V2_NO_INDEX && (scalar(slot.tag) || slot.tag==TAG_STRING || slot.tag==TAG_ARRAY));
        if(!valid)return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,0,"I require modeled exact local declarations.");
        a->proof->locals[f][n]=la_declaration(facts,slot);
        if(n<facts->params)signature->parameters[n]=a->proof->locals[f][n];
    }
    return true;
}
static bool la_optional_comparison(uint8_t op) {
    switch(op) {
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:return true;
    default:return false;
    }
}
static int la_scalar(uint8_t op,uint16_t *required) {
    *required=0;
    switch(op) {
    case OP_PUSH_I64:return TAG_INT;case OP_PUSH_F64:return TAG_FLOAT;
    case OP_PUSH_U8:return TAG_U8;case OP_PUSH_BOOL:return TAG_BOOL;case OP_PUSH_VOID:return TAG_VOID;
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:case OP_I64_DIV_S:case OP_I64_REM_S:case OP_I64_NEG:
        *required=LA_BIT(TAG_INT);return TAG_INT;
    case OP_F64_ADD:case OP_F64_SUB:case OP_F64_MUL:case OP_F64_DIV:case OP_F64_NEG:
        *required=LA_BIT(TAG_FLOAT);return TAG_FLOAT;
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
        *required=LA_BIT(TAG_INT);return TAG_BOOL;
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:
        *required=LA_BIT(TAG_FLOAT);return TAG_BOOL;
    case OP_BOOL_AND:case OP_BOOL_OR:case OP_BOOL_NOT:*required=LA_BIT(TAG_BOOL);return TAG_BOOL;
    case OP_EQ:case OP_NE:return TAG_BOOL;
    default:return -1;
    }
}
static bool la_supported(uint8_t op) {
    uint16_t required;if(la_scalar(op,&required)>=0)return true;
    switch(op) {
    case OP_PUSH_STR:case OP_NOP:case OP_DUP:case OP_POP:case OP_SWAP:case OP_ROT3:
    case OP_LOAD_LOCAL:case OP_STORE_LOCAL:case OP_OWN_MOVE_LOCAL:case OP_OWN_STORE_LOCAL:
    case OP_OWN_PACK:case OP_OWN_UNPACK_LOCAL:case OP_AGG_GET:
    case OP_ARR_NEW:case OP_ARR_LITERAL:case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN:
    case OP_JMP:case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_CALL:case OP_RET:case OP_ASSERT:
    case OP_PRINT:case OP_PRINTLN:return true;
    default:return false;
    }
}
static bool la_decode(LAAnalysis *a) {
    NvmOwnerOriginCounts origins;nvm_owned_array_origin_counts(a->proof->origins,&origins);
    for(uint32_t f=0;f<a->module->function_count;f++) {
        if(!la_facts(a,f))return false;
        const NvmFunctionEntry *fn=&a->module->functions[f];LAFunction *af=&a->functions[f];af->start=a->count;
        for(uint32_t pc=0;pc<fn->code_length;) {
            if(a->count==LA_INSTRUCTIONS)return la_stop(a,NVM_OWNER_AUTH_LIMIT,f,pc,"I reached my independent instruction budget.");
            LAInstruction *i=&a->instructions[a->count];i->function=f;i->pc=pc;
            uint32_t width=isa_decode(a->module->code+fn->code_offset+pc,fn->code_length-pc,&i->in);
            if(!width || !la_supported(i->in.opcode))return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,pc,"I require an explicit independent instruction transfer.");
            NvmOwnerOriginObligation *row=&a->proof->obligations[a->count];
            if(!nvm_owned_array_origin_obligation(a->proof->origins,a->count,row) || row->function!=f || row->pc!=pc || row->opcode!=i->in.opcode)
                return la_stop(a,NVM_OWNER_AUTH_INVALID,f,pc,"I require identical independent and origin instruction locations.");
            a->count++;pc+=width;i->next=a->count;i->target=UINT32_MAX;
        }
        af->end=a->count;
    }
    if(a->count!=origins.obligations)return la_stop(a,NVM_OWNER_AUTH_INVALID,0,0,"I require complete independent instruction coverage.");
    for(uint32_t f=0;f<a->module->function_count;f++)for(uint32_t n=a->functions[f].start;n<a->functions[f].end;n++) {
        LAInstruction *i=&a->instructions[n];uint8_t op=i->in.opcode;
        if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
            int64_t target=(int64_t)i->pc+i->in.operands[0].i32;
            for(uint32_t t=a->functions[f].start;t<a->functions[f].end;t++)if(target==a->instructions[t].pc)i->target=t;
            if(i->target==UINT32_MAX)return la_stop(a,NVM_OWNER_AUTH_INVALID,f,i->pc,"I require exact independent branch destinations.");
        }
    }
    return true;
}
static void la_frame_free(LAFrame *f) {
    if(!f)return;
    nvm_affine_state_free(f->owners);free(f->values);free(f);
}
static LAFrame *la_clone(const LAFrame *from,uint16_t locals) {
    LAFrame *f=calloc(1,sizeof *f);if(!f)return NULL;
    f->owners=nvm_affine_state_clone(from->owners);f->values=calloc((uint32_t)locals+LA_SLOTS,sizeof *f->values);
    if(!f->owners || !f->values){la_frame_free(f);return NULL;}
    f->depth=from->depth;memcpy(f->values,from->values,((uint32_t)locals+from->depth)*sizeof *f->values);return f;
}
static void la_enqueue(LAAnalysis *a,uint32_t n) {
    if(a->instructions[n].queued)return;
    a->instructions[n].queued=true;a->queue[a->tail]=n;a->tail=(a->tail+1)%LA_INSTRUCTIONS;a->pending++;
}
static bool la_join(LAAnalysis *a,uint32_t n,const LAFrame *from) {
    if(n>=a->count)return la_stop(a,NVM_OWNER_AUTH_INVALID,0,0,"I require an existing independent state destination.");
    LAInstruction *i=&a->instructions[n];uint16_t locals=a->module->functions[i->function].local_count;
    if(!la_work(a,(uint32_t)locals*2+from->depth))return false;
    bool changed=false;
    if(!i->frame) {
        uint32_t cells=(uint32_t)locals+LA_SLOTS;
        if(cells>LA_CELLS-a->cells)return la_stop(a,NVM_OWNER_AUTH_LIMIT,i->function,i->pc,"I reached my independent state-cell budget.");
        i->frame=la_clone(from,locals);
        if(!i->frame)return la_stop(a,NVM_OWNER_AUTH_MEMORY,i->function,i->pc,"I could not allocate an independent join frame.");
        a->cells+=cells;changed=true;
    } else {
        LAFrame *to=i->frame;
        if(to->depth!=from->depth)return la_stop(a,NVM_OWNER_AUTH_INVALID,i->function,i->pc,"I require equal independent join heights.");
        for(uint32_t k=0;k<(uint32_t)locals+to->depth;k++) {
            LAValue x=to->values[k],y=from->values[k];
            if((x.kind || y.kind) && (x.kind!=y.kind || x.layout!=y.layout || x.root!=y.root || x.tags!=y.tags || x.unknown!=y.unknown))
                return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,i->function,i->pc,"I require compatible unique stack owners and observations.");
        }
        bool met=false;
        if(!nvm_affine_state_meet_initialization(to->owners,from->owners,&met))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,i->function,i->pc,"I require compatible independent owner lifetimes at joins.");
        changed|=met;
        for(uint32_t k=0;k<(uint32_t)locals+to->depth;k++) {
            uint16_t tags=to->values[k].tags;uint8_t unknown=to->values[k].unknown;
            to->values[k].tags|=from->values[k].tags;to->values[k].unknown|=from->values[k].unknown;
            changed|=tags!=to->values[k].tags || unknown!=to->values[k].unknown;
        }
    }
    if(changed)la_enqueue(a,n);
    return true;
}
static bool la_observed(const LAFrame *frame,uint16_t locals,uint16_t root) {
    for(uint16_t n=0;n<frame->depth;n++)if(frame->values[locals+n].kind==2 && frame->values[locals+n].root==root)return true;
    return false;
}
static bool la_transfer(LAAnalysis *a,uint32_t index,LAFrame *frame) {
    LAInstruction *i=&a->instructions[index];uint32_t f=i->function;
    NvmAffineState *owners=frame->owners;Facts *facts=owners->facts;
    uint16_t locals=facts->count;LAValue *stack=frame->values+locals;
    DecodedInstruction *in=&i->in;uint8_t op=in->opcode;const InstructionInfo *info=isa_get_info(op);
    if(!info)return la_stop(a,NVM_OWNER_AUTH_INVALID,f,i->pc,"I require complete instruction metadata.");
    int pops=info->pop_count,pushes=info->push_count;
    NvmAffineType fields[LA_SLOTS];uint16_t field_count=0,local=in->operands[0].u16;
    if(op==OP_ARR_LITERAL){pops=in->operands[1].u16;pushes=1;}
    if(op==OP_OWN_PACK || op==OP_OWN_UNPACK_LOCAL) {
        uint32_t layout=op==OP_OWN_PACK?in->operands[0].u32:facts->locals[local].layout;
        if(!la_owner_layout(facts,layout) || !nvm_affine_record_fields(owners,layout,fields,LA_SLOTS,&field_count))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact owner authority before field transfer.");
        if(!la_work(a,field_count))return false;
        pops=op==OP_OWN_PACK?field_count:0;pushes=op==OP_OWN_PACK?1:field_count;
    }
    if(op==OP_CALL) {const NvmFunctionEntry *callee=&a->module->functions[in->operands[0].u32];pops=callee->arity;pushes=callee->result_count;}
    if(op==OP_RET) {
        if(!la_work(a,locals))return false;
        if(frame->depth!=a->module->functions[f].result_count ||
           (frame->depth && !la_type(facts,stack[0],facts->result)) ||
           !nvm_affine_can_exit_type(owners,(NvmAffineType){facts->result.tag,facts->result.layout}))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require the exact return and complete independent owner consumption.");
        return true;
    }
    if(pops<0 || pushes<0 || frame->depth<pops)return la_stop(a,NVM_OWNER_AUTH_INVALID,f,i->pc,"I require complete independent operand stacks.");
    if((uint32_t)(frame->depth-pops+pushes)>LA_SLOTS)return la_stop(a,NVM_OWNER_AUTH_LIMIT,f,i->pc,"I reached my independent stack budget.");
    uint16_t base=(uint16_t)(frame->depth-pops);LAValue result={0};uint16_t required;
    int produced=la_scalar(op,&required);
    if(produced>=0) {
        result=la_tag((uint8_t)produced);
        if(pops>2)return la_stop(a,NVM_OWNER_AUTH_INVALID,f,i->pc,"I require bounded independent scalar operand rows.");
        i->operand_count=(uint8_t)pops;
        for(uint16_t n=base;n<frame->depth;n++) {
            LAValue v=stack[n];
            if(v.kind || v.unknown || !v.tags || (v.tags&~LA_SCALARS))
                return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require independently modeled scalar operands.");
            uint16_t operand=(uint16_t)(n-base);
            i->operand_actual[operand]|=v.tags;i->operand_required[operand]=required;
            if(required && v.tags!=required) {
                if(!la_optional_comparison(op) || !(v.tags&LA_BIT(TAG_FLOAT)) ||
                   (v.tags&~(LA_BIT(TAG_FLOAT)|LA_BIT(TAG_VOID))))
                    return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require independently exact operands or a checked optional FLOAT comparison.");
                i->runtime_checks|=(uint8_t)(1u<<operand);
            }
        }
    } else switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_PUSH_STR:result=la_tag(TAG_STRING);break;
    case OP_DUP:
        if(stack[frame->depth-1].kind)return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I cannot copy a unique owner or observation.");
        stack[frame->depth]=stack[frame->depth-1];frame->depth++;goto successors;
    case OP_SWAP:result=stack[frame->depth-1];stack[frame->depth-1]=stack[frame->depth-2];stack[frame->depth-2]=result;goto successors;
    case OP_ROT3:result=stack[frame->depth-1];stack[frame->depth-1]=stack[frame->depth-2];stack[frame->depth-2]=stack[frame->depth-3];stack[frame->depth-3]=result;goto successors;
    case OP_POP:
        if(stack[base].kind)return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I cannot discard an unconsumed owner or observation.");
        break;
    case OP_LOAD_LOCAL: {
        Slot slot=facts->locals[local];
        if(slot.tag==TAG_STRUCT) {
            if(!la_owner_layout(facts,slot.layout) || !nvm_affine_owner_access(owners,local,NULL,0,false))
                return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require a live owner before observation.");
            result=la_owner(slot.layout);result.kind=2;result.root=local;
        } else {
            result=frame->values[local];
            if(!la_type(facts,result,slot) || ((scalar(slot.tag) || slot.tag==TAG_STRING) && !owners->live[local]))
                return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require independently initialized ordinary locals.");
        }
        break;
    }
    case OP_STORE_LOCAL: {
        Slot slot=facts->locals[local];result=stack[base];
        if(slot.tag==TAG_STRUCT || result.kind || !la_type(facts,result,slot))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I cannot overwrite owner authority through an ordinary store.");
        if(scalar(slot.tag) && !nvm_affine_scalar_define(owners,local))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require a checked scalar initialization.");
        if(slot.tag==TAG_STRING && !nvm_affine_string_define(owners,local))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require a checked STRING initialization.");
        /* ARRAY initialization belongs to the ordinary value lattice, not the
         * unique-owner live bitmap. No owned fact is replaced or cleared. */
        frame->values[local]=result;break;
    }
    case OP_OWN_MOVE_LOCAL:case OP_OWN_UNPACK_LOCAL: {
        NvmAffineType type;
        if(!la_work(a,frame->depth))return false;
        if(!la_owner_layout(facts,facts->locals[local].layout) || la_observed(frame,locals,local) ||
           !nvm_affine_take_local(owners,local,&type))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require a live unobserved owner before consuming it.");
        if(op==OP_OWN_MOVE_LOCAL)result=la_owner(type.layout);
        else {
            for(uint16_t n=0;n<field_count;n++)
                stack[frame->depth++]=fields[n].tag==TAG_STRUCT?la_owner(fields[n].layout):la_tag(fields[n].tag);
            goto successors;
        }
        break;
    }
    case OP_OWN_STORE_LOCAL:
        if(facts->locals[local].tag!=TAG_STRUCT || !la_type(facts,stack[base],facts->locals[local]) ||
           !nvm_affine_put_local(owners,local,(NvmAffineType){TAG_STRUCT,stack[base].layout}))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require an empty exact owner destination.");
        break;
    case OP_OWN_PACK:
        for(uint16_t n=0;n<field_count;n++)if(!la_type(facts,stack[base+n],(Slot){fields[n].tag,0,fields[n].layout}))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact transferred owner constructor fields.");
        result=la_owner(in->operands[0].u32);break;
    case OP_AGG_GET: {
        LAValue receiver=stack[base];uint16_t field=in->operands[0].u16;
        if(receiver.kind!=2 || receiver.unknown || receiver.root>=locals || !owners->live[receiver.root] ||
           !la_owner_layout(facts,receiver.layout) || field>=facts->layouts.items[receiver.layout].field_count)
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require an exact live owner observation path.");
        NvmV2LayoutField declared=facts->layouts.items[receiver.layout].fields[field];
        result=declared.type_tag==TAG_STRUCT?la_owner(declared.nested_idx):la_tag(declared.type_tag);
        if(result.kind){result.kind=2;result.root=receiver.root;}
        break;
    }
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        if(in->operands[0].u8!=TAG_FLOAT)return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require declared FLOAT array construction.");
        for(uint16_t n=base;n<frame->depth;n++)if(!la_exact(stack[n],TAG_FLOAT))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact FLOAT initializer values.");
        result=la_tag(TAG_ARRAY);break;
    case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN:
        if(!la_exact(stack[base],TAG_ARRAY))return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require an independently initialized ARRAY receiver.");
        if((op==OP_ARR_GET || op==OP_ARR_SET) && !la_exact(stack[base+1],TAG_INT))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact INT indices.");
        if((op==OP_ARR_PUSH || op==OP_ARR_SET) && !la_exact(stack[frame->depth-1],TAG_FLOAT))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact FLOAT writes through aliases.");
        if(op==OP_ARR_GET){result=la_tag(TAG_FLOAT);result.tags|=LA_BIT(TAG_VOID);}
        else result=la_tag(op==OP_ARR_LEN?TAG_INT:TAG_ARRAY);
        break;
    case OP_CALL: {
        uint32_t callee=in->operands[0].u32;Facts *called=a->functions[callee].seed->facts;
        for(uint16_t n=0;n<called->params;n++)if(!la_type(facts,stack[base+n],called->locals[n]))
            return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact independently transferred call arguments.");
        result=called->result.tag==TAG_STRUCT?la_owner(called->result.layout):la_tag(called->result.tag);break;
    }
    case OP_PRINT:case OP_PRINTLN:
        i->operand_count=1;i->operand_actual[0]|=stack[base].tags;
        i->operand_required[0]=LA_BIT(TAG_INT);
        if(!la_exact(stack[base],TAG_INT))return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require independently exact INT output operands.");
        break;
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        i->operand_count=1;i->operand_actual[0]|=stack[base].tags;
        i->operand_required[0]=LA_BIT(TAG_BOOL);
        if(!la_exact(stack[base],TAG_BOOL))return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require exact BOOL control operands.");
        break;
    default:return la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,f,i->pc,"I require an explicit independent lifetime transfer.");
    }
    frame->depth=(uint16_t)(base+pushes);if(pushes)stack[base]=result;
successors:
    if(frame->depth>a->proof->signatures[f].max_stack)a->proof->signatures[f].max_stack=frame->depth;
    if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
        if(!la_join(a,i->target,frame))return false;
        if(op==OP_JMP)return true;
    }
    return la_join(a,i->next,frame);
}
void nvm_owned_array_lifetime_facts_free(NvmOwnerLifetimeFacts *p) {
    if(!p)return;
    nvm_owned_array_origins_free(p->origins);nvm_owned_array_layouts_free(p->layouts);free(p);
}
static NvmOwnerAuthorityStatus la_origin_status(NvmOwnerOriginStatus status) {
    switch(status) {
    case NVM_OWNER_ORIGIN_MEMORY:return NVM_OWNER_AUTH_MEMORY;
    case NVM_OWNER_ORIGIN_LIMIT:return NVM_OWNER_AUTH_LIMIT;
    case NVM_OWNER_ORIGIN_INVALID:return NVM_OWNER_AUTH_INVALID;
    default:return NVM_OWNER_AUTH_UNRESOLVED;
    }
}
NvmOwnerAuthorityResult nvm_analyze_owned_array_lifetimes(const NvmModule *m,NvmOwnerLifetimeFacts **out) {
    NvmOwnerAuthorityResult invalid={NVM_OWNER_AUTH_INVALID,0,0,"I require a complete immutable module and private output."};
    if(!out)return invalid;
    NvmOwnedArrayOrigins *origins=NULL;NvmOwnerOriginResult origin=nvm_analyze_owned_array_origins(m,&origins);
    if(origin.status!=NVM_OWNER_ORIGIN_PROVED)return (NvmOwnerAuthorityResult){la_origin_status(origin.status),origin.function,origin.pc,origin.message};
    NvmOwnedArrayLayouts *layouts=NULL;NvmRecordPlanResult described=nvm_describe_owned_array_layouts(m,&layouts);
    if(described.status!=NVM_RECORD_DESCRIBED) {
        nvm_owned_array_origins_free(origins);
        NvmOwnerAuthorityStatus status=described.status==NVM_RECORD_MEMORY?NVM_OWNER_AUTH_MEMORY:
            described.status==NVM_RECORD_LIMIT?NVM_OWNER_AUTH_LIMIT:
            described.status==NVM_RECORD_INVALID?NVM_OWNER_AUTH_INVALID:NVM_OWNER_AUTH_UNRESOLVED;
        return (NvmOwnerAuthorityResult){status,0,0,described.message};
    }
    LAAnalysis *a=calloc(1,sizeof *a);NvmOwnerLifetimeFacts *p=calloc(1,sizeof *p);
    if(!a || !p) {free(a);free(p);nvm_owned_array_origins_free(origins);nvm_owned_array_layouts_free(layouts);
        return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_MEMORY,0,0,"I could not allocate independent owner analysis."};}
    p->origins=origins;p->layouts=layouts;a->proof=p;a->module=m;a->error=invalid;
    if(!la_decode(a))goto done;
    for(uint32_t f=0;f<m->function_count;f++) {
        uint16_t locals=m->functions[f].local_count;LAValue values[LA_SLOTS*2]={0};
        LAFrame seed={a->functions[f].seed,values,0};
        for(uint16_t n=0;n<locals;n++){values[n]=la_tag(TAG_VOID);values[n].unknown=1;}
        for(uint16_t n=0;n<m->functions[f].arity;n++)if(seed.owners->facts->locals[n].tag!=TAG_STRUCT)
            values[n]=la_tag(seed.owners->facts->locals[n].tag);
        if(!la_join(a,a->functions[f].start,&seed))goto done;
    }
    while(a->pending) {
        if(a->visits++==LA_VISITS){la_stop(a,NVM_OWNER_AUTH_LIMIT,0,0,"I reached my independent transfer visit budget.");goto done;}
        uint32_t n=a->queue[a->head];a->head=(a->head+1)%LA_INSTRUCTIONS;a->pending--;
        LAInstruction *i=&a->instructions[n];i->queued=false;
        uint16_t locals=m->functions[i->function].local_count;
        if(!la_work(a,(uint32_t)locals*2+i->frame->depth))goto done;
        LAFrame *frame=la_clone(i->frame,locals);
        if(!frame){la_stop(a,NVM_OWNER_AUTH_MEMORY,i->function,i->pc,"I could not copy independent transfer state.");goto done;}
        bool ok=la_transfer(a,n,frame);la_frame_free(frame);if(!ok)goto done;
    }
    for(uint32_t n=0;n<a->count;n++)if(!a->instructions[n].frame) {
        la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,a->instructions[n].function,a->instructions[n].pc,"I require independent state for every declared instruction.");goto done;
    }
    for(uint32_t n=0;n<a->count;n++) {
        LAInstruction *i=&a->instructions[n];const NvmOwnerOriginObligation *row=&p->obligations[n];
        if(!la_work(a,2))goto done;
        if(row->operand_count!=i->operand_count || row->runtime_checks!=i->runtime_checks ||
           row->operand_actual[0]!=i->operand_actual[0] || row->operand_actual[1]!=i->operand_actual[1] ||
           row->operand_required[0]!=i->operand_required[0] || row->operand_required[1]!=i->operand_required[1]) {
            la_stop(a,NVM_OWNER_AUTH_UNRESOLVED,i->function,i->pc,"I require independently matching per-operand obligations.");goto done;
        }
    }
    p->counts=(NvmOwnerAuthorityCounts){m->function_count,a->count,a->cells,a->visits,a->work};
    a->error=(NvmOwnerAuthorityResult){NVM_OWNER_AUTH_PREPARED,0,0,"I checked origin, lifetime and scalar transfers; common structure and runtime admission remain separate."};
done:
    for(uint32_t n=0;n<a->count;n++)la_frame_free(a->instructions[n].frame);
    for(uint32_t f=0;f<m->function_count;f++)nvm_affine_state_free(a->functions[f].seed);
    NvmOwnerAuthorityResult result=a->error;free(a);
    if(result.status==NVM_OWNER_AUTH_PREPARED)*out=p;else nvm_owned_array_lifetime_facts_free(p);
    return result;
}
