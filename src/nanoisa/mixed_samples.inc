/* I live inside affine_state.c: my checked constructor and Facts never escape. */
#include "mixed_samples.h"
#define MC_FUNCS 8u
#define MC_SLOTS 256u
#define MC_CELLS 1048576u
#define MC_VISITS 262144u
#define MC_BIT(t) ((uint16_t)(1u << (t)))
#define MC_FLOAT_VOID (MC_BIT(TAG_FLOAT)|MC_BIT(TAG_VOID))
typedef struct {
    uint16_t tags, root;
    uint64_t origins;
    uint32_t nominal;
    uint8_t kind; /* 0 scalar/ordinary, 1 owner token, 2 owner observation. */
} MCValue;
typedef struct {
    NvmAffineState *owners;
    MCValue *values; /* Original local indices, then operand stack. */
    uint16_t depth;
} MCFrame;
typedef struct {
    DecodedInstruction in;
    uint32_t function, pc, next, target;
    int16_t origin;
    MCFrame *frame;
    bool queued;
} MCInstruction;
typedef struct {
    uint32_t start, end;
    NvmAffineState *seed;
    uint8_t calls, color;
} MCFunction;
typedef struct {
    const NvmModule *module;
    NvmMixedSamplesProof *proof;
    MCFunction functions[MC_FUNCS];
    MCInstruction instructions[NVM_MIXED_FLOAT_INSTRUCTIONS];
    uint32_t count, cells, visits;
    uint32_t queue[NVM_MIXED_FLOAT_INSTRUCTIONS], head, tail, pending;
    NvmMixedShapeResult error;
} MCAnalysis;
static bool mc_stop(MCAnalysis *a,NvmMixedShapeStatus status,uint32_t f,uint32_t pc,const char *message) {
    a->error=(NvmMixedShapeResult){status,f,pc,message};return false;
}
static MCValue mc_tag(uint8_t tag) { MCValue v={0};v.tags=MC_BIT(tag);return v; }
static MCValue mc_owner(uint32_t layout) { MCValue v=mc_tag(TAG_STRUCT);v.kind=1;v.nominal=layout;return v; }
static bool mc_exact(MCValue v,uint8_t tag) { return !v.kind && !v.origins && v.tags==MC_BIT(tag); }
static bool mc_plain(MCValue v) {
    const uint16_t tags=MC_FLOAT_VOID|MC_BIT(TAG_INT)|MC_BIT(TAG_BOOL)|MC_BIT(TAG_U8);
    return !v.kind && !v.origins && v.tags && !(v.tags&~tags);
}
static bool mc_owned_layout(const MCAnalysis *a,uint32_t layout) {
    const NvmMixedLayoutView *v=a->proof->shape->view;
    if(layout>=v->layouts.count || v->classes[layout]!=NVM_MIXED_RESOURCE)return false;
    const NvmV2Layout *l=&v->layouts.items[layout];
    if(l->kind!=NVM_V2_LAYOUT_STRUCT || l->field_count>MC_SLOTS)return false;
    for(uint16_t f=0;f<l->field_count;f++) {
        const NvmV2LayoutField *field=&l->fields[f];
        if((field->type_tag!=TAG_INT && field->type_tag!=TAG_BOOL && field->type_tag!=TAG_U8) ||
           field->nested_idx!=NVM_V2_NO_INDEX)return false;
    }
    return true;
}
static bool mc_owned_local(const MCAnalysis *a,const NvmAffineState *s,uint16_t local) {
    return local<s->facts->count && !s->facts->locals[local].mode &&
        s->facts->locals[local].tag==TAG_STRUCT && mc_owned_layout(a,s->facts->locals[local].layout);
}
static bool mc_signature(const MCAnalysis *a,Slot slot,bool result) {
    if(slot.mode)return false;
    if(slot.tag==TAG_STRUCT)return mc_owned_layout(a,slot.layout);
    return slot.layout==NVM_V2_NO_INDEX && (slot.tag==TAG_INT || slot.tag==TAG_BOOL ||
        slot.tag==TAG_U8 || (result && slot.tag==TAG_VOID));
}
/* The caller has freshly built shape/view facts from this same immutable module.
 * I retain real descriptors; ordinary locals are not changed to scalar/VOID. */
static NvmAffineState *mc_checked_state(MCAnalysis *a,uint32_t function) {
    const NvmMixedLayoutView *v=a->proof->shape->view;
    NvmAffineState *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    s->facts=calloc(1,sizeof *s->facts);
    if(!s->facts){free(s);return NULL;}
    Facts *f=s->facts;f->users=1;f->module=a->module;f->function=function;s->invocation=1;
    if(nvm_v2_layouts_decode(a->module->layout_data,a->module->layout_size,&f->layouts)!=NVM_V2_OK)goto fail;
    f->flags=malloc(v->layouts.count?v->layouts.count:1);
    if(!f->flags)goto fail;
    memcpy(f->flags,v->ownership+8,v->layouts.count);
    NvmV2Cursor c;uint32_t ignored,count;const uint8_t *flags;
    nvm_v2_cursor_init(&c,v->ownership,v->ownership_size);
    if(nvm_v2_u32(&c,&ignored)!=NVM_V2_OK || nvm_v2_u32(&c,&count)!=NVM_V2_OK ||
       nvm_v2_take(&c,count,&flags)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&ignored)!=NVM_V2_OK)goto fail;
    for(uint32_t n=0;n<=function;n++) {
        uint16_t locals,params;Slot result;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK || !slot_read(&c,&result))goto fail;
        if(n==function) {
            f->count=locals;f->params=params;f->result=result;
            f->locals=calloc(locals?locals:1,sizeof *f->locals);
            s->live=calloc(locals?locals:1,sizeof *s->live);
            s->variants=malloc((locals?locals:1)*sizeof *s->variants);
            /* No references are admitted, but clones retain their normal storage contract. */
            s->refs=calloc(1,sizeof *s->refs);
            if(!f->locals || !s->live || !s->variants || !s->refs)goto fail;
            for(uint16_t local=0;local<locals;local++)s->variants[local]=NVM_AFFINE_UNKNOWN_VARIANT;
        }
        for(uint16_t local=0;local<locals;local++) {
            Slot slot;if(!slot_read(&c,&slot))goto fail;
            if(n==function){f->locals[local]=slot;s->live[local]=local<params;}
        }
    }
    return s;
fail:nvm_affine_state_free(s);return NULL;
}
static bool mc_declarations(MCAnalysis *a,uint32_t f) {
    const NvmFunctionEntry *fn=&a->module->functions[f];
    NvmAffineState *s=mc_checked_state(a,f);
    if(!s)return mc_stop(a,NVM_MIXED_SHAPE_MEMORY,f,0,"I could not materialize checked mixed facts.");
    a->functions[f].seed=s;
    if(s->facts->count!=fn->local_count || s->facts->params!=fn->arity ||
       !mc_signature(a,s->facts->result,true) || s->facts->result.tag!=fn->result_tag ||
       fn->result_count!=(s->facts->result.tag==TAG_VOID?0:1))
        return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I require exact mixed value signatures.");
    for(uint16_t n=0;n<fn->local_count;n++) {
        Slot slot=s->facts->locals[n];bool valid=false;
        if(n<fn->arity)valid=mc_signature(a,slot,false);
        else if(!slot.mode) {
            if(slot.tag==TAG_STRUCT) {
                const NvmMixedLayoutView *v=a->proof->shape->view;
                valid=slot.layout<v->layouts.count && (mc_owned_layout(a,slot.layout) ||
                    v->classes[slot.layout]==NVM_MIXED_ORDINARY_STRUCTURAL ||
                    v->classes[slot.layout]==NVM_MIXED_PENDING_ARRAY_PROOF);
            } else valid=slot.layout==NVM_V2_NO_INDEX && (scalar(slot.tag) || slot.tag==TAG_ARRAY);
        }
        if(!valid)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I keep unavailable mixed local categories explicit.");
    }
    return true;
}
static void mc_frame_free(MCFrame *f) { if(f){nvm_affine_state_free(f->owners);free(f->values);free(f);} }
static MCFrame *mc_frame_clone(const MCFrame *f,uint16_t locals) {
    MCFrame *out=calloc(1,sizeof *out);if(!out)return NULL;
    out->owners=nvm_affine_state_clone(f->owners);
    out->values=calloc((uint32_t)locals+MC_SLOTS,sizeof *out->values);
    if(!out->owners || !out->values){mc_frame_free(out);return NULL;}
    out->depth=f->depth;memcpy(out->values,f->values,((uint32_t)locals+f->depth)*sizeof *out->values);
    return out;
}
static void mc_enqueue(MCAnalysis *a,uint32_t index) {
    MCInstruction *i=&a->instructions[index];if(i->queued)return;
    i->queued=true;a->queue[a->tail]=index;a->tail=(a->tail+1)%NVM_MIXED_FLOAT_INSTRUCTIONS;a->pending++;
}
static bool mc_join(MCAnalysis *a,uint32_t index,const MCFrame *from) {
    MCInstruction *i=&a->instructions[index];uint16_t locals=a->module->functions[i->function].local_count;
    bool changed=false;
    if(!i->frame) {
        uint32_t cells=(uint32_t)locals+MC_SLOTS;
        if(cells>MC_CELLS-a->cells)return mc_stop(a,NVM_MIXED_SHAPE_LIMIT,i->function,i->pc,"I reached my composed state budget.");
        i->frame=mc_frame_clone(from,locals);
        if(!i->frame)return mc_stop(a,NVM_MIXED_SHAPE_MEMORY,i->function,i->pc,"I could not allocate a composed edge.");
        a->cells+=cells;changed=true;
    } else {
        MCFrame *to=i->frame;
        if(to->depth!=from->depth)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,i->function,i->pc,"I require equal composed stack heights.");
        /* Validate every owner/observation edge before changing any ordinary facts. */
        for(uint32_t n=0;n<(uint32_t)locals+to->depth;n++) {
            MCValue x=to->values[n],y=from->values[n];
            if((x.kind || y.kind) && (x.kind!=y.kind || x.tags!=y.tags || x.nominal!=y.nominal || x.root!=y.root))
                return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,i->function,i->pc,"I require exact owner stack and observation joins.");
        }
        if(!nvm_affine_state_meet_initialization(to->owners,from->owners,&changed))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,i->function,i->pc,"I require exact owner state at composed joins.");
        for(uint32_t n=0;n<(uint32_t)locals+to->depth;n++) {
            MCValue *x=&to->values[n],y=from->values[n];uint16_t tags=x->tags;uint64_t origins=x->origins;
            x->tags|=y.tags;x->origins|=y.origins;changed|=tags!=x->tags || origins!=x->origins;
        }
    }
    if(changed)mc_enqueue(a,index);
    return true;
}
static int mc_scalar_op(uint8_t op,uint8_t *required) {
    *required=TAG_COUNT;
    switch(op) {
    case OP_PUSH_I64:return TAG_INT;case OP_PUSH_F64:return TAG_FLOAT;
    case OP_PUSH_U8:return TAG_U8;case OP_PUSH_BOOL:return TAG_BOOL;case OP_PUSH_VOID:return TAG_VOID;
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:case OP_I64_DIV_S:case OP_I64_REM_S:case OP_I64_NEG:*required=TAG_INT;return TAG_INT;
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:*required=TAG_INT;return TAG_BOOL;
    case OP_F64_ADD:case OP_F64_SUB:case OP_F64_MUL:case OP_F64_DIV:case OP_F64_NEG:*required=TAG_FLOAT;return TAG_FLOAT;
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:*required=TAG_FLOAT;return TAG_BOOL;
    case OP_BOOL_AND:case OP_BOOL_OR:case OP_BOOL_NOT:*required=TAG_BOOL;return TAG_BOOL;
    default:return -1;
    }
}
static bool mc_supported(uint8_t op) {
    uint8_t required;if(mc_scalar_op(op,&required)>=0)return true;
    switch(op) {
    case OP_NOP:case OP_DUP:case OP_POP:case OP_SWAP:case OP_ROT3:
    case OP_EQ:case OP_NE:case OP_LT:case OP_LE:case OP_GT:case OP_GE:
    case OP_LOAD_LOCAL:case OP_STORE_LOCAL:case OP_OWN_MOVE_LOCAL:case OP_OWN_STORE_LOCAL:
    case OP_OWN_PACK:case OP_OWN_UNPACK_LOCAL:case OP_AGG_PACK:case OP_AGG_GET:
    case OP_ARR_NEW:case OP_ARR_LITERAL:case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN:
    case OP_JMP:case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_CALL:case OP_RET:case OP_ASSERT:return true;
    default:return false;
    }
}
static bool mc_graph(MCAnalysis *a,uint32_t f) {
    MCFunction *fn=&a->functions[f];if(fn->color==2)return true;
    if(fn->color==1)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,0,"I require acyclic composed calls.");
    fn->color=1;for(uint32_t n=0;n<a->module->function_count;n++)if((fn->calls&(1u<<n)) && !mc_graph(a,n))return false;
    fn->color=2;return true;
}
static bool mc_preflight(MCAnalysis *a) {
    const NvmModule *m=a->module;const NvmMixedLayoutView *view=a->proof->shape->view;
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];MCFunction *af=&a->functions[f];af->start=a->count;
        if(!fn->code_length || fn->code_offset>m->code_size || fn->code_length>m->code_size-fn->code_offset ||
           fn->name_idx>=m->string_count || fn->result_tag>=TAG_COUNT || !mc_declarations(a,f))return false;
        for(uint32_t prior=0;prior<f;prior++) {
            const NvmFunctionEntry *other=&m->functions[prior];
            if(fn->code_offset<(uint64_t)other->code_offset+other->code_length && other->code_offset<(uint64_t)fn->code_offset+fn->code_length)
                return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,0,"I require distinct composed function ranges.");
        }
        for(uint32_t pc=0;pc<fn->code_length;) {
            if(a->count==NVM_MIXED_FLOAT_INSTRUCTIONS)return mc_stop(a,NVM_MIXED_SHAPE_LIMIT,f,pc,"I reached my composed instruction bound.");
            MCInstruction *i=&a->instructions[a->count++];i->function=f;i->pc=pc;i->origin=-1;i->target=UINT32_MAX;
            uint32_t width=isa_decode(m->code+fn->code_offset+pc,fn->code_length-pc,&i->in);
            if(!width)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require a complete composed instruction.");
            uint8_t op=i->in.opcode;
            if(!mc_supported(op))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I require an explicit independent composed transfer.");
            if(op==OP_PUSH_BOOL && i->in.operands[0].u8>1)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require a Boolean literal.");
            if((op==OP_LOAD_LOCAL || op==OP_STORE_LOCAL || op==OP_OWN_MOVE_LOCAL || op==OP_OWN_STORE_LOCAL || op==OP_OWN_UNPACK_LOCAL) &&
               i->in.operands[0].u16>=fn->local_count)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require an existing composed local.");
            if(op==OP_CALL) {
                uint32_t target=i->in.operands[0].u32;
                if(!target || target>=m->function_count)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I require an existing nonentry callee.");
                af->calls|=(uint8_t)(1u<<target);
            }
            if(op==OP_OWN_PACK && !mc_owned_layout(a,i->in.operands[0].u32))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I refuse ordinary or unavailable owner construction.");
            if(op==OP_AGG_PACK || op==OP_ARR_NEW || op==OP_ARR_LITERAL) {
                for(uint32_t n=0;n<a->proof->shape->origin_count;n++) {
                    const NvmMixedFloatOrigin *o=&a->proof->shape->origins[n];
                    if(o->function==f && o->pc==pc)i->origin=(int16_t)n;
                }
                if(i->origin<0)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I require this exact closed allocation origin.");
                if(op==OP_AGG_PACK) {
                    uint32_t source=i->in.operands[1].u32;
                    if(i->in.operands[0].u8 || i->in.operands[2].u16 || source>=view->record_count ||
                       view->source_to_global[source]!=a->proof->shape->origins[i->origin].global_layout ||
                       i->in.operands[3].u16!=a->proof->shape->origins[i->origin].field_count)
                        return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,pc,"I preserve constructor source and global identities.");
                } else if(i->in.operands[0].u8!=TAG_FLOAT)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,pc,"I require packed FLOAT array origins.");
            }
            pc+=width;i->next=a->count;
        }
        af->end=a->count;
    }
    for(uint32_t f=0;f<m->function_count;f++) {
        MCFunction *fn=&a->functions[f];
        for(uint32_t n=fn->start;n<fn->end;n++) {
            MCInstruction *i=&a->instructions[n];uint8_t op=i->in.opcode;
            if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
                int64_t target=(int64_t)i->pc+i->in.operands[0].i32;
                for(uint32_t j=fn->start;j<fn->end;j++)if(target==a->instructions[j].pc)i->target=j;
                if(i->target==UINT32_MAX)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,i->pc,"I require an explicit instruction-boundary branch.");
            }
            if(i->next==fn->end && op!=OP_RET && op!=OP_JMP)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require explicit composed termination.");
        }
        if(!mc_graph(a,f))return false;
    }
    return true;
}
static bool mc_array(const MCAnalysis *a,MCValue v) {
    if(v.kind || v.tags!=MC_BIT(TAG_ARRAY) || !v.origins)return false;
    for(uint32_t n=0;n<a->proof->shape->origin_count;n++)if((v.origins&(UINT64_C(1)<<n)) && a->proof->shape->origins[n].tag!=TAG_ARRAY)return false;
    return true;
}
static bool mc_field(const MCAnalysis *a,MCValue v,uint8_t tag,uint32_t layout) {
    if(v.kind || v.tags!=MC_BIT(tag))return false;
    if(tag==TAG_ARRAY)return layout==NVM_V2_NO_INDEX && mc_array(a,v);
    if(tag!=TAG_STRUCT)return layout==NVM_V2_NO_INDEX && mc_plain(v);
    if(!v.origins)return false;
    for(uint32_t n=0;n<a->proof->shape->origin_count;n++)if(v.origins&(UINT64_C(1)<<n)) {
        const NvmMixedFloatOrigin *o=&a->proof->shape->origins[n];
        if(o->tag!=TAG_STRUCT || o->global_layout!=layout)return false;
    }
    return true;
}
static bool mc_type(const MCAnalysis *a,MCValue v,Slot type) {
    if(type.mode)return false;
    if(type.tag==TAG_STRUCT)return mc_owned_layout(a,type.layout) && v.kind==1 &&
        v.tags==MC_BIT(TAG_STRUCT) && !v.origins && v.nominal==type.layout;
    return mc_exact(v,type.tag) && type.layout==NVM_V2_NO_INDEX;
}
static bool mc_observed(const MCFrame *f,uint16_t locals,uint16_t root) {
    for(uint16_t n=0;n<f->depth;n++)if(f->values[locals+n].kind==2 && f->values[locals+n].root==root)return true;
    return false;
}
static bool mc_check(MCAnalysis *a,const MCInstruction *i,uint16_t operand,uint16_t actual,uint16_t required,NvmMixedScalarPolicy policy) {
    for(uint32_t n=0;n<a->proof->check_count;n++) {
        NvmMixedScalarCheck *c=&a->proof->checks[n];
        if(c->function==i->function && c->pc==i->pc && c->operand==operand && c->policy==policy) {
            c->actual_tags|=actual;return c->required_tags==required;
        }
    }
    if(a->proof->check_count==NVM_MIXED_SCALAR_CHECKS)return mc_stop(a,NVM_MIXED_SHAPE_LIMIT,i->function,i->pc,"I reached my scalar obligation budget.");
    a->proof->checks[a->proof->check_count++]=(NvmMixedScalarCheck){i->function,i->pc,operand,actual,required,policy};return true;
}
static bool mc_transfer(MCAnalysis *a,uint32_t index,MCFrame *frame) {
    MCInstruction *i=&a->instructions[index];uint32_t f=i->function;
    uint16_t locals=a->module->functions[f].local_count;
    NvmAffineState *owners=frame->owners;MCValue *stack=frame->values+locals;
    DecodedInstruction *in=&i->in;uint8_t op=in->opcode;
    const InstructionInfo *info=isa_get_info(op);
    int pops=info->pop_count,pushes=info->push_count;
    NvmAffineType fields[MC_SLOTS];uint16_t fields_count=0,local=in->operands[0].u16;
    if(op==OP_ARR_LITERAL){pops=in->operands[1].u16;pushes=1;}
    if(op==OP_AGG_PACK){pops=in->operands[3].u16;pushes=1;}
    if(op==OP_OWN_PACK || op==OP_OWN_UNPACK_LOCAL) {
        uint32_t layout=op==OP_OWN_PACK?in->operands[0].u32:owners->facts->locals[local].layout;
        if(!mc_owned_layout(a,layout) || !nvm_affine_record_fields(owners,layout,fields,MC_SLOTS,&fields_count))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require RESOURCE authority before owner field transfer.");
        pops=op==OP_OWN_PACK?fields_count:0;pushes=op==OP_OWN_PACK?1:fields_count;
    }
    if(op==OP_CALL) {
        const NvmFunctionEntry *callee=&a->module->functions[in->operands[0].u32];pops=callee->arity;pushes=callee->result_count;
    }
    if(op==OP_RET) {
        Slot result=owners->facts->result;
        if(frame->depth!=a->module->functions[f].result_count ||
           (frame->depth && !mc_type(a,stack[0],result)) ||
           !nvm_affine_can_exit_type(owners,(NvmAffineType){result.tag,result.layout}))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an exact return and complete independent owner consumption.");
        return true;
    }
    if(pops<0 || pushes<0 || frame->depth<pops)return mc_stop(a,NVM_MIXED_SHAPE_INVALID,f,i->pc,"I require a complete independent operand stack.");
    if((uint32_t)(frame->depth-pops+pushes)>MC_SLOTS)return mc_stop(a,NVM_MIXED_SHAPE_LIMIT,f,i->pc,"I reached my independent operand stack budget.");
    uint16_t base=(uint16_t)(frame->depth-pops);MCValue result={0};uint8_t required;
    int produced=mc_scalar_op(op,&required);
    if(produced>=0) {
        for(uint16_t n=base;n<frame->depth;n++) {
            if(required==TAG_COUNT || !mc_plain(stack[n]))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I refuse heap or owner scalar operands.");
            if(!mc_exact(stack[n],required)) {
                if(required!=TAG_FLOAT || stack[n].tags!=MC_FLOAT_VOID)
                    return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact scalar inputs or an explicit FLOAT-or-VOID check.");
                if(!mc_check(a,i,(uint16_t)(n-base),stack[n].tags,MC_BIT(TAG_FLOAT),NVM_MIXED_CHECK_FLOAT))return false;
            }
        }
        result=mc_tag((uint8_t)produced);
    } else switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_EQ:case OP_NE:case OP_LT:case OP_LE:case OP_GT:case OP_GE: {
        MCValue x=stack[base],y=stack[base+1];bool exact=mc_plain(x) && mc_plain(y) && x.tags==y.tags && !(x.tags&(x.tags-1));
        bool optional=(op==OP_EQ || op==OP_NE) && mc_plain(x) && mc_plain(y) && !(x.tags&~MC_FLOAT_VOID) && !(y.tags&~MC_FLOAT_VOID);
        if(!exact && !optional)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require supported scalar value comparisons, never owner or ordinary heap equality.");
        if(op!=OP_EQ && op!=OP_NE && (x.tags&MC_BIT(TAG_VOID)))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I keep VOID ordering outside my first composition.");
        if(!mc_check(a,i,0,x.tags,0,NVM_MIXED_COMPARE_VALUE) || !mc_check(a,i,1,y.tags,0,NVM_MIXED_COMPARE_VALUE))return false;
        result=mc_tag(TAG_BOOL);break;
    }
    case OP_DUP:
        if(stack[frame->depth-1].kind)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I cannot duplicate an owner or observation.");
        stack[frame->depth]=stack[frame->depth-1];frame->depth++;goto successors;
    case OP_POP:
        if(stack[base].kind)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require explicit owner consumption rather than POP.");
        break;
    case OP_SWAP:
        result=stack[frame->depth-1];stack[frame->depth-1]=stack[frame->depth-2];stack[frame->depth-2]=result;goto successors;
    case OP_ROT3:
        result=stack[frame->depth-1];stack[frame->depth-1]=stack[frame->depth-2];stack[frame->depth-2]=stack[frame->depth-3];stack[frame->depth-3]=result;goto successors;
    case OP_LOAD_LOCAL: {
        Slot slot=owners->facts->locals[local];
        if(mc_owned_local(a,owners,local)) {
            uint8_t tag,mode;
            if(!nvm_affine_local_info(owners,local,&tag,&mode) || tag!=TAG_STRUCT || mode)
                return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require a live exact owner observation.");
            result=mc_owner(slot.layout);result.kind=2;result.root=local;
        } else {
            result=frame->values[local];
            if(!mc_field(a,result,slot.tag,slot.layout))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require initialized exact ordinary local alternatives.");
        }
        break;
    }
    case OP_STORE_LOCAL: {
        Slot slot=owners->facts->locals[local];
        if(mc_owned_local(a,owners,local) || !mc_field(a,stack[base],slot.tag,slot.layout))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I refuse owner or mismatched ordinary local stores.");
        if(scalar(slot.tag) && !nvm_affine_scalar_define(owners,local))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require a checked scalar destination.");
        /* ARRAY/ordinary STRUCT initialization stays in its own value state. */
        frame->values[local]=stack[base];break;
    }
    case OP_OWN_MOVE_LOCAL:case OP_OWN_UNPACK_LOCAL: {
        NvmAffineType type;
        if(!mc_owned_local(a,owners,local) || mc_observed(frame,locals,local) || !nvm_affine_take_local(owners,local,&type))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require a live unobserved RESOURCE for move or unpack.");
        if(op==OP_OWN_MOVE_LOCAL)result=mc_owner(type.layout);
        else {
            for(uint16_t n=0;n<fields_count;n++)stack[frame->depth++]=mc_tag(fields[n].tag);
            goto successors;
        }
        break;
    }
    case OP_OWN_STORE_LOCAL:
        if(!mc_owned_local(a,owners,local) || !mc_type(a,stack[base],owners->facts->locals[local]) ||
           !nvm_affine_put_local(owners,local,(NvmAffineType){TAG_STRUCT,stack[base].nominal}))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an available exact RESOURCE destination.");
        break;
    case OP_OWN_PACK:
        for(uint16_t n=0;n<fields_count;n++)if(!mc_exact(stack[base+n],fields[n].tag))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact scalar fields before producing an owner token.");
        result=mc_owner(in->operands[0].u32);break;
    case OP_AGG_PACK: {
        const NvmMixedFloatOrigin *o=&a->proof->shape->origins[i->origin];
        const NvmV2Layout *layout=&a->proof->shape->view->layouts.items[o->global_layout];
        if(mc_owned_layout(a,o->global_layout))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I never construct an owner through ordinary packing.");
        for(uint16_t n=0;n<o->field_count;n++) {
            const NvmMixedFieldFact *fact=&a->proof->shape->fields[o->field_start+n];
            const NvmV2LayoutField *field=&layout->fields[n];MCValue value=stack[base+n];
            if(!mc_field(a,value,field->type_tag,field->nested_idx) || (value.tags&~fact->tags) || (value.origins&~fact->origins))
                return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require every independent ordinary field alternative in the closed proof.");
        }
        result=mc_tag(TAG_STRUCT);result.origins=UINT64_C(1)<<i->origin;break;
    }
    case OP_AGG_GET: {
        MCValue receiver=stack[base];
        if(receiver.kind) {
            uint8_t tag;
            if(receiver.kind!=2 || !mc_owned_local(a,owners,receiver.root) ||
               owners->facts->locals[receiver.root].layout!=receiver.nominal ||
               !nvm_affine_scalar_field(owners,receiver.root,local,&tag))
                return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an intact independent owner observation.");
            result=mc_tag(tag);
        } else {
            if(receiver.tags!=MC_BIT(TAG_STRUCT) || !receiver.origins)return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact ordinary receiver alternatives.");
            for(uint32_t n=0;n<a->proof->shape->origin_count;n++)if(receiver.origins&(UINT64_C(1)<<n)) {
                const NvmMixedFloatOrigin *o=&a->proof->shape->origins[n];
                if(o->tag!=TAG_STRUCT || local>=o->field_count || mc_owned_layout(a,o->global_layout))
                    return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require a proved field on every ordinary receiver.");
                const NvmMixedFieldFact *fact=&a->proof->shape->fields[o->field_start+local];
                result.tags|=fact->tags;result.origins|=fact->origins;
            }
        }
        break;
    }
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        for(uint16_t n=base;n<frame->depth;n++)if(!mc_exact(stack[n],TAG_FLOAT))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact FLOAT elements for array construction.");
        result=mc_tag(TAG_ARRAY);result.origins=UINT64_C(1)<<i->origin;break;
    case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN:
        if(!mc_array(a,stack[base]))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require closed FLOAT-array receiver alternatives.");
        if((op==OP_ARR_SET || op==OP_ARR_GET) && !mc_exact(stack[base+1],TAG_INT))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact integer array indices.");
        if((op==OP_ARR_PUSH || op==OP_ARR_SET) && !mc_exact(stack[frame->depth-1],TAG_FLOAT))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact FLOAT alias writes.");
        if(op==OP_ARR_GET){result=mc_tag(TAG_FLOAT);result.tags|=MC_BIT(TAG_VOID);}
        else if(op==OP_ARR_LEN)result=mc_tag(TAG_INT);else result=stack[base];
        break;
    case OP_CALL: {
        const NvmAffineState *callee=a->functions[in->operands[0].u32].seed;
        for(uint16_t n=0;n<callee->facts->params;n++)if(!mc_type(a,stack[base+n],callee->facts->locals[n]))
            return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require exact positional call values before owner transfer.");
        Slot returned=callee->facts->result;result=returned.tag==TAG_STRUCT?mc_owner(returned.layout):mc_tag(returned.tag);break;
    }
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        if(!mc_exact(stack[base],TAG_BOOL))return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an exact Boolean control operand.");
        break;
    default:return mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,f,i->pc,"I require an independent transfer for every mixed instruction.");
    }
    frame->depth=(uint16_t)(base+pushes);if(pushes)stack[base]=result;
successors:
    if(frame->depth>a->proof->max_stack[f])a->proof->max_stack[f]=frame->depth;
    if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
        if(!mc_join(a,i->target,frame))return false;
        if(op==OP_JMP)return true;
    }
    return mc_join(a,i->next,frame);
}
void nvm_mixed_samples_proof_free(NvmMixedSamplesProof *proof) {
    if(proof){nvm_mixed_float_proof_free(proof->shape);free(proof);}
}
NvmMixedShapeResult nvm_analyze_mixed_samples(const NvmModule *module,NvmMixedSamplesProof **out) {
    NvmMixedShapeResult error={NVM_MIXED_SHAPE_INVALID,0,0,"I require an immutable module and output for private composition."};
    if(!out)return error;
    NvmMixedFloatProof *shape=NULL;error=nvm_analyze_mixed_float_origins(module,&shape);
    if(error.status!=NVM_MIXED_SHAPE_PROVED)return error;
    MCAnalysis *a=calloc(1,sizeof *a);NvmMixedSamplesProof *proof=calloc(1,sizeof *proof);
    if(!a || !proof){free(a);free(proof);nvm_mixed_float_proof_free(shape);return (NvmMixedShapeResult){NVM_MIXED_SHAPE_MEMORY,0,0,"I could not allocate private composition."};}
    a->module=module;a->proof=proof;proof->shape=shape;
    a->error=(NvmMixedShapeResult){NVM_MIXED_SHAPE_INVALID,0,0,"I require complete independent mixed structure."};
    for(uint32_t n=0;n<NVM_RECORD_PLAN_MAX_LAYOUTS;n++)proof->global_to_managed[n]=proof->managed_to_global[n]=NVM_V2_NO_INDEX;
    if(!mc_preflight(a))goto done;
    for(uint32_t f=0;f<module->function_count;f++) {
        uint16_t locals=module->functions[f].local_count;MCValue values[MC_SLOTS*2]={0};
        MCFrame seed={a->functions[f].seed,values,0};
        for(uint16_t n=0;n<locals;n++)values[n]=mc_tag(TAG_VOID);
        for(uint16_t n=0;n<module->functions[f].arity;n++)if(!mc_owned_local(a,seed.owners,n))values[n]=mc_tag(seed.owners->facts->locals[n].tag);
        if(!mc_join(a,a->functions[f].start,&seed))goto done;
    }
    while(a->pending) {
        if(a->visits++==MC_VISITS){mc_stop(a,NVM_MIXED_SHAPE_LIMIT,0,0,"I reached my composed worklist bound.");goto done;}
        uint32_t index=a->queue[a->head];a->head=(a->head+1)%NVM_MIXED_FLOAT_INSTRUCTIONS;a->pending--;
        MCInstruction *i=&a->instructions[index];i->queued=false;
        MCFrame *frame=mc_frame_clone(i->frame,module->functions[i->function].local_count);
        if(!frame){mc_stop(a,NVM_MIXED_SHAPE_MEMORY,i->function,i->pc,"I could not copy composed transfer state.");goto done;}
        bool ok=mc_transfer(a,index,frame);mc_frame_free(frame);if(!ok)goto done;
    }
    for(uint32_t n=0;n<a->count;n++)if(!a->instructions[n].frame){mc_stop(a,NVM_MIXED_SHAPE_UNRESOLVED,a->instructions[n].function,a->instructions[n].pc,"I require checked state for every declared mixed instruction.");goto done;}
    /* These are separate certified identities; I do not promote descriptive classes. */
    for(uint32_t global=0;global<shape->view->layouts.count;global++) {
        bool included=shape->view->classes[global]==NVM_MIXED_ORDINARY_STRUCTURAL;
        if(shape->view->classes[global]==NVM_MIXED_PENDING_ARRAY_PROOF)
            for(uint32_t n=0;n<shape->origin_count;n++)if(shape->origins[n].tag==TAG_STRUCT && shape->origins[n].global_layout==global)included=true;
        if(included){proof->global_to_managed[global]=proof->managed_count;proof->managed_to_global[proof->managed_count++]=global;}
    }
    proof->checked_functions=module->function_count;proof->checked_instructions=a->count;proof->visits=a->visits;proof->affine_checked=true;
    a->error=(NvmMixedShapeResult){NVM_MIXED_SHAPE_PROVED,0,0,"I checked private mixed composition; runtime obligations and admission remain separate."};
done:
    for(uint32_t n=0;n<a->count;n++)mc_frame_free(a->instructions[n].frame);
    for(uint32_t f=0;f<module->function_count;f++)nvm_affine_state_free(a->functions[f].seed);
    error=a->error;free(a);
    if(error.status==NVM_MIXED_SHAPE_PROVED)*out=proof;else nvm_mixed_samples_proof_free(proof);
    return error;
}
