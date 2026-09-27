/* I retain conditional owner-field summaries separately from execution authority. */
#include "owned_array_origins.h"
#include "isa.h"
#define OQ_FUNCTIONS 8u
#define OQ_SLOTS 256u
#define OQ_INSTRUCTIONS 4096u
#define OQ_CELLS 1048576u
#define OQ_FIELDS 65536u
#define OQ_VISITS 262144u
#define OQ_BIT(t) ((uint16_t)(1u << (t)))
#define OQ_SCALARS (OQ_BIT(TAG_VOID)|OQ_BIT(TAG_INT)|OQ_BIT(TAG_U8)|OQ_BIT(TAG_FLOAT)|OQ_BIT(TAG_BOOL))
typedef struct { uint8_t tag; uint32_t layout; } OQType;
typedef struct {
    uint16_t tags, root;
    uint32_t layout, fields;
    NvmOwnerOriginSet origins;
    uint8_t kind, unknown; /* 1 unique shell, 2 nonescaping observation. */
} OQValue;
typedef struct {
    DecodedInstruction in;
    uint32_t function, pc, next, target;
    int16_t origin, depth;
    bool queued;
    OQValue *state;
} OQInstruction;
typedef struct {
    OQType result, locals[OQ_SLOTS];
    uint32_t start, end;
    uint16_t formal_offset[8];
    uint8_t calls, color;
    bool returned;
    OQValue returned_value;
} OQFunction;
struct NvmOwnedArrayOrigins {
    NvmOwnedArrayLayouts *layouts;
    NvmOwnerOriginCounts counts;
    NvmOwnerOriginSite sites[64];
    NvmOwnerOriginSummary summaries[OQ_FUNCTIONS];
    NvmOwnerOriginPath inputs[OQ_FUNCTIONS][NVM_OWNER_ORIGIN_FORMALS];
    NvmOwnerOriginPath results[OQ_FUNCTIONS][NVM_OWNER_ORIGIN_FORMALS];
    NvmOwnerOriginObligation obligations[OQ_INSTRUCTIONS];
};
typedef struct {
    const NvmModule *module;
    NvmOwnedArrayOrigins *proof;
    OQFunction functions[OQ_FUNCTIONS];
    OQInstruction instructions[OQ_INSTRUCTIONS];
    /* Immutable vectors: joins/copies never mutate facts shared by another state. */
    NvmOwnerOriginSet *fields;
    uint32_t field_count, cells, count, visits, field_work;
    uint32_t array_count[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t array_start[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t *paths;
    uint32_t queue[OQ_INSTRUCTIONS], head, tail, queued;
    NvmOwnerOriginResult error;
} OQAnalysis;
static bool oq_stop(OQAnalysis *a,NvmOwnerOriginStatus status,uint32_t f,uint32_t pc,const char *message) {
    a->error=(NvmOwnerOriginResult){status,f,pc,message};return false;
}
/* I charge bounded transfer work separately from dequeued instruction visits. */
static bool oq_work(OQAnalysis *a,uint32_t count) {
    if(count>OQ_VISITS-a->field_work)
        return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,0,0,"I reached my field-transfer work budget.");
    a->field_work+=count;return true;
}
static bool oq_nonempty(NvmOwnerOriginSet s) {
    for(unsigned w=0;w<NVM_OWNER_ORIGIN_WORDS;w++)if(s.words[w])return true;
    return false;
}
static bool oq_union(NvmOwnerOriginSet *to,NvmOwnerOriginSet from) {
    bool changed=false;
    for(unsigned w=0;w<NVM_OWNER_ORIGIN_WORDS;w++) {
        uint64_t n=to->words[w]|from.words[w];changed|=n!=to->words[w];to->words[w]=n;
    }
    return changed;
}
static OQValue oq_tag(uint8_t tag) {
    OQValue v={0};v.tags=OQ_BIT(tag);v.layout=NVM_V2_NO_INDEX;return v;
}
static bool oq_exact(OQValue v,uint16_t tags) {
    return !v.unknown && !v.kind && !oq_nonempty(v.origins) && v.tags==tags;
}
static bool oq_owner_layout(OQAnalysis *a,uint32_t layout) {
    NvmOwnedArrayLayoutFact fact;
    return nvm_owned_array_layout_fact(a->proof->layouts,layout,&fact) &&
           (fact.flags&NVM_LAYOUT_RESOURCE)!=0;
}
static bool oq_vector(OQAnalysis *a,uint32_t count,uint32_t *start) {
    if(count>OQ_FIELDS-a->field_count)
        return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,0,0,"I reached my intermediate owner-field fact budget.");
    if(!oq_work(a,count))return false;
    *start=a->field_count;a->field_count+=count;return true;
}
static bool oq_paths(OQAnalysis *a) {
    NvmOwnedArrayLayoutCounts counts;
    nvm_owned_array_layout_counts(a->proof->layouts,&counts);
    if(counts.leaf_paths) {
        a->paths=calloc(counts.leaf_paths,sizeof *a->paths);
        if(!a->paths)return oq_stop(a,NVM_OWNER_ORIGIN_MEMORY,0,0,"I could not allocate ARRAY path indices.");
    }
    uint32_t used=0;
    for(uint32_t l=0;l<counts.layouts;l++) {
        NvmOwnedArrayLayoutFact fact;nvm_owned_array_layout_fact(a->proof->layouts,l,&fact);
        a->array_start[l]=used;
        for(uint32_t p=0;p<fact.path_count;p++) {
            NvmOwnedArrayLeafPath path;nvm_owned_array_leaf_path(a->proof->layouts,fact.path_start+p,&path);
            if(path.tag==TAG_ARRAY) {a->paths[used++]=fact.path_start+p;a->array_count[l]++;}
        }
    }
    a->fields=calloc(OQ_FIELDS,sizeof *a->fields);
    return a->fields!=NULL || oq_stop(a,NVM_OWNER_ORIGIN_MEMORY,0,0,"I could not allocate bounded intermediate field facts.");
}
static NvmOwnedArrayLeafPath oq_path(OQAnalysis *a,uint32_t layout,uint32_t p) {
    NvmOwnedArrayLeafPath path={0};
    nvm_owned_array_leaf_path(a->proof->layouts,a->paths[a->array_start[layout]+p],&path);return path;
}
static bool oq_signature(OQAnalysis *a,OQType t) {
    if(t.tag==TAG_STRUCT)return oq_owner_layout(a,t.layout);
    return t.layout==NVM_V2_NO_INDEX && (t.tag==TAG_VOID || t.tag==TAG_INT ||
        t.tag==TAG_BOOL || t.tag==TAG_U8 || t.tag==TAG_STRING);
}
static bool oq_descriptors(OQAnalysis *a) {
    NvmOwnedArrayTransport transport;nvm_owned_array_transport(a->proof->layouts,&transport);
    NvmV2Cursor c;uint32_t n,version;const uint8_t *unused;
    nvm_v2_cursor_init(&c,transport.ownership,transport.ownership_size);
    if(nvm_v2_u32(&c,&version)!=NVM_V2_OK || nvm_v2_u32(&c,&n)!=NVM_V2_OK ||
       nvm_v2_take(&c,n,&unused)!=NVM_V2_OK || nvm_v2_align4(&c)!=NVM_V2_OK ||
       nvm_v2_u32(&c,&n)!=NVM_V2_OK || n!=a->module->function_count)return false;
    for(uint32_t f=0;f<n;f++) {
        const NvmFunctionEntry *fn=&a->module->functions[f];
        uint16_t locals,params;
        if(nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK ||
           locals!=fn->local_count || params!=fn->arity)return false;
        for(uint32_t j=0;j<=locals;j++) {
            OQType t;uint8_t mode;uint16_t reserved;
            if(nvm_v2_u8(&c,&t.tag)!=NVM_V2_OK || nvm_v2_u8(&c,&mode)!=NVM_V2_OK ||
               nvm_v2_u16(&c,&reserved)!=NVM_V2_OK || nvm_v2_u32(&c,&t.layout)!=NVM_V2_OK)return false;
            if(mode || reserved)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I keep borrowed modes outside owner ARRAY origins.");
            if(j==0 || j<=params) {
                if(!oq_signature(a,t) || (j && t.tag==TAG_VOID))
                    return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require exact scalar or owner signatures.");
            } else if(!oq_signature(a,t) &&
                      !(t.layout==NVM_V2_NO_INDEX && (t.tag==TAG_ARRAY || t.tag==TAG_FLOAT)))
                return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require modeled exact local types.");
            if(!j) {
                if((fn->result_count==0 && t.tag!=TAG_VOID) ||
                   (fn->result_count && fn->result_tag!=t.tag))return false;
                if(!f && t.tag!=TAG_INT && t.tag!=TAG_BOOL && t.tag!=TAG_U8 && t.tag!=TAG_VOID)
                    return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I retain a scalar or VOID entry result.");
                if(t.tag==TAG_STRUCT && a->array_count[t.layout]>NVM_OWNER_ORIGIN_FORMALS)
                    return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,f,0,"I reached my result ARRAY path budget.");
                a->functions[f].result=t;
            } else a->functions[f].locals[j-1]=t;
        }
    }
    return true;
}
/* Exact accepted arithmetic/control inventory; everything else refuses. */
static bool oq_optional_comparison(uint8_t op) {
    switch(op) {
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:return true;
    default:return false;
    }
}
static int oq_scalar(uint8_t op,uint16_t *required) {
    *required=0;
    switch(op) {
    case OP_PUSH_I64:return TAG_INT;case OP_PUSH_F64:return TAG_FLOAT;
    case OP_PUSH_U8:return TAG_U8;case OP_PUSH_BOOL:return TAG_BOOL;case OP_PUSH_VOID:return TAG_VOID;
    case OP_I64_ADD:case OP_I64_SUB:case OP_I64_MUL:case OP_I64_DIV_S:case OP_I64_REM_S:case OP_I64_NEG:
        *required=OQ_BIT(TAG_INT);return TAG_INT;
    case OP_F64_ADD:case OP_F64_SUB:case OP_F64_MUL:case OP_F64_DIV:case OP_F64_NEG:
        *required=OQ_BIT(TAG_FLOAT);return TAG_FLOAT;
    case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
        *required=OQ_BIT(TAG_INT);return TAG_BOOL;
    case OP_F64_EQ:case OP_F64_NE:case OP_F64_LT:case OP_F64_LE:case OP_F64_GT:case OP_F64_GE:
        *required=OQ_BIT(TAG_FLOAT);return TAG_BOOL;
    case OP_BOOL_AND:case OP_BOOL_OR:case OP_BOOL_NOT:*required=OQ_BIT(TAG_BOOL);return TAG_BOOL;
    case OP_EQ:case OP_NE:return TAG_BOOL;
    default:return -1;
    }
}
static bool oq_supported(uint8_t op) {
    uint16_t required;if(oq_scalar(op,&required)>=0)return true;
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
static bool oq_decode(OQAnalysis *a) {
    const NvmModule *m=a->module;
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];OQFunction *af=&a->functions[f];af->start=a->count;
        if(fn->name_idx>=m->string_count || !m->strings || !m->strings[fn->name_idx])
            return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,0,"I require complete function names.");
        if(!strcmp(m->strings[fn->name_idx],"__init__"))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I keep initializer entry roots outside this query.");
        if(fn->code_offset>m->code_size || fn->code_length>m->code_size-fn->code_offset || !fn->code_length)
            return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,0,"I require complete nonempty code ranges.");
        for(uint32_t e=0;e<f;e++) {
            const NvmFunctionEntry *other=&m->functions[e];
            if(fn->code_offset<(uint64_t)other->code_offset+other->code_length &&
               other->code_offset<(uint64_t)fn->code_offset+fn->code_length)
                return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require distinct function code ranges.");
        }
        for(uint32_t pc=0;pc<fn->code_length;) {
            if(a->count==OQ_INSTRUCTIONS)return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,f,pc,"I reached my instruction budget.");
            OQInstruction *i=&a->instructions[a->count++];i->function=f;i->pc=pc;i->depth=-1;i->origin=-1;
            uint32_t width=isa_decode(m->code+fn->code_offset+pc,fn->code_length-pc,&i->in);
            if(!width)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,pc,"I require complete decoded instructions.");
            a->proof->obligations[a->count-1]=(NvmOwnerOriginObligation){
                .function=f,.pc=pc,.opcode=i->in.opcode};
            uint8_t op=i->in.opcode;
            if(!oq_supported(op))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,pc,"I have no owner-origin transfer for this opcode.");
            if(op==OP_PUSH_BOOL && i->in.operands[0].u8>1)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,pc,"I require canonical BOOL immediates.");
            if(op==OP_PUSH_STR && (i->in.operands[0].u32>=m->string_count || !m->strings[i->in.operands[0].u32]))
                return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,pc,"I require an existing STRING constant.");
            if(op==OP_LOAD_LOCAL || op==OP_STORE_LOCAL || op==OP_OWN_MOVE_LOCAL ||
               op==OP_OWN_STORE_LOCAL || op==OP_OWN_UNPACK_LOCAL) {
                if(i->in.operands[0].u16>=fn->local_count)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,pc,"I require an existing local.");
            }
            if(op==OP_CALL) {
                uint32_t callee=i->in.operands[0].u32;
                if(callee>=m->function_count)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,pc,"I require an existing callee.");
                if(!callee)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,pc,"I do not call my selected entry.");
                af->calls|=(uint8_t)(1u<<callee);
            }
            if(op==OP_OWN_PACK || op==OP_OWN_UNPACK_LOCAL) {
                OQType t=op==OP_OWN_PACK?(OQType){TAG_STRUCT,i->in.operands[0].u32}:af->locals[i->in.operands[0].u16];
                if(t.tag!=TAG_STRUCT || !oq_owner_layout(a,t.layout))
                    return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,pc,"I require an exact owner declaration.");
            }
            if(op==OP_ARR_NEW || op==OP_ARR_LITERAL) {
                if(i->in.operands[0].u8!=TAG_FLOAT)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,pc,"I require declared FLOAT array sites.");
                if(a->proof->counts.sites==64)return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,f,pc,"I reached my concrete origin budget.");
                i->origin=(int16_t)a->proof->counts.sites++;
                a->proof->sites[i->origin]=(NvmOwnerOriginSite){f,pc,op};
            }
            pc+=width;i->next=a->count;i->target=UINT32_MAX;
        }
        af->end=a->count;
    }
    for(uint32_t f=0;f<m->function_count;f++)for(uint32_t n=a->functions[f].start;n<a->functions[f].end;n++) {
        OQInstruction *i=&a->instructions[n];uint8_t op=i->in.opcode;
        if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
            int64_t target=(int64_t)i->pc+i->in.operands[0].i32;
            for(uint32_t j=a->functions[f].start;j<a->functions[f].end;j++)
                if(target==a->instructions[j].pc)i->target=j;
            if(i->target==UINT32_MAX)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,i->pc,"I require an instruction-boundary branch target.");
        }
        if(i->next==a->functions[f].end && op!=OP_JMP && op!=OP_RET)
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require explicit terminal control flow.");
    }
    return true;
}
static bool oq_array(OQValue v) {
    return !v.kind && !v.unknown && v.tags==OQ_BIT(TAG_ARRAY) && oq_nonempty(v.origins);
}
static bool oq_type(OQAnalysis *a,OQValue v,OQType t) {
    if(t.tag==TAG_STRUCT)return v.kind==1 && !v.unknown && v.tags==OQ_BIT(TAG_STRUCT) &&
        v.layout==t.layout && oq_owner_layout(a,t.layout);
    if(t.tag==TAG_ARRAY)return oq_array(v);
    return oq_exact(v,OQ_BIT(t.tag));
}
static bool oq_require(OQAnalysis *a,uint32_t f,OQValue v) {
    if(v.kind) {
        if(!oq_work(a,a->array_count[v.layout]))return false;
        for(uint32_t p=0;p<a->array_count[v.layout];p++) {
            NvmOwnerOriginSet s=a->fields[v.fields+p];
            if(!oq_nonempty(s))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require every owner ARRAY path origin.");
            oq_union(&a->proof->summaries[f].required,s);
        }
    } else if(v.tags==OQ_BIT(TAG_ARRAY)) {
        if(!oq_array(v))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require positive ARRAY origins.");
        oq_union(&a->proof->summaries[f].required,v.origins);
    }
    return true;
}
static bool oq_merge_value(OQAnalysis *a,OQValue *to,OQValue from,bool *changed,uint32_t f,uint32_t pc) {
    if((to->kind || from.kind) && (to->kind!=from.kind || to->layout!=from.layout ||
                                 to->root!=from.root || to->tags!=from.tags))
        return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,pc,"I require matching initialized owner identities at joins.");
    uint16_t tags=to->tags;uint8_t unknown=to->unknown;
    to->tags|=from.tags;to->unknown|=from.unknown;
    *changed|=tags!=to->tags || unknown!=to->unknown;
    *changed|=oq_union(&to->origins,from.origins);
    if(to->kind) {
        uint32_t count=a->array_count[to->layout];bool difference=false;
        if(!oq_work(a,count))return false;
        for(uint32_t p=0;p<count;p++) {
            NvmOwnerOriginSet s=a->fields[to->fields+p];difference|=oq_union(&s,a->fields[from.fields+p]);
        }
        if(difference) {
            uint32_t start;if(!oq_vector(a,count,&start))return false;
            for(uint32_t p=0;p<count;p++) {
                a->fields[start+p]=a->fields[to->fields+p];oq_union(&a->fields[start+p],a->fields[from.fields+p]);
            }
            to->fields=start;*changed=true;
        }
    }
    return true;
}
static void oq_enqueue(OQAnalysis *a,uint32_t index) {
    if(a->instructions[index].queued)return;
    a->instructions[index].queued=true;a->queue[a->tail]=index;a->tail=(a->tail+1)%OQ_INSTRUCTIONS;a->queued++;
}
static bool oq_join(OQAnalysis *a,uint32_t index,const OQValue *state,uint16_t depth) {
    if(index>=a->count)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,0,0,"I require a valid state destination.");
    OQInstruction *i=&a->instructions[index];uint32_t locals=a->module->functions[i->function].local_count;
    if(depth>OQ_SLOTS)return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,i->function,i->pc,"I reached my stack limit.");
    if(i->depth>=0 && i->depth!=depth)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,i->function,i->pc,"I require equal join stack heights.");
    bool changed=false;
    if(!i->state) {
        uint32_t cells=locals+OQ_SLOTS;
        if(cells>OQ_CELLS-a->cells)return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,i->function,i->pc,"I reached my state cell budget.");
        i->state=calloc(cells,sizeof *i->state);
        if(!i->state)return oq_stop(a,NVM_OWNER_ORIGIN_MEMORY,i->function,i->pc,"I could not allocate instruction state.");
        a->cells+=cells;i->depth=(int16_t)depth;memcpy(i->state,state,(locals+depth)*sizeof *state);changed=true;
    } else for(uint32_t n=0;n<locals+depth;n++)
        if(!oq_merge_value(a,&i->state[n],state[n],&changed,i->function,i->pc))return false;
    if(changed)oq_enqueue(a,index);
    return true;
}
static bool oq_make_owner(OQAnalysis *a,uint32_t layout,OQValue *out) {
    OQValue v=oq_tag(TAG_STRUCT);v.kind=1;v.layout=layout;
    if(!oq_vector(a,a->array_count[layout],&v.fields))return false;
    *out=v;return true;
}
/* The descriptor expansion order is declaration-field order, including nested
 * children. Its ARRAY-only subsequence preserves that same exact order. */
static bool oq_child(OQAnalysis *a,OQValue parent,uint16_t field,OQValue *out) {
    NvmV2LayoutField declaration;
    if(!nvm_owned_array_layout_field(a->proof->layouts,parent.layout,field,&declaration))
        return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,0,0,"I require an existing exact owner field.");
    OQValue v=oq_tag(declaration.type_tag);uint32_t start=0,count=0;
    if(!oq_work(a,a->array_count[parent.layout]))return false;
    for(uint32_t p=0;p<a->array_count[parent.layout];p++) {
        NvmOwnedArrayLeafPath path=oq_path(a,parent.layout,p);
        if(path.fields[0]<field)start++;
        else if(path.fields[0]==field)count++;
    }
    if(declaration.type_tag==TAG_ARRAY) {
        if(count!=1)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,0,0,"I require one direct ARRAY leaf.");
        v.origins=a->fields[parent.fields+start];
    } else if(declaration.type_tag==TAG_STRUCT) {
        v.kind=parent.kind;v.root=parent.root;v.layout=declaration.nested_idx;v.fields=parent.fields+start;
        if(count!=a->array_count[v.layout])return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,0,0,"I require exact child path expansion.");
    }
    *out=v;return true;
}
static bool oq_substitute(OQAnalysis *a,uint32_t caller,uint32_t callee,const OQValue *args,
                          NvmOwnerOriginSet input,NvmOwnerOriginSet *out) {
    NvmOwnerOriginSet result={{input.words[0],0,0,0,0}};
    NvmOwnerOriginSummary *summary=&a->proof->summaries[callee];
    if(!oq_work(a,NVM_OWNER_ORIGIN_FORMALS))return false;
    for(uint32_t bit=0;bit<NVM_OWNER_ORIGIN_FORMALS;bit++)if(input.words[1+bit/64]&(UINT64_C(1)<<(bit%64))) {
        if(bit>=summary->formal_count)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,caller,0,"I refuse an unbound callee formal bit.");
        const NvmOwnerOriginPath *formal=&a->proof->inputs[callee][bit];
        OQValue owner=args[formal->ordinal];
        uint32_t offset=a->functions[callee].formal_offset[formal->ordinal];
        if(owner.layout!=formal->path.root_layout || bit<offset || bit-offset>=a->array_count[owner.layout])
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,caller,0,"I require exact actual nominal path indexing.");
        NvmOwnerOriginSet actual=a->fields[owner.fields+bit-offset];
        if(!oq_nonempty(actual))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,caller,0,"I require initialized actual field origins.");
        oq_union(&result,actual);
    }
    *out=result;return true;
}
static bool oq_step(OQAnalysis *a,uint32_t index) {
    OQInstruction *i=&a->instructions[index];uint32_t f=i->function;
    OQFunction *af=&a->functions[f];const NvmFunctionEntry *fn=&a->module->functions[f];
    OQValue state[OQ_SLOTS*2];memcpy(state,i->state,(fn->local_count+(uint16_t)i->depth)*sizeof *state);
    OQValue *stack=state+fn->local_count;uint16_t depth=(uint16_t)i->depth;
    DecodedInstruction *in=&i->in;uint8_t op=in->opcode;const InstructionInfo *info=isa_get_info(op);
    if(!info)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,i->pc,"I require opcode metadata.");
    int pops=info->pop_count,pushes=info->push_count;
    if(op==OP_ARR_LITERAL) {pops=in->operands[1].u16;pushes=1;}
    if(op==OP_OWN_PACK || op==OP_OWN_UNPACK_LOCAL) {
        uint32_t layout=op==OP_OWN_PACK?in->operands[0].u32:af->locals[in->operands[0].u16].layout;
        NvmOwnedArrayLayoutFact fact;nvm_owned_array_layout_fact(a->proof->layouts,layout,&fact);
        if(op==OP_OWN_PACK) {pops=(int)fact.field_count;pushes=1;}
        else {pops=0;pushes=(int)fact.field_count;}
    }
    if(op==OP_CALL) {const NvmFunctionEntry *callee=&a->module->functions[in->operands[0].u32];pops=callee->arity;pushes=callee->result_count;}
    if(op==OP_RET) {
        if(depth!=fn->result_count)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,i->pc,"I require the declared return count.");
        if(depth && !oq_type(a,stack[0],af->result))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require an exact complete returned value.");
        if(depth && !oq_require(a,f,stack[0]))return false;
        if(!af->returned) {af->returned=true;if(depth)af->returned_value=stack[0];}
        else if(depth) {bool changed=false;if(!oq_merge_value(a,&af->returned_value,stack[0],&changed,f,i->pc))return false;}
        return true;
    }
    if(pops<0 || pushes<0 || depth<pops)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,i->pc,"I require complete operand stacks.");
    if((uint32_t)(depth-pops+pushes)>OQ_SLOTS)return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,f,i->pc,"I reached my stack budget.");
    uint16_t base=(uint16_t)(depth-pops);OQValue result={0};uint16_t required;
    NvmOwnerOriginObligation *ob=&a->proof->obligations[index];
    int scalar=oq_scalar(op,&required);
    if(scalar>=0) {
        result=oq_tag((uint8_t)scalar);
        if(pops>2)return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,i->pc,"I require bounded scalar operand rows.");
        ob->operand_count=(uint8_t)pops;
        for(uint16_t n=base;n<depth;n++) {
            OQValue v=stack[n];
            if(v.kind || v.unknown || oq_nonempty(v.origins) || !v.tags || (v.tags&~OQ_SCALARS))
                return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require modeled scalar operand alternatives.");
            uint16_t operand=(uint16_t)(n-base);
            ob->actual_tags|=v.tags;ob->required_tags|=required;
            ob->operand_actual[operand]|=v.tags;ob->operand_required[operand]=required;
            if(required && v.tags!=required) {
                if(!oq_optional_comparison(op) || !(v.tags&OQ_BIT(TAG_FLOAT)) ||
                   (v.tags&~(OQ_BIT(TAG_FLOAT)|OQ_BIT(TAG_VOID))))
                    return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact typed scalar operands or a checked optional FLOAT comparison.");
                ob->runtime_checks|=(uint8_t)(1u<<operand);
            }
        }
    } else switch(op) {
    case OP_NOP:case OP_JMP:break;
    case OP_PUSH_STR:result=oq_tag(TAG_STRING);break;
    case OP_DUP:
        if(stack[depth-1].kind)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I do not duplicate owners or observations.");
        stack[depth]=stack[depth-1];depth++;goto successors;
    case OP_SWAP:result=stack[depth-1];stack[depth-1]=stack[depth-2];stack[depth-2]=result;goto successors;
    case OP_ROT3:result=stack[depth-1];stack[depth-1]=stack[depth-2];stack[depth-2]=stack[depth-3];stack[depth-3]=result;goto successors;
    case OP_POP:
        if(stack[base].kind)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require explicit owner consumption.");
        break;
    case OP_LOAD_LOCAL: {
        uint16_t local=in->operands[0].u16;result=state[local];
        if(!oq_type(a,result,af->locals[local]))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require initialized exact local alternatives.");
        if(result.kind) {result.kind=2;result.root=local;}
        break;
    }
    case OP_STORE_LOCAL: {
        uint16_t local=in->operands[0].u16;result=stack[base];
        if(result.kind || state[local].kind || !oq_type(a,result,af->locals[local]))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require an exact ordinary local store.");
        state[local]=result;break;
    }
    case OP_OWN_MOVE_LOCAL:case OP_OWN_UNPACK_LOCAL: {
        uint16_t local=in->operands[0].u16;result=state[local];
        for(uint16_t n=0;n<depth;n++)if(stack[n].kind==2 && stack[n].root==local)
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I keep an observed owner live until projection completes.");
        if(result.kind!=1 || !oq_type(a,result,af->locals[local]))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require an intact exact owner source.");
        state[local]=oq_tag(TAG_VOID);state[local].unknown=1;
        if(op==OP_OWN_UNPACK_LOCAL) {
            for(int n=0;n<pushes;n++)if(!oq_child(a,result,(uint16_t)n,&stack[depth++]))return false;
            goto successors;
        }
        break;
    }
    case OP_OWN_STORE_LOCAL: {
        uint16_t local=in->operands[0].u16;
        if(state[local].kind || state[local].tags!=OQ_BIT(TAG_VOID) ||
           stack[base].kind!=1 || !oq_type(a,stack[base],af->locals[local]))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require an empty exact owner destination.");
        state[local]=stack[base];break;
    }
    case OP_OWN_PACK: {
        uint32_t layout=in->operands[0].u32;
        if(!oq_make_owner(a,layout,&result))return false;
        uint32_t next=0;
        for(uint16_t n=0;n<(uint16_t)pops;n++) {
            NvmV2LayoutField field;nvm_owned_array_layout_field(a->proof->layouts,layout,n,&field);
            OQValue value=stack[base+n];
            if(!oq_type(a,value,(OQType){field.type_tag,field.nested_idx}))
                return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact owner constructor field facts.");
            if(!oq_require(a,f,value))return false;
            if(field.type_tag==TAG_ARRAY)a->fields[result.fields+next++]=value.origins;
            else if(field.type_tag==TAG_STRUCT)for(uint32_t p=0;p<a->array_count[value.layout];p++)
                a->fields[result.fields+next++]=a->fields[value.fields+p];
        }
        if(next!=a->array_count[layout])return oq_stop(a,NVM_OWNER_ORIGIN_INVALID,f,i->pc,"I require exact constructor path counts.");
        break;
    }
    case OP_AGG_GET: {
        OQValue receiver=stack[base];
        if(receiver.kind!=2 || receiver.unknown || receiver.root>=fn->local_count || state[receiver.root].kind!=1)
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require a live exact owner observation.");
        if(!oq_child(a,receiver,in->operands[0].u16,&result))return false;
        if(!oq_require(a,f,result))return false;
        break;
    }
    case OP_ARR_NEW:case OP_ARR_LITERAL:
        for(uint16_t n=base;n<depth;n++)if(!oq_exact(stack[n],OQ_BIT(TAG_FLOAT)))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact FLOAT literal elements.");
        result=oq_tag(TAG_ARRAY);result.origins.words[0]=UINT64_C(1)<<i->origin;break;
    case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN: {
        OQValue receiver=stack[base];
        if(!oq_array(receiver))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require concrete or conditional FLOAT array origins.");
        if(!oq_require(a,f,receiver))return false;
        if((op==OP_ARR_SET || op==OP_ARR_GET) && !oq_exact(stack[base+1],OQ_BIT(TAG_INT)))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact INT indices.");
        if((op==OP_ARR_SET || op==OP_ARR_PUSH) && !oq_exact(stack[depth-1],OQ_BIT(TAG_FLOAT)))
            return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require FLOAT writes through every alias.");
        if(op==OP_ARR_GET) {result=oq_tag(TAG_FLOAT);result.tags|=OQ_BIT(TAG_VOID);ob->read_tags=result.tags;}
        else if(op==OP_ARR_LEN)result=oq_tag(TAG_INT);else result=receiver;
        break;
    }
    case OP_CALL: {
        uint32_t callee=in->operands[0].u32;OQFunction *cf=&a->functions[callee];
        for(uint16_t n=0;n<(uint16_t)pops;n++) {
            if(!oq_type(a,stack[base+n],cf->locals[n]))
                return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact positional call field facts.");
            if(!oq_require(a,f,stack[base+n]))return false;
        }
        NvmOwnerOriginSet requirement;
        if(!oq_substitute(a,f,callee,stack+base,a->proof->summaries[callee].required,&requirement))return false;
        oq_union(&a->proof->summaries[f].required,requirement);
        if(pushes) {
            result=cf->returned_value;
            if(result.kind) {
                if(!oq_make_owner(a,result.layout,&result))return false;
                for(uint32_t p=0;p<a->array_count[result.layout];p++)
                    if(!oq_substitute(a,f,callee,stack+base,a->fields[cf->returned_value.fields+p],&a->fields[result.fields+p]))return false;
            }
        }
        break;
    }
    case OP_PRINT:case OP_PRINTLN:
        ob->operand_count=1;ob->operand_actual[0]|=stack[base].tags;
        ob->operand_required[0]=OQ_BIT(TAG_INT);
        ob->actual_tags|=stack[base].tags;ob->required_tags=OQ_BIT(TAG_INT);
        if(!oq_exact(stack[base],OQ_BIT(TAG_INT)))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact INT output operands.");
        break;
    case OP_JMP_TRUE:case OP_JMP_FALSE:case OP_ASSERT:
        ob->operand_count=1;ob->operand_actual[0]|=stack[base].tags;
        ob->operand_required[0]=OQ_BIT(TAG_BOOL);
        ob->actual_tags|=stack[base].tags;ob->required_tags=OQ_BIT(TAG_BOOL);
        if(!oq_exact(stack[base],OQ_BIT(TAG_BOOL)))return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require exact BOOL control operands.");
        break;
    default:return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,i->pc,"I require an explicit origin transfer.");
    }
    depth=(uint16_t)(base+pushes);if(pushes)stack[base]=result;
successors:
    if(op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
        if(!oq_join(a,i->target,state,depth))return false;
        if(op==OP_JMP)return true;
    }
    return oq_join(a,i->next,state,depth);
}
static bool oq_function(OQAnalysis *a,uint32_t f) {
    OQFunction *fn=&a->functions[f];
    if(fn->color==2)return true;
    if(fn->color==1)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require an acyclic complete call graph.");
    fn->color=1;
    for(uint32_t callee=0;callee<a->module->function_count;callee++)
        if((fn->calls&(1u<<callee)) && !oq_function(a,callee))return false;
    NvmOwnerOriginSummary *summary=&a->proof->summaries[f];summary->conditional=f!=0;
    OQValue seed[OQ_SLOTS*2]={0};const NvmFunctionEntry *declaration=&a->module->functions[f];
    for(uint16_t n=0;n<declaration->local_count;n++) {seed[n]=oq_tag(TAG_VOID);seed[n].unknown=1;}
    for(uint16_t n=0;n<declaration->arity;n++) {
        OQType t=fn->locals[n];seed[n]=oq_tag(t.tag);
        if(t.tag!=TAG_STRUCT)continue;
        fn->formal_offset[n]=(uint16_t)summary->formal_count;
        if(a->array_count[t.layout]>NVM_OWNER_ORIGIN_FORMALS-summary->formal_count)
            return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,f,0,"I reached my formal ARRAY path budget.");
        if(!oq_make_owner(a,t.layout,&seed[n]))return false;
        for(uint32_t p=0;p<a->array_count[t.layout];p++) {
            uint32_t bit=summary->formal_count++;
            NvmOwnerOriginSet formal={{0}};formal.words[1+bit/64]=UINT64_C(1)<<(bit%64);
            a->fields[seed[n].fields+p]=formal;
            a->proof->inputs[f][bit]=(NvmOwnerOriginPath){n,oq_path(a,t.layout,p),formal};
        }
    }
    if(!oq_join(a,fn->start,seed,0))return false;
    while(a->queued) {
        if(a->visits++==OQ_VISITS)return oq_stop(a,NVM_OWNER_ORIGIN_LIMIT,f,0,"I reached my origin fixed-point visit budget.");
        uint32_t index=a->queue[a->head];a->head=(a->head+1)%OQ_INSTRUCTIONS;a->queued--;
        a->instructions[index].queued=false;
        if(!oq_step(a,index))return false;
    }
    for(uint32_t n=fn->start;n<fn->end;n++)if(!a->instructions[n].state)
        return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,a->instructions[n].pc,"I require checked state for every declared instruction.");
    if(!fn->returned)return oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,f,0,"I require at least one checked return per function.");
    if(fn->result.tag==TAG_STRUCT) {
        summary->result_count=a->array_count[fn->result.layout];
        for(uint32_t p=0;p<summary->result_count;p++)
            a->proof->results[f][p]=(NvmOwnerOriginPath){0,oq_path(a,fn->result.layout,p),a->fields[fn->returned_value.fields+p]};
    }
    fn->color=2;return true;
}
static void oq_reachable(OQAnalysis *a,uint32_t f) {
    if(a->proof->summaries[f].reachable)return;
    a->proof->summaries[f].reachable=true;
    for(uint32_t c=0;c<a->module->function_count;c++)if(a->functions[f].calls&(1u<<c))oq_reachable(a,c);
}
void nvm_owned_array_origins_free(NvmOwnedArrayOrigins *p) {
    if(!p)return;
    nvm_owned_array_layouts_free(p->layouts);free(p);
}
NvmOwnerOriginResult nvm_analyze_owned_array_origins(const NvmModule *m,NvmOwnedArrayOrigins **out) {
    NvmOwnerOriginResult invalid={NVM_OWNER_ORIGIN_INVALID,0,0,"I require a complete immutable module envelope and output."};
    if(!m || !out)return invalid;
    if((nvm_capture_bindings_present(m) || nvm_service_execution_pending(m)))return (NvmOwnerOriginResult){NVM_OWNER_ORIGIN_UNRESOLVED,0,0,"I require service and capture admission before origin analysis."};
    if(!m->functions || !m->code || !m->function_count || !m->code_size ||
       m->header.magic[0]!=NVM_MAGIC_0 || m->header.magic[1]!=NVM_MAGIC_1 ||
       m->header.magic[2]!=NVM_MAGIC_2 || m->header.magic[3]!=NVM_MAGIC_3 ||
       m->header.format_version!=NVM_FORMAT_VERSION)return invalid;
    if(m->function_count>OQ_FUNCTIONS || m->header.entry_point ||
       !(m->header.flags&NVM_FLAG_HAS_MAIN) || (m->header.flags&~(NVM_FLAG_HAS_MAIN|NVM_FLAG_DEBUG_INFO)) ||
       m->import_count || m->module_ref_count || m->callback_contract_count || m->passive_size || m->passive_data)
        return (NvmOwnerOriginResult){NVM_OWNER_ORIGIN_UNRESOLVED,0,0,"I require one explicit standalone entry and at most eight functions."};
    for(uint32_t f=0;f<m->function_count;f++) {
        const NvmFunctionEntry *fn=&m->functions[f];
        if(fn->local_count>OQ_SLOTS || fn->arity>8)return (NvmOwnerOriginResult){NVM_OWNER_ORIGIN_LIMIT,f,0,"I reached my local or argument limit."};
        if(fn->arity>fn->local_count || fn->result_count>1 || fn->upvalue_count || (!f && fn->arity))return invalid;
    }
    NvmOwnedArrayLayouts *layouts=NULL;NvmRecordPlanResult described=nvm_describe_owned_array_layouts(m,&layouts);
    if(described.status!=NVM_RECORD_DESCRIBED) {
        NvmOwnerOriginStatus status=described.status==NVM_RECORD_MEMORY?NVM_OWNER_ORIGIN_MEMORY:
            described.status==NVM_RECORD_LIMIT?NVM_OWNER_ORIGIN_LIMIT:
            described.status==NVM_RECORD_INVALID?NVM_OWNER_ORIGIN_INVALID:NVM_OWNER_ORIGIN_UNRESOLVED;
        return (NvmOwnerOriginResult){status,0,0,described.message};
    }
    OQAnalysis *a=calloc(1,sizeof *a);NvmOwnedArrayOrigins *p=calloc(1,sizeof *p);
    if(!a || !p) {free(a);free(p);nvm_owned_array_layouts_free(layouts);return (NvmOwnerOriginResult){NVM_OWNER_ORIGIN_MEMORY,0,0,"I could not allocate my bounded origin query."};}
    a->module=m;a->proof=p;a->error=invalid;p->layouts=layouts;p->counts.functions=m->function_count;
    if(!oq_paths(a) || !oq_descriptors(a) || !oq_decode(a))goto done;
    for(uint32_t f=0;f<m->function_count;f++)if(!oq_function(a,f))goto done;
    oq_reachable(a,0);
    for(unsigned w=1;w<NVM_OWNER_ORIGIN_WORDS;w++)if(p->summaries[0].required.words[w]) {
        oq_stop(a,NVM_OWNER_ORIGIN_UNRESOLVED,0,0,"I require closed entry obligations without formal bits.");goto done;
    }
    if(p->counts.sites<64 && (p->summaries[0].required.words[0]>>p->counts.sites)) {
        oq_stop(a,NVM_OWNER_ORIGIN_INVALID,0,0,"I require existing concrete entry origins.");goto done;
    }
    p->counts.obligations=a->count;
    a->error=(NvmOwnerOriginResult){NVM_OWNER_ORIGIN_PROVED,0,0,"I proved conditional owner origins and a closed entry only; execution authority remains separate."};
done:
    for(uint32_t n=0;n<a->count;n++)free(a->instructions[n].state);
    NvmOwnerOriginResult result=a->error;free(a->fields);free(a->paths);free(a);
    if(result.status==NVM_OWNER_ORIGIN_PROVED)*out=p;else nvm_owned_array_origins_free(p);
    return result;
}
bool nvm_owned_array_origin_counts(const NvmOwnedArrayOrigins *p,NvmOwnerOriginCounts *out) {
    if(!p || !out)return false;
    *out=p->counts;return true;
}
bool nvm_owned_array_origin_site(const NvmOwnedArrayOrigins *p,uint32_t n,NvmOwnerOriginSite *out) {
    if(!p || !out || n>=p->counts.sites)return false;
    *out=p->sites[n];return true;
}
bool nvm_owned_array_origin_summary(const NvmOwnedArrayOrigins *p,uint32_t f,NvmOwnerOriginSummary *out) {
    if(!p || !out || f>=p->counts.functions)return false;
    *out=p->summaries[f];return true;
}
bool nvm_owned_array_origin_input(const NvmOwnedArrayOrigins *p,uint32_t f,uint32_t n,NvmOwnerOriginPath *out) {
    if(!p || !out || f>=p->counts.functions || n>=p->summaries[f].formal_count)return false;
    *out=p->inputs[f][n];return true;
}
bool nvm_owned_array_origin_result(const NvmOwnedArrayOrigins *p,uint32_t f,uint32_t n,NvmOwnerOriginPath *out) {
    if(!p || !out || f>=p->counts.functions || n>=p->summaries[f].result_count)return false;
    *out=p->results[f][n];return true;
}
bool nvm_owned_array_origin_obligation(const NvmOwnedArrayOrigins *p,uint32_t n,NvmOwnerOriginObligation *out) {
    if(!p || !out || n>=p->counts.obligations)return false;
    *out=p->obligations[n];return true;
}
