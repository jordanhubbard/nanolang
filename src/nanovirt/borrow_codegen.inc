/* I lower a closed source borrow profile; ordinary lowering stays separate.
 * Every accepted module passes its exact owned or mixed verifier conjunction. */
#include "nanoisa/retained_layouts.h"
#include "nanoisa/ownership_contracts.h"
#include "nanoisa/verifier.h"
#include "nanoisa/mixed_samples_internal.h"
#include "nanoisa/owned_array_admission.h"
#include "nanoisa/service_bindings_module.h"

typedef struct BorrowLoop {
    struct BorrowLoop *parent;
    unsigned first, break_count;
    int index;
    uint32_t top, breaks[256];
    bool live[256], pending[256];
} BorrowLoop;

typedef struct {
    UnionDef *definition;
    TypeInfo *arguments;
    char *name;
    bool resource;
} BorrowUnion;

typedef struct {
    CG *cg;
    Environment *env;
    BorrowLoop *loop;
    ASTNode *current_source;
    ASTNode **programs;
    unsigned program_count;
    ASTNode *records[MAX_STRUCT_DEFS], *helper, *functions[8];
    BorrowUnion unions[256];
    unsigned function_count, current_function;
    bool value_graph, mixed, owner_arrays, union_graph;
    bool ordinary[MAX_STRUCT_DEFS];
    uint32_t result_layout, expression_layout;
    unsigned record_count, union_count, count, control_depth;
    unsigned depths[MAX_STRUCT_DEFS], path_count;
    uint16_t paths[256][32], path_lengths[256];
    int projection_parent[256], projection_field[256];
    const char *names[256], *advisory_names[256];
    uint32_t name_begin[256], name_end[256];
    bool name_open[256];
    NvmLocalBinding pending_names[8*256];
    unsigned pending_name_count;
    uint8_t tags[256], modes[256];
    uint32_t layouts[256];
    int16_t variants[256];
    bool live[256], disposal_pending[256], mutable[256], helper_body, returned, transferred, consuming;
    uint8_t result;
    uint8_t *contracts;
    size_t contract_size, contract_capacity;
} BorrowCG;

static void borrow_error(BorrowCG *b, ASTNode *n, const char *message) {
    cg_error(b->cg,n?n->line:0,"I require %s in my source borrow profile",message);
}
static uint8_t borrow_tag(Type type) {
    return type==TYPE_INT?TAG_INT:type==TYPE_BOOL?TAG_BOOL:TAG_VOID;
}
/* I keep retained fields separate from numeric and reference scalar tags. */
static uint8_t borrow_field_tag(Type type) {
    return type==TYPE_STRING?TAG_STRING:borrow_tag(type);
}
/* I do not grant float fields or signatures by admitting value locals. */
static uint8_t borrow_local_tag(Type type) {
    return type==TYPE_FLOAT?TAG_FLOAT:borrow_field_tag(type);
}
/* I keep parameter transport separate from resource-leaf eligibility. */
static uint8_t borrow_parameter_tag(Type type) {
    return type==TYPE_STRING?TAG_STRING:borrow_tag(type);
}
static uint8_t borrow_type_info_tag(const TypeInfo *info) {
    if(!info || info->generic_name || info->type_param_count)return TAG_VOID;
    return borrow_local_tag(info->base_type);
}
#ifdef NANOVIRT_TEST_CONTRACT_REALLOC
extern void *nanovirt_test_contract_realloc(void *pointer,size_t size);
#define BORROW_CONTRACT_REALLOC(pointer,size) nanovirt_test_contract_realloc((pointer),(size))
#else
#define BORROW_CONTRACT_REALLOC(pointer,size) realloc((pointer),(size))
#endif
static void borrow_word(BorrowCG *b,uint32_t value,unsigned width) {
    if(b->cg->had_error)return;
    if(b->contract_size+width>b->contract_capacity) {
        size_t capacity=b->contract_capacity?b->contract_capacity*2:128;
        uint8_t *next=BORROW_CONTRACT_REALLOC(b->contracts,capacity);
        if(!next){borrow_error(b,NULL,"available contract storage");return;}
        b->contracts=next;b->contract_capacity=capacity;
    }
    for(unsigned i=0;i<width;i++)b->contracts[b->contract_size++]=(uint8_t)(value>>(8*i));
}
static bool borrow_patch_word(BorrowCG *b,size_t at,uint32_t value,unsigned width) {
    if(b->cg->had_error)return false;
    if(width>4 || at>b->contract_size || width>b->contract_size-at) {
        borrow_error(b,NULL,"a retained contract length reservation");return false;
    }
    for(unsigned i=0;i<width;i++)b->contracts[at+i]=(uint8_t)(value>>(8*i));
    return true;
}
#undef BORROW_CONTRACT_REALLOC
static void borrow_descriptor(BorrowCG *b,uint8_t tag,uint8_t mode,uint32_t layout) {
    if(tag==TAG_UNION && layout!=NVM_V2_NO_INDEX)layout+=b->record_count;
    borrow_word(b,tag,1);borrow_word(b,mode,1);borrow_word(b,0,2);borrow_word(b,layout,4);
}
static int borrow_record(BorrowCG *b,const char *name) {
    for(unsigned i=0;name && i<b->record_count;i++)
        if(!strcmp(name,b->records[i]->as.struct_def.name))return (int)i;
    return -1;
}
/* I classify ordinary storage without granting an owner field or signature. */
static bool borrow_ordinary_local(BorrowCG *b,unsigned local) {
    return b->tags[local]==TAG_STRUCT && b->layouts[local]<b->record_count &&
           b->ordinary[b->layouts[local]];
}
static bool borrow_float_array_field(ASTNode *record,int field) {
    return record->as.struct_def.field_types[field]==TYPE_ARRAY &&
           record->as.struct_def.field_element_types &&
           record->as.struct_def.field_element_types[field]==TYPE_FLOAT;
}
/* I share exact ARRAY expression support, not ordinary record authority. */
static bool borrow_arrays(BorrowCG *b) { return b->mixed || b->owner_arrays; }
static uint8_t borrow_owner_field_tag(BorrowCG *b,unsigned layout,int field) {
    ASTNode *record=b->records[layout];
    if(b->owner_arrays && !b->ordinary[layout] && borrow_float_array_field(record,field))return TAG_ARRAY;
    return borrow_field_tag(record->as.struct_def.field_types[field]);
}
/* I select only the scalar instructions admitted by the owner ARRAY plan. */
static uint8_t borrow_scalar_opcode(BorrowCG *b,uint8_t opcode,uint8_t tag) {
    if(!b->owner_arrays)return opcode;
    if(tag==TAG_INT)switch(opcode) {
        case OP_ADD:return OP_I64_ADD;case OP_SUB:return OP_I64_SUB;
        case OP_MUL:return OP_I64_MUL;case OP_DIV:return OP_I64_DIV_S;
        case OP_MOD:return OP_I64_REM_S;case OP_NEG:return OP_I64_NEG;
        case OP_EQ:case OP_NE:return opcode;
        case OP_LT:return OP_I64_LT_S;case OP_LE:return OP_I64_LE_S;
        case OP_GT:return OP_I64_GT_S;case OP_GE:return OP_I64_GE_S;
        default:return 0;
    }
    if(tag==TAG_BOOL)switch(opcode) {
        case OP_AND:return OP_BOOL_AND;case OP_OR:return OP_BOOL_OR;
        case OP_NOT:return OP_BOOL_NOT;case OP_EQ:case OP_NE:return opcode;
        default:return 0;
    }
    return tag==TAG_FLOAT?opcode:0;
}
static int borrow_local(BorrowCG *b,const char *name) {
    for(unsigned i=b->count;i>0;i--)
        if(name && b->names[i-1] && !strcmp(name,b->names[i-1]))return (int)i-1;
    return -1;
}
static bool borrow_name_append(char *out,size_t capacity,size_t *used,const char *text) {
    size_t n=text?strlen(text):0;
    if(!out || !used || *used>=capacity || n>=capacity-*used)return false;
    memcpy(out+*used,text,n);*used+=n;out[*used]='\0';return true;
}
static bool borrow_type_name(const TypeInfo *info,char *out,size_t capacity,size_t *used,unsigned depth) {
    if(!info || depth>32)return false;
    const char *scalar=info->base_type==TYPE_INT?"int":info->base_type==TYPE_FLOAT?"float":
        info->base_type==TYPE_BOOL?"bool":info->base_type==TYPE_STRING?"string":NULL;
    if(!info->generic_name)return scalar && borrow_name_append(out,capacity,used,scalar);
    if(!borrow_name_append(out,capacity,used,info->generic_name))return false;
    if(!info->type_param_count)return true;
    if(!info->type_params || !borrow_name_append(out,capacity,used,"<"))return false;
    for(int i=0;i<info->type_param_count;i++) {
        if(i && !borrow_name_append(out,capacity,used,","))return false;
        if(!borrow_type_name(info->type_params[i],out,capacity,used,depth+1))return false;
    }
    return borrow_name_append(out,capacity,used,">");
}
static bool borrow_union_arguments_equal(const TypeInfo *left,const TypeInfo *right) {
    if(!left || !right || left->type_param_count!=right->type_param_count ||
       left->type_param_count<0 || (left->type_param_count && (!left->type_params || !right->type_params)))return false;
    for(int i=0;i<left->type_param_count;i++)
        if(!type_infos_equal(left->type_params[i],right->type_params[i]))return false;
    return true;
}
static int borrow_union_instance(BorrowCG *b,const TypeInfo *info) {
    if(!info || !info->generic_name)return -1;
    UnionDef *definition=env_get_union(b->env,info->generic_name);
    if(!definition)return -1;
    for(unsigned i=0;i<b->union_count;i++)
        if(b->unions[i].definition==definition && borrow_union_arguments_equal(b->unions[i].arguments,info))return (int)i;
    return -1;
}
static int borrow_named_union(BorrowCG *b,const TypeInfo *info,const char *name) {
    if(info)return borrow_union_instance(b,info);
    TypeInfo concrete={0};concrete.base_type=TYPE_UNION;concrete.generic_name=(char *)name;
    return borrow_union_instance(b,&concrete);
}
/* I validate identity arguments without treating unused parameters as owners. */
static bool borrow_concrete_type(BorrowCG *b,const TypeInfo *info,unsigned depth) {
    if(!info || depth>32)return false;
    if(borrow_type_info_tag(info))return true;
    if(!info->generic_name || info->type_param_count<0 ||
       (info->type_param_count && !info->type_params))return false;
    int record=borrow_record(b,info->generic_name);
    if(record>=0)return !info->type_param_count;
    UnionDef *definition=env_get_union(b->env,info->generic_name);
    if(!definition || definition->is_extern || definition->generic_param_count!=info->type_param_count)return false;
    for(int i=0;i<info->type_param_count;i++)
        if(!borrow_concrete_type(b,info->type_params[i],depth+1))return false;
    return true;
}
static bool borrow_register_union_depth(BorrowCG *b,const TypeInfo *info,ASTNode *at,unsigned depth) {
    if(!info || !info->generic_name || !env_get_union(b->env,info->generic_name))return true;
    if(borrow_union_instance(b,info)>=0)return true;
    UnionDef *definition=env_get_union(b->env,info->generic_name);
    if(depth>32 || !borrow_concrete_type(b,info,0) || definition->variant_count<=0) {
        borrow_error(b,at,"a bounded exact concrete union instance");return false;
    }
    bool resource=false;unsigned fields=0;
    for(int variant=0;variant<definition->variant_count;variant++) {
        int count=definition->variant_field_counts[variant];
        if(count<0 || fields+(unsigned)count>UINT16_MAX) {
            borrow_error(b,at,"bounded concrete union fields");return false;
        }
        for(int field=0;field<count;field++) {
            TypeInfo *payload=resolve_union_payload_type_info(definition,variant,field,info);
            uint8_t tag=borrow_type_info_tag(payload);
            if(!tag && payload && payload->generic_name) {
                int record=borrow_record(b,payload->generic_name);
                if(record>=0 && !payload->type_param_count && !b->ordinary[record]) {
                    tag=TAG_STRUCT;resource=true;b->union_graph=true;
                } else if(env_get_union(b->env,payload->generic_name) &&
                          borrow_register_union_depth(b,payload,at,depth+1)) {
                    int child=borrow_union_instance(b,payload);
                    if(child>=0) {tag=TAG_UNION;resource|=b->unions[child].resource;b->union_graph=true;}
                }
            }
            free_payload_type_info(payload);
            if(!tag || b->cg->had_error) {
                if(!b->cg->had_error)borrow_error(b,at,"exact scalar, resource-record or concrete union payloads");
                return false;
            }
        }
        fields+=(unsigned)count;
    }
    if(b->union_count==256){borrow_error(b,at,"at most 256 concrete unions");return false;}
    char spelling[4096]="";size_t used=0;
    if(!borrow_type_name(info,spelling,sizeof(spelling),&used,0)) {
        borrow_error(b,at,"a bounded concrete union identity");return false;
    }
    char *name=strdup(spelling);
    if(!name){borrow_error(b,at,"available concrete union identity storage");return false;}
    /* Resolved nested payload trees are freed by my caller. I own this copy. */
    TypeInfo *arguments=copy_payload_type_info(info);
    b->unions[b->union_count++]=(BorrowUnion){definition,arguments,name,resource};return true;
}
static bool borrow_register_union(BorrowCG *b,const TypeInfo *info,ASTNode *at) {
    return borrow_register_union_depth(b,info,at,0);
}
static bool borrow_register_named_union(BorrowCG *b,const TypeInfo *info,const char *name,ASTNode *at) {
    if(info)return borrow_register_union(b,info,at);
    TypeInfo concrete={0};concrete.base_type=TYPE_UNION;concrete.generic_name=(char *)name;
    return borrow_register_union(b,&concrete,at);
}
static bool borrow_owned_union(BorrowCG *b,uint32_t instance) {
    return instance<b->union_count && b->unions[instance].resource;
}
static bool borrow_owned_local(BorrowCG *b,unsigned local) {
    return (b->tags[local]==TAG_STRUCT && !borrow_ordinary_local(b,local)) ||
           (b->tags[local]==TAG_UNION && borrow_owned_union(b,b->layouts[local]));
}
static int borrow_slot(BorrowCG *b,const char *name,uint8_t tag,uint32_t layout,uint8_t mode,bool mutable) {
    if(b->count==256){borrow_error(b,NULL,"at most 256 local slots");return 0;}
    unsigned i=b->count++;b->projection_parent[i]=-1;b->projection_field[i]=-1;b->advisory_names[i]=NULL;b->name_open[i]=false;b->names[i]=name;b->tags[i]=tag;b->layouts[i]=layout;
    b->variants[i]=-1;b->modes[i]=mode;b->live[i]=true;b->disposal_pending[i]=false;b->mutable[i]=mutable;return (int)i;
}
/* I retain lexical positions, not runtime owner liveness. */
static void borrow_name_begin(BorrowCG *b,unsigned slot,const char *name) {
    if(b->cg->had_error || !name || !*name)return;
    b->advisory_names[slot]=name;b->name_begin[slot]=b->cg->code_size;b->name_open[slot]=true;
}
static void borrow_names_end(BorrowCG *b,unsigned first) {
    for(unsigned i=first;i<b->count;i++)if(b->name_open[i]) {
        b->name_end[i]=b->cg->code_size;b->name_open[i]=false;
    }
}
static void borrow_name_retain(BorrowCG *b,unsigned slot) {
    if(!b->advisory_names[slot] || b->cg->had_error)return;
    if(b->pending_name_count>=8*256){borrow_error(b,NULL,"bounded lexical-name storage");return;}
    NvmLocalBinding binding={.function=b->current_function,.slot=(uint16_t)slot,
        .begin=b->name_begin[slot],.end=b->name_end[slot],
        .name=(const uint8_t *)b->advisory_names[slot],.name_size=(uint32_t)strlen(b->advisory_names[slot])};
    b->pending_names[b->pending_name_count++]=binding;b->advisory_names[slot]=NULL;
}
static int borrow_child(BorrowCG *b,unsigned layout,int field) {
    ASTNode *record=b->records[layout];
    return record->as.struct_def.field_types[field]==TYPE_STRUCT
        ?borrow_record(b,record->as.struct_def.field_type_names[field]):-1;
}
static bool borrow_leaf(BorrowCG *b,unsigned layout) {
    ASTNode *record=b->records[layout];
    for(int i=0;i<record->as.struct_def.field_count;i++)
        if(!borrow_tag(record->as.struct_def.field_types[i]))return false;
    return true;
}
static int borrow_layout_field(BorrowCG *b,unsigned layout,const char *name,ASTNode *n) {
    ASTNode *record=b->records[layout];
    for(int i=0;i<record->as.struct_def.field_count;i++)
        if(!strcmp(name,record->as.struct_def.field_names[i]))return i;
    borrow_error(b,n,"a declared resource field");return -1;
}
typedef struct { int root;unsigned layout,length;uint16_t fields[32]; } BorrowPlace;
static bool borrow_place(BorrowCG *b,ASTNode *n,BorrowPlace *place) {
    if(!n || b->cg->had_error)return false;
    if(n->type==AST_IDENTIFIER) {
        int local=borrow_local(b,n->as.identifier);
        if(local<0 || !b->live[local] || b->tags[local]!=TAG_STRUCT || borrow_ordinary_local(b,(unsigned)local)) {
            borrow_error(b,n,"a live exact resource place");return false;
        }
        place->root=local;place->layout=b->layouts[local];place->length=0;return true;
    }
    if(n->type==AST_FIELD_ACCESS && borrow_place(b,n->as.field_access.object,place)) {
        int field=borrow_layout_field(b,place->layout,n->as.field_access.field_name,n);
        if(field<0)return false;
        int child=borrow_child(b,place->layout,field);
        if(child<0 || place->length==32){borrow_error(b,n,"a resource-record path of at most 32 fields");return false;}
        place->fields[place->length++]=(uint16_t)field;place->layout=(unsigned)child;return true;
    }
    borrow_error(b,n,"a root-local resource field chain");return false;
}
static void borrow_reference(BorrowCG *b,const BorrowPlace *place,unsigned reference,uint8_t mode) {
    if(!borrow_leaf(b,place->layout)){borrow_error(b,NULL,"a scalar-leaf resource referent");return;}
    if(!place->length) {
        emit_op(b->cg,mode==2?OP_BORROW_LOCAL_EXCLUSIVE:OP_BORROW_LOCAL_SHARED,(int)reference,place->root);return;
    }
    unsigned path=0;
    while(path<b->path_count && (b->path_lengths[path]!=place->length ||
          memcmp(b->paths[path],place->fields,place->length*sizeof(uint16_t))))path++;
    if(path==b->path_count) {
        if(path==256){borrow_error(b,NULL,"at most 256 numeric resource paths");return;}
        b->path_lengths[path]=(uint16_t)place->length;
        memcpy(b->paths[path],place->fields,place->length*sizeof(uint16_t));b->path_count++;
    }
    emit_op(b->cg,mode==2?OP_BORROW_PATH_EXCLUSIVE:OP_BORROW_PATH_SHARED,(int)reference,place->root,(uint32_t)path);
}
static bool borrow_overlap(const BorrowPlace *a,const BorrowPlace *b) {
    unsigned common=a->length<b->length?a->length:b->length;
    return a->root==b->root && !memcmp(a->fields,b->fields,common*sizeof(uint16_t));
}
/* I consume the shell once and retain each child in its exact owned slot. */
static void borrow_unpack(BorrowCG *b,unsigned local) {
    ASTNode *record=b->records[b->layouts[local]];int slots[256];
    for(int f=0;f<record->as.struct_def.field_count && !b->cg->had_error;f++) {
        int child=borrow_child(b,b->layouts[local],f);
        slots[f]=borrow_slot(b,NULL,child>=0?TAG_STRUCT:borrow_owner_field_tag(b,b->layouts[local],f),
                            child>=0?(uint32_t)child:NVM_V2_NO_INDEX,0,false);
        b->projection_parent[slots[f]]=(int)local;b->projection_field[slots[f]]=f;
    }
    if(b->cg->had_error)return;
    emit_op(b->cg,OP_OWN_UNPACK_LOCAL,(int)local);b->live[local]=false;b->disposal_pending[local]=false;
    for(int f=record->as.struct_def.field_count-1;f>=0;f--)
        emit_op(b->cg,b->tags[slots[f]]==TAG_STRUCT?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slots[f]);
}
static void borrow_cleanup(BorrowCG *b,unsigned first) {
    for(unsigned i=first;i<b->count && !b->cg->had_error;i++) {
        if(b->live[i] && b->tags[i]==TAG_UNION && borrow_owned_union(b,b->layouts[i]) && b->disposal_pending[i]) {
            int variant=b->variants[i];
            if(variant<0){borrow_error(b,NULL,"a selected ordinary union arm for disposal");return;}
            int fields=b->unions[b->layouts[i]].definition->variant_field_counts[variant];
            emit_op(b->cg,OP_OWN_UNPACK_VARIANT,(int)i,variant,fields);
            for(int j=0;j<fields;j++)emit_op(b->cg,OP_POP);
            b->live[i]=false;b->disposal_pending[i]=false;
        }
        if(b->live[i] && b->tags[i]==TAG_STRUCT && !borrow_ordinary_local(b,i) && !b->modes[i]) {
            if(!borrow_leaf(b,b->layouts[i])){borrow_unpack(b,i);continue;}
            emit_op(b->cg,OP_OWN_UNPACK_LOCAL,(int)i);
            int fields=b->records[b->layouts[i]]->as.struct_def.field_count;
            for(int j=0;j<fields;j++)emit_op(b->cg,OP_POP);
            b->live[i]=false;b->disposal_pending[i]=false;
        }
    }
}
static bool borrow_consumed(BorrowCG *b,unsigned first,ASTNode *at) {
    for(unsigned i=first;i<b->count;i++)
        if(b->live[i] && borrow_owned_local(b,i) && !b->modes[i] && !b->disposal_pending[i]) {
            borrow_error(b,at,"explicit source owner consumption at scope exit");return false;
        }
    return !b->cg->had_error;
}
static bool borrow_same_owners(BorrowCG *b,unsigned count,const bool *live,const bool *pending,ASTNode *at) {
    for(unsigned i=0;i<count;i++)if(borrow_owned_local(b,i) &&
       (b->live[i]!=live[i] || b->disposal_pending[i]!=pending[i])) {
        borrow_error(b,at,"matching ownership on every reaching edge");return false;
    }
    return !b->cg->had_error;
}
static int borrow_function_index(BorrowCG *b,const char *name) {
    for(unsigned i=0;name && i<b->function_count;i++)
        if(b->functions[i] && !strcmp(name,b->functions[i]->as.function.name))return (int)i;
    return -1;
}
static uint8_t borrow_result_tag(BorrowCG *b,ASTNode *function) {
    if(borrow_named_union(b,function->as.function.return_type_info,function->as.function.return_struct_type_name)>=0)return TAG_UNION;
    return function->as.function.return_type==TYPE_STRUCT?TAG_STRUCT:borrow_local_tag(function->as.function.return_type);
}
static bool borrow_owner(BorrowCG *b,ASTNode *value,unsigned layout);
static uint8_t borrow_expr(BorrowCG *b,ASTNode *n);
static bool borrow_ordinary(BorrowCG *b,ASTNode *value,unsigned layout);
static uint8_t borrow_match_expr(BorrowCG *b,ASTNode *n);
static uint8_t borrow_owned_match(BorrowCG *b,ASTNode *n,int instance,int expected,bool value,bool shadow);
static int borrow_match_source(BorrowCG *b,ASTNode *n);
static void borrow_match_statement(BorrowCG *b,ASTNode *n,bool shadow);
static void borrow_body(BorrowCG *b,ASTNode *body,bool shadow);
static bool borrow_declares_match_binding(ASTNode *n,const char *name) {
    if(!n || !name)return false;
    switch(n->type) {
        case AST_MATCH:
            for(int i=0;i<n->as.match_expr.arm_count;i++) {
                if(n->as.match_expr.pattern_bindings[i] &&
                   !strcmp(n->as.match_expr.pattern_bindings[i],name))return true;
                if(n->as.match_expr.guard_exprs && borrow_declares_match_binding(n->as.match_expr.guard_exprs[i],name))return true;
                if(borrow_declares_match_binding(n->as.match_expr.arm_bodies[i],name))return true;
            }
            return borrow_declares_match_binding(n->as.match_expr.expr,name);
        case AST_BLOCK:
            for(int i=0;i<n->as.block.count;i++)if(borrow_declares_match_binding(n->as.block.statements[i],name))return true;
            return false;
        case AST_UNSAFE_BLOCK:
            for(int i=0;i<n->as.unsafe_block.count;i++)if(borrow_declares_match_binding(n->as.unsafe_block.statements[i],name))return true;
            return false;
        case AST_IF:return borrow_declares_match_binding(n->as.if_stmt.condition,name) ||
            borrow_declares_match_binding(n->as.if_stmt.then_branch,name) ||
            borrow_declares_match_binding(n->as.if_stmt.else_branch,name);
        case AST_WHILE:return borrow_declares_match_binding(n->as.while_stmt.condition,name) ||
            borrow_declares_match_binding(n->as.while_stmt.body,name);
        case AST_FOR:return borrow_declares_match_binding(n->as.for_stmt.range_expr,name) ||
            borrow_declares_match_binding(n->as.for_stmt.body,name);
        case AST_LET:return borrow_declares_match_binding(n->as.let.value,name);
        case AST_SET:return borrow_declares_match_binding(n->as.set.value,name);
        case AST_RETURN:return borrow_declares_match_binding(n->as.return_stmt.value,name);
        case AST_ASSERT:return borrow_declares_match_binding(n->as.assert.condition,name);
        case AST_PRINT:return borrow_declares_match_binding(n->as.print.expr,name);
        case AST_FIELD_ACCESS:return borrow_declares_match_binding(n->as.field_access.object,name);
        case AST_CALL:
            if(borrow_declares_match_binding(n->as.call.func_expr,name))return true;
            for(int i=0;i<n->as.call.arg_count;i++)if(borrow_declares_match_binding(n->as.call.args[i],name))return true;
            return false;
        case AST_PREFIX_OP:
            for(int i=0;i<n->as.prefix_op.arg_count;i++)if(borrow_declares_match_binding(n->as.prefix_op.args[i],name))return true;
            return false;
        default:return false;
    }
}
static uint8_t borrow_union_payload(BorrowCG *b,unsigned instance,int variant,int field,
                                    uint32_t *layout,ASTNode *at) {
    *layout=NVM_V2_NO_INDEX;
    if(instance>=b->union_count){borrow_error(b,at,"a registered concrete union");return TAG_VOID;}
    UnionDef *definition=b->unions[instance].definition;
    if(variant<0 || variant>=definition->variant_count || field<0 ||
       field>=definition->variant_field_counts[variant]) {
        borrow_error(b,at,"a declared concrete union payload field");return TAG_VOID;
    }
    TypeInfo *payload=resolve_union_payload_type_info(definition,variant,field,b->unions[instance].arguments);
    uint8_t tag=borrow_type_info_tag(payload);
    if(!tag && payload && payload->generic_name) {
        int record=borrow_record(b,payload->generic_name),child=borrow_union_instance(b,payload);
        if(record>=0 && !payload->type_param_count && !b->ordinary[record]) {tag=TAG_STRUCT;*layout=(uint32_t)record;}
        else if(child>=0 && (unsigned)child<instance) {tag=TAG_UNION;*layout=(uint32_t)child;}
    }
    free_payload_type_info(payload);
    if(!tag)borrow_error(b,at,"an exact prior concrete union payload layout");
    return tag;
}
static uint8_t borrow_union_payload_tag(BorrowCG *b,unsigned instance,int variant,int field,ASTNode *at) {
    uint32_t layout;return borrow_union_payload(b,instance,variant,field,&layout,at);
}
static int borrow_union_variant(BorrowCG *b,unsigned instance,const char *name,ASTNode *at) {
    if(instance>=b->union_count){borrow_error(b,at,"a registered concrete scalar union");return -1;}
    UnionDef *definition=b->unions[instance].definition;
    for(int i=0;i<definition->variant_count;i++)
        if(name && !strcmp(name,definition->variant_names[i]))return i;
    borrow_error(b,at,"a declared concrete union variant");return -1;
}
static uint8_t borrow_union_expr(BorrowCG *b,ASTNode *n,int expected) {
    if(!n || b->cg->had_error)return TAG_VOID;
    ASTNode constructor;char declaration[4096];
    if(n->type==AST_STRUCT_LITERAL && n->as.struct_literal.struct_name) {
        const char *name=n->as.struct_literal.struct_name,*dot=strrchr(name,'.');
        size_t length=dot?(size_t)(dot-name):0;
        if(!dot || length>=sizeof(declaration) || n->as.struct_literal.spread_source) {
            borrow_error(b,n,"an exact named union constructor without spread");return TAG_VOID;
        }
        memcpy(declaration,name,length);declaration[length]='\0';
        constructor=*n;memset(&constructor.as,0,sizeof(constructor.as));constructor.type=AST_UNION_CONSTRUCT;
        constructor.as.union_construct.union_name=declaration;
        constructor.as.union_construct.variant_name=(char *)(dot+1);
        constructor.as.union_construct.field_names=n->as.struct_literal.field_names;
        constructor.as.union_construct.field_values=n->as.struct_literal.field_values;
        constructor.as.union_construct.field_count=n->as.struct_literal.field_count;
        n=&constructor;
    }
    if(n->type==AST_MATCH) {
        int instance=borrow_match_source(b,n);
        if(instance<0)return TAG_VOID;
        if(borrow_owned_union(b,(uint32_t)instance))return borrow_owned_match(b,n,instance,expected,true,false);
        borrow_error(b,n,"an owned source for contextual union-valued matching");return TAG_VOID;
    }
    if(n->type!=AST_UNION_CONSTRUCT) {
        uint8_t tag=borrow_expr(b,n);
        if(tag!=TAG_UNION || (expected>=0 && b->expression_layout!=(uint32_t)expected))
            borrow_error(b,n,"the exact concrete union value");
        return b->cg->had_error?TAG_VOID:TAG_UNION;
    }
    int instance=borrow_named_union(b,n->as.union_construct.type_info,n->as.union_construct.union_name);
    if(instance<0)instance=expected;
    if(instance<0 || (expected>=0 && instance!=expected) || (unsigned)instance>=b->union_count) {
        borrow_error(b,n,"a registered exact concrete scalar union constructor");return TAG_VOID;
    }
    BorrowUnion *u=&b->unions[instance];
    if(env_get_union(b->env,n->as.union_construct.union_name)!=u->definition) {
        borrow_error(b,n,"the exact union constructor declaration");return TAG_VOID;
    }
    int variant=borrow_union_variant(b,(unsigned)instance,n->as.union_construct.variant_name,n);
    if(variant<0)return TAG_VOID;
    int count=u->definition->variant_field_counts[variant];int slots[256];
    if(count<0 || count>256 || n->as.union_construct.field_count!=count) {
        borrow_error(b,n,"every union field exactly once");return TAG_VOID;
    }
    for(int i=0;i<count;i++)slots[i]=-1;
    for(int i=0;i<n->as.union_construct.field_count && !b->cg->had_error;i++) {
        int selected=-1;
        for(int f=0;f<count;f++)
            if(!strcmp(n->as.union_construct.field_names[i],u->definition->variant_field_names[variant][f]))selected=f;
        if(selected<0 || slots[selected]>=0) {
            borrow_error(b,n,"each declared union field once");return TAG_VOID;
        }
        uint32_t child;uint8_t expected_tag=borrow_union_payload(b,(unsigned)instance,variant,selected,&child,n);
        ASTNode *value=n->as.union_construct.field_values[i];
        uint8_t actual=expected_tag==TAG_STRUCT?(borrow_owner(b,value,child)?TAG_STRUCT:TAG_VOID):
                       expected_tag==TAG_UNION?borrow_union_expr(b,value,(int)child):borrow_expr(b,value);
        if(actual!=expected_tag) {
            borrow_error(b,n,"the exact concrete union field type");return TAG_VOID;
        }
        slots[selected]=borrow_slot(b,NULL,expected_tag,child,0,false);
        emit_op(b->cg,borrow_owned_local(b,(unsigned)slots[selected])?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slots[selected]);
    }
    for(int i=0;i<count;i++) {
        bool owned=borrow_owned_local(b,(unsigned)slots[i]);
        emit_op(b->cg,owned?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,slots[i]);
        if(owned)b->live[slots[i]]=false;
    }
    emit_op(b->cg,OP_AGG_PACK,AGG_VARIANT,instance,variant,count);
    b->expression_layout=(uint32_t)instance;return TAG_UNION;
}
static uint8_t borrow_float_array(BorrowCG *b,ASTNode *n,bool expected) {
    if(n->type!=AST_ARRAY_LITERAL)return borrow_expr(b,n);
    if(!borrow_arrays(b) || (!expected && !n->as.array_literal.element_count)) {borrow_error(b,n,"an exact contextual flat FLOAT array");return TAG_VOID;}
    for(int i=0;i<n->as.array_literal.element_count;i++)
        if(borrow_expr(b,n->as.array_literal.elements[i])!=TAG_FLOAT) {borrow_error(b,n,"exact FLOAT array elements");return TAG_VOID;}
    if(b->cg->had_error)return TAG_VOID;
    emit_op(b->cg,OP_ARR_LITERAL,TAG_FLOAT,n->as.array_literal.element_count);return TAG_ARRAY;
}
static uint8_t borrow_expr(BorrowCG *b,ASTNode *n) {
    CG *cg=b->cg;if(cg->had_error || !n)return TAG_VOID;
    if(n->type==AST_MATCH)return borrow_match_expr(b,n);
    if(n->type==AST_UNION_CONSTRUCT)return borrow_union_expr(b,n,-1);
    if(n->type==AST_ARRAY_LITERAL)return borrow_float_array(b,n,false);
    if(n->type==AST_STRING) {
        if(!b->value_graph){borrow_error(b,n,"string literals only in a value-call graph");return TAG_VOID;}
        const char *raw=n->as.string_val;
        for(size_t i=0;raw[i];i++)if(raw[i]=='\\' && raw[i+1]) {
            if(raw[++i]=='0'){borrow_error(b,n,"string literals without embedded NUL");return TAG_VOID;}
        }
        char *decoded=nl_unescape_string(raw);
        if(!decoded){borrow_error(b,n,"available literal storage");return TAG_VOID;}
        uint32_t index=nvm_add_string(cg->module,decoded,(uint32_t)strlen(decoded));free(decoded);
        if(index==UINT32_MAX){borrow_error(b,n,"available literal storage");return TAG_VOID;}
        emit_op(cg,OP_PUSH_STR,index);return TAG_STRING;
    }
    if(n->type==AST_NUMBER){emit_op(cg,OP_PUSH_I64,(int64_t)n->as.number);return TAG_INT;}
    if(n->type==AST_FLOAT){emit_op(cg,OP_PUSH_F64,n->as.float_val);return TAG_FLOAT;}
    if(n->type==AST_BOOL){emit_op(cg,OP_PUSH_BOOL,(int)n->as.bool_val);return TAG_BOOL;}
    if(n->type==AST_IDENTIFIER) {
        int local=borrow_local(b,n->as.identifier);
        if(local<0 || !b->live[local] || (b->tags[local]==TAG_STRUCT && !borrow_ordinary_local(b,(unsigned)local))) {
            borrow_error(b,n,"a live scalar value, not a copied owner/reference");return TAG_VOID;
        }
        b->expression_layout=b->layouts[local];
        bool owned=borrow_owned_local(b,(unsigned)local);
        emit_op(cg,owned?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,local);
        if(owned){b->live[local]=false;b->disposal_pending[local]=false;}
        return b->tags[local];
    }
    if(n->type==AST_FIELD_ACCESS) {
        ASTNode *object=n->as.field_access.object;
        int local=object->type==AST_IDENTIFIER?borrow_local(b,object->as.identifier):-1;
        if(local<0 && object->type==AST_IDENTIFIER &&
           borrow_declares_match_binding(b->current_source?b->current_source->as.function.body:NULL,
                                         object->as.identifier)) {
            borrow_error(b,n,"a payload binding inside its lexical match arm");return TAG_VOID;
        }
        if(local>=0 && b->tags[local]==TAG_UNION) {
            if(!b->live[local] || b->layouts[local]>=b->union_count || b->variants[local]<0) {
                borrow_error(b,n,"a live exact matched union payload");return TAG_VOID;
            }
            BorrowUnion *u=&b->unions[b->layouts[local]];int variant=b->variants[local],field=-1;
            for(int i=0;i<u->definition->variant_field_counts[variant];i++)
                if(!strcmp(n->as.field_access.field_name,u->definition->variant_field_names[variant][i]))field=i;
            if(field<0){borrow_error(b,n,"a field from the checked union variant");return TAG_VOID;}
            uint8_t tag=borrow_union_payload_tag(b,b->layouts[local],variant,field,n);
            if(tag==TAG_STRUCT || tag==TAG_UNION) {borrow_error(b,n,"complete selected extraction of aggregate payloads");return TAG_VOID;}
            emit_op(cg,OP_LOAD_LOCAL,local);emit_op(cg,OP_AGG_GET,field);return tag;
        }
        if(local>=0 && borrow_ordinary_local(b,(unsigned)local)) {
            if(!b->live[local]){borrow_error(b,n,"a live ordinary receiver");return TAG_VOID;}
            int field=borrow_layout_field(b,b->layouts[local],n->as.field_access.field_name,n);
            if(field<0)return TAG_VOID;
            ASTNode *record=b->records[b->layouts[local]];
            uint8_t tag=borrow_float_array_field(record,field)?TAG_ARRAY:borrow_tag(record->as.struct_def.field_types[field]);
            emit_op(cg,OP_LOAD_LOCAL,local);emit_op(cg,OP_AGG_GET,field);return tag;
        }
        BorrowPlace place;
        if(!borrow_place(b,n->as.field_access.object,&place))return TAG_VOID;
        int field=borrow_layout_field(b,place.layout,n->as.field_access.field_name,n);
        if(field<0)return TAG_VOID;
        uint8_t tag=borrow_owner_field_tag(b,place.layout,field);
        if(b->owner_arrays) {
            if(!tag || b->modes[place.root] || b->disposal_pending[place.root]) {
                borrow_error(b,n,"an intact mode-zero owner leaf observation");return TAG_VOID;
            }
            emit_op(cg,OP_LOAD_LOCAL,place.root);
            for(unsigned i=0;i<place.length;i++)emit_op(cg,OP_AGG_GET,place.fields[i]);
            emit_op(cg,OP_AGG_GET,field);return tag;
        }
        if(tag==TAG_STRING && b->value_graph && !place.length && !b->modes[place.root]) {
            emit_op(cg,OP_LOAD_LOCAL,place.root);emit_op(cg,OP_AGG_GET,field);return TAG_STRING;
        }
        if(!tag || !borrow_leaf(b,place.layout)){borrow_error(b,n,"a scalar field of a leaf resource record");return TAG_VOID;}
        /* I keep temporary local observations after inherited helper formals. */
        unsigned reference=b->modes[place.root]?(unsigned)place.root:
            b->helper_body?(unsigned)b->helper->as.function.param_count:0;
        if(!b->modes[place.root]) { emit_op(cg,OP_REGION_BEGIN);borrow_reference(b,&place,reference,1); }
        emit_op(cg,OP_REF_GET,reference,field);
        if(!b->modes[place.root])emit_op(cg,OP_REGION_END);
        return tag;
    }
    if(n->type==AST_PREFIX_OP) {
        int count=n->as.prefix_op.arg_count;TokenType op=n->as.prefix_op.op;
        if(count<1 || count>2){borrow_error(b,n,"one unary or two binary scalar operands");return TAG_VOID;}
        if(op==TOKEN_NOT && (borrow_local(b,"not")>=0 || borrow_function_index(b,"not")>=0)) {
            borrow_error(b,n,"an unshadowed Boolean not builtin");return TAG_VOID;
        }
        uint8_t left=borrow_expr(b,n->as.prefix_op.args[0]);
        if(count==1) {
            if(op==TOKEN_NOT && left==TAG_BOOL){emit_op(cg,borrow_scalar_opcode(b,OP_NOT,TAG_BOOL));return TAG_BOOL;}
            if(op==TOKEN_MINUS && left==TAG_INT){emit_op(cg,borrow_scalar_opcode(b,OP_NEG,TAG_INT));return TAG_INT;}
            if(op==TOKEN_MINUS && left==TAG_FLOAT){emit_op(cg,OP_F64_NEG);return TAG_FLOAT;}
        } else {
            uint8_t right=borrow_expr(b,n->as.prefix_op.args[1]);uint8_t opcode=0,result=left;
            if(left!=right){borrow_error(b,n,"matching scalar operand types");return TAG_VOID;}
            if(left==TAG_INT) switch(op) {
                case TOKEN_PLUS:opcode=OP_ADD;break;case TOKEN_MINUS:opcode=OP_SUB;break;
                case TOKEN_STAR:opcode=OP_MUL;break;case TOKEN_SLASH:opcode=OP_DIV;break;
                case TOKEN_PERCENT:opcode=OP_MOD;break;default:break;
            }
            if(left==TAG_BOOL) {
                if(op==TOKEN_AND)opcode=OP_AND;
                if(op==TOKEN_OR)opcode=OP_OR;
            }
            if(left==TAG_FLOAT) switch(op) {
                case TOKEN_PLUS:opcode=OP_F64_ADD;break;case TOKEN_MINUS:opcode=OP_F64_SUB;break;
                case TOKEN_STAR:opcode=OP_F64_MUL;break;case TOKEN_SLASH:opcode=OP_F64_DIV;break;
                case TOKEN_EQ:opcode=OP_F64_EQ;result=TAG_BOOL;break;
                case TOKEN_NE:opcode=OP_F64_NE;result=TAG_BOOL;break;
                case TOKEN_LT:opcode=OP_F64_LT;result=TAG_BOOL;break;
                case TOKEN_LE:opcode=OP_F64_LE;result=TAG_BOOL;break;
                case TOKEN_GT:opcode=OP_F64_GT;result=TAG_BOOL;break;
                case TOKEN_GE:opcode=OP_F64_GE;result=TAG_BOOL;break;default:break;
            }
            if(left==TAG_STRING && b->value_graph) {
                if(op==TOKEN_EQ){opcode=OP_EQ;result=TAG_BOOL;}
                if(op==TOKEN_NE){opcode=OP_NE;result=TAG_BOOL;}
            }
            if(left==TAG_INT || left==TAG_BOOL) switch(op) {
                case TOKEN_EQ:opcode=OP_EQ;result=TAG_BOOL;break;
                case TOKEN_NE:opcode=OP_NE;result=TAG_BOOL;break;
                case TOKEN_LT:opcode=OP_LT;result=TAG_BOOL;break;
                case TOKEN_LE:opcode=OP_LE;result=TAG_BOOL;break;
                case TOKEN_GT:opcode=OP_GT;result=TAG_BOOL;break;
                case TOKEN_GE:opcode=OP_GE;result=TAG_BOOL;break;default:break;
            }
            opcode=borrow_scalar_opcode(b,opcode,left);
            if(opcode){emit_op(cg,opcode);return result;}
        }
        borrow_error(b,n,"supported exact-tag scalar operators");return TAG_VOID;
    }
    if(n->type==AST_MODULE_QUALIFIED_CALL && b->value_graph) {
        ASTNode call={0};
        call.type=AST_CALL;call.line=n->line;call.column=n->column;
        call.as.call.name=n->as.module_qualified_call.function_name;
        call.as.call.args=n->as.module_qualified_call.args;
        call.as.call.arg_count=n->as.module_qualified_call.arg_count;
        return borrow_expr(b,&call);
    }
    if(n->type==AST_CALL && b->owner_arrays && !n->as.call.func_expr && !n->as.call.borrow_mode &&
       n->as.call.name && borrow_function_index(b,n->as.call.name)<0 && borrow_local(b,n->as.call.name)<0 &&
       (!strcmp(n->as.call.name,"array_set") || !strcmp(n->as.call.name,"array_push") ||
        !strcmp(n->as.call.name,"array_length"))) {
        bool setter=!strcmp(n->as.call.name,"array_set");
        bool length=!strcmp(n->as.call.name,"array_length");
        int arity=setter?3:(length?1:2);
        if(n->as.call.arg_count!=arity) {borrow_error(b,n,"the exact array builtin arity");return TAG_VOID;}
        if(borrow_expr(b,n->as.call.args[0])!=TAG_ARRAY) {borrow_error(b,n,"an exact FLOAT array receiver");return TAG_VOID;}
        if(length){emit_op(cg,OP_ARR_LEN);return TAG_INT;}
        if(setter && borrow_expr(b,n->as.call.args[1])!=TAG_INT) {borrow_error(b,n,"an exact integer array index");return TAG_VOID;}
        if(borrow_expr(b,n->as.call.args[arity-1])!=TAG_FLOAT) {borrow_error(b,n,"an exact FLOAT mutation value");return TAG_VOID;}
        if(cg->had_error)return TAG_VOID;
        if(setter){emit_op(cg,OP_ARR_SET);emit_op(cg,OP_POP);return TAG_VOID;}
        emit_op(cg,OP_ARR_PUSH);return TAG_ARRAY;
    }
    if(n->type==AST_CALL && borrow_arrays(b) && !n->as.call.func_expr && !n->as.call.borrow_mode &&
       n->as.call.name && !strcmp(n->as.call.name,"at") &&
       borrow_function_index(b,"at")<0 && borrow_local(b,"at")<0) {
        if(n->as.call.arg_count!=2 || borrow_expr(b,n->as.call.args[0])!=TAG_ARRAY ||
           borrow_expr(b,n->as.call.args[1])!=TAG_INT) {borrow_error(b,n,"an exact FLOAT array and integer index");return TAG_VOID;}
        /* ARR_GET may produce VOID; the complete mixed verifier retains that obligation. */
        emit_op(cg,OP_ARR_GET);return TAG_FLOAT;
    }
    if(n->type==AST_CALL && b->value_graph && !n->as.call.func_expr &&
       !n->as.call.borrow_mode && n->as.call.name &&
       (!strcmp(n->as.call.name,"print") || !strcmp(n->as.call.name,"println")) &&
       borrow_function_index(b,n->as.call.name)<0 && borrow_local(b,n->as.call.name)<0) {
        if(b->owner_arrays){borrow_error(b,n,"STRING print outside the owner ARRAY profile");return TAG_VOID;}
        if(n->as.call.arg_count!=1 || borrow_expr(b,n->as.call.args[0])!=TAG_STRING) {
            borrow_error(b,n,"one exact immutable string print operand");return TAG_VOID;
        }
        emit_op(cg,!strcmp(n->as.call.name,"println")?OP_PRINTLN:OP_PRINT);return TAG_VOID;
    }
    if(n->type==AST_CALL && b->value_graph) {
        int target=n->as.call.func_expr?-1:borrow_function_index(b,n->as.call.name);
        if(target<=0 || n->as.call.borrow_mode || borrow_local(b,n->as.call.name)>=0) {
            borrow_error(b,n,"an unshadowed direct non-entry value-call target");return TAG_VOID;
        }
        ASTNode *function=b->functions[target];
        if(n->as.call.arg_count!=function->as.function.param_count) {
            borrow_error(b,n,"the exact declared value-call arity");return TAG_VOID;
        }
        for(int i=0;i<n->as.call.arg_count && !cg->had_error;i++) {
            Parameter *p=&function->as.function.params[i];ASTNode *argument=n->as.call.args[i];
            int union_instance=borrow_named_union(b,p->type_info,p->struct_type_name);
            if(union_instance>=0) {
                if(borrow_union_expr(b,argument,union_instance)!=TAG_UNION)return TAG_VOID;
            } else if(p->type==TYPE_STRUCT) {
                int layout=borrow_record(b,p->struct_type_name);
                if(layout<0) {borrow_error(b,argument,"an exact declared owner parameter");return TAG_VOID;}
                if(argument->type==AST_IDENTIFIER) {
                    int local=borrow_local(b,argument->as.identifier);
                    if(local<0 || !b->live[local] || b->modes[local] || b->disposal_pending[local] ||
                       b->tags[local]!=TAG_STRUCT || b->layouts[local]!=(uint32_t)layout) {
                        borrow_error(b,argument,"a named live exact whole owner argument");return TAG_VOID;
                    }
                    emit_op(cg,OP_OWN_MOVE_LOCAL,local);b->live[local]=false;
                } else if(!borrow_owner(b,argument,(unsigned)layout)) return TAG_VOID;
            } else if(borrow_expr(b,argument)!=borrow_parameter_tag(p->type)) {
                borrow_error(b,argument,"the exact positional scalar argument");return TAG_VOID;
            }
        }
        if(cg->had_error)return TAG_VOID;
        emit_op(cg,OP_CALL,(uint32_t)target);
        uint8_t result=borrow_result_tag(b,function);
        b->expression_layout=result==TAG_UNION?(uint32_t)borrow_named_union(b,function->as.function.return_type_info,function->as.function.return_struct_type_name):
            (uint32_t)borrow_record(b,function->as.function.return_struct_type_name);
        return result;
    }
    if(n->type==AST_CALL && !b->helper_body && !n->as.call.func_expr &&
       n->as.call.name && !strcmp(n->as.call.name,b->helper->as.function.name) &&
       n->as.call.arg_count==b->helper->as.function.param_count) {
        if(b->consuming) {
            ASTNode *argument=n->as.call.args[0];
            int local=argument->type==AST_IDENTIFIER?borrow_local(b,argument->as.identifier):-1;
            int layout=borrow_record(b,b->helper->as.function.params[0].struct_type_name);
            if(n->as.call.borrow_mode || local<0 || !b->live[local] || b->modes[local] ||
               b->tags[local]!=TAG_STRUCT || b->disposal_pending[local] || layout<0 ||
               b->layouts[local]!=(uint32_t)layout) {
                borrow_error(b,n,"one named live exact whole owner for my consuming helper");return TAG_VOID;
            }
            emit_op(cg,OP_OWN_MOVE_LOCAL,local);b->live[local]=false;
            emit_op(cg,OP_CALL,(uint32_t)1);
            return borrow_tag(b->helper->as.function.return_type);
        }
        BorrowPlace places[8];uint8_t modes[8];
        emit_op(cg,OP_REGION_BEGIN);
        for(int i=0;i<n->as.call.arg_count;i++) {
            ASTNode *arg=n->as.call.args[i];Parameter *parameter=&b->helper->as.function.params[i];
            uint8_t mode=parameter->type==TYPE_BORROW_MUT?2:1;
            if(arg->type!=AST_CALL || arg->as.call.borrow_mode!=mode || arg->as.call.arg_count!=1 ||
               (arg->as.call.args[0]->type!=AST_IDENTIFIER && arg->as.call.args[0]->type!=AST_FIELD_ACCESS)) {
                borrow_error(b,n,"an explicit matching root-local borrowed argument");return TAG_VOID;
            }
            BorrowPlace place;
            if(!borrow_place(b,arg->as.call.args[0],&place))return TAG_VOID;
            int local=place.root;
            int layout=borrow_record(b,parameter->struct_type_name);
            if(local<0 || !b->live[local] || b->modes[local] || b->tags[local]!=TAG_STRUCT ||
               layout<0 || place.layout!=(uint32_t)layout || (mode==2 && !b->mutable[local])) {
                borrow_error(b,n,"exact live caller ownership and mutable exclusive authority");return TAG_VOID;
            }
            for(int j=0;j<i;j++)if(borrow_overlap(&places[j],&place) && (mode==2 || modes[j]==2)) {
                borrow_error(b,n,"disjoint places for every exclusive argument");return TAG_VOID;
            }
            places[i]=place;modes[i]=mode;
            borrow_reference(b,&place,(unsigned)i,mode);
        }
        emit_op(cg,OP_CALL_REF,(uint32_t)1,0);emit_op(cg,OP_REGION_END);
        return borrow_tag(b->helper->as.function.return_type);
    }
    borrow_error(b,n,"straight-line scalar expressions and one borrowed helper");return TAG_VOID;
}
static bool borrow_owner(BorrowCG *b,ASTNode *value,unsigned layout) {
    if(layout>=b->record_count || b->ordinary[layout]) {borrow_error(b,value,"a resource layout for owner transfer");return false;}
    CG *cg=b->cg;
    if(!value || cg->had_error)return false;
    if(value->type==AST_STRUCT_LITERAL && !value->as.struct_literal.spread_source &&
       !strcmp(value->as.struct_literal.struct_name,b->records[layout]->as.struct_def.name)) {
        ASTNode *record=b->records[layout];int fields=record->as.struct_def.field_count;
        int slots[256];for(int f=0;f<256;f++)slots[f]=-1;
        if(value->as.struct_literal.field_count!=fields){borrow_error(b,value,"every declared scalar field exactly once");return false;}
        for(int f=0;f<fields && !cg->had_error;f++) {
            int index=-1;for(int j=0;j<fields;j++)
                if(!strcmp(value->as.struct_literal.field_names[f],record->as.struct_def.field_names[j]))index=j;
            if(index<0 || slots[index]>=0){borrow_error(b,value,"distinct declared constructor fields");return false;}
            int child=borrow_child(b,(unsigned)layout,index);uint8_t tag;
            if(child>=0) {
                /* I root each completed child before the next source field. */
                if((unsigned)child>=layout){borrow_error(b,value,"a strictly earlier exact child layout");return false;}
                if(!borrow_owner(b,value->as.struct_literal.field_values[f],(unsigned)child))return false;
                tag=TAG_STRUCT;
            } else {
                bool array=b->owner_arrays && borrow_float_array_field(record,index);
                tag=array?borrow_float_array(b,value->as.struct_literal.field_values[f],true):
                          borrow_expr(b,value->as.struct_literal.field_values[f]);
                if(tag!=borrow_owner_field_tag(b,layout,index)){borrow_error(b,value,"exact constructor field types");return false;}
            }
            slots[index]=borrow_slot(b,NULL,tag,child>=0?(uint32_t)child:NVM_V2_NO_INDEX,0,false);
            if(cg->had_error)return false;
            emit_op(cg,child>=0?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slots[index]);
        }
        if(cg->had_error)return false;
        for(int f=0;f<fields;f++) {
            emit_op(cg,b->tags[slots[f]]==TAG_STRUCT?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,slots[f]);
            if(b->tags[slots[f]]==TAG_STRUCT)b->live[slots[f]]=false;
        }
        emit_op(cg,OP_OWN_PACK,(uint32_t)layout);
    } else if(value->type==AST_IDENTIFIER) {
        int old=borrow_local(b,value->as.identifier);
        if(old<0 || !b->live[old] || b->modes[old] || b->disposal_pending[old] || b->tags[old]!=TAG_STRUCT || b->layouts[old]!=(uint32_t)layout) {
            borrow_error(b,value,"an unborrowed exact owner for destructuring");return false;
        }
        emit_op(cg,OP_OWN_MOVE_LOCAL,old);b->live[old]=false;b->disposal_pending[old]=false;
    } else if(value->type==AST_CALL && b->value_graph) {
        if(borrow_expr(b,value)!=TAG_STRUCT || b->expression_layout!=layout) {
            borrow_error(b,value,"an exact owned call result");return false;
        }
    } else {borrow_error(b,value,"a resource constructor, exact owner move or explicit destructive pattern");return false;}
    return !cg->had_error;
}
static bool borrow_ordinary(BorrowCG *b,ASTNode *value,unsigned layout) {
    if(!b->mixed || !b->ordinary[layout])return false;
    if(value->type==AST_IDENTIFIER) {
        int local=borrow_local(b,value->as.identifier);
        if(local<0 || !b->live[local] || !borrow_ordinary_local(b,(unsigned)local) || b->layouts[local]!=layout) {
            borrow_error(b,value,"an exact live ordinary record alias");return false;
        }
        emit_op(b->cg,OP_LOAD_LOCAL,local);return true;
    }
    ASTNode *record=b->records[layout];int slots[256];
    if(value->type!=AST_STRUCT_LITERAL || value->as.struct_literal.spread_source ||
       strcmp(value->as.struct_literal.struct_name,record->as.struct_def.name) ||
       value->as.struct_literal.field_count!=record->as.struct_def.field_count) {
        borrow_error(b,value,"an exact complete ordinary constructor");return false;
    }
    for(int i=0;i<record->as.struct_def.field_count;i++)slots[i]=-1;
    for(int i=0;i<value->as.struct_literal.field_count;i++) {
        int f=borrow_layout_field(b,layout,value->as.struct_literal.field_names[i],value);
        if(f<0)return false;
        if(slots[f]>=0){borrow_error(b,value,"distinct ordinary constructor fields");return false;}
        bool array=borrow_float_array_field(record,f);
        ASTNode *field=value->as.struct_literal.field_values[i];
        uint8_t tag=array?borrow_float_array(b,field,true):borrow_expr(b,field);
        if(tag!=(array?TAG_ARRAY:borrow_tag(record->as.struct_def.field_types[f]))) {
            borrow_error(b,value,"exact ordinary field types");return false;
        }
        slots[f]=borrow_slot(b,NULL,tag,NVM_V2_NO_INDEX,0,false);
        if(b->cg->had_error)return false;
        emit_op(b->cg,OP_STORE_LOCAL,slots[f]);
    }
    for(int i=0;i<record->as.struct_def.field_count;i++)emit_op(b->cg,OP_LOAD_LOCAL,slots[i]);
    emit_op(b->cg,OP_AGG_PACK,AGG_RECORD,layout,0,record->as.struct_def.field_count);
    return !b->cg->had_error;
}
static bool borrow_match_coverage(BorrowCG *b,ASTNode *n,unsigned instance) {
    UnionDef *definition=b->unions[instance].definition;bool covered[256]={false};
    if(definition->variant_count<=0 || definition->variant_count>256 || n->as.match_expr.arm_count<=0 ||
       n->as.match_expr.arm_count>64)return false;
    for(int i=0;i<n->as.match_expr.arm_count;i++) {
        const char *name=n->as.match_expr.pattern_variants[i];int variant=-1;
        for(int j=0;j<definition->variant_count;j++)if(name && !strcmp(name,definition->variant_names[j]))variant=j;
        if(variant<0)return false;
        ASTNode *guard=n->as.match_expr.guard_exprs?n->as.match_expr.guard_exprs[i]:NULL;
        if(!guard || (guard->type==AST_BOOL && guard->as.bool_val))covered[variant]=true;
    }
    for(int i=0;i<definition->variant_count;i++)if(!covered[i])return false;
    return true;
}
static int borrow_match_source(BorrowCG *b,ASTNode *n) {
    if(borrow_expr(b,n->as.match_expr.expr)!=TAG_UNION || b->expression_layout>=b->union_count) {
        borrow_error(b,n,"an exact concrete scalar union match source");return -1;
    }
    int instance=(int)b->expression_layout;
    if(!borrow_match_coverage(b,n,(unsigned)instance)) {
        borrow_error(b,n,"complete named scalar union match coverage");return -1;
    }
    return instance;
}
static int borrow_match_binding(BorrowCG *b,unsigned instance,int variant,const char *name) {
    const char *binding=name && strcmp(name,"_")?name:NULL;
    int slot=borrow_slot(b,binding,TAG_UNION,instance,0,false);
    b->variants[slot]=(int16_t)variant;emit_op(b->cg,OP_DUP);emit_op(b->cg,OP_STORE_LOCAL,slot);
    if(binding)borrow_name_begin(b,(unsigned)slot,binding);
    return slot;
}
static void borrow_match_scope_end(BorrowCG *b,unsigned first) {
    borrow_names_end(b,first);for(unsigned i=first;i<b->count;i++)b->names[i]=NULL;
}
static int borrow_selected_field(BorrowCG *b,unsigned instance,int variant,const char *name) {
    UnionDef *u=b->unions[instance].definition;
    if(variant<0 || variant>=u->variant_count)return -1;
    for(int f=0;f<u->variant_field_counts[variant];f++)
        if(!strcmp(name,u->variant_field_names[variant][f]))return f;
    return -1;
}
static bool borrow_selected_owns(BorrowCG *b,unsigned instance,int variant) {
    UnionDef *u=b->unions[instance].definition;
    for(int f=0;f<u->variant_field_counts[variant];f++) {
        uint32_t child;uint8_t tag=borrow_union_payload(b,instance,variant,f,&child,NULL);
        if(tag==TAG_STRUCT || (tag==TAG_UNION && borrow_owned_union(b,child)))return true;
    }
    return false;
}
static bool borrow_union_pattern(BorrowCG *b,ASTNode *n) {
    if(!n->as.let.is_destructure || !n->as.let.value || n->as.let.value->type!=AST_IDENTIFIER)return false;
    int root=borrow_local(b,n->as.let.value->as.identifier);
    if(root<0 || b->tags[root]!=TAG_UNION || !borrow_owned_union(b,b->layouts[root]))return false;
    unsigned instance=b->layouts[root];int variant=b->variants[root];
    UnionDef *u=b->unions[instance].definition;
    const char *pattern=n->as.let.type_name,*dot=pattern?strrchr(pattern,'.'):NULL;
    size_t length=dot?(size_t)(dot-pattern):0;
    char declaration[4096];
    if(!b->live[root] || variant<0 || !dot || length>=sizeof(declaration)) {
        borrow_error(b,n,"a live selected union pattern");return true;
    }
    memcpy(declaration,pattern,length);declaration[length]='\0';
    if(env_get_union(b->env,declaration)!=u || strcmp(dot+1,u->variant_names[variant])) {
        borrow_error(b,n,"the exact selected union declaration and variant");return true;
    }
    int count=u->variant_field_counts[variant],slots[256];bool seen[256]={false};
    if(count>256 || n->as.let.destructure_count!=count) {borrow_error(b,n,"every selected union field once");return true;}
    for(int i=0;i<count;i++) {
        int field=borrow_selected_field(b,instance,variant,n->as.let.destructure_names[i]);
        if(field<0 || seen[field]){borrow_error(b,n,"distinct exact selected union fields");return true;}
        seen[field]=true;
    }
    b->names[root]=n->as.let.name;
    for(int f=0;f<count;f++) {
        uint32_t child;uint8_t tag=borrow_union_payload(b,instance,variant,f,&child,n);
        slots[f]=borrow_slot(b,NULL,tag,child,0,false);
        b->projection_parent[slots[f]]=root;b->projection_field[slots[f]]=f;
    }
    if(b->cg->had_error)return true;
    emit_op(b->cg,OP_OWN_UNPACK_VARIANT,root,variant,count);
    b->live[root]=false;b->disposal_pending[root]=false;
    for(int f=count-1;f>=0;f--)
        emit_op(b->cg,borrow_owned_local(b,(unsigned)slots[f])?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slots[f]);
    return true;
}
static uint8_t borrow_owned_match(BorrowCG *b,ASTNode *n,int instance,int expected,bool value,bool shadow) {
    unsigned incoming=b->count;
    int root=borrow_slot(b,NULL,TAG_UNION,(uint32_t)instance,0,false);
    emit_op(b->cg,OP_OWN_STORE_LOCAL,root);emit_op(b->cg,OP_LOAD_LOCAL,root);
    bool live[256],pending[256],joined_live[256]={false},joined_pending[256]={false};
    memcpy(live,b->live,incoming*sizeof(bool));memcpy(pending,b->disposal_pending,incoming*sizeof(bool));
    bool joined=false,all_return=true;uint32_t ends[64];unsigned end_count=0;
    uint8_t result=TAG_VOID;uint32_t result_layout=NVM_V2_NO_INDEX;
    for(int i=0;i<n->as.match_expr.arm_count && !b->cg->had_error;i++) {
        if(n->as.match_expr.guard_exprs && n->as.match_expr.guard_exprs[i]) {
            borrow_error(b,n,"exhaustive unguarded owned match arms");break;
        }
        memcpy(b->live,live,incoming*sizeof(bool));memcpy(b->disposal_pending,pending,incoming*sizeof(bool));
        b->live[root]=true;b->returned=false;b->transferred=false;
        int variant=borrow_union_variant(b,(unsigned)instance,n->as.match_expr.pattern_variants[i],n);
        if(variant<0)break;
        b->variants[root]=(int16_t)variant;
        b->disposal_pending[root]=!borrow_selected_owns(b,(unsigned)instance,variant);
        uint32_t matched=emit_op(b->cg,OP_MATCH_TAG,variant,(int32_t)0);
        uint32_t next=emit_op(b->cg,OP_JMP,(int32_t)0);
        patch_jump(b->cg,matched+3,matched,b->cg->code_size);emit_op(b->cg,OP_POP);
        const char *name=n->as.match_expr.pattern_bindings[i];
        b->names[root]=name && strcmp(name,"_")?name:NULL;
        if(b->names[root])borrow_name_begin(b,(unsigned)root,b->names[root]);
        ASTNode *body=n->as.match_expr.arm_bodies[i];
        uint8_t tag=TAG_VOID;
        if(value) {
            ASTNode *tail=body;
            if(body && (body->type==AST_BLOCK || body->type==AST_UNSAFE_BLOCK)) {
                ASTNode prefix=*body;ASTNode **items=body->type==AST_BLOCK?body->as.block.statements:body->as.unsafe_block.statements;
                int count=body->type==AST_BLOCK?body->as.block.count:body->as.unsafe_block.count;
                if(!count){borrow_error(b,n,"a value at the end of every owned match arm");break;}
                if(body->type==AST_BLOCK)prefix.as.block.count=count-1;else prefix.as.unsafe_block.count=count-1;
                unsigned depth=b->control_depth;b->control_depth=0;borrow_body(b,&prefix,false);b->control_depth=depth;
                if(b->returned || b->transferred){borrow_error(b,n,"a value arm without early control transfer");break;}
                tail=items[count-1];
            }
            tag=expected>=0?borrow_union_expr(b,tail,expected):borrow_expr(b,tail);
            uint32_t layout=(tag==TAG_STRUCT || tag==TAG_UNION)?b->expression_layout:NVM_V2_NO_INDEX;
            if(!tag || (i && (tag!=result || layout!=result_layout))) {
                borrow_error(b,n,"one exact value type across owned match arms");break;
            }
            result=tag;result_layout=layout;
        } else {b->control_depth++;borrow_body(b,body,shadow);b->control_depth--;}
        if(b->transferred){borrow_error(b,n,"owned match arms without loop transfers");break;}
        bool returned=b->returned;
        if(!returned) {
            if(!borrow_consumed(b,incoming,n))break;
            borrow_cleanup(b,incoming);
            if(joined)borrow_same_owners(b,incoming,joined_live,joined_pending,n);
            else {memcpy(joined_live,b->live,incoming*sizeof(bool));memcpy(joined_pending,b->disposal_pending,incoming*sizeof(bool));joined=true;}
            ends[end_count++]=emit_op(b->cg,OP_JMP,(int32_t)0);
        }
        borrow_match_scope_end(b,incoming);
        borrow_name_retain(b,(unsigned)root);
        for(unsigned slot=incoming;slot<b->count;slot++){b->live[slot]=false;b->disposal_pending[slot]=false;}
        all_return=all_return && returned;
        patch_jump(b->cg,next+1,next,b->cg->code_size);
    }
    if(b->cg->had_error)return TAG_VOID;
    emit_op(b->cg,OP_POP);emit_op(b->cg,OP_PUSH_BOOL,0);emit_op(b->cg,OP_ASSERT);emit_op(b->cg,OP_HALT);
    for(unsigned i=0;i<end_count;i++)patch_jump(b->cg,ends[i]+1,ends[i],b->cg->code_size);
    if(joined){memcpy(b->live,joined_live,incoming*sizeof(bool));memcpy(b->disposal_pending,joined_pending,incoming*sizeof(bool));}
    b->returned=all_return;b->transferred=false;b->expression_layout=result_layout;return result;
}
static uint8_t borrow_match_expr(BorrowCG *b,ASTNode *n) {
    int instance=borrow_match_source(b,n);if(instance<0)return TAG_VOID;
    if(borrow_owned_union(b,(uint32_t)instance))return borrow_owned_match(b,n,instance,-1,true,false);
    uint32_t ends[64];unsigned end_count=0;uint8_t result=TAG_VOID;
    for(int i=0;i<n->as.match_expr.arm_count && !b->cg->had_error;i++) {
        ASTNode *body=n->as.match_expr.arm_bodies[i];
        if(!body || body->type==AST_BLOCK || body->type==AST_UNSAFE_BLOCK) {
            borrow_error(b,n,"scalar expressions in value match arms");break;
        }
        int variant=borrow_union_variant(b,(unsigned)instance,n->as.match_expr.pattern_variants[i],n);
        if(variant<0)break;
        uint32_t matched=emit_op(b->cg,OP_MATCH_TAG,variant,(int32_t)0);
        uint32_t next=emit_op(b->cg,OP_JMP,(int32_t)0);
        patch_jump(b->cg,matched+3,matched,b->cg->code_size);
        unsigned first=b->count;borrow_match_binding(b,(unsigned)instance,variant,n->as.match_expr.pattern_bindings[i]);
        uint32_t guard=UINT32_MAX;
        ASTNode *condition=n->as.match_expr.guard_exprs?n->as.match_expr.guard_exprs[i]:NULL;
        if(condition) {
            if(borrow_expr(b,condition)!=TAG_BOOL){borrow_error(b,condition,"an exact Boolean scalar union match guard");break;}
            guard=emit_op(b->cg,OP_JMP_FALSE,(int32_t)0);
        }
        emit_op(b->cg,OP_POP);
        uint8_t tag=borrow_expr(b,body);
        if(tag!=TAG_INT && tag!=TAG_FLOAT && tag!=TAG_BOOL && tag!=TAG_STRING) {
            borrow_error(b,body,"one exact scalar value-match result");break;
        }
        if(result==TAG_VOID)result=tag;
        else if(result!=tag){borrow_error(b,body,"one matching scalar value-match result type");break;}
        borrow_match_scope_end(b,first);
        ends[end_count++]=emit_op(b->cg,OP_JMP,(int32_t)0);
        patch_jump(b->cg,next+1,next,b->cg->code_size);
        if(guard!=UINT32_MAX)patch_jump(b->cg,guard+1,guard,b->cg->code_size);
    }
    if(b->cg->had_error)return TAG_VOID;
    emit_op(b->cg,OP_POP);emit_op(b->cg,OP_PUSH_BOOL,0);emit_op(b->cg,OP_ASSERT);emit_op(b->cg,OP_HALT);
    for(unsigned i=0;i<end_count;i++)patch_jump(b->cg,ends[i]+1,ends[i],b->cg->code_size);
    return result;
}
static void borrow_match_statement(BorrowCG *b,ASTNode *n,bool shadow) {
    int instance=borrow_match_source(b,n);if(instance<0)return;
    if(borrow_owned_union(b,(uint32_t)instance)){borrow_owned_match(b,n,instance,-1,false,shadow);return;}
    unsigned incoming_count=b->count;bool incoming_live[256],incoming_pending[256];
    bool joined_live[256]={false},joined_pending[256]={false};
    memcpy(incoming_live,b->live,incoming_count*sizeof(bool));
    memcpy(incoming_pending,b->disposal_pending,incoming_count*sizeof(bool));
    bool has_join=false,all_return=true,all_stop=true;uint32_t ends[64];unsigned end_count=0;
    for(int i=0;i<n->as.match_expr.arm_count && !b->cg->had_error;i++) {
        ASTNode *body=n->as.match_expr.arm_bodies[i];
        if(!body || (body->type!=AST_BLOCK && body->type!=AST_UNSAFE_BLOCK)) {
            borrow_error(b,n,"statement blocks in scalar union match arms");break;
        }
        memcpy(b->live,incoming_live,incoming_count*sizeof(bool));
        memcpy(b->disposal_pending,incoming_pending,incoming_count*sizeof(bool));
        b->returned=false;b->transferred=false;
        int variant=borrow_union_variant(b,(unsigned)instance,n->as.match_expr.pattern_variants[i],n);
        if(variant<0)break;
        uint32_t matched=emit_op(b->cg,OP_MATCH_TAG,variant,(int32_t)0);
        uint32_t next=emit_op(b->cg,OP_JMP,(int32_t)0);
        patch_jump(b->cg,matched+3,matched,b->cg->code_size);
        unsigned first=b->count;borrow_match_binding(b,(unsigned)instance,variant,n->as.match_expr.pattern_bindings[i]);
        uint32_t guard=UINT32_MAX;
        ASTNode *condition=n->as.match_expr.guard_exprs?n->as.match_expr.guard_exprs[i]:NULL;
        if(condition) {
            if(borrow_expr(b,condition)!=TAG_BOOL){borrow_error(b,condition,"an exact Boolean scalar union match guard");break;}
            guard=emit_op(b->cg,OP_JMP_FALSE,(int32_t)0);
        }
        emit_op(b->cg,OP_POP);b->control_depth++;borrow_body(b,body,shadow);b->control_depth--;
        bool arm_return=b->returned,arm_stop=b->returned || b->transferred;
        borrow_match_scope_end(b,first);
        if(!arm_stop) {
            if(has_join)borrow_same_owners(b,incoming_count,joined_live,joined_pending,n);
            else {
                memcpy(joined_live,b->live,incoming_count*sizeof(bool));
                memcpy(joined_pending,b->disposal_pending,incoming_count*sizeof(bool));has_join=true;
            }
            ends[end_count++]=emit_op(b->cg,OP_JMP,(int32_t)0);
        }
        all_return=all_return && arm_return;all_stop=all_stop && arm_stop;
        patch_jump(b->cg,next+1,next,b->cg->code_size);
        if(guard!=UINT32_MAX)patch_jump(b->cg,guard+1,guard,b->cg->code_size);
    }
    if(b->cg->had_error)return;
    emit_op(b->cg,OP_POP);emit_op(b->cg,OP_PUSH_BOOL,0);emit_op(b->cg,OP_ASSERT);emit_op(b->cg,OP_HALT);
    for(unsigned i=0;i<end_count;i++)patch_jump(b->cg,ends[i]+1,ends[i],b->cg->code_size);
    if(has_join) {
        memcpy(b->live,joined_live,incoming_count*sizeof(bool));
        memcpy(b->disposal_pending,joined_pending,incoming_count*sizeof(bool));
    }
    b->returned=all_return;b->transferred=all_stop && !all_return;
}
static void borrow_increment(BorrowCG *b,int slot) {
    if(slot<0 || b->cg->had_error)return;
    emit_op(b->cg,OP_LOAD_LOCAL,slot);emit_op(b->cg,OP_PUSH_I64,(int64_t)1);
    emit_op(b->cg,borrow_scalar_opcode(b,OP_ADD,TAG_INT));emit_op(b->cg,OP_STORE_LOCAL,slot);
}
static void borrow_body(BorrowCG *b,ASTNode *body,bool shadow) {
    if(!body || (body->type!=AST_BLOCK && body->type!=AST_UNSAFE_BLOCK)) {
        borrow_error(b,body,"a lexical function/shadow body");return;
    }
    ASTNode **statements=body->type==AST_BLOCK?body->as.block.statements:body->as.unsafe_block.statements;
    int statement_count=body->type==AST_BLOCK?body->as.block.count:body->as.unsafe_block.count;
    unsigned first=b->count;
    for(int i=0;i<statement_count && !b->cg->had_error && !b->returned && !b->transferred;i++) {
        ASTNode *n=statements[i];CG *cg=b->cg;
        if(n->type==AST_MATCH) {
            if(b->control_depth==32){borrow_error(b,n,"at most 32 nested control-flow bodies");break;}
            borrow_match_statement(b,n,shadow);
        } else if(n->type==AST_UNSAFE_BLOCK) {
            if(b->control_depth==32){borrow_error(b,n,"at most 32 nested control-flow bodies");break;}
            b->control_depth++;borrow_body(b,n,shadow);b->control_depth--;
        } else if(n->type==AST_IF || n->type==AST_WHILE || n->type==AST_FOR) {
            if(b->control_depth==32){borrow_error(b,n,"at most 32 nested control-flow bodies");break;}
            unsigned for_first=b->count;int index=-1,end=-1;
            if(n->type==AST_FOR) {
                ASTNode *range=n->as.for_stmt.range_expr;
                if(range->type!=AST_CALL || range->as.call.func_expr || range->as.call.borrow_mode || !range->as.call.name ||
                   strcmp(range->as.call.name,"range") || borrow_local(b,"range")>=0 ||
                   borrow_function_index(b,"range")>=0 ||
                   range->as.call.arg_count!=2) {
                    borrow_error(b,n,"a direct builtin range with two integer bounds");break;
                }
                if(borrow_expr(b,range->as.call.args[0])!=TAG_INT){borrow_error(b,n,"integer range bounds");break;}
                if(borrow_expr(b,range->as.call.args[1])!=TAG_INT){borrow_error(b,n,"integer range bounds");break;}
                end=borrow_slot(b,NULL,TAG_INT,NVM_V2_NO_INDEX,0,false);emit_op(cg,OP_STORE_LOCAL,end);
                index=borrow_slot(b,n->as.for_stmt.var_name,TAG_INT,NVM_V2_NO_INDEX,0,false);emit_op(cg,OP_STORE_LOCAL,index);
                borrow_name_begin(b,(unsigned)index,n->as.for_stmt.var_name);
            }
            uint32_t top=cg->code_size;
            if(n->type==AST_FOR) {
                emit_op(cg,OP_LOAD_LOCAL,index);emit_op(cg,OP_LOAD_LOCAL,end);emit_op(cg,borrow_scalar_opcode(b,OP_LT,TAG_INT));
            } else {
                ASTNode *condition=n->type==AST_IF?n->as.if_stmt.condition:n->as.while_stmt.condition;
                if(borrow_expr(b,condition)!=TAG_BOOL){borrow_error(b,n,"a Boolean control-flow condition");break;}
            }
            uint32_t skip=emit_op(cg,OP_JMP_FALSE,(int32_t)0);
            unsigned incoming_count=b->count;
            bool incoming_live[256],incoming_disposal[256];
            memcpy(incoming_live,b->live,incoming_count*sizeof(bool));
            memcpy(incoming_disposal,b->disposal_pending,incoming_count*sizeof(bool));
            BorrowLoop loop={0};
            if(n->type!=AST_IF) {
                loop.parent=b->loop;loop.first=incoming_count;loop.top=top;loop.index=index;
                memcpy(loop.live,incoming_live,incoming_count*sizeof(bool));
                memcpy(loop.pending,incoming_disposal,incoming_count*sizeof(bool));b->loop=&loop;
            }
            b->control_depth++;
            borrow_body(b,n->type==AST_IF?n->as.if_stmt.then_branch:n->type==AST_FOR?n->as.for_stmt.body:n->as.while_stmt.body,shadow);
            b->control_depth--;
            if(n->type!=AST_IF)b->loop=loop.parent;
            if(cg->had_error)break;
            bool then_returns=b->returned,then_stops=b->returned || b->transferred;
            bool then_live[256],then_disposal[256];
            memcpy(then_live,b->live,incoming_count*sizeof(bool));
            memcpy(then_disposal,b->disposal_pending,incoming_count*sizeof(bool));
            uint32_t next=UINT32_MAX;
            if(!then_stops){borrow_increment(b,index);next=emit_op(cg,OP_JMP,(int32_t)0);}
            patch_jump(cg,skip+1,skip,cg->code_size);
            memcpy(b->live,incoming_live,incoming_count*sizeof(bool));
            memcpy(b->disposal_pending,incoming_disposal,incoming_count*sizeof(bool));b->returned=false;b->transferred=false;
            if(n->type==AST_IF) {
                if(n->as.if_stmt.else_branch) {
                    b->control_depth++;borrow_body(b,n->as.if_stmt.else_branch,shadow);b->control_depth--;
                }
                bool else_returns=b->returned,else_stops=b->returned || b->transferred;
                if(!then_stops) {
                    if(!else_stops)borrow_same_owners(b,incoming_count,then_live,then_disposal,n);
                    else {
                        memcpy(b->live,then_live,incoming_count*sizeof(bool));
                        memcpy(b->disposal_pending,then_disposal,incoming_count*sizeof(bool));
                    }
                }
                b->returned=then_returns && else_returns;
                b->transferred=then_stops && else_stops && !b->returned;
                if(!cg->had_error && next!=UINT32_MAX)patch_jump(cg,next+1,next,cg->code_size);
            } else {
                if(next!=UINT32_MAX && borrow_same_owners(b,incoming_count,then_live,then_disposal,n))patch_jump(cg,next+1,next,top);
                for(unsigned j=0;j<loop.break_count;j++)patch_jump(cg,loop.breaks[j]+1,loop.breaks[j],cg->code_size);
                if(n->type==AST_FOR){borrow_names_end(b,for_first);for(unsigned j=for_first;j<b->count;j++)b->names[j]=NULL;}
            }
        } else if(n->type==AST_BREAK || n->type==AST_CONTINUE) {
            BorrowLoop *loop=b->loop;
            if(!loop){borrow_error(b,n,"an enclosing loop for loop transfer");break;}
            if(!borrow_consumed(b,loop->first,n) ||
               !borrow_same_owners(b,loop->first,loop->live,loop->pending,n))break;
            borrow_cleanup(b,loop->first);
            if(cg->had_error)break;
            if(n->type==AST_BREAK && loop->break_count==256){borrow_error(b,n,"at most 256 break edges per loop");break;}
            if(n->type==AST_CONTINUE)borrow_increment(b,loop->index);
            uint32_t edge=emit_op(cg,OP_JMP,(int32_t)0);
            if(n->type==AST_CONTINUE)patch_jump(cg,edge+1,edge,loop->top);
            else loop->breaks[loop->break_count++]=edge;
            b->transferred=true;
        } else if(n->type==AST_LET) {
            if(borrow_union_pattern(b,n))continue;
            if(n->as.let.is_destructure_projection && n->as.let.value->type==AST_FIELD_ACCESS &&
               n->as.let.value->as.field_access.object->type==AST_IDENTIFIER) {
                ASTNode *field=n->as.let.value;
                int parent=borrow_local(b,field->as.field_access.object->as.identifier);
                if(parent>=0 && !b->live[parent] && borrow_owned_local(b,(unsigned)parent) &&
                   (b->tags[parent]==TAG_UNION || borrow_arrays(b) || !borrow_leaf(b,b->layouts[parent]))) {
                    int index=b->tags[parent]==TAG_UNION?
                        borrow_selected_field(b,b->layouts[parent],b->variants[parent],field->as.field_access.field_name):
                        borrow_layout_field(b,b->layouts[parent],field->as.field_access.field_name,n),source=-1;
                    for(unsigned j=0;j<b->count;j++)if(b->projection_parent[j]==parent && b->projection_field[j]==index)source=(int)j;
                    if(source<0 || !b->live[source]){borrow_error(b,n,"an unconsumed exact destructured field");break;}
                    emit_op(cg,borrow_owned_local(b,(unsigned)source)?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,source);
                    if(borrow_owned_local(b,(unsigned)source))b->live[source]=false;
                    int slot=borrow_slot(b,n->as.let.name,b->tags[source],b->layouts[source],0,n->as.let.is_mut);
                    emit_op(cg,borrow_owned_local(b,(unsigned)source)?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slot);
                    borrow_name_begin(b,(unsigned)slot,n->as.let.name);continue;
                }
            }
            int layout=borrow_record(b,n->as.let.type_name);
            int union_instance=borrow_named_union(b,n->as.let.type_info,n->as.let.type_name);
            if(union_instance>=0) {
                if(n->as.let.is_destructure || borrow_union_expr(b,n->as.let.value,union_instance)!=TAG_UNION) {
                    if(!cg->had_error)borrow_error(b,n,"an exact concrete union binding without destructuring");
                    break;
                }
                int slot=borrow_slot(b,n->as.let.name,TAG_UNION,(uint32_t)union_instance,0,n->as.let.is_mut);
                emit_op(cg,borrow_owned_local(b,(unsigned)slot)?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slot);borrow_name_begin(b,(unsigned)slot,n->as.let.name);
            } else if(layout>=0 && b->ordinary[layout]) {
                if(n->as.let.is_destructure || !borrow_ordinary(b,n->as.let.value,(unsigned)layout)) {
                    if(!cg->had_error)borrow_error(b,n,"ordinary record bindings without destructive patterns");
                    break;
                }
                int slot=borrow_slot(b,n->as.let.name,TAG_STRUCT,(uint32_t)layout,0,n->as.let.is_mut);
                emit_op(cg,OP_STORE_LOCAL,slot);borrow_name_begin(b,(unsigned)slot,n->as.let.name);
            } else if(layout>=0) {
                ASTNode *value=n->as.let.value;
                if(!borrow_owner(b,value,(unsigned)layout))break;
                int slot=borrow_slot(b,n->as.let.name,TAG_STRUCT,(uint32_t)layout,0,n->as.let.is_mut);
                emit_op(cg,OP_OWN_STORE_LOCAL,slot);
                if(!n->as.let.is_destructure)borrow_name_begin(b,(unsigned)slot,n->as.let.name);
                else {
                    ASTNode *record=b->records[layout];
                    if(n->as.let.destructure_count!=record->as.struct_def.field_count){borrow_error(b,n,"every pattern field once");break;}
                    bool seen[256]={false};
                    for(int f=0;f<n->as.let.destructure_count && !cg->had_error;f++) {
                        int field=borrow_layout_field(b,(unsigned)layout,n->as.let.destructure_names[f],n);
                        if(field<0)break;
                        if(seen[field])borrow_error(b,n,"distinct pattern fields");
                        seen[field]=true;
                    }
                    if(!cg->had_error) {
                        if(!borrow_arrays(b) && borrow_leaf(b,(unsigned)layout))b->disposal_pending[slot]=true;
                        else borrow_unpack(b,(unsigned)slot);
                    }
                }
            } else {
                uint8_t tag=borrow_expr(b,n->as.let.value),declared=borrow_local_tag(n->as.let.var_type);
                if(tag==TAG_STRUCT && b->expression_layout<b->record_count && b->ordinary[b->expression_layout] && n->as.let.var_type==TYPE_UNKNOWN) {
                    int slot=borrow_slot(b,n->as.let.name,tag,b->expression_layout,0,n->as.let.is_mut);
                    emit_op(cg,OP_STORE_LOCAL,slot);borrow_name_begin(b,(unsigned)slot,n->as.let.name);continue;
                }
                if(borrow_arrays(b) && n->as.let.var_type==TYPE_ARRAY && n->as.let.element_type==TYPE_FLOAT)declared=TAG_ARRAY;
                if((n->as.let.var_type!=TYPE_UNKNOWN && tag!=declared) || (tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_FLOAT && !(b->value_graph && tag==TAG_STRING) && !(borrow_arrays(b) && tag==TAG_ARRAY))) {
                    borrow_error(b,n,"int/bool/float locals or value-graph STRING locals");break;
                }
                int slot=borrow_slot(b,n->as.let.name,tag,NVM_V2_NO_INDEX,0,n->as.let.is_mut);
                emit_op(cg,OP_STORE_LOCAL,slot);
                borrow_name_begin(b,(unsigned)slot,n->as.let.name);
            }
        } else if(n->type==AST_PRINT) {
            if(b->owner_arrays){borrow_error(b,n,"STRING print outside the owner ARRAY profile");break;}
            const char *name=n->as.print.is_println?"println":"print";
            if(!b->value_graph || borrow_function_index(b,name)>=0 || borrow_local(b,name)>=0 ||
               borrow_expr(b,n->as.print.expr)!=TAG_STRING) {
                borrow_error(b,n,"an unshadowed immutable string print in a value-call graph");break;
            }
            emit_op(cg,n->as.print.is_println?OP_PRINTLN:OP_PRINT);
        } else if(n->type==AST_ASSERT) {
            if(borrow_expr(b,n->as.assert.condition)!=TAG_BOOL)borrow_error(b,n,"a Boolean assertion");
            emit_op(cg,OP_ASSERT);
        } else if(n->type==AST_SET) {
            int local=borrow_local(b,n->as.set.name);
            if(local<0){borrow_error(b,n,"an existing mutable target");break;}
            if(borrow_ordinary_local(b,(unsigned)local) || b->tags[local]==TAG_ARRAY) {borrow_error(b,n,"ordinary managed bindings without assignment");break;}
            if(!n->as.set.field_name && b->tags[local]==TAG_STRUCT) {
                ASTNode *value=n->as.set.value;
                int source=value->type==AST_IDENTIFIER?borrow_local(b,value->as.identifier):-1;
                if(!b->mutable[local] || b->live[local] || b->modes[local] ||
                   source<0 || !b->live[source] || b->modes[source] || b->disposal_pending[source] ||
                   b->tags[source]!=TAG_STRUCT || b->layouts[source]!=b->layouts[local]) {
                    borrow_error(b,n,"an exact live owner assigned to a consumed mutable owner");break;
                }
                emit_op(cg,OP_OWN_MOVE_LOCAL,source);
                b->live[source]=false;b->disposal_pending[source]=false;
                emit_op(cg,OP_OWN_STORE_LOCAL,local);
                b->live[local]=true;b->disposal_pending[local]=false;
                continue;
            }
            if(!b->live[local]){borrow_error(b,n,"a live mutable target");break;}
            if(n->as.set.field_name) {
                if(b->tags[local]!=TAG_STRUCT || b->modes[local]!=2){borrow_error(b,n,"exclusive borrowed field mutation");break;}
                int field=borrow_layout_field(b,b->layouts[local],n->as.set.field_name,n);
                if(field<0)break;
                if(borrow_expr(b,n->as.set.value)!=borrow_tag(b->records[b->layouts[local]]->as.struct_def.field_types[field]))
                    borrow_error(b,n,"an exact scalar field value");
                emit_op(cg,OP_REF_SET,local,field);
            } else {
                if(!b->mutable[local] || b->tags[local]==TAG_STRUCT || borrow_expr(b,n->as.set.value)!=b->tags[local]) {
                    borrow_error(b,n,"a mutable scalar assignment of the same type");break;
                }
                emit_op(cg,OP_STORE_LOCAL,local);
            }
        } else if(n->type==AST_RETURN && !shadow) {
            if(b->result==TAG_STRUCT) {
                if(!borrow_owner(b,n->as.return_stmt.value,b->result_layout))break;
            } else if(b->result==TAG_UNION) {
                if(borrow_union_expr(b,n->as.return_stmt.value,(int)b->result_layout)!=TAG_UNION)break;
            } else if(b->result==TAG_VOID) {
                if(n->as.return_stmt.value){borrow_error(b,n,"a zero-value void return");break;}
            } else if(borrow_expr(b,n->as.return_stmt.value)!=b->result)borrow_error(b,n,"the declared scalar return type");
            if(!borrow_consumed(b,0,n))break;
            borrow_cleanup(b,0);emit_op(cg,OP_RET);b->returned=true;
        } else if(n->type==AST_CALL) {
            uint8_t tag=borrow_expr(b,n);
            if(tag==TAG_STRUCT || (tag==TAG_UNION && borrow_owned_union(b,b->expression_layout))){borrow_error(b,n,"explicit consumption of an owned call result");break;}
            if(tag!=TAG_VOID)emit_op(cg,OP_POP);
        } else {borrow_error(b,n,"straight-line let/set/assert/call/return statements");break;}
    }
    if((shadow || b->control_depth) && borrow_consumed(b,first,body))borrow_cleanup(b,first);
    if(shadow || b->control_depth){borrow_names_end(b,first);for(unsigned i=first;i<b->count;i++)b->names[i]=NULL;}
}
static void borrow_function(BorrowCG *b,ASTNode *function,bool shadows,unsigned index) {
    CG *cg=b->cg;b->count=0;b->control_depth=0;b->returned=false;b->transferred=false;b->loop=NULL;b->helper_body=index!=0;
    b->current_function=index;b->current_source=function;if(b->value_graph)b->helper=function;
    b->result=index?borrow_result_tag(b,function):TAG_INT;
    b->result_layout=b->result==TAG_STRUCT?(uint32_t)borrow_record(b,function->as.function.return_struct_type_name):
        b->result==TAG_UNION?(uint32_t)borrow_named_union(b,function->as.function.return_type_info,function->as.function.return_struct_type_name):NVM_V2_NO_INDEX;cg->code_size=0;
    if(index) {
        for(int i=0;i<function->as.function.param_count;i++) {
            Parameter *p=&function->as.function.params[i];
            int union_instance=borrow_named_union(b,p->type_info,p->struct_type_name);
            uint8_t tag=union_instance>=0?TAG_UNION:
                b->value_graph && p->type!=TYPE_STRUCT?borrow_parameter_tag(p->type):TAG_STRUCT;
            uint32_t layout=union_instance>=0?(uint32_t)union_instance:(uint32_t)borrow_record(b,p->struct_type_name);
            borrow_slot(b,p->name,tag,layout,b->value_graph?0:p->type==TYPE_BORROW_MUT?2:1,false);
            borrow_name_begin(b,(unsigned)i,p->name);
        }
    }
    if(shadows && !index) {
        for(unsigned p=0;p<b->program_count && !cg->had_error;p++) {
            ASTNode *selected=b->programs[p];
            for(int i=0;selected && i<selected->as.program.count && !cg->had_error;i++) {
                ASTNode *item=selected->as.program.items[i];
                if(item->type==AST_SHADOW)borrow_body(b,item->as.shadow.body,true);
            }
        }
        emit_op(cg,OP_PUSH_I64,(int64_t)0);emit_op(cg,OP_RET);
    } else {
        borrow_body(b,function->as.function.body,false);
        if(!b->returned) {
            if(b->result==TAG_VOID && borrow_consumed(b,0,function)) {borrow_cleanup(b,0);emit_op(cg,OP_RET);}
            else borrow_error(b,function,"an explicit declared result return");
        }
    }
    borrow_names_end(b,0);
    borrow_word(b,b->count,2);borrow_word(b,index?(uint32_t)function->as.function.param_count:0,2);
    borrow_descriptor(b,b->result,0,b->result_layout);
    for(unsigned i=0;i<b->count;i++)borrow_descriptor(b,b->tags[i],b->modes[i],b->layouts[i]);
    if(cg->had_error)return;
    uint32_t offset=nvm_append_code(cg->module,cg->code,cg->code_size);
    if(offset==UINT32_MAX){borrow_error(b,NULL,"available code storage");return;}
    cg->module->functions[index].code_offset=offset;
    cg->module->functions[index].code_length=cg->code_size;
    cg->module->functions[index].local_count=(uint16_t)b->count;
    for(unsigned i=0;i<b->count && !cg->had_error;i++)borrow_name_retain(b,i);
}
static void borrow_collect_unions(BorrowCG *b,ASTNode *n) {
    if(!n || b->cg->had_error)return;
    if(n->lambda_definition)borrow_collect_unions(b,n->lambda_definition);
    switch(n->type) {
        case AST_PROGRAM:
            for(int i=0;i<n->as.program.count;i++)borrow_collect_unions(b,n->as.program.items[i]);
            break;
        case AST_BLOCK:
            for(int i=0;i<n->as.block.count;i++)borrow_collect_unions(b,n->as.block.statements[i]);
            break;
        case AST_UNSAFE_BLOCK:
            for(int i=0;i<n->as.unsafe_block.count;i++)borrow_collect_unions(b,n->as.unsafe_block.statements[i]);
            break;
        case AST_FUNCTION:
            borrow_register_named_union(b,n->as.function.return_type_info,n->as.function.return_struct_type_name,n);
            for(int i=0;i<n->as.function.param_count;i++)borrow_register_named_union(b,n->as.function.params[i].type_info,n->as.function.params[i].struct_type_name,n);
            borrow_collect_unions(b,n->as.function.body);break;
        case AST_SHADOW:borrow_collect_unions(b,n->as.shadow.body);break;
        case AST_LET:
            borrow_register_named_union(b,n->as.let.type_info,n->as.let.type_name,n);borrow_collect_unions(b,n->as.let.value);break;
        case AST_SET:borrow_collect_unions(b,n->as.set.value);break;
        case AST_PRINT:borrow_collect_unions(b,n->as.print.expr);break;
        case AST_ASSERT:borrow_collect_unions(b,n->as.assert.condition);break;
        case AST_RETURN:borrow_collect_unions(b,n->as.return_stmt.value);break;
        case AST_IF:
            borrow_collect_unions(b,n->as.if_stmt.condition);borrow_collect_unions(b,n->as.if_stmt.then_branch);
            borrow_collect_unions(b,n->as.if_stmt.else_branch);break;
        case AST_WHILE:
            borrow_collect_unions(b,n->as.while_stmt.condition);borrow_collect_unions(b,n->as.while_stmt.body);break;
        case AST_FOR:
            borrow_collect_unions(b,n->as.for_stmt.range_expr);borrow_collect_unions(b,n->as.for_stmt.body);break;
        case AST_MATCH:
            borrow_collect_unions(b,n->as.match_expr.expr);
            for(int i=0;i<n->as.match_expr.arm_count;i++) {
                if(n->as.match_expr.guard_exprs)borrow_collect_unions(b,n->as.match_expr.guard_exprs[i]);
                borrow_collect_unions(b,n->as.match_expr.arm_bodies[i]);
            }
            break;
        case AST_CALL:
            if(n->as.call.checked_signature) {
                borrow_register_union(b,n->as.call.checked_signature->return_type_info,n);
                for(int i=0;i<n->as.call.checked_signature->param_count;i++)
                    borrow_register_union(b,n->as.call.checked_signature->param_type_info?
                        n->as.call.checked_signature->param_type_info[i]:NULL,n);
            }
            borrow_collect_unions(b,n->as.call.func_expr);
            for(int i=0;i<n->as.call.arg_count;i++)borrow_collect_unions(b,n->as.call.args[i]);
            break;
        case AST_MODULE_QUALIFIED_CALL:
            for(int i=0;i<n->as.module_qualified_call.arg_count;i++)
                borrow_collect_unions(b,n->as.module_qualified_call.args[i]);
            break;
        case AST_PREFIX_OP:
            for(int i=0;i<n->as.prefix_op.arg_count;i++)borrow_collect_unions(b,n->as.prefix_op.args[i]);
            break;
        case AST_FIELD_ACCESS:borrow_collect_unions(b,n->as.field_access.object);break;
        case AST_UNION_CONSTRUCT:
            borrow_register_named_union(b,n->as.union_construct.type_info,n->as.union_construct.union_name,n);
            for(int i=0;i<n->as.union_construct.field_count;i++)borrow_collect_unions(b,n->as.union_construct.field_values[i]);
            break;
        case AST_ARRAY_LITERAL:
            for(int i=0;i<n->as.array_literal.element_count;i++)borrow_collect_unions(b,n->as.array_literal.elements[i]);
            break;
        case AST_TUPLE_LITERAL:
            for(int i=0;i<n->as.tuple_literal.element_count;i++)borrow_collect_unions(b,n->as.tuple_literal.elements[i]);
            break;
        default:break;
    }
}
/* I inspect lexical syntax, not the lifetime of checker scope entries. */
static bool borrow_source_uses_owner(ASTNode *n,Environment *env) {
    if(!n)return false;
    if(n->lambda_definition && borrow_source_uses_owner(n->lambda_definition,env))return true;
    const char *name=n->type==AST_LET?n->as.let.type_name:
        n->type==AST_STRUCT_LITERAL?n->as.struct_literal.struct_name:NULL;
    StructDef *record=name?env_get_struct(env,name):NULL;
    if(record && is_resource_type(env,name))return true;
    switch(n->type) {
        case AST_PROGRAM:
            for(int i=0;i<n->as.program.count;i++)if(borrow_source_uses_owner(n->as.program.items[i],env))return true;
            break;
        case AST_BLOCK:
            for(int i=0;i<n->as.block.count;i++)if(borrow_source_uses_owner(n->as.block.statements[i],env))return true;
            break;
        case AST_UNSAFE_BLOCK:
            for(int i=0;i<n->as.unsafe_block.count;i++)if(borrow_source_uses_owner(n->as.unsafe_block.statements[i],env))return true;
            break;
        case AST_FUNCTION:
            if(n->as.function.return_struct_type_name) {
                StructDef *returned=env_get_struct(env,n->as.function.return_struct_type_name);
                if(returned && is_resource_type(env,n->as.function.return_struct_type_name))return true;
            }
            for(int i=0;i<n->as.function.param_count;i++) {
                Parameter *p=&n->as.function.params[i];
                StructDef *parameter=p->struct_type_name?env_get_struct(env,p->struct_type_name):NULL;
                if(p->type==TYPE_BORROW_SHARED || p->type==TYPE_BORROW_MUT || (parameter && is_resource_type(env,p->struct_type_name)))return true;
            }
            return borrow_source_uses_owner(n->as.function.body,env);
        case AST_SHADOW:return borrow_source_uses_owner(n->as.shadow.body,env);
        case AST_LET:return borrow_source_uses_owner(n->as.let.value,env);
        case AST_SET:return borrow_source_uses_owner(n->as.set.value,env);
        case AST_PRINT:return borrow_source_uses_owner(n->as.print.expr,env);
        case AST_ASSERT:return borrow_source_uses_owner(n->as.assert.condition,env);
        case AST_FIELD_ACCESS:return borrow_source_uses_owner(n->as.field_access.object,env);
        case AST_TUPLE_INDEX:return borrow_source_uses_owner(n->as.tuple_index.tuple,env);
        case AST_TRY_OP:return borrow_source_uses_owner(n->as.try_op.operand,env);
        case AST_ASYNC_FN:return borrow_source_uses_owner(n->as.async_fn.function,env);
        case AST_AWAIT:return borrow_source_uses_owner(n->as.await_expr.expr,env);
        case AST_RETURN:return borrow_source_uses_owner(n->as.return_stmt.value,env);
        case AST_IF:return borrow_source_uses_owner(n->as.if_stmt.condition,env) ||
            borrow_source_uses_owner(n->as.if_stmt.then_branch,env) || borrow_source_uses_owner(n->as.if_stmt.else_branch,env);
        case AST_WHILE:return borrow_source_uses_owner(n->as.while_stmt.condition,env) || borrow_source_uses_owner(n->as.while_stmt.body,env);
        case AST_FOR:return borrow_source_uses_owner(n->as.for_stmt.range_expr,env) || borrow_source_uses_owner(n->as.for_stmt.body,env);
        case AST_MATCH:
            if(borrow_source_uses_owner(n->as.match_expr.expr,env))return true;
            for(int i=0;i<n->as.match_expr.arm_count;i++)
                if(borrow_source_uses_owner(n->as.match_expr.arm_bodies[i],env) ||
                   (n->as.match_expr.guard_exprs && borrow_source_uses_owner(n->as.match_expr.guard_exprs[i],env)))return true;
            break;
        case AST_CALL:
            if(borrow_source_uses_owner(n->as.call.func_expr,env))return true;
            for(int i=0;i<n->as.call.arg_count;i++)if(borrow_source_uses_owner(n->as.call.args[i],env))return true;
            break;
        case AST_COND:
            for(int i=0;i<n->as.cond_expr.clause_count;i++)
                if(borrow_source_uses_owner(n->as.cond_expr.conditions[i],env) || borrow_source_uses_owner(n->as.cond_expr.values[i],env))return true;
            return borrow_source_uses_owner(n->as.cond_expr.else_value,env);
        case AST_STRUCT_LITERAL:
            if(borrow_source_uses_owner(n->as.struct_literal.spread_source,env))return true;
            for(int i=0;i<n->as.struct_literal.field_count;i++)if(borrow_source_uses_owner(n->as.struct_literal.field_values[i],env))return true;
            break;
        case AST_UNION_CONSTRUCT:
            for(int i=0;i<n->as.union_construct.field_count;i++)if(borrow_source_uses_owner(n->as.union_construct.field_values[i],env))return true;
            break;
        case AST_ARRAY_LITERAL:
            for(int i=0;i<n->as.array_literal.element_count;i++)if(borrow_source_uses_owner(n->as.array_literal.elements[i],env))return true;
            break;
        case AST_TUPLE_LITERAL:
            for(int i=0;i<n->as.tuple_literal.element_count;i++)if(borrow_source_uses_owner(n->as.tuple_literal.elements[i],env))return true;
            break;
        case AST_MODULE_QUALIFIED_CALL:
            for(int i=0;i<n->as.module_qualified_call.arg_count;i++)if(borrow_source_uses_owner(n->as.module_qualified_call.args[i],env))return true;
            break;
        case AST_PAR_BLOCK:
            for(int i=0;i<n->as.par_block.count;i++)if(borrow_source_uses_owner(n->as.par_block.bindings[i],env))return true;
            break;
        case AST_PAR_LET:
            for(int i=0;i<n->as.par_let.count;i++)if(borrow_source_uses_owner(n->as.par_let.values[i],env))return true;
            return borrow_source_uses_owner(n->as.par_let.body,env);
        case AST_HANDLE_EXPR:
            for(int i=0;i<n->as.handle_expr.handler_count;i++)if(borrow_source_uses_owner(n->as.handle_expr.handler_bodies[i],env))return true;
            return borrow_source_uses_owner(n->as.handle_expr.body,env);
        case AST_EFFECT_HANDLER:
            for(int i=0;i<n->as.effect_handler.handler_count;i++)if(borrow_source_uses_owner(n->as.effect_handler.handler_bodies[i],env))return true;
            return borrow_source_uses_owner(n->as.effect_handler.body,env);
        case AST_EFFECT_OP:
            for(int i=0;i<n->as.effect_op.arg_count;i++)if(borrow_source_uses_owner(n->as.effect_op.args[i],env))return true;
            break;
        case AST_PREFIX_OP:
            for(int i=0;i<n->as.prefix_op.arg_count;i++)if(borrow_source_uses_owner(n->as.prefix_op.args[i],env))return true;
            break;
        default:break;
    }
    return false;
}
static CodegenResult codegen_borrow_compile(ASTNode *program,Environment *env,ModuleList *modules,bool shadows) {
    CodegenResult out={0};CG *cg=calloc(1,sizeof(*cg));BorrowCG *b=calloc(1,sizeof(*b));
    if(!cg || !b){free(cg);free(b);snprintf(out.error_msg,sizeof(out.error_msg),"I cannot allocate my source borrow compiler");return out;}
    b->cg=cg;b->env=env;cg->module=nvm_module_new();cg->code=malloc(CODE_INITIAL);cg->code_cap=CODE_INITIAL;
    ASTNode *entry=NULL,*declared[8];unsigned declared_count=0;bool has_borrow=false;
    if(!cg->module || !cg->code)borrow_error(b,NULL,"available module storage");
    unsigned module_count=modules && modules->count>0?(unsigned)modules->count:0;
    if(module_count>256)borrow_error(b,NULL,"at most 256 dependency source modules");
    else {
        b->programs=calloc(module_count+1,sizeof(*b->programs));
        if(!b->programs)borrow_error(b,NULL,"available dependency closure storage");
    }
    for(unsigned i=0;i<module_count && !cg->had_error;i++) {
        ASTNode *dependency=get_cached_module_ast(modules->module_paths[i]);
        if(!dependency || dependency->type!=AST_PROGRAM)borrow_error(b,NULL,"a loaded dependency source closure");
        else b->programs[b->program_count++]=dependency;
    }
    if(!cg->had_error)b->programs[b->program_count++]=program;
    for(unsigned p=0;p<b->program_count && !cg->had_error;p++) for(int i=0;i<b->programs[p]->as.program.count && !cg->had_error;i++) {
        ASTNode *n=b->programs[p]->as.program.items[i];
        if(n->type==AST_STRUCT_DEF) {
            if(n->as.struct_def.is_extern || b->record_count==MAX_STRUCT_DEFS ||
               n->as.struct_def.field_count>256 || strchr(n->as.struct_def.name,'<') || borrow_record(b,n->as.struct_def.name)>=0) {
                borrow_error(b,n,"distinct finite plain resource records");break;
            }
            bool owned=n->as.struct_def.is_resource, array=false;
            for(int f=0;f<n->as.struct_def.field_count;f++) {
                if(borrow_float_array_field(n,f))array=true;
                else if(!borrow_field_tag(n->as.struct_def.field_types[f])) {
                    int child=n->as.struct_def.field_types[f]==TYPE_STRUCT
                        ?borrow_record(b,n->as.struct_def.field_type_names[f]):-1;
                    if(child<0 || b->ordinary[child] || b->depths[child]>=32)borrow_error(b,n,"earlier resource child declarations and depth at most 32");
                    else {
                        owned=true;
                        if(b->depths[b->record_count]<b->depths[child]+1)b->depths[b->record_count]=b->depths[child]+1;
                    }
                }
                for(int prior=0;prior<f;prior++)
                    if(!strcmp(n->as.struct_def.field_names[prior],n->as.struct_def.field_names[f]))
                        borrow_error(b,n,"distinct field declarations");
            }
            if(owned && array)b->owner_arrays=true;
            if(!owned) {
                if(!array){borrow_error(b,n,"a closed ordinary record with a FLOAT array field");break;}
                for(int f=0;f<n->as.struct_def.field_count;f++)
                    if(!borrow_float_array_field(n,f) && !borrow_tag(n->as.struct_def.field_types[f]))borrow_error(b,n,"ordinary int/bool or flat FLOAT array fields");
                b->ordinary[b->record_count]=true;b->mixed=true;
            }
            b->records[b->record_count++]=n;
        } else if(n->type==AST_UNION_DEF) {
            if(n->as.union_def.is_extern || !env_get_union(env,n->as.union_def.name))
                borrow_error(b,n,"defined local scalar union declarations");
        } else if(n->type==AST_FUNCTION && !n->as.function.is_extern) {
            if(!strcmp(n->as.function.name,"main")){if(entry)borrow_error(b,n,"one main entry");entry=n;}
            else b->helper=n;
            if(declared_count==8){borrow_error(b,n,"at most eight emitted functions");break;}
            for(unsigned j=0;j<declared_count;j++)if(!strcmp(declared[j]->as.function.name,n->as.function.name))borrow_error(b,n,"distinct function declarations");
            declared[declared_count++]=n;
            for(int j=0;j<n->as.function.param_count;j++)if(n->as.function.params[j].type==TYPE_BORROW_SHARED || n->as.function.params[j].type==TYPE_BORROW_MUT)has_borrow=true;
        } else if(n->type!=AST_SHADOW && n->type!=AST_IMPORT && n->type!=AST_MODULE_DECL)
            borrow_error(b,n,"only resource records, scalar unions, main, one helper and selected shadows");
    }
    for(unsigned p=0;p<b->program_count && !cg->had_error;p++)borrow_collect_unions(b,b->programs[p]);
    b->value_graph=!has_borrow;b->consuming=b->value_graph;
    if(b->union_graph && (borrow_arrays(b) || has_borrow))borrow_error(b,NULL,"standalone union value graphs without managed arrays or borrows");
    if(borrow_arrays(b) && has_borrow)borrow_error(b,NULL,"managed arrays only in my value graph");
    if(b->owner_arrays && b->mixed)borrow_error(b,NULL,"separate ordinary Samples and owner ARRAY profiles");
    if(!entry || !b->record_count)borrow_error(b,NULL,"main and exact resource declarations");
    if(!cg->had_error && (entry->as.function.param_count || entry->as.function.return_type!=TYPE_INT))borrow_error(b,entry,"a zero-argument int main");
    if(!cg->had_error && !b->value_graph && declared_count!=2)borrow_error(b,NULL,"exactly main and one borrowed helper");
    if(!cg->had_error) {
        b->functions[b->function_count++]=b->value_graph && shadows?NULL:entry;
        for(unsigned i=0;i<declared_count;i++)if(declared[i]!=entry || (b->value_graph && shadows)) {
            if(b->function_count==8){borrow_error(b,declared[i],"at most eight emitted functions including shadow entry");break;}
            b->functions[b->function_count++]=declared[i];
        }
    }
    for(unsigned f=1;f<b->function_count && !cg->had_error;f++) {
        ASTNode *h=b->functions[f];Parameter *p=h->as.function.params;
        int result_layout=borrow_record(b,h->as.function.return_struct_type_name);
        if(h->as.function.param_count>8 || (h->as.function.param_count && !p))borrow_error(b,h,"at most eight exact parameters");
        if(b->value_graph) {
            if(!borrow_result_tag(b,h) && h->as.function.return_type!=TYPE_VOID &&
               !(h->as.function.return_type==TYPE_STRUCT && result_layout>=0 && !b->ordinary[result_layout]))
                borrow_error(b,h,"an int/bool/float/string, void, exact declared owner or scalar union result");
        } else if(h->as.function.param_count<1 || !borrow_tag(h->as.function.return_type))borrow_error(b,h,"a scalar-result borrowed helper");
        for(int i=0;i<h->as.function.param_count && !cg->had_error;i++) {
            int layout=borrow_record(b,p[i].struct_type_name);
            bool valid=b->value_graph?(borrow_parameter_tag(p[i].type) || borrow_named_union(b,p[i].type_info,p[i].struct_type_name)>=0 ||
                (p[i].type==TYPE_STRUCT && layout>=0 && !b->ordinary[layout])):
                ((p[i].type==TYPE_BORROW_SHARED || p[i].type==TYPE_BORROW_MUT) && layout>=0 && borrow_leaf(b,(unsigned)layout));
            if(!valid)borrow_error(b,h,"exact mode-zero scalar/owner or separate borrowed parameters");
            for(int j=0;j<i;j++)if(!strcmp(p[i].name,p[j].name))borrow_error(b,h,"distinct parameter names");
        }
    }
    NvmV2Layouts layouts={0};
    if(!cg->had_error) {
        layouts.count=b->record_count+b->union_count;layouts.items=calloc(layouts.count,sizeof(*layouts.items));
        if(!layouts.items){layouts.count=0;borrow_error(b,NULL,"available layout storage");}
    }
    for(unsigned i=0;i<layouts.count && !cg->had_error;i++) {
        if(i>=b->record_count) {
            unsigned instance=i-b->record_count;BorrowUnion *u=&b->unions[instance];
            NvmV2Layout *layout=&layouts.items[i];unsigned fields=0;
            for(int variant=0;variant<u->definition->variant_count;variant++)
                fields+=(unsigned)u->definition->variant_field_counts[variant];
            layout->kind=NVM_V2_LAYOUT_UNION;layout->field_count=(uint16_t)fields;
            layout->name_idx=nvm_add_string(cg->module,u->name,(uint32_t)strlen(u->name));
            layout->fields=calloc(fields?fields:1,sizeof(*layout->fields));
            if(layout->name_idx==UINT32_MAX || !layout->fields) {borrow_error(b,NULL,"available union layout storage");break;}
            unsigned at=0;
            for(int variant=0;variant<u->definition->variant_count;variant++)
                for(int field=0;field<u->definition->variant_field_counts[variant];field++,at++) {
                    uint32_t child;
                    uint8_t tag=borrow_union_payload(b,instance,variant,field,&child,NULL);
                    layout->fields[at].type_tag=tag;
                    layout->fields[at].nested_idx=tag==TAG_UNION?b->record_count+child:child;
                    const char *name=u->definition->variant_field_names[variant][field];
                    layout->fields[at].name_idx=nvm_add_string(cg->module,name,(uint32_t)strlen(name));
                    if(layout->fields[at].name_idx==UINT32_MAX)borrow_error(b,NULL,"available union field-name storage");
                }
            continue;
        }
        ASTNode *n=b->records[i];NvmV2Layout *layout=&layouts.items[i];
        layout->kind=NVM_V2_LAYOUT_STRUCT;layout->field_count=(uint16_t)n->as.struct_def.field_count;
        layout->name_idx=nvm_add_string(cg->module,n->as.struct_def.name,(uint32_t)strlen(n->as.struct_def.name));
        if(layout->name_idx==UINT32_MAX){borrow_error(b,n,"available record-name storage");break;}
        layout->fields=calloc(layout->field_count?layout->field_count:1,sizeof(*layout->fields));
        if(!layout->fields){borrow_error(b,n,"available field storage");break;}
        for(unsigned f=0;f<layout->field_count && !cg->had_error;f++) {
            const char *name=n->as.struct_def.field_names[f];layout->fields[f].type_tag=borrow_owner_field_tag(b,i,(int)f);
            int child=borrow_child(b,i,(int)f);
            if(b->ordinary[i] && borrow_float_array_field(n,(int)f))layout->fields[f].type_tag=TAG_ARRAY;
            if(child>=0)layout->fields[f].type_tag=TAG_STRUCT;
            layout->fields[f].nested_idx=child>=0?(uint32_t)child:NVM_V2_NO_INDEX;layout->fields[f].name_idx=nvm_add_string(cg->module,name,(uint32_t)strlen(name));
            if(layout->fields[f].name_idx==UINT32_MAX)borrow_error(b,n,"available field-name storage");
        }
    }
    if(!cg->had_error) {
        cg->module->struct_count=b->record_count;cg->module->union_count=b->union_count;
        if(nvm_retain_layouts(cg->module,&layouts)!=NVM_V2_OK)borrow_error(b,NULL,"complete retained layouts");
        char shadow_name[128]="__nanoisa_shadow_entry";
        while(borrow_function_index(b,shadow_name)>=0 && strlen(shadow_name)<sizeof(shadow_name)-2)strcat(shadow_name,"_");
        if(borrow_function_index(b,shadow_name)>=0)borrow_error(b,NULL,"an available synthetic entry name");
        for(unsigned i=0;i<b->function_count && !cg->had_error;i++) {
            ASTNode *source=b->functions[i];const char *name=i || !shadows?source->as.function.name:shadow_name;
            NvmFunctionEntry f={0};f.name_idx=nvm_add_string(cg->module,name,(uint32_t)strlen(name));
            if(f.name_idx==UINT32_MAX){borrow_error(b,NULL,"available function-name storage");break;}
            f.arity=i?(uint16_t)source->as.function.param_count:0;
            f.result_tag=i?borrow_result_tag(b,source):TAG_INT;f.result_count=f.result_tag!=TAG_VOID;
            if(nvm_add_function(cg->module,&f)!=i)borrow_error(b,NULL,"available function storage");
            uint8_t parameters[8];
            for(unsigned j=0;j<f.arity;j++) {
                Parameter *parameter=&source->as.function.params[j];
                parameters[j]=borrow_named_union(b,parameter->type_info,parameter->struct_type_name)>=0?TAG_UNION:
                    b->value_graph && parameter->type!=TYPE_STRUCT?borrow_parameter_tag(parameter->type):TAG_STRUCT;
            }
            if(!nvm_set_function_param_types(cg->module,i,parameters,f.arity))borrow_error(b,NULL,"exact positional parameter tags");
        }
        borrow_word(b,b->union_graph?NVM_OWNERSHIP_UNION_GRAPH_VERSION:
                    b->union_count?NVM_OWNERSHIP_EXTENSION_VERSION:NVM_OWNERSHIP_VERSION,4);
        borrow_word(b,b->record_count+b->union_count,4);
        for(unsigned i=0;i<b->record_count;i++)borrow_word(b,b->ordinary[i]?1:3,1);
        for(unsigned i=0;i<b->union_count;i++)borrow_word(b,b->union_graph?
            NVM_LAYOUT_COMPLETE|(b->unions[i].resource?NVM_LAYOUT_RESOURCE:0):0,1);
        while(!cg->had_error && b->contract_size%4)borrow_word(b,0,1);
        borrow_word(b,b->function_count,4);
        for(unsigned i=0;i<b->function_count && !cg->had_error;i++)borrow_function(b,i?b->functions[i]:entry,shadows && !i,i);
        /* I intern executable literals before advisory strings in both producers. */
        for(unsigned i=0;i<b->pending_name_count && !cg->had_error;i++)
            if(!nvm_add_local_binding(cg->module,&b->pending_names[i]))
                borrow_error(b,NULL,"available lexical-name storage");
        if((b->path_count || b->union_count) && !cg->had_error) {
            if(!b->union_count)borrow_patch_word(b,0,NVM_OWNERSHIP_PATH_VERSION,1);
            size_t paths_at=b->contract_size;
            if(b->union_count)borrow_word(b,0,4);
            size_t paths_begin=b->contract_size;
            borrow_word(b,b->path_count,4);
            for(unsigned path=0;path<b->path_count;path++) {
                borrow_word(b,b->path_lengths[path],2);borrow_word(b,0,2);
                for(unsigned f=0;f<b->path_lengths[path];f++)borrow_word(b,b->paths[path][f],2);
                while(!cg->had_error && b->contract_size%4)borrow_word(b,0,1);
            }
            if(b->union_count && !cg->had_error) {
                uint32_t path_bytes=(uint32_t)(b->contract_size-paths_begin);
                borrow_patch_word(b,paths_at,path_bytes,4);
                borrow_word(b,1,4);
                borrow_word(b,NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS,2);
                borrow_word(b,NVM_OWNERSHIP_EXTENSION_REVISION_1,2);
                size_t size_at=b->contract_size;borrow_word(b,0,4);size_t payload=b->contract_size;
                borrow_word(b,b->union_count,4);
                for(unsigned instance=0;instance<b->union_count && !cg->had_error;instance++) {
                    BorrowUnion *u=&b->unions[instance];unsigned offset=0;
                    borrow_word(b,b->record_count+instance,4);
                    borrow_word(b,(uint32_t)u->definition->variant_count,2);borrow_word(b,0,2);
                    for(int variant=0;variant<u->definition->variant_count && !cg->had_error;variant++) {
                        const char *name=u->definition->variant_names[variant];
                        uint32_t string=nvm_add_string(cg->module,name,(uint32_t)strlen(name));
                        if(string==UINT32_MAX){borrow_error(b,NULL,"available union variant-name storage");break;}
                        borrow_word(b,string,4);borrow_word(b,offset,2);
                        borrow_word(b,(uint32_t)u->definition->variant_field_counts[variant],2);
                        offset+=(unsigned)u->definition->variant_field_counts[variant];
                    }
                }
                uint32_t bytes=(uint32_t)(b->contract_size-payload);
                borrow_patch_word(b,size_at,bytes,4);
                while(!cg->had_error && b->contract_size%4)borrow_word(b,0,1);
            }
        }
        if(!cg->had_error) {
            cg->module->ownership_data=b->contracts;b->contracts=NULL;cg->module->ownership_size=(uint32_t)b->contract_size;
            cg->module->header.flags|=NVM_FLAG_HAS_MAIN;cg->module->header.entry_point=0;
            if(nvm_service_bindings_present(cg->module))borrow_error(b,NULL,"a source graph without service bindings");
            NvmVerifyResult verified={0};
            if(!cg->had_error)verified=nvm_verify(cg->module);
            if(verified.ok && b->owner_arrays) {
                NvmOwnedArrayPlan *plan=NULL;
                if(nvm_owned_array_route(cg->module)!=NVM_OWNER_ARRAY_SELECTED) {
                    borrow_error(b,NULL,"a positive complete owner ARRAY execution route");verified.ok=false;
                } else {
                    NvmOwnerAuthorityResult admitted=nvm_owned_array_admit(cg->module,&plan);
                    if(admitted.status!=NVM_OWNER_AUTH_PREPARED) {
                        borrow_error(b,NULL,admitted.message);verified.ok=false;
                    }
                }
                nvm_owned_array_plan_free(plan);
            }
            if(verified.ok && b->mixed && !nvm_mixed_samples_candidate(cg->module)) {
                borrow_error(b,NULL,"a positive complete mixed execution candidate");verified.ok=false;
            }
            if(verified.ok && !borrow_arrays(b))verified=nvm_verify_owned_module(cg->module);
            if(!verified.ok && !cg->had_error)borrow_error(b,NULL,verified.error_msg);
        }
    }
    out.ok=!cg->had_error;out.error_line=cg->error_line;snprintf(out.error_msg,sizeof(out.error_msg),"%s",cg->error_msg);
    if(out.ok)out.module=cg->module;else if(cg->module)nvm_module_free(cg->module);
    nvm_v2_layouts_free(&layouts);for(unsigned i=0;i<b->union_count;i++) {
        free(b->unions[i].name);free_payload_type_info(b->unions[i].arguments);
    }
    free(b->contracts);free(b->programs);free(cg->code);free(cg);free(b);return out;
}
