/* I lower a closed source borrow profile; ordinary lowering stays separate.
 * Every accepted module still passes the shared owned verifier. */
#include "nanoisa/retained_layouts.h"
#include "nanoisa/ownership_contracts.h"
#include "nanoisa/verifier.h"

typedef struct {
    CG *cg;
    ASTNode *records[MAX_STRUCT_DEFS], *helper;
    unsigned record_count, count, control_depth;
    unsigned depths[MAX_STRUCT_DEFS], path_count;
    uint16_t paths[256][32], path_lengths[256];
    int projection_parent[256], projection_field[256];
    const char *names[256], *advisory_names[256];
    uint32_t name_begin[256], name_end[256];
    bool name_open[256];
    uint8_t tags[256], modes[256];
    uint32_t layouts[256];
    bool live[256], disposal_pending[256], mutable[256], helper_body, returned;
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
static void borrow_word(BorrowCG *b,uint32_t value,unsigned width) {
    if(b->cg->had_error)return;
    if(b->contract_size+width>b->contract_capacity) {
        size_t capacity=b->contract_capacity?b->contract_capacity*2:128;
        uint8_t *next=realloc(b->contracts,capacity);
        if(!next){borrow_error(b,NULL,"available contract storage");return;}
        b->contracts=next;b->contract_capacity=capacity;
    }
    for(unsigned i=0;i<width;i++)b->contracts[b->contract_size++]=(uint8_t)(value>>(8*i));
}
static void borrow_descriptor(BorrowCG *b,uint8_t tag,uint8_t mode,uint32_t layout) {
    borrow_word(b,tag,1);borrow_word(b,mode,1);borrow_word(b,0,2);borrow_word(b,layout,4);
}
static int borrow_record(BorrowCG *b,const char *name) {
    for(unsigned i=0;name && i<b->record_count;i++)
        if(!strcmp(name,b->records[i]->as.struct_def.name))return (int)i;
    return -1;
}
static int borrow_local(BorrowCG *b,const char *name) {
    for(unsigned i=b->count;i>0;i--)
        if(name && b->names[i-1] && !strcmp(name,b->names[i-1]))return (int)i-1;
    return -1;
}
static int borrow_slot(BorrowCG *b,const char *name,uint8_t tag,uint32_t layout,uint8_t mode,bool mutable) {
    if(b->count==256){borrow_error(b,NULL,"at most 256 local slots");return 0;}
    unsigned i=b->count++;b->projection_parent[i]=-1;b->projection_field[i]=-1;b->advisory_names[i]=NULL;b->name_open[i]=false;b->names[i]=name;b->tags[i]=tag;b->layouts[i]=layout;
    b->modes[i]=mode;b->live[i]=true;b->disposal_pending[i]=false;b->mutable[i]=mutable;return (int)i;
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
        if(local<0 || !b->live[local] || b->tags[local]!=TAG_STRUCT) {
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
        slots[f]=borrow_slot(b,NULL,child>=0?TAG_STRUCT:borrow_tag(record->as.struct_def.field_types[f]),
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
        if(b->live[i] && b->tags[i]==TAG_STRUCT && !b->modes[i]) {
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
        if(b->live[i] && b->tags[i]==TAG_STRUCT && !b->modes[i] && !b->disposal_pending[i]) {
            borrow_error(b,at,"explicit source owner consumption at scope exit");return false;
        }
    return !b->cg->had_error;
}
static bool borrow_same_owners(BorrowCG *b,unsigned count,const bool *live,const bool *pending,ASTNode *at) {
    for(unsigned i=0;i<count;i++)if(b->tags[i]==TAG_STRUCT &&
       (b->live[i]!=live[i] || b->disposal_pending[i]!=pending[i])) {
        borrow_error(b,at,"matching ownership on every reaching edge");return false;
    }
    return !b->cg->had_error;
}
static uint8_t borrow_expr(BorrowCG *b,ASTNode *n) {
    CG *cg=b->cg;if(cg->had_error || !n)return TAG_VOID;
    if(n->type==AST_NUMBER){emit_op(cg,OP_PUSH_I64,(int64_t)n->as.number);return TAG_INT;}
    if(n->type==AST_BOOL){emit_op(cg,OP_PUSH_BOOL,(int)n->as.bool_val);return TAG_BOOL;}
    if(n->type==AST_IDENTIFIER) {
        int local=borrow_local(b,n->as.identifier);
        if(local<0 || !b->live[local] || b->tags[local]==TAG_STRUCT) {
            borrow_error(b,n,"a live scalar value, not a copied owner/reference");return TAG_VOID;
        }
        emit_op(cg,OP_LOAD_LOCAL,local);return b->tags[local];
    }
    if(n->type==AST_FIELD_ACCESS) {
        BorrowPlace place;
        if(!borrow_place(b,n->as.field_access.object,&place))return TAG_VOID;
        int field=borrow_layout_field(b,place.layout,n->as.field_access.field_name,n);
        if(field<0)return TAG_VOID;
        uint8_t tag=borrow_tag(b->records[place.layout]->as.struct_def.field_types[field]);
        if(!tag || !borrow_leaf(b,place.layout)){borrow_error(b,n,"a scalar field of a leaf resource record");return TAG_VOID;}
        if(!b->modes[place.root]) { emit_op(cg,OP_REGION_BEGIN);borrow_reference(b,&place,0,1); }
        emit_op(cg,OP_REF_GET,b->modes[place.root]?place.root:0,field);
        if(!b->modes[place.root])emit_op(cg,OP_REGION_END);
        return tag;
    }
    if(n->type==AST_PREFIX_OP) {
        int count=n->as.prefix_op.arg_count;TokenType op=n->as.prefix_op.op;
        if(count<1 || count>2){borrow_error(b,n,"one unary or two binary scalar operands");return TAG_VOID;}
        uint8_t left=borrow_expr(b,n->as.prefix_op.args[0]);
        if(count==1) {
            if(op==TOKEN_NOT && left==TAG_BOOL){emit_op(cg,OP_NOT);return TAG_BOOL;}
            if(op==TOKEN_MINUS && left==TAG_INT){emit_op(cg,OP_NEG);return TAG_INT;}
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
            if(left==TAG_INT || left==TAG_BOOL) switch(op) {
                case TOKEN_EQ:opcode=OP_EQ;result=TAG_BOOL;break;
                case TOKEN_NE:opcode=OP_NE;result=TAG_BOOL;break;
                case TOKEN_LT:opcode=OP_LT;result=TAG_BOOL;break;
                case TOKEN_LE:opcode=OP_LE;result=TAG_BOOL;break;
                case TOKEN_GT:opcode=OP_GT;result=TAG_BOOL;break;
                case TOKEN_GE:opcode=OP_GE;result=TAG_BOOL;break;default:break;
            }
            if(opcode){emit_op(cg,opcode);return result;}
        }
        borrow_error(b,n,"supported exact-tag scalar operators");return TAG_VOID;
    }
    if(n->type==AST_CALL && !b->helper_body && !n->as.call.func_expr &&
       n->as.call.name && !strcmp(n->as.call.name,b->helper->as.function.name) &&
       n->as.call.arg_count==b->helper->as.function.param_count) {
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
static void borrow_body(BorrowCG *b,ASTNode *body,bool shadow) {
    if(!body || body->type!=AST_BLOCK){borrow_error(b,body,"a lexical function/shadow body");return;}
    unsigned first=b->count;
    for(int i=0;i<body->as.block.count && !b->cg->had_error && !b->returned;i++) {
        ASTNode *n=body->as.block.statements[i];CG *cg=b->cg;
        if(n->type==AST_IF || n->type==AST_WHILE) {
            if(b->control_depth==32){borrow_error(b,n,"at most 32 nested control-flow bodies");break;}
            uint32_t top=cg->code_size;
            ASTNode *condition=n->type==AST_IF?n->as.if_stmt.condition:n->as.while_stmt.condition;
            if(borrow_expr(b,condition)!=TAG_BOOL){borrow_error(b,n,"a Boolean control-flow condition");break;}
            uint32_t skip=emit_op(cg,OP_JMP_FALSE,(int32_t)0);
            unsigned incoming_count=b->count;
            bool incoming_live[256],incoming_disposal[256];
            memcpy(incoming_live,b->live,incoming_count*sizeof(bool));
            memcpy(incoming_disposal,b->disposal_pending,incoming_count*sizeof(bool));
            b->control_depth++;
            borrow_body(b,n->type==AST_IF?n->as.if_stmt.then_branch:n->as.while_stmt.body,shadow);
            b->control_depth--;
            if(cg->had_error)break;
            bool then_returns=b->returned;
            bool then_live[256],then_disposal[256];
            memcpy(then_live,b->live,incoming_count*sizeof(bool));
            memcpy(then_disposal,b->disposal_pending,incoming_count*sizeof(bool));
            uint32_t next=UINT32_MAX;
            if(!then_returns)next=emit_op(cg,OP_JMP,(int32_t)0);
            patch_jump(cg,skip+1,skip,cg->code_size);
            memcpy(b->live,incoming_live,incoming_count*sizeof(bool));
            memcpy(b->disposal_pending,incoming_disposal,incoming_count*sizeof(bool));b->returned=false;
            if(n->type==AST_IF) {
                if(n->as.if_stmt.else_branch) {
                    b->control_depth++;borrow_body(b,n->as.if_stmt.else_branch,shadow);b->control_depth--;
                }
                bool else_returns=b->returned;
                if(!then_returns) {
                    if(!else_returns)borrow_same_owners(b,incoming_count,then_live,then_disposal,n);
                    else {
                        memcpy(b->live,then_live,incoming_count*sizeof(bool));
                        memcpy(b->disposal_pending,then_disposal,incoming_count*sizeof(bool));
                    }
                }
                b->returned=then_returns && else_returns;
                if(!cg->had_error && next!=UINT32_MAX)patch_jump(cg,next+1,next,cg->code_size);
            } else if(next!=UINT32_MAX) {
                if(borrow_same_owners(b,incoming_count,then_live,then_disposal,n))patch_jump(cg,next+1,next,top);
            }
        } else if(n->type==AST_LET) {
            if(n->as.let.is_destructure_projection && n->as.let.value->type==AST_FIELD_ACCESS &&
               n->as.let.value->as.field_access.object->type==AST_IDENTIFIER) {
                ASTNode *field=n->as.let.value;
                int parent=borrow_local(b,field->as.field_access.object->as.identifier);
                if(parent>=0 && !b->live[parent] && b->tags[parent]==TAG_STRUCT && !borrow_leaf(b,b->layouts[parent])) {
                    int index=borrow_layout_field(b,b->layouts[parent],field->as.field_access.field_name,n),source=-1;
                    for(unsigned j=0;j<b->count;j++)if(b->projection_parent[j]==parent && b->projection_field[j]==index)source=(int)j;
                    if(source<0 || !b->live[source]){borrow_error(b,n,"an unconsumed exact destructured field");break;}
                    emit_op(cg,b->tags[source]==TAG_STRUCT?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,source);
                    if(b->tags[source]==TAG_STRUCT)b->live[source]=false;
                    int slot=borrow_slot(b,n->as.let.name,b->tags[source],b->layouts[source],0,n->as.let.is_mut);
                    emit_op(cg,b->tags[source]==TAG_STRUCT?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slot);
                    borrow_name_begin(b,(unsigned)slot,n->as.let.name);continue;
                }
            }
            int layout=borrow_record(b,n->as.let.type_name);
            if(layout>=0) {
                if(b->helper_body){borrow_error(b,n,"only scalar helper locals");break;}
                ASTNode *value=n->as.let.value;
                if(value->type==AST_STRUCT_LITERAL && !value->as.struct_literal.spread_source &&
                   !strcmp(value->as.struct_literal.struct_name,n->as.let.type_name)) {
                    ASTNode *record=b->records[layout];int fields=record->as.struct_def.field_count;
                    int slots[256];for(int f=0;f<256;f++)slots[f]=-1;
                    if(value->as.struct_literal.field_count!=fields){borrow_error(b,n,"every declared scalar field exactly once");break;}
                    for(int f=0;f<fields && !cg->had_error;f++) {
                        int index=-1;for(int j=0;j<fields;j++)
                            if(!strcmp(value->as.struct_literal.field_names[f],record->as.struct_def.field_names[j]))index=j;
                        if(index<0 || slots[index]>=0){borrow_error(b,n,"distinct declared constructor fields");break;}
                        int child=borrow_child(b,(unsigned)layout,index);uint8_t tag;
                        if(child>=0) {
                            ASTNode *item=value->as.struct_literal.field_values[f];
                            int source=item->type==AST_IDENTIFIER?borrow_local(b,item->as.identifier):-1;
                            if(source<0 || !b->live[source] || b->modes[source] || b->tags[source]!=TAG_STRUCT || b->layouts[source]!=(uint32_t)child) {
                                borrow_error(b,n,"an exact live child owner in a nested constructor");break;
                            }
                            emit_op(cg,OP_OWN_MOVE_LOCAL,source);b->live[source]=false;tag=TAG_STRUCT;
                        } else {
                            tag=borrow_expr(b,value->as.struct_literal.field_values[f]);
                            if(tag!=borrow_tag(record->as.struct_def.field_types[index])){borrow_error(b,n,"exact constructor field types");break;}
                        }
                        slots[index]=borrow_slot(b,NULL,tag,child>=0?(uint32_t)child:NVM_V2_NO_INDEX,0,false);
                        emit_op(cg,child>=0?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slots[index]);
                    }
                    if(cg->had_error)break;
                    for(int f=0;f<fields;f++) {
                        emit_op(cg,b->tags[slots[f]]==TAG_STRUCT?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,slots[f]);
                        if(b->tags[slots[f]]==TAG_STRUCT)b->live[slots[f]]=false;
                    }
                    emit_op(cg,OP_OWN_PACK,(uint32_t)layout);
                } else if(value->type==AST_IDENTIFIER) {
                    int old=borrow_local(b,value->as.identifier);
                    if(old<0 || !b->live[old] || b->modes[old] || b->tags[old]!=TAG_STRUCT || b->layouts[old]!=(uint32_t)layout) {
                        borrow_error(b,n,"an unborrowed exact owner for destructuring");break;
                    }
                    emit_op(cg,OP_OWN_MOVE_LOCAL,old);b->live[old]=false;b->disposal_pending[old]=false;
                } else {borrow_error(b,n,"a resource constructor, exact owner move or explicit destructive pattern");break;}
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
                        if(borrow_leaf(b,(unsigned)layout))b->disposal_pending[slot]=true;
                        else borrow_unpack(b,(unsigned)slot);
                    }
                }
            } else {
                uint8_t tag=borrow_expr(b,n->as.let.value),declared=borrow_tag(n->as.let.var_type);
                if((n->as.let.var_type!=TYPE_UNKNOWN && tag!=declared) || (tag!=TAG_INT && tag!=TAG_BOOL)) {
                    borrow_error(b,n,"int/bool scalar local types");break;
                }
                int slot=borrow_slot(b,n->as.let.name,tag,NVM_V2_NO_INDEX,0,n->as.let.is_mut);
                emit_op(cg,OP_STORE_LOCAL,slot);
                borrow_name_begin(b,(unsigned)slot,n->as.let.name);
            }
        } else if(n->type==AST_ASSERT) {
            if(borrow_expr(b,n->as.assert.condition)!=TAG_BOOL)borrow_error(b,n,"a Boolean assertion");
            emit_op(cg,OP_ASSERT);
        } else if(n->type==AST_SET) {
            int local=borrow_local(b,n->as.set.name);
            if(local<0 || !b->live[local]){borrow_error(b,n,"an existing live mutable target");break;}
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
            if(!borrow_consumed(b,0,n))break;
            if(borrow_expr(b,n->as.return_stmt.value)!=b->result)borrow_error(b,n,"the declared scalar return type");
            borrow_cleanup(b,0);emit_op(cg,OP_RET);b->returned=true;
        } else if(n->type==AST_CALL) {
            if(borrow_expr(b,n)==TAG_VOID)break;
            emit_op(cg,OP_POP);
        } else {borrow_error(b,n,"straight-line let/set/assert/call/return statements");break;}
    }
    if((shadow || b->control_depth) && borrow_consumed(b,first,body))borrow_cleanup(b,first);
    if(shadow || b->control_depth){borrow_names_end(b,first);for(unsigned i=first;i<b->count;i++)b->names[i]=NULL;}
}
static void borrow_function(BorrowCG *b,ASTNode *program,ASTNode *function,bool shadows,unsigned index) {
    CG *cg=b->cg;b->count=0;b->control_depth=0;b->returned=false;b->helper_body=index==1;
    b->result=index?borrow_tag(function->as.function.return_type):TAG_INT;cg->code_size=0;
    if(index) {
        for(int i=0;i<function->as.function.param_count;i++) {
            Parameter *p=&function->as.function.params[i];
            borrow_slot(b,p->name,TAG_STRUCT,(uint32_t)borrow_record(b,p->struct_type_name),p->type==TYPE_BORROW_MUT?2:1,false);
            borrow_name_begin(b,(unsigned)i,p->name);
        }
    }
    if(shadows && !index) {
        for(int i=0;i<program->as.program.count && !cg->had_error;i++) {
            ASTNode *item=program->as.program.items[i];
            if(item->type==AST_SHADOW)borrow_body(b,item->as.shadow.body,true);
        }
        emit_op(cg,OP_PUSH_I64,(int64_t)0);emit_op(cg,OP_RET);
    } else {
        borrow_body(b,function->as.function.body,false);
        if(!b->returned)borrow_error(b,function,"an explicit scalar return");
    }
    borrow_names_end(b,0);
    borrow_word(b,b->count,2);borrow_word(b,index?(uint32_t)function->as.function.param_count:0,2);
    borrow_descriptor(b,b->result,0,NVM_V2_NO_INDEX);
    for(unsigned i=0;i<b->count;i++)borrow_descriptor(b,b->tags[i],b->modes[i],b->layouts[i]);
    if(cg->had_error)return;
    uint32_t offset=nvm_append_code(cg->module,cg->code,cg->code_size);
    if(offset==UINT32_MAX){borrow_error(b,NULL,"available code storage");return;}
    cg->module->functions[index].code_offset=offset;
    cg->module->functions[index].code_length=cg->code_size;
    cg->module->functions[index].local_count=(uint16_t)b->count;
    for(unsigned i=0;i<b->count && !cg->had_error;i++)if(b->advisory_names[i]) {
        NvmLocalBinding binding={.function=index,.slot=(uint16_t)i,
            .begin=b->name_begin[i],.end=b->name_end[i],
            .name=(const uint8_t *)b->advisory_names[i],.name_size=(uint32_t)strlen(b->advisory_names[i])};
        if(!nvm_add_local_binding(cg->module,&binding))borrow_error(b,NULL,"available lexical-name storage");
    }
}
static CodegenResult codegen_borrow_compile(ASTNode *program,ModuleList *modules,bool shadows) {
    CodegenResult out={0};CG *cg=calloc(1,sizeof(*cg));BorrowCG *b=calloc(1,sizeof(*b));
    if(!cg || !b){free(cg);free(b);snprintf(out.error_msg,sizeof(out.error_msg),"I cannot allocate my source borrow compiler");return out;}
    b->cg=cg;cg->module=nvm_module_new();cg->code=malloc(CODE_INITIAL);cg->code_cap=CODE_INITIAL;
    ASTNode *entry=NULL;
    if(!cg->module || !cg->code)borrow_error(b,NULL,"available module storage");
    if(modules && modules->count)borrow_error(b,NULL,"an import-free source module");
    for(int i=0;i<program->as.program.count && !cg->had_error;i++) {
        ASTNode *n=program->as.program.items[i];
        if(n->type==AST_STRUCT_DEF) {
            if(!n->as.struct_def.is_resource || n->as.struct_def.is_extern || b->record_count==MAX_STRUCT_DEFS ||
               n->as.struct_def.field_count>256 || strchr(n->as.struct_def.name,'<') || borrow_record(b,n->as.struct_def.name)>=0) {
                borrow_error(b,n,"distinct finite plain resource records");break;
            }
            for(int f=0;f<n->as.struct_def.field_count;f++) {
                if(!borrow_tag(n->as.struct_def.field_types[f])) {
                    int child=n->as.struct_def.field_types[f]==TYPE_STRUCT
                        ?borrow_record(b,n->as.struct_def.field_type_names[f]):-1;
                    if(child<0 || b->depths[child]>=32)borrow_error(b,n,"earlier resource child declarations and depth at most 32");
                    else if(b->depths[b->record_count]<b->depths[child]+1)b->depths[b->record_count]=b->depths[child]+1;
                }
                for(int prior=0;prior<f;prior++)
                    if(!strcmp(n->as.struct_def.field_names[prior],n->as.struct_def.field_names[f]))
                        borrow_error(b,n,"distinct field declarations");
            }
            b->records[b->record_count++]=n;
        } else if(n->type==AST_FUNCTION && !n->as.function.is_extern) {
            if(!strcmp(n->as.function.name,"main")){if(entry)borrow_error(b,n,"one main entry");entry=n;}
            else {if(b->helper)borrow_error(b,n,"one borrowed helper");b->helper=n;}
        } else if(n->type!=AST_SHADOW)borrow_error(b,n,"only resource records, main, one helper and selected shadows");
    }
    if(!entry || !b->helper || !b->record_count)borrow_error(b,NULL,"main, one borrowed helper and a resource record");
    if(!cg->had_error) {
        ASTNode *h=b->helper;Parameter *p=h->as.function.params;
        if(entry->as.function.param_count || entry->as.function.return_type!=TYPE_INT ||
           h->as.function.param_count<1 || h->as.function.param_count>8 || !p || !borrow_tag(h->as.function.return_type))
            borrow_error(b,h,"entry0 and one scalar-result helper with one through eight borrowed parameters");
        for(int i=0;i<h->as.function.param_count && !cg->had_error;i++) {
            if((p[i].type!=TYPE_BORROW_SHARED && p[i].type!=TYPE_BORROW_MUT) || borrow_record(b,p[i].struct_type_name)<0 ||
               !borrow_leaf(b,(unsigned)borrow_record(b,p[i].struct_type_name)))
                borrow_error(b,h,"exact borrowed record parameters");
            for(int j=0;j<i;j++)if(!strcmp(p[i].name,p[j].name))borrow_error(b,h,"distinct parameter names");
        }
    }
    NvmV2Layouts layouts={0};
    if(!cg->had_error) {
        layouts.count=b->record_count;layouts.items=calloc(layouts.count,sizeof(*layouts.items));
        if(!layouts.items){layouts.count=0;borrow_error(b,NULL,"available layout storage");}
    }
    for(unsigned i=0;i<layouts.count && !cg->had_error;i++) {
        ASTNode *n=b->records[i];NvmV2Layout *layout=&layouts.items[i];
        layout->kind=NVM_V2_LAYOUT_STRUCT;layout->field_count=(uint16_t)n->as.struct_def.field_count;
        layout->name_idx=nvm_add_string(cg->module,n->as.struct_def.name,(uint32_t)strlen(n->as.struct_def.name));
        if(layout->name_idx==UINT32_MAX){borrow_error(b,n,"available record-name storage");break;}
        layout->fields=calloc(layout->field_count?layout->field_count:1,sizeof(*layout->fields));
        if(!layout->fields){borrow_error(b,n,"available field storage");break;}
        for(unsigned f=0;f<layout->field_count && !cg->had_error;f++) {
            const char *name=n->as.struct_def.field_names[f];layout->fields[f].type_tag=borrow_tag(n->as.struct_def.field_types[f]);
            int child=borrow_child(b,i,(int)f);
            if(child>=0)layout->fields[f].type_tag=TAG_STRUCT;
            layout->fields[f].nested_idx=child>=0?(uint32_t)child:NVM_V2_NO_INDEX;layout->fields[f].name_idx=nvm_add_string(cg->module,name,(uint32_t)strlen(name));
            if(layout->fields[f].name_idx==UINT32_MAX)borrow_error(b,n,"available field-name storage");
        }
    }
    if(!cg->had_error) {
        cg->module->struct_count=b->record_count;
        if(nvm_retain_layouts(cg->module,&layouts)!=NVM_V2_OK)borrow_error(b,NULL,"complete retained layouts");
        const char *shadow_name = !strcmp(b->helper->as.function.name,"__nanoisa_shadow_entry")
            ? "__nanoisa_shadow_entry_" : "__nanoisa_shadow_entry";
        const char *names[2]={shadows?shadow_name:"main",b->helper->as.function.name};
        for(unsigned i=0;i<2;i++) {
            NvmFunctionEntry f={0};f.name_idx=nvm_add_string(cg->module,names[i],(uint32_t)strlen(names[i]));
            if(f.name_idx==UINT32_MAX){borrow_error(b,NULL,"available function-name storage");break;}
            f.arity=i?(uint16_t)b->helper->as.function.param_count:0;f.result_count=1;f.result_tag=i?borrow_tag(b->helper->as.function.return_type):TAG_INT;
            if(nvm_add_function(cg->module,&f)!=i)borrow_error(b,NULL,"available function storage");
        }
        uint8_t parameters[8];memset(parameters,TAG_STRUCT,sizeof(parameters));
        if(!nvm_set_function_param_types(cg->module,1,parameters,(uint16_t)b->helper->as.function.param_count))borrow_error(b,NULL,"exact borrowed parameter tags");
        borrow_word(b,1,4);borrow_word(b,b->record_count,4);
        for(unsigned i=0;i<b->record_count;i++)borrow_word(b,3,1);
        while(!cg->had_error && b->contract_size%4)borrow_word(b,0,1);
        borrow_word(b,2,4);
        borrow_function(b,program,entry,shadows,0);
        if(!cg->had_error)borrow_function(b,program,b->helper,false,1);
        if(b->path_count && !cg->had_error) {
            b->contracts[0]=2;borrow_word(b,b->path_count,4);
            for(unsigned path=0;path<b->path_count;path++) {
                borrow_word(b,b->path_lengths[path],2);borrow_word(b,0,2);
                for(unsigned f=0;f<b->path_lengths[path];f++)borrow_word(b,b->paths[path][f],2);
                while(!cg->had_error && b->contract_size%4)borrow_word(b,0,1);
            }
        }
        cg->module->ownership_data=b->contracts;b->contracts=NULL;cg->module->ownership_size=(uint32_t)b->contract_size;
        cg->module->header.flags|=NVM_FLAG_HAS_MAIN;cg->module->header.entry_point=0;
        if(!cg->had_error){NvmVerifyResult verified=nvm_verify(cg->module);if(!verified.ok)borrow_error(b,NULL,verified.error_msg);}
    }
    out.ok=!cg->had_error;out.error_line=cg->error_line;snprintf(out.error_msg,sizeof(out.error_msg),"%s",cg->error_msg);
    if(out.ok)out.module=cg->module;else if(cg->module)nvm_module_free(cg->module);
    nvm_v2_layouts_free(&layouts);free(b->contracts);free(cg->code);free(cg);free(b);return out;
}
