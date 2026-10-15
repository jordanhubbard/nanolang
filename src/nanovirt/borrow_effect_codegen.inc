/* I retain exact operation declarations and move owners across handler edges. */
static NvmOwnershipEffectValue borrow_effect_type(BorrowCG *b,Type type,const TypeInfo *info,
                                                  const char *name,ASTNode *at,bool result) {
    NvmOwnershipEffectValue value={TAG_VOID,NVM_V2_NO_INDEX};
    if(type==TYPE_ARRAY) {
        value.tag=TAG_ARRAY;value.layout=borrow_array_type(b,info,at);return value;
    }
    borrow_register_named_union(b,info,name,at);
    int instance=borrow_named_union_instance(b,info,name),record=borrow_record(b,name);
    if(instance>=0) {
        value.tag=TAG_UNION;value.layout=(uint32_t)instance;b->unions[instance].complete=true;
    } else if(type==TYPE_STRUCT && record>=0) {
        value.tag=TAG_STRUCT;value.layout=(uint32_t)record;
    } else value.tag=borrow_local_tag(type);
    if(!value.tag && !(result && type==TYPE_VOID))
        borrow_error(b,at,"an exact scalar, STRING, record or union effect signature");
    return value;
}
static BorrowEffect *borrow_effect_register(BorrowCG *b,const char *effect_name,const char *operation_name,ASTNode *at) {
    EffectDef *effect=env_get_effect(b->env,effect_name);
    EffectOp *operation=effect?effect_get_op(effect,operation_name):NULL;
    if(!operation || operation->param_count<0 || operation->param_count>256) {
        borrow_error(b,at,"a resolved bounded effect operation");return NULL;
    }
    for(unsigned i=0;i<b->effect_count;i++)if(b->effects[i].operation==operation)return &b->effects[i];
    if(b->effect_count==256){borrow_error(b,at,"at most 256 retained effect operations");return NULL;}
    BorrowEffect *entry=&b->effects[b->effect_count++];entry->definition=effect;entry->operation=operation;
    entry->signature.key_idx=effect_operation(b->cg,effect->name,operation->name);
    if(entry->signature.key_idx==UINT32_MAX){borrow_error(b,at,"available effect key storage");return NULL;}
    entry->signature.parameter_count=(uint16_t)operation->param_count;
    entry->signature.result=borrow_effect_type(b,operation->return_type,operation->return_type_info,operation->return_type_name,at,true);
    for(int i=0;i<operation->param_count;i++) {
        Parameter *parameter=&operation->params[i];
        entry->signature.parameters[i]=borrow_effect_type(b,parameter->type,parameter->type_info,parameter->struct_type_name,at,false);
    }
    return b->cg->had_error?NULL:entry;
}
static void borrow_effect_contracts(BorrowCG *b) {
    unsigned order[256];for(unsigned i=0;i<b->effect_count;i++)order[i]=i;
    for(unsigned i=1;i<b->effect_count;i++)for(unsigned j=i;j &&
        b->effects[order[j]].signature.key_idx<b->effects[order[j-1]].signature.key_idx;j--) {
        unsigned previous=order[j-1];order[j-1]=order[j];order[j]=previous;
    }
    borrow_word(b,NVM_OWNERSHIP_EXTENSION_EFFECTS,2);borrow_word(b,b->array_types.count?NVM_OWNERSHIP_EFFECT_CONSTRUCTOR_REVISION:NVM_OWNERSHIP_EXTENSION_REVISION_1,2);
    size_t size_at=b->contract_size;borrow_word(b,0,4);size_t payload=b->contract_size;
    if(b->array_types.count) {
        borrow_word(b,b->array_types.count,4);
        for(unsigned i=0;i<b->array_types.count;i++)
            borrow_descriptor(b,b->array_types.items[i].tag,0,b->array_types.items[i].layout);
        borrow_word(b,b->array_site_count,4);
        for(unsigned i=0;i<b->array_site_count;i++) {
            borrow_word(b,b->array_sites[i].function,4);borrow_word(b,b->array_sites[i].pc,4);borrow_word(b,b->array_sites[i].element,4);
        }
    }
    borrow_word(b,b->effect_count,4);
    for(unsigned i=0;i<b->effect_count;i++) {
        NvmOwnershipEffect *effect=&b->effects[order[i]].signature;
        borrow_word(b,effect->key_idx,4);borrow_word(b,effect->parameter_count,2);borrow_word(b,0,2);
        borrow_descriptor(b,effect->result.tag,0,effect->result.layout);
        for(unsigned p=0;p<effect->parameter_count;p++)
            borrow_descriptor(b,effect->parameters[p].tag,0,effect->parameters[p].layout);
    }
    borrow_patch_word(b,size_at,(uint32_t)(b->contract_size-payload),4);
}
/* Destructuring already consumed these source owners. I finish their retained
 * scalar shells before a suspension can leave this lexical function. */
static void borrow_dispose_patterns(BorrowCG *b) {
    for(unsigned i=0;i<b->count && !b->cg->had_error;i++)if(b->live[i] && b->disposal_pending[i]) {
        emit_op(b->cg,OP_OWN_UNPACK_LOCAL,i);
        for(int f=0;f<b->records[b->layouts[i]]->as.struct_def.field_count;f++)emit_op(b->cg,OP_POP);
        b->live[i]=false;b->disposal_pending[i]=false;
    }
}
static uint8_t borrow_effect_value(BorrowCG *b,ASTNode *node,uint8_t expected,uint32_t layout) {
    if(expected==TAG_ARRAY)return borrow_array_expr(b,node,layout);
    if(expected==TAG_UNION)return borrow_union_expr(b,node,(int)layout);
    if(expected==TAG_STRUCT) {
        bool emitted=b->ordinary[layout]?borrow_ordinary(b,node,layout):borrow_owner(b,node,layout);
        b->expression_layout=layout;return emitted?TAG_STRUCT:TAG_VOID;
    }
    if(node && node->type==AST_IDENTIFIER) {
        int local=borrow_local(b,node->as.identifier);
        if(local>=0 && b->tags[local]==TAG_STRUCT && !b->ordinary[b->layouts[local]]) {
            uint32_t owner=b->layouts[local];bool emitted=borrow_owner(b,node,owner);
            b->expression_layout=owner;return emitted?TAG_STRUCT:TAG_VOID;
        }
    }
    if(node && node->type==AST_STRUCT_LITERAL) {
        int record=borrow_record(b,node->as.struct_literal.struct_name);
        if(record>=0) {
            bool emitted=b->ordinary[record]?borrow_ordinary(b,node,(unsigned)record):borrow_owner(b,node,(unsigned)record);
            b->expression_layout=(uint32_t)record;return emitted?TAG_STRUCT:TAG_VOID;
        }
    }
    return borrow_expr(b,node);
}
static uint8_t borrow_effect_block(BorrowCG *b,ASTNode *body,uint8_t expected,uint32_t layout) {
    unsigned first=b->count;uint8_t result=TAG_VOID;ASTNode *last=body;
    if(body && body->type==AST_BLOCK) {
        int count=body->as.block.count;
        bool value=count && ast_is_value_expression(body->as.block.statements[count-1]->type);
        ASTNode prefix=*body;if(value)prefix.as.block.count--;
        unsigned depth=b->control_depth;b->control_depth=0;
        borrow_body(b,&prefix,false);b->control_depth=depth;
        last=value?body->as.block.statements[count-1]:NULL;
    }
    if(!b->returned && !b->transferred && !b->cg->had_error) {
        if(last) {
            result=borrow_effect_value(b,last,expected,layout);
            /* A void call leaves no operand; perform and handle carry a void token. */
            if(result==TAG_VOID && last->type!=AST_EFFECT_OP && last->type!=AST_HANDLE_EXPR && last->type!=AST_EFFECT_HANDLER)
                emit_op(b->cg,OP_PUSH_VOID);
        } else emit_op(b->cg,OP_PUSH_VOID);
        if(borrow_consumed(b,first,body))borrow_cleanup(b,first);
    }
    borrow_names_end(b,first);for(unsigned i=first;i<b->count;i++)b->names[i]=NULL;
    return result;
}
typedef struct {
    unsigned count;
    const char *names[256];
    bool live[256],pending[256];
    int16_t variants[256];
} BorrowEffectScope;
static void borrow_effect_save(BorrowCG *b,BorrowEffectScope *scope) {
    scope->count=b->count;
    for(unsigned i=0;i<b->count;i++) {
        scope->names[i]=b->names[i];scope->live[i]=b->live[i];
        scope->pending[i]=b->disposal_pending[i];scope->variants[i]=b->variants[i];
    }
}
static void borrow_effect_restore(BorrowCG *b,const BorrowEffectScope *scope) {
    for(unsigned i=0;i<b->count;i++) {
        b->names[i]=i<scope->count?scope->names[i]:NULL;
        b->live[i]=i<scope->count?scope->live[i]:false;
        b->disposal_pending[i]=i<scope->count?scope->pending[i]:false;
        b->variants[i]=i<scope->count?scope->variants[i]:-1;
    }
}
static uint8_t borrow_effect_expr(BorrowCG *b,ASTNode *node) {
    CG *cg=b->cg;
    if(node->type==AST_EFFECT_HANDLER) {
        ASTNode normalized=*node;int count=node->as.effect_handler.handler_count;
        int counts[64];char **parameters[64];
        if(count<1 || count>64){borrow_error(b,node,"one through 64 handlers");return TAG_VOID;}
        for(int i=0;i<count;i++) {
            counts[i]=node->as.effect_handler.handler_param_names[i]?1:0;
            parameters[i]=&node->as.effect_handler.handler_param_names[i];
        }
        normalized.type=AST_HANDLE_EXPR;normalized.as.handle_expr.body=node->as.effect_handler.body;
        normalized.as.handle_expr.effect_name=node->as.effect_handler.effect_name;
        normalized.as.handle_expr.handler_count=count;
        normalized.as.handle_expr.handler_op_names=node->as.effect_handler.handler_op_names;
        normalized.as.handle_expr.handler_param_names=parameters;normalized.as.handle_expr.handler_param_counts=counts;
        normalized.as.handle_expr.handler_bodies=node->as.effect_handler.handler_bodies;
        return borrow_effect_expr(b,&normalized);
    }
    if(node->type==AST_EFFECT_OP) {
        BorrowEffect *effect=borrow_effect_register(b,node->as.effect_op.effect_name,node->as.effect_op.op_name,node);
        if(!effect)return TAG_VOID;
        NvmOwnershipEffect *signature=&effect->signature;
        if(node->as.effect_op.arg_count!=signature->parameter_count) {
            borrow_error(b,node,"the exact performed argument count");return TAG_VOID;
        }
        borrow_dispose_patterns(b);
        for(unsigned i=0;i<signature->parameter_count && !cg->had_error;i++) {
            NvmOwnershipEffectValue parameter=signature->parameters[i];
            uint8_t tag=borrow_effect_value(b,node->as.effect_op.args[i],parameter.tag,parameter.layout);
            if(tag!=parameter.tag || ((tag==TAG_STRUCT || tag==TAG_UNION || tag==TAG_ARRAY) && b->expression_layout!=parameter.layout))
                borrow_error(b,node,"the exact positional operation argument");
        }
        borrow_dispose_patterns(b);
        emit_op(cg,OP_PERFORM,signature->key_idx,signature->parameter_count);
        b->expression_layout=signature->result.layout;return signature->result.tag;
    }
    int count=node->as.handle_expr.handler_count;unsigned depth=b->handler_depth;
    if(count<1 || count>64 || depth+(unsigned)count>64) {
        borrow_error(b,node,"at most 64 active lexical handlers");return TAG_VOID;
    }
    BorrowEffectScope before,normal;borrow_effect_save(b,&before);
    BorrowEffect *effects[64];uint32_t patches[64];unsigned starts[64];
    for(int i=0;i<count && !cg->had_error;i++) {
        effects[i]=borrow_effect_register(b,node->as.handle_expr.effect_name,node->as.handle_expr.handler_op_names[i],node);
        if(!effects[i])return TAG_VOID;
        NvmOwnershipEffect *signature=&effects[i]->signature;
        if(node->as.handle_expr.handler_param_counts[i]!=signature->parameter_count) {
            borrow_error(b,node,"exact handler parameter bindings");return TAG_VOID;
        }
        starts[i]=b->count;
        for(unsigned p=0;p<signature->parameter_count;p++) {
            NvmOwnershipEffectValue value=signature->parameters[p];
            int slot=borrow_slot(b,NULL,value.tag,value.layout,0,false);b->live[slot]=false;
        }
        patches[i]=emit_op(cg,OP_HANDLER_PUSH,signature->key_idx,(int32_t)0,starts[i],signature->parameter_count);
    }
    if(cg->had_error)return TAG_VOID;
    b->handler_depth=depth+(unsigned)count;
    uint8_t result=borrow_effect_block(b,node->as.handle_expr.body,TAG_VOID,NVM_V2_NO_INDEX);
    uint32_t result_layout=b->expression_layout;bool returned=b->returned,transferred=b->transferred;
    uint32_t skip=UINT32_MAX;
    if(!returned && !transferred) {
        emit_op(cg,OP_HANDLER_POP,count);skip=emit_op(cg,OP_JMP,(int32_t)0);
    }
    borrow_effect_save(b,&normal);BorrowLoop *loop=b->loop;
    for(int i=0;i<count && !cg->had_error;i++) {
        patch_jump(cg,patches[i]+5,patches[i],cg->code_size);
        borrow_effect_restore(b,&before);b->returned=false;b->transferred=false;b->loop=NULL;b->handler_depth=depth+(unsigned)i+1;
        for(unsigned j=0;j<before.count;j++)if(borrow_owned_type(b,b->tags[j],b->layouts[j]) || b->modes[j]) {
            b->live[j]=false;b->disposal_pending[j]=false;
        }
        NvmOwnershipEffect *signature=&effects[i]->signature;
        for(unsigned p=0;p<signature->parameter_count;p++) {
            unsigned slot=starts[i]+p;b->names[slot]=node->as.handle_expr.handler_param_names[i][p];b->live[slot]=true;
            borrow_name_begin(b,slot,b->names[slot]);
        }
        uint8_t tag=borrow_effect_block(b,node->as.handle_expr.handler_bodies[i],signature->result.tag,signature->result.layout);
        if(!b->returned && !b->transferred) {
            if(tag!=signature->result.tag || ((tag==TAG_STRUCT || tag==TAG_UNION || tag==TAG_ARRAY) && b->expression_layout!=signature->result.layout))
                borrow_error(b,node,"the exact resumed operation result");
            if(borrow_consumed(b,0,node))borrow_cleanup(b,0);
            emit_op(cg,OP_EFFECT_RESUME);
        }
        borrow_names_end(b,starts[i]);
    }
    b->loop=loop;b->handler_depth=depth;
    borrow_effect_restore(b,&normal);b->returned=returned;b->transferred=transferred;b->expression_layout=result_layout;
    if(skip!=UINT32_MAX)patch_jump(cg,skip+1,skip,cg->code_size);
    return result;
}
