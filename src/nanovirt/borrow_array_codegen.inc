/* I preserve exact array identities while executable collection cleanup is gated. */
static uint32_t borrow_array_intern(BorrowCG *b,NvmOwnershipEffectValue element,ASTNode *at) {
    for(unsigned i=0;i<b->array_types.count;i++)
        if(b->array_types.items[i].tag==element.tag && b->array_types.items[i].layout==element.layout)return i;
    if(b->array_types.count==NVM_OWNERSHIP_MAX_EFFECT_TYPES) {
        borrow_error(b,at,"bounded exact array element declarations");return NVM_V2_NO_INDEX;
    }
    unsigned index=b->array_types.count++;b->array_types.items[index]=element;return index;
}
static uint32_t borrow_array_type_depth(BorrowCG *b,const TypeInfo *info,ASTNode *at,unsigned depth) {
    if(!info || info->base_type!=TYPE_ARRAY || !info->element_type || depth>=32) {
        borrow_error(b,at,"a complete bounded array element annotation");return NVM_V2_NO_INDEX;
    }
    const TypeInfo *child=info->element_type;
    NvmOwnershipEffectValue element={TAG_VOID,NVM_V2_NO_INDEX};
    if(child->base_type==TYPE_ARRAY) {
        element.tag=TAG_ARRAY;element.layout=borrow_array_type_depth(b,child,at,depth+1);
    } else {
        borrow_register_named_union(b,child,child->generic_name,at);
        int instance=borrow_named_union_instance(b,child,child->generic_name);
        int record=borrow_record(b,child->generic_name);
        if(instance>=0) {
            element.tag=TAG_UNION;element.layout=(uint32_t)instance;b->unions[instance].complete=true;
        } else if(record>=0 && child->base_type==TYPE_STRUCT) {
            element.tag=TAG_STRUCT;element.layout=(uint32_t)record;
        } else element.tag=borrow_local_tag(child->base_type);
    }
    if(!element.tag)borrow_error(b,at,"an exact scalar, STRING, nominal or nested array element");
    return b->cg->had_error?NVM_V2_NO_INDEX:borrow_array_intern(b,element,at);
}
static uint32_t borrow_array_type(BorrowCG *b,const TypeInfo *info,ASTNode *at) {
    return borrow_array_type_depth(b,info,at,0);
}
static bool borrow_array_value_matches(BorrowCG *b,uint8_t tag,NvmOwnershipEffectValue expected) {
    return tag==expected.tag && ((tag!=TAG_ARRAY && tag!=TAG_STRUCT && tag!=TAG_UNION) || b->expression_layout==expected.layout);
}
static uint8_t borrow_array_expr(BorrowCG *b,ASTNode *n,uint32_t expected) {
    if(!n)return TAG_VOID;
    if(n->type!=AST_ARRAY_LITERAL) {
        uint8_t tag=n->type==AST_MATCH?borrow_match_expr(b,n,TAG_ARRAY,expected):borrow_expr(b,n);
        if(tag!=TAG_ARRAY || b->expression_layout>=b->array_types.count ||
            (expected!=NVM_V2_NO_INDEX && expected!=b->expression_layout)) {
            borrow_error(b,n,"the exact declared array value");return TAG_VOID;
        }
        return TAG_ARRAY;
    }
    int count=n->as.array_literal.element_count;
    if(count<0 || count>UINT16_MAX || (expected!=NVM_V2_NO_INDEX && expected>=b->array_types.count)) {
        borrow_error(b,n,"bounded exact array literal elements");return TAG_VOID;
    }
    NvmOwnershipEffectValue element={TAG_VOID,NVM_V2_NO_INDEX};
    if(expected!=NVM_V2_NO_INDEX)element=b->array_types.items[expected];
    if(!count && !element.tag) {
        element.tag=borrow_local_tag(n->as.array_literal.element_type);
        if(!element.tag){borrow_error(b,n,"an exact contextual empty array type");return TAG_VOID;}
    }
    for(int i=0;i<count && !b->cg->had_error;i++) {
        uint8_t tag=borrow_effect_value(b,n->as.array_literal.elements[i],element.tag,element.layout);
        if(!element.tag) {
            element.tag=tag;element.layout=tag==TAG_ARRAY || tag==TAG_STRUCT || tag==TAG_UNION?b->expression_layout:NVM_V2_NO_INDEX;
        }
        if(!tag || !borrow_array_value_matches(b,tag,element))borrow_error(b,n,"matching exact array literal elements");
    }
    if(b->cg->had_error)return TAG_VOID;
    uint32_t type=borrow_array_intern(b,element,n);
    if(!count) {
        if(b->array_site_count==NVM_OWNERSHIP_MAX_ARRAY_CONSTRUCTORS) {
            borrow_error(b,n,"bounded empty array constructor sites");return TAG_VOID;
        }
        unsigned site=b->array_site_count++;
        b->array_sites[site].function=b->current_function;b->array_sites[site].pc=b->cg->code_size;b->array_sites[site].element=type;
    }
    emit_op(b->cg,OP_ARR_LITERAL,element.tag,count);b->expression_layout=type;return TAG_ARRAY;
}
static bool borrow_array_builtin(BorrowCG *b,ASTNode *n,uint8_t *out) {
    const char *name=n->as.call.name;
    if(!name || n->as.call.func_expr || n->as.call.borrow_mode || borrow_function_index(b,name)>=0 || borrow_value_bound(b,name))return false;
    uint8_t op=0;unsigned count=0;
    if(!strcmp(name,"at")){op=OP_ARR_GET;count=2;}
    else if(!strcmp(name,"array_length")){op=OP_ARR_LEN;count=1;}
    else if(!strcmp(name,"array_push")){op=OP_ARR_PUSH;count=2;}
    else if(!strcmp(name,"array_set")){op=OP_ARR_SET;count=3;}
    else if(!strcmp(name,"array_pop")){op=OP_ARR_POP;count=1;}
    else if(!strcmp(name,"array_remove_at")){op=OP_ARR_REMOVE;count=2;}
    else return false;
    *out=TAG_VOID;
    if(n->as.call.arg_count!=(int)count){borrow_error(b,n,"the exact array builtin arity");return true;}
    ASTNode *receiver=n->as.call.args[0];uint32_t type;
    int local=receiver->type==AST_IDENTIFIER?borrow_local(b,receiver->as.identifier):-1;
    if(op==OP_ARR_LEN && local>=0 && b->tags[local]==TAG_ARRAY && borrow_owned_type(b,TAG_ARRAY,b->layouts[local])) {
        if(!b->live[local]){borrow_error(b,n,"a live array owner for length");return true;}
        type=b->layouts[local];emit_op(b->cg,OP_LOAD_LOCAL,local);
    } else {
        if(borrow_array_expr(b,receiver,NVM_V2_NO_INDEX)!=TAG_ARRAY)return true;
        type=b->expression_layout;
    }
    NvmOwnershipEffectValue element=b->array_types.items[type];
    if((op==OP_ARR_GET || op==OP_ARR_REMOVE || op==OP_ARR_SET) && borrow_expr(b,n->as.call.args[1])!=TAG_INT) {
        borrow_error(b,n,"an exact integer array index");return true;
    }
    if(op==OP_ARR_PUSH || op==OP_ARR_SET) {
        uint8_t tag=borrow_effect_value(b,n->as.call.args[count-1],element.tag,element.layout);
        if(!borrow_array_value_matches(b,tag,element)){borrow_error(b,n,"the exact array mutation element");return true;}
    }
    if(b->cg->had_error)return true;
    emit_op(b->cg,op);b->expression_layout=type;*out=TAG_ARRAY;
    if(op==OP_ARR_LEN){*out=TAG_INT;b->expression_layout=NVM_V2_NO_INDEX;}
    if(op==OP_ARR_GET || op==OP_ARR_POP){*out=element.tag;b->expression_layout=element.layout;}
    if(op==OP_ARR_SET){emit_op(b->cg,OP_POP);*out=TAG_VOID;b->expression_layout=NVM_V2_NO_INDEX;}
    return true;
}
