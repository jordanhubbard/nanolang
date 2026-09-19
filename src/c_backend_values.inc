/* I insert scoped statements at supported value-consuming statement boundaries. */
typedef struct { Type type; ASTNode *owner; bool continues; } CBValueInfo;

static bool cb_needs_lift(ASTNode *n) {
    if (!n) return false;
    switch (n->type) {
    case AST_BLOCK: case AST_MATCH: case AST_IF: case AST_COND:
    case AST_RETURN: case AST_BREAK: case AST_CONTINUE: return true;
    case AST_PREFIX_OP:
        for (int i=0;i<n->as.prefix_op.arg_count;i++)
            if (cb_needs_lift(n->as.prefix_op.args[i])) return true;
        return false;
    case AST_CALL:
        if (cb_needs_lift(n->as.call.func_expr)) return true;
        for (int i=0;i<n->as.call.arg_count;i++)
            if (cb_needs_lift(n->as.call.args[i])) return true;
        return false;
    case AST_FIELD_ACCESS: return cb_needs_lift(n->as.field_access.object);
    case AST_UNION_CONSTRUCT:
        for (int i=0;i<n->as.union_construct.field_count;i++)
            if (cb_needs_lift(n->as.union_construct.field_values[i])) return true;
        return false;
    case AST_STRUCT_LITERAL:
        if (cb_needs_lift(n->as.struct_literal.spread_source)) return true;
        for (int i=0;i<n->as.struct_literal.field_count;i++)
            if (cb_needs_lift(n->as.struct_literal.field_values[i])) return true;
        return false;
    default: return false;
    }
}

static bool cb_scalar(Type t) {
    return t==TYPE_INT || t==TYPE_BOOL || t==TYPE_FLOAT || t==TYPE_STRING;
}
static CBValueInfo cb_info(CBCtx *c, ASTNode *n);
static bool cb_statement_exits(CBCtx *c, ASTNode *n) {
    if (!n) return false;
    if (n->type==AST_RETURN || n->type==AST_BREAK || n->type==AST_CONTINUE) return true;
    if (n->type==AST_BLOCK) {
        bool exits=false;ctx_push_scope(c);
        for(int i=0;i<n->as.block.count;i++) {
            ASTNode *s=n->as.block.statements[i];
            if(cb_statement_exits(c,s)){exits=true;break;}
            if(s && s->type==AST_LET)ctx_add_nominal(c,s->as.let.name,s->as.let.var_type,s->as.let.type_name,NULL);
        }
        ctx_pop_scope(c);return exits;
    }
    ASTNode *value=NULL;
    switch(n->type) {
    case AST_LET:value=n->as.let.value;break;
    case AST_SET:value=n->as.set.value;break;
    case AST_ASSERT:value=n->as.assert.condition;break;
    case AST_PRINT:value=n->as.print.expr;break;
    case AST_IF:value=n->as.if_stmt.condition;break;
    case AST_CALL:case AST_PREFIX_OP:case AST_COND:value=n;break;
    default:break;
    }
    if(cb_needs_lift(value) && !cb_info(c,value).continues)return true;
    if(n->type==AST_IF)return n->as.if_stmt.else_branch &&
        cb_statement_exits(c,n->as.if_stmt.then_branch) && cb_statement_exits(c,n->as.if_stmt.else_branch);
    return false;
}
static CBValueInfo cb_join(CBCtx *c, CBValueInfo a, CBValueInfo b) {
    if(!a.continues) return b;
    if(!b.continues) return a;
    if(a.type!=b.type || a.owner!=b.owner) {
        ctx_error(c,"I require identical continuing C value types and nominal owners.");
        return (CBValueInfo){TYPE_UNKNOWN,NULL,true};
    }
    return a;
}
static void cb_payload_scope(CBCtx *c, ASTNode *n, int arm, ASTNode *owner) {
    const char *variant=n->as.match_expr.pattern_variants[arm];
    const char *binding=n->as.match_expr.pattern_bindings ? n->as.match_expr.pattern_bindings[arm] : NULL;
    int declaration,index;
    if(binding && strcmp(binding,"_") && strcmp(variant,"_") &&
       ctx_union_variant(c,owner->as.union_def.name,variant,&declaration,&index)==owner &&
       owner->as.union_def.variant_field_counts[index]>0)
        ctx_add_nominal(c,binding,TYPE_STRUCT,owner->as.union_def.name,variant);
}
static CBValueInfo cb_info(CBCtx *c, ASTNode *n) {
    CBValueInfo no={TYPE_VOID,NULL,false}, bad={TYPE_UNKNOWN,NULL,true};
    if(!n) {ctx_error(c,"I require a value on every continuing C path.");return bad;}
    if(cb_record_spread(c,n)) return bad;
    if(n->type==AST_RETURN || n->type==AST_BREAK || n->type==AST_CONTINUE) return no;
    if(n->type==AST_BLOCK) {
        ctx_push_scope(c); CBValueInfo result=bad;
        if(!n->as.block.count) ctx_error(c,"I require a final expression in a C value block.");
        for(int i=0;i<n->as.block.count;i++) {
            ASTNode *s=n->as.block.statements[i];
            if(i==n->as.block.count-1) {result=cb_info(c,s);break;}
            if(cb_statement_exits(c,s)) {result=no;break;}
            if(s && s->type==AST_LET)
                ctx_add_nominal(c,s->as.let.name,s->as.let.var_type,s->as.let.type_name,NULL);
        }
        ctx_pop_scope(c);return result;
    }
    if(n->type==AST_MATCH) {
        if(!cb_info(c,n->as.match_expr.expr).continues)return no;
        int declaration;ASTNode *owner=ctx_guarded_union_profile(c,n,&declaration);
        if(!owner)return bad;
        CBValueInfo result=no;
        for(int i=0;i<n->as.match_expr.arm_count;i++) {
            ctx_push_scope(c);cb_payload_scope(c,n,i,owner);
            result=cb_join(c,result,cb_info(c,n->as.match_expr.arm_bodies[i]));
            ctx_pop_scope(c);
        }
        return result;
    }
    if(n->type==AST_IF) {
        if(!cb_info(c,n->as.if_stmt.condition).continues)return no;
        return cb_join(c,cb_info(c,n->as.if_stmt.then_branch),cb_info(c,n->as.if_stmt.else_branch));
    }
    if(n->type==AST_COND) {
        CBValueInfo result=no;
        for(int i=0;i<n->as.cond_expr.clause_count;i++) {
            if(!cb_info(c,n->as.cond_expr.conditions[i]).continues)return result;
            result=cb_join(c,result,cb_info(c,n->as.cond_expr.values[i]));
        }
        return cb_join(c,result,cb_info(c,n->as.cond_expr.else_value));
    }
    if(n->type==AST_PREFIX_OP && cb_needs_lift(n)) {
        int count=n->as.prefix_op.arg_count;
        if(count<1 || count>2) {ctx_error(c,"I require a bounded scalar C value operator.");return bad;}
        CBValueInfo a=cb_info(c,n->as.prefix_op.args[0]);
        if(!a.continues)return no;
        TokenType op=n->as.prefix_op.op;
        CBValueInfo b=count==2?cb_info(c,n->as.prefix_op.args[1]):a;
        if(!b.continues && op!=TOKEN_AND && op!=TOKEN_OR)return no;
        if(op==TOKEN_AND || op==TOKEN_OR || op==TOKEN_NOT) {
            if(a.type!=TYPE_BOOL || (b.continues && b.type!=TYPE_BOOL))ctx_error(c,"I require exact BOOL lifted logical operands.");
            return (CBValueInfo){TYPE_BOOL,NULL,true};
        }
        if(op==TOKEN_EQ || op==TOKEN_NE || op==TOKEN_LT || op==TOKEN_LE || op==TOKEN_GT || op==TOKEN_GE)
            return (CBValueInfo){TYPE_BOOL,NULL,true};
        if(a.type==TYPE_STRING && b.type==TYPE_STRING && op==TOKEN_PLUS)
            return (CBValueInfo){TYPE_STRING,NULL,true};
        if((a.type==TYPE_INT || a.type==TYPE_FLOAT) && (b.type==TYPE_INT || b.type==TYPE_FLOAT))
            return (CBValueInfo){a.type==TYPE_FLOAT || b.type==TYPE_FLOAT?TYPE_FLOAT:TYPE_INT,NULL,true};
        ctx_error(c,"I require resolved scalar lifted operator results.");return bad;
    }
    if(n->type==AST_CALL || n->type==AST_UNION_CONSTRUCT || n->type==AST_STRUCT_LITERAL) {
        ASTNode **values=NULL;int count=0;
        if(n->type==AST_CALL){values=n->as.call.args;count=n->as.call.arg_count;}
        else if(n->type==AST_UNION_CONSTRUCT){values=n->as.union_construct.field_values;count=n->as.union_construct.field_count;}
        else {values=n->as.struct_literal.field_values;count=n->as.struct_literal.field_count;}
        for(int i=0;i<count;i++)if(cb_needs_lift(values[i]) && !cb_info(c,values[i]).continues)return no;
    }
    if(n->type==AST_UNION_CONSTRUCT)
        return (CBValueInfo){TYPE_UNION,ctx_result_union(c,n->as.union_construct.union_name),true};
    if(n->type==AST_STRUCT_LITERAL && n->as.struct_literal.struct_name) {
        const char *name=n->as.struct_literal.struct_name;
        ASTNode **items=c->root->type==AST_PROGRAM?c->root->as.program.items:&c->root;
        int count=c->root->type==AST_PROGRAM?c->root->as.program.count:1;
        for(int i=0;i<count;i++) {
            ASTNode *d=items[i];if(!d || d->type!=AST_UNION_DEF)continue;
            size_t length=strlen(d->as.union_def.name);
            if(!strncmp(name,d->as.union_def.name,length) && name[length]=='.')
                return (CBValueInfo){TYPE_UNION,ctx_result_union(c,d->as.union_def.name),true};
        }
    }
    if(n->type==AST_CALL && cb_needs_lift(n) && n->as.call.name && !n->as.call.func_expr &&
       !ctx_has_binding(c,n->as.call.name) && !ctx_function(c,n->as.call.name) && !n->as.call.checked_signature) {
        const char *name=n->as.call.name;Type result=TYPE_UNKNOWN;
        if(!strcmp(name,"float_from_bits"))result=TYPE_FLOAT;
        else if(!strcmp(name,"float_to_bits") || !strcmp(name,"str_length"))result=TYPE_INT;
        else if(!strcmp(name,"float_to_string") || !strcmp(name,"int_to_string") || !strcmp(name,"str_concat"))result=TYPE_STRING;
        else if(!strcmp(name,"print") || !strcmp(name,"println"))result=TYPE_VOID;
        if(result!=TYPE_UNKNOWN)return (CBValueInfo){result,NULL,true};
    }
    Type t=infer_expr_type(c,n);ASTNode *owner=NULL;
    if(t==TYPE_UNION)owner=ctx_union_value(c,n);
    if(!cb_scalar(t) && t!=TYPE_UNION && t!=TYPE_VOID)ctx_error(c,"I require an exact supported C lifted result type.");
    return (CBValueInfo){t,owner,true};
}

static void cb_free_temps(CBCtx *c, CBLiftTemp *mark) {
    while(c->lift_temps!=mark) {CBLiftTemp *t=c->lift_temps;c->lift_temps=t->next;free(t);}
}
static CBLiftTemp *cb_new_temp(CBCtx *c, CBValueInfo info) {
    if(!cb_scalar(info.type) && !(info.type==TYPE_UNION && info.owner)) {
        ctx_error(c,"I require a concrete C result-slot type.");return NULL;
    }
    if(c->lift_counter==SIZE_MAX){ctx_error(c,"I exceeded my C value-slot capacity.");return NULL;}
    CBLiftTemp *t=calloc(1,sizeof *t);
    if(!t){ctx_error(c,"I could not allocate a C value plan.");return NULL;}
    snprintf(t->name,sizeof t->name,"%svalue_%zu",c->prefix,c->lift_counter++);
    t->type=info.type;t->owner=info.owner;t->reference.type=AST_IDENTIFIER;t->reference.as.identifier=t->name;
    t->next=c->lift_temps;c->lift_temps=t;
    emit_indent(c);
    if(t->owner)fprintf(c->out,"NanoUnion_%s",t->owner->as.union_def.name);
    else fputs(c_type(c, t->type),c->out);
    fprintf(c->out," %s;\n",t->name);return t;
}
static int cb_value(CBCtx *c, ASTNode *n, CBLiftTemp *destination, bool *continues);
static CBLiftTemp *cb_materialize(CBCtx *c, ASTNode *n, bool *continues) {
    CBValueInfo info=cb_info(c,n);if(c->error)return NULL;
    if(!info.continues) {cb_value(c,n,NULL,continues);return NULL;}
    CBLiftTemp *t=cb_new_temp(c,info);if(!t)return NULL;
    if(cb_value(c,n,t,continues))return NULL;
    return t;
}
static int cb_assign(CBCtx *c, ASTNode *n, CBLiftTemp *destination) {
    if(destination) {
        CBValueInfo info=cb_info(c,n);
        if(info.type!=destination->type || info.owner!=destination->owner) {
            ctx_error(c,"I require an exact C result-slot assignment.");return -1;
        }
        if(info.type==TYPE_UNION && !ctx_require_union_value(c,n,info.owner))return -1;
        emit_indent(c);fprintf(c->out,"%s = ",destination->name);
    } else {emit_indent(c);fputs("(void)(",c->out);}
    if(emit_expr(c,n))return -1;
    fputs(destination?";\n":");\n",c->out);return 0;
}

static int cb_value(CBCtx *c, ASTNode *n, CBLiftTemp *destination, bool *continues) {
    *continues=true;
    if(!n){ctx_error(c,"I require a continuing C value expression.");return -1;}
    if(n->type==AST_RETURN || n->type==AST_BREAK || n->type==AST_CONTINUE) {
        emit_indent(c);*continues=false;return emit_stmt(c,n);
    }
    if(n->type==AST_BLOCK) {
        int rc=0;emit_indent(c);fputs("{\n",c->out);c->indent++;ctx_push_scope(c);
        if(!n->as.block.count){ctx_error(c,"I require a final expression in a C value block.");rc=-1;}
        for(int i=0;!rc && i<n->as.block.count;i++) {
            ASTNode *s=n->as.block.statements[i];
            if(i==n->as.block.count-1){rc=cb_value(c,s,destination,continues);break;}
            bool exits=cb_statement_exits(c,s);
            emit_indent(c);rc=emit_stmt(c,s);
            if(exits){*continues=false;break;}
        }
        ctx_pop_scope(c);c->indent--;emit_indent(c);fputs("}\n",c->out);return rc;
    }
    if(n->type==AST_COND) {
        if(!n->as.cond_expr.clause_count)return cb_value(c,n->as.cond_expr.else_value,destination,continues);
        ASTNode rest=*n;rest.as.cond_expr.clause_count--;
        rest.as.cond_expr.conditions++;rest.as.cond_expr.values++;
        ASTNode branch={.type=AST_IF};branch.as.if_stmt.condition=n->as.cond_expr.conditions[0];
        branch.as.if_stmt.then_branch=n->as.cond_expr.values[0];branch.as.if_stmt.else_branch=&rest;
        return cb_value(c,&branch,destination,continues);
    }
    if(n->type==AST_IF) {
        bool live;CBLiftTemp *condition=cb_materialize(c,n->as.if_stmt.condition,&live);
        if(c->error)return -1;
        if(!live){*continues=false;return 0;}
        if(!condition || condition->type!=TYPE_BOOL){ctx_error(c,"I require an exact BOOL C value condition.");return -1;}
        emit_indent(c);fprintf(c->out,"if (%s) {\n",condition->name);c->indent++;ctx_push_scope(c);
        bool yes=false,no=false;int rc=cb_value(c,n->as.if_stmt.then_branch,destination,&yes);
        ctx_pop_scope(c);c->indent--;emit_indent(c);fputs("} else {\n",c->out);c->indent++;ctx_push_scope(c);
        if(!rc)rc=cb_value(c,n->as.if_stmt.else_branch,destination,&no);
        ctx_pop_scope(c);c->indent--;emit_indent(c);fputs("}\n",c->out);
        *continues=yes || no;return rc;
    }
    if(n->type==AST_MATCH) {
        int declaration;ASTNode *owner=ctx_guarded_union_profile(c,n,&declaration);
        if(!owner)return -1;
        bool live;CBLiftTemp *scrutinee=cb_materialize(c,n->as.match_expr.expr,&live);
        if(c->error)return -1;
        if(!live){*continues=false;return 0;}
        if(!scrutinee || scrutinee->owner!=owner){ctx_error(c,"I require the exact lifted C match owner.");return -1;}
        if(c->match_labels==SIZE_MAX){ctx_error(c,"I exceeded my C match label capacity.");return -1;}
        size_t label=c->match_labels++;bool any=false;int rc=0;
        for(int i=0;!rc && i<n->as.match_expr.arm_count;i++) {
            const char *variant=n->as.match_expr.pattern_variants[i];bool wildcard=!strcmp(variant,"_");
            int index=0,di=0;
            if(!wildcard && ctx_union_variant(c,owner->as.union_def.name,variant,&di,&index)!=owner){ctx_error(c,"I require an exact lifted match variant.");return -1;}
            emit_indent(c);
            if(wildcard)fputs("{\n",c->out);
            else fprintf(c->out,"if (%s.tag == NanoUnion_%s_TAG_%s) {\n",scrutinee->name,owner->as.union_def.name,variant);
            c->indent++;ctx_push_scope(c);cb_payload_scope(c,n,i,owner);
            const char *binding=n->as.match_expr.pattern_bindings?n->as.match_expr.pattern_bindings[i]:NULL;
            if(!wildcard && binding && strcmp(binding,"_") && owner->as.union_def.variant_field_counts[index]>0) {
                emit_indent(c);fprintf(c->out,"%spayload_%d_%d %s = %s.as.%s;\n",c->prefix,declaration,index,binding,scrutinee->name,variant);
            }
            ASTNode *guard=n->as.match_expr.guard_exprs?n->as.match_expr.guard_exprs[i]:NULL;
            bool guard_live=true,opened=false;
            if(guard) {
                CBLiftTemp *test=cb_materialize(c,guard,&guard_live);
                if(c->error)rc=-1;
                else if(guard_live) {
                    if(!test || test->type!=TYPE_BOOL){ctx_error(c,"I require an exact BOOL lifted match guard.");rc=-1;}
                    else {emit_indent(c);fprintf(c->out,"if (%s) {\n",test->name);c->indent++;opened=true;}
                }
            }
            bool arm_live=false;
            if(!rc && guard_live)rc=cb_value(c,n->as.match_expr.arm_bodies[i],destination,&arm_live);
            if(!rc && arm_live) {emit_indent(c);fprintf(c->out,"goto %smatch_end_%zu;\n",c->prefix,label);any=true;}
            if(opened){c->indent--;emit_indent(c);fputs("}\n",c->out);}
            ctx_pop_scope(c);c->indent--;emit_indent(c);fputs("}\n",c->out);
        }
        if(rc)return rc;
        emit_indent(c);fputs("fputs(\"I require a covered C union tag for match.\\n\", stderr); exit(EXIT_FAILURE);\n",c->out);
        if(any){emit_indent(c);fprintf(c->out,"%smatch_end_%zu: ;\n",c->prefix,label);}
        *continues=any;return 0;
    }
    if(!cb_needs_lift(n) && n->type!=AST_PREFIX_OP && n->type!=AST_CALL &&
       n->type!=AST_UNION_CONSTRUCT && n->type!=AST_STRUCT_LITERAL)
        return cb_assign(c,n,destination);
    if(n->type==AST_PREFIX_OP && (n->as.prefix_op.op==TOKEN_AND || n->as.prefix_op.op==TOKEN_OR)) {
        if(n->as.prefix_op.arg_count!=2){ctx_error(c,"I require two lifted logical operands.");return -1;}
        bool live;CBLiftTemp *left=cb_materialize(c,n->as.prefix_op.args[0],&live);
        if(c->error)return -1;
        if(!live){*continues=false;return 0;}
        if(!left || left->type!=TYPE_BOOL){ctx_error(c,"I require exact BOOL lifted logical operands.");return -1;}
        bool conjunction=n->as.prefix_op.op==TOKEN_AND;
        emit_indent(c);fprintf(c->out,"if (%s%s) {\n",conjunction?"":"!",left->name);c->indent++;
        bool right_live;int rc=cb_value(c,n->as.prefix_op.args[1],destination,&right_live);
        c->indent--;emit_indent(c);fputs("} else {\n",c->out);c->indent++;
        ASTNode constant={.type=AST_BOOL};constant.as.bool_val=!conjunction;
        if(!rc)rc=cb_assign(c,&constant,destination);
        c->indent--;emit_indent(c);fputs("}\n",c->out);return rc;
    }
    ASTNode view=*n;ASTNode **children=NULL;int count=0;
    if(n->type==AST_PREFIX_OP){children=n->as.prefix_op.args;count=n->as.prefix_op.arg_count;}
    else if(n->type==AST_CALL){
        if(n->as.call.func_expr || !n->as.call.name || ctx_has_binding(c,n->as.call.name)) {
            ctx_error(c,"I require a direct declared or builtin lifted C call.");return -1;
        }
        children=n->as.call.args;count=n->as.call.arg_count;
    } else if(n->type==AST_UNION_CONSTRUCT){children=n->as.union_construct.field_values;count=n->as.union_construct.field_count;}
    else if(n->type==AST_STRUCT_LITERAL){children=n->as.struct_literal.field_values;count=n->as.struct_literal.field_count;}
    else {ctx_error(c,"I do not support this compound lifted C value context.");return -1;}
    if(count<0 || (size_t)count>SIZE_MAX/sizeof(ASTNode *)){ctx_error(c,"I exceeded my lifted C operand capacity.");return -1;}
    ASTNode **args=calloc(count? (size_t)count:1,sizeof *args);
    if(!args){ctx_error(c,"I could not allocate lifted C operands.");return -1;}
    int rc=0;
    for(int i=0;i<count;i++) {
        bool live;CBLiftTemp *t=cb_materialize(c,children[i],&live);
        if(c->error){rc=-1;break;}
        if(!live){*continues=false;break;}
        if(!t){ctx_error(c,"I require a value-producing lifted C operand.");rc=-1;break;}
        args[i]=&t->reference;
    }
    if(!rc && *continues) {
        if(n->type==AST_PREFIX_OP)view.as.prefix_op.args=args;
        else if(n->type==AST_CALL)view.as.call.args=args;
        else if(n->type==AST_UNION_CONSTRUCT)view.as.union_construct.field_values=args;
        else view.as.struct_literal.field_values=args;
        rc=cb_assign(c,&view,destination);
    }
    free(args);return rc;
}

static int cb_lift_statement(CBCtx *c, ASTNode *n, bool *handled) {
    ASTNode *value=NULL;ASTNode view=*n;
    switch(n->type) {
    case AST_LET:value=n->as.let.value;break;
    case AST_SET:value=n->as.set.value;break;
    case AST_RETURN:value=n->as.return_stmt.value;break;
    case AST_ASSERT:value=n->as.assert.condition;break;
    case AST_PRINT:value=n->as.print.expr;break;
    case AST_IF:value=n->as.if_stmt.condition;break;
    case AST_CALL:case AST_PREFIX_OP:case AST_COND:value=n;break;
    default:return 0;
    }
    if(!cb_needs_lift(value))return 0;
    *handled=true;CBLiftTemp *mark=c->lift_temps;bool live=true;int rc=0;
    CBValueInfo info=cb_info(c,value);
    bool discard=value==n;
    if(c->error)rc=-1;
    else if(discard)rc=cb_value(c,value,NULL,&live);
    else {
        CBLiftTemp *t=cb_materialize(c,value,&live);
        if(c->error)rc=-1;
        else if(live && t) {
            if(n->type==AST_LET)view.as.let.value=&t->reference;
            else if(n->type==AST_SET)view.as.set.value=&t->reference;
            else if(n->type==AST_RETURN)view.as.return_stmt.value=&t->reference;
            else if(n->type==AST_ASSERT)view.as.assert.condition=&t->reference;
            else if(n->type==AST_IF)view.as.if_stmt.condition=&t->reference;
            else view.as.print.expr=&t->reference;
            emit_indent(c);rc=emit_stmt(c,&view);
        } else if(live){ctx_error(c,"I require a continuing typed C value.");rc=-1;}
    }
    (void)info;
    cb_free_temps(c,mark);return rc;
}
