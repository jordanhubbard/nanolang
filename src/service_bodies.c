#include "service_bodies.h"
#include <stdlib.h>
#include <string.h>

#define SB_LOCALS 1024u
#define SB_FACTS 16384u
#define SB_DEPTH 128u
typedef struct { TypeInfo type; unsigned borrow; bool returns; } BodyValue;
typedef struct { const char *name; BodyValue value; bool mutable; } BodyLocal;
typedef struct {
    const NlServiceNamespace *space;
    NlServiceBodyCheck *out;
    const char *source;
    BodyValue result;
    BodyLocal locals[SB_LOCALS];
    size_t count;
    unsigned depth;
    bool pure;
} BodyCheck;
static BodyValue value(Type type) { return (BodyValue){.type = {.base_type = type}}; }
static BodyValue fail(BodyCheck *c, const ASTNode *node, unsigned status, const char *why) {
    if (!c->out->status || (c->out->status == 2 && status != 2)) {
        c->out->status = status; c->out->diagnostic = why;
        c->out->line = node ? node->line : 0; c->out->column = node ? node->column : 0;
    }
    return value(TYPE_UNKNOWN);
}
static bool equal(BodyValue a, BodyValue b) {
    return a.type.base_type != TYPE_UNKNOWN && b.type.base_type != TYPE_UNKNOWN &&
        a.borrow == b.borrow && type_infos_equal(&a.type, &b.type);
}
static BodyValue annotation(BodyCheck *c, const ASTNode *node, Type type, const TypeInfo *info) {
    BodyValue v = value(type);
    if (type == TYPE_BORROW_SHARED || type == TYPE_BORROW_MUT) {
        if (!info || !info->element_type) return fail(c,node,1,"I require a retained borrow referent.");
        v = annotation(c,node,info->element_type->base_type,info->element_type);
        v.borrow = type == TYPE_BORROW_MUT ? 2 : 1;
        return v;
    }
    if (info && info->service_declaration) {
        v.type.service_declaration = info->service_declaration;
        v.type.service_module = info->service_module;
        v.type.service_ordinal = info->service_ordinal;
        v.type.service_category = info->service_category;
        return v;
    }
    if (type == TYPE_INT || type == TYPE_BOOL || type == TYPE_VOID || type == TYPE_UNKNOWN) return v;
    return fail(c,node,2,"I have not checked this ordinary type in my service body path.");
}
static int local(BodyCheck *c, const char *name) {
    for (size_t i=c->count; i>0; --i) if (!strcmp(c->locals[i-1].name,name)) return (int)i-1;
    return -1;
}
static bool bind(BodyCheck *c, const ASTNode *node, const char *name, BodyValue v, bool mutable) {
    if (c->count == SB_LOCALS) { fail(c,node,3,"I exceeded my checked service local bound."); return false; }
    c->locals[c->count++] = (BodyLocal){name,v,mutable}; return true;
}
static BodyValue remember(BodyCheck *c, const ASTNode *node, BodyValue v, uint32_t declaration) {
    if (v.type.base_type == TYPE_UNKNOWN) return v;
    if (c->out->count == SB_FACTS) return fail(c,node,3,"I exceeded my checked service fact bound.");
    c->out->facts[c->out->count++] = (NlServiceBodyFact){node,v.type,declaration,v.borrow};
    return v;
}
static BodyValue expression(BodyCheck *, const ASTNode *);
static BodyValue require(BodyCheck *c, const ASTNode *node, BodyValue actual, BodyValue expected) {
    if (actual.type.base_type != TYPE_UNKNOWN && expected.type.base_type != TYPE_UNKNOWN && !equal(actual,expected))
        return fail(c,node,1,"I require the exact nominal type and borrow mode.");
    return actual;
}
static BodyValue call(BodyCheck *c, const ASTNode *node, const char *name, ASTNode **args, int count) {
    if (!name || local(c,name) >= 0) return fail(c,node,2,"I have not checked an indirect service call.");
    const NlServiceName *row=nl_service_namespace_lookup(c->space,c->source,name);
    if (!row) return fail(c,node,1,"I require a declared service or helper call.");
    if (row->kind == NL_SERVICE_METHOD) {
        NlServiceSignature sig;
        if (!nl_service_method_type(c->space,c->source,name,&sig) || count != (int)sig.parameter_count)
            return fail(c,node,1,"I require the catalog method argument count.");
        if (c->pure) return fail(c,node,1,"I cannot call a File service from a pure function.");
        for (int i=0;i<count;++i) {
            BodyValue expected={.type=sig.parameters[i],.borrow=i==0 && sig.input_mode==1 ? 2u:0u};
            require(c,args[i],expression(c,args[i]),expected);
        }
        return remember(c,node,(BodyValue){.type=sig.result},row->target);
    }
    if (row->kind != NL_SERVICE_FUNCTION || !row->declaration)
        return fail(c,node,1,"I require a callable declaration.");
    const ASTNode *fn=row->declaration;
    if (fn->as.function.is_extern) return fail(c,node,2,"I have not admitted extern transport in my service body path.");
    if (count != fn->as.function.param_count) return fail(c,node,1,"I require the helper argument count.");
    if (c->pure && !fn->as.function.is_pure) return fail(c,node,1,"I require a pure helper in a pure function.");
    for(int i=0;i<count;++i) {
        const Parameter *p=&fn->as.function.params[i];
        require(c,args[i],expression(c,args[i]),annotation(c,fn,p->type,p->type_info));
    }
    return remember(c,node,annotation(c,fn,fn->as.function.return_type,fn->as.function.return_type_info),row->target);
}
static BodyValue match(BodyCheck *c, const ASTNode *node) {
    BodyValue input=expression(c,node->as.match_expr.expr);
    if (input.type.base_type == TYPE_UNKNOWN) return input;
    if (input.borrow || input.type.base_type != TYPE_UNION || !input.type.service_declaration)
        return fail(c,node,2,"I have not checked this non-catalog match in my service body path.");
    unsigned seen=0; BodyValue result=value(TYPE_UNKNOWN); bool all_return=true;
    for(int i=0;i<node->as.match_expr.arm_count;++i) {
        const char *arm=node->as.match_expr.pattern_variants[i];
        unsigned bit=!strcmp(arm,"Ok")?1:!strcmp(arm,"Error")?2:0;
        if (!bit || (seen&bit)) return fail(c,node,1,"I require distinct catalog Result arms.");
        seen|=bit;
        if (node->as.match_expr.guard_exprs && node->as.match_expr.guard_exprs[i])
            return fail(c,node,2,"I have not checked guarded catalog Result exhaustiveness.");
        TypeInfo payload;
        if (!nl_service_member_type(c->space,&input.type,arm,&payload))
            return fail(c,node,1,"I require the exact catalog Result payload.");
        const char *name=node->as.match_expr.pattern_bindings[i];
        bool has=name && *name && strcmp(name,"_");
        if (has != (payload.base_type!=TYPE_VOID))
            return fail(c,node,1,"I require the exact catalog Result payload arity.");
        size_t saved=c->count;
        if (has) bind(c,node,name,(BodyValue){.type=payload},false);
        BodyValue body=expression(c,node->as.match_expr.arm_bodies[i]);
        c->count=saved; all_return &= body.returns;
        if (!body.returns) {
            if (result.type.base_type==TYPE_UNKNOWN) result=body;
            else require(c,node,body,result);
        }
    }
    if (seen!=3) return fail(c,node,1,"I require both catalog Result arms.");
    if (all_return) result=value(TYPE_VOID);
    result.returns=all_return;
    return result;
}
static BodyValue expression_impl(BodyCheck *c, const ASTNode *node) {
    if (node->lambda_definition)
        return fail(c,node,2,"I have not checked a captured callable in my service body path.");
    switch(node->type) {
    case AST_NUMBER: return value(TYPE_INT);
    case AST_BOOL: return value(TYPE_BOOL);
    case AST_IDENTIFIER: {
        int index=local(c,node->as.identifier);
        if (index>=0) return c->locals[index].value;
        if (nl_service_namespace_lookup(c->space,c->source,node->as.identifier))
            return fail(c,node,2,"I have not checked this non-local value in my service body path.");
        return fail(c,node,1,"I require a bound service body value.");
    }
    case AST_CALL:
        if (node->as.call.borrow_mode) {
            if (node->as.call.arg_count!=1 || node->as.call.args[0]->type!=AST_IDENTIFIER)
                return fail(c,node,1,"I require a named File borrow root.");
            int index=local(c,node->as.call.args[0]->as.identifier);
            if (index<0) return fail(c,node,1,"I require a bound File borrow root.");
            BodyValue v=c->locals[index].value;
            if (v.type.service_category!=1 || (node->as.call.borrow_mode==2 &&
                !(c->locals[index].mutable || v.borrow==2)) || (v.borrow==1 && node->as.call.borrow_mode==2))
                return fail(c,node,1,"I require a mutable File root for an exclusive borrow.");
            v.borrow=(unsigned)node->as.call.borrow_mode; return v;
        }
        return call(c,node,node->as.call.name,node->as.call.args,node->as.call.arg_count);
    case AST_MODULE_QUALIFIED_CALL: {
        char name[4096];
        int length=snprintf(name,sizeof name,"%s.%s",node->as.module_qualified_call.module_alias,node->as.module_qualified_call.function_name);
        if (length<0 || (size_t)length>=sizeof name) return fail(c,node,3,"I exceeded my qualified call name bound.");
        return call(c,node,name,node->as.module_qualified_call.args,node->as.module_qualified_call.arg_count);
    }
    case AST_PREFIX_OP: {
        TokenType op=node->as.prefix_op.op;
        bool logical=op==TOKEN_AND || op==TOKEN_OR || op==TOKEN_NOT;
        bool compare=op==TOKEN_EQ || op==TOKEN_NE || op==TOKEN_LT || op==TOKEN_LE || op==TOKEN_GT || op==TOKEN_GE;
        bool arithmetic=op==TOKEN_PLUS || op==TOKEN_MINUS || op==TOKEN_STAR || op==TOKEN_SLASH || op==TOKEN_PERCENT;
        if (!logical && !compare && !arithmetic) return fail(c,node,2,"I have not checked this service body operator.");
        int n=node->as.prefix_op.arg_count;
        if (n != (op==TOKEN_NOT?1:2) && !(op==TOKEN_MINUS && n==1))
            return fail(c,node,1,"I require the operator argument count.");
        BodyValue first=expression(c,node->as.prefix_op.args[0]);
        Type expected=logical?TYPE_BOOL:TYPE_INT;
        if ((op==TOKEN_EQ || op==TOKEN_NE) && first.type.base_type==TYPE_BOOL) expected=TYPE_BOOL;
        require(c,node,first,value(expected));
        for(int i=1;i<n;++i) require(c,node,expression(c,node->as.prefix_op.args[i]),value(expected));
        return value(logical||compare?TYPE_BOOL:TYPE_INT);
    }
    case AST_FIELD_ACCESS: {
        BodyValue object=expression(c,node->as.field_access.object); TypeInfo field;
        if (object.type.base_type==TYPE_UNKNOWN) return object;
        if (object.borrow || object.type.service_category!=3 ||
            !nl_service_member_type(c->space,&object.type,node->as.field_access.field_name,&field))
            return fail(c,node,1,"I require a declared catalog record field.");
        return (BodyValue){.type=field};
    }
    case AST_LET: {
        BodyValue actual=expression(c,node->as.let.value);
        BodyValue declared=annotation(c,node,node->as.let.var_type,node->as.let.type_info);
        if (node->as.let.var_type!=TYPE_UNKNOWN) require(c,node,actual,declared);
        if (actual.borrow) return fail(c,node,1,"I cannot store a call-scoped File borrow.");
        bind(c,node,node->as.let.name,actual,node->as.let.is_mut);
        remember(c,node,actual,0); return value(TYPE_VOID);
    }
    case AST_SET: {
        int index=local(c,node->as.set.name);
        if(index<0 || !c->locals[index].mutable) return fail(c,node,1,"I require a mutable assignment target.");
        if(node->as.set.field_name) return fail(c,node,2,"I have not checked service field assignment.");
        require(c,node,expression(c,node->as.set.value),c->locals[index].value); return value(TYPE_VOID);
    }
    case AST_RETURN: {
        BodyValue actual=expression(c,node->as.return_stmt.value);
        require(c,node,actual,c->result);
        actual.returns=true; return actual;
    }
    case AST_ASSERT:
        require(c,node,expression(c,node->as.assert.condition),value(TYPE_BOOL)); return value(TYPE_VOID);
    case AST_BLOCK: {
        size_t saved=c->count; BodyValue last=value(TYPE_VOID); bool returned=false;
        for(int i=0;i<node->as.block.count;++i) {
            last=expression(c,node->as.block.statements[i]); returned |= last.returns;
        }
        c->count=saved; last.returns=returned; return last;
    }
    case AST_IF: {
        require(c,node,expression(c,node->as.if_stmt.condition),value(TYPE_BOOL));
        BodyValue yes=expression(c,node->as.if_stmt.then_branch);
        BodyValue no=expression(c,node->as.if_stmt.else_branch);
        BodyValue result=value(TYPE_VOID); result.returns=yes.returns && no.returns; return result;
    }
    case AST_WHILE:
        require(c,node,expression(c,node->as.while_stmt.condition),value(TYPE_BOOL));
        expression(c,node->as.while_stmt.body); return value(TYPE_VOID);
    case AST_MATCH: return match(c,node);
    case AST_STRUCT_LITERAL: case AST_UNION_CONSTRUCT: {
        const char *name=node->type==AST_STRUCT_LITERAL ? node->as.struct_literal.struct_name : node->as.union_construct.union_name;
        TypeInfo type;
        if(nl_service_type(c->space,c->source,name,&type) && type.service_category==1)
            return fail(c,node,1,"I cannot fabricate a catalog File value.");
        return fail(c,node,2,"I have not checked this ordinary constructor in my service body path.");
    }
    default: return fail(c,node,2,"I have not checked this source form in my service body path.");
    }
}
static BodyValue expression(BodyCheck *c, const ASTNode *node) {
    if (!node) return value(TYPE_VOID);
    if (++c->depth>SB_DEPTH) { --c->depth; return fail(c,node,3,"I exceeded my service body nesting bound."); }
    BodyValue result=expression_impl(c,node); --c->depth;
    if (c->out->count && c->out->facts[c->out->count-1].node==node) return result;
    return remember(c,node,result,0);
}
NlServiceBodyCheck *nl_service_check_bodies(const NlServiceNamespace *space) {
    NlServiceBodyCheck *out=calloc(1,sizeof *out);
    BodyCheck *c=calloc(1,sizeof *c);
    if (!out || !c) { free(out); free(c); return NULL; }
    out->facts=calloc(SB_FACTS,sizeof *out->facts);
    if (!out->facts) { free(out); free(c); return NULL; }
    c->space=space; c->out=out;
    if (!space) { fail(c,NULL,1,"I require a complete service namespace."); }
    for(uint32_t module=0; space && nl_service_namespace_program(space,module); ++module) {
        const ASTNode *program=nl_service_namespace_program(space,module);
        c->source=nl_service_namespace_module(space,module);
        for(int i=0;i<program->as.program.count;++i) {
            const ASTNode *node=program->as.program.items[i];
            c->count=0; c->depth=0; c->pure=false; c->result=value(TYPE_VOID);
            const ASTNode *body=NULL;
            if (node->type==AST_FUNCTION) {
                ++out->functions; body=node->as.function.body; c->pure=node->as.function.is_pure;
                if(node->as.function.is_extern || node->as.function.is_anonymous) { fail(c,node,2,"I have not checked this callable in my service body path."); continue; }
                c->result=annotation(c,node,node->as.function.return_type,node->as.function.return_type_info);
                for(int j=0;j<node->as.function.param_count;++j) {
                    const Parameter *p=&node->as.function.params[j];
                    for(int k=0;k<j;++k) if(!strcmp(p->name,node->as.function.params[k].name)) fail(c,node,1,"I require distinct parameter names.");
                    bind(c,node,p->name,annotation(c,node,p->type,p->type_info),false);
                }
            } else if(node->type==AST_SHADOW) { ++out->shadows; body=node->as.shadow.body; }
            else if(node->type!=AST_SERVICE_DECL && node->type!=AST_IMPORT && node->type!=AST_MODULE_DECL)
                fail(c,node,2,"I have not checked this declaration beside service bodies.");
            if(body) {
                BodyValue result=expression(c,body);
                if(node->type==AST_FUNCTION && c->result.type.base_type!=TYPE_VOID && c->result.type.base_type!=TYPE_UNKNOWN && !result.returns)
                    fail(c,node,1,"I require a return on every service helper path.");
            }
        }
    }
    free(c); return out;
}
void nl_service_body_check_free(NlServiceBodyCheck *check) {
    if(check) { free(check->facts); free(check); }
}
