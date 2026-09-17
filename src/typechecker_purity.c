/* I prove empty closed effects by traversing every reachable body once.
 * Unknown nodes/calls remain open. Recursion cannot hide a reachable effect. */
#include "generated/purity_intrinsics.h"
enum { PURE_IO=1, PURE_READ=2, PURE_WRITE=4, PURE_UNKNOWN=8, PURE_UNSAFE=16, PURE_RESOURCE=32 };
typedef struct {
    Environment *env;
    const char *source_file;
    bool *seen;
    const char **locals;
    bool *mutable;
    int count, capacity;
} PurityWalk;
static unsigned purity_node(PurityWalk *, ASTNode *);
static unsigned purity_function(PurityWalk *, Function *);

static bool purity_builtin(const char *name) {
    for (int i=0; purity_intrinsic_names[i]; ++i)
        if (!strcmp(name, purity_intrinsic_names[i])) return true;
    return false;
}
static int purity_local(PurityWalk *p, const char *name) {
    for (int i=p->count-1; i>=0; --i) if (!strcmp(name,p->locals[i])) return i;
    return -1;
}
static bool purity_add(PurityWalk *p, const char *name, bool mut) {
    if (!name) return false;
    if (p->count == p->capacity) {
        int cap=p->capacity ? p->capacity*2 : 16;
        const char **names=realloc(p->locals,(size_t)cap*sizeof(*names));
        if (!names) return false;
        p->locals=names;
        bool *flags=realloc(p->mutable,(size_t)cap*sizeof(*flags));
        if (!flags) return false;
        p->mutable=flags; p->capacity=cap;
    }
    p->locals[p->count]=name; p->mutable[p->count++]=mut; return true;
}
static unsigned purity_type(PurityWalk *p, Type type, const char *name, TypeInfo *info, int depth) {
    if (depth>32) return PURE_UNKNOWN;
    if (type==TYPE_OPAQUE || is_resource_type(p->env,name)) return PURE_RESOURCE;
    unsigned result=0;
    if (info) {
        if (info->base_type==TYPE_OPAQUE || is_resource_type(p->env,info->generic_name)) result|=PURE_RESOURCE;
        if (info->element_type) result|=purity_type(p,info->element_type->base_type,NULL,info->element_type,depth+1);
        for (int i=0;i<info->type_param_count;++i)
            if (info->type_params && info->type_params[i]) result|=purity_type(p,info->type_params[i]->base_type,NULL,info->type_params[i],depth+1);
        for (int i=0;i<info->tuple_element_count;++i)
            result|=purity_type(p,info->tuple_types ? info->tuple_types[i] : TYPE_UNKNOWN,
                                info->tuple_type_names ? info->tuple_type_names[i] : NULL,NULL,depth+1);
    }
    return result;
}
/* I accept external values only when their complete value shape is immutable.
 * An immutable binding does not freeze an aliased array/map/list allocation. */
static bool purity_input_value(PurityWalk *p, Type type, const char *name, int depth) {
    if (depth>32) return false;
    switch (type) {
    case TYPE_INT: case TYPE_U8: case TYPE_FLOAT: case TYPE_BOOL:
    case TYPE_STRING: case TYPE_VOID: case TYPE_ENUM: return true;
    case TYPE_STRUCT: {
        if (!name) return false;
        StructDef *record=env_get_struct(p->env,name);
        if (!record || record->is_resource) return false;
        char *saved=p->env->current_module;
        p->env->current_module=record->module_name;
        bool safe=true;
        for (int i=0; safe && i<record->field_count; ++i) {
            if (!record->field_types || !purity_input_value(p,record->field_types[i],
                    record->field_type_names ? record->field_type_names[i] : NULL,depth+1)) safe=false;
        }
        p->env->current_module=saved;
        return safe;
    }
    default: return false;
    }
}
static Function *purity_function_decl(PurityWalk *p, const char *name) {
    if (!name) return NULL;
    Function *fn=env_get_function(p->env,name);
    /* I prefer an explicit declaration in this owner over a registry spelling. */
    for (int i=0; i<p->env->function_count; ++i) {
        Function *decl=&p->env->functions[i];
        if (decl->name && !strcmp(decl->name,name) &&
            ((!decl->module_name && !p->env->current_module) ||
             (decl->module_name && p->env->current_module &&
              !strcmp(decl->module_name,p->env->current_module)))) {
            fn=decl;
            break;
        }
    }
    return fn;
}
static unsigned purity_call(PurityWalk *p, const char *name) {
    if (!name || purity_local(p,name)>=0) return PURE_UNKNOWN;
    Function *fn=purity_function_decl(p,name);
    if (fn && fn->is_extern) return PURE_UNKNOWN;
    if (fn && fn->body) return purity_function(p,fn);
    if (purity_builtin(name)) return 0;
    for (int i=0; purity_io_names[i]; ++i)
        if (!strcmp(name,purity_io_names[i])) return PURE_IO;
    return PURE_UNKNOWN;
}
static unsigned purity_identifier(PurityWalk *p,const char *name) {
    int local=purity_local(p,name);
    if (local>=0) return p->mutable[local] ? PURE_READ : 0;
    unsigned effects=0; bool found=false;
    for (int i=0;i<p->env->symbol_count;++i) {
        Symbol *sym=&p->env->symbols[i];
        if (!sym->is_global || strcmp(sym->name,name)) continue;
        if (sym->def_file && p->source_file && strcmp(sym->def_file,p->source_file)) continue;
        found=true;
        if (sym->is_mut || !purity_input_value(p,sym->type,sym->struct_type_name,0)) effects|=PURE_READ;
        effects|=purity_type(p,sym->type,sym->struct_type_name,sym->type_info,0);
    }
    if (found || env_get_enum(p->env,name)) return effects;
    /* Taking a resolved declaration's function value has no observable effect.
     * Calling a function-typed local remains unknown in purity_call. */
    if (purity_function_decl(p,name)) return 0;
    return PURE_UNKNOWN;
}
static unsigned purity_node(PurityWalk *p, ASTNode *node) {
    if (!node) return 0;
    unsigned result=0;
#define CHILD(n) (result |= purity_node(p,(n)))
#define MANY(a,n) do { for(int j=0;j<(n);++j) CHILD((a)[j]); } while(0)
    switch(node->type) {
    case AST_NUMBER: case AST_FLOAT: case AST_STRING: case AST_BOOL: break;
    case AST_IDENTIFIER: result|=purity_identifier(p,node->as.identifier); break;
    case AST_PREFIX_OP: MANY(node->as.prefix_op.args,node->as.prefix_op.arg_count); break;
    case AST_CALL:
        result|=node->as.call.func_expr ? PURE_UNKNOWN : purity_call(p,node->as.call.name);
        CHILD(node->as.call.func_expr); MANY(node->as.call.args,node->as.call.arg_count); break;
    case AST_MODULE_QUALIFIED_CALL: {
        const char *module=node->as.module_qualified_call.module_alias;
        const char *function=node->as.module_qualified_call.function_name;
        size_t length=strlen(module)+strlen(function)+2;
        char *name=malloc(length);
        if (!name) result|=PURE_UNKNOWN;
        else {
            snprintf(name,length,"%s.%s",module,function);
            result|=purity_call(p,name);
            free(name);
        }
        MANY(node->as.module_qualified_call.args,node->as.module_qualified_call.arg_count); break;
    }
    case AST_LET:
        CHILD(node->as.let.value);
        result|=purity_type(p,node->as.let.var_type,node->as.let.type_name,node->as.let.type_info,0);
        if (node->as.let.is_mut) result|=PURE_WRITE;
        if (!purity_add(p,node->as.let.name,node->as.let.is_mut)) result|=PURE_UNKNOWN;
        break;
    case AST_SET: result|=PURE_WRITE; CHILD(node->as.set.value); break;
    case AST_BLOCK: {
        int saved=p->count;
        MANY(node->as.block.statements,node->as.block.count);
        p->count=saved; break;
    }
    case AST_IF: CHILD(node->as.if_stmt.condition); CHILD(node->as.if_stmt.then_branch); CHILD(node->as.if_stmt.else_branch); break;
    case AST_COND:
        MANY(node->as.cond_expr.conditions,node->as.cond_expr.clause_count);
        MANY(node->as.cond_expr.values,node->as.cond_expr.clause_count); CHILD(node->as.cond_expr.else_value); break;
    case AST_MATCH:
        CHILD(node->as.match_expr.expr);
        for (int i=0; i<node->as.match_expr.arm_count; ++i) {
            int saved=p->count;
            const char *binding=node->as.match_expr.pattern_bindings
                ? node->as.match_expr.pattern_bindings[i] : NULL;
            if (binding && binding[0] && strcmp(binding,"_") &&
                !purity_add(p,binding,false)) result|=PURE_UNKNOWN;
            if (node->as.match_expr.guard_exprs) CHILD(node->as.match_expr.guard_exprs[i]);
            if (node->as.match_expr.arm_bodies) CHILD(node->as.match_expr.arm_bodies[i]);
            p->count=saved;
        }
        break;
    case AST_WHILE: case AST_FOR: result|=PURE_UNKNOWN; break;
    case AST_RETURN: CHILD(node->as.return_stmt.value); break;
    case AST_PRINT: result|=PURE_IO; CHILD(node->as.print.expr); break;
    case AST_ASSERT: CHILD(node->as.assert.condition); break;
    case AST_ARRAY_LITERAL: MANY(node->as.array_literal.elements,node->as.array_literal.element_count); break;
    case AST_STRUCT_LITERAL:
        if (is_resource_type(p->env,node->as.struct_literal.struct_name)) result|=PURE_RESOURCE;
        MANY(node->as.struct_literal.field_values,node->as.struct_literal.field_count); CHILD(node->as.struct_literal.spread_source); break;
    case AST_FIELD_ACCESS: CHILD(node->as.field_access.object); break;
    case AST_TUPLE_LITERAL: MANY(node->as.tuple_literal.elements,node->as.tuple_literal.element_count); break;
    case AST_TUPLE_INDEX: CHILD(node->as.tuple_index.tuple); break;
    case AST_UNSAFE_BLOCK: result|=PURE_UNSAFE|PURE_UNKNOWN; break;
    /* I keep unclassified effects, async and higher-order constructs open. */
    default: result|=PURE_UNKNOWN; break;
    }
#undef CHILD
#undef MANY
    return result;
}
static unsigned purity_function(PurityWalk *parent,Function *fn) {
    if (!fn || !fn->body) return PURE_UNKNOWN;
    int index=(int)(fn-parent->env->functions);
    if (index<0 || index>=parent->env->function_count) return PURE_UNKNOWN;
    if (parent->seen[index]) return 0;
    parent->seen[index]=true;
    PurityWalk p={0}; p.env=parent->env; p.seen=parent->seen;
    p.source_file=parent->source_file;
    if (fn->source_file) p.source_file=fn->source_file;
    char *saved=p.env->current_module; p.env->current_module=fn->module_name;
    unsigned result=purity_type(&p,fn->return_type,fn->return_struct_type_name,fn->return_type_info,0);
    for(int i=0;i<fn->param_count;++i) {
        Parameter *param=&fn->params[i];
        result|=purity_type(&p,param->type,param->struct_type_name,param->type_info,0);
        if (!purity_input_value(&p,param->type,param->struct_type_name,0)) result|=PURE_READ;
        if (!purity_add(&p,param->name,false)) result|=PURE_UNKNOWN;
    }
    result|=purity_node(&p,fn->body);
    p.env->current_module=saved;
    free(p.locals); free(p.mutable);
    return result;
}
static void check_purity(ASTNode *body,Environment *env,const char *name) {
    if (!body) return; /* An extern annotation is not a verified call summary. */
    PurityWalk p={0}; p.env=env; p.source_file=env_current_file(env);
    p.seen=calloc((size_t)env->function_count,sizeof(bool));
    unsigned effects=p.seen ? purity_function(&p,env_get_function(env,name)) : PURE_UNKNOWN;
    free(p.seen);
    if (effects) {
        emit_context_error("E002 PURITY VIOLATION",body->line,body->column,1,
            "I cannot prove a closed empty effect summary for this pure fn",
            "I reject observable state, mutation, resources and unknown calls or constructs");
        g_typecheck_error_count++;
    }
}
