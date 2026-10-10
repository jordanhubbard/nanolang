/* I keep generic templates out of executable signatures. Each checked direct
 * call selects an owned concrete signature keyed by source body and types. */
extern const char *get_struct_type_name(ASTNode *, Environment *);

typedef struct CgGenericType {
    TypeInfo *info;
    struct CgGenericType *next;
} CgGenericType;

struct CgGeneric {
    GenericBindings bindings;
    CgGenericType *local_types;
    ASTNode declaration;
    ASTNode *template;
    struct CgGeneric *parent;
    ASTNode *program;
    const char *source;
    uint32_t index;
    struct CgGeneric *next;
};

static bool cg_generic_declaration(CG *cg, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION || node->as.function.is_extern) return false;
    for (int p=0;p<node->as.function.param_count;++p) {
        TypeInfo view;
        if (generic_contains(cg->env,generic_parameter(&node->as.function.params[p],&view),0)) return true;
    }
    return false;
}
static const char *cg_generic_nominal(const TypeInfo *info) {
    if (!info) return NULL;
    if (info->base_type==TYPE_ARRAY && info->element_type) return info->element_type->generic_name;
    return info->generic_name;
}
static ASTNode *cg_generic_definition(CG *cg, ASTNode *body, ASTNode **owner, const char **source) {
    for(int m=-1; m<(cg->source_modules?cg->source_modules->count:0); ++m) {
        ASTNode *program=m<0?cg->root_program:get_cached_module_ast(cg->source_modules->module_paths[m]);
        if(!program || program->type!=AST_PROGRAM)continue;
        for(int i=0;i<program->as.program.count;i++) {
            ASTNode *node=bytecode_declaration(program->as.program.items[i]);
            if(node->type==AST_FUNCTION && node->as.function.body==body) {
                *owner=program;*source=m<0?cg->source_file:cg->source_modules->module_paths[m];
                return node;
            }
        }
    }
    return NULL;
}
static void cg_generic_free(CgGeneric *instance) {
    generic_bindings_free(&instance->bindings);
    while (instance->local_types) {
        CgGenericType *item=instance->local_types;
        instance->local_types=item->next;
        free_payload_type_info(item->info);free(item);
    }
    free_function_signature(instance->declaration.as.function.return_fn_sig);
    for(int p=0;p<instance->declaration.as.function.param_count;p++) {
        free(instance->declaration.as.function.params[p].struct_type_name);
        free_payload_type_info(instance->declaration.as.function.params[p].type_info);
    }
    free_payload_type_info(instance->declaration.as.function.return_type_info);
    free(instance->declaration.as.function.params);
    free(instance->declaration.as.function.return_struct_type_name);
    free(instance->declaration.as.function.name);
    free(instance);
}
static bool cg_generic_type_supported(CG *cg, const TypeInfo *info) {
    if (!info) return false;
    if (info->base_type == TYPE_FUNCTION) {
        if (!info->fn_sig) return false;
        TypeInfo view;
        for (int p=0;p<info->fn_sig->param_count;++p)
            if (!cg_generic_type_supported(cg,generic_signature_param(info->fn_sig,p,&view))) return false;
        return cg_generic_type_supported(cg,generic_signature_result(info->fn_sig,&view));
    }
    if (info->base_type == TYPE_ARRAY) return cg_generic_type_supported(cg, info->element_type);
    if (info->base_type == TYPE_STRUCT)
        return info->generic_name && env_get_struct(cg->env, info->generic_name);
    return info->base_type == TYPE_VOID || info->base_type == TYPE_INT || info->base_type == TYPE_U8 ||
        info->base_type == TYPE_FLOAT || info->base_type == TYPE_BOOL || info->base_type == TYPE_STRING;
}
static int32_t cg_generic_call(CG *cg, ASTNode *call) {
    Function *function=call->as.call.name?env_get_function(cg->env,call->as.call.name):NULL;
    /* Builtins have arity but no source body or Parameter array. */
    if(!function || function->is_extern || !function->body || !function->params)return -1;
    Function retained_function=*function;function=&retained_function;
    if(!generic_function(cg->env,function))return -1;
    if (check_expression(call, cg->env) == TYPE_UNKNOWN || !call->as.call.checked_signature) {
        cg_error(cg, call->line, "I require a complete checked generic signature");return -1;
    }
    /* Argument checking may grow my environment function table. */
    function = env_get_function(cg->env, call->as.call.name);
    if (!function) { cg_error(cg, call->line, "I lost the checked generic declaration"); return -1; }
    const FunctionSignature *signature = call->as.call.checked_signature;
    ASTNode *owner=NULL;const char *source=NULL;
    ASTNode *definition=cg_generic_definition(cg,function->body,&owner,&source);
    if(!definition || function->param_count!=call->as.call.arg_count || function->param_count>MAX_LOCALS) {
        cg_error(cg,call->line,"I require a retained generic declaration and exact argument count");return -1;
    }
    CgGeneric *instance=calloc(1,sizeof *instance);
    Parameter *params=calloc((size_t)function->param_count,sizeof *params);
    if(!instance || !params){free(instance);free(params);cg_error(cg,call->line,"I cannot allocate a generic signature");return -1;}
    instance->declaration=*definition;
    instance->template=definition;
    instance->parent=cg->generic_active;
    instance->declaration.as.function.params=params;
    instance->declaration.as.function.name=NULL;
    instance->declaration.as.function.return_struct_type_name=NULL;
    instance->declaration.as.function.return_type_info=copy_payload_type_info(signature->return_type_info);
    instance->declaration.as.function.return_fn_sig=copy_function_signature(signature->return_fn_sig);
    instance->declaration.as.function.return_type=signature->return_type;
    instance->declaration.as.function.return_element_type=signature->return_type_info && signature->return_type_info->element_type
        ? signature->return_type_info->element_type->base_type : TYPE_UNKNOWN;
    instance->program=owner;instance->source=source;
    for(int p=0;p<function->param_count;p++) {
        const Parameter *formal=&definition->as.function.params[p];
        params[p]=*formal;
        params[p].struct_type_name=NULL;
        params[p].type=signature->param_types[p];
        params[p].type_info=copy_payload_type_info(signature->param_type_info[p]);
        params[p].fn_sig=params[p].type_info ? params[p].type_info->fn_sig : NULL;
        TypeInfo view;
        const TypeInfo *pattern=generic_parameter(formal,&view);
        if (generic_contains(cg->env,pattern,0) &&
            (!cg_generic_type_supported(cg,params[p].type_info) ||
             !generic_bind(cg->env,pattern,params[p].type_info,&instance->bindings,0))) {
            cg_error(cg,call->line,"I require a complete supported generic parameter identity");goto fail;
        }
        const char *name=cg_generic_nominal(params[p].type_info);
        params[p].struct_type_name=name?strdup(name):NULL;
        if (params[p].type_info && params[p].type_info->element_type)
            params[p].element_type=params[p].type_info->element_type->base_type;
    }
    const char *result_name=cg_generic_nominal(signature->return_type_info);
    instance->declaration.as.function.return_struct_type_name=result_name?strdup(result_name):NULL;
    if (generic_contains(cg->env,signature->return_type_info,0)) {
        cg_error(cg,call->line,"I require a concrete generic result identity");goto fail;
    }
    for(CgGeneric *prior=cg->generics;prior;prior=prior->next) {
        if(prior->declaration.as.function.body!=definition->as.function.body)continue;
        bool same=true;
        for(int p=0;p<function->param_count;p++) {
            Parameter *a=&params[p],*b=&prior->declaration.as.function.params[p];
            if(a->type!=b->type || !type_infos_equal(a->type_info,b->type_info) ||
               !!a->struct_type_name!=!!b->struct_type_name ||
               (a->struct_type_name && strcmp(a->struct_type_name,b->struct_type_name)))same=false;
        }
        if(same){uint32_t index=prior->index;cg_generic_free(instance);return (int32_t)index;}
    }
    unsigned recursive_instances = 0;
    for (CgGeneric *parent = instance->parent; parent; parent = parent->parent) {
        if (parent->template == definition && ++recursive_instances >= 64) {
            cg_error(cg, call->line, "I reached my generic specialization budget (64 recursive instances per declaration)");goto fail;
        }
    }
    if(cg->module->function_count>=MAX_FUNCTIONS){cg_error(cg,call->line,"I reached my concrete function limit");goto fail;}
    char name[64];snprintf(name,sizeof name,"$generic.%u",cg->module->function_count);
    instance->declaration.as.function.name=strdup(name);
    if(!instance->declaration.as.function.name){cg_error(cg,call->line,"I cannot allocate a concrete function name");goto fail;}
    NvmFunctionEntry entry={0};
    entry.name_idx=nvm_add_string(cg->module,name,(uint32_t)strlen(name));
    if(entry.name_idx==UINT32_MAX){cg_error(cg,call->line,"I cannot retain a generic function name");goto fail;}
    entry.arity=(uint16_t)function->param_count;
    entry.result_tag=type_to_tag(instance->declaration.as.function.return_type,result_name,cg->env);
    entry.result_count=entry.result_tag==TAG_VOID?0:1;
    instance->index=nvm_add_function(cg->module,&entry);
    uint8_t tags[MAX_LOCALS];
    for(int p=0;p<function->param_count;p++)tags[p]=type_to_tag(params[p].type,params[p].struct_type_name,cg->env);
    if(instance->index==UINT32_MAX || !nvm_set_function_param_types(cg->module,instance->index,tags,entry.arity)) {
        cg_error(cg,call->line,"I cannot retain a concrete generic signature");goto fail;
    }
    if(cg->generic_tail)cg->generic_tail->next=instance;else cg->generics=instance;
    cg->generic_tail=instance;
    return (int32_t)instance->index;
fail:
    cg_generic_free(instance);return -1;
}
