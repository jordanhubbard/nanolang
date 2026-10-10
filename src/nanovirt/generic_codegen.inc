/* I keep generic templates out of executable signatures. Each checked direct
 * call selects an owned concrete signature keyed by source body and types. */
extern const char *get_struct_type_name(ASTNode *, Environment *);

struct CgGeneric {
    ASTNode declaration;
    ASTNode *template;
    ASTNode *program;
    const char *source;
    uint32_t index;
    struct CgGeneric *next;
};

static bool cg_type_variable(CG *cg, Type type, const char *name) {
    return type == TYPE_STRUCT && name && name[0] >= 'A' && name[0] <= 'Z' &&
        !name[1] && !env_get_struct(cg->env, name);
}
static bool cg_generic_declaration(CG *cg, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION || node->as.function.is_extern) return false;
    for (int p=0; p<node->as.function.param_count; ++p)
        if(cg_type_variable(cg,node->as.function.params[p].type,
                            node->as.function.params[p].struct_type_name)) return true;
    return false;
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
    for(int p=0;p<instance->declaration.as.function.param_count;p++)
        free(instance->declaration.as.function.params[p].struct_type_name);
    free(instance->declaration.as.function.params);
    free(instance->declaration.as.function.return_struct_type_name);
    free(instance->declaration.as.function.name);
    free(instance);
}
static int32_t cg_generic_call(CG *cg, ASTNode *call) {
    Function *function=call->as.call.name?env_get_function(cg->env,call->as.call.name):NULL;
    if(!function || function->is_extern)return -1;
    Function retained_function=*function;function=&retained_function;
    bool generic=false;
    for(int p=0;p<function->param_count;p++)
        generic |= cg_type_variable(cg,function->params[p].type,function->params[p].struct_type_name);
    if(!generic)return -1;
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
    instance->declaration.as.function.params=params;
    instance->declaration.as.function.name=NULL;
    instance->declaration.as.function.return_struct_type_name=NULL;
    instance->program=owner;instance->source=source;
    for(int p=0;p<function->param_count;p++) {
        params[p]=definition->as.function.params[p];
        const char *name=params[p].struct_type_name;
        if(cg_type_variable(cg,params[p].type,name)) {
            params[p].type=check_expression(call->as.call.args[p],cg->env);
            params[p].type_info=NULL;
            name=params[p].type==TYPE_STRUCT?get_struct_type_name(call->as.call.args[p],cg->env):NULL;
            if(params[p].type!=TYPE_INT && params[p].type!=TYPE_U8 && params[p].type!=TYPE_FLOAT &&
               params[p].type!=TYPE_BOOL && params[p].type!=TYPE_STRING &&
               !(params[p].type==TYPE_STRUCT && name && env_get_struct(cg->env,name))) {
                params[p].struct_type_name=NULL;
                cg_error(cg,call->line,"I require a concrete scalar or declared record generic argument");goto fail;
            }
        }
        params[p].struct_type_name=name?strdup(name):NULL;
        if(name && !params[p].struct_type_name){cg_error(cg,call->line,"I cannot copy a generic record identity");goto fail;}
        for(int q=0;q<p;q++) {
            const Parameter *formal=&definition->as.function.params[p], *prior=&definition->as.function.params[q];
            if(cg_type_variable(cg,formal->type,formal->struct_type_name) &&
               cg_type_variable(cg,prior->type,prior->struct_type_name) && !strcmp(formal->struct_type_name,prior->struct_type_name) &&
               (params[p].type!=params[q].type || !!params[p].struct_type_name!=!!params[q].struct_type_name ||
                (params[p].struct_type_name && strcmp(params[p].struct_type_name,params[q].struct_type_name)))) {
                cg_error(cg,call->line,"I require one concrete identity for each generic type variable");goto fail;
            }
        }
    }
    const char *result_name=definition->as.function.return_struct_type_name;
    if(cg_type_variable(cg,definition->as.function.return_type,result_name)) {
        bool found=false;
        for(int p=0;p<function->param_count;p++) {
            const Parameter *formal=&definition->as.function.params[p];
            if(cg_type_variable(cg,formal->type,formal->struct_type_name) && !strcmp(formal->struct_type_name,result_name)) {
                instance->declaration.as.function.return_type=params[p].type;
                instance->declaration.as.function.return_type_info=NULL;
                result_name=params[p].struct_type_name;found=true;break;
            }
        }
        if(!found){cg_error(cg,call->line,"I cannot infer this generic result identity");goto fail;}
    }
    instance->declaration.as.function.return_struct_type_name=result_name?strdup(result_name):NULL;
    if(result_name && !instance->declaration.as.function.return_struct_type_name){cg_error(cg,call->line,"I cannot copy a generic result identity");goto fail;}
    for(CgGeneric *prior=cg->generics;prior;prior=prior->next) {
        if(prior->declaration.as.function.body!=definition->as.function.body)continue;
        bool same=true;
        for(int p=0;p<function->param_count;p++) {
            Parameter *a=&params[p],*b=&prior->declaration.as.function.params[p];
            if(a->type!=b->type || !!a->struct_type_name!=!!b->struct_type_name ||
               (a->struct_type_name && strcmp(a->struct_type_name,b->struct_type_name)))same=false;
        }
        if(same){uint32_t index=prior->index;cg_generic_free(instance);return (int32_t)index;}
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
