/* I am included after the ordinary checker entry points. */
#ifndef CHECKER_SDK_ALLOCATE
#define CHECKER_SDK_ALLOCATE(count,size) calloc((count),(size))
#endif
#ifndef CHECKER_SDK_FREE
#define CHECKER_SDK_FREE(pointer) free(pointer)
#endif
struct CheckerSdkProjection {
    ASTNode *program;
    Environment *env;
    CheckerSdkSourceRow *rows;
    size_t count;
    int functions,records,unions,enums,opaques;
    NvmPreparationBudget remaining;
    CheckerSdkStatus status;
};
static CheckerSdkProjection *checker_sdk_active;
static bool checker_sdk_charge(CheckerSdkProjection *p,size_t bytes,uint32_t steps) {
    if(p->status!=CHECKER_SDK_CAPTURED)return false;
    if(nvm_preparation_charge(&p->remaining,bytes,steps))return true;
    p->status=CHECKER_SDK_LIMIT;return false;
}
static const ASTNode *checker_sdk_declaration(const ASTNode *n) {
    return n&&n->type==AST_ASYNC_FN?n->as.async_fn.function:n;
}
static Type checker_sdk_kind(const ASTNode *n) {
    if(!n)return TYPE_UNKNOWN;
    switch(n->type) {
        case AST_FUNCTION:return TYPE_FUNCTION;
        case AST_STRUCT_DEF:return TYPE_STRUCT;
        case AST_UNION_DEF:return TYPE_UNION;
        case AST_ENUM_DEF:return TYPE_ENUM;
        case AST_OPAQUE_TYPE:return TYPE_OPAQUE;
        default:return TYPE_UNKNOWN;
    }
}
static bool checker_sdk_copy_owner(CheckerSdkProjection *p,const char *owner,char **out) {
    if(!owner){*out=NULL;return true;}
    size_t length=strnlen(owner,4097);
    if(length>4096){p->status=CHECKER_SDK_LIMIT;return false;}
    if(!checker_sdk_charge(p,length+1,(uint32_t)length+1))return false;
    char *copy=CHECKER_SDK_ALLOCATE(length+1,1);
    if(!copy){p->status=CHECKER_SDK_MEMORY;return false;}
    memcpy(copy,owner,length+1);*out=copy;return true;
}
static void checker_sdk_capture(Environment *env,const ASTNode *node,Type kind,size_t ordinal) {
    CheckerSdkProjection *p=checker_sdk_active;
    if(!p||p->env!=env||p->status!=CHECKER_SDK_CAPTURED)return;
    if(p->program->as.program.count<0||(size_t)p->program->as.program.count!=p->count||
       (p->count&&!p->program->as.program.items)){p->status=CHECKER_SDK_INVALID;return;}
    for(size_t i=0;i<p->count;i++) {
        if(!checker_sdk_charge(p,0,1))return;
        if(checker_sdk_declaration(p->program->as.program.items[i])!=node)continue;
        if(checker_sdk_kind(node)!=kind||p->rows[i].declaration){p->status=CHECKER_SDK_INVALID;return;}
        const char *owner=NULL,*declaration_owner=NULL;
        switch(kind) {
            case TYPE_FUNCTION:if(ordinal>=(size_t)env->function_count)goto invalid;owner=env_function_signature_owner(env,&env->functions[ordinal]);declaration_owner=env->functions[ordinal].module_name;break;
            case TYPE_STRUCT:if(ordinal>=(size_t)env->struct_count)goto invalid;owner=declaration_owner=env->structs[ordinal].module_name;break;
            case TYPE_UNION:if(ordinal>=(size_t)env->union_count)goto invalid;owner=declaration_owner=env->unions[ordinal].module_name;break;
            case TYPE_ENUM:if(ordinal>=(size_t)env->enum_count)goto invalid;owner=declaration_owner=env->enums[ordinal].module_name;break;
            case TYPE_OPAQUE:if(ordinal>=(size_t)env->opaque_type_count)goto invalid;owner=env->current_module;declaration_owner=env->opaque_types[ordinal].origin;break;
            default:goto invalid;
        }
        char *copy=NULL,*origin=NULL;
        if(!checker_sdk_copy_owner(p,owner,&copy))return;
        if(!checker_sdk_copy_owner(p,declaration_owner,&origin)){CHECKER_SDK_FREE(copy);return;}
        p->rows[i]=(CheckerSdkSourceRow){node,kind,i,ordinal,copy,origin};return;
    }
    /* Nested declarations and separately checked dependencies do not become
     * occurrences in this source merely because registration called my hook. */
    return;
invalid:p->status=CHECKER_SDK_INVALID;
}
static void checker_sdk_capture_opaque(Environment *env,const ASTNode *node,const char *name) {
    if(!checker_sdk_active||checker_sdk_active->env!=env||checker_sdk_active->status!=CHECKER_SDK_CAPTURED)return;
    OpaqueTypeDef *definition=env_get_opaque_type(env,name);
    if(!definition){checker_sdk_active->status=CHECKER_SDK_INVALID;return;}
    checker_sdk_capture(env,node,TYPE_OPAQUE,(size_t)(definition-env->opaque_types));
}
void checker_sdk_projection_free(CheckerSdkProjection *p) {
    if(!p)return;
    for(size_t i=0;p->rows&&i<p->count;i++) {
        CHECKER_SDK_FREE((char *)p->rows[i].owner);
        CHECKER_SDK_FREE((char *)p->rows[i].declaration_owner);
    }
    CHECKER_SDK_FREE(p->rows);CHECKER_SDK_FREE(p);
}
static bool checker_sdk_projection_stable(const CheckerSdkProjection *p) {
    return p&&p->env&&p->program&&p->program->type==AST_PROGRAM&&
        p->program->as.program.count>=0&&(size_t)p->program->as.program.count==p->count&&
        p->functions==p->env->function_count&&p->records==p->env->struct_count&&
        p->unions==p->env->union_count&&p->enums==p->env->enum_count&&p->opaques==p->env->opaque_type_count;
}
bool checker_sdk_projection_row(const CheckerSdkProjection *p,size_t source,CheckerSdkSourceRow *out) {
    if(!out||!checker_sdk_projection_stable(p)||source>=p->count||!p->rows[source].declaration||
       checker_sdk_declaration(p->program->as.program.items[source])!=p->rows[source].declaration)return false;
    *out=p->rows[source];return true;
}
const Environment *checker_sdk_projection_environment(const CheckerSdkProjection *p) {
    return checker_sdk_projection_stable(p)?p->env:NULL;
}
CheckerSdkStatus type_check_sdk_projection(ASTNode *program,Environment *env,bool module,
        NvmPreparationBudget *budget,CheckerSdkProjection **out) {
    if(!program||!env||!budget||!out||checker_sdk_active||program->type!=AST_PROGRAM||
       program->as.program.count<0||(program->as.program.count&&!program->as.program.items))return CHECKER_SDK_INVALID;
    if(!nvm_preparation_budget_valid(budget)||program->as.program.count>65536)return CHECKER_SDK_LIMIT;
    size_t count=(size_t)program->as.program.count;
    NvmPreparationBudget remaining=*budget;
    if(!nvm_preparation_charge(&remaining,sizeof(CheckerSdkProjection)+(count?count:1)*sizeof(CheckerSdkSourceRow),0))return CHECKER_SDK_LIMIT;
    CheckerSdkProjection *p=CHECKER_SDK_ALLOCATE(1,sizeof *p);if(!p)return CHECKER_SDK_MEMORY;
    p->count=count;p->program=program;p->env=env;p->remaining=remaining;p->status=CHECKER_SDK_CAPTURED;
    p->rows=CHECKER_SDK_ALLOCATE(count?count:1,sizeof *p->rows);
    if(!p->rows){checker_sdk_projection_free(p);return CHECKER_SDK_MEMORY;}
    for(size_t i=0;i<count;i++)if(!checker_sdk_declaration(program->as.program.items[i])){checker_sdk_projection_free(p);return CHECKER_SDK_INVALID;}
    checker_sdk_active=p;
    bool checked=module?type_check_module(program,env):type_check(program,env);
    checker_sdk_active=NULL;
    if(!checked)p->status=CHECKER_SDK_CHECK_FAILED;
    else if(program->as.program.count<0||(size_t)program->as.program.count!=count||
            (count&&!program->as.program.items))p->status=CHECKER_SDK_INVALID;
    for(size_t i=0;i<count&&p->status==CHECKER_SDK_CAPTURED;i++) {
        if(!checker_sdk_charge(p,0,1))break;
        const ASTNode *node=checker_sdk_declaration(program->as.program.items[i]);
        if(checker_sdk_kind(node)!=TYPE_UNKNOWN&&!p->rows[i].declaration)p->status=CHECKER_SDK_INVALID;
    }
    if(p->status!=CHECKER_SDK_CAPTURED){CheckerSdkStatus status=p->status;checker_sdk_projection_free(p);return status;}
    p->functions=env->function_count;p->records=env->struct_count;p->unions=env->union_count;
    p->enums=env->enum_count;p->opaques=env->opaque_type_count;
    *budget=p->remaining;*out=p;return CHECKER_SDK_CAPTURED;
}
