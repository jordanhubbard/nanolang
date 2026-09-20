/* I keep the original parsed lifecycle test verbatim in its owning fixture. */
#define main checker_metadata_all_tests_main
#include "test_module_metadata.c"
#undef main

static void checker_orders(bool module, bool ast_first, bool discard_slots) {
    const char *source =
        "fn transform(values: array<array<int>>) -> array<array<int>> { return values } "
        "fn invoke(callback: fn(array<array<int>>) -> array<array<int>>, values: array<array<int>>) -> array<array<int>> { return (callback values) } "
        "fn factory() -> fn(array<array<int>>) -> array<array<int>> { return transform } "
        "fn first(values: array<int>) -> int { return (at values 0) } "
        "fn main() -> int { return 0 }";
    int count=0;
    Token *tokens=tokenize(source,&count);
    ASSERT_NOT_NULL(tokens);
    ASTNode *program=parse_program(tokens,count);
    ASSERT_NOT_NULL(program);
    Environment *env=create_environment();
    env->suppress_shadow_warnings=true;
    typecheck_set_current_file("<checker-ownership>");
    ASSERT(module?type_check_module(program,env):type_check(program,env));
    Function *invoke=env_get_function(env,"invoke");
    ASSERT_NOT_NULL(invoke);
    ASSERT_NOT_NULL(invoke->params[0].fn_sig);
    FunctionSignature *signature=invoke->params[0].fn_sig;
    ASSERT(signature->param_type_info[0]->element_type->element_type->base_type==TYPE_INT);
    ModuleMetadata *metadata=extract_module_metadata(env,"owned");
    ASSERT_NOT_NULL(metadata);
    if (discard_slots) {
        /* I simulate replaced metadata and removed function entries. Allocation
         * ownership must not depend on those mutable pointers or counts. */
        for(int i=0;i<env->symbol_count;i++) {
            if(env->symbols[i].type==TYPE_ARRAY || env->symbols[i].type==TYPE_FUNCTION) {
                env->symbols[i].value=create_void();
                env->symbols[i].type_info=NULL;
            }
        }
        /* Symbol slot truncation releases its ordinary owned names, but the
         * checker-only metadata/placeholder registry must survive the slots. */
        for(int i=0;i<env->symbol_count;i++) {
            free(env->symbols[i].name);
            free(env->symbols[i].struct_type_name);
        }
        env->symbol_count=0;
        env->function_count=0;
    }
    if(ast_first) {
        free_ast(program);free_tokens(tokens,count);free_environment(env);
    } else {
        free_environment(env);
        /* The registry must not deep-free the AST's callback annotation. */
        ASSERT(signature->param_type_info[0]->element_type->element_type->base_type==TYPE_INT);
        free_ast(program);free_tokens(tokens,count);
    }
    invoke=NULL;
    for(int i=0;i<metadata->function_count;i++)
        if(!strcmp(metadata->functions[i].name,"invoke"))invoke=&metadata->functions[i];
    ASSERT_NOT_NULL(invoke);
    ASSERT(invoke->params[0].fn_sig->return_type_info->element_type->element_type->base_type==TYPE_INT);
    free_module_metadata(metadata);
}

static void checker_borrowed_control(void) {
    TypeInfo integer={.base_type=TYPE_INT};
    Type tags[]={TYPE_INT};
    TypeInfo *infos[]={&integer};
    FunctionSignature signature={.param_count=1,.param_types=tags,.param_type_info=infos,
        .return_type=TYPE_INT,.return_type_info=&integer};
    Parameter parameter={.name="borrowed",.type=TYPE_FUNCTION,.fn_sig=&signature};
    Function function={.name="manual",.params=&parameter,.param_count=1,
        .return_type=TYPE_FUNCTION,.return_fn_sig=&signature};
    Environment *env=create_environment();
    env_define_function(env,function);
    TypeInfo callback={.base_type=TYPE_FUNCTION,.fn_sig=&signature};
    env_define_var_with_type_info(env,"callback",TYPE_FUNCTION,TYPE_UNKNOWN,&callback,false,create_void());
    /* I retain ownership of this ordinary runtime array outside the registry. */
    Value array=create_array(VAL_INT,1,1);
    ((long long *)array.as.array_val->data)[0]=37;
    env_define_var_with_element_type(env,"array",TYPE_ARRAY,TYPE_INT,false,array);
    free_environment(env);
    ASSERT(signature.param_type_info[0]==&integer);
    ASSERT(((long long *)array.as.array_val->data)[0]==37);
    free(array.as.array_val->data);free(array.as.array_val);
}

int main(void) {
    TEST(function_metadata_lifetime);
    for(int module=0;module<2;module++)
        for(int ast_first=0;ast_first<2;ast_first++)
            for(int discard=0;discard<2;discard++)
                checker_orders(module!=0,ast_first!=0,discard!=0);
    checker_borrowed_control();
    puts("Checker ownership: original parsed lifecycle, eight orders/slot controls, borrowed metadata/runtime array PASS");
    return 0;
}
