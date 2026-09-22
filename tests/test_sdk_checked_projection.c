/* I check actual parsed declarations, never foreign code or synthetic selectors. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#ifdef NDEBUG
#error I require active projection assertions.
#endif
static size_t sdk_calls,sdk_live,sdk_fail=SIZE_MAX;
static bool sdk_persistent;
static void *sdk_allocate(size_t n,size_t size) {
    size_t at=sdk_calls++;
    if(at==sdk_fail||(sdk_persistent&&at>sdk_fail))return NULL;
    void *p=calloc(n,size);if(p)++sdk_live;return p;
}
static void sdk_free(void *p){if(p){assert(sdk_live);--sdk_live;free(p);}}
#define CHECKER_SDK_ALLOCATE(n,s) sdk_allocate(n,s)
#define CHECKER_SDK_FREE(p) sdk_free(p)
#include "../src/typechecker.c"
int g_argc;char **g_argv;char g_project_root[4096]=".";
const char *get_project_root(void){return g_project_root;}
typedef struct {Token *tokens;int count;ASTNode *program;Environment *env;} Source;
static Source source(const char *text) {
    Source s={0};s.tokens=tokenize(text,&s.count);assert(s.tokens);
    s.program=parse_program(s.tokens,s.count);assert(s.program);
    s.env=create_environment();assert(s.env);return s;
}
static void source_free(Source *s) {
    free_environment(s->env);free_ast(s->program);free_tokens(s->tokens,s->count);
    memset(s,0,sizeof *s);
}
static NvmPreparationBudget budget(void){return (NvmPreparationBudget){NVM_PREPARATION_MAX_BYTES,NVM_PREPARATION_MAX_STEPS};}
static const char *program_text=
    "module Provider\n"
    "struct Item { value:int }\n"
    "union Box<T> { Value { item:T } }\n"
    "enum Mode { First, Second }\n"
    "extern fn consume(items:array<Item>, callback:fn(Item)->Item)->Box<Item>\n"
    "extern fn consume(items:array<Item>, callback:fn(Item)->Item)->Box<Item>\n"
    "fn main()->int { return 0 }\n"
    "shadow main { assert true }\n";
static void facts(CheckerSdkProjection *p,Source *s) {
    assert(checker_sdk_projection_environment(p)==s->env);
    CheckerSdkSourceRow row,prior;
    assert(!checker_sdk_projection_row(p,0,&row)); /* Module marker is not a declaration. */
    assert(checker_sdk_projection_row(p,1,&row)&&row.kind==TYPE_STRUCT&&row.source_ordinal==1);
    assert(row.declaration==s->program->as.program.items[1]&&!strcmp(row.owner,"Provider"));
    size_t record_ordinal=row.environment_ordinal;
    assert(&s->env->structs[row.environment_ordinal]==env_get_struct_owned(s->env,row.declaration->as.struct_def.name,"Provider"));
    assert(checker_sdk_projection_row(p,2,&row)&&row.kind==TYPE_UNION);
    assert(s->env->unions[row.environment_ordinal].generic_param_count==1);
    assert(!strcmp(s->env->unions[row.environment_ordinal].generic_params[0],"T"));
    assert(checker_sdk_projection_row(p,3,&row)&&row.kind==TYPE_ENUM);
    assert(s->env->enums[row.environment_ordinal].variant_count==2);
    assert(checker_sdk_projection_row(p,4,&prior)&&prior.kind==TYPE_FUNCTION);
    assert(checker_sdk_projection_row(p,5,&row)&&row.kind==TYPE_FUNCTION);
    assert(row.declaration!=prior.declaration&&row.environment_ordinal==prior.environment_ordinal);
    const Function *fn=&s->env->functions[row.environment_ordinal];
    assert(fn->is_extern&&!fn->checker_builtin_placeholder&&fn->param_count==2);
    assert(fn->params[0].type_info&&fn->params[0].type_info->element_type);
    assert(fn->params[1].fn_sig|| (fn->params[1].type_info&&fn->params[1].type_info->fn_sig));
    assert(fn->return_type_info&&fn->return_type_info->type_param_count==1);
    NominalIdentity argument=env_nominal_identity(s->env,fn->return_type_info->type_params[0]->generic_name,row.owner,TYPE_STRUCT);
    assert(argument.ordinal==record_ordinal+1);
    assert(!checker_sdk_projection_row(p,7,&row));
    assert(!checker_sdk_projection_row(p,SIZE_MAX,&row));
    CheckerSdkSourceRow sentinel={.source_ordinal=999};row=sentinel;
    assert(!checker_sdk_projection_row(NULL,0,&row)&&row.source_ordinal==999);
    assert(!checker_sdk_projection_row(p,6,NULL));
}
static void positive(bool module) {
    Source s=source(program_text);CheckerSdkProjection *p=NULL;NvmPreparationBudget b=budget();
    assert(type_check_sdk_projection(s.program,s.env,module,&b,&p)==CHECKER_SDK_CAPTURED);facts(p,&s);
    /* The same basename in another actual checked owner cannot substitute for
     * this source occurrence. Ordinary registration invalidates the transient view. */
    Source other=source("module Elsewhere\nstruct Item { value:int }\nfn extra()->int{return 1}\nshadow extra{assert true}\n");
    assert(type_check_module(other.program,s.env));
    assert(!checker_sdk_projection_environment(p));
    CheckerSdkSourceRow row={.source_ordinal=999};
    assert(!checker_sdk_projection_row(p,1,&row)&&row.source_ordinal==999);
    checker_sdk_projection_free(p);source_free(&s);source_free(&other);assert(!sdk_live);
}
static void opaque_and_async(void) {
    Source s=source("module Foreign\nopaque type Handle\nextern fn handle_id(value:Handle)->Handle\nasync fn task_value()->int{return 7}\nshadow task_value{assert true}\n");
    NvmPreparationBudget b=budget();CheckerSdkProjection *p=NULL;
    assert(type_check_sdk_projection(s.program,s.env,true,&b,&p)==CHECKER_SDK_CAPTURED);
    CheckerSdkSourceRow row;
    assert(checker_sdk_projection_row(p,1,&row)&&row.kind==TYPE_OPAQUE&&!strcmp(row.owner,"Foreign"));
    const OpaqueTypeDef *opaque=&s.env->opaque_types[row.environment_ordinal];
    assert(opaque->origin&&opaque->identity&&opaque->c_type_name);
    assert(row.declaration_owner&&!strcmp(row.declaration_owner,opaque->origin));
    assert(checker_sdk_projection_row(p,2,&row)&&row.kind==TYPE_FUNCTION);
    assert(s.env->functions[row.environment_ordinal].is_extern);
    assert(checker_sdk_projection_row(p,3,&row)&&row.kind==TYPE_FUNCTION);
    assert(row.declaration==s.program->as.program.items[3]->as.async_fn.function);
    checker_sdk_projection_free(p);source_free(&s);assert(!sdk_live);
}
static void limits_and_failures(void) {
    Source s=source(program_text);NvmPreparationBudget b=budget(),start=b;CheckerSdkProjection *p=NULL;sdk_calls=0;
    assert(type_check_sdk_projection(s.program,s.env,true,&b,&p)==CHECKER_SDK_CAPTURED);
    size_t allocations=sdk_calls,bytes=start.bytes-b.bytes;uint32_t steps=start.steps-b.steps;
    checker_sdk_projection_free(p);source_free(&s);assert(!sdk_live&&allocations>2&&bytes&&steps);
    for(unsigned mode=0;mode<3;mode++) {
        s=source(program_text);b=(NvmPreparationBudget){bytes-(mode==1),steps-(mode==2)};start=b;p=NULL;
        CheckerSdkStatus result=type_check_sdk_projection(s.program,s.env,true,&b,&p);
        if(!mode){assert(result==CHECKER_SDK_CAPTURED&&!b.bytes&&!b.steps);facts(p,&s);checker_sdk_projection_free(p);}
        else assert(result==CHECKER_SDK_LIMIT&&p==NULL&&b.bytes==start.bytes&&b.steps==start.steps);
        source_free(&s);assert(!sdk_live);
    }
    for(unsigned mode=0;mode<2;mode++)for(size_t i=0;i<allocations;i++) {
        s=source(program_text);b=budget();start=b;p=NULL;sdk_calls=0;sdk_fail=i;sdk_persistent=mode!=0;
        assert(type_check_sdk_projection(s.program,s.env,true,&b,&p)==CHECKER_SDK_MEMORY);
        assert(p==NULL&&b.bytes==start.bytes&&b.steps==start.steps&&!sdk_live);
        sdk_fail=SIZE_MAX;sdk_persistent=false;source_free(&s);
        s=source(program_text);b=budget();p=NULL;
        assert(type_check_sdk_projection(s.program,s.env,true,&b,&p)==CHECKER_SDK_CAPTURED);facts(p,&s);
        checker_sdk_projection_free(p);source_free(&s);assert(!sdk_live);
    }
    s=source("fn main()->int{return false}\nshadow main{assert true}\n");b=budget();start=b;p=NULL;
    assert(type_check_sdk_projection(s.program,s.env,false,&b,&p)==CHECKER_SDK_CHECK_FAILED);
    assert(p==NULL&&b.bytes==start.bytes&&b.steps==start.steps&&!sdk_live);source_free(&s);
    s=source(program_text);b=budget();start=b;p=(void *)(uintptr_t)1;
    size_t before_calls=sdk_calls;int before_functions=s.env->function_count;
    assert(type_check_sdk_projection(s.program,s.env,true,&b,&p)==CHECKER_SDK_INVALID);
    assert(p==(void *)(uintptr_t)1&&b.bytes==start.bytes&&b.steps==start.steps);
    assert(sdk_calls==before_calls&&s.env->function_count==before_functions&&!sdk_live);source_free(&s);
    b=budget();start=b;p=NULL;
    assert(type_check_sdk_projection(NULL,NULL,false,&b,&p)==CHECKER_SDK_INVALID);
    assert(p==NULL&&b.bytes==start.bytes&&b.steps==start.steps);
}
int main(void){positive(false);positive(true);opaque_and_async();limits_and_failures();puts("PASS actual checked SDK declaration capture; no binding or execution admission");return 0;}
