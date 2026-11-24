/* I adopt only completed literal records and preserve every copying boundary. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/nanolang.h"
#include "../src/runtime/gc.h"
#include "../src/eval/carrier_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;
const char *get_project_root(void) { return "."; }

static void check_case(ASTNode *expression, Type type, int signal, bool plain,
                       bool adopted, bool copied) {
    gc_init();
    Environment *env=create_environment();
    assert(env);
    Value text=create_string("fresh");
    char *names[]={"text"};
    Value original=create_struct("Probe",names,&text,1);
    original.is_return=signal==1;
    original.is_break=signal==2;
    original.is_continue=signal==3;
    TypeInfo info={.base_type=TYPE_STRUCT,.generic_name="Probe"};
    GCStats before=gc_get_stats();
    if (plain) env_define_var_with_type_info(env,"record",type,TYPE_UNKNOWN,&info,true,original);
    else env_define_evaluated_var(env,"record",type,TYPE_UNKNOWN,&info,true,original,expression);
    Symbol *binding=env_get_var(env,"record");
    assert(binding && binding->type==type && binding->type_info==&info && binding->is_mut);
    assert(binding->value.type==VAL_STRUCT);
    bool same=binding->value.as.struct_val==original.as.struct_val;
    assert(same==(adopted || !copied));
    assert(strcmp(binding->value.as.struct_val->field_values[0].as.string_val,"fresh")==0);
    assert((binding->value.as.struct_val->field_values[0].as.string_val==
            original.as.struct_val->field_values[0].as.string_val)==same);
    GCStats after=gc_get_stats();
    assert(after.num_objects-before.num_objects==(copied?1u:0u));
    assert((after.total_allocated==before.total_allocated)==!copied);
    free_environment(env);
    if (!adopted) {
        assert(strcmp(original.as.struct_val->field_values[0].as.string_val,"fresh")==0);
        gc_release(original.as.struct_val->field_values[0].as.string_val);
        release_struct_carrier_names(original.as.struct_val);
        free(original.as.struct_val->field_values);
        free(original.as.struct_val);
    }
    gc_release(text.as.string_val);
    gc_shutdown();
    assert(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}

/* I retain an already-owned constructor across the next collection. */
static void tracked_literal(void) {
    gc_init();
    Environment *env=create_environment();
    if (!env_carrier_enabled(env)) assert(env_carrier_enable(env));
    assert(env_carrier_enable_collection(env,NULL,0));
    env_carrier_enter(env);
    Value text=create_string("tracked");
    char *names[]={"text"};
    Value original=create_struct("Probe",names,&text,1);
    env_carrier_track(env,original,true);
    assert(env_carrier_owns(env,original));
    ASTNode literal={.type=AST_STRUCT_LITERAL};
    GCStats before=gc_get_stats();
    env_define_evaluated_var(env,"record",TYPE_STRUCT,TYPE_UNKNOWN,NULL,false,original,&literal);
    assert(env_get_var(env,"record")->value.as.struct_val==original.as.struct_val);
    assert(gc_get_stats().total_allocated==before.total_allocated);
    env_carrier_leave(env);
    assert(env_carrier_collect(env));
    Value retained=env_get_var(env,"record")->value;
    assert(retained.as.struct_val==original.as.struct_val);
    assert(strcmp(retained.as.struct_val->field_values[0].as.string_val,"tracked")==0);
    free_environment(env);
    gc_release(text.as.string_val);
    gc_shutdown();
    assert(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}

int main(void) {
    ASTNode literal={.type=AST_STRUCT_LITERAL};
    check_case(&literal,TYPE_STRUCT,0,false,true,false);
    ASTNode spread={.type=AST_STRUCT_LITERAL};
    ASTNode source={.type=AST_IDENTIFIER};
    spread.as.struct_literal.spread_source=&source;
    check_case(&spread,TYPE_STRUCT,0,false,true,false);
    check_case(NULL,TYPE_STRUCT,0,false,false,true);
    ASTNodeType kinds[]={AST_IDENTIFIER,AST_CALL,AST_FIELD_ACCESS,AST_COND,AST_STRING};
    for (size_t i=0;i<sizeof kinds/sizeof *kinds;++i) {
        ASTNode expression={.type=kinds[i]};
        check_case(&expression,TYPE_STRUCT,0,false,false,true);
    }
    for (int signal=1;signal<=3;++signal) check_case(&literal,TYPE_STRUCT,signal,false,false,true);
    check_case(&literal,TYPE_STRUCT,0,true,false,true);
    check_case(&literal,TYPE_BORROW_SHARED,0,false,false,false);
    check_case(&literal,TYPE_BORROW_MUT,0,false,false,false);
    puts("I preserve fourteen literal/copy/borrowed boundaries before tracked adoption.");
    tracked_literal();
    puts("I adopt fresh and already-owned literals without duplicate registration; copies and borrows retain their policies.");
    return 0;
}
