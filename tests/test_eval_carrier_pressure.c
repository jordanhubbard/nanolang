/* I defer reclamation under bounded registration pressure, never root facts. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do {if(!(c)){fprintf(stderr,"I fail %s at %d.\n",#c,__LINE__);exit(1);}}while(0)
int g_argc;char **g_argv;char g_project_root[4096]=".";
const char *get_project_root(void){return g_project_root;}
static Value record(Environment *env){
    Value text=create_string("kept");char *name="text";
    Value value=create_struct("PressureItem",&name,&text,1);gc_release(text.as.string_val);
    env_carrier_track(env,value,false);return value;
}
static void enter(Environment *env){for(int i=0;i<3;i++)env_carrier_enter(env);}
static void leave(Environment *env){for(int i=0;i<3;i++)env_carrier_leave(env);}
static void retained_headroom(size_t live,size_t pressure) {
    Environment *env=create_environment();CHECK(env_carrier_enable(env));
    CHECK(env_carrier_enable_collection(env,NULL,0));enter(env);
    NlCarrierScopeMark empty=env_carrier_root_mark(env);
    for(size_t i=0;i<live;i++)(void)record(env);
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==live);
    NlCarrierScopeMark roots=env_carrier_root_mark(env);
    for(size_t i=1;i<=pressure;i++) {
        (void)record(env);env_carrier_root_reset(env,roots);CHECK(env_carrier_auto_collect(env));
        CHECK(env_carrier_stats(env).live==(i<pressure ? live+i : live));
    }
    env_carrier_root_reset(env,empty);CHECK(env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live==0);leave(env);free_environment(env);
    CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}
int main(void){
    retained_headroom(32,32);retained_headroom(128,64);
    Environment *env=create_environment();CHECK(env_carrier_enable(env));
    CHECK(env_carrier_enable_collection(env,NULL,0));enter(env);
    CHECK(env_carrier_auto_collect(env));
    NlCarrierScopeMark empty=env_carrier_root_mark(env);
    Value anchor=record(env);CHECK(env_carrier_collect(env));
    NlCarrierScopeMark retained=env_carrier_root_mark(env);
    for(int i=1;i<=16;i++){
        (void)record(env);env_carrier_root_reset(env,retained);
        CHECK(env_carrier_auto_collect(env));
        CHECK(env_carrier_stats(env).live==(i<16 ? (size_t)i+1 : 1));
    }
    CHECK(!strcmp(anchor.as.struct_val->field_values[0].as.string_val,"kept"));
    env_carrier_root_reset(env,empty);CHECK(env_carrier_auto_collect(env));
    CHECK(env_carrier_stats(env).live==1); /* I permit deferred retirement. */
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==0);
    List_int *lists[16];for(int i=0;i<16;i++)lists[i]=list_int_new();
    for(int i=1;i<=16;i++){
        env_carrier_consume_legacy_list(env,lists[i-1]);CHECK(env_carrier_auto_collect(env));
        CHECK(env_carrier_consumed_count(env)==(i<16 ? (size_t)i : 0));
    }
    Value caller=record(env);env_carrier_export(env,caller,false);
    CHECK(env_carrier_stats(env).transferred==1);env_carrier_root_reset(env,empty);
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==0);
    CHECK(!strcmp(caller.as.struct_val->field_values[0].as.string_val,"kept"));
    gc_release(caller.as.struct_val->field_values[0].as.string_val);
    release_struct_carrier_names(caller.as.struct_val);free(caller.as.struct_val->field_values);free(caller.as.struct_val);
    Value host_value=record(env);env_carrier_root_reset(env,empty);leave(env);
    NlCarrierRef host=nl_carrier_value(host_value);
    CHECK(env_carrier_enable_collection(env,&host,1));enter(env);
    CHECK(env_carrier_auto_collect(env));CHECK(env_carrier_stats(env).live==1);
    leave(env);CHECK(env_carrier_enable_collection(env,NULL,0));enter(env);
    CHECK(env_carrier_auto_collect(env));CHECK(env_carrier_stats(env).live==0);
    env_define_var(env,"bad",TYPE_LIST_GENERIC,true,create_void());CHECK(env_carrier_collect(env));
    Symbol *bad=env_get_var(env,"bad");CHECK(bad);bad->value=create_bool(true);
    CHECK(!env_carrier_collect(env));CHECK(!env_carrier_auto_collect(env));
    bad->value=create_void();CHECK(env_carrier_auto_collect(env));
    env_carrier_leave(env);bad->value=create_bool(true);CHECK(!env_carrier_auto_collect(env));
    bad->value=create_void();CHECK(env_carrier_auto_collect(env));env_carrier_enter(env);
    leave(env);host.record_list=255;
    CHECK(env_carrier_enable_collection(env,&host,1));enter(env);CHECK(!env_carrier_auto_collect(env));leave(env);
    CHECK(env_carrier_enable_collection(env,NULL,0));CHECK(env_carrier_auto_collect(env));
    free_environment(env);CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
    puts("I preserve16-registration/consumption pressure, explicit walks, live roots, transfers, host reconfiguration, failed-walk recovery and shallow malformed-root refusals.");
    return 0;
}
