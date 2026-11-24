/* I retain actual carrier kinds without rooting ordinary scalar pointer bits. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static void roots(int kind) {
    Environment *env = create_environment(); CHECK(env_carrier_enable(env));
    CHECK(env_carrier_enable_collection(env,NULL,0));
    Value text = create_string("kept"); char *name = "text";
    Value child = create_struct("SelectedItem",&name,&text,1);
    gc_release(text.as.string_val);
    env_carrier_track(env,child,false);
    Value parent = child;
    Type root_type = TYPE_BORROW_SHARED;
    if (kind == 1) parent = create_tuple(&child,1);
    if (kind == 2) {
        parent = create_union("SelectedUnion",0,"Some",&name,&child,1);
        child = parent.as.union_val->field_values[0];
        env_carrier_track(env,parent,true);
    }
    if (kind == 3) {
        parent = create_array(VAL_STRUCT,1,1);
        ((StructValue **)parent.as.array_val->data)[0] = child.as.struct_val;
    }
    if (kind == 4) {
        DynArray *list = dyn_array_new(ELEM_INT); CHECK(list);
        dyn_array_push_int(list,(intptr_t)child.as.struct_val);
        env_carrier_track_list(env,list);
        parent = create_int((intptr_t)list); root_type = TYPE_LIST_GENERIC;
    }
    if (kind == 5) {
        List_int *list = list_int_new(); list_int_push(list,(intptr_t)child.as.struct_val);
        env_carrier_track_legacy_list(env,list);
        parent = create_int((intptr_t)list); root_type = TYPE_LIST_GENERIC;
    }
    if (kind == 1 || kind == 3) env_carrier_track(env,parent,false);
    env_define_var(env,"selected",root_type,true,parent);
    for (int i=0;i<2000;++i) {
        char label[64];snprintf(label,sizeof label,"scalar_%d",i);
        env_define_var(env,label,TYPE_INT,false,create_int((intptr_t)child.as.struct_val));
    }
    CHECK(env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live == (kind ? 2u : 1u));
    CHECK(!strcmp(child.as.struct_val->field_values[0].as.string_val,"kept"));
    Symbol *selected = env_get_var(env,"selected");CHECK(selected);
    if (kind < 4) {
        /* I honor the actual C/API value even when its declared scalar type differs. */
        selected->type=TYPE_INT;
        CHECK(env_carrier_collect(env));
        CHECK(env_carrier_stats(env).live == (kind ? 2u : 1u));
    } else {
        selected->value=create_bool(true);
        CHECK(!env_carrier_collect(env));
        selected->value=parent;
        CHECK(env_carrier_collect(env));
    }
    NlCarrierRef invalid=nl_carrier_value(create_int(0));invalid.record_list=255;
    CHECK(env_carrier_enable_collection(env,&invalid,1));
    CHECK(!env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live == (kind ? 2u : 1u));
    CHECK(env_carrier_enable_collection(env,NULL,0));
    selected=env_get_var(env,"selected");CHECK(selected);
    selected->type=TYPE_INT;
    selected->value=kind>=4 ? parent : create_void();
    CHECK(env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live==0);
    selected->value=create_void();
    free_environment(env);
    CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}
static void borrowed_scalar_array(void) {
    Environment *env=create_environment();CHECK(env_carrier_enable(env));
    CHECK(env_carrier_enable_collection(env,NULL,0));
    Value text=create_string("not an array edge");char *name="text";
    Value child=create_struct("ScalarArrayBits",&name,&text,1);gc_release(text.as.string_val);
    env_carrier_track(env,child,false);
    Value array=create_array(VAL_INT,1024,1024);
    ((int64_t *)array.as.array_val->data)[0]=(intptr_t)child.as.struct_val;
    env_define_var(env,"caller_array",TYPE_BORROW_SHARED,true,array);
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==0);
    CHECK(array.as.array_val->length==1024);
    int saved=array.as.array_val->capacity;array.as.array_val->capacity=0;
    CHECK(!env_carrier_collect(env));array.as.array_val->capacity=saved;
    CHECK(env_carrier_collect(env));
    /* I preserve owned scalar-array identity and eventual destruction. */
    env_carrier_track(env,array,false);CHECK(env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live==1);
    env_get_var(env,"caller_array")->value=create_void();
    CHECK(env_carrier_collect(env));CHECK(env_carrier_stats(env).live==0);
    free_environment(env);CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}
int main(void) {
    borrowed_scalar_array();
    for(int kind=0;kind<6;++kind)roots(kind);
    puts("I preserve record/union/tuple/array/canonical/legacy roots, malformed host/list refusals and scalar-bit nonownership across 2000 irrelevant bindings.");
    return 0;
}
