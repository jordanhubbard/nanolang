/* I check resource propagation independently of control-flow checking. */
#include "resource_tracking.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

int main(void) {
    enum { COUNT = 300 };
    StructDef records[COUNT] = {0};
    char names[COUNT][24];
    char *targets[COUNT];
    Type type = TYPE_STRUCT;
    for (int i = 0; i < COUNT; i++) {
        snprintf(names[i], sizeof(names[i]), "Record%d", i);
        records[i].name = names[i];
        records[i].field_count = 1;
        records[i].field_types = &type;
        targets[i] = names[(i + 1) % COUNT];
        records[i].field_type_names = &targets[i];
    }
    Environment env = {.structs = records, .struct_count = COUNT};
    assert(!is_resource_type(&env, names[0]));
    records[COUNT - 1].is_resource = true;
    assert(is_resource_type(&env, names[0]));
    assert(is_resource_type(&env, names[COUNT - 1]));
    records[COUNT - 1].is_resource = false;
    assert(!is_resource_type(&env, names[0]));
    assert(!is_resource_type(&env, "Missing"));
    assert(!is_resource_type(&env, NULL));

    /* I register distinct module declarations instead of bypassing registration. */
    Environment registered = {0};
    registered.struct_capacity = 4;
    registered.structs = calloc(4, sizeof(StructDef));
    assert(registered.structs);
    env_define_struct(&registered, (StructDef){.name = "Handle", .module_name = "plain"});
    env_define_struct(&registered, (StructDef){.name = "Handle", .module_name = "owned", .is_resource = true});
    env_define_struct(&registered, (StructDef){.name = "Handle", .module_name = "owned"});
    assert(registered.struct_count == 2);
    registered.current_module = "plain";
    assert(!is_resource_type(&registered, "Handle"));
    registered.current_module = "owned";
    assert(is_resource_type(&registered, "Handle"));
    assert(!env_get_struct_owned(&registered, "Handle", "unrelated"));
    free(registered.structs);

    char *field = "Handle";
    StructDef scoped[] = {
        {.name = "Handle", .module_name = "caller"},
        {.name = "Handle", .module_name = "owner", .is_resource = true},
        {.name = "Box", .module_name = "owner", .field_count = 1,
         .field_types = &type, .field_type_names = &field}
    };
    env.structs = scoped;
    env.struct_count = 3;
    env.current_module = "caller";
    char *caller = env.current_module;
    assert(is_resource_type(&env, "Box"));
    assert(env.current_module == caller);
    assert(!is_resource_type(&env, "Handle"));
    Type union_type = TYPE_UNION;
    char *wrapper_name = "Choice";
    char *payload_names[] = {"Box", "Handle"};
    Type payload_types[] = {TYPE_STRUCT, TYPE_STRUCT};
    int counts[] = {0, 2};
    Type *arm_types[] = {NULL, payload_types};
    char **arm_names[] = {NULL, payload_names};
    UnionDef choices[] = {
        {.name = "Choice", .module_name = "caller"},
        {.name = "Choice", .module_name = "owner", .variant_count = 2,
         .variant_field_counts = counts, .variant_field_types = arm_types,
         .variant_field_type_names = arm_names}
    };
    scoped[2].field_types = &union_type;
    scoped[2].field_type_names = &wrapper_name;
    env.unions = choices;
    env.union_count = 2;
    assert(is_resource_type(&env, "Box"));
    assert(!is_resource_type(&env, "Choice"));
    assert(env.current_module == caller);
    env.current_module = "owner";
    assert(is_resource_type(&env, "Choice"));
    scoped[1].is_resource = false;
    assert(!is_resource_type(&env, "Choice"));
    assert(!is_resource_type(&env, "Box"));
    scoped[1].is_resource = true;
    payload_types[0] = TYPE_UNION;
    payload_names[0] = "Choice";
    assert(is_resource_type(&env, "Choice"));
    counts[1] = 1;
    assert(!is_resource_type(&env, "Choice"));
    puts("I passed nested, cyclic, deep and module-owned record/union classification checks.");
    return 0;
}
