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
    puts("I passed nested, cyclic, deep and module-owned record classification checks.");
    return 0;
}
