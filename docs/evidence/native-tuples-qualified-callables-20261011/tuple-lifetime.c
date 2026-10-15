#include "nanolang.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
int main(void) {
    Type params[] = {TYPE_INT};
    FunctionSignature *signature = create_function_signature(params, 1, TYPE_FUNCTION);
    signature->return_fn_sig = create_function_signature(params, 1, TYPE_BOOL);
    Value original = create_function("retained_callback", signature);
    Value fields[] = {original, original};
    Value tuple = create_tuple(fields, 2);
    free(original.as.function_val.function_name);
    free_function_signature(original.as.function_val.signature);
    for (int i=0; i<2; ++i) {
        Value retained = tuple.as.tuple_val->elements[i];
        assert(strcmp(retained.as.function_val.function_name, "retained_callback") == 0);
        assert(retained.as.function_val.signature->param_types[0] == TYPE_INT);
        assert(retained.as.function_val.signature->return_fn_sig->return_type == TYPE_BOOL);
    }
    assert(tuple.as.tuple_val->elements[0].as.function_val.signature != tuple.as.tuple_val->elements[1].as.function_val.signature);
    free_tuple(tuple.as.tuple_val);
    return 0;
}
