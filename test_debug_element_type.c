#include <stdio.h>
#include "src/nanolang.h"

int main() {
    Environment *env = create_environment();
    
    // Create a Symbol with element_type
    Value val = create_array(VAL_STRING, 0, 0);
    env_define_var_with_element_type(env, "test_array", TYPE_ARRAY, TYPE_STRING, false, val);
    
    // Look it up
    Symbol *sym = env_get_var(env, "test_array");
    if (sym) {
        printf("Symbol found!\n");
        printf("Type: %d (TYPE_ARRAY=%d)\n", sym->type, TYPE_ARRAY);
        printf("Element type: %d (TYPE_STRING=%d)\n", sym->element_type, TYPE_STRING);
    } else {
        printf("Symbol not found!\n");
    }
    
    return 0;
}
