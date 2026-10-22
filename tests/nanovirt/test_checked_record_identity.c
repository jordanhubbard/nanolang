#include "nanovirt/checked_record_identity.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    char name[] = "Byte", same_name[] = "Byte";
    char *fields[] = {"value"}, *same_fields[] = {"value"};
    ASTNode declaration = {.type = AST_STRUCT_DEF};
    declaration.as.struct_def.name = name;
    declaration.as.struct_def.field_names = fields;
    declaration.as.struct_def.field_count = 1;
    assert(nanovirt_checked_record_identity(&declaration,name,fields,1,false));
    assert(!nanovirt_checked_record_identity(NULL,name,fields,1,false));
    assert(!nanovirt_checked_record_identity(&declaration,name,fields,1,true));
    assert(!nanovirt_checked_record_identity(&declaration,same_name,fields,1,false));
    assert(!nanovirt_checked_record_identity(&declaration,name,same_fields,1,false));
    assert(!nanovirt_checked_record_identity(&declaration,name,fields,0,false));
    declaration.type = AST_NUMBER;
    assert(!nanovirt_checked_record_identity(&declaration,name,fields,1,false));
    puts("I preserve checked record identity and refuse missing/ambiguous registry entries.");
    return 0;
}
