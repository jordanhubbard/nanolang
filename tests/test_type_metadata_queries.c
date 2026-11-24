/* I preserve annotation identity and the parser's program-scope query boundary. */
#include <assert.h>
#include <stdio.h>
#define NL_DEFINE_ROW_FIELD_INFO
#define NL_DEFINE_AST_SERVICE_QUERY
#include "../src/type_metadata_queries.inc"
#undef NL_DEFINE_AST_SERVICE_QUERY
#undef NL_DEFINE_ROW_FIELD_INFO

int main(void) {
    Type kinds[] = {TYPE_INT, TYPE_ENUM};
    char *names[] = {"ignored", "Mode"};
    TypeInfo exact = {.base_type = TYPE_ARRAY};
    TypeInfo *annotations[] = {&exact, NULL};
    TypeInfo row = {.row_field_count = 2, .row_field_types = kinds,
                    .row_field_type_names = names, .row_field_type_info = annotations};
    TypeInfo fallback = {.base_type = TYPE_FUNCTION, .element_type = &exact};
    assert(row_field_info(&row, 0, NULL) == &exact);
    assert(row_field_info(&row, 1, &fallback) == &fallback);
    assert(fallback.base_type == TYPE_ENUM && fallback.generic_name == names[1]);
    assert(fallback.element_type == NULL && fallback.fn_sig == NULL);
    assert(row.row_field_type_info[0] == &exact && row.row_field_types[1] == TYPE_ENUM);
    assert(row_field_info(NULL, 0, &fallback) == NULL);
    assert(row_field_info(&row, -1, &fallback) == NULL);
    assert(row_field_info(&row, 2, &fallback) == NULL);
    assert(row_field_info(&row, 1, NULL) == NULL);
    row.row_field_types = NULL;
    assert(row_field_info(&row, 0, NULL) == &exact);
    assert(row_field_info(&row, 1, &fallback) == NULL);
    row.row_field_types = kinds;
    row.row_field_type_names = NULL;
    assert(row_field_info(&row, 1, &fallback) == &fallback);
    assert(fallback.base_type == TYPE_ENUM && fallback.generic_name == NULL);

    ASTNode ordinary = {.type = AST_NUMBER};
    ASTNode service = {.type = AST_SERVICE_DECL};
    ASTNode *items[] = {&ordinary, &service};
    ASTNode program = {.type = AST_PROGRAM};
    program.as.program.items = items;
    program.as.program.count = 1;
    assert(!ast_has_service_declaration(NULL));
    assert(!ast_has_service_declaration(&ordinary));
    assert(ast_has_service_declaration(&service));
    assert(!ast_has_service_declaration(&program));
    program.as.program.count = 2;
    assert(ast_has_service_declaration(&program));
    program.as.program.count = 0;
    assert(!ast_has_service_declaration(&program));
    puts("I preserve annotation priority and program-scope service queries.");
    return 0;
}
