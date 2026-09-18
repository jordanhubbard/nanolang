/* I test public C staging, declaration resolution and recovery with fresh ASTs. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t allocation_index, fail_at;
static void *fixture_malloc(size_t bytes) {
    ++allocation_index;
    return fail_at && allocation_index == fail_at ? NULL : malloc(bytes);
}
#define malloc fixture_malloc
#include "../src/c_backend.c"
#undef malloc
/* I do not admit passive nodes in this isolated scalar API fixture. */
int *passive_binding_order(const ASTNode *block) {
    (void)block;
    assert(!"I reached a passive path outside this fixture.");
    return NULL;
}

static void previous(const char *path) {
    FILE *out = fopen(path, "w");
    assert(out); assert(fputs("previous", out) >= 0); assert(fclose(out) == 0);
}
static void retained(const char *path) {
    FILE *in = fopen(path, "r"); char text[32] = {0};
    assert(in); assert(fread(text, 1, sizeof text, in) == 8);
    assert(strcmp(text, "previous") == 0); assert(fclose(in) == 0);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    ASTNode zero = {.type = AST_NUMBER};
    ASTNode result = {.type = AST_RETURN}; result.as.return_stmt.value = &zero;
    ASTNode *statements[] = {&result};
    ASTNode block = {.type = AST_BLOCK}; block.as.block.statements = statements; block.as.block.count = 1;
    ASTNode main_function = {.type = AST_FUNCTION};
    main_function.as.function.name = "main"; main_function.as.function.return_type = TYPE_INT;
    main_function.as.function.body = &block;
    ASTNode *items[] = {&main_function};
    ASTNode root = {.type = AST_PROGRAM}; root.as.program.items = items; root.as.program.count = 1;
    CBOptions options = {0};

    /* I refuse an unresolved FLOAT producer without publishing partial source. */
    ASTNode call = {.type = AST_CALL}; call.as.call.name = "missing_result";
    main_function.as.function.name = "ordinary_float";
    main_function.as.function.return_type = TYPE_FLOAT;
    result.as.return_stmt.value = &call;
    previous(argv[1]); assert(c_backend_emit(&root, argv[1], "ordinary.nano", &options) != 0); retained(argv[1]);
    FILE *stream = tmpfile(); assert(stream); assert(fputs("previous", stream) >= 0);
    assert(c_backend_emit_fp(&root, stream, "ordinary.nano", &options) != 0);
    assert(ftell(stream) == 8); assert(fclose(stream) == 0);

    /* I keep a checked indirect callee's result separate from its spelling. */
    CBCtx context = {0}; context.root = &root;
    assert(infer_expr_type(&context, &call) == TYPE_UNKNOWN);
    call.as.call.name = "ordinary_float";
    assert(infer_expr_type(&context, &call) == TYPE_FLOAT);
    ctx_add_sym(&context, "ordinary_float", TYPE_FUNCTION);
    assert(infer_expr_type(&context, &call) == TYPE_UNKNOWN);
    FunctionSignature signature = {0}; signature.return_type = TYPE_FLOAT;
    call.as.call.checked_signature = &signature;
    assert(infer_expr_type(&context, &call) == TYPE_FLOAT);
    ASTNode field = {.type = AST_FIELD_ACCESS};
    assert(infer_expr_type(&context, &field) == TYPE_UNKNOWN);
    TypeInfo field_type = {.base_type = TYPE_FLOAT};
    field.as.field_access.resolved_type_info = &field_type;
    assert(infer_expr_type(&context, &field) == TYPE_FLOAT);
    field_type.base_type = TYPE_INT;
    assert(infer_expr_type(&context, &field) == TYPE_INT);
    field_type.base_type = TYPE_UNKNOWN;
    assert(infer_expr_type(&context, &field) == TYPE_UNKNOWN);
    /* I keep absent annotations tied to their exact declaration and scope. */
    ASTNode declaration = {.type = AST_STRUCT_DEF};
    char *field_names[] = {"value"}; Type field_types[] = {TYPE_FLOAT};
    declaration.as.struct_def.name = "Measure";
    declaration.as.struct_def.field_count = 1;
    declaration.as.struct_def.field_names = field_names;
    declaration.as.struct_def.field_types = field_types;
    ASTNode *declarations[] = {&declaration};
    ASTNode field_root = {.type = AST_PROGRAM};
    field_root.as.program.items = declarations; field_root.as.program.count = 1;
    CBCtx fields = {0}; fields.root = &field_root;
    ctx_add_nominal(&fields, "m", TYPE_STRUCT, "Measure", NULL);
    ASTNode object = {.type = AST_IDENTIFIER}; object.as.identifier = "m";
    field.as.field_access.object = &object;
    field.as.field_access.field_name = "value";
    field.as.field_access.resolved_type_info = NULL;
    assert(infer_expr_type(&fields, &field) == TYPE_FLOAT);
    ctx_push_scope(&fields);
    ctx_add_nominal(&fields, "m", TYPE_STRUCT, "Unrelated", NULL);
    assert(infer_expr_type(&fields, &field) == TYPE_UNKNOWN);
    ctx_pop_scope(&fields);
    assert(infer_expr_type(&fields, &field) == TYPE_FLOAT);
    field.as.field_access.field_name = "absent";
    assert(infer_expr_type(&fields, &field) == TYPE_UNKNOWN);
    field.as.field_access.field_name = "value";
    field.as.field_access.resolved_type_info = &field_type;
    assert(infer_expr_type(&fields, &field) == TYPE_UNKNOWN);
    ASTNode union_declaration = {.type = AST_UNION_DEF};
    char *variant_names[] = {"Reading"}; int counts[] = {1};
    char **variant_fields[] = {field_names}; Type *variant_types[] = {field_types};
    char **absent_name_rows[] = {NULL};
    union_declaration.as.union_def.name = "Sample";
    union_declaration.as.union_def.variant_count = 1;
    union_declaration.as.union_def.variant_names = variant_names;
    union_declaration.as.union_def.variant_field_counts = counts;
    union_declaration.as.union_def.variant_field_names = variant_fields;
    union_declaration.as.union_def.variant_field_types = variant_types;
    union_declaration.as.union_def.variant_field_type_names = absent_name_rows;
    declarations[0] = &union_declaration;
    ctx_add_nominal(&fields, "m", TYPE_STRUCT, "Sample", "Reading");
    field.as.field_access.resolved_type_info = NULL;
    assert(infer_expr_type(&fields, &field) == TYPE_FLOAT);
    union_declaration.as.union_def.generic_param_count = 1;
    assert(infer_expr_type(&fields, &field) == TYPE_UNKNOWN);
    ctx_error(&context, "first"); ctx_error(&context, "second");
    assert(strcmp(context.error, "first") == 0);

    /* I recover independently, including every allocator failure in path staging. */
    main_function.as.function.name = "main"; main_function.as.function.return_type = TYPE_INT;
    result.as.return_stmt.value = &zero;
    allocation_index = 0; fail_at = 0;
    assert(c_backend_emit(&root, argv[1], "ordinary.nano", &options) == 0);
    size_t allocations = allocation_index;
    assert(allocations >= 2);
    for (size_t failure = 1; failure <= allocations; ++failure) {
        previous(argv[1]); allocation_index = 0; fail_at = failure;
        assert(c_backend_emit(&root, argv[1], "ordinary.nano", &options) != 0);
        fail_at = 0; retained(argv[1]);
        assert(c_backend_emit(&root, argv[1], "ordinary.nano", &options) == 0);
    }
    stream = tmpfile(); assert(stream);
    assert(c_backend_emit_fp(&root, stream, "ordinary.nano", &options) == 0);
    assert(ftell(stream) > 8); assert(fclose(stream) == 0);
    puts("I retained output and recovered after semantic/allocation refusals.");
    return 0;
}
