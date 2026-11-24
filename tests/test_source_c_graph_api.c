/* I retain exact borrowed graph members, owners and prior output. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t allocations, denied;
static void *fixture_calloc(size_t count, size_t size) {
    allocations++;
    return allocations == denied ? NULL : calloc(count, size);
}
#define calloc fixture_calloc
#include "../src/c_backend.c"
#undef calloc
int *passive_binding_order(const ASTNode *block) {
    (void)block; assert(!"I reached an unrelated passive fixture."); return NULL;
}
typedef struct { ASTNode *root, *entry, *answer, *foreign; } Lookup;
static ASTNode *resolve(void *opaque, ASTNode *owner, const char *name) {
    Lookup *lookup = opaque;
    if (owner == lookup->root && !strcmp(name, "main")) return lookup->entry;
    if (!strcmp(name, "lib.answer")) return lookup->foreign ? lookup->foreign : lookup->answer;
    return NULL;
}
static void previous(const char *path) {
    FILE *file = fopen(path, "w"); assert(file);
    assert(fputs("previous", file) >= 0); assert(!fclose(file));
}
static void retained(const char *path) {
    char text[16] = {0}; FILE *file = fopen(path, "r"); assert(file);
    assert(fread(text, 1, sizeof text, file) == 8); assert(!strcmp(text, "previous"));
    assert(!fclose(file));
}
int main(int argc, char **argv) {
    assert(argc == 2);
    ASTNode first = {.type = AST_IDENTIFIER}; first.as.identifier = "first";
    ASTNode second = {.type = AST_IDENTIFIER}; second.as.identifier = "second";
    ASTNode ten = {.type = AST_NUMBER}; ten.as.number = 10;
    ASTNode *multiply_args[] = {&first, &ten};
    ASTNode multiply = {.type = AST_PREFIX_OP}; multiply.as.prefix_op.op = TOKEN_STAR; multiply.as.prefix_op.args = multiply_args; multiply.as.prefix_op.arg_count = 2;
    ASTNode *sum_args[] = {&multiply, &second};
    ASTNode sum = {.type = AST_PREFIX_OP}; sum.as.prefix_op.op = TOKEN_PLUS; sum.as.prefix_op.args = sum_args; sum.as.prefix_op.arg_count = 2;
    ASTNode result = {.type = AST_RETURN}; result.as.return_stmt.value = &sum;
    ASTNode *body_items[] = {&result};
    ASTNode body = {.type = AST_BLOCK}; body.as.block.statements = body_items; body.as.block.count = 1;
    ASTNode answer = {.type = AST_FUNCTION}; answer.as.function.name = "answer";
    answer.as.function.return_type = TYPE_INT; answer.as.function.body = &body;
    Parameter parameters[] = {{.name="first", .type=TYPE_INT}, {.name="second", .type=TYPE_INT}};
    answer.as.function.params = parameters; answer.as.function.param_count = 2;
    ASTNode *leaf_items[] = {&answer};
    ASTNode leaf = {.type = AST_PROGRAM}; leaf.as.program.items = leaf_items; leaf.as.program.count = 1;
    ASTNode call = {.type = AST_MODULE_QUALIFIED_CALL};
    call.as.module_qualified_call.module_alias = "lib"; call.as.module_qualified_call.function_name = "answer";
    ASTNode four = {.type = AST_NUMBER}; four.as.number = 4;
    ASTNode two = {.type = AST_NUMBER}; two.as.number = 2;
    ASTNode *arguments[] = {&four, &two};
    call.as.module_qualified_call.args = arguments; call.as.module_qualified_call.arg_count = 2;
    Type parameter_types[] = {TYPE_INT, TYPE_INT};
    FunctionSignature signature = {0}; signature.return_type = TYPE_INT;
    signature.param_types = parameter_types; signature.param_count = 2;
    call.as.module_qualified_call.checked_signature = &signature;
    ASTNode root_result = {.type = AST_RETURN}; root_result.as.return_stmt.value = &call;
    ASTNode *root_body_items[] = {&root_result};
    ASTNode root_body = {.type = AST_BLOCK}; root_body.as.block.statements = root_body_items; root_body.as.block.count = 1;
    ASTNode entry = {.type = AST_FUNCTION}; entry.as.function.name = "main";
    entry.as.function.return_type = TYPE_INT; entry.as.function.body = &root_body;
    ASTNode *root_items[] = {&entry};
    ASTNode root = {.type = AST_PROGRAM}; root.as.program.items = root_items; root.as.program.count = 1;
    ASTNode *items[] = {&answer, &entry};
    ASTNode merged = {.type = AST_PROGRAM}; merged.as.program.items = items; merged.as.program.count = 2;
    CBSourceDeclaration declarations[] = {{&answer, &leaf}, {&entry, &root}};
    Lookup lookup = {&root, &entry, &answer, NULL};
    CBSourceGraph graph = {&root, declarations, 2, &lookup, resolve};
    CBOptions options = {0};
    unsigned char before_answer[sizeof answer], before_entry[sizeof entry], before_call[sizeof call];
    memcpy(before_answer, &answer, sizeof answer);
    memcpy(before_entry, &entry, sizeof entry);
    memcpy(before_call, &call, sizeof call);
    previous(argv[1]);
    assert(c_backend_emit(&root, argv[1], "root.nano", &options)); retained(argv[1]);
    CBSourceGraph invalid = graph; invalid.count = 1;
    assert(c_backend_emit_graph(&merged, &invalid, argv[1], "root.nano", &options)); retained(argv[1]);
    declarations[0].owner = &root;
    assert(c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options)); retained(argv[1]);
    declarations[0].owner = &leaf;
    ASTNode foreign = answer; lookup.foreign = &foreign;
    assert(c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options)); retained(argv[1]);
    lookup.foreign = NULL;
    ASTNode omitted = answer; omitted.as.function.name = "omitted";
    ASTNode *incomplete_items[] = {&answer, &omitted};
    leaf.as.program.items = incomplete_items; leaf.as.program.count = 2;
    assert(c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options)); retained(argv[1]);
    leaf.as.program.items = leaf_items; leaf.as.program.count = 1;
    ASTNode unknown_global = {.type = AST_LET}; unknown_global.as.let.name = "missing"; unknown_global.as.let.var_type = TYPE_INT;
    ASTNode global_reference = {.type = AST_IDENTIFIER}; global_reference.as.identifier = "missing"; global_reference.checked_global = &unknown_global;
    root_result.as.return_stmt.value = &global_reference;
    assert(c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options)); retained(argv[1]);
    root_result.as.return_stmt.value = &call;
    allocations = 0;
    assert(!c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options));
    size_t positions = allocations; assert(positions > 0);
    for (denied = 1; denied <= positions; denied++) {
        previous(argv[1]); allocations = 0;
        assert(c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options)); retained(argv[1]);
    }
    denied = 0;
    assert(!c_backend_emit_graph(&merged, &graph, argv[1], "root.nano", &options));
    printf("I preserve output through %zu allocation failures.\n", positions);
    assert(!memcmp(&answer, before_answer, sizeof answer));
    assert(!memcmp(&entry, before_entry, sizeof entry));
    assert(!memcmp(&call, before_call, sizeof call));
    assert(declarations[0].declaration == &answer && declarations[0].owner == &leaf);
    return 0;
}
