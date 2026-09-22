/* I exercise the real checker adapter below the public scalar-borrow admission guard. */
#include "nanolang.h"
#include "runtime/gc.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int g_argc;
char **g_argv;
static bool refuse_result_binding;
static bool split_context_bind(Environment *env, const ASTNode *node, const TypeInfo *info) {
    if (refuse_result_binding) return false;
    return env_bind_array_expression(env, node, info);
}
#define env_bind_array_expression split_context_bind
#include "../src/typechecker.c"
#undef env_bind_array_expression

static void check_context(const char *element, bool refuse) {
    char source[512];
    int n = snprintf(source, sizeof source,
        "resource struct Words { items: array<%s> }\n"
        "fn main() -> int { return 0 }\nshadow main { assert true }\n", element);
    assert(n > 0 && (size_t)n < sizeof source);
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens && count > 0);
    ASTNode *program = parse_program(tokens, count);
    assert(program);
    Environment *env = create_environment();
    assert(env && type_check(program, env));
    env_define_var(env, "view", TYPE_BORROW_MUT, false, create_void());
    Symbol *view = env_get_var(env, "view");
    assert(view);
    view->struct_type_name = strdup("Words");
    assert(view->struct_type_name);
    const char *body = "fn probe() -> void { set view.items (str_split \"a,b\" \",\") }";
    int body_count = 0;
    Token *body_tokens = tokenize(body, &body_count);
    assert(body_tokens);
    ASTNode *body_program = parse_program(body_tokens, body_count);
    assert(body_program && body_program->as.program.count == 1);
    ASTNode *function = body_program->as.program.items[0];
    assert(function->type == AST_FUNCTION && function->as.function.body);
    ASTNode *block = function->as.function.body;
    assert(block->type == AST_BLOCK && block->as.block.count == 1);
    ASTNode *assignment = block->as.block.statements[0];
    assert(assignment->type == AST_SET);
    TypeChecker checker = {.env = env, .current_function_return_type = TYPE_VOID};
    int errors = g_typecheck_error_count;
    size_t rows = env->array_expression_binding_count;
    refuse_result_binding = refuse;
    (void)check_statement(&checker, assignment);
    refuse_result_binding = false;
    if (refuse) {
        assert(env->opaque_resolution_failed);
        assert(env->array_expression_binding_count == rows);
        assert(!env_array_expression_info(env, assignment->as.set.value));
    } else {
        assert(!env->opaque_resolution_failed);
        const TypeInfo *result = env_array_expression_info(env, assignment->as.set.value);
        assert(result && result->base_type == TYPE_ARRAY && result->element_type);
        assert(result->element_type->base_type == TYPE_STRING);
        assert(env->array_expression_binding_count == rows + 1);
        assert((g_typecheck_error_count > errors) == (strcmp(element, "string") != 0));
        if (!strcmp(element, "string")) {
            assert(!checker.has_error);
            (void)check_statement(&checker, assignment);
            assert(g_typecheck_error_count == errors);
            assert(env->array_expression_binding_count == rows + 1);
        }
    }
    free_environment(env);
    free_ast(body_program); free_tokens(body_tokens, body_count);
    free_ast(program); free_tokens(tokens, count);
}
/* I check the contextual helper directly, without publishing invalid code. */
static void check_literal_context(Type expected_element, const char *literal, bool mismatch) {
    char source[512];
    int n = snprintf(source, sizeof source,
        "fn probe() -> void { let values = %s }", literal);
    assert(n > 0 && (size_t)n < sizeof source);
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program && program->as.program.count == 1);
    ASTNode *function = program->as.program.items[0];
    assert(function->type == AST_FUNCTION && function->as.function.body);
    ASTNode *body = function->as.function.body;
    assert(body->type == AST_BLOCK && body->as.block.count == 1);
    ASTNode *declaration = body->as.block.statements[0];
    assert(declaration->type == AST_LET);
    ASTNode *value = declaration->as.let.value;
    assert(value && value->type == AST_ARRAY_LITERAL);
    Environment *env = create_environment();
    assert(env && check_expression(value, env) == TYPE_ARRAY);
    TypeInfo element = {.base_type = expected_element};
    TypeInfo expected = {.base_type = TYPE_ARRAY, .element_type = &element};
    int errors = g_typecheck_error_count;
    check_concrete_union_arrays(env, &expected, value, 0);
    assert((g_typecheck_error_count > errors) == mismatch);
    assert(!env->opaque_resolution_failed);
    free_environment(env);
    free_ast(program);
    free_tokens(tokens, count);
}
int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    check_context("string", false);
    check_context("int", false);
    check_context("string", true);
    check_literal_context(TYPE_STRING, "[41]", true);
    check_literal_context(TYPE_INT, "[\"x\"]", true);
    check_literal_context(TYPE_STRING, "[\"x\"]", false);
    check_literal_context(TYPE_STRING, "[]", false);
    check_literal_context(TYPE_INT, "[]", false);
    check_literal_context(TYPE_FLOAT, "[41]", false);
    gc_shutdown();
    puts("I checked the string-array field adapter and binding-refusal publication boundary.");
    return 0;
}
