/* I test ownership storage growth and fail-closed allocation boundaries. */
#include "resource_tracking.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static size_t allocations, attempts, fail_at;
static void *checked_malloc(size_t size) {
    if (++attempts == fail_at) return NULL;
    void *result = malloc(size);
    if (result) allocations++;
    return result;
}
static void *checked_realloc(void *old, size_t size) {
    if (++attempts == fail_at) return NULL;
    bool fresh = old == NULL;
    void *result = realloc(old, size);
    if (result && fresh) allocations++;
    return result;
}
static void checked_free(void *value) {
    if (value) { assert(allocations > 0); allocations--; }
    free(value);
}

bool is_resource_type(Environment *env, const char *name) {
    (void)env;
    return name && !strcmp(name, "Handle");
}
Function *env_get_function(Environment *env, const char *name) {
    (void)env; (void)name; return NULL;
}
StructDef *env_get_struct(Environment *env, const char *name) {
    (void)env; (void)name; return NULL;
}

#define malloc checked_malloc
#define realloc checked_realloc
#define free checked_free
#include "../src/resource_flow.c"
#undef malloc
#undef realloc
#undef free

int main(void) {
    enum { COUNT = 300 };
    char names[COUNT][24];
    ASTNode values[COUNT] = {0}, bindings[COUNT] = {0};
    ASTNode identifiers[COUNT] = {0}, calls[COUNT] = {0};
    ASTNode *arguments[COUNT], *statements[COUNT * 2 + 1];
    for (int i = 0; i < COUNT; ++i) {
        snprintf(names[i], sizeof(names[i]), "owner_%d", i);
        values[i].type = AST_STRUCT_LITERAL;
        values[i].as.struct_literal.struct_name = "Handle";
        bindings[i].type = AST_LET;
        bindings[i].as.let.name = names[i];
        bindings[i].as.let.type_name = "Handle";
        bindings[i].as.let.value = &values[i];
        identifiers[i].type = AST_IDENTIFIER;
        identifiers[i].as.identifier = names[i];
        arguments[i] = &identifiers[i];
        calls[i].type = AST_CALL;
        calls[i].as.call.name = "consume";
        calls[i].as.call.args = &arguments[i];
        calls[i].as.call.arg_count = 1;
        statements[i * 2] = &bindings[i];
        statements[i * 2 + 1] = &calls[i];
    }
    ASTNode condition = {.type = AST_BOOL}, empty = {.type = AST_BLOCK};
    ASTNode branch = {.type = AST_IF};
    condition.as.bool_val = true;
    branch.as.if_stmt.condition = &condition;
    branch.as.if_stmt.then_branch = &empty;
    branch.as.if_stmt.else_branch = &empty;
    statements[COUNT * 2] = &branch;
    ASTNode body = {.type = AST_BLOCK};
    body.as.block.statements = statements;
    body.as.block.count = COUNT * 2 + 1;
    ASTNode function = {.type = AST_FUNCTION};
    function.as.function.body = &body;
    StructDef record = {.name = "Handle", .is_resource = true};
    Environment env = {.structs = &record, .struct_count = 1};
    bool error = false;
    check_function_ownership(&env, &function, &error);
    assert(!error);
    assert(allocations == 0);
    size_t sites = attempts;
    assert(sites > 1);
    for (size_t failure = 1; failure <= sites; ++failure) {
        attempts = 0;
        fail_at = failure;
        error = false;
        check_function_ownership(&env, &function, &error);
        assert(error);
        assert(allocations == 0);
    }
    fail_at = 0;
    body.as.block.count = COUNT * 2 - 1;
    error = false;
    check_function_ownership(&env, &function, &error);
    assert(error);
    assert(allocations == 0);
    OwnFlow overflow = {.env = &env, .count = SIZE_MAX, .capacity = SIZE_MAX, .error = &error};
    error = false;
    assert(!own_add(&overflow, &function, "overflow", "Handle"));
    assert(error);
    assert(allocations == 0);
    printf("I checked 300 owners and %zu allocation-failure positions.\n", sites);
    return 0;
}
