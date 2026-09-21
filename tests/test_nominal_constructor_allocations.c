/* I isolate the actual nominal binder's allocation domain. Other providers and
 * parser setup remain ordinary; transferred results use the real AST destructor. */
#define _POSIX_C_SOURCE 200809L
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#define _DARWIN_C_SOURCE
#include "../src/nanolang.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef NDEBUG
#error I require active constructor assertions.
#endif
static bool observing, transient;
static size_t attempts, failures, denied_at, live, released_original;
static void *owned[8];
static void *original_name;
static void *constructor_malloc(size_t bytes) {
    if (observing) {
        size_t index = attempts++;
        if (transient ? index == denied_at : index >= denied_at) { ++failures; return NULL; }
    }
    void *p = malloc(bytes);
    if (observing && p) {
        assert(live < sizeof owned / sizeof *owned);
        size_t slot = 0; while (owned[slot]) ++slot;
        owned[slot] = p; ++live;
    }
    return p;
}
static char *constructor_strdup(const char *text) {
    size_t bytes = strlen(text) + 1;
    char *p = constructor_malloc(bytes);
    if (p) memcpy(p, text, bytes);
    return p;
}
static void constructor_free(void *p) {
    if (p && p == original_name) { ++released_original; original_name = NULL; }
    for (size_t i = 0; p && i < sizeof owned / sizeof *owned; ++i)
        if (owned[i] == p) { owned[i] = NULL; assert(live); --live; break; }
    free(p);
}
#define malloc constructor_malloc
#define strdup constructor_strdup
#define free constructor_free
#define bind_nominal_records fixture_bind_nominal_records
#include "../src/nominal_types.c"
#undef bind_nominal_records
#undef malloc
#undef strdup
#undef free

static ASTNode *constructor_literal(const char *name) {
    ASTNode *node = calloc(1, sizeof *node); assert(node);
    node->type = AST_STRUCT_LITERAL;
    node->as.struct_literal.struct_name = strdup(name);
    node->as.struct_literal.field_count = 1;
    node->as.struct_literal.field_names = calloc(1, sizeof(char *));
    node->as.struct_literal.field_values = calloc(1, sizeof(ASTNode *));
    assert(node->as.struct_literal.struct_name && node->as.struct_literal.field_names && node->as.struct_literal.field_values);
    node->as.struct_literal.field_names[0] = strdup("value");
    node->as.struct_literal.field_values[0] = calloc(1, sizeof(ASTNode));
    assert(node->as.struct_literal.field_names[0] && node->as.struct_literal.field_values[0]);
    node->as.struct_literal.field_values[0]->type = AST_NUMBER;
    return node;
}
static void constructor_foreign(Environment *env, bool exported) {
    UnionDef definition = {0};
    definition.name = strdup("Box"); definition.module_name = strdup("Foreign");
    definition.variant_count = 1; definition.variant_names = calloc(1, sizeof(char *));
    assert(definition.name && definition.module_name && definition.variant_names);
    definition.variant_names[0] = strdup("Value"); assert(definition.variant_names[0]);
    env_define_union(env, definition);
    char **names = exported ? calloc(1, sizeof *names) : NULL;
    if (exported) { assert(names); names[0] = strdup("Box"); assert(names[0]); }
    env_register_namespace(env, "alias", "Foreign", NULL, 0, NULL, 0, NULL, 0, names, exported ? 1 : 0);
}
static size_t constructor_attempt(bool imported, size_t prefix, bool once) {
    const char *source = imported ? "fn main()->int{return 0}" : "union Box<T>{Value{value:T}}";
    int token_count = 0; Token *tokens = tokenize(source, &token_count); assert(tokens);
    ASTNode *program = parse_program(tokens, token_count); assert(program);
    Environment *env = create_environment(); assert(env); env->current_module = "Caller";
    if (imported) constructor_foreign(env, true);
    ASTNode *node = constructor_literal(imported ? "alias.Box.Value" : "Box.Value");
    ASTNode saved = *node;
    ASTNode *child = node->as.struct_literal.field_values[0];
    char *field = node->as.struct_literal.field_names[0];
    assert(!live); attempts = failures = released_original = 0;
    denied_at = prefix; transient = once; original_name = node->as.struct_literal.struct_name;
    observing = true;
    bool ok = nominal_literal_constructor(program, env, node);
    observing = false;
    size_t measured = attempts;
    if (prefix == SIZE_MAX) {
        assert(ok && !failures && attempts == 2 && released_original == 1 && !original_name);
        assert(node->type == AST_UNION_CONSTRUCT && !node->as.union_construct.type_info);
        assert(!strcmp(node->as.union_construct.union_name, imported ? "alias.Box" : "Box"));
        assert(!strcmp(node->as.union_construct.variant_name, "Value"));
        assert(node->as.union_construct.field_names == saved.as.struct_literal.field_names);
        assert(node->as.union_construct.field_values == saved.as.struct_literal.field_values);
        assert(node->as.union_construct.field_names[0] == field && node->as.union_construct.field_values[0] == child);
        assert(node->as.union_construct.field_count == 1 && live == 2);
        /* I transfer tracking to the real owning AST; no hooked-destructor claim. */
        bool prefix_owned = false, variant_owned = false;
        for (size_t i = 0; i < sizeof owned / sizeof *owned; ++i) {
            if (owned[i] == node->as.union_construct.union_name) prefix_owned = true;
            if (owned[i] == node->as.union_construct.variant_name) variant_owned = true;
            owned[i] = NULL;
        }
        assert(prefix_owned && variant_owned); live = 0;
    } else {
        assert(!ok && failures == 1 && attempts == prefix + 1 && !released_original && !live);
        assert(!memcmp(node, &saved, sizeof saved));
        assert(node->as.struct_literal.field_values[0] == child && node->as.struct_literal.field_names[0] == field);
        original_name = NULL;
    }
    free_ast(node); free_environment(env); free_ast(program); free_tokens(tokens, token_count);
    return measured;
}
static void constructor_refusals(void) {
    const char *sources[] = {"union Box{Value{value:int}}", "union Box{Value{value:int}}", "union Box{Value{value:int}} struct Box{value:int}", "union Box{Value{value:int},Value{value:int}}"};
    for (size_t i = 0; i < sizeof sources / sizeof *sources; ++i) {
        int count = 0; Token *tokens = tokenize(sources[i], &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program);
        Environment *env = create_environment(); assert(env);
        ASTNode *node = constructor_literal(i == 0 ? "Box.Missing" : "Box.Value");
        if (i == 1) { node->as.struct_literal.spread_source = calloc(1, sizeof(ASTNode)); assert(node->as.struct_literal.spread_source); node->as.struct_literal.spread_source->type = AST_NUMBER; }
        ASTNode saved = *node;
        assert(!nominal_literal_constructor(program, env, node));
        assert(!memcmp(node, &saved, sizeof saved));
        free_ast(node); free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
    for (int which = 0; which < 3; ++which) {
        Environment *env = create_environment(); assert(env); env->current_module = "Caller";
        constructor_foreign(env, which != 0);
        if (which == 1) env->current_module = "Other";
        ASTNode *node = constructor_literal(which == 2 ? "unknown.Box.Value" : "alias.Box.Value");
        ASTNode *items[] = {node}; ASTNode program = {.type = AST_PROGRAM};
        program.as.program.items = items; program.as.program.count = 1;
        ASTNode saved = *node;
        assert(!fixture_bind_nominal_records(&program, env));
        assert(!memcmp(node, &saved, sizeof saved));
        free_ast(node); free_environment(env);
    }
}
static void constructor_parsed_imports(void) {
    for (int record = 0; record < 2; ++record) {
        const char *source = record
            ? "fn sample()->int { let value =alias.Record{value:3} return 0 }"
            : "fn sample()->int { let value =alias.Box.Value{value:3} return 0 }";
        int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
        ASTNode *program = parse_program(tokens, count); assert(program && program->as.program.count == 1);
        ASTNode *node = program->as.program.items[0]->as.function.body->as.block.statements[0]->as.let.value;
        assert(node && node->type == AST_STRUCT_LITERAL);
        char **names = node->as.struct_literal.field_names;
        ASTNode **values = node->as.struct_literal.field_values;
        Environment *env = create_environment(); assert(env); env->current_module = "Caller";
        if (record) {
            StructDef definition = {0}; definition.name = strdup("Record"); definition.module_name = "Foreign";
            assert(definition.name); env_define_struct(env, definition);
            char **exports = calloc(1, sizeof *exports); assert(exports);
            exports[0] = strdup("Record"); assert(exports[0]);
            env_register_namespace(env, "alias", "Foreign", NULL, 0, exports, 1, NULL, 0, NULL, 0);
        } else constructor_foreign(env, true);
        assert(fixture_bind_nominal_records(program, env));
        if (record) {
            assert(node->type == AST_STRUCT_LITERAL && !strcmp(node->as.struct_literal.struct_name, "Record"));
            assert(node->as.struct_literal.field_names == names && node->as.struct_literal.field_values == values);
        } else {
            assert(node->type == AST_UNION_CONSTRUCT && !strcmp(node->as.union_construct.variant_name, "Value"));
            assert(node->as.union_construct.field_names == names && node->as.union_construct.field_values == values);
            NominalIdentity identity = env_nominal_identity(env, node->as.union_construct.union_name, "Caller", TYPE_UNION);
            assert(identity.kind == TYPE_UNION && identity.ordinal == 1);
        }
        free_environment(env); free_ast(program); free_tokens(tokens, count);
    }
}
void test_nominal_constructor_allocations(void) {
    for (int imported = 0; imported < 2; ++imported) {
        size_t total = constructor_attempt(imported != 0, SIZE_MAX, false);
        for (int once = 0; once < 2; ++once) for (size_t i = 0; i < total; ++i) {
            constructor_attempt(imported != 0, i, once != 0);
            assert(constructor_attempt(imported != 0, SIZE_MAX, false) == total);
        }
    }
    constructor_refusals();
    constructor_parsed_imports();
    puts("I checked exact union constructor ownership and both allocation prefixes.");
}
