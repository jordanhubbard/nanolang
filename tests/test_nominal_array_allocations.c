/* I observe actual env/checker allocations only; setup and other providers
 * remain outside this independent domain. Legacy fatal signature copies are
 * not recoverable-prefix acceptance. */
#define _POSIX_C_SOURCE 200809L
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#define _DARWIN_C_SOURCE
#include "../src/nanolang.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t checks, attempts, failed, at, live;
static bool observing, once;
static void *pointers[4096];
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "I failed %s at %d\n", #x, __LINE__); exit(90); } } while (0)
static bool failure(void) {
    if (!observing) return false;
    size_t index = attempts++;
    if (once ? index == at : index >= at) { ++failed; return true; }
    return false;
}
static void remember(void *p) {
    if (!p || !observing) return;
    for (size_t i = 0; i < sizeof pointers / sizeof *pointers; ++i)
        if (!pointers[i]) { pointers[i] = p; ++live; return; }
    CHECK(false);
}
void *array_alloc_malloc(size_t size) {
    if (failure()) return NULL;
    void *p = malloc(size); remember(p); return p;
}
void *array_alloc_calloc(size_t count, size_t size) {
    if (failure()) return NULL;
    void *p = calloc(count, size); remember(p); return p;
}
void *array_alloc_realloc(void *p, size_t size) {
    if (failure()) return NULL;
    size_t slot = SIZE_MAX;
    for (size_t i = 0; i < sizeof pointers / sizeof *pointers; ++i)
        if (p && pointers[i] == p) { slot = i; break; }
    void *next = realloc(p, size);
    if (next) { if (slot != SIZE_MAX) pointers[slot] = next; else remember(next); }
    return next;
}
char *array_alloc_strdup(const char *s) {
    size_t bytes = strlen(s) + 1;
    char *p = array_alloc_malloc(bytes); if (p) memcpy(p, s, bytes); return p;
}
void array_alloc_free(void *p) {
    for (size_t i = 0; p && i < sizeof pointers / sizeof *pointers; ++i)
        if (pointers[i] == p) { pointers[i] = NULL; CHECK(live); --live; break; }
    free(p);
}
#define malloc array_alloc_malloc
#define calloc array_alloc_calloc
#define realloc array_alloc_realloc
#define strdup array_alloc_strdup
#define free array_alloc_free
#include "../src/env.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
extern bool array_test_view(Environment *, ASTNode *, unsigned, TypeInfo **, const char **);
static void begin(size_t prefix, bool transient) {
    CHECK(!observing && !live); observing = true; attempts = failed = 0;
    at = prefix; once = transient;
}
static size_t stop(void) { observing = false; return attempts; }

static size_t copy_attempt(size_t prefix, bool transient) {
    char name[] = "Item"; Type tags[] = {TYPE_STRUCT}; char *names[] = {name};
    TypeInfo leaf = {.base_type = TYPE_STRUCT, .generic_name = name};
    TypeInfo *params[] = {&leaf};
    FunctionSignature signature = {.param_count = 1, .param_types = tags,
        .param_struct_names = names, .param_type_info = params, .return_type = TYPE_STRUCT,
        .return_struct_name = name, .return_type_info = &leaf};
    TypeInfo root = {.base_type = TYPE_ARRAY, .element_type = &leaf, .type_params = params,
        .type_param_count = 1, .tuple_types = tags, .tuple_type_names = names, .tuple_element_count = 1,
        .row_field_names = names, .row_field_types = tags, .row_field_type_names = names,
        .row_field_count = 1, .fn_sig = &signature};
    TypeInfo *out = &root;
    begin(prefix, transient); bool ok = copy_payload_type_info_checked(&root, &out); size_t count = stop();
    if (prefix != SIZE_MAX) {
        CHECK(!ok && failed && out == &root);
        CHECK(!strcmp(name, "Item") && root.element_type == &leaf && signature.param_type_info[0] == &leaf);
    }
    else {
        CHECK(ok && out != &root && out->element_type != &leaf);
        name[0] = 'X';
        CHECK(!strcmp(out->element_type->generic_name, "Item"));
        CHECK(!strcmp(out->fn_sig->param_type_info[0]->generic_name, "Item"));
        CHECK(!strcmp(out->tuple_type_names[0], "Item"));
        CHECK(!strcmp(out->row_field_type_names[0], "Item"));
        free_payload_type_info(out);
    }
    CHECK(!live); return count;
}
static void registration_controls(void) {
    for (int transient = 0; transient < 2; ++transient) {
        Environment *env = create_environment(); CHECK(env);
        TypeInfo leaf = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
        TypeInfo source = {.base_type = TYPE_ARRAY, .element_type = &leaf};
        TypeInfo *owned = NULL;
        begin(SIZE_MAX, false); CHECK(copy_payload_type_info_checked(&source, &owned)); stop();
        size_t retained = live; CHECK(retained >= 3);
        struct EnvCheckerAllocation *before = env->checker_allocations;
        /* I fail only the registry node, keeping the already-owned tree alive. */
        attempts = failed = 0; at = 0; once = transient != 0; observing = true;
        CHECK(!env_own_checker_type_info(env, owned)); stop();
        CHECK(attempts == 1 && failed == 1 && live == retained && env->checker_allocations == before);
        CHECK(!strcmp(owned->element_type->generic_name, "Item"));
        attempts = failed = 0; at = SIZE_MAX; observing = true;
        CHECK(env_own_checker_type_info(env, owned)); stop();
        CHECK(attempts == 1 && !failed && live == retained + 1);
        CHECK(env->checker_allocations->allocation == owned && env->checker_allocations->owned_type_info);
        free_environment(env); CHECK(!live);
    }
    TypeInfo borrowed = {.base_type = TYPE_INT};
    CHECK(!env_own_checker_type_info(NULL, &borrowed));
    CHECK(borrowed.base_type == TYPE_INT);
    CHECK(!env_own_checker_type_info(NULL, NULL));
}
static size_t view_attempt(Environment *env, ASTNode *expression, size_t prefix, bool transient, unsigned expected_depth) {
    TypeInfo sentinel = {.base_type = TYPE_BOOL}, *out = &sentinel;
    const char *owner = "unchanged";
    begin(prefix, transient); bool ok = array_test_view(env, expression, 0, &out, &owner); size_t count = stop();
    if (prefix != SIZE_MAX) CHECK(failed && (!ok || transient));
    if (!ok) CHECK(out == &sentinel && !strcmp(owner, "unchanged"));
    else {
        CHECK(out != &sentinel && owner == NULL);
        TypeInfo *leaf = out;
        unsigned depth = 0;
        while (leaf->base_type == TYPE_ARRAY) { CHECK(leaf->element_type && ++depth <= 2); leaf = leaf->element_type; }
        CHECK(depth == expected_depth);
        CHECK(leaf->base_type == TYPE_STRUCT && leaf->generic_name && !strcmp(leaf->generic_name, "Item"));
        free_payload_type_info(out);
    }
    if (prefix == SIZE_MAX) CHECK(ok && !failed);
    CHECK(!live); return count;
}
static void view_controls(void) {
    Environment *env = create_environment(); CHECK(env);
    StructDef definition = {0}; definition.name = strdup("Item"); CHECK(definition.name);
    env_define_struct(env, definition);
    TypeInfo leaf = {.base_type = TYPE_STRUCT, .generic_name = "Item"};
    TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = &leaf};
    TypeInfo nested = {.base_type = TYPE_ARRAY, .element_type = &array};
    env_define_var_with_type_info(env, "values", TYPE_ARRAY, TYPE_STRUCT, &array, false, create_void());
    env_define_var_with_type_info(env, "nested", TYPE_ARRAY, TYPE_ARRAY, &nested, false, create_void());
    env_define_var_with_type_info(env, "item", TYPE_STRUCT, TYPE_UNKNOWN, &leaf, false, create_void());
    ASTNode values = {0}, item = {0}, nested_value = {0}, zero = {0};
    values.type = item.type = nested_value.type = AST_IDENTIFIER;
    values.as.identifier = "values"; item.as.identifier = "item"; nested_value.as.identifier = "nested";
    zero.type = AST_NUMBER;
    ASTNode *elements[] = {&item, &item};
    ASTNode literal = {0}; literal.type = AST_ARRAY_LITERAL;
    literal.as.array_literal.elements = elements; literal.as.array_literal.element_count = 2;
    ASTNode branch = {0}; branch.type = AST_IF;
    branch.as.if_stmt.then_branch = &literal; branch.as.if_stmt.else_branch = &values;
    ASTNode empty = {0}; empty.type = AST_ARRAY_LITERAL;
    ASTNode *arguments[][3] = {{&zero, &item, NULL}, {&values, &item, NULL},
        {&empty, &item, NULL}, {&values, &zero, NULL}, {&nested_value, &zero, NULL}, {&values, &zero, &zero}};
    const char *names[] = {"array_new", "array_push", "array_push", "at", "array_get", "array_slice"};
    ASTNode calls[6] = {{0}};
    for (size_t i = 0; i < 6; ++i) {
        CHECK(env_get_function(env, names[i])); /* I initialize builtin lookup outside observation. */
        calls[i].type = AST_CALL; calls[i].as.call.name = (char *)names[i];
        calls[i].as.call.args = arguments[i]; calls[i].as.call.arg_count = i == 5 ? 3 : 2;
    }
    ASTNode *cases[] = {&values, &nested_value, &literal, &branch,
        &calls[0], &calls[1], &calls[2], &calls[3], &calls[4], &calls[5]};
    const unsigned depths[] = {1, 2, 1, 1, 1, 1, 1, 0, 1, 1};
    for (size_t c = 0; c < sizeof cases / sizeof *cases; ++c) {
        size_t count = view_attempt(env, cases[c], SIZE_MAX, false, depths[c]); CHECK(count > 0);
        printf("I measure view %zu: %zu allocation attempts.\n", c, count);
        for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
            view_attempt(env, cases[c], i, transient != 0, depths[c]);
            CHECK(view_attempt(env, cases[c], SIZE_MAX, false, depths[c]) == count);
        }
    }
    TypeInfo sentinel = {.base_type = TYPE_BOOL}, *out = &sentinel; const char *owner = "unchanged";
    CHECK(!array_test_view(env, &values, 129, &out, &owner) && out == &sentinel);
    free_environment(env); CHECK(!live);
}
extern bool array_test_owned_context(Environment *, Symbol *);
static size_t owned_context_attempt(size_t prefix, bool transient) {
    Environment *env = create_environment(); CHECK(env);
    StructDef definition = {0}; definition.name = strdup("Item"); definition.module_name = "Caller";
    CHECK(definition.name); env_define_struct(env, definition);
    TypeInfo sentinel = {.base_type = TYPE_BOOL};
    Symbol output = {0}; output.type_info = &sentinel; output.nominal_owner = "unchanged";
    begin(prefix, transient); bool ok = array_test_owned_context(env, &output); size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(ok && !failed && output.checker_nominal_view);
    else CHECK(!ok && failed && output.type_info == &sentinel && !output.checker_nominal_view &&
               !strcmp(output.nominal_owner, "unchanged"));
    free_environment(env); CHECK(!live);
    return count;
}
static void owned_context_controls(void) {
    size_t count = owned_context_attempt(SIZE_MAX, false); CHECK(count > 15);
    printf("I measure the owned nominal context: %zu allocation attempts.\n", count);
    for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
        owned_context_attempt(i, transient != 0);
        CHECK(owned_context_attempt(SIZE_MAX, false) == count);
    }
}
extern bool array_test_prepare_callable(Environment *, bool);
extern bool array_test_callable_consumer(Environment *, TypeInfo **);
static size_t callable_context_attempt(size_t prefix, bool transient, bool legacy_return) {
    Environment *env = create_environment(); CHECK(env);
    for (int i = 0; i < 2; ++i) {
        StructDef record = {0}; record.name = strdup("Item"); record.module_name = i ? "Caller" : "Definitions";
        CHECK(record.name); env_define_struct(env, record);
    }
    env_define_var(env, "callback", TYPE_FUNCTION, false, create_void());
    CHECK(array_test_prepare_callable(env, legacy_return));
    Symbol *binding = env_get_var(env, "callback");
    const void *proof = binding->checker_nominal_view;
    TypeInfo *prior = binding->type_info;
    TypeInfo sentinel = {.base_type = TYPE_BOOL}, *output = &sentinel;
    begin(prefix, transient); bool ok = array_test_callable_consumer(env, &output); size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(ok && !failed && output != &sentinel);
    else CHECK(!ok && failed && output == &sentinel);
    CHECK(binding->type_info == prior && binding->checker_nominal_view == proof);
    if (ok) free_payload_type_info(output);
    CHECK(!live);
    free_environment(env); CHECK(!live);
    return count;
}
static void callable_context_controls(void) {
    for (int legacy = 0; legacy < 2; ++legacy) {
        size_t count = callable_context_attempt(SIZE_MAX, false, legacy != 0); CHECK(count > 30);
        printf("I measure retained callable form %d: %zu allocation attempts.\n", legacy, count);
        for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
            callable_context_attempt(i, transient != 0, legacy != 0);
            CHECK(callable_context_attempt(SIZE_MAX, false, legacy != 0) == count);
        }
    }
}

static size_t native_callable_attempt(size_t prefix, bool transient, bool legacy_return) {
    Environment *env = create_environment(); CHECK(env);
    for (int i = 0; i < 2; ++i) {
        StructDef record = {0}; record.name = strdup("Item"); record.module_name = i ? "Caller" : "Definitions";
        CHECK(record.name); env_define_struct(env, record);
    }
    env_define_var(env, "callback", TYPE_FUNCTION, false, create_void());
    CHECK(array_test_prepare_callable(env, legacy_return));
    Symbol *binding = env_get_var(env, "callback");
    const void *proof = binding->checker_nominal_view; TypeInfo *prior = binding->type_info;
    ASTNode identifier = {.type = AST_IDENTIFIER}; identifier.as.identifier = "callback";
    begin(prefix, transient);
    FunctionSignature *copy = checked_callable_signature_copy(&identifier, env);
    size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(copy && !failed && copy != prior->fn_sig && copy->return_fn_sig);
    else CHECK(!copy && failed);
    CHECK(binding->checker_nominal_view == proof && binding->type_info == prior);
    free_function_signature(copy); CHECK(!live);
    free_environment(env); CHECK(!live);
    return count;
}
static void native_callable_controls(void) {
    for (int legacy = 0; legacy < 2; ++legacy) {
        size_t count = native_callable_attempt(SIZE_MAX, false, legacy != 0); CHECK(count > 30);
        printf("I measure native callable form %d: %zu allocation attempts.\n", legacy, count);
        for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
            native_callable_attempt(i, transient != 0, legacy != 0);
            CHECK(native_callable_attempt(SIZE_MAX, false, legacy != 0) == count);
        }
    }
}

extern bool array_test_prepare_tuple_leaves(Environment *);
extern bool array_test_tuple_consumer(Environment *, Symbol *, TypeInfo **);
static size_t tuple_context_attempt(size_t prefix, bool transient, bool publication) {
    Environment *env = create_environment(); CHECK(env);
    for (int i = 0; i < 2; ++i) {
        StructDef record = {0}; record.name = strdup("Item"); record.module_name = i ? "Caller" : "Definitions";
        CHECK(record.name); env_define_struct(env, record);
    }
    env_define_var(env, "callback", TYPE_FUNCTION, false, create_void());
    CHECK(array_test_prepare_callable(env, true) && array_test_prepare_tuple_leaves(env));
    Symbol *binding = env_get_var(env, "callback");
    const void *proof = binding->checker_nominal_view;
    TypeInfo *prior = binding->type_info;
    TypeInfo sentinel = {.base_type = TYPE_BOOL}, *output = &sentinel;
    Symbol published = {.type_info = &sentinel, .nominal_owner = "unchanged"};
    begin(prefix, transient);
    bool ok = array_test_tuple_consumer(env, publication ? &published : NULL, &output);
    size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(ok && !failed && (publication ?
        published.checker_nominal_view != NULL : output != &sentinel));
    else CHECK(!ok && failed && output == &sentinel && published.type_info == &sentinel &&
        !published.checker_nominal_view && !strcmp(published.nominal_owner, "unchanged"));
    CHECK(binding->type_info == prior && binding->checker_nominal_view == proof);
    if (ok && !publication) free_payload_type_info(output);
    /* Failed publication can retain an unpublished concrete copy until teardown. */
    if (!publication) CHECK(!live);
    free_environment(env); CHECK(!live);
    return count;
}
static void tuple_context_controls(void) {
    for (int publication = 0; publication < 2; ++publication) {
        size_t count = tuple_context_attempt(SIZE_MAX, false, publication != 0); CHECK(count > 30);
        printf("I measure composed tuple route %d: %zu allocation attempts.\n", publication, count);
        for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
            tuple_context_attempt(i, transient != 0, publication != 0);
            CHECK(tuple_context_attempt(SIZE_MAX, false, publication != 0) == count);
        }
    }
}

static size_t tuple_tags_attempt(size_t prefix, bool transient) {
    Environment *env = create_environment(); CHECK(env);
    ASTNode number = {.type = AST_NUMBER}, literal = {.type = AST_TUPLE_LITERAL};
    ASTNode *elements[] = {&number};
    literal.as.tuple_literal.elements = elements; literal.as.tuple_literal.element_count = 1;
    CHECK(check_expression(&literal, env) == TYPE_TUPLE);
    Type *prior = literal.as.tuple_literal.element_types; CHECK(prior && prior[0] == TYPE_INT);
    begin(prefix, transient); Type result = check_expression(&literal, env); size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(result == TYPE_TUPLE && !failed &&
        literal.as.tuple_literal.element_types != prior && literal.as.tuple_literal.element_types[0] == TYPE_INT);
    else CHECK(result == TYPE_UNKNOWN && failed && literal.as.tuple_literal.element_types == prior && prior[0] == TYPE_INT);
    CHECK(check_expression(&literal, env) == TYPE_TUPLE); CHECK(!live);
    prior = literal.as.tuple_literal.element_types;
    ASTNode unresolved = {.type = AST_IDENTIFIER}; unresolved.as.identifier = "missing_tuple_child";
    elements[0] = &unresolved;
    CHECK(check_expression(&literal, env) == TYPE_UNKNOWN && literal.as.tuple_literal.element_types == prior && prior[0] == TYPE_INT);
    literal.as.tuple_literal.element_count = 0;
    CHECK(check_expression(&literal, env) == TYPE_TUPLE && !literal.as.tuple_literal.element_types);
    free_environment(env); CHECK(!live);
    return count;
}
static void tuple_tags_controls(void) {
    size_t count = tuple_tags_attempt(SIZE_MAX, false); CHECK(count == 1);
    for (int transient = 0; transient < 2; ++transient) {
        tuple_tags_attempt(0, transient != 0);
        CHECK(tuple_tags_attempt(SIZE_MAX, false) == count);
    }
}

extern bool array_test_native_publish(Environment *, ASTNode *, bool);
static size_t native_emission_attempt(size_t prefix, bool transient, int mode, bool seeded) {
    Environment *env = create_environment(); CHECK(env);
    TypeInfo scalar = {.base_type = TYPE_INT}, array = {.base_type = TYPE_ARRAY, .element_type = &scalar};
    Type tags[] = {TYPE_INT}; TypeInfo *children[] = {&scalar};
    TypeInfo prior_tuple = {.base_type = TYPE_TUPLE, .tuple_element_count = 1,
        .tuple_types = tags, .type_param_count = 1, .type_params = children};
    ASTNode prior = {.type = AST_TUPLE_LITERAL}, prior_array = {.type = AST_ARRAY_LITERAL};
    prior.as.tuple_literal.element_count = 1;
    if (seeded) {
        CHECK(env_bind_tuple_literal(env, &prior, &prior_tuple));
        CHECK(env_bind_array_expression(env, &prior_array, &array));
    }
    const TypeInfo *old_tuple = env_tuple_literal_info(env, &prior);
    const TypeInfo *old_array = env_array_expression_info(env, &prior_array);
    ASTNode one = {.type = AST_NUMBER}, inner = {.type = AST_TUPLE_LITERAL};
    ASTNode list = {.type = AST_ARRAY_LITERAL}, outer = {.type = AST_TUPLE_LITERAL};
    ASTNode *single[] = {&one}, *pair[] = {&inner, &list};
    inner.as.tuple_literal.elements = single; inner.as.tuple_literal.element_count = 1;
    list.as.array_literal.elements = single; list.as.array_literal.element_count = 1;
    outer.as.tuple_literal.elements = pair; outer.as.tuple_literal.element_count = 2;
    begin(prefix, transient);
    const TypeInfo *published = NULL;
    bool ok = mode == 2 ? (published = checked_expression_type_info(&outer, env)) != NULL
                        : array_test_native_publish(env, &outer, mode == 1);
    size_t count = stop();
    if (mode == 1) CHECK(!ok && !failed);
    else if (prefix == SIZE_MAX) CHECK(ok && !failed);
    else CHECK(!ok && failed);
    CHECK(env_tuple_literal_info(env, &prior) == old_tuple && (!seeded || type_infos_equal(old_tuple, &prior_tuple)));
    CHECK(env_array_expression_info(env, &prior_array) == old_array && (!seeded || type_infos_equal(old_array, &array)));
    CHECK(inner.as.tuple_literal.element_count == 1);
    if (ok) {
        CHECK(env->tuple_literal_binding_count == (size_t)seeded + 2 &&
              env->array_expression_binding_count == (size_t)seeded + 1);
        CHECK(env_tuple_literal_info(env, &outer) && env_tuple_literal_info(env, &inner));
        CHECK(env_array_expression_info(env, &list));
        if (published) CHECK(published->type_param_count == 2 &&
            published->type_params[0]->base_type == TYPE_TUPLE &&
            published->type_params[1]->base_type == TYPE_ARRAY);
    } else {
        CHECK(env->tuple_literal_binding_count == (size_t)seeded &&
              env->array_expression_binding_count == (size_t)seeded);
        CHECK(!env_tuple_literal_info(env, &outer) && !env_tuple_literal_info(env, &inner));
        CHECK(!env_array_expression_info(env, &list));
        if (!seeded) CHECK(!env->tuple_literal_bindings && !env->array_expression_bindings);
    }
    free_environment(env); CHECK(!live);
    return count;
}
static void native_emission_controls(void) {
    for (int seeded = 0; seeded < 2; ++seeded) {
        native_emission_attempt(SIZE_MAX, false, 1, seeded != 0);
        for (int mode = 0; mode <= 2; mode += 2) {
            size_t count = native_emission_attempt(SIZE_MAX, false, mode, seeded != 0); CHECK(count > 30);
            printf("I measure transactional native metadata mode %d seeded %d: %zu allocation attempts.\n", mode, seeded, count);
            for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
                native_emission_attempt(i, transient != 0, mode, seeded != 0);
                CHECK(native_emission_attempt(SIZE_MAX, false, mode, seeded != 0) == count);
            }
        }
    }
}

static size_t tuple_emission_binding_attempt(size_t prefix, bool transient) {
    Environment *env = create_environment(); CHECK(env);
    TypeInfo scalar = {.base_type = TYPE_INT};
    Type tags[] = {TYPE_INT}; TypeInfo *children[] = {&scalar};
    TypeInfo tuple = {.base_type = TYPE_TUPLE, .tuple_element_count = 1,
        .tuple_types = tags, .type_param_count = 1, .type_params = children};
    ASTNode first = {.type = AST_TUPLE_LITERAL}, second = first;
    first.as.tuple_literal.element_count = second.as.tuple_literal.element_count = 1;
    CHECK(env_bind_tuple_literal(env, &first, &tuple));
    const TypeInfo *retained = env_tuple_literal_info(env, &first);
    CHECK(retained && retained != &tuple && retained->type_params[0] != &scalar);
    begin(prefix, transient); bool ok = env_bind_tuple_literal(env, &second, &tuple); size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(ok && !failed && env_tuple_literal_info(env, &second));
    else CHECK(!ok && failed && !env_tuple_literal_info(env, &second));
    CHECK(env_tuple_literal_info(env, &first) == retained && type_infos_equal(retained, &tuple));
    CHECK(env_bind_tuple_literal(env, &first, &tuple));
    tags[0] = TYPE_BOOL;
    CHECK(!env_bind_tuple_literal(env, &first, &tuple) && env_tuple_literal_info(env, &first) == retained);
    free_environment(env); CHECK(!live);
    return count;
}
static void tuple_emission_binding_controls(void) {
    size_t count = tuple_emission_binding_attempt(SIZE_MAX, false); CHECK(count > 3);
    printf("I measure complete tuple emission binding: %zu allocation attempts.\n", count);
    for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
        tuple_emission_binding_attempt(i, transient != 0);
        CHECK(tuple_emission_binding_attempt(SIZE_MAX, false) == count);
    }
}

extern bool array_test_constructor_registry(Environment *, ASTNode *);
static size_t constructor_registry_attempt(size_t prefix, bool transient) {
    Environment *env = create_environment(); CHECK(env);
    StructDef record = {0}; record.name = strdup("Item"); record.module_name = "Caller";
    CHECK(record.name); env_define_struct(env, record);
    UnionDef box = {0}; box.name = strdup("Box"); box.module_name = strdup("Caller");
    box.generic_param_count = 1; box.generic_params = calloc(1, sizeof(char *));
    CHECK(box.name && box.module_name && box.generic_params);
    box.generic_params[0] = strdup("T"); CHECK(box.generic_params[0]);
    env_define_union(env, box);
    ASTNode prior = {.type = AST_UNION_CONSTRUCT}, next = {.type = AST_UNION_CONSTRUCT};
    CHECK(array_test_constructor_registry(env, &prior));
    const void *head = env->checker_nominal_expressions;
    begin(prefix, transient); bool ok = array_test_constructor_registry(env, &next); size_t count = stop();
    if (prefix == SIZE_MAX) CHECK(ok && !failed && env->checker_nominal_expressions != head);
    else CHECK(!ok && failed && env->checker_nominal_expressions == head);
    CHECK(array_test_constructor_registry(env, &prior));
    free_environment(env); CHECK(!live);
    return count;
}
static void constructor_registry_controls(void) {
    size_t count = constructor_registry_attempt(SIZE_MAX, false); CHECK(count > 5);
    printf("I measure constructor proof publication: %zu allocation attempts.\n", count);
    for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
        constructor_registry_attempt(i, transient != 0);
        CHECK(constructor_registry_attempt(SIZE_MAX, false) == count);
    }
}
static void struct_auxiliary_teardown_controls(void) {
    for (int fields = 0; fields < 2; ++fields) {
        Environment *env = create_environment(); CHECK(env);
        TypeInfo borrowed = {.base_type = TYPE_BOOL}; TypeInfo *annotations[] = {&borrowed};
        StructDef record = {.field_count = fields, .field_type_info = annotations,
                            .module_name = "BorrowedOwner"};
        begin(SIZE_MAX, false);
        record.name = array_alloc_strdup("Auxiliary");
        record.field_names = array_alloc_calloc(1, sizeof(char *));
        record.field_types = array_alloc_calloc(1, sizeof(Type));
        record.field_type_names = array_alloc_calloc(1, sizeof(char *));
        record.field_element_types = array_alloc_calloc(1, sizeof(Type));
        CHECK(record.name && record.field_names && record.field_types && record.field_type_names && record.field_element_types);
        if (fields) {
            record.field_names[0] = array_alloc_strdup("value");
            record.field_type_names[0] = array_alloc_strdup("Child");
            CHECK(record.field_names[0] && record.field_type_names[0]);
        }
        env_define_struct(env, record);
        CHECK(stop() >= 5 && !failed);
        free_environment(env);
        CHECK(!live && borrowed.base_type == TYPE_BOOL && annotations[0] == &borrowed);
    }
}

int main(void) {
    size_t count = copy_attempt(SIZE_MAX, false); CHECK(count > 20);
    printf("I measure the complete TypeInfo copy: %zu allocation attempts.\n", count);
    for (int transient = 0; transient < 2; ++transient) for (size_t i = 0; i < count; ++i) {
        copy_attempt(i, transient != 0); CHECK(copy_attempt(SIZE_MAX, false) == count);
    }
    TypeInfo chain[129] = {{0}}, sentinel = {.base_type = TYPE_BOOL}, *out = &sentinel;
    for (int i = 0; i < 128; ++i) { chain[i].base_type = TYPE_ARRAY; chain[i].element_type = &chain[i + 1]; }
    chain[128].base_type = TYPE_INT;
    CHECK(!copy_payload_type_info_checked(chain, &out) && out == &sentinel);
    CHECK(copy_payload_type_info_checked(chain + 1, &out)); free_payload_type_info(out);
    CHECK(copy_payload_type_info_checked(NULL, &out) && out == NULL);
    CHECK(!copy_payload_type_info_checked(chain, NULL));
    struct_auxiliary_teardown_controls(); registration_controls(); view_controls(); owned_context_controls(); callable_context_controls(); native_callable_controls(); tuple_context_controls(); tuple_tags_controls(); tuple_emission_binding_controls(); native_emission_controls(); constructor_registry_controls();
    printf("I passed %zu separate checker annotation allocation assertions.\n", checks);
    return 0;
}
