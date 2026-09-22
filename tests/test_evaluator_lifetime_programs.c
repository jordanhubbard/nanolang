#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../src/nanolang.h"
#include "../src/coroutine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "I failed %s at %d\n", #x, __LINE__); exit(90); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
extern EnvEvaluationProvider *lifetime_cache_provider(void);
extern int lifetime_cache_count(void);
extern bool lifetime_private_compile_fault(const char *, const char *, Environment *, int);
extern int lifetime_private_close_count(void);
extern bool lifetime_task_clone(Value, Value *);
extern void lifetime_task_drop(Value);
static char *read_source(const char *path) {
    FILE *file = fopen(path, "rb"); CHECK(file && !fseek(file, 0, SEEK_END));
    long count = ftell(file); CHECK(count >= 0 && count < 16777216 && !fseek(file, 0, SEEK_SET));
    char *text = malloc((size_t)count + 1); CHECK(text);
    CHECK(fread(text, 1, (size_t)count, file) == (size_t)count && !fclose(file));
    text[count] = 0; return text;
}
static void program(const char *path, bool escape, bool reject) {
    char *source = read_source(path); int count;
    Token *tokens = tokenize(source, &count); CHECK(tokens);
    ASTNode *ast = parse_program(tokens, count); CHECK(ast);
    Environment *env = create_environment(); ModuleList *modules = create_module_list(); CHECK(env && modules);
    CHECK(process_imports(ast, env, modules, path));
    typecheck_set_current_file(path);
    bool checked = type_check(ast, env);
    if (reject) CHECK(!checked);
    else {
        CHECK(checked && run_program(ast, env));
        Value main_result = call_function("main", NULL, 0, env);
        CHECK(main_result.type == VAL_INT && main_result.as.int_val == 0);
    }
    Value escaped = {0};
    if (escape) {
        escaped = call_function("capture", NULL, 0, env);
        CHECK(escaped.type == VAL_TUPLE && escaped.as.tuple_val->element_count == 2);
    }
    CHECK(env_can_destroy(env)); free_environment(env); free_ast(ast); free_tokens(tokens, count);
    free_module_list(modules); clear_module_cache(); free(source);
    if (escape) {
        Value *values = escaped.as.tuple_val->elements;
        CHECK(values[0].type == VAL_STRUCT && values[1].type == VAL_STRING);
        CHECK(!strcmp(values[1].as.string_val, "tail"));
        CHECK(!strcmp(values[0].as.struct_val->field_values[0].as.string_val, "retained"));
        CHECK(values[0].as.struct_val->field_values[1].as.int_val == 41);
        env_discard_value_snapshot(escaped);
    }
    puts("I retained my parsed evaluator result and cleanup assertions.");
}
/* I inspect declaration facts without executing an unbound foreign ABI. */
static void declarations(const char *path, const char *mode) {
    char *source = read_source(path); int count;
    Token *tokens = tokenize(source, &count); CHECK(tokens);
    ASTNode *ast = parse_program(tokens, count); CHECK(ast);
    Environment *env = create_environment(); ModuleList *modules = create_module_list(); CHECK(env && modules);
    bool imported = process_imports(ast, env, modules, path);
    if (!strcmp(mode, "declarations-import-refuse")) CHECK(!imported);
    else {
        CHECK(imported); typecheck_set_current_file(path);
        bool checked = type_check(ast, env);
        if (!strcmp(mode, "declarations-refuse")) CHECK(!checked);
        else {
            CHECK(checked);
            NominalIdentity token = env_nominal_identity(env, "NativeToken", "Contracts", TYPE_STRUCT);
            if (!strcmp(mode, "declarations-pending")) {
                CHECK(!token.ordinal && env->generic_instance_count == 0);
            } else {
                CHECK(token.ordinal && env->structs[token.ordinal - 1].is_extern);
                CHECK(!strcmp(env_nominal_owner(env, token), "Tokens"));
                if (!strcmp(mode, "declarations-alias"))
                    CHECK(env_nominal_identity(env, "native.NativeToken", NULL, TYPE_STRUCT).ordinal == token.ordinal);
                bool found = false;
                for (int i = 0; i < env->generic_instance_count; ++i) {
                    GenericInstantiation *instance = &env->generic_instances[i];
                    if (instance->list_element.kind == TYPE_STRUCT && instance->list_element.ordinal == token.ordinal)
                        found = true;
                }
                CHECK(found);
            }
        }
    }
    free_environment(env); free_ast(ast); free_tokens(tokens, count);
    free_module_list(modules); clear_module_cache(); free(source);
    puts("I checked pending and resolved foreign declaration facts without foreign execution.");
}
static void foreign_facts(void) {
    for (int reverse = 0; reverse < 2; ++reverse) {
        Environment *env = create_environment(); CHECK(env);
        StructDef ordinary = {0}, foreign = {0};
        ordinary.name = strdup("OtherRecord"); ordinary.module_name = "Other";
        foreign.name = strdup("NativeToken"); foreign.module_name = "Tokens"; foreign.is_extern = true;
        CHECK(ordinary.name && ordinary.module_name && foreign.name && foreign.module_name);
        env_define_struct(env, reverse ? foreign : ordinary);
        env_define_struct(env, reverse ? ordinary : foreign);
        NominalIdentity token = env_nominal_identity(env, "NativeToken", "Contracts", TYPE_STRUCT);
        CHECK(token.ordinal == (size_t)(reverse ? 1 : 2));
        CHECK(!strcmp(env_nominal_owner(env, token), "Tokens"));
        CHECK(!env_nominal_identity(env, "OtherRecord", "Contracts", TYPE_STRUCT).ordinal);
        char **exported = calloc(1, sizeof(char *)); CHECK(exported);
        exported[0] = strdup("NativeToken"); CHECK(exported[0]);
        env_register_namespace(env, "native", "Tokens", NULL, 0, exported, 1, NULL, 0, NULL, 0);
        CHECK(env_nominal_identity(env, "native.NativeToken", NULL, TYPE_STRUCT).ordinal == token.ordinal);
        CHECK(!env_nominal_identity(env, "native.NativeToken", "Elsewhere", TYPE_STRUCT).ordinal);
        CHECK(!env_nominal_identity(env, "native.Missing", NULL, TYPE_STRUCT).ordinal);
        /* Direct registrations bypass the AST binder, so the resolver itself
         * must refuse both ordinary and foreign competing declarations. */
        StructDef collision = {0}; collision.name = strdup("NativeToken");
        collision.module_name = "Competing"; collision.is_extern = reverse != 0;
        CHECK(collision.name && collision.module_name); env_define_struct(env, collision);
        CHECK(!env_nominal_identity(env, "NativeToken", "Contracts", TYPE_STRUCT).ordinal);
        CHECK(!env_nominal_identity(env, "native.NativeToken", NULL, TYPE_STRUCT).ordinal);
        CHECK(!env_register_list_instantiation(env, "NativeToken"));
        free_environment(env);
    }
}
static void caches(const char *path, const char *object, const char *mode) {
    Environment *a = create_environment(), *b = create_environment(); CHECK(a && b);
    ASTNode *first = load_module(path, a); CHECK(first);
    EnvEvaluationProvider *provider = lifetime_cache_provider(); CHECK(provider);
    int before = lifetime_cache_count(); CHECK(before > 0);
    CHECK(env_acquire_evaluation_lease(a));
    CHECK(load_module(path, b) == first); /* Real cached hit must register B. */
    CHECK(env_acquire_evaluation_lease(b));
    if (!strcmp(mode, "cache-refuse")) {
        env_release_evaluation_lease(a); free_environment(a);
        clear_module_cache(); CHECK(false);
    }
    if (!strcmp(mode, "env-refuse")) { free_environment(a); CHECK(false); }
    /* The actual compiler temporarily swaps a distinct module environment/cache. */
    CHECK(compile_module_to_object(path, object, a, false, NULL, 0));
    CHECK(lifetime_cache_provider() == provider && lifetime_cache_count() == before);
    CHECK(get_cached_module_ast(path) == first);
    CHECK(!compile_module_to_object("/no-such-nanolang-lifetime-module.nano", object, a, false, NULL, 0));
    CHECK(lifetime_cache_provider() == provider && get_cached_module_ast(path) == first);
    /* I model allocation/transpile/write reporting faults; I separately invoke
     * the real selected C compiler with a deliberately invalid option. */
    for (int fault = 1; fault <= 6; ++fault) {
        if (fault == 2) continue;
        CHECK(!lifetime_private_compile_fault(path, object, a, fault));
        CHECK(lifetime_cache_provider() == provider && get_cached_module_ast(path) == first);
        CHECK(!env_can_destroy(a) && !env_can_destroy(b));
        if (fault == 3 || fault == 4) CHECK(lifetime_private_close_count() == 1);
    }
    CHECK(!compile_module_to_object(path, "/no-such-nanolang-lifetime-directory/object.o", a, false, NULL, 0));
    CHECK(lifetime_cache_provider() == provider && get_cached_module_ast(path) == first);
    env_release_evaluation_lease(a); env_release_evaluation_lease(b);
    clear_module_cache(); CHECK(!lifetime_cache_provider());
    CHECK(!env_acquire_evaluation_lease(a) && !env_acquire_evaluation_lease(b));
    free_environment(a); free_environment(b);
    puts("I retained cache-hit leases and real private compiler restoration.");
}
static void task_callable(void) {
    Type tags[] = {TYPE_STRING};
    TypeInfo argument = {.base_type = TYPE_STRING}; TypeInfo *arguments[] = {&argument};
    FunctionSignature signature = {.param_types = tags, .param_count = 1, .param_type_info = arguments,
                                  .return_type = TYPE_STRING, .return_type_info = &argument};
    Value source = {0}; source.type = VAL_FUNCTION;
    source.as.function_val.function_name = "callable"; source.as.function_val.signature = &signature;
    Value out; CHECK(lifetime_task_clone(source, &out));
    CHECK(out.as.function_val.signature != &signature && out.as.function_val.signature->param_type_info[0] != &argument);
    CHECK(!strcmp(out.as.function_val.function_name, "callable")); lifetime_task_drop(out);
}
extern int lifetime_enqueue_named(Environment *, const char *);
extern Value lifetime_task_result(Environment *, int, bool);
static Environment *task_env, *task_foreign;
static ASTNode *task_ast;
static Token *task_tokens;
static int task_token_count, task_id = -1, task_calls;
static bool task_expect_ready;
static void task_control_cleanup(void) {
    if (task_id >= 0) {
        if (task_expect_ready) {
            CHECK(!nano_coro_is_done(task_id) && task_calls == 0);
            Symbol *calls = env_get_var(task_env, "calls");
            CHECK(calls && calls->value.type == VAL_INT && calls->value.as.int_val == 0);
        }
        if (!nano_coro_is_done(task_id)) CHECK(nano_coro_cancel(task_id));
        CHECK(nano_coro_release(task_id)); task_id = -1;
    }
    if (task_foreign) { free_environment(task_foreign); task_foreign = NULL; }
    if (task_env) { CHECK(env_can_destroy(task_env)); free_environment(task_env); task_env = NULL; }
    if (task_ast) { free_ast(task_ast); task_ast = NULL; }
    if (task_tokens) { free_tokens(task_tokens, task_token_count); task_tokens = NULL; }
    clear_module_cache();
}
static Value task_raw_callback(void *unused, int id) {
    (void)unused; (void)id; ++task_calls; return create_int(7);
}
static void task_context_control(const char *mode) {
    CHECK(atexit(task_control_cleanup) == 0);
    const char *source =
        "let mut calls: int = 0\n"
        "fn scalar() -> int { set calls (+ calls 1) return 7 }\n"
        "shadow scalar { assert (== (scalar) 7) }\n"
        "fn words() -> string { return \"kept\" }\n"
        "shadow words { assert (== (words) \"kept\") }\n";
    task_tokens = tokenize(source, &task_token_count); CHECK(task_tokens);
    task_ast = parse_program(task_tokens, task_token_count); CHECK(task_ast);
    task_env = create_environment(); task_foreign = create_environment(); CHECK(task_env && task_foreign);
    typecheck_set_current_file("<task-context>");
    CHECK(type_check(task_ast, task_env) && run_program(task_ast, task_env));
    bool raw = !strcmp(mode, "task-raw");
    task_id = raw ? nano_coro_spawn(task_raw_callback, NULL) : lifetime_enqueue_named(task_env, "scalar");
    CHECK(task_id >= 0);
    if (!strcmp(mode, "task-foreign-ready") || raw) {
        task_expect_ready = true;
        (void)lifetime_task_result(raw ? task_env : task_foreign, task_id, true);
        CHECK(false);
    }
    CHECK(nano_coro_context_matches(task_id, task_env->task_identity));
    Value result = lifetime_task_result(task_env, task_id, true);
    CHECK(nano_coro_is_done(task_id) && result.type == VAL_INT && result.as.int_val == 7);
    CHECK(env_can_destroy(task_env));
    if (!strcmp(mode, "task-foreign-done")) {
        (void)lifetime_task_result(task_foreign, task_id, false); CHECK(false);
    }
    EnvTaskIdentity *retained = task_env->task_identity;
    free_environment(task_env); task_env = NULL;
    CHECK(nano_coro_context_matches(task_id, retained));
    task_env = create_environment(); CHECK(task_env && task_env->task_identity != retained);
    CHECK(!nano_coro_context_matches(task_id, task_env->task_identity));
    CHECK(nano_coro_release(task_id)); task_id = -1;
    /* I reuse the checked declarations in a fresh Environment. */
    CHECK(type_check(task_ast, task_env) && run_program(task_ast, task_env));
    task_id = lifetime_enqueue_named(task_env, "words"); CHECK(task_id >= 0);
    result = lifetime_task_result(task_env, task_id, true);
    CHECK(result.type == VAL_STRING && !strcmp(result.as.string_val, "kept"));
    env_discard_value_snapshot(result);
    CHECK(nano_coro_is_done(task_id) && !env_can_destroy(task_env));
    CHECK(nano_coro_release(task_id)); task_id = -1;
    CHECK(env_can_destroy(task_env));
    task_control_cleanup();
    puts("I retained task identity, scalar teardown and completed result leases.");
}
int main(int argc, char **argv) {
    CHECK(argc >= 2); nano_scheduler_init();
    if (!strncmp(argv[1], "task-", 5)) { CHECK(argc == 2); task_context_control(argv[1]); }
    else if (!strcmp(argv[1], "program") || !strcmp(argv[1], "escape") || !strcmp(argv[1], "reject")) {
        CHECK(argc == 3); program(argv[2], !strcmp(argv[1], "escape"), !strcmp(argv[1], "reject"));
    } else if (!strncmp(argv[1], "declarations-", 13)) {
        CHECK(argc == 3); declarations(argv[2], argv[1]);
    } else if (!strcmp(argv[1], "foreign-facts")) foreign_facts();
    else if (!strcmp(argv[1], "callable")) task_callable();
    else { CHECK(argc == 4); caches(argv[2], argv[3], argv[1]); }
    return 0;
}
