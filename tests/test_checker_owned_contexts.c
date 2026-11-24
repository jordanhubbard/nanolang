/* I retain checker context identity without taking caller or AST storage. */
#include "nanolang.h"
#include "runtime/gc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "I failed checker ownership at line %d: %s\n", __LINE__, #condition); \
    exit(1); } } while (0)

typedef struct { Token *tokens; int count; ASTNode *program; } Input;
static Input parse(const char *source) {
    Input input = {0};
    input.tokens = tokenize(source, &input.count);
    CHECK(input.tokens);
    input.program = parse_program(input.tokens, input.count);
    CHECK(input.program);
    return input;
}
static void retire(Input input) {
    free_ast(input.program);
    free_tokens(input.tokens, input.count);
}
static void contexts(void) {
    Environment *env = create_environment();
    char caller[] = "caller-owned-context";
    env->current_module = caller;
    env_set_current_file(env, "root-context.nano");
    typecheck_set_current_file("root-context.nano");
    Input root = parse("module RootOwner\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n");
    CHECK(type_check(root.program, env));
    CHECK(!strcmp(caller, "caller-owned-context"));
    CHECK(env->current_module == env_get_module(env, "RootOwner")->name);
    CHECK(env->current_module != root.program->as.program.items[0]->as.module_decl.name);
    char *retained = env->current_module;
    for (int i = 0; i < 96; ++i) {
        char name[40];
        snprintf(name, sizeof name, "OtherOwner_%d", i);
        env_register_module(env, name, "other.nano", false);
    }
    CHECK(env->current_module == retained && !strcmp(retained, "RootOwner"));
    env->current_module = caller;
    env_set_current_file(env, "module-context.nano");
    typecheck_set_current_file("module-context.nano");
    Input module = parse("module NextOwner\nfn answer() -> int { return 42 }\nshadow answer { assert (== (answer) 42) }\n");
    CHECK(type_check_module(module.program, env));
    CHECK(!strcmp(caller, "caller-owned-context"));
    CHECK(env->current_module == env_get_module(env, "NextOwner")->name);
    CHECK(!strcmp(retained, "RootOwner"));
    char *prior = env->current_module;
    struct EnvModuleCheckpoint *mark = env_module_checkpoint(env, "RejectedOwner", "rejected-context.nano");
    CHECK(mark && env_module_checkpoint_declarations(env, mark));
    env_set_current_file(env, "rejected-context.nano");
    typecheck_set_current_file("rejected-context.nano");
    Input bad = parse("module RejectedOwner\nfn rejected() -> int { return \"wrong\" }\nshadow rejected { assert true }\n");
    CHECK(!type_check_module(bad.program, env));
    CHECK(env_get_module(env, "RejectedOwner"));
    env->current_module = prior;
    env_module_rollback(env, mark);
    CHECK(!env_get_module(env, "RejectedOwner"));
    CHECK(env->current_module == prior && !strcmp(prior, "NextOwner"));
    retire(bad); retire(module); retire(root);
    CHECK(!strcmp(env->current_module, "NextOwner"));
    CHECK(!strcmp(caller, "caller-owned-context"));
    free_environment(env);
}
static void match_bindings(bool expression) {
    const char *prefix = "union Choice { Some { number: int }, None {} }\n";
    const char *body = expression ?
        "fn pair(first: int, second: int) -> int { return (+ first second) }\n"
        "shadow pair { assert (== (pair 1 2) 3) }\n"
        "fn main() -> int { let chosen: Choice = Choice.Some { number: 1 }\n"
        " let value: int = (pair (match chosen { Some(data) => { return 42 } None(_) => { return 42 } })\n"
        "                       (match chosen { Some(data) => { assert false 7 } None(_) => 7 }))\n"
        " return value }\nshadow main { assert (== (main) 42) }\n" :
        "fn main() -> int { let chosen: Choice = Choice.Some { number: 1 }\n"
        " match chosen { Some(data) => { assert (== data.number 1) } None(_) => { assert false } }\n"
        " match chosen { Some(data) => { assert (== data.number 1) } None(_) => { assert false } }\n"
        " return 0 }\nshadow main { assert (== (main) 0) }\n";
    size_t length = strlen(prefix) + strlen(body) + 1;
    char *source = malloc(length); CHECK(source);
    snprintf(source, length, "%s%s", prefix, body);
    Input input = parse(source);
    Environment *env = create_environment();
    env_set_current_file(env, expression ? "expression-binding.nano" : "statement-binding.nano");
    typecheck_set_current_file(env_current_file(env));
    CHECK(type_check(input.program, env));
    CHECK(type_check_root_shadows(input.program, env));
    free_environment(env); retire(input); free(source);
}
int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "contexts") &&
        strcmp(argv[1], "expression") && strcmp(argv[1], "statement"))) return 2;
    clear_module_cache();
    if (argc == 1 || !strcmp(argv[1], "contexts")) contexts();
    if (argc == 1 || !strcmp(argv[1], "expression")) match_bindings(true);
    if (argc == 1 || !strcmp(argv[1], "statement")) match_bindings(false);
    clear_module_cache(); gc_shutdown();
    puts("I preserve caller/registry contexts, growth, rollback and repeated match binding ownership.");
    return 0;
}
