/* I check emitter context restoration across successive and enclosing calls. */
#include "../src/transpiler.c"
#include <assert.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static void emit_checked(const char *source, const char *expected_declaration) {
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program);
    Environment *env = create_environment();
    assert(env);
    typecheck_set_current_file("<native-nominal-context>");
    assert(type_check(program, env));
    char *text = transpile_to_c(program, env, "<native-nominal-context>");
    assert(text);
    if (expected_declaration) {
        assert(strstr(text, expected_declaration));
        assert(!strstr(text, "typedef struct void*"));
    } else {
        assert(!strstr(text, "nl_T"));
    }
    free(text);
    free_environment(env);
    free_ast(program);
    free_tokens(tokens, count);
}

int main(void) {
    const char *declared = "union T { Some { value: int } } fn main() -> int { return 0 } shadow main { assert (== (main) 0) }";
    const char *generic = "fn identity(value: T) -> T { return value } shadow identity { assert (== (identity 7) 7) } fn main() -> int { return (- (identity 7) 7) } shadow main { assert (== (main) 0) }";
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    emit_checked(declared, "typedef struct nl_T");
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    emit_checked(generic, NULL);
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));

    emit_checked("enum T { First, Second } fn main() -> int { return 0 } shadow main { assert (== (main) 0) }", "} nl_T;");
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    emit_checked(generic, NULL);

    /* An enclosing emission's snapshot survives an inner success or refusal. */
    native_declared_letters = UINT32_C(1) << ('U' - 'A');
    emit_checked(declared, "typedef struct nl_T");
    assert(!strcmp(get_prefixed_type_name("U"), "nl_U"));
    assert(!strcmp(get_prefixed_type_name("T"), "void*"));
    assert(!transpile_to_c(NULL, NULL, "<refused>"));
    assert(!strcmp(get_prefixed_type_name("U"), "nl_U"));
    native_declared_letters = 0;
    puts("I passed native nominal context restoration checks.");
    return 0;
}
