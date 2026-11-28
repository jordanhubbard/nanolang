/* I retain diagnostics after a declaration refuses registration. */
#include "nanolang.h"
#include "runtime/gc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static void check(const char *name, const char *declarations, bool module, bool accepted) {
    const char *entry = "fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n";
    char source[4096];
    int length = snprintf(source, sizeof source, "%s\n%s", declarations, module ? "" : entry);
    assert(length > 0 && (size_t)length < sizeof source);
    int count = 0;
    Token *tokens = tokenize(source, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    assert(program);
    clear_module_cache();
    Environment *env = create_environment();
    assert(env);
    env_set_current_file(env, "<rejected-declaration>");
    typecheck_set_current_file("<rejected-declaration>");
    bool result = module ? type_check_module(program, env) : type_check(program, env);
    assert(result == accepted);
    if (!module && !strncmp(name, "reserved", 8))
        assert(env_get_function(env, "bstr_utf8_char_at") == NULL);
    printf("{\"case\":\"%s\",\"module\":%s,\"accepted\":%s}\n",
           name, module ? "true" : "false", result ? "true" : "false");
    free_environment(env);
    free_ast(program);
    free_tokens(tokens, count);
    clear_module_cache();
    gc_shutdown();
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *legal = "fn bstr_custom(x: int, y: int) -> int { return (+ x y) }\n"
                        "shadow bstr_custom { assert (== (bstr_custom 1 2) 3) }\n";
    check("legal-name", legal, false, true);
    check("legal-name", legal, true, true);
    check("orphan-shadow", "shadow absent { assert true }\n", false, false);
    check("orphan-shadow", "shadow absent { assert true }\n", true, false);
    check("reserved-with-shadow",
          "fn bstr_utf8_char_at(x: int, y: int) -> int { return (+ x y) }\n"
          "shadow bstr_utf8_char_at { assert (== (bstr_utf8_char_at 1 2) 3) }\n", false, false);
    check("reserved-without-shadow",
          "fn bstr_utf8_char_at(x: int, y: int) -> int { return (+ x y) }\n", false, false);
    return 0;
}
