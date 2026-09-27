/* I exercise actual module lookup/preflight and both alias-copy allocation sites. */
#include "nanolang.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc;
char **g_argv;
static size_t alias_calls, alias_fail_at, alias_live;
static void *alias_allocations[4];
static char *alias_strdup(const char *text) {
    ++alias_calls;
    if (alias_calls == alias_fail_at) return NULL;
    char *copy = strdup(text);
    assert(copy);
    assert(alias_live < 4);
    alias_allocations[alias_live++] = copy;
    return copy;
}
static void alias_free(void *pointer) {
    if (!pointer) return;
    size_t index = 0;
    while (index < alias_live && alias_allocations[index] != pointer) ++index;
    assert(index < alias_live);
    alias_allocations[index] = alias_allocations[--alias_live];
    free(pointer);
}
#define strdup alias_strdup
#define free alias_free
#include "../src/module.c"
#undef free
#undef strdup

static void check_alias_copy(void) {
    char source_name[] = "declared";
    char original_name[] = "original";
    char selected_name[] = "chosen";
    Function source = {0};
    source.name = source_name;
    source.alias_of = original_name;
    source.module_name = "owner";
    source.is_pub = true;
    source.param_count = 3;
    for (size_t failure = 1; failure <= 2; ++failure) {
        Function output;
        unsigned char before[sizeof output];
        memset(&output, 0xa5, sizeof output);
        memcpy(before, &output, sizeof output);
        alias_calls = 0; alias_fail_at = failure;
        assert(!copy_module_function_alias(&source, selected_name, &output));
        assert(alias_calls == failure && alias_live == 0);
        assert(memcmp(before, &output, sizeof output) == 0);
        alias_calls = 0; alias_fail_at = 0;
        assert(copy_module_function_alias(&source, selected_name, &output));
        assert(alias_calls == 2 && alias_live == 2);
        assert(output.name != selected_name && strcmp(output.name, "chosen") == 0);
        assert(output.alias_of != original_name && strcmp(output.alias_of, "original") == 0);
        assert(output.module_name == source.module_name && output.is_pub);
        assert(output.param_count == 3 && output.params == source.params);
        alias_free(output.name); alias_free(output.alias_of);
        assert(alias_live == 0);
    }
    Function output;
    alias_calls = 0; alias_fail_at = 0;
    assert(copy_module_function_alias(&source, selected_name, &output));
    memset(source_name, 'x', sizeof source_name - 1);
    memset(original_name, 'y', sizeof original_name - 1);
    memset(selected_name, 'z', sizeof selected_name - 1);
    assert(strcmp(output.name, "chosen") == 0);
    assert(strcmp(output.alias_of, "original") == 0);
    alias_free(output.name); alias_free(output.alias_of);
    assert(alias_live == 0);
}

static void check_selected_owner(void) {
    Function functions[3] = {0};
    for (int i = 0; i < 3; ++i) functions[i].name = "member";
    functions[0].is_pub = true;
    functions[1].module_name = "other";
    functions[1].is_pub = true;
    functions[2].module_name = "owner";
    Environment env = {0};
    env.functions = functions; env.function_count = 3;
    assert(find_module_function(&env, "owner", "member") == &functions[2]);
    assert(find_module_function(&env, "absent", "member") == NULL);
    assert(find_module_function(&env, NULL, "member") == &functions[0]);
    char *names[] = {"member", "OnlyType"};
    ASTNode item = {0}; item.type = AST_IMPORT; item.line = 1; item.column = 1;
    item.as.import_stmt.import_symbols = names;
    item.as.import_stmt.import_symbol_count = 2;
    assert(!selected_module_functions_public(&env, "owner", &item));
    assert(env.function_count == 3);
    functions[2].is_pub = true;
    assert(selected_module_functions_public(&env, "owner", &item));
    assert(env.function_count == 3);
}
int main(int argc, char **argv) {
    g_argc = argc; g_argv = argv;
    check_selected_owner(); check_alias_copy();
    puts("I checked exact function owners, selective visibility and both alias allocation failures.");
    return 0;
}
