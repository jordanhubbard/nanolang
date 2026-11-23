/* I exercise the accepted lower-case source helpers with private collection. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "I fail %s at line %d.\n", #c, __LINE__); exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
int main(void) {
    const char *source =
        "struct LowerRootItem { number: int, text: string }\n"
        "fn lower_list() -> List<LowerRootItem> { let items: List<LowerRootItem> = (list_LowerRootItem_new) (list_LowerRootItem_push items LowerRootItem { number: 7, text: \"kept\" }) return items }\n"
        "shadow lower_list { let items: List<LowerRootItem> = (lower_list) assert (== (list_LowerRootItem_length items) 1) }\n"
        "fn lower_probe() -> int { let items: List<LowerRootItem> = (lower_list) let item: LowerRootItem = (list_LowerRootItem_get items 0) assert (== item.text \"kept\") return item.number }\n"
        "shadow lower_probe { assert (== (lower_probe) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n"
        "struct LowerRootBox { items: List<LowerRootItem> }\n"
        "fn lower_box() -> LowerRootBox { return LowerRootBox { items: (lower_list) } }\n"
        "shadow lower_box { let box: LowerRootBox = (lower_box) assert (== (lower_probe) 7) assert (== (list_LowerRootItem_length box.items) 1) let item: LowerRootItem = (list_LowerRootItem_get box.items 0) assert (== item.text \"kept\") }\n";
    int count = 0;
    Token *tokens = tokenize(source, &count); CHECK(tokens);
    ASTNode *program = parse_program(tokens, count); CHECK(program);
    Environment *env = create_environment();
    typecheck_set_current_file("<legacy-record-list>");
    CHECK(type_check(program, env));
    CHECK(type_check_root_shadows(program, env));
    CHECK(env_carrier_enable(env));
    CHECK(env_carrier_enable_collection(env, NULL, 0));
    CHECK(run_program(program, env));
    CHECK(run_shadow_tests(program, env, false));
    for (int i = 0; i < 128; ++i) {
        Value value = call_function("lower_probe", NULL, 0, env);
        CHECK(value.type == VAL_INT && value.as.int_val == 7);
    }
    CHECK(env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live == 0);
    free_environment(env);
    free_ast(program);
    free_tokens(tokens, count);
    clear_module_cache();
    CHECK(env_carrier_last_retired_stats().live == 0);
    CHECK(gc_get_stats().num_objects == 0 && gc_get_stats().current_usage == 0);
    puts("I preserve lower-case record-list shadows and 128 returned/local/STRING source calls.");
    return 0;
}
