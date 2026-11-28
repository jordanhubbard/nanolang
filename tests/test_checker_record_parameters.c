/* I check nominal parameter metadata without constructing runtime records. */
#include "../src/nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "I failed %s at line %d.\n", #condition, __LINE__); exit(1); \
} } while (0)

typedef struct {
    const char *source;
    bool accepted;
    const char *parameter;
    Type type;
    const char *nominal;
} Case;

int main(void) {
    const Case cases[] = {
        {"struct Point { value: int } fn read(item: Point) -> int { return item.value } fn main() -> int { return 0 }", true, "item", TYPE_STRUCT, "Point"},
        {"struct Point { value: int } struct Box { point: Point } fn read(item: Box) -> int { return item.point.value } fn main() -> int { return 0 }", true, "item", TYPE_STRUCT, "Box"},
        {"struct Point { value: int } fn read(items: array<Point>) -> int { let item: Point = (array_get items 0) return item.value } fn main() -> int { return 0 }", true, "items", TYPE_ARRAY, NULL},
        {"struct Point { value: int } fn read(item: Point) -> int { return item.absent } fn main() -> int { return 0 }", false, "item", TYPE_STRUCT, "Point"},
        {"struct Point { value: int } fn read(item: Point) -> string { return item.value } fn main() -> int { return 0 }", false, "item", TYPE_STRUCT, "Point"},
        {"struct Point { value: int } struct Other { value: int } fn read(item: Point) -> int { return item.value } fn main() -> int { return (read Other { value: 1 }) }", false, "item", TYPE_STRUCT, "Point"},
    };
    unsigned completed = 0;
    for (int round = 0; round < 3; ++round) {
        for (int module = 0; module < 2; ++module) {
            for (size_t index = 0; index < sizeof cases / sizeof *cases; ++index) {
                int count = 0;
                Token *tokens = tokenize(cases[index].source, &count);
                REQUIRE(tokens);
                ASTNode *program = parse_program(tokens, count);
                REQUIRE(program);
                Environment *env = create_environment();
                env->suppress_shadow_warnings = true;
                env->current_file = "<checker-record-parameters>";
                typecheck_set_current_file(env->current_file);
                bool accepted = module ? type_check_module(program, env) : type_check(program, env);
                REQUIRE(accepted == cases[index].accepted);
                Symbol *parameter = env_get_var(env, cases[index].parameter);
                REQUIRE(parameter && parameter->type == cases[index].type);
                if (cases[index].nominal) {
                    REQUIRE(parameter->struct_type_name);
                    REQUIRE(!strcmp(parameter->struct_type_name, cases[index].nominal));
                } else {
                    REQUIRE(parameter->element_type == TYPE_STRUCT);
                }
                free_environment(env);
                free_ast(program);
                free_tokens(tokens, count);
                ++completed;
            }
        }
    }
    printf("I pass %u repeated checker record-parameter controls.\n", completed);
    return 0;
}
