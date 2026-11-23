/* I free legacy list storage immediately and retain copied records safely. */
#include "../src/nanolang.h"
#include "../src/eval/carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NANO_LIST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(NANO_LIST_ASAN)
#define NANO_LIST_NATIVE_BYTES 1
extern size_t __sanitizer_get_current_allocated_bytes(void);
static size_t allocated(void) { return __sanitizer_get_current_allocated_bytes(); }
#elif defined(__GLIBC__)
#define NANO_LIST_NATIVE_BYTES 1
#include <malloc.h>
static size_t allocated(void) { struct mallinfo2 m = mallinfo2(); return m.uordblks + m.hblkhd; }
#else
static size_t allocated(void) { return 0; }
#endif
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
        "fn lower_probe() -> int { let items: List<LowerRootItem> = (lower_list) let item: LowerRootItem = (list_LowerRootItem_get items 0) assert (== item.text \"kept\") (list_LowerRootItem_free items) return item.number }\n"
        "shadow lower_probe { assert (== (lower_probe) 7) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n"
        "struct LowerRootBox { items: List<LowerRootItem> }\n"
        "fn lower_box() -> LowerRootBox { return LowerRootBox { items: (lower_list) } }\n"
        "shadow lower_box { let box: LowerRootBox = (lower_box) assert (== (lower_probe) 7) assert (== (list_LowerRootItem_length box.items) 1) let item: LowerRootItem = (list_LowerRootItem_get box.items 0) assert (== item.text \"kept\") }\n"
        "fn lower_consume(items: List<LowerRootItem>) -> LowerRootItem { let item: LowerRootItem = (list_LowerRootItem_get items 0) (list_LowerRootItem_free items) return item }\n"
        "shadow lower_consume { let item: LowerRootItem = (lower_consume (lower_list)) assert (== item.text \"kept\") }\n"
        "fn lower_free_alias() -> int { let items: List<LowerRootItem> = (lower_list) let alias: List<LowerRootItem> = items let box: LowerRootBox = LowerRootBox { items: items } let item: LowerRootItem = (list_LowerRootItem_get items 0) (list_LowerRootItem_free alias) assert (== (lower_probe) 7) return item.number }\n"
        "shadow lower_free_alias { assert (== (lower_free_alias) 7) }\n"
        "fn lower_free_field() -> int { let box: LowerRootBox = (lower_box) let item: LowerRootItem = (list_LowerRootItem_get box.items 0) (list_LowerRootItem_free box.items) assert (== (lower_probe) 7) return item.number }\n"
        "shadow lower_free_field { assert (== (lower_free_field) 7) }\n";
    size_t initial = allocated();
    void *positive = malloc(2 * 1024 * 1024); CHECK(positive);
    ((volatile unsigned char *)positive)[0] = 1;
    ((volatile unsigned char *)positive)[2 * 1024 * 1024 - 1] = 2;
    size_t during = allocated(); free(positive);
#if defined(NANO_LIST_NATIVE_BYTES)
    CHECK(during >= initial + 2 * 1024 * 1024);
    CHECK(allocated() <= initial + 4096);
#else
    puts("I have no native byte counter here; I still check all 512 source rounds and metadata bounds.");
#endif
    size_t middle = 0;
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
    for (int i = 0; i < 512; ++i) {
        Value value = call_function("lower_probe", NULL, 0, env);
        CHECK(value.type == VAL_INT && value.as.int_val == 7);
        value = call_function("lower_free_alias", NULL, 0, env);
        CHECK(value.type == VAL_INT && value.as.int_val == 7);
        value = call_function("lower_free_field", NULL, 0, env);
        CHECK(value.type == VAL_INT && value.as.int_val == 7);
        CHECK(env_carrier_collect(env));
        CHECK(env_carrier_consumed_count(env) == 0);
        CHECK(env_carrier_consumed_capacity(env) <= 16);
        CHECK(env_carrier_stats(env).live == 0);
        if (i == 127) middle = allocated();
    }
    size_t last = allocated();
    CHECK(last <= middle + 1024);
    printf("I observe native free storage: positive=%zu after128=%zu after512=%zu delta=%lld rounds=512.\n",
           during - initial, middle, last, (long long)last - (long long)middle);
    Value inputs[] = {create_int(7), create_string("kept")};
    char *names[] = {"number", "text"};
    Value caller = create_struct("LowerRootItem", names, inputs, 2);
    gc_release(inputs[1].as.string_val);
    List_int *list = list_int_new(); list_int_push(list, (intptr_t)caller.as.struct_val);
    CHECK(env_carrier_begin_list_borrow(env, nl_carrier_legacy_record_list(list)));
    NlCarrierRef host = nl_carrier_legacy_record_list(list);
    CHECK(env_carrier_enable_collection(env, &host, 1));
    Value argument = create_int((intptr_t)list);
    size_t registered = env_carrier_stats(env).registered;
    Value returned = call_function("lower_consume", &argument, 1, env);
    CHECK(returned.type == VAL_STRUCT && returned.as.struct_val != caller.as.struct_val);
    CHECK(returned.as.struct_val->field_values[0].as.int_val == 7);
    CHECK(!strcmp(returned.as.struct_val->field_values[1].as.string_val, "kept"));
    CHECK(env_carrier_stats(env).registered == registered + 1);
    CHECK(env_carrier_collect(env));
    CHECK(env_carrier_consumed_count(env) == 1);
    CHECK(env_carrier_enable_collection(env, NULL, 0));
    CHECK(env_carrier_collect(env));
    CHECK(env_carrier_consumed_count(env) == 0);
    CHECK(env_carrier_collect(env));
    CHECK(env_carrier_stats(env).live == 0);
    free_environment(env);
    CHECK(!strcmp(returned.as.struct_val->field_values[1].as.string_val, "kept"));
    gc_release(returned.as.struct_val->field_values[1].as.string_val);
    release_struct_carrier_names(returned.as.struct_val);
    free(returned.as.struct_val->field_values); free(returned.as.struct_val);
    CHECK(!strcmp(caller.as.struct_val->field_values[1].as.string_val, "kept"));
    gc_release(caller.as.struct_val->field_values[1].as.string_val);
    release_struct_carrier_names(caller.as.struct_val);
    free(caller.as.struct_val->field_values); free(caller.as.struct_val);
    free_ast(program);
    free_tokens(tokens, count);
    clear_module_cache();
    CHECK(env_carrier_last_retired_stats().live == 0);
    CHECK(gc_get_stats().num_objects == 0 && gc_get_stats().current_usage == 0);
    puts("I preserve immediate free, copied STRING/record values, arguments, aliases and fields across 512 source rounds with bounded consumption facts.");
    return 0;
}
