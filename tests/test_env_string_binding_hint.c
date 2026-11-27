/* I preserve exact STRING retain behavior while skipping never-bound scans. */
#include "nanolang.h"
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
#include "runtime/list_string.h"
#include "runtime/ffi_loader.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/env.c"

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static void fresh_and_sticky(void) {
    Value value = create_string("fresh");
    GCHeader *header = gc_get_header(value.as.string_val);
    assert(header->flags == 0 && !env_string_binding_seen(value));
    header->flags = UINT16_C(0x8000);
    env_note_string_binding(create_int(1));
    assert(!env_string_binding_seen(value));
    env_note_string_binding(value);
    assert(header->flags == (UINT16_C(0x8000) | GC_FLAG_ENV_STRING_BOUND));
    assert(header->ref_count == 1 && env_string_binding_seen(value));
    env_note_string_binding(value);
    assert(header->ref_count == 1);
    gc_release(value.as.string_val);
    for (int i = 0; i < 1024; ++i) {
        value = create_string("recycled allocation");
        assert(!env_string_binding_seen(value));
        env_note_string_binding(value);
        assert(env_string_binding_seen(value));
        gc_release(value.as.string_val);
    }
    Value plain = create_void(); plain.type = VAL_STRING;
    plain.as.string_val = strdup("unmanaged"); assert(plain.as.string_val);
    env_note_string_binding(plain); assert(!env_string_binding_seen(plain));
    free(plain.as.string_val);
    plain.as.string_val = NULL; env_note_string_binding(plain);
    assert(!env_string_binding_seen(plain));
}

static void aliases_at_every_position(void) {
    ASTNode call = {.type = AST_CALL}; call.as.call.name = "untrusted_spelling";
    for (int position = 0; position < 128; ++position) {
        Environment *env = create_environment();
        Value value = create_string("original"); char *pointer = value.as.string_val;
        for (int i = 0; i < 128; ++i) {
            char name[32]; snprintf(name, sizeof name, "slot_%d", i);
            if (i == position) env_define_var(env, name, TYPE_STRING, true, value);
            else env_define_var(env, name, TYPE_INT, false, create_int(i));
        }
        assert(env_string_binding_seen(value) && gc_get_header(pointer)->ref_count == 1);
        for (int i = 0; i < 8; ++i) {
            char name[32]; snprintf(name, sizeof name, "alias_%d", i);
            env_define_evaluated_var(env, name, TYPE_STRING, TYPE_UNKNOWN, NULL, true, value, &call);
            assert(gc_get_header(pointer)->ref_count == (uint32_t)(i + 2));
        }
        char source[32]; snprintf(source, sizeof source, "slot_%d", position);
        env_set_var(env, source, create_string("replacement"));
        assert(gc_get_header(pointer)->ref_count == 8);
        env_set_var(env, "alias_0", value);
        assert(gc_get_header(pointer)->ref_count == 8);
        Value fresh = create_string("new target"); char *replacement = fresh.as.string_val;
        env_set_evaluated_var(env, "alias_0", fresh, &call);
        assert(gc_get_header(pointer)->ref_count == 7 && gc_get_header(replacement)->ref_count == 1);
        assert(env_string_binding_seen(fresh));
        env_set_var(env, "alias_0", value);
        assert(!gc_is_managed(replacement) && gc_get_header(pointer)->ref_count == 8);
        free_environment(env); assert(!gc_is_managed(pointer));
    }
}

static void borrowed_and_retained_outside_scope(void) {
    Environment *env = create_environment(); Value value = create_string("caller");
    env_define_var(env, "borrow", TYPE_BORROW_SHARED, false, value);
    assert(env_string_binding_seen(value) && gc_get_header(value.as.string_val)->ref_count == 1);
    free_environment(env);
    assert(gc_is_managed(value.as.string_val) && env_string_binding_seen(value));
    env = create_environment();
    env_define_var(env, "owner", TYPE_STRING, true, value);
    assert(gc_get_header(value.as.string_val)->ref_count == 1);
    gc_retain(value.as.string_val);
    struct EnvModuleCheckpoint *mark = env_module_checkpoint(env, "HintOwner", "hint.nano");
    assert(mark && env_module_checkpoint_declarations(env, mark));
    env_module_rollback(env, mark);
    assert(env_string_binding_seen(env_get_var(env, "owner")->value));
    free_environment(env);
    assert(gc_is_managed(value.as.string_val) && env_string_binding_seen(value));
    gc_release(value.as.string_val);
}

static void imported_literal_publication(void) {
    const char *temporary = getenv("TMPDIR"); if (!temporary) temporary = "/tmp";
    char directory[4096]; assert(snprintf(directory, sizeof directory, "%s/nano-string-hint-XXXXXX", temporary) < (int)sizeof directory);
    assert(mkdtemp(directory));
    char provider[4096], consumer[4096], source[8192];
    assert(snprintf(provider, sizeof provider, "%s/provider.nano", directory) < (int)sizeof provider);
    assert(snprintf(consumer, sizeof consumer, "%s/consumer.nano", directory) < (int)sizeof consumer);
    FILE *file = fopen(provider, "wb"); assert(file);
    assert(fputs("let legacy_string: string = \"legacy\"\n", file) >= 0); assert(!fclose(file));
    /* I exercise the retained legacy folding path, not a new import style. */
    assert(snprintf(source, sizeof source, "import \"%s\"\n", provider) < (int)sizeof source);
    int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
    ASTNode *program = parse_program(tokens, count); assert(program);
    Environment *env = create_environment(); ModuleList *modules = create_module_list();
    assert(process_imports(program, env, modules, consumer));
    Symbol *symbol = env_global_import_symbol_at(env, consumer, "legacy_string");
    assert(symbol && symbol->value.type == VAL_STRING && env_string_binding_seen(symbol->value));
    Value value = symbol->value; char *pointer = value.as.string_val;
    env_define_var(env, "retained", TYPE_STRING, false, value);
    assert(gc_get_header(pointer)->ref_count == 2);
    free_environment(env); free_module_list(modules); free_ast(program); free_tokens(tokens, count); clear_module_cache();
    assert(!gc_is_managed(pointer)); assert(!unlink(provider)); assert(!rmdir(directory));
}

static int loop_probe_calls, loop_probe_misses;
bool nano_test_string_binding_probe(const char *text) {
    ++loop_probe_calls;
    bool seen = gc_is_managed((void *)text) &&
        (gc_get_header((void *)text)->flags & GC_FLAG_ENV_STRING_BOUND) != 0;
    if (!seen) ++loop_probe_misses;
    return seen;
}

static void direct_loop_publications(void) {
    const char *source =
        "extern fn nano_test_string_binding_probe(value: string) -> bool\n"
        "fn loop_probe(values: array<string>) -> void {\n"
        " for item in values { unsafe { assert (nano_test_string_binding_probe item) } }\n"
        "}\n"
        "shadow loop_probe { (loop_probe [\"shadow\"]) }\n"
        "fn main() -> int { return 0 }\n"
        "shadow main { assert (== (main) 0) }\n";
    int count = 0; Token *tokens = tokenize(source, &count); assert(tokens);
    ASTNode *program = parse_program(tokens, count); assert(program);
    Environment *env = create_environment(); typecheck_set_current_file("<test>");
    assert(type_check(program, env)); assert(run_program(program, env));
    assert(ffi_loader_init(false));
    Value array = create_array(VAL_STRING, 1, 1);
    ((char **)array.as.array_val->data)[0] = strdup("static");
    (void)call_function("loop_probe", &array, 1, env);
    assert(loop_probe_calls == 1 && loop_probe_misses == 0);
    DynArray *dynamic = dyn_array_new(ELEM_STRING);
    dyn_array_push_string_copy(dynamic, "dynamic");
    Value dyn = {.type = VAL_DYN_ARRAY, .as.dyn_array_val = dynamic};
    (void)call_function("loop_probe", &dyn, 1, env);
    assert(loop_probe_calls == 2 && loop_probe_misses == 0);
    Function *function = env_get_function(env, "loop_probe"); assert(function);
    function->params[0].type = TYPE_LIST_STRING;
    List_string *list = list_string_new(); list_string_push(list, "legacy");
    Value legacy = create_int((int64_t)(intptr_t)list);
    (void)call_function("loop_probe", &legacy, 1, env);
    assert(loop_probe_calls == 3 && loop_probe_misses == 0);
    function->params[0].type = TYPE_ARRAY;
    assert(run_shadow_tests(program, env, false));
    assert(loop_probe_calls == 4 && loop_probe_misses == 0);
    free(((char **)array.as.array_val->data)[0]);
    free(array.as.array_val->data); free(array.as.array_val);
    gc_release(dynamic); list_string_free(list);
    free_environment(env); free_ast(program); free_tokens(tokens, count);
    clear_module_cache(); ffi_loader_shutdown();
}

int main(void) {
    gc_init(); fresh_and_sticky(); aliases_at_every_position();
    borrowed_and_retained_outside_scope(); imported_literal_publication();
    direct_loop_publications();
    puts("I preserve fresh, seen, borrowed, imported, rebound and recycled STRING bindings.");
    gc_shutdown(); return 0;
}
