/* I bind service declarations to physical files before aliases or lowering. */
#include "nanolang.h"
#include "nanoisa/file_source_snapshot.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;

static ASTNode *parse(const char *text) {
    int count = 0;
    Token *tokens = tokenize(text, &count);
    assert(tokens);
    ASTNode *program = parse_program(tokens, count);
    free_tokens(tokens, count);
    assert(program);
    return program;
}

static const char *declaration =
    "service \"nsi:nanolang/filesystem\" catalog 1 from \"interface.nsi.json\"\n";

int main(int argc, char **argv) {
    assert(argc == 2);
    char path[4096], alias[4096];
    Environment *env = create_environment();
    ASTNode *plain = parse("fn main() -> int { return 0 }\nshadow main { assert true }\n");
    assert(bind_service_origin(plain, env, NULL));
    assert(env->service_origin_count == 0);
    free_ast(plain);
    ASTNode *first = parse(declaration);
    snprintf(path, sizeof path, "%s/one/binding.nano", argv[1]);
    char *canonical = realpath(path, NULL);
    assert(canonical);
    assert(bind_service_origin(first, env, path));
    assert(first->as.program.items[0]->as.service_decl.origin_index == 0);
    assert(!strcmp(env->service_origins[0], canonical));
    free(canonical);
    snprintf(alias, sizeof alias, "%s/alias.nano", argv[1]);
    assert(bind_service_origin(first, env, alias));
    assert(env->service_origin_count == 1);
    snprintf(path, sizeof path, "%s/two/binding.nano", argv[1]);
    assert(!bind_service_origin(first, env, path));
    assert(first->as.program.items[0]->as.service_decl.origin_index == 0);
    assert(env->service_origin_count == 1);
    ASTNode *second = parse(declaration);
    assert(bind_service_origin(second, env, path));
    assert(second->as.program.items[0]->as.service_decl.origin_index == 1);
    assert(strcmp(env->service_origins[0], env->service_origins[1]));
    assert(env->service_inputs == NULL);
    assert(acquire_service_input(first, env));
    assert(acquire_service_input(second, env));
    assert(nl_file_source_snapshot_count(env->service_inputs) == 2);
    assert(env->service_snapshot_bound[0] && env->service_snapshot_bound[1]);
    size_t bytes = 0;
    const unsigned char *retained = nl_file_source_snapshot_bytes(env->service_inputs,
        env->service_snapshot_indices[0], 1, &bytes);
    assert(retained && bytes > 0);
    unsigned char *copy = malloc(bytes);
    assert(copy); memcpy(copy, retained, bytes);
    snprintf(path, sizeof path, "%s/one/interface.nsi.json", argv[1]);
    FILE *changed = fopen(path, "wb"); assert(changed);
    assert(fwrite("{}", 1, 2, changed) == 2); assert(!fclose(changed));
    assert(acquire_service_input(first, env));
    assert(nl_file_source_snapshot_count(env->service_inputs) == 2);
    assert(!memcmp(copy, retained, bytes));
    ASTNode *node = first->as.program.items[0];
    char *original_path = node->as.service_decl.document_path;
    int64_t original_size = node->as.service_decl.path_bytes;
    node->as.service_decl.document_path = "other.json";
    node->as.service_decl.path_bytes = 10;
    assert(!acquire_service_input(first, env));
    node->as.service_decl.document_path = original_path;
    node->as.service_decl.path_bytes = -1;
    assert(!acquire_service_input(first, env));
    node->as.service_decl.path_bytes = original_size;
    node->as.service_decl.origin_index = INT64_C(4294967296);
    assert(!acquire_service_input(first, env));
    node->as.service_decl.origin_index = 0;
    assert(acquire_service_input(first, env));
    changed = fopen(path, "wb"); assert(changed);
    assert(fwrite(copy, 1, bytes, changed) == bytes); assert(!fclose(changed)); free(copy);
    for (int i = 2; i < 16; ++i) {
        snprintf(path, sizeof path, "%s/%d.nano", argv[1], i);
        ASTNode *next = parse(declaration);
        assert(bind_service_origin(next, env, path));
        assert(next->as.program.items[0]->as.service_decl.origin_index == i);
        free_ast(next);
    }
    ASTNode *extra = parse(declaration);
    snprintf(path, sizeof path, "%s/16.nano", argv[1]);
    assert(!bind_service_origin(extra, env, path));
    assert(extra->as.program.items[0]->as.service_decl.origin_index == -1);
    assert(env->service_origin_count == 16);
    free_ast(extra);
    free_ast(second);
    free_ast(first);
    free_environment(env);

    env = create_environment();
    char duplicates[512];
    snprintf(duplicates, sizeof duplicates, "%s%s", declaration, declaration);
    ASTNode *duplicate = parse(duplicates);
    assert(!bind_service_origin(duplicate, env, path));
    assert(env->service_origin_count == 0);
    assert(duplicate->as.program.items[0]->as.service_decl.origin_index == -1);
    assert(duplicate->as.program.items[1]->as.service_decl.origin_index == -1);
    free_ast(duplicate);
    ASTNode *missing = parse(declaration);
    assert(!bind_service_origin(missing, env, "/absent-nanolang-origin/file.nano"));
    assert(env->service_origin_count == 0);
    free_ast(missing);

    /* I exercise the real recursive import loader, not a second graph reader.
     * Service lowering remains unsupported, but its retained origin is exact. */
    ASTNode *root = parse("module \"bridge.nano\" as wrapper\n");
    snprintf(path, sizeof path, "%s/main.nano", argv[1]);
    assert(!process_imports(root, env, NULL, path));
    assert(env->service_origin_count == 1);
    assert(nl_file_source_snapshot_count(env->service_inputs) == 1);
    assert(env->service_snapshot_bound[0]);
    snprintf(path, sizeof path, "%s/one/binding.nano", argv[1]);
    canonical = realpath(path, NULL);
    assert(canonical && !strcmp(canonical, env->service_origins[0]));
    free(canonical);
    ASTNode *unavailable = parse(declaration);
    snprintf(path, sizeof path, "%s/two/binding.nano", argv[1]);
    assert(bind_service_origin(unavailable, env, path));
    snprintf(path, sizeof path, "%s/two/interface.nsi.json", argv[1]);
    assert(!remove(path));
    assert(!acquire_service_input(unavailable, env));
    assert(nl_file_source_snapshot_count(env->service_inputs) == 1);
    assert(!env->service_snapshot_bound[1]);
    free_ast(unavailable);
    free_ast(root);
    free_environment(env);
    clear_module_cache();
    puts("PASS service origins: canonical identity, capacity, atomicity, transitive loader");
    return 0;
}
