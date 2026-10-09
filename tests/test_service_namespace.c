/* I inspect the namespace retained by my actual recursive C loader. */
#include "nanolang.h"
#include "service_namespace.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* I fail each allocation made by the namespace builder, after parsing/loading. */
static long allocations_before_failure = -1;
static void *namespace_malloc(size_t size) {
    if (allocations_before_failure == 0) return NULL;
    if (allocations_before_failure > 0) --allocations_before_failure;
    return malloc(size);
}
static void *namespace_calloc(size_t count, size_t size) {
    if (allocations_before_failure == 0) return NULL;
    if (allocations_before_failure > 0) --allocations_before_failure;
    return calloc(count, size);
}
#define malloc namespace_malloc
#define calloc namespace_calloc
#include "../src/service_namespace.c"
#undef malloc
#undef calloc

int g_argc;
char **g_argv;
int main(int argc, char **argv) {
    assert(argc >= 3);
    FILE *file = fopen(argv[1], "rb"); assert(file);
    assert(!fseek(file, 0, SEEK_END)); long size = ftell(file); assert(size >= 0);
    rewind(file); char *text = calloc((size_t)size + 1, 1); assert(text);
    assert(fread(text, 1, (size_t)size, file) == (size_t)size); fclose(file);
    int count = 0; Token *tokens = tokenize(text, &count); assert(tokens);
    ASTNode *root = parse_program(tokens, count); assert(root);
    free_tokens(tokens, count); free(text);
    Environment *env = create_environment();
    ModuleList *modules = create_module_list();
    /* I retain the execution guard until source checking and lowering exist. */
    assert(!process_imports(root, env, modules, argv[1]));
    NlServiceNamespace *space = env->service_namespace;
    if (!strcmp(argv[2], "reject")) { assert(!space); }
    else {
        assert(space && nl_service_namespace_plan(space));
        char *owner = realpath(argv[1], NULL); assert(owner);
        printf("PLAN %zu\n", nl_file_source_plan_count(nl_service_namespace_plan(space)));
        for (int i = 3; i < argc; ++i) {
            const NlServiceName *name = nl_service_namespace_lookup(space, owner, argv[i]);
            if (!name) { printf("MISSING %s\n", argv[i]); continue; }
            const NlServiceName *origin = nl_service_namespace_name(space, name->target - 1);
            assert(origin);
            printf("NAME %s %u %s %s\n", argv[i], name->kind,
                nl_service_namespace_module(space, origin->module), origin->name);
        }
        /* Failed construction must preserve the caller's published pointer. */
        NlServiceNamespace *prior = space;
        assert(nl_service_namespace_build(root, env, NULL, argv[1], &prior) != NL_FILE_SOURCE_OK);
        assert(prior == space);
        free(owner);
        bool reached_success = false;
        for (long prefix = 0; prefix < 1024; ++prefix) {
            NlServiceNamespace *candidate = space;
            allocations_before_failure = prefix;
            NlFileSourceStatus status = nl_service_namespace_build(root, env, modules, argv[1], &candidate);
            allocations_before_failure = -1;
            if (status == NL_FILE_SOURCE_OK) {
                assert(candidate != space);
                nl_service_namespace_free(candidate);
                reached_success = true;
                break;
            }
            assert(status == NL_FILE_SOURCE_MEMORY && candidate == space);
        }
        assert(reached_success);
        /* A new failing top-level load must not reuse these resolved facts. */
        assert(!process_imports(root, env, modules, "/missing/namespace/root.nano"));
        assert(!env->service_namespace);
    }
    free_environment(env); free_module_list(modules); free_ast(root); clear_module_cache();
    puts("PASS actual C namespace");
    return 0;
}
