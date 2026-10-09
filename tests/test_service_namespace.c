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
        for (size_t i = 0; i < nl_service_namespace_count(space); ++i) {
            const NlServiceName *row = nl_service_namespace_name(space, i);
            const char *source = nl_service_namespace_module(space, row->module);
            if (row->kind == NL_SERVICE_TYPE) {
                TypeInfo type;
                assert(nl_service_type(space, source, row->name, &type));
                assert(type.service_declaration == row->target);
                TypeInfo *copy = copy_payload_type_info(&type);
                assert(type_infos_equal(&type, copy));
                copy->generic_name = strdup("visible_alias");
                assert(type_infos_equal(&type, copy));
                ++copy->service_declaration;
                assert(!type_infos_equal(&type, copy));
                free_payload_type_info(copy);
                TypeInfo ordinary = {.base_type = type.base_type};
                assert(!type_infos_equal(&type, &ordinary));
                int64_t members = nl_file_source_catalog_number(1, row->ordinal, 1, 0);
                for (int64_t member = 0; member < members; ++member) {
                    TypeInfo payload;
                    const char *name = nl_file_source_catalog_string(1, row->ordinal, 3, member);
                    assert(nl_service_member_type(space, &type, name, &payload));
                    const char *id = nl_file_source_catalog_string(1, row->ordinal, 4, member);
                    if (!*id) assert(payload.base_type == TYPE_VOID && !payload.service_declaration);
                    else if (!strcmp(id, "nsi:core/int")) assert(payload.base_type == TYPE_INT);
                    else if (!strcmp(id, "nsi:core/bool")) assert(payload.base_type == TYPE_BOOL);
                    else assert(payload.service_declaration && payload.service_module == type.service_module);
                    TypeInfo forged = type;
                    ++forged.service_declaration;
                    TypeInfo unchanged = payload;
                    assert(!nl_service_member_type(space, &forged, name, &payload));
                    assert(type_infos_equal(&unchanged, &payload));
                }
                TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = &type};
                TypeInfo *nested = copy_payload_type_info(&array);
                assert(type_infos_equal(&array, nested));
                ++nested->element_type->service_module;
                assert(!type_infos_equal(&array, nested));
                free_payload_type_info(nested);
            } else if (row->kind == NL_SERVICE_METHOD) {
                NlServiceSignature signature;
                assert(nl_service_method_type(space, source, row->name, &signature));
                assert(signature.declaration == row->target);
                assert(signature.result.service_module == row->target_module);
                assert(signature.parameter_count == (row->ordinal == 0 ? 0u : row->ordinal == 1 ? 2u : 1u));
                assert(signature.input_mode == (row->ordinal == 0 ? 0u : row->ordinal == 4 ? 2u : 1u));
                if (signature.parameter_count) assert(signature.parameters[0].service_ordinal == 0);
            } else {
                if (row->kind == NL_SERVICE_FUNCTION && !strcmp(row->name, "preserve")) {
                    ASTNode *function = row->declaration;
                    TypeInfo *parameter = function->as.function.params[0].type_info;
                    assert(parameter && parameter->service_declaration);
                    assert(type_infos_equal(parameter, function->as.function.return_type_info));
                    ASTNode *local = function->as.function.body->as.block.statements[0];
                    assert(local->type == AST_LET);
                    assert(type_infos_equal(parameter, local->as.let.type_info));
                }
                if (row->kind == NL_SERVICE_FUNCTION && !strcmp(row->name, "factory")) {
                    FunctionSignature *signature = row->declaration->as.function.return_fn_sig;
                    assert(signature && signature->param_type_info && signature->param_type_info[0] && signature->param_type_info[0]->service_declaration);
                    assert(type_infos_equal(signature->param_type_info[0], signature->return_type_info));
                    FunctionSignature other = *signature;
                    TypeInfo changed = *signature->return_type_info;
                    other.return_type_info = &changed;
                    assert(function_signatures_equal(signature, &other));
                    ++changed.service_declaration;
                    assert(!function_signatures_equal(signature, &other));
                }
                if (row->kind == NL_SERVICE_FUNCTION && !strcmp(row->name, "observe")) {
                    Parameter *parameter = &row->declaration->as.function.params[0];
                    assert(parameter->type == TYPE_BORROW_MUT);
                    assert(parameter->type_info->element_type->service_declaration);
                }
                if (row->kind == NL_SERVICE_UNION && !strcmp(row->name, "Envelope")) {
                    ASTNode *node = row->declaration;
                    for (int arm = 0; arm < 2; ++arm) {
                        TypeInfo *payload = node->as.union_def.variant_field_type_info[arm][0];
                        assert(payload && payload->service_declaration);
                        assert(payload->service_ordinal == (arm == 0 ? 0u : 3u));
                        assert(node->as.union_def.variant_field_types[arm][0] == payload->base_type);
                    }
                }
                if (row->kind == NL_SERVICE_UNION && !strcmp(row->name, "Generic")) {
                    TypeInfo *formal = row->declaration->as.union_def.variant_field_type_info[0][0];
                    assert(formal && !formal->service_declaration && !strcmp(formal->generic_name, "Handle"));
                }
                if (row->kind == NL_SERVICE_RECORD && !strcmp(row->name, "Wrapper")) {
                    TypeInfo **fields = row->declaration->as.struct_def.field_type_info;
                    assert(fields[0]->element_type->service_declaration);
                    assert(fields[1]->fn_sig->param_type_info[0]->service_declaration);
                }
                TypeInfo sentinel = {.base_type = TYPE_INT};
                assert(!nl_service_type(space, source, row->name, &sentinel));
                assert(sentinel.base_type == TYPE_INT);
            }
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
