/* I retain the existing fixture helpers and its original main references. */
#define main nominal_original_metadata_main
#include "test_module_metadata.c"
#undef main

static void namespace_record(Environment *env, const char *owner, const char *canonical) {
    StructDef record = {0};
    record.name = strdup(canonical);
    record.original_name = strdup("Record");
    record.module_name = strdup(owner);
    record.is_pub = true;
    env_define_struct(env, record);
    char **names = malloc(sizeof *names);
    names[0] = strdup("Record");
    env_register_namespace(env, "p", owner, NULL, 0, names, 1, NULL, 0, NULL, 0);
}

int main(void) {
    Environment *env = create_environment();
    env->current_module = "declared";
    namespace_record(env, "types", "__owned_types_Record");
    env->current_module = "filename";
    namespace_record(env, "other", "__owned_other_Record");
    char *saved = strdup("filename");
    env->current_module = saved;
    int count = 0;
    Token *tokens = tokenize("module declared\nfn forward(value: p.Record) -> p.Record { return value }", &count);
    ASSERT_NOT_NULL(tokens);
    ASTNode *program = parse_program(tokens, count);
    ASSERT_NOT_NULL(program);
    ASSERT(bind_nominal_records(program, env));
    ASSERT(env->current_module == saved);
    ASTNode *function = program->as.program.items[1];
    ASSERT(!strcmp(function->as.function.return_struct_type_name, "__owned_types_Record"));
    ASSERT(!strcmp(function->as.function.params[0].struct_type_name, "__owned_types_Record"));
    free_ast(program);
    free_tokens(tokens, count);
    free_environment(env);
    free(saved);

    env = create_environment();
    saved = strdup("filename");
    env->current_module = saved;
    StructDef foreign = {0};
    foreign.name = strdup("Clash");
    foreign.module_name = strdup("other");
    foreign.is_extern = true;
    env_define_struct(env, foreign);
    tokens = tokenize("module declared\nstruct Clash { value: int }", &count);
    ASSERT_NOT_NULL(tokens);
    program = parse_program(tokens, count);
    ASSERT_NOT_NULL(program);
    ASSERT(!bind_nominal_records(program, env));
    ASSERT(env->current_module == saved);
    free_ast(program);
    free_tokens(tokens, count);
    free_environment(env);
    free(saved);
    env = create_environment();
    saved = strdup("filename");
    env->current_module = saved;
    tokens = tokenize("module first_owner\npub struct Record { value: int }\nextern struct Foreign { value: int }\nfn make() -> Record { return Record { value: 42 } }", &count);
    ASSERT_NOT_NULL(tokens);
    program = parse_program(tokens, count);
    ASSERT_NOT_NULL(program);
    ASSERT(bind_nominal_records(program, env));
    ASSERT(env->current_module == saved);
    ASTNode *record = program->as.program.items[1];
    ASSERT(record->as.struct_def.original_name != NULL);
    ASSERT(!strcmp(record->as.struct_def.original_name, "Record"));
    ASSERT(strcmp(record->as.struct_def.name, "Record") != 0);
    ASTNode *foreign_record = program->as.program.items[2];
    ASSERT(foreign_record->as.struct_def.original_name == NULL);
    ASSERT(!strcmp(foreign_record->as.struct_def.name, "Foreign"));
    function = program->as.program.items[3];
    ASSERT(!strcmp(function->as.function.return_struct_type_name, record->as.struct_def.name));
    ASTNode *returned = function->as.function.body->as.block.statements[0]->as.return_stmt.value;
    ASSERT(!strcmp(returned->as.struct_literal.struct_name, record->as.struct_def.name));
    free_ast(program);
    free_tokens(tokens, count);
    free_environment(env);
    free(saved);
    puts("I preserve declared-owner identity and restore context after success and refusal.");
    return 0;
}
