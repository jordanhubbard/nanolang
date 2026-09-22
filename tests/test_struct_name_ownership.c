/* I exercise the real lookup and Environment owner without parsed metadata leaks. */
#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int g_argc;
char **g_argv;
extern const char *get_struct_type_name(ASTNode *, Environment *);
static size_t calls, fail_at;
static int persistent;
static int refuse(void) {
    ++calls;
    return fail_at && (calls == fail_at || (persistent && calls >= fail_at));
}
void *struct_name_test_malloc(size_t size) { return refuse() ? NULL : malloc(size); }
char *struct_name_test_strdup(const char *text) { return refuse() ? NULL : strdup(text); }

/* I arm these independent counters only around the real snapshot extractor. */
static int metadata_armed, metadata_persistent;
static size_t metadata_calls, metadata_fail;
static unsigned metadata_kinds[256];
static int metadata_refuse(unsigned kind) {
    if (!metadata_armed) return 0;
    assert(metadata_calls<sizeof(metadata_kinds)/sizeof(metadata_kinds[0]));
    metadata_kinds[metadata_calls++]=kind;
    return metadata_fail && (metadata_calls==metadata_fail ||
        (metadata_persistent && metadata_calls>=metadata_fail));
}
void *struct_metadata_test_calloc(size_t count,size_t width) {
    return metadata_refuse(0) ? NULL : calloc(count,width);
}
char *struct_metadata_test_strdup(const char *name) {
    return metadata_refuse(1) ? NULL : strdup(name);
}
void *struct_payload_test_calloc(size_t count,size_t width) {
    return metadata_refuse(2) ? NULL : calloc(count,width);
}

#include "struct_ownership_worker.h"

static size_t lookup_case(int kind, size_t failure, int mode) {
    Environment *env = create_environment();
    assert(env && !env->struct_count && !env->union_count && !env->symbol_count);
    char spelling[] = "Owner.Target";
    char *field_names[] = {"child"};
    Type field_types[] = {TYPE_STRUCT};
    char *field_type_names[] = {spelling};
    StructDef outer = {.name="Outer", .field_count=1, .field_names=field_names,
        .field_types=field_types, .field_type_names=field_type_names};
    env->structs[0] = outer;
    env->structs[1] = (StructDef){.name="Target_Name"};
    env->struct_count = 2;
    char *variant_names[] = {"Some"};
    int counts[] = {1};
    char **variant_fields[] = {field_names};
    Type *variant_types[] = {field_types};
    char *parameter_names[] = {"T"};
    char *generic_field_names[] = {"T"};
    char **variant_type_names[] = {kind == 2 ? generic_field_names : field_type_names};
    env->unions[0] = (UnionDef){.name="Choice", .variant_count=1,
        .variant_names=variant_names, .variant_field_counts=counts,
        .variant_field_names=variant_fields, .variant_field_types=variant_types,
        .variant_field_type_names=variant_type_names,
        .generic_params=parameter_names, .generic_param_count=kind == 2};
    env->union_count = 1;
    TypeInfo concrete = {.base_type=TYPE_STRUCT, .generic_name=spelling};
    TypeInfo *parameters[] = {&concrete};
    TypeInfo instance = {.base_type=TYPE_UNION, .type_param_count=1, .type_params=parameters};
    env->symbols[0] = (Symbol){.name="object", .type=kind == 3 ? TYPE_STRUCT : TYPE_UNION,
        .struct_type_name=kind == 3 ? "Outer" : "Choice.Some", .type_info=&instance};
    env->symbol_count = 1;
    ASTNode object = {.type=AST_IDENTIFIER}; object.as.identifier="object";
    ASTNode expr = {.type=kind == 0 ? AST_CALL : AST_FIELD_ACCESS};
    if (kind == 0) expr.as.call.name="List_Target_Name_get";
    else { expr.as.field_access.object=&object; expr.as.field_access.field_name="child"; }
    void *before = env->checker_allocations;
    calls=0; fail_at=failure; persistent=mode;
    const char *result=get_struct_type_name(&expr, env);
    size_t measured=calls;
    fail_at=0;
    if (failure) {
        assert(!result);
        assert(env->checker_allocations == before);
    } else {
        assert(result && !strcmp(result, kind == 0 ? "Target_Name" : spelling));
        assert(env->checker_allocations != before);
        const char *again=get_struct_type_name(&expr, env);
        assert(again && again != result && !strcmp(again, result));
        spelling[0]='X';
        assert(!strcmp(result, kind == 0 ? "Target_Name" : "Owner.Target"));
        assert(!strcmp(again, result));
        /* Borrowed identifier names are never registered or freed as copies. */
        before=env->checker_allocations;
        assert(get_struct_type_name(&object, env) == env->symbols[0].struct_type_name);
        assert(env->checker_allocations == before);
    }
    /* The fixture owns its stack-backed descriptors; the real registry still
     * destroys every lookup copy independently of these removed slots. */
    env->struct_count=env->union_count=env->symbol_count=0;
    free_environment(env);
    return measured;
}

static void parsed_parameter_names(void) {
    int count=0;
    Token *tokens=tokenize("fn names(a: Child, b: Owner.Child)->int { return 0 }", &count);
    assert(tokens);
    ASTNode *program=parse_program(tokens,count); assert(program);
    assert(program->as.program.count==1);
    ASTNode *function=program->as.program.items[0]; assert(function->type==AST_FUNCTION);
    assert(function->as.function.param_count==2);
    assert(!strcmp(function->as.function.params[0].struct_type_name,"Child"));
    assert(!strcmp(function->as.function.params[1].struct_type_name,"Owner.Child"));
    free_tokens(tokens,count);
    assert(!strcmp(function->as.function.params[1].struct_type_name,"Owner.Child"));
    free_ast(program);
}

static void parsed_record_lifetimes(int ast_first) {
    const char *source="struct Child { value: int } struct Outer { child: Child, values: array<int> } "
        "fn read(value: Child)->int { return value.value } fn main()->int { return 0 }";
    int count=0; Token *tokens=tokenize(source,&count); assert(tokens);
    ASTNode *program=parse_program(tokens,count); assert(program);
    Environment *env=create_environment(); assert(env); env->suppress_shadow_warnings=true;
    typecheck_set_current_file("<struct-name-ownership>");
    assert(type_check(program,env));
    StructDef *outer=env_get_struct(env,"Outer"); assert(outer);
    assert(outer->field_type_names && !strcmp(outer->field_type_names[0],"Child"));
    assert(outer->field_element_types[1]==TYPE_INT);
    if (ast_first) {
        free_ast(program); free_tokens(tokens,count);
        assert(!strcmp(outer->field_type_names[0],"Child"));
        assert(outer->field_element_types[1]==TYPE_INT);
        free_environment(env);
    } else {
        free_environment(env);
        ASTNode *record=program->as.program.items[1];
        assert(record->type==AST_STRUCT_DEF);
        assert(!strcmp(record->as.struct_def.field_type_names[0],"Child"));
        assert(record->as.struct_def.field_element_types[1]==TYPE_INT);
        free_ast(program); free_tokens(tokens,count);
    }
}

static void auxiliary_vectors(void) {
    for (int fields=0; fields<2; ++fields) {
        Environment *env=create_environment(); assert(env);
        TypeInfo borrowed={.base_type=TYPE_BOOL}; TypeInfo *annotations[]={&borrowed};
        StructDef record={.name=strdup("Auxiliary"),.field_count=fields,
            .field_type_info=annotations,.field_names=calloc(1,sizeof(char *)),
            .field_types=calloc(1,sizeof(Type)),.field_type_names=calloc(1,sizeof(char *)),
            .field_element_types=calloc(1,sizeof(Type))};
        assert(record.name && record.field_names && record.field_types &&
               record.field_type_names && record.field_element_types);
        if (fields) {
            record.field_names[0]=strdup("value"); record.field_type_names[0]=strdup("Child");
            assert(record.field_names[0] && record.field_type_names[0]);
        }
        env_define_struct(env,record); free_environment(env);
        assert(borrowed.base_type==TYPE_BOOL && annotations[0]==&borrowed);
    }
}

static void metadata_snapshot_lifetimes(int snapshot_first) {
    const char *source="struct Child { value: int } struct Outer { child: Child, values: array<int> } "
        "fn main()->int { return 0 }";
    int count=0; Token *tokens=tokenize(source,&count); assert(tokens);
    ASTNode *program=parse_program(tokens,count); assert(program);
    Environment *env=create_environment(); assert(env); env->suppress_shadow_warnings=true;
    assert(type_check(program,env));
    StructDef *outer=env_get_struct(env,"Outer"); assert(outer);
    free(outer->original_name); outer->original_name=strdup("WrittenOuter");
    outer->module_name=env_own_checker_allocation(env,strdup("Owner"));
    ModuleMetadata *snapshot=extract_module_metadata(env,"Owner"); assert(snapshot);
    StructDef *copy=NULL;
    for (int i=0;i<snapshot->struct_count;++i)
        if (!strcmp(snapshot->structs[i].name,"Outer")) copy=&snapshot->structs[i];
    assert(copy && copy->original_name!=outer->original_name && copy->module_name!=outer->module_name);
    assert(copy->field_names!=outer->field_names && copy->field_types!=outer->field_types);
    assert(copy->field_type_names!=outer->field_type_names && copy->field_element_types!=outer->field_element_types);
    assert(copy->field_type_info!=outer->field_type_info);
    if (snapshot_first) {
        free_module_metadata(snapshot);
        assert(!strcmp(outer->original_name,"WrittenOuter"));
        assert(outer->field_type_info[1]->element_type->base_type==TYPE_INT);
        free_environment(env); free_ast(program); free_tokens(tokens,count);
    } else {
        free_ast(program); free_tokens(tokens,count); free_environment(env);
        assert(!strcmp(copy->name,"Outer") && !strcmp(copy->original_name,"WrittenOuter"));
        assert(!strcmp(copy->module_name,"Owner") && !strcmp(copy->field_names[0],"child"));
        assert(copy->field_types[0]==TYPE_STRUCT && !strcmp(copy->field_type_names[0],"Child"));
        assert(copy->field_element_types[1]==TYPE_INT);
        assert(copy->field_type_info[1]->base_type==TYPE_ARRAY);
        assert(copy->field_type_info[1]->element_type->base_type==TYPE_INT);
        free_module_metadata(snapshot);
    }
}

static void metadata_callback_annotation(void) {
    char name[]="ExactOwner.Child";
    TypeInfo child={.base_type=TYPE_STRUCT,.generic_name=name};
    TypeInfo array={.base_type=TYPE_ARRAY,.element_type=&child};
    TypeInfo *parameters[]={&array}; Type parameter_types[]={TYPE_ARRAY};
    FunctionSignature signature={.param_count=1,.param_types=parameter_types,
        .param_type_info=parameters,.return_type=TYPE_STRUCT,.return_struct_name=name,
        .return_type_info=&child};
    TypeInfo callback={.base_type=TYPE_FUNCTION,.fn_sig=&signature};
    Type tuple_types[]={TYPE_FUNCTION,TYPE_ARRAY}; char *tuple_names[]={NULL,NULL};
    TypeInfo *children[]={&callback,&array};
    TypeInfo tuple={.base_type=TYPE_TUPLE,.type_params=children,.type_param_count=2,
        .tuple_types=tuple_types,.tuple_type_names=tuple_names,.tuple_element_count=2};
    TypeInfo *annotations[]={&tuple};
    Environment *env=create_environment(); assert(env);
    StructDef record={.name=strdup("CallbackHolder"),.field_count=1,.field_type_info=annotations,
        .field_names=calloc(1,sizeof(char *)),.field_types=calloc(1,sizeof(Type))};
    assert(record.name && record.field_names && record.field_types);
    record.field_names[0]=strdup("callback"); assert(record.field_names[0]);
    record.field_types[0]=TYPE_TUPLE; env_define_struct(env,record);
    ModuleMetadata *snapshot=extract_module_metadata(env,"ExactOwner"); assert(snapshot);
    TypeInfo *tuple_copy=snapshot->structs[0].field_type_info[0];
    assert(tuple_copy!=&tuple && tuple_copy->type_params!=children && tuple_copy->tuple_types!=tuple_types);
    assert(tuple_copy->tuple_type_names!=tuple_names && tuple_copy->type_param_count==2);
    assert(tuple_copy->tuple_element_count==2 && tuple_copy->tuple_types[0]==TYPE_FUNCTION);
    assert(tuple_copy->type_params[1]!=&array && tuple_copy->type_params[1]->element_type!=&child);
    TypeInfo *copy=tuple_copy->type_params[0];
    assert(copy!=&callback && copy->fn_sig!=&signature);
    assert(copy->fn_sig->param_type_info[0]!=&array);
    assert(copy->fn_sig->param_type_info[0]->element_type!=&child);
    free_environment(env); name[0]='X'; child.base_type=TYPE_BOOL;
    assert(copy->fn_sig->param_types[0]==TYPE_ARRAY);
    assert(copy->fn_sig->param_type_info[0]->element_type->base_type==TYPE_STRUCT);
    assert(!strcmp(copy->fn_sig->param_type_info[0]->element_type->generic_name,"ExactOwner.Child"));
    assert(!strcmp(copy->fn_sig->return_struct_name,"ExactOwner.Child"));
    assert(copy->fn_sig->return_type_info->base_type==TYPE_STRUCT);
    assert(!strcmp(copy->fn_sig->return_type_info->generic_name,"ExactOwner.Child"));
    free_module_metadata(snapshot);
    assert(callback.fn_sig==&signature && signature.param_type_info==parameters);
}

static void metadata_empty_vectors(void) {
    for (int allocated=0;allocated<2;++allocated) {
        Environment *env=create_environment(); assert(env);
        StructDef record={.name=strdup("Empty")}; assert(record.name);
        if (allocated) {
            record.field_names=calloc(1,sizeof(char *)); record.field_types=calloc(1,sizeof(Type));
            record.field_type_names=calloc(1,sizeof(char *)); record.field_element_types=calloc(1,sizeof(Type));
        }
        env_define_struct(env,record);
        ModuleMetadata *snapshot=extract_module_metadata(env,"EmptyOwner"); assert(snapshot);
        StructDef *copy=&snapshot->structs[0]; assert(copy->field_count==0);
        assert(!!copy->field_names==allocated && !!copy->field_types==allocated);
        assert(!!copy->field_type_names==allocated && !!copy->field_element_types==allocated);
        assert(!copy->field_type_info && !copy->module_name && !copy->original_name);
        free_environment(env); assert(!strcmp(copy->name,"Empty")); free_module_metadata(snapshot);
    }
}

static void checker_module_name_ownership(void) {
    const char *sources[] = {
        "module First struct A{value:int} enum E{One}",
        "module Second struct B{value:int} enum F{One}"
    };
    Environment *env = create_environment(); assert(env);
    env->current_module = "BorrowedPrior";
    ASTNode *programs[2]; Token *tokens[2]; int counts[2];
    char *first_context = NULL;
    for (int i = 0; i < 2; ++i) {
        tokens[i] = tokenize(sources[i], &counts[i]); assert(tokens[i]);
        programs[i] = parse_program(tokens[i], counts[i]); assert(programs[i]);
        assert(type_check_module(programs[i], env));
        if (!i) first_context = env->current_module;
    }
    assert(!strcmp(first_context, "First") && !strcmp(env->current_module, "Second"));
    assert(env->struct_count == 2 && env->enum_count == 2);
    ModuleMetadata *meta = extract_module_metadata(env, "Snapshot"); assert(meta);
    assert(meta->struct_count == 2 && meta->enum_count == 2);
    for (int i = 0; i < 2; ++i) {
        assert(meta->structs[i].module_name != env->structs[i].module_name);
        assert(meta->enums[i].module_name != env->enums[i].module_name);
    }
    free_environment(env);
    for (int i = 0; i < 2; ++i) { free_ast(programs[i]); free_tokens(tokens[i], counts[i]); }
    for (int i = 0; i < 2; ++i) {
        assert(!strcmp(meta->structs[i].module_name, i ? "Second" : "First"));
        assert(!strcmp(meta->enums[i].module_name, i ? "Second" : "First"));
    }
    free_module_metadata(meta);
}

static const StructDef *fatal_source;
static StructDef fatal_before;
static ModuleMetadata *fatal_published;
static void verify_fatal_snapshot(void) {
    assert(fatal_source && !memcmp(fatal_source,&fatal_before,sizeof(fatal_before)));
    assert(!fatal_published);
    assert(!strcmp(fatal_source->original_name,"WrittenHolder"));
    assert(!strcmp(fatal_source->field_type_info[0]->fn_sig->return_struct_name,"ExactOwner.Child"));
}
static size_t metadata_fault_case(size_t position,int mode) {
    TypeInfo child={.base_type=TYPE_STRUCT,.generic_name="ExactOwner.Child"};
    TypeInfo array={.base_type=TYPE_ARRAY,.element_type=&child};
    TypeInfo *parameters[]={&array}; Type parameter_types[]={TYPE_ARRAY};
    FunctionSignature signature={.param_count=1,.param_types=parameter_types,
        .param_type_info=parameters,.return_type=TYPE_STRUCT,
        .return_struct_name="ExactOwner.Child",.return_type_info=&child};
    TypeInfo callback={.base_type=TYPE_FUNCTION,.fn_sig=&signature};
    TypeInfo *annotations[]={&callback}; char *names[]={"callback"};
    char *type_names[]={"ExactOwner.Child"}; Type types[]={TYPE_FUNCTION},elements[]={TYPE_UNKNOWN};
    Environment *env=create_environment(); assert(env && !env->function_count && !env->struct_count);
    env->structs[0]=(StructDef){.name="Holder",.original_name="WrittenHolder",.module_name="ExactOwner",
        .field_count=1,.field_names=names,.field_types=types,.field_type_names=type_names,
        .field_element_types=elements,.field_type_info=annotations};
    env->struct_count=1;
    StructDef before; memcpy(&before,&env->structs[0],sizeof(before));
    if (position) {
        fatal_source=&env->structs[0]; memcpy(&fatal_before,fatal_source,sizeof(fatal_before));
        fatal_published=NULL; assert(atexit(verify_fatal_snapshot)==0);
    }
    metadata_calls=0; metadata_fail=position; metadata_persistent=mode; metadata_armed=1;
    ModuleMetadata *snapshot=extract_module_metadata(env,"ExactOwner");
    metadata_armed=0;
    if (position) { fatal_published=snapshot; _Exit(88); }
    size_t measured=metadata_calls;
    assert(snapshot && !memcmp(&before,&env->structs[0],sizeof(before)));
    assert(!strcmp(snapshot->structs[0].field_type_info[0]->fn_sig->return_struct_name,"ExactOwner.Child"));
    free_module_metadata(snapshot); env->struct_count=0; free_environment(env);
    return measured;
}
static void metadata_fault_positions(void) {
    size_t count=metadata_fault_case(0,0); assert(count>15 && count<256);
    unsigned kinds[256]; memcpy(kinds,metadata_kinds,sizeof(kinds));
    const char *messages[]={"I cannot allocate a module metadata copy.\n",
        "I cannot copy a module metadata owner.\n","I cannot allocate payload type metadata\n"};
    for (int mode=0;mode<2;++mode) for (size_t pos=1;pos<=count;++pos) {
        int errors[2]; assert(pipe(errors)==0);
        pid_t child=spawn_fault_case(errors,0,0,pos,mode);
        close(errors[1]); read_child_diagnostic(errors[0],child,messages[kinds[pos-1]]);
        int status=0;
        assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==1);
        assert(metadata_fault_case(0,0)==count);
    }
    printf("Struct snapshot allocation positions: %zu, two failure modes and fresh recovery PASS\n",count);
}

int main(int argc,char **argv) {
    fixture_executable=argv[0];
    unsigned kind=0,position=0,mode=0;
    if (argc==5 && !strcmp(argv[1],"_lookup_fault") &&
        fixture_number(argv[2],3,&kind) && fixture_number(argv[3],kind==3 ? 2 : 3,&position) &&
        position && fixture_number(argv[4],1,&mode)) {
        (void)lookup_case((int)kind,position,(int)mode); return 0;
    }
    if (argc==4 && !strcmp(argv[1],"_snapshot_fault") &&
        fixture_number(argv[2],255,&position) && position && fixture_number(argv[3],1,&mode)) {
        (void)metadata_fault_case(position,(int)mode); return 89;
    }
    if (argc>2 || (argc==2 && strcmp(argv[1],"snapshot"))) {
        fprintf(stderr,"I accept only the optional snapshot fixture selector.\n"); return 2;
    }
    int snapshot_only=argc==2;
    if (!snapshot_only) {
    for (int kind=0; kind<4; ++kind) {
        size_t count=lookup_case(kind,0,0);
        assert(count == (kind == 3 ? 2u : 3u));
        for (int mode=0; mode<2; ++mode) for (size_t pos=1; pos<=count; ++pos) {
            int errors[2]; assert(pipe(errors)==0);
            pid_t child=spawn_fault_case(errors,1,kind,pos,mode);
            close(errors[1]);
            read_child_diagnostic(errors[0],child,pos == count ?
                "I could not allocate checker ownership metadata\n" : "");
            int status=0; assert(waitpid(child,&status,0)==child && WIFEXITED(status));
            assert(WEXITSTATUS(status) == (pos == count ? 1 : 0));
            (void)lookup_case(kind,0,0);
        }
    }
    parsed_parameter_names(); parsed_record_lifetimes(0); parsed_record_lifetimes(1); auxiliary_vectors();
    puts("Parser/record ownership: qualified parameters, both destruction orders, zero/nonzero auxiliary vectors and borrowed annotations PASS");
    puts("Struct name ownership: four paths, exact copies, borrowed controls, all allocation positions/two modes/recovery PASS");
    }
    metadata_snapshot_lifetimes(0); metadata_snapshot_lifetimes(1); metadata_empty_vectors(); metadata_callback_annotation(); metadata_fault_positions(); checker_module_name_ownership();
    puts("Struct metadata snapshot ownership: lifetime, complete annotations, module owners and all allocation positions PASS");
    return 0;
}
