/* I exercise the real lookup and Environment owner without parsed metadata leaks. */
#define _POSIX_C_SOURCE 200809L
#include "../src/nanolang.h"
#include <assert.h>
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

int main(void) {
    for (int kind=0; kind<4; ++kind) {
        size_t count=lookup_case(kind,0,0);
        assert(count == (kind == 3 ? 2u : 3u));
        for (int mode=0; mode<2; ++mode) for (size_t pos=1; pos<=count; ++pos) {
            pid_t child=fork(); assert(child>=0);
            if (!child) { (void)lookup_case(kind,pos,mode); exit(0); }
            int status=0; assert(waitpid(child,&status,0)==child && WIFEXITED(status));
            /* The last allocation is the existing fatal registry metadata path. */
            assert(WEXITSTATUS(status) == (pos == count ? 1 : 0));
            (void)lookup_case(kind,0,0);
        }
    }
    puts("Struct name ownership: four paths, exact copies, borrowed controls, all allocation positions/two modes/recovery PASS");
    return 0;
}
