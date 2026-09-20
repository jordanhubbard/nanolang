/* I preserve the first affine contract-allocation error and recover cleanly. */
#include "nanolang.h"
#include "nanovirt/codegen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;

static size_t allocation_attempts;
static size_t fail_at;

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NANOVIRT_CONTRACT_TEST_HAS_LSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define NANOVIRT_CONTRACT_TEST_HAS_LSAN 1
#endif
#ifdef NANOVIRT_CONTRACT_TEST_HAS_LSAN
extern void __lsan_disable(void);
extern void __lsan_enable(void);
static void frontend_leak_scope_begin(void) { __lsan_disable(); }
static void frontend_leak_scope_end(void) { __lsan_enable(); }
#else
static void frontend_leak_scope_begin(void) {}
static void frontend_leak_scope_end(void) {}
#endif

void *nanovirt_test_contract_realloc(void *pointer,size_t size) {
    allocation_attempts++;
    if(fail_at && allocation_attempts==fail_at)return NULL;
    return realloc(pointer,size);
}

static void require(bool condition,const char *message) {
    if(condition)return;
    fprintf(stderr,"FAIL: %s\n",message);exit(1);
}

static char *read_source(const char *path) {
    FILE *file=fopen(path,"rb");require(file!=NULL,"I open the scalar-union source fixture");
    require(!fseek(file,0,SEEK_END),"I seek the scalar-union source fixture");
    long length=ftell(file);require(length>=0,"I measure the scalar-union source fixture");
    require(!fseek(file,0,SEEK_SET),"I rewind the scalar-union source fixture");
    char *source=malloc((size_t)length+1);require(source!=NULL,"I allocate source text");
    require(fread(source,1,(size_t)length,file)==(size_t)length,"I read the complete source fixture");
    source[length]='\0';require(!fclose(file),"I close the scalar-union source fixture");return source;
}

static CodegenResult compile_fixture(void) {
    /* I exclude the separately tracked checker binding leak, then restore LSan
     * before the first contract allocation and leave ASan/UBSan active. */
    frontend_leak_scope_begin();
    char *source=read_source("tests/nanoisa/fixtures/affine_scalar_union_instances.nano");
    int token_count=0;Token *tokens=tokenize(source,&token_count);
    require(tokens!=NULL,"I tokenize the scalar-union source fixture");
    ASTNode *program=parse_program(tokens,token_count);
    require(program!=NULL,"I parse the scalar-union source fixture");
    Environment *environment=create_environment();require(environment!=NULL,"I allocate the checked environment");
    environment->suppress_shadow_warnings=true;
    require(type_check(program,environment),"I type-check the scalar-union source fixture");
    frontend_leak_scope_end();
    CodegenResult result=codegen_compile(program,environment,NULL,NULL);
    free_environment(environment);free_ast(program);free_tokens(tokens,token_count);free(source);
    return result;
}

int main(void) {
    allocation_attempts=0;fail_at=0;
    CodegenResult baseline=compile_fixture();
    require(baseline.ok && baseline.module,"I publish the complete baseline contract");
    size_t sites=allocation_attempts;nvm_module_free(baseline.module);
    require(sites>=2,"I reach contract growth after the initial reservation");

    for(size_t failure=1;failure<=sites;failure++) {
        allocation_attempts=0;fail_at=failure;
        CodegenResult failed=compile_fixture();
        require(!failed.ok && !failed.module,"I publish no partial contract after failed growth");
        require(strstr(failed.error_msg,"available contract storage")!=NULL,
                "I preserve the first contract-storage diagnostic");
        require(allocation_attempts==failure,"I stop contract growth at the selected failure");

        allocation_attempts=0;fail_at=0;
        CodegenResult recovered=compile_fixture();
        require(recovered.ok && recovered.module,"I recover in the same process after failed growth");
        require(allocation_attempts==sites,"I rebuild the complete contract after recovery");
        nvm_module_free(recovered.module);
    }
    printf("I preserved %zu contract allocation failures and same-process recoveries.\n",sites);
    return 0;
}
