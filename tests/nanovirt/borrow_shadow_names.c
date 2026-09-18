/* I expose the C producer's selected-shadow module for paired advisory tests. */
#include "nanolang.h"
#include "nanovirt/codegen.h"
#include "nanoisa/disassembler.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int g_argc;
char **g_argv;
int main(int argc,char **argv) {
    assert(argc==2);
    FILE *file=fopen(argv[1],"rb");assert(file);
    assert(!fseek(file,0,SEEK_END));long size=ftell(file);assert(size>=0);
    rewind(file);char *source=calloc((size_t)size+1,1);assert(source);
    assert(fread(source,1,(size_t)size,file)==(size_t)size);fclose(file);
    int count=0;Token *tokens=tokenize(source,&count);assert(tokens);
    ASTNode *program=parse_program(tokens,count);assert(program);
    Environment *env=create_environment();assert(env);
    env->suppress_shadow_warnings=true;
    assert(type_check(program,env));
    CodegenResult result=codegen_compile_shadows(program,env,NULL,argv[1]);
    if(!result.ok)fprintf(stderr,"%s\n",result.error_msg);
    assert(result.ok);
    char *text=disasm_module_styled(result.module,DISASM_STYLE_CANONICAL);assert(text);
    fputs(text,stdout);free(text);nvm_module_free(result.module);
    free_ast(program);free_environment(env);free_tokens(tokens,count);free(source);
    return 0;
}
