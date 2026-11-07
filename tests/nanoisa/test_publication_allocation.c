/* I call the production CLI and host helper, not replacement publisher logic. */
#include "publication_alloc.h"
#include "assembler.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int publisher_cli_main(int,char **);
static void require(bool ok,const char *why){if(!ok){fprintf(stderr,"I failed publication qualification: %s (calls=%zu denied=%zu live=%zu)\n",why,pa_calls,pa_denied,pa_live);exit(90);}}
static char *read_text(const char *path) {
    FILE *f=fopen(path,"rb");require(f!=NULL,"open input");
    require(fseek(f,0,SEEK_END)==0,"seek input");long n=ftell(f);require(n>=0,"size input");rewind(f);
    char *text=malloc((size_t)n+1);require(text!=NULL,"read allocation");
    require(fread(text,1,(size_t)n,f)==(size_t)n,"read input");text[n]=0;fclose(f);return text;
}
int main(int argc,char **argv){
    require(argc==6,"mode/source/assembly/output/failure arguments");
    bool query=!strcmp(argv[1],"query");
    bool save=!strcmp(argv[1],"save");
    const char *position=getenv("PUBLICATION_FAIL");pa_fail=position?(size_t)strtoull(position,NULL,10):SIZE_MAX;
    pa_persistent=getenv("PUBLICATION_PERSISTENT")!=NULL;
    char *source=read_text(argv[3]);
    if(save){
        FILE *f=fopen(argv[4],"wb");require(f!=NULL,"save sentinel open");fputs("prior-output",f);fclose(f);
        pa_active=true;int64_t answer=nl_nanoisa_assemble_text_save(source,argv[4]);pa_active=false;
        if(pa_denied){
            require(answer!=0,"save memory failure is hard error");
            char *output=read_text(argv[4]);require(!strcmp(output,"prior-output"),"prior assembler output preserved");free(output);
        }else require(answer==0,"save success");
    }else if(query){
        pa_active=true;int64_t answer=nl_nanoisa_array_publication(source);pa_active=false;
        if(pa_denied)require(answer==-1,"observed memory failure is hard error");
        else require(answer==1,"eligible query");
    }else{
        FILE *f=fopen(argv[4],"wb");require(f!=NULL,"sentinel open");fputs("prior-output",f);fclose(f);
        char *args[]={"nano_virt",argv[2],"--emit-nvm","-o",argv[4],NULL};
        int result=publisher_cli_main(5,args);
        if(pa_denied){
            require(result!=0,"publisher memory failure is hard error");
            char *output=read_text(argv[4]);require(!strcmp(output,"prior-output"),"prior CLI output preserved");free(output);
        }else require(result==0,"publisher success");
    }
    if(position || getenv("PUBLICATION_GROWTH_FAIL"))require(pa_denied>0,"requested failure reached");
    require(pa_live==0,"selected allocations cleaned");
    size_t count=pa_calls,denials=pa_denied;
    if(getenv("PUBLICATION_GROWTH_FAIL")){
        pa_calls=0;pa_denied=0;pa_active=true;AsmResult error={0};
        NvmModule *module=asm_assemble_unverified(source,&error);pa_active=false;
        require(module==NULL && error.error==ASM_ERR_MEMORY,"typed assembler memory error");
        require(pa_denied>0 && pa_live==0,"typed failure injection and cleanup");
    }
    pa_fail=SIZE_MAX;pa_calls=0;pa_denied=0;pa_persistent=false;
    unsetenv("PUBLICATION_GROWTH_FAIL");
    /* I run an independent valid invocation after each failure, in this process. */
    if(save){pa_active=true;require(nl_nanoisa_assemble_text_save(source,argv[4])==0,"save recovery");pa_active=false;}
    else if(query){pa_active=true;require(nl_nanoisa_array_publication(source)==1,"query recovery");pa_active=false;}
    else {char *args[]={"nano_virt",argv[2],"--emit-nvm","-o",argv[4],NULL};require(publisher_cli_main(5,args)==0,"publisher recovery");}
    require(pa_live==0,"recovery cleanup");free(source);
    printf("I qualified %s calls=%zu denials=%zu with independent recovery\n",argv[1],count,denials);
    return 0;
}
