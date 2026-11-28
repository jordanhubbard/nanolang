/* I inspect actual completed/recursive/guard paths, never a fresh body walk. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

static void result_snapshot(NvmAffineAnalysis result) {
    printf("%u,%u,%u,%u,%s",result.ok,result.byte_offset,result.reachable,result.visits,result.message);
}
static void cache_paths(Reader *r) {
    uint32_t present=word(r),functions_present=word(r),function=word(r),caller_present=word(r),globals_valid=word(r),graph=word(r),effect=word(r);
    uint32_t status=word(r),ok=word(r),offset=word(r),reachable=word(r),visits=word(r),allowed=word(r),returns=word(r),keys=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || present>1 || functions_present>1 || caller_present>1 || globals_valid>1 || graph>1 || effect>1 ||
       status<1 || status>2 || ok>1 || allowed>3 || returns>15 || keys>256 ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count || model.function_count>8 || r->pos!=r->size){r->ok=false;free_model(&model);return;}
    AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
    if(nvm_ownership_globals(&model,calls->globals,256,&calls->global_count)!=NVM_V2_OK){r->ok=false;free(calls);free_model(&model);return;}
    calls->globals_valid=globals_valid!=0;calls->value_graph=graph!=0;calls->effect_globals=effect!=0;
    bool guarded=!present || !functions_present || function>=model.function_count || !globals_valid || (calls->global_count && !graph);
    if(!guarded && (caller_present || !graph)){r->ok=false;free(calls);free_model(&model);return;}
    for(uint32_t f=0;f<8;f++) {
        calls->status[f]=(uint8_t)status;calls->results[f]=(NvmAffineAnalysis){.ok=ok!=0,.byte_offset=offset,.reachable=reachable,.visits=visits};
        snprintf(calls->results[f].message,sizeof(calls->results[f].message),"cached-%u",f);
        for(uint32_t g=0;g<calls->global_count;g++){calls->global_allowed[f][g]=(uint8_t)((allowed+f+g)%4);calls->global_returns[f][g]=(uint8_t)((returns+f+g)%16);}
        calls->escape_count[f]=(uint16_t)keys;for(uint32_t k=0;k<keys;k++)calls->escapes[f][k]=f*100+k;
    }
    NvmAffineState *caller=caller_present?nvm_affine_state_create(&model,0,model.functions[0].local_count):NULL;
    if(caller_present && !caller){r->ok=false;free(calls);free_model(&model);return;}
    NvmFunctionEntry *functions=model.functions;if(!functions_present)model.functions=NULL;
    NvmAffineAnalysis result=analyze(present?&model:NULL,function,caller,0,calls);
    model.functions=functions;nvm_affine_state_free(caller);
    bool cached=!strncmp(result.message,"cached-",7);
    printf("%u|",cached);result_snapshot(result);printf("|%u,%u,%u,%u|",calls->globals_valid,calls->value_graph,calls->effect_globals,calls->global_count);
    for(uint32_t g=0;g<calls->global_count;g++){number(calls->globals[g].tag);number(calls->globals[g].mutable);number(calls->globals[g].layout);putchar(';');}
    putchar('|');
    for(uint32_t f=0;f<8;f++) {
        printf("%u~",calls->status[f]);result_snapshot(calls->results[f]);putchar('~');
        for(uint32_t g=0;g<calls->global_count;g++){number(calls->global_allowed[f][g]);}putchar('/');
        for(uint32_t g=0;g<calls->global_count;g++){number(calls->global_returns[f][g]);}putchar('/');
        for(uint32_t k=0;k<calls->escape_count[f];k++){number(calls->escapes[f][k]);}putchar('#');
    }
    free(calls);free_model(&model);
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);
        else if(kind==15)cache_paths(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
