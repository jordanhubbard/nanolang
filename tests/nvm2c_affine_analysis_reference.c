/* I compare the actual unchanged private C body analyzer and its complete cache/facts. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

static void result_snapshot(NvmAffineAnalysis result) {
    printf("%u,%u,%u,%u,%s",result.ok,result.byte_offset,result.reachable,result.visits,result.message);
}
static void cache_snapshot(const AnalysisCalls *calls) {
    printf("%u,%u,%u,%u|",calls->globals_valid,calls->value_graph,calls->effect_globals,calls->global_count);
    for(uint32_t g=0;g<calls->global_count;g++) {number(calls->globals[g].tag);number(calls->globals[g].mutable);number(calls->globals[g].layout);putchar(';');}
    putchar('|');
    for(uint32_t f=0;f<8;f++) {
        printf("%u~",calls->status[f]);result_snapshot(calls->results[f]);putchar('~');
        for(uint32_t g=0;g<calls->global_count;g++){number(calls->global_allowed[f][g]);}putchar('/');
        for(uint32_t g=0;g<calls->global_count;g++){number(calls->global_returns[f][g]);}putchar('/');
        for(uint32_t k=0;k<calls->escape_count[f];k++){number(calls->escapes[f][k]);}putchar('#');
    }
}
static void analysis(Reader *r) {
    uint32_t function=word(r),facts_present=word(r),capacity=word(r),seeds=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || facts_present>1 || seeds>4096 || !take(&r->pos,size,r->size,&at) ||
       !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(r->pos!=r->size){r->ok=false;free_model(&model);return;}
    AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
    analysis_calls_init(calls,&model);
    NvmAffineInstructionFact *facts=calloc(4096,sizeof(*facts));if(!facts)exit(2);
    for(uint32_t i=0;i<seeds;i++)facts[i]=(NvmAffineInstructionFact){.byte_offset=1234567,.reachable=true,.stack_depth=7,.top_tag=99,.unpack_count=11};
    calls->facts=facts_present?facts:NULL;calls->fact_function=function;calls->fact_capacity=capacity;
    NvmAffineAnalysis result=analyze(&model,function,NULL,0,calls);
    result_snapshot(result);putchar('|');cache_snapshot(calls);putchar('|');
    uint32_t count=seeds;
    if(result.ok && facts_present) {
        VmDecodedFunction decoded={0};char error[VM_DECODE_ERROR_SIZE];
        if(!vm_decode_function(&model,function,&decoded,error))exit(2);
        count=decoded.instruction_count;vm_decoded_function_free(&decoded);
    }
    number(count);
    for(uint32_t i=0;i<count;i++) {
        number(facts[i].byte_offset);number(facts[i].reachable);number(facts[i].stack_depth);number(facts[i].top_tag);number(facts[i].unpack_count);putchar(';');
    }
    free(facts);free(calls);free_model(&model);
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);
        else if(kind==17)analysis(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
