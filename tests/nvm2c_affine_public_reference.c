/* I exercise the unchanged C public analyzers and private effect/global checker. */
#include <stdint.h>
static uint32_t requested_limit=262144u;
uint32_t public_effect_test_limit(uint32_t limit) { return requested_limit<limit?requested_limit:limit; }
#ifndef NANOLANG_EFFECT_OBSERVER_DISABLED
#define NVM_AFFINE_EFFECT_TEST_LIMIT public_effect_test_limit
#endif
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main
#ifdef NVM_AFFINE_EFFECT_TEST_LIMIT
#undef NVM_AFFINE_EFFECT_TEST_LIMIT
#endif

static void result_snapshot(NvmAffineAnalysis result) {
    printf("%u,%u,%u,%u,%s",result.ok,result.byte_offset,result.reachable,result.visits,result.message);
}
static void public_analysis(Reader *r) {
    uint32_t mode=word(r),function=word(r),present=word(r),capacity=word(r),seeds=word(r),limit=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || mode>2 || present>1 || seeds>4096 || !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(r->pos!=r->size){r->ok=false;free_model(&model);return;}
    NvmAffineInstructionFact *facts=calloc(4096,sizeof(*facts));if(!facts)exit(2);
    for(uint32_t i=0;i<seeds;i++)facts[i]=(NvmAffineInstructionFact){.byte_offset=1234567,.reachable=true,.stack_depth=7,.top_tag=99,.unpack_count=11};
    NvmAffineAnalysis result={0};requested_limit=mode==2?limit:262144u;
    if(mode==0)result=nvm_affine_analyze_function(&model,function);
    else if(mode==1)result=nvm_affine_analyze_instructions(&model,function,present?facts:NULL,capacity);
    else {
        AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);analysis_calls_init(calls,&model);
        if(!calls->globals_valid || !calls->value_graph){r->ok=false;free(calls);free(facts);free_model(&model);return;}
        result=effect_global_proof(&model,calls);free(calls);
    }
    result_snapshot(result);putchar('|');
    uint32_t count=result.ok && mode==1?capacity:seeds;number(count);
    for(uint32_t i=0;i<count;i++) {number(facts[i].byte_offset);number(facts[i].reachable);number(facts[i].stack_depth);number(facts[i].top_tag);number(facts[i].unpack_count);putchar(';');}
    free(facts);free_model(&model);
}
static void hash_control(Reader *r) {
    uint32_t a=word(r),b=word(r),c=word(r),d=word(r),e=word(r);
    if(!r->ok || r->pos!=r->size){r->ok=false;return;}
    number(eg_hash(a,b,c,d,e));
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);
        else if(kind==18)public_analysis(&r);else if(kind==19)hash_control(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
