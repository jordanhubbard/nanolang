/* I observe actual C preflight and compare entry queries with public C state APIs. */
#include <stdint.h>
uint32_t capture_preflight(uint32_t);
#ifndef NANOLANG_PREFLIGHT_OBSERVER_DISABLED
#define NVM_AFFINE_TEST_VISIT_LIMIT capture_preflight
#endif
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main
#undef NVM_AFFINE_TEST_VISIT_LIMIT

static bool captured;
static const NvmModule *observed_module;
static AnalysisCalls *observed_calls;
static const NvmAffineState *observed_caller;
static uint32_t observed_function,observed_reference;
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
uint32_t capture_preflight(uint32_t limit) {
    if(captured) {fputs("I refuse nested body calls in my preflight diagnostic\n",stderr);exit(2);}
    captured=true;AnalysisCalls *calls=observed_calls;uint32_t function=observed_function;
    /* I reconstruct this separate query view with actual public C APIs. I do not
     * claim access to analyze's private initial pointer through the limit hook. */
    Frame value={0};Frame *f=&value;VmDecodedFunction body={0};VmDecodedFunction *decoded=&body;
    f->locals=nvm_affine_state_create(observed_module,function,observed_module->functions[function].local_count);
    if(!f->locals || (observed_caller && !nvm_affine_bind_caller(f->locals,observed_caller,observed_reference)))exit(2);
    char detail[VM_DECODE_ERROR_SIZE];if(!vm_decode_function(observed_module,function,decoded,detail))exit(2);
    f->global_count=(uint16_t)calls->global_count;memset(f->globals,9,calls->global_count);
    fputs("1|0,0,0,0,|",stdout);cache_snapshot(calls);
    printf("|%u,%u,%u,%u,%u,%u,|",limit,decoded->instruction_count,f->count,f->handler_count,f->handler_floor,f->activation);
    for(uint32_t g=0;g<f->global_count;g++){number(f->globals[g]);}putchar('|');
    for(uint16_t i=0;i<observed_module->functions[function].local_count;i++) {
        uint8_t tag=0,mode=0,read_tag=0,write_tag=0;uint16_t variant=UINT16_MAX;
        NvmAffineType type={0},parameter={0};NvmReferenceMode parameter_mode=0;
        bool live=nvm_affine_local_info(f->locals,i,&tag,&mode);
        (void)nvm_affine_union_variant(f->locals,i,&variant);
        bool typed=nvm_affine_local_type(f->locals,i,&type);
        bool param=nvm_affine_parameter_at(f->locals,i,&parameter,&parameter_mode);
        bool read=nvm_affine_reference_field(f->locals,i,0,false,&read_tag);
        bool write=nvm_affine_reference_field(f->locals,i,0,true,&write_tag);
        number(live);number(tag);number(mode);number(variant);number(typed);number(type.tag);number(type.layout);
        number(param);number(parameter.tag);number(parameter.layout);number(parameter_mode);
        number(read);number(read_tag);number(write);number(write_tag);putchar(';');
    }
    nvm_affine_state_free(f->locals);vm_decoded_function_free(decoded);return limit;
}
static void preflight(Reader *r) {
    uint32_t present=word(r),functions_present=word(r),function=word(r),caller_present=word(r),reference=word(r);
    uint32_t globals_valid=word(r),graph=word(r),effect=word(r),facts_present=word(r),fact_function=word(r),fact_capacity=word(r),borrow_mode=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || present>1 || functions_present>1 || caller_present>1 || globals_valid>1 || graph>1 || effect>1 || facts_present>1 || borrow_mode>2 ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count || model.function_count>8 || r->pos!=r->size){r->ok=false;free_model(&model);return;}
    AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
    if(nvm_ownership_globals(&model,calls->globals,256,&calls->global_count)!=NVM_V2_OK)calls->global_count=0;
    calls->globals_valid=globals_valid!=0;calls->value_graph=graph!=0;calls->effect_globals=effect!=0;
    NvmAffineState *caller=caller_present?nvm_affine_state_create(&model,0,model.functions[0].local_count):NULL;
    if(caller_present && (!caller || (borrow_mode && (!nvm_affine_region_begin(caller) || !nvm_affine_borrow(caller,1,0,NULL,0,(NvmReferenceMode)borrow_mode))))) {
        r->ok=false;nvm_affine_state_free(caller);free(calls);free_model(&model);return;
    }
    /* I use a valid pointer even for a mismatching capacity; analyze checks it before writes. */
    NvmAffineInstructionFact *facts=facts_present?calloc(4096,sizeof(*facts)):NULL;
    if(facts_present && !facts)exit(2);
    calls->facts=facts;calls->fact_function=fact_function;calls->fact_capacity=fact_capacity;
    NvmFunctionEntry *functions=model.functions;if(!functions_present)model.functions=NULL;
    observed_module=&model;observed_calls=calls;observed_caller=caller;observed_function=function;observed_reference=reference;captured=false;
    NvmAffineAnalysis result=analyze(present?&model:NULL,function,caller,reference,calls);
    model.functions=functions;
    if(!captured) {fputs("0|",stdout);result_snapshot(result);putchar('|');cache_snapshot(calls);fputs("|-",stdout);}
    fprintf(stderr,"preflight=%u function=%u body_ok=%u offset=%u reachable=%u visits=%u message=%s\n",captured,function,result.ok,result.byte_offset,result.reachable,result.visits,result.message);
    free(facts);nvm_affine_state_free(caller);free(calls);free_model(&model);
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);
        else if(kind==16)preflight(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
