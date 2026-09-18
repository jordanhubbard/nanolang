#define CONSUMING_ALLOC_TEST
#include "test_consuming_calls.c"
#include <stdarg.h>

static void append(char *s,size_t capacity,const char *format,...) {
    size_t used=strlen(s);va_list args;va_start(args,format);
    int written=vsnprintf(s+used,capacity-used,format,args);va_end(args);
    CHECK(written>=0 && (size_t)written<capacity-used);
}
static uint8_t multiple_tag(unsigned index,unsigned p) {
    const uint8_t tags[8]={TAG_STRUCT,TAG_INT,TAG_STRUCT,TAG_BOOL,TAG_U8,TAG_STRUCT,TAG_INT,TAG_STRUCT};
    return index<2?TAG_STRUCT:tags[index==3?(p+1)%8:p];
}
static unsigned multiple_layout(unsigned index,unsigned p) {
    return index<2?((p+index)%2?2:0):p==2?1:p%2?2:0;
}
static int multiple_value(uint8_t tag,unsigned p) {
    return tag==TAG_BOOL?(int)(p&1):tag==TAG_U8?200+(int)p:10+(int)p;
}
static NvmModule *multiple_fixture(unsigned index,const char *call_override,const char *helper_override) {
    (void)consuming_fixture;(void)consuming_refusals;
    unsigned count=index<2?2:8;
    uint8_t result_tag=index==6?TAG_BOOL:index==7?TAG_U8:TAG_INT;
    const char *result_name=index==6?"bool":index==7?"u8":"int";
    char setup[8192]="",args[4096]="",helper[16384]="",params[256]="",source[49152];
    for(unsigned p=0;p<count;p++) {
        uint8_t tag=multiple_tag(index,p);unsigned layout=multiple_layout(index,p);
        int value=multiple_value(tag,p);
        const char *name=tag==TAG_STRUCT?"struct":tag==TAG_BOOL?"bool":tag==TAG_U8?"u8":"int";
        append(params,sizeof(params)," %s",name);
        if(tag==TAG_STRUCT) {
            if(layout==1) append(setup,sizeof(setup),"PUSH_I64 %d\nOWN_PACK 0\nPUSH_I64 99\nOWN_PACK 0\nOWN_PACK 1\n",value);
            else append(setup,sizeof(setup),"PUSH_I64 %d\nOWN_PACK %u\n",value,layout);
            append(setup,sizeof(setup),"OWN_STORE_LOCAL %u\n",p);
            append(args,sizeof(args),"OWN_MOVE_LOCAL %u\n",p);
            append(helper,sizeof(helper),"REGION_BEGIN\n%s 0 %u%s\nREF_GET 0 0\nPUSH_I64 %d\nEQ\nASSERT\nREGION_END\n",
                layout==1?"BORROW_PATH_SHARED":"BORROW_LOCAL_SHARED",p,layout==1?" 0":"",value);
        } else {
            const char *push=tag==TAG_BOOL?"PUSH_BOOL":tag==TAG_U8?"PUSH_U8":"PUSH_I64";
            append(setup,sizeof(setup),"%s %d\nSTORE_LOCAL %u\n",push,value,p);
            if(index==3 && p==0) append(args,sizeof(args),
                "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 14\nREF_GET 0 0\nPUSH_I64 1\nADD\nREGION_END\n");
            else append(args,sizeof(args),"LOAD_LOCAL %u\n",p);
            append(helper,sizeof(helper),"LOAD_LOCAL %u\n%s %d\nEQ\nASSERT\n",p,push,value);
        }
    }
    append(helper,sizeof(helper),
        "PUSH_I64 8\nOWN_PACK 0\nOWN_STORE_LOCAL 14\n"
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 14\nREF_GET 0 0\nPUSH_I64 8\nEQ\nASSERT\nREGION_END\n%s",
        index==4?"PUSH_BOOL 0\nASSERT\n":"");
    for(unsigned p=0;p<count;p++) if(multiple_tag(index,p)==TAG_STRUCT) {
        append(helper,sizeof(helper),"OWN_UNPACK_LOCAL %u\n",p);
        if(multiple_layout(index,p)==1) append(helper,sizeof(helper),
            "OWN_STORE_LOCAL 13\nOWN_STORE_LOCAL 12\nOWN_UNPACK_LOCAL 12\nPOP\nOWN_UNPACK_LOCAL 13\nPOP\n");
        else append(helper,sizeof(helper),"POP\n");
    }
    append(helper,sizeof(helper),"OWN_UNPACK_LOCAL 14\nPOP\n%s\nRET\n",
        index==6?"PUSH_BOOL 1":index==7?"PUSH_U8 207":"PUSH_I64 42");
    snprintf(source,sizeof(source),
        ".types 3 0 0\n.entry 0\n.function main 0 16 0 %s 1\n"
        "PUSH_I64 9\nOWN_PACK 0\nOWN_STORE_LOCAL 14\n"
        "%s%sCALL 1\nSTORE_LOCAL 15\n%s%sCALL 1\nSTORE_LOCAL 15\n"
        "REGION_BEGIN\nBORROW_LOCAL_SHARED 0 14\nREF_GET 0 0\nPUSH_I64 9\nEQ\nASSERT\nREGION_END\n%s"
        "OWN_UNPACK_LOCAL 14\nPOP\nLOAD_LOCAL 15\nRET\n.end\n"
        ".function inspect %u 16 0 %s 1\n%s.end\n.parameters 1%s\n",
        result_name,setup,call_override?call_override:args,setup,call_override?call_override:args,
        index==5?"PUSH_BOOL 0\nASSERT\n":"",count,result_name,helper_override?helper_override:helper,params);
    NvmModule *m=fixture();AsmResult result;NvmModule *code=asm_assemble_unverified(source,&result);
    if(!code)fprintf(stderr,"%s\n",result.message);
    CHECK(code);
    free(m->code);m->code=code->code;code->code=NULL;m->code_size=code->code_size;
    memcpy(m->functions,code->functions,2*sizeof(*m->functions));
    for(unsigned fn=0;fn<2;fn++) {
        free(m->function_param_types[fn]);
        m->function_param_types[fn]=code->function_param_types[fn];
        code->function_param_types[fn]=NULL;
    }
    nvm_module_free(code);
    free(m->ownership_data);m->ownership_size=308;m->ownership_data=calloc(308,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,3);data[8]=data[9]=data[10]=3;word(data+12,2);
    for(unsigned fn=0;fn<2;fn++) {
        unsigned header=16+fn*140,base=header+12;data[header]=16;data[header+2]=fn?count:0;
        slot(data+header+4,result_tag,0,NVM_V2_NO_INDEX);
        for(unsigned p=0;p<16;p++) {
            uint8_t tag=p<count?multiple_tag(index,p):p>=12&&p<=14?TAG_STRUCT:p==15?result_tag:TAG_INT;
            unsigned layout=p<count?multiple_layout(index,p):0;
            slot(data+base+8*p,tag,0,tag==TAG_STRUCT?layout:NVM_V2_NO_INDEX);
        }
    }
    word(data+296,1);data[300]=1;
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);return m;
}
static void multiple_refusals(void) {
    const char *args[]={"OWN_MOVE_LOCAL 0\nOWN_MOVE_LOCAL 0\n", /* repeated move */
        "OWN_MOVE_LOCAL 1\nOWN_MOVE_LOCAL 0\n", /* exact nominal order */
        "OWN_MOVE_LOCAL 0\nLOAD_LOCAL 1\n", /* resource observation */
        "OWN_MOVE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 1\nOWN_MOVE_LOCAL 1\n"};
    for(unsigned i=0;i<4;i++) {NvmModule *m=multiple_fixture(0,args[i],NULL);
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);}
    NvmModule *m=multiple_fixture(2,NULL,"PUSH_I64 42\nRET\n");
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=multiple_fixture(2,NULL,NULL);slot(m->ownership_data+168+7*8,TAG_STRUCT,0,0);
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m); /* final nominal mismatch */
    m=multiple_fixture(2,NULL,NULL);slot(m->ownership_data+168+6*8,TAG_BOOL,0,NVM_V2_NO_INDEX);
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=multiple_fixture(0,NULL,NULL);m->ownership_data[169]=1;
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=multiple_fixture(0,NULL,"OWN_MOVE_LOCAL 0\nOWN_MOVE_LOCAL 1\nCALL 1\nRET\n");
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    for(unsigned arity=0;arity<=9;arity+=9){m=multiple_fixture(2,NULL,NULL);
        m->functions[1].arity=arity;m->ownership_data[158]=arity;
        uint8_t tags[9];for(unsigned p=0;p<arity;p++) tags[p]=m->ownership_data[168+p*8];
        CHECK(nvm_set_function_param_types(m,1,tags,arity));
        bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);
        NvmAffineState *bounded=nvm_affine_state_create(m,1,16);CHECK(bounded);
        NvmAffineType parameters[8];uint16_t count=99;
        CHECK(!nvm_affine_consuming_parameters(bounded,parameters,8,&count));CHECK(count==99);
        nvm_affine_state_free(bounded);
        CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);}
    m=multiple_fixture(0,NULL,NULL);
    for(unsigned p=0;p<2;p++) {slot(m->ownership_data+168+p*8,TAG_INT,0,NVM_V2_NO_INDEX);m->function_param_types[1][p]=TAG_INT;}
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m); /* scalar-only CALL stays separate */
    m=multiple_fixture(2,NULL,NULL);slot(m->ownership_data+168+6*8,TAG_FLOAT,0,NVM_V2_NO_INDEX);m->function_param_types[1][6]=TAG_FLOAT;
    CHECK(!nvm_verify_owned_module(m).ok);nvm_module_free(m);
    m=multiple_fixture(0,NULL,NULL);NvmAffineState *s=nvm_affine_state_create(m,1,16);CHECK(s);
    NvmAffineType types[8];memset(types,0xa5,sizeof(types));NvmAffineType before[8];memcpy(before,types,sizeof(types));uint16_t count=99;
    CHECK(!nvm_affine_consuming_parameters(s,types,1,&count));CHECK(count==99&&!memcmp(types,before,sizeof(types)));
    CHECK(!nvm_affine_owned_parameter_type(s,&types[0]));
    CHECK(nvm_affine_consuming_parameters(s,types,8,&count)&&count==2);
    CHECK(types[0].layout==0&&types[1].layout==2);nvm_affine_state_free(s);nvm_module_free(m);
}
#ifndef MULTIPLE_CONSUMING_ALLOC_TEST
#ifndef MULTIPLE_CONSUMING_REPEATS
#define MULTIPLE_CONSUMING_REPEATS 16u
#endif
#if MULTIPLE_CONSUMING_REPEATS < 2
#error I require repeated invocation coverage.
#endif
int main(int argc,char **argv) {
    CHECK(argc==2);multiple_refusals();
    for(unsigned index=0;index<8;index++) {
        bool failure=index==4||index==5;uint8_t tag=index==6?TAG_BOOL:index==7?TAG_U8:TAG_INT;
        int expected=index==6?1:index==7?207:42;NvmModule *m=multiple_fixture(index,NULL,NULL);
        consuming_verified(m);artifacts(m,argv[1],index);VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<MULTIPLE_CONSUMING_REPEATS;repeat++) {
            NanoValue result=val_void();uint64_t generation=vm.reference_generation;
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):
                api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            CHECK(status==(failure?VM_ERR_ASSERT_FAILED:VM_OK));CHECK(vm.reference_generation>generation);
            if(!failure){if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}
                CHECK(result.tag==tag);CHECK((tag==TAG_BOOL?result.as.boolean:tag==TAG_U8?result.as.u8:result.as.i64)==expected);vm_release(&vm.heap,result);}
            CHECK(vm.stack_size==0&&vm.frame_count==0);CHECK(!vm.references.active&&!vm.callee_references.active);CHECK(vm.heap.stats.num_objects==baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u %d\n",index,failure?2:0,expected);
    }
    printf("%u multiple consuming-call checks passed\n",checks);return 0;
}
#endif
