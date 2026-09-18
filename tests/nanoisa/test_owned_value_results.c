/* I transfer exact returned owners while preserving unrelated caller holds. */
#define OWNED_GRAPH_ALLOC_TEST
#include "test_owned_value_graphs.c"
static NvmModule *result_fixture(unsigned index,unsigned refusal) {
    (void)graph_fixture;(void)graph_refusals;(void)ordinary_chain;
    unsigned functions=index==5?8:4,layout=index>=1&&index<=3?2:0;
    uint8_t leaf=index==2?TAG_BOOL:index==3?TAG_U8:TAG_INT;
    const char *push=leaf==TAG_BOOL?"PUSH_BOOL 1":leaf==TAG_U8?"PUSH_U8 200":"PUSH_I64 42";
    char source[24000]=".types 3 0 0\n.entry 0\n",factory[4096]="";
    append(source,sizeof(source),".function main 0 8 0 int 1\nPUSH_I64 99\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 0\n");
    if(index==5)append(source,sizeof(source),"PUSH_I64 42\nOWN_PACK 0\nCALL 1\nOWN_STORE_LOCAL 1\n");
    else {
        append(source,sizeof(source),"CALL 1\n%s\nPUSH_I64 2\nOWN_MOVE_LOCAL 1\nPUSH_BOOL 1\nCALL 2\nOWN_STORE_LOCAL 1\nPUSH_I64 2\nOWN_MOVE_LOCAL 1\nPUSH_BOOL 0\nCALL 2\nOWN_STORE_LOCAL 1\n",refusal==2?"POP":"OWN_STORE_LOCAL 1");
    }
    if(index==8)append(source,sizeof(source),"PUSH_BOOL 0\nASSERT\n");
    if(index!=5)append(source,sizeof(source),"CALL 1\nCALL 3\n");
    append(source,sizeof(source),"REF_GET 0 0\nPUSH_I64 99\nEQ\nASSERT\n");
    if(index==5)append(source,sizeof(source),"OWN_UNPACK_LOCAL 1\nPUSH_I64 42\nEQ\nASSERT\n");
    else append(source,sizeof(source),"OWN_MOVE_LOCAL 1\nCALL 3\n");
    append(source,sizeof(source),"REF_GET 0 0\nPUSH_I64 99\nEQ\nASSERT\nREGION_END\nOWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n.end\n");
    if(index==5) {
        for(unsigned f=1;f<8;f++) {
            append(source,sizeof(source),".function forward%u 1 8 0 struct 1\nOWN_MOVE_LOCAL 0\n",f);
            if(f<7)append(source,sizeof(source),"CALL %u\n",f+1);
            append(source,sizeof(source),"RET\n.end\n.parameters %u struct\n",f);
        }
    } else {
        if(refusal==10)append(factory,sizeof(factory),"PUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 4\n");
        /* Nested results can now be declared, but this case still assigns the
         * different returned layout to an exact scalar-leaf caller local. */
        if(refusal==6)append(factory,sizeof(factory),"PUSH_I64 42\nOWN_PACK 0\nPUSH_I64 43\nOWN_PACK 0\nOWN_PACK 1\nRET\n");
        else {
            append(factory,sizeof(factory),"PUSH_BOOL %u\nJMP_FALSE alternate\n%s\nOWN_PACK %u\n",index==4?0:1,push,refusal==1?2:layout);
            if(index==7)append(factory,sizeof(factory),"PUSH_BOOL 0\nASSERT\n");
            if(refusal==3)append(factory,sizeof(factory),"PUSH_I64 9\n");
            if(refusal==5)append(factory,sizeof(factory),"OWN_STORE_LOCAL 4\n");
            append(factory,sizeof(factory),"RET\nalternate:\n%s\nOWN_PACK %u\nRET\n",push,layout);
        }
        append(source,sizeof(source),".function factory 0 8 0 struct 1\n%s.end\n",factory);
        append(source,sizeof(source),".function forward 3 8 0 struct 1\nLOAD_LOCAL 0\nPUSH_I64 2\nEQ\nASSERT\n");
        if(refusal==4)append(source,sizeof(source),"LOAD_LOCAL 1\nRET\n");
        else if(refusal==9)append(source,sizeof(source),"REGION_BEGIN\nBORROW_LOCAL_SHARED 0 1\nLOAD_LOCAL 1\nRET\n");
        else {
            append(source,sizeof(source),"LOAD_LOCAL 2\nJMP_FALSE replace\nOWN_MOVE_LOCAL 1\nRET\nreplace:\nOWN_UNPACK_LOCAL 1\nPOP\nCALL 1\n");
            if(index==9)append(source,sizeof(source),"PUSH_BOOL 0\nASSERT\n");
            append(source,sizeof(source),"RET\n");
        }
        append(source,sizeof(source),".end\n.parameters 2 int struct bool\n.function consume 1 8 0 void 0\nOWN_UNPACK_LOCAL 0\n%s\nEQ\nASSERT\n%sRET\n.end\n.parameters 3 struct\n",push,index==6?"PUSH_BOOL 0\nASSERT\n":"");
    }
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmModule *base=fixture();m->layout_data=base->layout_data;base->layout_data=NULL;m->layout_size=base->layout_size;nvm_module_free(base);
    if(leaf!=TAG_INT) {
        NvmV2Layouts rows={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&rows)==NVM_V2_OK);
        rows.items[2].fields[0].type_tag=leaf;CHECK(nvm_retain_layouts(m,&rows)==NVM_V2_OK);nvm_v2_layouts_free(&rows);
    }
    m->ownership_size=16+functions*76+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *data=m->ownership_data;word(data,2);word(data+4,3);data[8]=data[9]=data[10]=3;word(data+12,functions);
    for(unsigned f=0;f<functions;f++) {
        unsigned h=16+f*76;data[h]=8;data[h+2]=(uint8_t)m->functions[f].arity;
        uint8_t result=f?index==5||f<3?TAG_STRUCT:TAG_VOID:TAG_INT;
        slot(data+h+4,result,0,result==TAG_STRUCT?(refusal==6&&f==1?1:layout):NVM_V2_NO_INDEX);
        for(unsigned local=0;local<8;local++) {
            uint8_t tag=TAG_INT;uint32_t nominal=NVM_V2_NO_INDEX;
            if(!f&&(local==0||local==1)){tag=TAG_STRUCT;nominal=local?layout:0;}
            if(f&&((index==5&&local==0)||(index!=5&&f==2&&local==1)||(index!=5&&f==3&&local==0)||local==4)){tag=TAG_STRUCT;nominal=layout;}
            if(index!=5&&f==2&&local==2)tag=TAG_BOOL;
            slot(data+h+12+8*local,tag,0,nominal);
        }
    }
    if(refusal==7||refusal==8){m->functions[0].result_tag=refusal==7?TAG_STRUCT:TAG_VOID;m->functions[0].result_count=refusal==7?1:0;slot(data+20,m->functions[0].result_tag,0,refusal==7?0:NVM_V2_NO_INDEX);}
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);return m;
}
static VmResult result_api(VmState *vm,unsigned api,NanoValue *result) {
    return api==0?vm_invoke(vm,0,NULL,0,result):api==1?vm_execute(vm):api==2?vm_call_function(vm,0,NULL,0):vm_invoke_callable(vm,val_function(0),NULL,0,result);
}
static void result_clean(VmState *vm,size_t baseline) {
    CHECK(!vm->stack_size&&!vm->frame_count);CHECK(!vm->references.active&&!vm->callee_references.active);
    for(unsigned f=0;f<NVM_OWNED_MAX_FUNCTIONS-2;f++)CHECK(!vm->value_references[f].active);
    CHECK(vm->heap.stats.num_objects==baseline);
}
#ifndef OWNED_RESULT_ALLOC_TEST
int main(int argc,char **argv) {
    CHECK(argc==2||argc==3);
    for(unsigned refusal=1;refusal<=10;refusal++) {
        NvmModule *m=result_fixture(0,refusal);CHECK(!nvm_verify_owned_module(m).ok);CHECK(!nvm_verify(m).ok);
        char error[256];CHECK(!nvm2c_emit(m,error,sizeof(error)));nvm_module_free(m);
    }
    unsigned begin=0,end=10;
    if(argc==3){char *stop=NULL;unsigned long n=strtoul(argv[2],&stop,10);CHECK(argv[2][0]&&!*stop&&n<10);begin=(unsigned)n;end=begin+1;}
    for(unsigned index=begin;index<end;index++) {
        NvmModule *m=result_fixture(index,0);consuming_verified(m);artifacts(m,argv[1],index);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            NanoValue result=val_int(-91);uint64_t generation=vm.reference_generation;
            VmResult status=result_api(&vm,api,&result);CHECK(status==(index>=6?VM_ERR_ASSERT_FAILED:VM_OK));CHECK(vm.reference_generation>generation);
            if(index<6){if(api==1||api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];}CHECK(result.tag==TAG_INT&&result.as.i64==42);vm_release(&vm.heap,result);}
            result_clean(&vm,baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u 42\n",index,index>=6?2:0);
    }
    printf("%u owned value result checks passed\n",checks);return 0;
}
#endif
