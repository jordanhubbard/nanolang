/* I reuse qualified graph accounting while retaining its original main. */
#define NANO_OWNER_ARRAY_RUNTIME_MAIN retained_mutation_runtime_main
#include "test_private_owned_array_runtime.c"
static VmState *observed;
static uint32_t expected_pc;
static unsigned fault_kind,site_hits;
static bool site(void) {return observed && observed->current_fn==0 && observed->ip==expected_pc;}
void *mutation_vm_realloc(void *pointer,size_t bytes) {
    if(fault_kind==1 && observed && pointer==observed->stack) {
        CHECK(site() && observed->frame_count==1 && observed->stack_size==6);
        CHECK(observed->stack[4].tag==TAG_ARRAY && observed->stack[5].tag==TAG_STRUCT);
        CHECK(observed->stack_capacity==8 && bytes>8*sizeof(NanoValue));
        fprintf(stderr,"mutation call realloc pc=%u bytes=%zu frames=%u depth=%u\n",expected_pc,bytes,observed->frame_count,observed->stack_size);
        site_hits++;fault_kind=0;return NULL;
    }
    return realloc(pointer,bytes);
}
void *mutation_heap_malloc(size_t bytes) {
    if(fault_kind==2 && site()) {
        CHECK(bytes==sizeof(VmStruct) && observed->stack_size==7);
        CHECK(observed->stack[4].tag==TAG_STRUCT && observed->stack[5].tag==TAG_ARRAY && observed->stack[6].tag==TAG_STRING);
        fprintf(stderr,"mutation pack shell pc=%u bytes=%zu\n",expected_pc,bytes);
        site_hits++;fault_kind=0;return NULL;
    }
    return malloc(bytes);
}
void *mutation_heap_calloc(size_t count,size_t bytes) {
    if(fault_kind==3 && site()) {
        CHECK(count==3 && bytes==sizeof(NanoValue) && observed->stack_size==7);
        fprintf(stderr,"mutation pack fields pc=%u count=%zu width=%zu\n",expected_pc,count,bytes);
        site_hits++;fault_kind=0;return NULL;
    }
    return calloc(count,bytes);
}
static NvmModule *mutation_module(unsigned which) {
    Function call[]={
      {"ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 1\n"
       "LOAD_LOCAL 0\nPUSH_F64 2.5\nARR_PUSH\nDUP\nARR_LEN\nPRINTLN\n"
       "OWN_MOVE_LOCAL 1\nCALL 1\nPUSH_F64 3.5\nARR_SET\nPOP\n"
       "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 3.5\nF64_EQ\nASSERT\n"
       "LOAD_LOCAL 0\nARR_LEN\nPRINTLN\nPUSH_I64 0\nRET\n",0,4,T(TAG_INT),{T(TAG_ARRAY),OWNER(0),T(TAG_INT),T(TAG_INT)}},
      {"OWN_UNPACK_LOCAL 0\nPRINTLN\nPUSH_I64 0\nRET\n",1,8,T(TAG_INT),{OWNER(0),T(TAG_INT),T(TAG_INT),T(TAG_INT),T(TAG_INT),T(TAG_INT),T(TAG_INT),T(TAG_INT)}}
    };
    Function pack={
      "ARR_NEW 3\nSTORE_LOCAL 0\nPUSH_I64 7\nOWN_PACK 0\nOWN_STORE_LOCAL 1\nPUSH_STR value\nSTORE_LOCAL 3\n"
      "OWN_MOVE_LOCAL 1\nLOAD_LOCAL 0\nPUSH_F64 2.5\nARR_PUSH\nDUP\nARR_LEN\nPRINTLN\nLOAD_LOCAL 3\nOWN_PACK 1\nOWN_STORE_LOCAL 2\n"
      "OWN_UNPACK_LOCAL 2\nSTORE_LOCAL 3\nSTORE_LOCAL 0\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nPRINTLN\n"
      "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\nLOAD_LOCAL 0\nARR_LEN\nPRINTLN\nPUSH_I64 0\nRET\n",
      0,4,T(TAG_INT),{T(TAG_ARRAY),OWNER(0),OWNER(1),T(TAG_STRING)}};
    NvmModule *m=which?build(&pack,1):build(call,2);NvmOwnedArrayPlan *plan=NULL;
    NvmOwnerAuthorityResult r=nvm_owned_array_admit(m,&plan);
    if(r.status!=NVM_OWNER_AUTH_PREPARED)fprintf(stderr,"mutation admission: %s\n",r.message);
    CHECK(r.status==NVM_OWNER_AUTH_PREPARED && nvm_verify(m).ok);nvm_owned_array_plan_free(plan);
    expected_pc=UINT32_MAX;
    for(uint32_t pc=0;pc<m->functions[0].code_length;) {
        DecodedInstruction in;uint32_t width=isa_decode(m->code+pc,m->functions[0].code_length-pc,&in);CHECK(width);
        if((!which && in.opcode==OP_CALL) || (which && in.opcode==OP_OWN_PACK && in.operands[0].u32==1))expected_pc=pc+width;
        pc+=width;
    }
    CHECK(expected_pc!=UINT32_MAX);return m;
}
static void mutation_run(NvmModule *m,unsigned fault,bool small,unsigned fused) {
    VmState vm;vm_init(&vm,m);CHECK(vm.last_error==VM_OK);
    VmDispatchProfile profile={.fuse_load_local_field=fused};vm_set_dispatch_profile(&vm,profile);CHECK(vm.dispatch_module_valid);
    if(small){NanoValue *stack=calloc(8,sizeof *stack);CHECK(stack);free(vm.stack);vm.stack=stack;vm.stack_capacity=8;}
    size_t roots=vm.heap.stats.num_objects,bytes=vm.heap.stats.allocated-vm.heap.stats.freed;
    FILE *output=tmpfile();CHECK(output);vm.output=output;observed=&vm;fault_kind=fault;site_hits=0;
    NanoValue result=val_int(-91);VmResult status=runtime_entry(&vm,&result);observed=NULL;fault_kind=0;
    fprintf(stderr,"mutation api=%u fault=%u hits=%u status=%d\n",public_api,fault,site_hits,status);
    CHECK(fault?(site_hits==1 && status==VM_ERR_MEMORY && runtime_failure_value(result)):(status==VM_OK && result.tag==TAG_INT && result.as.i64==0));
    CHECK(!fflush(output));long length=ftell(output);rewind(output);char text[32]={0};CHECK(length>=0 && length<(long)sizeof text);CHECK(fread(text,1,(size_t)length,output)==(size_t)length);
    CHECK(!strcmp(text,fault?"1\n":"1\n7\n1\n"));CHECK(!fclose(output));vm.output=NULL;
    clean(&vm,roots,bytes);vm_destroy(&vm);CHECK(!vm.heap.stats.num_objects);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    for(unsigned which=0;which<2;which++) {
        NvmModule *m=mutation_module(which);char error[256];char *source=nvm2c_emit(m,error,sizeof error);CHECK(source);
        char path[1024];snprintf(path,sizeof path,"%s/mutation%u.c",argv[1],which);FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(source,1,strlen(source),f)==strlen(source));CHECK(!fclose(f));free(source);
        for(public_api=0;public_api<4;public_api++)for(unsigned fused=0;fused<2;fused++) {
            mutation_run(m,0,!which,fused);
            mutation_run(m,which?2:1,!which,fused);mutation_run(m,0,!which,fused);
            if(which){mutation_run(m,3,false,fused);mutation_run(m,0,false,fused);}
        }
        nvm_module_free(m);
    }
    printf("%u prepared mutation checks passed\n",checks);return 0;
}
