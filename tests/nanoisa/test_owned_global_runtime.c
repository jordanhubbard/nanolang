#define main owned_record_test_main
#include "test_owned_runtime.c"
#undef main
#include "owned_global_fixture.h"

static NvmModule *global_union_module(bool wrong) {
    char body[512];snprintf(body,sizeof(body),
        "PUSH_I64 0\nSTORE_GLOBAL 1\nloop:\nPUSH_I64 42\nAGG_PACK 1 %u 0 1\nSTORE_GLOBAL 0\n"
        "LOAD_GLOBAL 0\nMATCH_TAG 0 observed\nHALT\nobserved:\nAGG_GET 0\nPOP\n"
        "LOAD_GLOBAL 1\nPUSH_I64 1\nADD\nSTORE_GLOBAL 1\nLOAD_GLOBAL 1\nPUSH_I64 20\nLT\nJMP_TRUE loop\n"
        "LOAD_GLOBAL 0\nMATCH_TAG 0 selected\nHALT\nselected:\nAGG_GET 0\nRET\n",wrong?1u:0u);
    NvmModule *m=global_fixture(body,NULL,NULL,TAG_INT,true);
    uint32_t first=nvm_add_string(m,"First",5),second=nvm_add_string(m,"Second",6);
    uint32_t variant=nvm_add_string(m,"Some",4),name=nvm_add_string(m,"value",5);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,name};
    NvmV2Layout items[]={{NVM_V2_LAYOUT_UNION,1,first,&field},{NVM_V2_LAYOUT_UNION,1,second,&field}};
    NvmV2Layouts layouts={items,2};m->struct_count=0;m->union_count=2;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    free(m->ownership_data);m->ownership_size=112;m->ownership_data=calloc(112,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;global_word(p,3);global_word(p+4,2);p[8]=p[9]=1;
    global_word(p+12,1);global_slot(p+20,TAG_INT,0);global_word(p+28,4);global_word(p+36,2);
    p[40]=1;p[42]=1;global_word(p+44,36);global_word(p+48,2);
    for(unsigned i=0;i<2;i++) {
        unsigned at=52+i*16;global_word(p+at,i);p[at+4]=1;global_word(p+at+8,variant);p[at+14]=1;
    }
    p[84]=3;p[86]=1;global_word(p+88,20);global_word(p+92,2);global_slot(p+96,TAG_UNION,1);global_slot(p+104,TAG_INT,1);
    return m;
}

static void global_public_routes(void) {
    const char *body="PUSH_STR 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nLOAD_GLOBAL 0\nPUSH_STR 0\nEQ\nASSERT\nPUSH_I64 42\nRET\n";
    const char *helper="LOAD_GLOBAL 0\nPRINTLN\nPUSH_I64 0\nRET\n";
    for(unsigned api=0;api<4;api++) {
        NvmModule *m=global_fixture(body,helper,NULL,TAG_STRING,true);
        CHECK(nvm_verify(m).ok);VmState vm;vm_init(&vm,m);
        FILE *output=tmpfile();CHECK(output);vm.output=output;size_t baseline=vm.heap.stats.num_objects;
        for(unsigned repeat=0;repeat<3;repeat++) {
            NanoValue result=val_void();
            VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):
                api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
            if(status!=VM_OK)fprintf(stderr,"global API %u: %s\n",api,vm.error_msg);
            CHECK(status==VM_OK);
            if(api==1 || api==2){CHECK(vm.stack_size==1);result=vm.stack[--vm.stack_size];vm.stack[vm.stack_size]=val_void();}
            CHECK(result.tag==TAG_INT && result.as.i64==42);
            CHECK(!vm.global_count && !vm.stack_size && !vm.frame_count && vm.heap.stats.num_objects==baseline);
        }
        NanoValue result=val_int(-99);
        CHECK(vm_invoke(&vm,1,NULL,0,&result)==VM_ERR_TYPE_ERROR);
        CHECK(vm_call_function(&vm,1,NULL,0)==VM_ERR_TYPE_ERROR);
        CHECK(vm_invoke_callable(&vm,val_function(1),NULL,0,&result)==VM_ERR_TYPE_ERROR);
        CHECK(!vm.global_count && vm.heap.stats.num_objects==baseline);
        vm_destroy(&vm);fclose(output);nvm_module_free(m);
    }
    /* Global roots must disappear after a trap on every entry API as well. */
    for(unsigned api=0;api<4;api++) {
        NvmModule *m=global_fixture("PUSH_STR 0\nSTORE_GLOBAL 0\nPUSH_BOOL 0\nASSERT\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_STRING,true);
        CHECK(nvm_verify(m).ok);VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        NanoValue result=val_void();
        VmResult status=api==0?vm_invoke(&vm,0,NULL,0,&result):api==1?vm_execute(&vm):
            api==2?vm_call_function(&vm,0,NULL,0):vm_invoke_callable(&vm,val_function(0),NULL,0,&result);
        CHECK(status==VM_ERR_ASSERT_FAILED && !vm.global_count && !vm.stack_size && !vm.frame_count);
        CHECK(vm.heap.stats.num_objects==baseline);vm_destroy(&vm);nvm_module_free(m);
    }
    NvmModule *m=global_fixture("PUSH_I64 42\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true);
    VmState vm;vm_init(&vm,m);
    vm.frame_count=1;vm.current_fn=0;vm.ip=m->functions[0].code_offset;
    vm.frames[0].module=m;vm.frames[0].fn_idx=0;
    VmTrap trap=vm_core_execute(&vm);CHECK(trap.type==TRAP_ERROR && trap.data.error.code==VM_ERR_TYPE_ERROR);
    CHECK(!vm.global_count);vm.frame_count=0;vm_destroy(&vm);nvm_module_free(m);
}

#ifndef OWNED_GLOBAL_NO_MAIN
int main(int argc,char **argv) {
    CHECK(argc==2);unsigned number=0;
    global_public_routes();
    execute_module(global_fixture("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nCALL 1\nRET\n",
        "LOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,TAG_INT,true),2,argv[1],number++,TAG_INT);
    execute_module(global_fixture("CALL 1\nPOP\nLOAD_GLOBAL 0\nRET\n",
        "CALL 2\nRET\n","PUSH_I64 42\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",TAG_INT,false),42,argv[1],number++,TAG_INT);
    execute_module(global_fixture("PUSH_BOOL 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nASSERT\nPUSH_I64 42\nRET\n",NULL,NULL,TAG_BOOL,true),42,argv[1],number++,TAG_INT);
    execute_module(global_fixture("PUSH_U8 42\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_U8 42\nEQ\nASSERT\nPUSH_I64 42\nRET\n",NULL,NULL,TAG_U8,true),42,argv[1],number++,TAG_INT);
    execute_module(global_fixture("PUSH_F64 2.0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_F64 2.0\nF64_EQ\nASSERT\nPUSH_I64 42\nRET\n",NULL,NULL,TAG_FLOAT,true),42,argv[1],number++,TAG_INT);
    execute_module(global_fixture("PUSH_STR 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nLOAD_GLOBAL 0\nPUSH_STR 0\nEQ\nASSERT\nPUSH_I64 42\nRET\n",
        "PUSH_STR 0\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,TAG_STRING,true),42,argv[1],number++,TAG_INT);
    execute_module(global_union_module(false),42,argv[1],number++,TAG_INT);
    execute_module(global_fixture("PUSH_I64 0\nSTORE_GLOBAL 0\nloop:\nLOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 42\nLT\nJMP_TRUE loop\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true),42,argv[1],number++,TAG_INT);
    refused(global_fixture("LOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,true),argv[1],0);
    refused(global_fixture("CALL 1\nRET\n","LOAD_GLOBAL 0\nRET\n",NULL,TAG_INT,true),argv[1],1);
    refused(global_fixture("PUSH_I64 1\nSTORE_GLOBAL 0\nPUSH_I64 2\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n",NULL,NULL,TAG_INT,false),argv[1],2);
    refused(global_fixture("PUSH_I64 1\nOWN_PACK 0\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_STRUCT,true),argv[1],3);
    refused(global_union_module(true),argv[1],4);
    NvmModule *trapped=global_fixture("PUSH_STR 0\nSTORE_GLOBAL 0\nPUSH_BOOL 0\nASSERT\nPUSH_I64 0\nRET\n",NULL,NULL,TAG_STRING,true);
    char diagnostic[256],path[1024];char *native=nvm2c_emit(trapped,diagnostic,sizeof(diagnostic));CHECK(native);
    snprintf(path,sizeof(path),"%s/trapped.c",argv[1]);FILE *file=fopen(path,"w");CHECK(file);
    CHECK(fputs(native,file)>=0 && fclose(file)==0);free(native);nvm_module_free(trapped);
    printf("%u typed global runtime checks passed\n",checks);return 0;
}
#endif
