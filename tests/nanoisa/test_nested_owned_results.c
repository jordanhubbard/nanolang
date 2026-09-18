/* I transfer nested result roots without losing their descendants. */
#define OWNED_RESULT_ALLOC_TEST
#include "test_owned_value_results.c"
#include "disassembler.h"
static NvmModule *nested_fixture(unsigned index) {
    bool empty=index==3;unsigned root=index==2?4:2;
    char source[18000]=".types 5 0 0\n.entry 0\n";
    append(source,sizeof(source),".function main 0 8 0 int 1\nPUSH_I64 99\nOWN_PACK 3\nOWN_STORE_LOCAL 0\nREGION_BEGIN\nBORROW_LOCAL_SHARED 0 0\n");
    for(unsigned i=0;i<2;i++)append(source,sizeof(source),
        "CALL 1\nCALL 2\nOWN_STORE_LOCAL 1\n%sREF_GET 0 0\nPUSH_I64 99\nEQ\nASSERT\nOWN_MOVE_LOCAL 1\nCALL 3\nPUSH_I64 42\nEQ\nASSERT\n",index==5?"PUSH_BOOL 0\nASSERT\n":"");
    append(source,sizeof(source),"REGION_END\nOWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 42\nRET\n.end\n.function factory 0 8 0 struct 1\nPUSH_BOOL %u\nJMP_FALSE alternate\n",index==1?0:1);
    for(unsigned branch=0;branch<2;branch++) {
        if(branch)append(source,sizeof(source),"alternate:\n");
        append(source,sizeof(source),"%sOWN_PACK 0\nPUSH_BOOL 1\nOWN_PACK 1\nPUSH_U8 200\nOWN_PACK %u\n%sRET\n",empty?"":"PUSH_I64 42\n",root,index==4?"PUSH_BOOL 0\nASSERT\n":"");
    }
    append(source,sizeof(source),".end\n.function relay 1 8 0 struct 1\nOWN_MOVE_LOCAL 0\nRET\n.end\n.parameters 2 struct\n.function consume 1 8 0 int 1\nOWN_UNPACK_LOCAL 0\nPUSH_U8 200\nEQ\nASSERT\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nASSERT\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\n%sRET\n.end\n.parameters 3 struct\n",empty?"PUSH_I64 42\n":"");
    AsmResult assembled;NvmModule *m=asm_assemble_unverified(source,&assembled);
    if(!m)fprintf(stderr,"%s\n",assembled.message);
    CHECK(m);
    NvmV2LayoutField scalar={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2LayoutField inner[]={{TAG_STRUCT,0,NVM_V2_NO_INDEX},{TAG_BOOL,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2LayoutField outer[]={{TAG_STRUCT,1,NVM_V2_NO_INDEX},{TAG_U8,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX}};
    NvmV2Layout rows[]={{NVM_V2_LAYOUT_STRUCT,empty?0:1,NVM_V2_NO_INDEX,&scalar},
        {NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,inner},{NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,outer},
        {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&scalar},{NVM_V2_LAYOUT_STRUCT,2,NVM_V2_NO_INDEX,outer}};
    NvmV2Layouts layouts={rows,5};CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    m->ownership_size=20+4*76+4;m->ownership_data=calloc(m->ownership_size,1);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;word(p,2);word(p+4,5);memset(p+8,3,5);word(p+16,4);
    for(unsigned f=0;f<4;f++) {
        unsigned h=20+f*76;p[h]=8;p[h+2]=(uint8_t)m->functions[f].arity;
        slot(p+h+4,f==1||f==2?TAG_STRUCT:TAG_INT,0,f==1||f==2?root:NVM_V2_NO_INDEX);
        for(unsigned l=0;l<8;l++) {
            unsigned nominal=NVM_V2_NO_INDEX;
            if(f==0&&l<2)nominal=l?root:3;
            if((f==2||f==3)&&l==0)nominal=root;
            if(f==3&&l==1)nominal=1;
            if(f==3&&l==2)nominal=0;
            slot(p+h+12+l*8,nominal==NVM_V2_NO_INDEX?TAG_INT:TAG_STRUCT,0,nominal);
        }
    }
    bool needs=false;CHECK(nvm_ownership_contracts_validate(m,&needs)==NVM_V2_OK&&needs);return m;
}
static void nested_roundtrip(NvmModule *m) {
    char *text=disasm_module_styled(m,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult error;NvmModule *copy=asm_assemble(text,&error);CHECK(copy);
    CHECK(m->layout_size==copy->layout_size&&!memcmp(m->layout_data,copy->layout_data,m->layout_size));
    CHECK(m->ownership_size==copy->ownership_size&&!memcmp(m->ownership_data,copy->ownership_data,m->ownership_size));
    consuming_verified(copy);free(text);nvm_module_free(copy);
}
#ifndef NESTED_RESULT_ALLOC_TEST
int main(int argc,char **argv) {
    (void)result_fixture;CHECK(argc==2);
    NvmModule *wrong=nested_fixture(0);
    /* I refuse a same-shaped but different exact factory result identity. */
    slot(wrong->ownership_data+20+76+4,TAG_STRUCT,0,4);
    CHECK(!nvm_verify(wrong).ok);CHECK(!nvm_verify_owned_module(wrong).ok);
    char error[256];CHECK(!nvm2c_emit(wrong,error,sizeof(error)));nvm_module_free(wrong);
    for(unsigned index=0;index<6;index++) {
        NvmModule *m=nested_fixture(index);consuming_verified(m);nested_roundtrip(m);artifacts(m,argv[1],index);
        VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
        for(unsigned api=0;api<4;api++)for(unsigned repeat=0;repeat<2;repeat++) {
            NanoValue value=val_int(-91);VmResult status=result_api(&vm,api,&value);
            CHECK(status==(index>=4?VM_ERR_ASSERT_FAILED:VM_OK));
            if(status==VM_OK){if(api==1||api==2)value=vm.stack[--vm.stack_size];CHECK(value.tag==TAG_INT&&value.as.i64==42);vm_release(&vm.heap,value);}
            result_clean(&vm,baseline);
        }
        vm_destroy(&vm);nvm_module_free(m);printf("case %u %u 42\n",index,index>=4?2:0);
    }
    printf("%u nested owned result checks passed\n",checks);return 0;
}
#endif
