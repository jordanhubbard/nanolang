/* I execute scalar effects alongside selected resource-union transfers. */
#include "affine_bytecode.h"
#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "assembler.h"
#include "verifier.h"
#include "nvm2c.h"
#include "isa.h"
#include "../../src/nanovm/vm.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int g_argc=0;char **g_argv=NULL;
static unsigned checks;
#define CHECK(c) do {checks++;assert(c);} while(0)
#include "union_fixture.h"
#include "scalar_global_fixture.h"
static void write_artifact(NvmModule *m,const char *path) {
    NvmV2Module v2;size_t size;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,size,f)==size);CHECK(!fclose(f));
    free(bytes);nvm_v2_module_free(&v2);
}
static void execute_module(NvmModule *m,int64_t expected,const char *dir,unsigned number,uint8_t tag,bool trap) {
    if (tag!=TAG_INT) {
        NvmV2Layouts layouts={0};CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&layouts)==NVM_V2_OK);
        layouts.items[0].fields[0].type_tag=tag;layouts.items[1].fields[0].type_tag=tag;
        CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&layouts);
        m->functions[0].result_tag=tag;m->ownership_data[20]=tag;
    }
    NvmVerifyResult valid=nvm_verify(m);
    if(!valid.ok)fprintf(stderr,"%s\n",valid.error_msg);
    CHECK(valid.ok);CHECK(nvm_verify_owned_module(m).ok);
    uint16_t max;CHECK(nvm_verify_function_max_stack(m,0,&max).ok && max<=256);
    VmState vm;vm_init(&vm,m);size_t baseline=vm.heap.stats.num_objects;
    if(m->function_count>1) {NanoValue ignored=val_void();CHECK(vm_invoke(&vm,1,NULL,0,&ignored)==VM_ERR_TYPE_ERROR);}
    for(unsigned iteration=0;iteration<20;iteration++) {
        vm_set_dispatch_profile(&vm,iteration%2?vm_dispatch_profile_all():vm_dispatch_profile_none());
        NanoValue result=val_void();CHECK(vm_invoke(&vm,0,NULL,0,&result)==(trap?VM_ERR_ASSERT_FAILED:VM_OK));
        if(!trap) {
        CHECK(result.tag==tag);
        CHECK((tag==TAG_BOOL?(int64_t)result.as.boolean:tag==TAG_U8?(int64_t)result.as.u8:result.as.i64)==expected);
        }
        CHECK(vm.stack_size==0 && vm.frame_count==0);
        CHECK(vm.heap.stats.num_objects==baseline);
    }
    vm_destroy(&vm);
    char error[256];char *source=nvm2c_emit(m,error,sizeof(error));
    if(!source)fprintf(stderr,"%s\n",error);
    CHECK(source);
    char path[1024];snprintf(path,sizeof(path),"%s/case%u.c",dir,number);
    FILE *f=fopen(path,"w");CHECK(f);CHECK(fputs(source,f)>=0);CHECK(!fclose(f));free(source);
    snprintf(path,sizeof(path),"%s/case%u.nvm",dir,number);write_artifact(m,path);
    printf("case %u %lld\n",number,(long long)expected);nvm_module_free(m);
}
static void refused(NvmModule *m,const char *dir,unsigned number) {
    CHECK(!nvm_verify_owned_module(m).ok);CHECK(!nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);NanoValue result=val_void();
    CHECK(vm_invoke(&vm,0,NULL,0,&result)!=VM_OK);vm_destroy(&vm);
    char error[256];CHECK(nvm2c_emit(m,error,sizeof(error))==NULL);
    char path[1024];snprintf(path,sizeof(path),"%s/refused%u.nvm",dir,number);write_artifact(m,path);
    nvm_module_free(m);
}

static NvmModule *program(const char *prefix,const uint8_t *tags,uint32_t count,const char *callee) {
    char code[4096];snprintf(code,sizeof(code),"%s%s",prefix,body);
    NvmModule *m=owned_union_fixture(code,0,TAG_INT);
    if(callee)helper(m,callee);
    if(count)attach(m,tags,count);
    return m;
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    uint8_t integer=TAG_INT,text=TAG_STRING,pair[]={TAG_STRING,TAG_STRING};
    const char *increment="LOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nRET\n";
    execute_module(program("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nCALL 1\nPOP\nLOAD_GLOBAL 0\nPUSH_I64 2\nEQ\nASSERT\n",&integer,1,increment),7,argv[1],0,TAG_INT,false);
    uint8_t scalars[]={TAG_INT,TAG_U8,TAG_BOOL,TAG_FLOAT};
    execute_module(program("PUSH_I64 3\nSTORE_GLOBAL 0\nPUSH_U8 5\nSTORE_GLOBAL 1\nPUSH_BOOL 1\nSTORE_GLOBAL 2\nPUSH_F64 2.5\nSTORE_GLOBAL 3\nLOAD_GLOBAL 0\nPUSH_I64 3\nEQ\nASSERT\nLOAD_GLOBAL 1\nPUSH_U8 5\nEQ\nASSERT\nLOAD_GLOBAL 2\nASSERT\nLOAD_GLOBAL 3\nPUSH_F64 2.5\nF64_EQ\nASSERT\n",scalars,4,NULL),7,argv[1],1,TAG_INT,false);
    execute_module(program("PUSH_STR 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nSTORE_GLOBAL 1\nPUSH_STR 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 1\nPUSH_STR 0\nEQ\nASSERT\nLOAD_GLOBAL 0\nPUSH_STR 1\nEQ\nASSERT\n",pair,2,NULL),7,argv[1],2,TAG_INT,false);
    for(unsigned selected=0;selected<2;selected++) {
        char prefix[1024];snprintf(prefix,sizeof(prefix),"PUSH_BOOL %u\nJMP_FALSE other\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP join\nother:\nPUSH_I64 2\nSTORE_GLOBAL 0\njoin:\nLOAD_GLOBAL 0\nPUSH_I64 %u\nEQ\nASSERT\n",selected,selected?1:2);
        execute_module(program(prefix,&integer,1,NULL),7,argv[1],3+selected,TAG_INT,false);
    }
    const char *consume=strstr(body,"LOAD_LOCAL 0");CHECK(consume);
    char union_entry[4096];snprintf(union_entry,sizeof(union_entry),"PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nOWN_STORE_LOCAL 0\nLOAD_GLOBAL 0\nPUSH_I64 1\nEQ\nASSERT\n%s",consume);
    NvmModule *owned=owned_union_fixture(union_entry,0,TAG_INT);
    helper(owned,"LOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nPUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nRET\n");
    owned->functions[1].result_tag=TAG_UNION;slot(owned->ownership_data+56,TAG_UNION,0);word(owned->ownership_data+60,1);
    attach(owned,&integer,1);execute_module(owned,7,argv[1],5,TAG_INT,false);
    char trap[4096];snprintf(trap,sizeof(trap),"PUSH_STR 0\nSTORE_GLOBAL 0\nPUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 0\nPUSH_BOOL 0\nASSERT\n%s",consume);
    NvmModule *trapped=owned_union_fixture(trap,0,TAG_INT);attach(trapped,&text,1);
    execute_module(trapped,7,argv[1],6,TAG_INT,true);
    execute_module(program("PUSH_I64 0\nSTORE_GLOBAL 0\nloop:\nLOAD_GLOBAL 0\nPUSH_I64 2\nLT\nJMP_FALSE after\nLOAD_GLOBAL 0\nPUSH_I64 1\nADD\nSTORE_GLOBAL 0\nJMP loop\nafter:\nLOAD_GLOBAL 0\nPUSH_I64 2\nEQ\nASSERT\n",&integer,1,NULL),7,argv[1],7,TAG_INT,false);
    execute_module(program("PUSH_STR 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\nLOAD_GLOBAL 0\nPUSH_STR 1\nEQ\nASSERT\n",&text,1,"PUSH_STR 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n"),7,argv[1],8,TAG_INT,false);
    refused(program("",&integer,1,NULL),argv[1],0);
    refused(program("PUSH_BOOL 1\nSTORE_GLOBAL 0\n",&integer,1,NULL),argv[1],1);
    refused(program("PUSH_I64 0\nSTORE_GLOBAL 1\n",&integer,1,NULL),argv[1],2);
    refused(program("PUSH_I64 7\nOWN_PACK 0\nSTORE_GLOBAL 0\n",&integer,1,NULL),argv[1],3);
    refused(program("PUSH_I64 7\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_GLOBAL 0\n",&integer,1,NULL),argv[1],4);
    refused(program("PUSH_BOOL 1\nJMP_FALSE other\nPUSH_I64 1\nSTORE_GLOBAL 0\nJMP join\nother:\nNOP\njoin:\nLOAD_GLOBAL 0\nPOP\n",&integer,1,NULL),argv[1],5);
    refused(program("CALL 1\nPOP\nPUSH_I64 0\nSTORE_GLOBAL 0\n",&integer,1,increment),argv[1],6);
    refused(program("PUSH_I64 0\nSTORE_GLOBAL 0\n",&integer,0,NULL),argv[1],7);
    refused(program("PUSH_I64 0\nSTORE_GLOBAL 0\nCALL 1\nPOP\n",&integer,1,"PUSH_BOOL 1\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n"),argv[1],8);
    printf("I passed %u scalar-global runtime checks.\n",checks);return 0;
}
