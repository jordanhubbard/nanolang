/* I compare real VM execution and generated native results for owned transfers. */
#include "affine_bytecode.h"
#include "reference_places.h"
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
int g_argc=0; char **g_argv=NULL;
static unsigned checks;
#define CHECK(c) do {checks++;assert(c);} while(0)
#include "owned_fixture.h"
static void write_artifact(NvmModule *m,const char *path) {
    NvmV2Module v2;size_t size;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(bytes,1,size,f)==size);CHECK(!fclose(f));
    free(bytes);nvm_v2_module_free(&v2);
}
static void execute_module(NvmModule *m,int64_t expected,const char *dir,unsigned number,uint8_t tag) {
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
    for(unsigned iteration=0;iteration<20;iteration++) {
        NanoValue result=val_void();CHECK(vm_invoke(&vm,0,NULL,0,&result)==VM_OK);
        CHECK(result.tag==tag);
        CHECK((tag==TAG_BOOL?(int64_t)result.as.boolean:tag==TAG_U8?(int64_t)result.as.u8:result.as.i64)==expected);
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
static void execute(const char *body,int64_t expected,const char *dir,unsigned number,uint8_t tag) {
    execute_module(fixture(body,false,false),expected,dir,number,tag);
}
static void refused(NvmModule *m,const char *dir,unsigned number) {
    CHECK(!nvm_verify_owned_module(m).ok);CHECK(!nvm_verify(m).ok);
    VmState vm;vm_init(&vm,m);NanoValue result=val_void();
    CHECK(vm_invoke(&vm,0,NULL,0,&result)!=VM_OK);vm_destroy(&vm);
    char error[256];CHECK(nvm2c_emit(m,error,sizeof(error))==NULL);
    char path[1024];snprintf(path,sizeof(path),"%s/refused%u.nvm",dir,number);write_artifact(m,path);
    nvm_module_free(m);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    const char *move="PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n";
    execute(move,42,argv[1],0,TAG_INT);
    execute("PUSH_I64 10\nOWN_PACK 0\nPUSH_I64 32\nOWN_PACK 0\nOWN_PACK 2\nOWN_STORE_LOCAL 2\nOWN_UNPACK_LOCAL 2\nOWN_STORE_LOCAL 3\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nOWN_UNPACK_LOCAL 3\nADD\nRET\n",42,argv[1],1,TAG_INT);
    execute("PUSH_I64 37\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_BOOL 0\nJMP_FALSE other\nOWN_MOVE_LOCAL 0\nJMP join\nother:\nOWN_MOVE_LOCAL 0\njoin:\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nRET\n",37,argv[1],2,TAG_INT);
    execute("PUSH_I64 0\nSTORE_LOCAL 4\nloop:\nLOAD_LOCAL 4\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_MOVE_LOCAL 0\nOWN_STORE_LOCAL 3\nOWN_UNPACK_LOCAL 3\nPUSH_I64 1\nADD\nSTORE_LOCAL 4\nLOAD_LOCAL 4\nPUSH_I64 1000\nLT\nJMP_TRUE loop\nLOAD_LOCAL 4\nRET\n",1000,argv[1],3,TAG_INT);
    execute("PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nAGG_GET 0\nOWN_UNPACK_LOCAL 0\nADD\nRET\n",84,argv[1],4,TAG_INT);
    execute("PUSH_BOOL 1\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nRET\n",1,argv[1],5,TAG_BOOL);
    execute("PUSH_U8 255\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nRET\n",255,argv[1],6,TAG_U8);
    execute("PUSH_I64 -9223372036854775808\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nPUSH_I64 -1\nDIV\nRET\n",INT64_MIN,argv[1],7,TAG_INT);
    /* A scalar local may exist on only one branch, provided no later read
     * relies on it. Both runtime paths preserve and consume the same owner. */
    for(unsigned branch=0;branch<2;branch++) {
        char body[512];snprintf(body,sizeof(body),
            "PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_BOOL %u\nJMP_FALSE joined\n"
            "PUSH_BOOL 1\nSTORE_LOCAL 4\nLOAD_LOCAL 4\nASSERT\njoined:\nOWN_UNPACK_LOCAL 0\nRET\n",branch);
        NvmModule *m=fixture(body,false,false);m->ownership_data[60]=TAG_BOOL;
        execute_module(m,42,argv[1],8+branch,TAG_INT);
    }
    for(unsigned iterations=0;iterations<4;iterations+=3) {
        char body[768];snprintf(body,sizeof(body),
            "PUSH_I64 42\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nPUSH_I64 0\nSTORE_LOCAL 4\n"
            "loop:\nLOAD_LOCAL 4\nPUSH_I64 %u\nLT\nJMP_FALSE done\n"
            "LOAD_LOCAL 4\nSTORE_LOCAL 5\nLOAD_LOCAL 5\nPUSH_I64 1\nADD\nSTORE_LOCAL 4\nJMP loop\n"
            "done:\nOWN_UNPACK_LOCAL 0\nLOAD_LOCAL 4\nADD\nRET\n",iterations);
        NvmModule *m=fixture(body,false,false);
        uint8_t *next=realloc(m->ownership_data,76);CHECK(next);m->ownership_data=next;
        memset(next+68,0,8);slot(next+68,TAG_INT,NVM_V2_NO_INDEX);
        m->ownership_size=76;next[16]=6;m->functions[0].local_count=6;
        execute_module(m,42+iterations,argv[1],10+iterations/3,TAG_INT);
    }
    refused(fixture("OWN_UNPACK_LOCAL 0\nRET\n",true,false),argv[1],0);
    refused(fixture("PUSH_I64 1\nOWN_PACK 0\nRET\n",false,true),argv[1],1);
    refused(fixture("CALL 0\nRET\n",false,false),argv[1],2);
    refused(fixture("PUSH_F64 1.0\nPOP\nPUSH_I64 0\nRET\n",false,false),argv[1],3);
    refused(fixture("PUSH_I64 1\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_MOVE_LOCAL 0\nOWN_MOVE_LOCAL 0\nRET\n",false,false),argv[1],4);
    NvmModule *borrow=fixture("LOAD_LOCAL 0\nAGG_GET 0\nRET\n",true,false);
    borrow->ownership_data[29]=NVM_REFERENCE_SHARED;
    refused(borrow,argv[1],5);
    NvmModule *floating=fixture("PUSH_F64 1.0\nOWN_PACK 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_LOCAL 0\nPOP\nPUSH_I64 0\nRET\n",false,false);
    NvmV2Layouts layouts={0};CHECK(nvm_v2_layouts_decode(floating->layout_data,floating->layout_size,&layouts)==NVM_V2_OK);
    layouts.items[0].fields[0].type_tag=TAG_FLOAT;
    CHECK(nvm_retain_layouts(floating,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&layouts);
    CHECK(nvm_verify_affine_function(floating,0).ok);
    refused(floating,argv[1],6);
    NvmModule *linked=fixture(move,false,false);
    const NvmModule *links[]={linked};CHECK(!nvm_verify_linked(linked,links,1).ok);
    linked->ownership_data[8]=linked->ownership_data[9]=linked->ownership_data[10]=NVM_LAYOUT_COMPLETE;
    CHECK(nvm_verify_owned_module(linked).ok);
    AsmResult error;
    NvmModule *root=asm_assemble(".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",&error);CHECK(root);
    CHECK(!nvm_verify_linked(root,links,1).ok);
    VmState vm;vm_init(&vm,root);CHECK(vm_link_module(&vm,linked)!=UINT32_MAX);
    CHECK(vm_execute(&vm)==VM_ERR_TYPE_ERROR);vm_destroy(&vm);nvm_module_free(root);
    nvm_module_free(linked);
    printf("%u owned runtime checks passed\n",checks);return 0;
}
