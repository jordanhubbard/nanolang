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
#include "union_fixture.h"
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

static NvmModule *entry(const char *body) {
    NvmModule *m=owned_union_fixture(body,0,TAG_INT);
    m->functions[0].arity=0;m->functions[0].result_tag=TAG_INT;m->functions[0].result_count=1;
    m->ownership_data[18]=0;slot(m->ownership_data+20,TAG_INT,0);
    return m;
}
static NvmModule *call_entry(const char *body) {
    NvmModule *m=entry(body);AsmResult assembled;
    NvmModule *helper=asm_assemble_unverified(".entry 0\n.function identity 1 3 0 union 1\nOWN_MOVE_LOCAL 0\nRET\n.end\n",&assembled);CHECK(helper);
    CHECK(nvm_add_function(m,&helper->functions[0])==1);
    uint8_t parameter=TAG_UNION;CHECK(nvm_set_function_param_types(m,1,&parameter,1));
    m->functions[1].code_offset=m->code_size;
    m->functions[1].name_idx=nvm_add_string(m,"identity",8);m->function_count=2;
    m->code=realloc(m->code,m->code_size+helper->code_size);CHECK(m->code);
    memcpy(m->code+m->code_size,helper->code,helper->code_size);m->code_size+=helper->code_size;
    nvm_module_free(helper);
    uint8_t *p=calloc(144,1);CHECK(p);
    memcpy(p,m->ownership_data,52);word(p+12,2);
    memcpy(p+52,m->ownership_data+16,36);
    p[54]=1;slot(p+56,TAG_UNION,0);word(p+60,1);
    memcpy(p+88,m->ownership_data+52,56);
    free(m->ownership_data);m->ownership_data=p;m->ownership_size=144;
    return m;
}
static NvmModule *nested_entry(const char *body) {
    NvmModule *m=entry(body);NvmV2Layouts old={0};
    CHECK(nvm_v2_layouts_decode(m->layout_data,m->layout_size,&old)==NVM_V2_OK);
    uint32_t name=nvm_add_string(m,"Box<Choice<Handle,int>>",sizeof("Box<Choice<Handle,int>>")-1);
    uint32_t some=nvm_add_string(m,"Some",4),none=nvm_add_string(m,"None",4);
    NvmV2LayoutField field={TAG_UNION,1,some};
    NvmV2Layout items[]={old.items[0],old.items[1],{NVM_V2_LAYOUT_UNION,1,name,&field}};
    NvmV2Layouts layouts={items,3};m->union_count=2;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);nvm_v2_layouts_free(&old);
    m->ownership_data=realloc(m->ownership_data,132);CHECK(m->ownership_data);
    uint8_t *p=m->ownership_data;memset(p+108,0,24);m->ownership_size=132;
    word(p+4,3);p[10]=NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE;word(p+48,2);
    word(p+68,60);word(p+72,2);word(p+108,2);p[112]=2;
    word(p+116,some);p[122]=1;word(p+124,none);p[128]=1;
    return m;
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    const char *select="LOAD_LOCAL 0\nMATCH_TAG 0 owner\nMATCH_TAG 1 ordinary\nMATCH_TAG 2 empty\nPOP\nHALT\n"
        "owner:\nPOP\nOWN_UNPACK_VARIANT 0 0 1\nOWN_STORE_LOCAL 1\nOWN_UNPACK_LOCAL 1\nRET\n"
        "ordinary:\nPOP\nLOAD_LOCAL 0\nAGG_GET 0\nPOP\nOWN_UNPACK_VARIANT 0 1 2\nPOP\nRET\n"
        "empty:\nPOP\nOWN_UNPACK_VARIANT 0 2 0\nPUSH_I64 0\nRET\n";
    const char *construct[]={"PUSH_I64 42\nOWN_PACK 0\nAGG_PACK 1 0 0 1\n",
        "PUSH_I64 17\nPUSH_BOOL 1\nAGG_PACK 1 0 1 2\n","AGG_PACK 1 0 2 0\n"};
    int64_t expected[]={42,17,0};
    for(unsigned arm=0;arm<3;arm++) {
        char body[2048];snprintf(body,sizeof(body),"%sOWN_STORE_LOCAL 2\nOWN_MOVE_LOCAL 2\nOWN_STORE_LOCAL 0\n%s",construct[arm],select);
        execute_module(entry(body),expected[arm],argv[1],arm,TAG_INT,false);
    }
    for(unsigned arm=0;arm<3;arm++) {
        char body[2048];snprintf(body,sizeof(body),"%sCALL 1\nOWN_STORE_LOCAL 0\n%s",construct[arm],select);
        execute_module(call_entry(body),expected[arm],argv[1],3+arm,TAG_INT,false);
    }
    for(unsigned arm=0;arm<3;arm++) {
        char body[2048];snprintf(body,sizeof(body),"%sAGG_PACK 1 1 0 1\nOWN_STORE_LOCAL 2\nLOAD_LOCAL 2\nMATCH_TAG 0 outer\nPOP\nHALT\nouter:\nPOP\nOWN_UNPACK_VARIANT 2 0 1\nOWN_STORE_LOCAL 0\n%s",construct[arm],select);
        execute_module(nested_entry(body),expected[arm],argv[1],6+arm,TAG_INT,false);
    }
    char trap_body[2048];snprintf(trap_body,sizeof(trap_body),"%sAGG_PACK 1 1 0 1\nOWN_STORE_LOCAL 2\nPUSH_BOOL 0\nASSERT\nLOAD_LOCAL 2\nMATCH_TAG 0 outer\nPOP\nHALT\nouter:\nPOP\nOWN_UNPACK_VARIANT 2 0 1\nOWN_STORE_LOCAL 0\n%s",construct[0],select);
    execute_module(nested_entry(trap_body),0,argv[1],9,TAG_INT,true);
    NvmModule *text=entry("PUSH_STR 0\nPUSH_BOOL 1\nAGG_PACK 1 0 1 2\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nMATCH_TAG 1 text\nPOP\nHALT\ntext:\nPOP\nLOAD_LOCAL 0\nAGG_GET 0\nPOP\nOWN_UNPACK_VARIANT 0 1 2\nPOP\nPOP\nPUSH_I64 0\nRET\n");
    NvmV2Layouts text_layouts={0};CHECK(nvm_v2_layouts_decode(text->layout_data,text->layout_size,&text_layouts)==NVM_V2_OK);
    text_layouts.items[1].fields[1].type_tag=TAG_STRING;
    CHECK(nvm_retain_layouts(text,&text_layouts)==NVM_V2_OK);nvm_v2_layouts_free(&text_layouts);
    execute_module(text,0,argv[1],10,TAG_INT,false);
    refused(entry("AGG_PACK 1 0 2 0\nOWN_STORE_LOCAL 0\nOWN_UNPACK_VARIANT 0 2 0\nPUSH_I64 0\nRET\n"),argv[1],0);
    refused(entry("PUSH_I64 1\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nPOP\nPUSH_I64 0\nRET\n"),argv[1],1);
    refused(entry("PUSH_I64 1\nOWN_PACK 0\nAGG_PACK 1 0 0 1\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nMATCH_TAG 0 arm\nPOP\nHALT\narm:\nPOP\nOWN_UNPACK_VARIANT 0 0 2\nPOP\nPOP\nPUSH_I64 0\nRET\n"),argv[1],2);
    refused(entry("AGG_PACK 1 0 2 0\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nMATCH_TAG 2 arm\nPOP\nHALT\narm:\nPOP\nOWN_UNPACK_VARIANT 0 2 0\nOWN_UNPACK_VARIANT 0 2 0\nPUSH_I64 0\nRET\n"),argv[1],3);
    refused(entry("AGG_PACK 1 0 2 0\nOWN_STORE_LOCAL 0\nLOAD_LOCAL 0\nOWN_MOVE_LOCAL 0\nPOP\nPOP\nPUSH_I64 0\nRET\n"),argv[1],4);
    printf("%u owned union runtime checks passed\n",checks);return 0;
}
