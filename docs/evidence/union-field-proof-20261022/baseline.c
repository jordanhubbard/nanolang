/* I qualify descriptive union proof without enrolling public consumers. */
#define main union_declaration_controls_main
#include "/Users/jkh/Src/nanolang-51-platform-qualification/tests/nanoisa/test_union_field_declarations.c"
#undef main
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/managed_union_field_proof_private.h"
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/verifier.h"
#include "/Users/jkh/Src/nanolang-51-platform-qualification/src/nanoisa/nvm2c.h"
static void proof_header(UnionCase *v,uint8_t *code,size_t length) {
    static char *strings[]={"Box","Outer","value","boxed","Some","None"};
    static uint32_t lengths[]={3,5,5,5,4,4};
    NvmModule *m=&v->c.module;
    m->strings=strings;m->string_lengths=lengths;m->code=code;m->code_size=(uint32_t)length;
    m->header.magic[0]=NVM_MAGIC_0;m->header.magic[1]=NVM_MAGIC_1;
    m->header.magic[2]=NVM_MAGIC_2;m->header.magic[3]=NVM_MAGIC_3;
    m->header.format_version=NVM_FORMAT_VERSION;m->header.flags=NVM_FLAG_HAS_MAIN;
    m->header.entry_point=0;v->function.code_length=(uint32_t)length;v->function.name_idx=0;
}
static void proof_i64(uint8_t *code,size_t *n,int64_t value) {
    u8(code,n,OP_PUSH_I64);
    for(unsigned i=0;i<8;i++)u8(code,n,(uint8_t)((uint64_t)value>>(i*8)));
}
static void proof_pack(uint8_t *code,size_t *n,uint8_t kind,uint16_t variant,uint16_t count) {
    u8(code,n,OP_AGG_PACK);u8(code,n,kind);u32(code,n,0);u16(code,n,variant);u16(code,n,count);
}
int main(void) {
 CHECK(union_declaration_controls_main()==0);
 UnionCase v;union_case(&v);uint8_t code[128];size_t n=0;
 proof_i64(code,&n,42);proof_pack(code,&n,AGG_VARIANT,0,1);proof_pack(code,&n,AGG_RECORD,0,1);
 u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_AGG_GET);u16(code,&n,0);u8(code,&n,OP_RET);
 proof_header(&v,code,n);
 NvmVerifyResult verify=nvm_verify(&v.c.module);char error[512]={0};char *native=nvm2c_emit(&v.c.module,error,sizeof error);
 printf("verify=%d message=%s native=%d error=%s\n",verify.ok,verify.error_msg,native!=NULL,error);CHECK(!verify.ok&&!native);
 NvmV2Module wire;memset(&wire,0,sizeof wire);NvmV2Result from=nvm_v2_from_nvm_module(&v.c.module,&wire);
 printf("from=%u\n",from);CHECK(from==NVM_V2_OK);
 NvmModule *copy=NULL;NvmV2Result to=nvm_v2_to_nvm_module(&wire,&copy);printf("to=%u\n",to);CHECK(to==NVM_V2_OK);
 CHECK(copy->layout_size==v.c.module.layout_size&&!memcmp(copy->layout_data,v.c.layouts,copy->layout_size));
 CHECK(copy->ownership_size==v.c.module.ownership_size&&!memcmp(copy->ownership_data,v.c.ownership,copy->ownership_size));
 verify=nvm_verify(copy);memset(error,0,sizeof error);native=nvm2c_emit(copy,error,sizeof error);
 printf("copy verify=%d message=%s native=%d error=%s layouts=%u ownership=%u\n",verify.ok,verify.error_msg,native!=NULL,error,copy->layout_size,copy->ownership_size);
 CHECK(!verify.ok&&!native);nvm_module_free(copy);nvm_v2_module_free(&wire);return 0;
}
