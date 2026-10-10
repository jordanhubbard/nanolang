/* I execute complete checked bytecode and emit a separate native program. */
#define WEBSOCKET_BODY_FIXTURE
#include "test_websocket_body.c"
#undef WEBSOCKET_BODY_FIXTURE
#include "../../src/nanoisa/websocket_codec.h"
#include "../../src/nanovm/websocket_vm_indirect_private.h"
#include "../../src/nanoisa/nvm2c_websocket_indirect_private.h"
#include "websocket_dispatch_host.h"
static void number(Body *c,int64_t value){op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)op(c,(uint8_t)((uint64_t)value>>(8*i)));}
static void string_call(Body *c,unsigned function,bool indirect) {
    if(indirect){op(c,OP_FUNCREF);u32(c,function);op(c,OP_CALL_INDIRECT);u16(c,0);u16(c,1);}
    else {op(c,OP_CALL);u32(c,function);}
}
static void pass(Body *c){op(c,OP_PUSH_BOOL);op(c,0);op(c,OP_ASSERT);}
static void write_file(const char *path,const void *data,size_t size){FILE *f=fopen(path,"wb");CHECK(f && fwrite(data,1,size,f)==size);CHECK(!fclose(f));}
static uint8_t *read_file(const char *path,size_t *size){FILE *f=fopen(path,"rb");CHECK(f && !fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>=0 && !fseek(f,0,SEEK_SET));uint8_t *p=malloc((size_t)n);CHECK(p && fread(p,1,(size_t)n,f)==(size_t)n);CHECK(!fclose(f));*size=(size_t)n;return p;}
static void emit(const char *directory,unsigned port,unsigned which) {
    Program p;memset(&p,0,sizeof p);fixture_locals(&p.f,which!=0);Fixture *f=&p.f;Body *c=&p.code;
    bool indirect=which!=0;int64_t close_timeout=which==2?-1:1000;
    string_call(c,1,indirect);number(c,1000);service(&p,0,UINT16_MAX);one(c,OP_OWN_STORE_LOCAL,1);
    uint32_t connect_error=branch(c,OP_FILE_RESULT_BRANCH,1);
    take(c,1,0);one(c,OP_OWN_STORE_LOCAL,0);op(c,OP_REGION_BEGIN);
    op(c,OP_BORROW_LOCAL_EXCLUSIVE);u16(c,20);u16(c,0);
    number(c,2);one(c,OP_STORE_LOCAL,7);uint32_t loop=c->n;
    op(c,OP_PUSH_BOOL);op(c,1);string_call(c,2,indirect);
    op(c,OP_AGG_PACK);op(c,AGG_RECORD);uint32_t source=1;for(unsigned i=0;i<3;i++)source+=f->bindings.layouts[i]<f->bindings.layouts[2];
    u32(c,source);u16(c,0);u16(c,2);number(c,1000);service(&p,1,20);one(c,OP_STORE_LOCAL,5);
    uint32_t send_error=branch(c,OP_FILE_RESULT_BRANCH,5);
    take(c,5,0);number(c,3);op(c,OP_EQ);op(c,OP_ASSERT);uint32_t sent=branch(c,OP_JMP,0);
    target(c,send_error,c->n);take(c,5,1);op(c,OP_POP);pass(c);target(c,sent,c->n);
    number(c,1000);service(&p,2,20);one(c,OP_STORE_LOCAL,3);uint32_t receive_error=branch(c,OP_FILE_RESULT_BRANCH,3);
    take(c,3,0);one(c,OP_STORE_LOCAL,2);one(c,OP_LOAD_LOCAL,2);one(c,OP_AGG_GET,0);op(c,OP_ASSERT);
    one(c,OP_LOAD_LOCAL,2);one(c,OP_AGG_GET,1);op(c,OP_DUP);op(c,OP_STR_LEN);number(c,3);op(c,OP_EQ);op(c,OP_ASSERT);
    string_call(c,2,indirect);op(c,OP_STR_EQ);op(c,OP_ASSERT);uint32_t received=branch(c,OP_JMP,0);
    target(c,receive_error,c->n);take(c,3,1);op(c,OP_POP);pass(c);target(c,received,c->n);
    one(c,OP_LOAD_LOCAL,7);number(c,1);op(c,OP_SUB);one(c,OP_STORE_LOCAL,7);
    one(c,OP_LOAD_LOCAL,7);number(c,0);op(c,OP_GT);uint32_t back=branch(c,OP_JMP_TRUE,0);target(c,back,loop);
    op(c,OP_REGION_END);one(c,OP_OWN_MOVE_LOCAL,0);number(c,close_timeout);service(&p,3,UINT16_MAX);one(c,OP_STORE_LOCAL,4);
    uint32_t close_error=branch(c,OP_FILE_RESULT_BRANCH,4);
    take(c,4,0);op(c,OP_POP);number(c,77);op(c,OP_RET);
    target(c,close_error,c->n);take(c,4,1);one(c,OP_AGG_GET,0);op(c,OP_RET);
    target(c,connect_error,c->n);take(c,1,1);one(c,OP_AGG_GET,0);op(c,OP_RET);
    NvmFunctionEntry functions[3]={f->function};uint8_t *params[3]={NULL,NULL,NULL};functions[0].code_length=c->n;
    char url[128];snprintf(url,sizeof url,"ws://%s:%u/?mode=test",which==1?"localhost":"127.0.0.1",port);
    uint32_t url_index=name(f,url),message_index=name(f,"aXb");f->names[message_index][1]=0;f->lengths[message_index]=3;
    for(unsigned i=1;i<3;i++) {
        uint32_t start=c->n;op(c,OP_PUSH_STR);u32(c,i==1?url_index:message_index);op(c,OP_RET);
        functions[i]=(NvmFunctionEntry){.name_idx=name(f,i==1?"url":"message"),.result_count=1,.result_tag=TAG_STRING,.code_offset=start,.code_length=c->n-start};
        wr16(f->ownership+104+(i-1)*12,0);wr16(f->ownership+106+(i-1)*12,0);descriptor(f->ownership+108+(i-1)*12,TAG_STRING,0,NVM_V2_NO_INDEX);
    }
    wr32(f->ownership+16,3);f->module.ownership_size=128;f->module.functions=functions;f->module.function_count=3;f->module.function_param_types=params;
    f->module.code=c->bytes;f->module.code_size=c->n;f->module.header.flags=NVM_FLAG_HAS_MAIN;
    NvmV2Module wire={0};CHECK(nvm_websocket_from_module(&f->module,&wire)==NVM_V2_OK);
    for(unsigned i=0;i<3;i++)wire.functions.items[i].max_stack=256;
    size_t size=0;CHECK(nvm_websocket_serialize(&wire,NULL,0,&size)==NVM_V2_OK);uint8_t *bytes=malloc(size);CHECK(bytes);
    CHECK(nvm_websocket_serialize(&wire,bytes,size,&size)==NVM_V2_OK);nvm_v2_module_free(&wire);
    char *text=NULL,error[256];NvmWebSocketRuntimeStatus status=nvm2c_websocket_indirect_private_emit(bytes,size,&text,error,sizeof error);
    if(status!=NVM_WEBSOCKET_RUNTIME_OK)fprintf(stderr,"emit status %u: %s\n",status,error);
    CHECK(status==NVM_WEBSOCKET_RUNTIME_OK);
    char path[4096];snprintf(path,sizeof path,"%s/case-%u.nvm",directory,which);write_file(path,bytes,size);
    snprintf(path,sizeof path,"%s/case-%u.c",directory,which);write_file(path,text,strlen(text));free(text);free(bytes);
}
int main(int argc,char **argv) {
    if(argc==5 && !strcmp(argv[1],"emit")){emit(argv[2],(unsigned)strtoul(argv[3],NULL,10),(unsigned)strtoul(argv[4],NULL,10));return 0;}
    if(argc==3 && !strcmp(argv[1],"refuse")) {
        size_t size;uint8_t *bytes=read_file(argv[2],&size);char *text=(void *)&bytes,error[256];
        CHECK(nvm2c_websocket_indirect_private_emit(bytes,size,&text,error,sizeof error)!=NVM_WEBSOCKET_RUNTIME_OK && text==(void *)&bytes);
        free(bytes);return 0;
    }
    CHECK(argc==6 && !strcmp(argv[1],"vm"));size_t size;uint8_t *bytes=read_file(argv[2],&size);
    NvmWebSocketIndirectOptions options={1,strtoull(argv[3],NULL,10)};NlWsTransportPolicy policy={atoi(argv[4])!=0,atoi(argv[4])!=2,getenv("NANOLANG_RESOLVER"),2000};
    fail_close=atoi(argv[5])!=0;NvmWebSocketRuntimeView out={.fields=99,.values={12345}};
    NvmWebSocketIndirectExecutionReport r=nvm_websocket_vm_indirect_execute(bytes,size,&options,&out,atoi(argv[4])<0?NULL:&policy);free(bytes);report(r,out);return 0;
}
