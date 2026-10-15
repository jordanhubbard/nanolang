/* I build checked TCP programs and compare independently compiled native execution. */
#define FILE_BODY_MAIN prior_socket_body_fixture_main
#include "test_socket_body.c"
#undef FILE_BODY_MAIN
#include "../../src/nanovm/socket_vm_indirect_private.h"
#include "../../src/nanoisa/nvm2c_socket_indirect_private.h"
#include "socket_dispatch_host.h"
static void number(Body *c,int64_t value){op(c,OP_PUSH_I64);for(unsigned i=0;i<8;i++)op(c,(uint8_t)((uint64_t)value>>(8*i)));}
static void jump(Body *c,uint32_t pc){uint32_t at=branch(c,OP_JMP,0);wr32(c->bytes+at+1,(uint32_t)(int32_t)((int64_t)pc-at));}
static void callable(Body *c,uint32_t target){op(c,OP_FUNCREF);u32(c,target);}
static void call_owned(Body *c,uint32_t target,bool indirect){if(indirect){callable(c,target);op(c,OP_CALL_INDIRECT);u16(c,1);u16(c,1);}else{op(c,OP_CALL);u32(c,target);}}
static void send_call(Body *c,NvmModule *m,bool indirect){
 number(c,165);
 if(indirect){const char refs[]={20,0,(char)255,(char)255};uint32_t index=nvm_add_string(m,refs,4);CHECK(index!=UINT32_MAX);
  callable(c,2);op(c,OP_FILE_CALL_INDIRECT_REFS);u16(c,2);u16(c,1);u32(c,index);
 }else{op(c,OP_CALL_REF);u32(c,2);u16(c,20);}
}
static Body program(NvmModule *m,NvmSocketNominalBindings b,int family,int port,bool indirect,bool trap){
 Body c={0};int64_t endpoint[]={family,family==4?INT64_C(0x7f000001):0,0,0,family==6?1:0,port,0};
 for(unsigned i=0;i<7;i++)number(&c,endpoint[i]);
 uint32_t source=0;for(unsigned i=0;i<3;i++)source+=b.layouts[i]<b.layouts[8];
 op(&c,OP_AGG_PACK);op(&c,AGG_RECORD);u32(&c,source);u16(&c,0);u16(&c,7);
 op(&c,OP_FILE_SERVICE);u32(&c,b.imports[0]);u16(&c,UINT16_MAX);
 call_owned(&c,4,indirect);one(&c,OP_OWN_STORE_LOCAL,1);uint32_t failed=branch(&c,OP_FILE_RESULT_BRANCH,1);
 take_result(&c,1,0);call_owned(&c,1,indirect);one(&c,OP_OWN_STORE_LOCAL,0);
 op(&c,OP_REGION_BEGIN);op(&c,OP_BORROW_LOCAL_EXCLUSIVE);u16(&c,20);u16(&c,0);
 if(trap){op(&c,OP_PUSH_BOOL);op(&c,0);op(&c,OP_ASSERT);}
 uint32_t finish=c.n;service(&c,b,2,20);one(&c,OP_STORE_LOCAL,7);uint32_t pending=branch(&c,OP_FILE_RESULT_BRANCH,7);
 take_result(&c,7,0);op(&c,OP_POP);
 uint32_t send=c.n;send_call(&c,m,indirect);one(&c,OP_STORE_LOCAL,6);uint32_t again=branch(&c,OP_FILE_RESULT_BRANCH,6);
 take_result(&c,6,0);number(&c,1);op(&c,OP_EQ);op(&c,OP_ASSERT);
 uint32_t receive=c.n;service(&c,b,3,20);one(&c,OP_STORE_LOCAL,8);uint32_t empty=branch(&c,OP_FILE_RESULT_BRANCH,8);
 take_result(&c,8,0);one(&c,OP_STORE_LOCAL,5);one(&c,OP_LOAD_LOCAL,5);one(&c,OP_AGG_GET,0);number(&c,90);op(&c,OP_EQ);op(&c,OP_ASSERT);
 one(&c,OP_LOAD_LOCAL,5);one(&c,OP_AGG_GET,1);op(&c,OP_NOT);op(&c,OP_ASSERT);
 op(&c,OP_REGION_END);one(&c,OP_OWN_MOVE_LOCAL,0);service(&c,b,4,UINT16_MAX);one(&c,OP_STORE_LOCAL,3);
 uint32_t closeerror=branch(&c,OP_FILE_RESULT_BRANCH,3);take_result(&c,3,0);op(&c,OP_POP);number(&c,77);op(&c,OP_RET);
 target(&c,closeerror);take_result(&c,3,1);op(&c,OP_POP);number(&c,77);op(&c,OP_RET);
 target(&c,empty);take_result(&c,8,1);op(&c,OP_POP);jump(&c,receive);
 target(&c,again);take_result(&c,6,1);op(&c,OP_POP);jump(&c,send);
 target(&c,pending);take_result(&c,7,1);op(&c,OP_POP);jump(&c,finish);
 target(&c,failed);take_result(&c,1,1);one(&c,OP_AGG_GET,0);op(&c,OP_RET);return c;
}
static uint8_t *wire(NvmModule *m,size_t *size){NvmV2Module v={0};CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
 CHECK(nvm_v2_module_serialize(&v,NULL,0,size)==NVM_V2_OK);uint8_t *data=malloc(*size);CHECK(data);
 CHECK(nvm_v2_module_serialize(&v,data,*size,size)==NVM_V2_OK);nvm_v2_module_free(&v);return data;}
static void capture(const char *directory,int port4,int port6){
 for(unsigned i=0;i<4;i++){
  NvmSocketNominalBindings b;NvmModule *m=bodymodule(&b,i==1);Body c=program(m,b,i==1?6:4,i==2?0:i==1?port6:port4,i!=0,i==3);setbody(m,0,c);
  size_t size;uint8_t *bytes=wire(m,&size);nvm_module_free(m);char *text=NULL,error[256];
  NvmSocketRuntimeStatus s=nvm2c_socket_indirect_private_emit(bytes,size,&text,error,sizeof error);
  if(s!=NVM_SOCKET_RUNTIME_OK){fprintf(stderr,"case %u: %u %s\n",i,(unsigned)s,error);}
  CHECK(s==NVM_SOCKET_RUNTIME_OK);
  char *prior=(void *)(uintptr_t)1;
  CHECK(nvm2c_socket_indirect_private_emit(bytes,size-1,&prior,error,sizeof error)!=NVM_SOCKET_RUNTIME_OK && prior==(void *)(uintptr_t)1);
  char path[4096];CHECK(snprintf(path,sizeof path,"%s/case-%u.nvm",directory,i)>0);FILE *f=fopen(path,"wb");CHECK(f && fwrite(bytes,1,size,f)==size && fclose(f)==0);
  CHECK(snprintf(path,sizeof path,"%s/case-%u.c",directory,i)>0);f=fopen(path,"wb");CHECK(f && fwrite(text,1,strlen(text),f)==strlen(text) && fclose(f)==0);free(text);free(bytes);
 }
 CHECK(!dispatch_opens && !dispatch_closes);puts("PASS TCP emission without host acquisition");
}
#ifndef SOCKET_DISPATCH_MAIN
#define SOCKET_DISPATCH_MAIN main
#endif
int SOCKET_DISPATCH_MAIN(int argc,char **argv){
 if(argc==5 && !strcmp(argv[1],"emit")){capture(argv[2],atoi(argv[3]),atoi(argv[4]));return 0;}
 CHECK(argc==5 && !strcmp(argv[1],"vm"));FILE *f=fopen(argv[2],"rb");CHECK(f && fseek(f,0,SEEK_END)==0);long n=ftell(f);CHECK(n>0 && fseek(f,0,SEEK_SET)==0);
 uint8_t *bytes=malloc((size_t)n);CHECK(bytes && fread(bytes,1,(size_t)n,f)==(size_t)n && fclose(f)==0);
 NvmSocketIndirectOptions options={1,strtoull(argv[3],NULL,10)};dispatch_close_fault=atoi(argv[4])!=0;NvmSocketRuntimeView out={.fields=99,.values={12345}};
 NvmSocketIndirectExecutionReport report=nvm_socket_vm_indirect_execute(bytes,(size_t)n,&options,&out);free(bytes);dispatch_report(report,out);return 0;
}
