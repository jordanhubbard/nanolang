/* I exercise private carrier calls, not a matched CODE dispatcher. */
#define FILE_BODY_MAIN prior_socket_body_fixture_main
#include "test_socket_body.c"
#undef FILE_BODY_MAIN
#include "../../src/nanoisa/socket_runtime.h"
#include "../../src/nanoisa/socket_cyclic_runtime.h"
#include "../../src/nanoisa/socket_indirect_runtime.h"
#include <arpa/inet.h>
#include <errno.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#define RT(x) CHECK((x)==NVM_SOCKET_RUNTIME_OK)
static unsigned acquisition_attempts,close_attempts;
static bool close_failure;
int socket_runtime_socket(int domain,int type,int protocol){acquisition_attempts++;return socket(domain,type,protocol);}
int socket_runtime_close(int fd){close_attempts++;int rc=close(fd);if(rc==0 && close_failure){close_failure=false;errno=EIO;return -1;}return rc;}
static int allocation_budget=-1;
static unsigned allocation_live;
void *socket_runtime_malloc(size_t n){if(!allocation_budget)return NULL;if(allocation_budget>0)allocation_budget--;void *p=malloc(n);if(p)allocation_live++;return p;}
void *socket_runtime_calloc(size_t n,size_t s){if(n && s>SIZE_MAX/n)return NULL;void *p=socket_runtime_malloc(n*s);if(p)memset(p,0,n*s);return p;}
void socket_runtime_free(void *p){if(p){CHECK(allocation_live);allocation_live--;}free(p);}
static uint8_t *runtime_bytes(NvmSocketNominalBindings *b,size_t *size){
 NvmModule *m=bodymodule(b,false);setbody(m,0,lifecycle_code(*b));NvmV2Module wire={0};
 CHECK(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);CHECK(nvm_v2_module_serialize(&wire,NULL,0,size)==NVM_V2_OK);
 uint8_t *bytes=malloc(*size);CHECK(bytes);CHECK(nvm_v2_module_serialize(&wire,bytes,*size,size)==NVM_V2_OK);
 nvm_v2_module_free(&wire);nvm_module_free(m);return bytes;
}
static void make_endpoint(NvmSocketRuntime *c,int family,int port){
 int64_t values[]={family,family==4?INT64_C(0x7f000001):0,0,0,family==6?1:0,port,0};uint32_t inputs[]={0,1,2,3,4,5,6};
 for(unsigned i=0;i<7;i++)RT(nvm_socket_runtime_scalar(c,i,TAG_INT,values[i]));
 RT(nvm_socket_runtime_construct(c,8,0,inputs,7,7));
}
static int listener(int family,int *port){
 int fd=socket(family==4?AF_INET:AF_INET6,SOCK_STREAM,0);CHECK(fd>=0);
 if(family==4){struct sockaddr_in addr={0};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
  CHECK(bind(fd,(struct sockaddr *)&addr,sizeof addr)==0);socklen_t n=sizeof addr;CHECK(getsockname(fd,(struct sockaddr *)&addr,&n)==0);*port=ntohs(addr.sin_port);
 }else{struct sockaddr_in6 addr={0};addr.sin6_family=AF_INET6;addr.sin6_addr=in6addr_loopback;
  CHECK(bind(fd,(struct sockaddr *)&addr,sizeof addr)==0);socklen_t n=sizeof addr;CHECK(getsockname(fd,(struct sockaddr *)&addr,&n)==0);*port=ntohs(addr.sin6_port);}
 CHECK(listen(fd,1)==0);return fd;
}
static NvmSocketRuntimeView view(NvmSocketRuntime *c,uint32_t root){NvmSocketRuntimeView v;CHECK(nvm_socket_runtime_view(c,root,&v));return v;}
static void publish(NvmSocketRuntime **c,bool failed_close){
 RT(nvm_socket_runtime_scalar(*c,0,TAG_INT,42));RT(nvm_socket_runtime_complete_root(*c,0));
 NvmSocketRuntimeView out={.fields=99};NvmSocketRuntimeReport r=nvm_socket_runtime_destroy(c,&out);
 if(failed_close){CHECK(r.status==NVM_SOCKET_RUNTIME_CLEANUP && r.cleanup.cleanup_failures && r.cleanup.first_cleanup.closure_unknown && out.fields==99);}
 else {CHECK(r.status==NVM_SOCKET_RUNTIME_OK && !r.cleanup.cleanup_failures && r.cleanup.execution==NL_SOCKET_VALUE_OK);CHECK(out.fields==1 && out.values[0]==42);}
 CHECK(!*c);
}
static void runtime_lifecycle(const uint8_t *bytes,size_t size,NvmSocketNominalBindings b,int family,NvmSocketRuntimeMode mode,unsigned disposition){
 bool abandon=(disposition&1)!=0,failed_close=(disposition&2)!=0;
 int port=0,server=listener(family,&port);NvmSocketRuntime *c=NULL;RT(nvm_socket_runtime_create(bytes,size,mode,&c));RT(nvm_socket_runtime_begin(c));
 make_endpoint(c,family,port);RT(nvm_socket_runtime_service(c,b.imports[0],UINT32_MAX,7,8));CHECK(!view(c,7).initialized);
 NvmSocketFlowArm arm;RT(nvm_socket_runtime_result_arm(c,8,&arm));CHECK(arm==NVM_SOCKET_FLOW_ARM_OK);
 RT(nvm_socket_runtime_take(c,8,arm,9));RT(nvm_socket_runtime_region_begin(c));RT(nvm_socket_runtime_borrow(c,9,0));
 struct pollfd ready={server,POLLIN,0};CHECK(poll(&ready,1,2000)==1);int peer=accept(server,NULL,NULL);CHECK(peer>=0);
 RT(nvm_socket_runtime_service(c,b.imports[2],0,UINT32_MAX,11));CHECK(view(c,11).arm==NVM_SOCKET_FLOW_ARM_OK);RT(nvm_socket_runtime_drop(c,11));
 if(abandon){
  close_failure=failed_close;unsigned before=close_attempts;
  NvmSocketRuntimeView out={.fields=99};NvmSocketRuntimeReport report=nvm_socket_runtime_destroy(&c,&out);
  CHECK(report.status==NVM_SOCKET_RUNTIME_STATE && (report.cleanup.cleanup_failures!=0)==failed_close && out.fields==99 && !c && close_attempts==before+1);
  if(failed_close)CHECK(report.cleanup.first_cleanup.closure_unknown);
 }else{
  RT(nvm_socket_runtime_scalar(c,10,TAG_INT,256));RT(nvm_socket_runtime_service(c,b.imports[1],0,10,11));
  NvmSocketRuntimeView error=view(c,11);CHECK(error.arm==NVM_SOCKET_FLOW_ARM_ERROR && error.fields==11 && error.values[0]==NL_SOCKET_ARGUMENT);
  RT(nvm_socket_runtime_drop(c,11));
  RT(nvm_socket_runtime_scalar(c,10,TAG_INT,173));RT(nvm_socket_runtime_service(c,b.imports[1],0,10,11));
  CHECK(view(c,11).arm==NVM_SOCKET_FLOW_ARM_OK && view(c,11).values[0]==1);RT(nvm_socket_runtime_drop(c,11));
  ready=(struct pollfd){peer,POLLIN,0};CHECK(poll(&ready,1,2000)==1);unsigned char byte=0;CHECK(read(peer,&byte,1)==1 && byte==173);
  RT(nvm_socket_runtime_service(c,b.imports[3],0,UINT32_MAX,11));CHECK(view(c,11).arm==NVM_SOCKET_FLOW_ARM_ERROR && view(c,11).values[0]==NL_SOCKET_WOULD_BLOCK);RT(nvm_socket_runtime_drop(c,11));
  byte=219;CHECK(write(peer,&byte,1)==1);
  bool received=false;
  for(unsigned i=0;i<1000;i++){
   RT(nvm_socket_runtime_service(c,b.imports[3],0,UINT32_MAX,11));NvmSocketRuntimeView got=view(c,11);
   if(got.arm==NVM_SOCKET_FLOW_ARM_OK){CHECK(got.fields==2 && got.values[0]==219 && !got.values[1]);received=true;}
   else CHECK(got.values[0]==NL_SOCKET_WOULD_BLOCK);
   RT(nvm_socket_runtime_drop(c,11));if(received)break;usleep(1000);
  }CHECK(received);
  CHECK(shutdown(peer,SHUT_WR)==0);bool eof=false;
  for(unsigned i=0;i<1000;i++){
   RT(nvm_socket_runtime_service(c,b.imports[3],0,UINT32_MAX,11));NvmSocketRuntimeView got=view(c,11);
   if(got.arm==NVM_SOCKET_FLOW_ARM_OK){CHECK(got.fields==2 && !got.values[0] && got.values[1]);eof=true;}
   else CHECK(got.values[0]==NL_SOCKET_WOULD_BLOCK);
   RT(nvm_socket_runtime_drop(c,11));if(eof)break;usleep(1000);
  }CHECK(eof);
  RT(nvm_socket_runtime_end_reference(c,0));
  RT(nvm_socket_runtime_borrow_shared(c,9,0));RT(nvm_socket_runtime_borrow_shared(c,9,1));
  RT(nvm_socket_runtime_bind_formal(c,0,10,2));CHECK(view(c,10).formal && view(c,10).type.mode==1);
  RT(nvm_socket_runtime_end_reference(c,2));CHECK(!view(c,10).initialized);
  RT(nvm_socket_runtime_end_reference(c,0));RT(nvm_socket_runtime_end_reference(c,1));RT(nvm_socket_runtime_region_end(c));
  RT(nvm_socket_runtime_move(c,9,10));close_failure=failed_close;unsigned before=close_attempts;RT(nvm_socket_runtime_service(c,b.imports[4],UINT32_MAX,10,11));
  NvmSocketRuntimeView closed=view(c,11);CHECK(!view(c,10).initialized && close_attempts==before+1);
  if(failed_close){CHECK(closed.arm==NVM_SOCKET_FLOW_ARM_ERROR && closed.fields==11 && closed.values[0]==NL_SOCKET_IO && closed.values[1]==EIO && closed.values[4]==1 && !closed.values[5] && closed.values[7] && closed.values[9]);}
  else CHECK(closed.arm==NVM_SOCKET_FLOW_ARM_OK);
  RT(nvm_socket_runtime_drop(c,11));publish(&c,failed_close);CHECK(close_attempts==before+1);
 }
 CHECK(close(peer)==0 && close(server)==0);
}
static void invalid_endpoint(const uint8_t *bytes,size_t size,NvmSocketNominalBindings b){
 NvmSocketRuntime *c=NULL;RT(nvm_socket_runtime_create(bytes,size,NVM_SOCKET_RUNTIME_VM,&c));RT(nvm_socket_runtime_begin(c));make_endpoint(c,4,0);unsigned before=acquisition_attempts;
 RT(nvm_socket_runtime_service(c,b.imports[0],UINT32_MAX,7,8));NvmSocketFlowArm arm;RT(nvm_socket_runtime_result_arm(c,8,&arm));CHECK(arm==NVM_SOCKET_FLOW_ARM_ERROR && acquisition_attempts==before);
 RT(nvm_socket_runtime_take(c,8,arm,9));NvmSocketRuntimeView v=view(c,9);CHECK(v.fields==11 && v.values[0]==NL_SOCKET_ARGUMENT);
 for(unsigned i=1;i<11;i++)CHECK(v.values[i]==0);
 RT(nvm_socket_runtime_project(c,9,10,10));CHECK(view(c,10).type.tag==TAG_BOOL && !view(c,10).values[0]);RT(nvm_socket_runtime_drop(c,10));publish(&c,false);
}
static void preparation(const uint8_t *bytes,size_t size){
 for(unsigned indirect=0;indirect<2;indirect++)for(unsigned mode=0;mode<2;mode++){
  NvmSocketRuntime *c=NULL;NvmSocketCyclicOptions cyclic={1,0};NvmSocketIndirectOptions call={1,0};
  if(indirect)RT(nvm_socket_runtime_indirect_create(bytes,size,(NvmSocketRuntimeMode)mode,&call,&c));
  else RT(nvm_socket_runtime_cyclic_create(bytes,size,(NvmSocketRuntimeMode)mode,&cyclic,&c));
  RT(nvm_socket_runtime_begin(c));RT(nvm_socket_runtime_frame_start(c));NvmSocketRuntimeView out={.fields=99};
  CHECK((indirect?nvm_socket_runtime_indirect_enter(c):nvm_socket_runtime_cyclic_enter(c))==NVM_SOCKET_RUNTIME_LIMIT);
  if(indirect){NvmSocketIndirectExecutionReport r=nvm_socket_runtime_indirect_destroy(&c,&out);CHECK(r.fuel_exhausted && !r.instructions_started && r.runtime.status==NVM_SOCKET_RUNTIME_LIMIT && !r.runtime.cleanup.cleanup_failures);}
  else {NvmSocketCyclicExecutionReport r=nvm_socket_runtime_cyclic_destroy(&c,&out);CHECK(r.fuel_exhausted && !r.instructions_started && r.runtime.status==NVM_SOCKET_RUNTIME_LIMIT && !r.runtime.cleanup.cleanup_failures);}
  CHECK(!c && out.fields==99);
 }
#ifdef SOCKET_RUNTIME_INSTRUMENT
 bool success=false;unsigned failures=0;
 for(int n=0;n<64;n++){
  NvmSocketRuntime *c=(void *)(uintptr_t)1;allocation_budget=n;NvmSocketRuntimeStatus status=nvm_socket_runtime_create(bytes,size,NVM_SOCKET_RUNTIME_VM,&c);allocation_budget=-1;
  if(status==NVM_SOCKET_RUNTIME_OK){nvm_socket_runtime_destroy(&c,NULL);success=true;CHECK(!allocation_live);break;}
  CHECK(status==NVM_SOCKET_RUNTIME_MEMORY && c==(void *)(uintptr_t)1 && !allocation_live);failures++;
 }CHECK(success && failures>=5);
#endif
}
/* I drive the Error path manually to check the concrete frame/variant protocol.
 * This small fixture is not a production instruction-coverage claim. */
static void checked_frames(const uint8_t *bytes,size_t size){
 for(unsigned indirect=0;indirect<2;indirect++)for(unsigned mode=0;mode<2;mode++){
  NvmSocketRuntime *c=NULL;NvmSocketCyclicOptions cyclic={1,100};NvmSocketIndirectOptions call={1,100};
  if(indirect)RT(nvm_socket_runtime_indirect_create(bytes,size,(NvmSocketRuntimeMode)mode,&call,&c));
  else RT(nvm_socket_runtime_cyclic_create(bytes,size,(NvmSocketRuntimeMode)mode,&cyclic,&c));
  RT(nvm_socket_runtime_begin(c));RT(nvm_socket_runtime_frame_start(c));bool completed=false;unsigned before=acquisition_attempts,steps=0;
  for(;steps<100;steps++){
   NvmSocketRuntimeFrameView f;CHECK(nvm_socket_runtime_frame_view(c,&f));NvmSocketCodeInstruction in;
   if(indirect){CHECK(nvm_socket_indirect_hosted_instruction(nvm_socket_runtime_indirect_plan(c),f.function,(uint16_t)f.instruction,&in));RT(nvm_socket_runtime_indirect_enter(c));}
   else {CHECK(nvm_socket_cyclic_hosted_instruction(nvm_socket_runtime_cyclic_plan(c),f.function,(uint16_t)f.instruction,&in));RT(nvm_socket_runtime_cyclic_enter(c));}
   uint32_t src=0,dst=0,scratch=0;uint8_t edge=0;NvmSocketFlowArm arm;
   switch(in.decoded.opcode){
   case OP_PUSH_I64:
    CHECK(nvm_socket_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst));RT(nvm_socket_runtime_scalar(c,dst,TAG_INT,0));break;
   case OP_AGG_PACK: {
    uint32_t inputs[7];CHECK(f.stack_count==7 && in.catalog_ordinal==8);
    for(uint16_t i=0;i<7;i++)CHECK(nvm_socket_runtime_frame_operand(c,i,&inputs[i]));
    CHECK(nvm_socket_runtime_frame_scratch(c,&scratch));RT(nvm_socket_runtime_construct(c,8,0,inputs,7,scratch));
    CHECK(nvm_socket_runtime_frame_reserve(c,0,&dst));RT(nvm_socket_runtime_move(c,scratch,dst));break;
   }
   case OP_FILE_SERVICE:
    CHECK(in.catalog_ordinal==0 && f.stack_count==1);CHECK(nvm_socket_runtime_frame_operand(c,0,&src));CHECK(nvm_socket_runtime_frame_scratch(c,&scratch));
    RT(nvm_socket_runtime_service(c,in.decoded.operands[0].u32,UINT32_MAX,src,scratch));CHECK(nvm_socket_runtime_frame_reserve(c,0,&dst));RT(nvm_socket_runtime_move(c,scratch,dst));break;
   case OP_OWN_STORE_LOCAL:RT(nvm_socket_runtime_frame_store(c));break;
   case OP_FILE_RESULT_BRANCH:
    CHECK(nvm_socket_runtime_frame_local(c,in.decoded.operands[0].u16,&src));RT(nvm_socket_runtime_result_arm(c,src,&arm));CHECK(arm==NVM_SOCKET_FLOW_ARM_ERROR);edge=1;break;
   case OP_FILE_RESULT_TAKE:
    CHECK(nvm_socket_runtime_frame_local(c,in.decoded.operands[0].u16,&src));CHECK(nvm_socket_runtime_frame_reserve(c,(uint16_t)f.stack_count,&dst));
    CHECK(in.decoded.operands[1].u8==1);RT(nvm_socket_runtime_take(c,src,NVM_SOCKET_FLOW_ARM_ERROR,dst));CHECK(view(c,dst).fields==11);break;
   case OP_POP:
    CHECK(f.stack_count && nvm_socket_runtime_frame_operand(c,(uint16_t)(f.stack_count-1),&src));RT(nvm_socket_runtime_drop(c,src));break;
   case OP_RET:RT(nvm_socket_runtime_frame_return(c));completed=true;break;
   default:CHECK(false);
   }
   if(completed){steps++;break;}
   RT(nvm_socket_runtime_frame_next(c,edge));
  }
  CHECK(completed && acquisition_attempts==before);NvmSocketRuntimeView out={.fields=99};
  if(indirect){NvmSocketIndirectExecutionReport r=nvm_socket_runtime_indirect_destroy(&c,&out);CHECK(r.runtime.status==NVM_SOCKET_RUNTIME_OK && r.instructions_started==steps && !r.runtime.cleanup.cleanup_failures);}
  else {NvmSocketCyclicExecutionReport r=nvm_socket_runtime_cyclic_destroy(&c,&out);CHECK(r.runtime.status==NVM_SOCKET_RUNTIME_OK && r.instructions_started==steps && !r.runtime.cleanup.cleanup_failures);}
  CHECK(!c && out.fields==1 && !out.values[0]);
 }
}
int main(void){NvmSocketNominalBindings b;size_t size;uint8_t *bytes=runtime_bytes(&b,&size);preparation(bytes,size);checked_frames(bytes,size);invalid_endpoint(bytes,size,b);
 for(unsigned mode=0;mode<2;mode++)for(unsigned family=4;family<=6;family+=2)for(unsigned abandon=0;abandon<4;abandon++)runtime_lifecycle(bytes,size,b,family,(NvmSocketRuntimeMode)mode,abandon);
 free(bytes);CHECK(!allocation_live);printf("PASS %u TCP carrier, real IPv4/IPv6 lifecycle and cleanup checks; no matched CODE dispatch\n",checks);return 0;}
