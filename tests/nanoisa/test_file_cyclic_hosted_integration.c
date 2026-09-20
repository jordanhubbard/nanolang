/* I use source-private query/fixture headers with the actual built public
 * archive. This is not an installed cyclic API or a cyclic execution test. */
#define FILE_HOSTED_MAIN prior_hosted_fixture_for_archive
#include "test_file_hosted.c"
#undef FILE_HOSTED_MAIN
#include "../../src/nanoisa/file_cyclic_hosted.h"
#include "../../src/nanoisa/file_public.h"
#include "../../src/nanoisa/file_native_public.h"
#include "../../src/nanoisa/file_host_grant_internal.h"

int main(void){
 NvmFileNominalBindings b;NvmModule *m=bodymodule(&b,false);Body c={0};
 op(&c,OP_PUSH_BOOL);op(&c,1);uint32_t leave=branch(&c,OP_JMP_FALSE,0);
 uint32_t backedge=branch(&c,OP_JMP,0);wr32(c.bytes+backedge+1,(uint32_t)(-(int32_t)backedge));
 target(&c,leave);retint(&c);setbody(m,0,c);size_t size;uint8_t *bytes=serialize(m,&size);
 NvmFileCyclicHostedPlan *query=NULL;OK(nvm_file_cyclic_hosted_prepare(bytes,size,&query));
 NvmFileCyclicHostedStartup facts;CHECK(nvm_file_cyclic_hosted_startup(query,&facts) && !facts.runtime_admitted);
 NvmFileCyclicHostedFunction f;CHECK(nvm_file_cyclic_hosted_function(query,0,&f));bool retained_backedge=false;
 for(uint16_t i=0;i<f.code.instruction_count;i++){
  NvmFileCodeInstruction in;CHECK(nvm_file_cyclic_hosted_instruction(query,0,i,&in));
  if(in.decoded.opcode==OP_JMP){CHECK(in.successor_count==1 && in.successors[0]==0);retained_backedge=true;}
 }
 CHECK(retained_backedge);expect_hosted(bytes,size,NVM_FILE_FLOW_UNRESOLVED);
 NvmFileHostGrant *grant=NULL;CHECK(nvm_file_host_grant_create_temporary_files(&grant)==NVM_FILE_HOST_OK);
 NvmFileScalar out;memset(&out,0xa5,sizeof out);NvmFileScalar before=out;
 NvmFileRuntimeReport report=nvm_file_execute_bytes(grant,bytes,size,&out);
 CHECK(report.status==NVM_FILE_RUNTIME_UNRESOLVED && !report.acquired && !memcmp(&out,&before,sizeof out));
 char *text=(char *)(uintptr_t)1;char diagnostic[256]={0};
 CHECK(nvm2c_emit_file_bytes(bytes,size,"cyclic_refused",&text,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_UNRESOLVED);
 CHECK(text==(char *)(uintptr_t)1);
 CHECK(nvm_file_host_enter_query()==NVM_FILE_HOST_OK);nvm_file_host_leave();
 CHECK(nvm_file_cyclic_hosted_startup(query,&facts) && !facts.runtime_admitted);
 nvm_file_cyclic_hosted_free(query);free(bytes);
 /* The same grant remains usable after both checked refusals. Only this
  * acyclic scalar entry executes; it contains no File service instruction. */
 c=(Body){0};retint(&c);setbody(m,0,c);bytes=serialize(m,&size);
 report=nvm_file_execute_bytes(grant,bytes,size,&out);
 CHECK(report.status==NVM_FILE_RUNTIME_OK && report.acquired && !report.cleanup.cleanup_failures && out.tag==TAG_INT && out.value==0);
 text=(char *)(uintptr_t)1;
 CHECK(nvm2c_emit_file_bytes(bytes,size,"acyclic_ok",&text,diagnostic,sizeof diagnostic)==NVM_FILE_RUNTIME_OK);
 CHECK(text && text!=(char *)(uintptr_t)1 && strstr(text,"nvm_file_program_acyclic_ok"));free(text);
 CHECK(nvm_file_host_grant_revoke(grant)==NVM_FILE_HOST_OK);out=before;
 report=nvm_file_execute_bytes(grant,bytes,size,&out);
 CHECK(report.status==NVM_FILE_RUNTIME_STATE && !report.acquired && !memcmp(&out,&before,sizeof out));
 CHECK(nvm_file_host_grant_destroy(&grant)==NVM_FILE_HOST_OK && !grant);
 free(bytes);nvm_module_free(m);
 printf("PASS %u linked archive cyclic refusal and acyclic scalar controls; no cyclic execution\n",checks);return 0;
}
