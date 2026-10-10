/* I run the same checked mixed graph through public VM and generated native C. */
#define MIXED_WEBSOCKET_FLOW_MAIN mixed_flow_fixture_main
#include "test_mixed_websocket_flow.c"
#include "mixed_websocket_runtime_host.h"
static void mixed_write(const char *path,const void *data,size_t size){FILE *f=fopen(path,"wb");CHECK(f && fwrite(data,1,size,f)==size && !fclose(f));}
static unsigned char *mixed_read(const char *path,size_t *size){
 FILE *f=fopen(path,"rb");CHECK(f && !fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>0 && !fseek(f,0,SEEK_SET));
 unsigned char *bytes=malloc((size_t)n);CHECK(bytes && fread(bytes,1,(size_t)n,f)==(size_t)n && !fclose(f));*size=(size_t)n;return bytes;
}
int main(int argc,char **argv){
 CHECK(argc==6);
 if(!strcmp(argv[1],"emit")){
  unsigned graph=(unsigned)strtoul(argv[4],NULL,10),shape=graph%3;unsigned catalogs[5],count=mixed_catalogs(shape,catalogs);
  unsigned live_port=(unsigned)strtoul(argv[5],NULL,10);
  ws_fixture_reply=live_port>2?"a\0b":"r\0b";
  ws_fixture_require_success=live_port>2;ws_fixture_timeout=live_port>2?1000:0;
  NvmMultiNominalBindings b;NvmModule *m=ws_program(catalogs,count,graph/3%2,graph/6%2,graph/12%2,atoi(argv[5])==2?6:0,&b);
  if(atoi(argv[5])==1)for(unsigned i=0;i<m->string_count;i++)if(m->string_lengths[i]==19 && !memcmp(m->strings[i],"ws://127.0.0.1/test",19))m->strings[i][0]='!';
  if(live_port>2)for(unsigned i=0;i<m->string_count;i++)if(m->string_lengths[i]==19 && !memcmp(m->strings[i],"ws://127.0.0.1/test",19)){
   char url[128];snprintf(url,sizeof url,"ws://127.0.0.1:%u/?mode=test",live_port);
   free(m->strings[i]);m->strings[i]=strdup(url);CHECK(m->strings[i]);m->string_lengths[i]=(uint32_t)strlen(url);
  }
  if(live_port>2)for(unsigned f=0;f<m->function_count;f++){
   unsigned at=m->functions[f].code_offset,end=at+m->functions[f].code_length,instructions=0;
   while(at<end){DecodedInstruction in;unsigned n=isa_decode(m->code+at,end-at,&in);CHECK(n);at+=n;instructions++;}
   printf("I count %u instructions in live function %u.\n",instructions,f);
  }
  size_t size=0;uint8_t *bytes=ws_wire(m,&size);char *native=NULL,why[256];
  NvmServicesRuntimeStatus status=nvm2c_emit_services_indirect_bytes(bytes,size,"mixed_test",&native,why,sizeof why);
  if(status)fprintf(stderr,"native status=%u: %s\n",status,why);
  CHECK(status==NVM_SERVICES_RUNTIME_OK);
  mixed_write(argv[2],bytes,size);mixed_write(argv[3],native,strlen(native));free(native);free(bytes);nvm_module_free(m);return 0;
 }
 CHECK(!strcmp(argv[1],"vm"));size_t size=0;unsigned char *bytes=mixed_read(argv[2],&size);
 mixed_mode=(unsigned)strtoul(argv[4],NULL,10);NvmServicesHostGrant *g=mixed_grant((unsigned)strtoul(argv[3],NULL,10),mixed_mode);
 NvmServicesIndirectOptions options={1,strtoull(argv[5],NULL,10)};NvmServicesScalar value={TAG_INT,12345};
 mixed_budget_start();
 NvmServicesIndirectExecutionReport r=nvm_services_execute_indirect_bytes(g,bytes,size,&options,&value);
 CHECK(nvm_services_host_grant_destroy(&g)==NVM_SERVICES_HOST_OK && !g);free(bytes);mixed_report(r,value);return 0;
}
