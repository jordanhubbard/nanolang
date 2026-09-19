#include "../../src/nanoisa/service_bindings.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do{checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
static const uint8_t golden[56]={
 1,0,1,0,5,0,0,0,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,
 1,0,0,0,4,3,2,1,
 2,0,0,0,254,255,255,255,
 3,0,0,0,0,0,0,128,
 4,0,0,0,7,0,0,0
};
static void refused(const uint8_t *bytes,size_t length,NvmServiceResult expected){
 NvmServiceBindings out={{11,12,13,14,15}},saved=out;
 CHECK(nvm_service_bindings_decode(bytes,length,&out)==expected);
 CHECK(!memcmp(&out,&saved,sizeof out));
}
static void encode_refused(const NvmServiceBindings *value,size_t cap,NvmServiceResult expected){
 uint8_t out[64],saved[64];memset(out,0xa5,sizeof out);memcpy(saved,out,sizeof out);size_t length=99;
 CHECK(nvm_service_bindings_encode(value,out,cap,&length)==expected);
 CHECK(length==99);CHECK(!memcmp(out,saved,sizeof out));
}
int main(void){
 NvmServiceBindings value={{0,UINT32_C(0x01020304),UINT32_MAX-1,UINT32_C(0x80000000),7}},decoded;
 CHECK(nvm_service_bindings_check(&value)==NVM_SERVICE_OK);
 CHECK(nvm_service_bindings_decode(golden,sizeof golden,&decoded)==NVM_SERVICE_OK);
 CHECK(!memcmp(&value,&decoded,sizeof value));
 uint8_t output[64];memset(output,0xa5,sizeof output);size_t size=99;
 CHECK(nvm_service_bindings_encode(&value,output,sizeof output,&size)==NVM_SERVICE_OK);
 CHECK(size==56 && !memcmp(output,golden,56));for(size_t i=56;i<64;i++)CHECK(output[i]==0xa5);
 size=99;CHECK(nvm_service_bindings_encode(&value,NULL,0,&size)==NVM_SERVICE_OK && size==56);
 size=99;CHECK(nvm_service_bindings_encode(&value,NULL,SIZE_MAX,&size)==NVM_SERVICE_OK && size==56);
 for(size_t n=0;n<56;n++){refused(golden,n,NVM_SERVICE_SIZE);encode_refused(&value,n,NVM_SERVICE_SIZE);}
 refused(golden,57,NVM_SERVICE_SIZE);refused(golden,SIZE_MAX,NVM_SERVICE_SIZE);
 refused(NULL,56,NVM_SERVICE_ARGUMENT);CHECK(nvm_service_bindings_decode(golden,56,NULL)==NVM_SERVICE_ARGUMENT);
 CHECK(nvm_service_bindings_check(NULL)==NVM_SERVICE_ARGUMENT);encode_refused(NULL,56,NVM_SERVICE_ARGUMENT);
 memset(output,0xa5,sizeof output);CHECK(nvm_service_bindings_encode(&value,output,56,NULL)==NVM_SERVICE_ARGUMENT);
 for(size_t i=0;i<64;i++)CHECK(output[i]==0xa5);
 for(size_t i=0;i<56;i++){
  uint8_t bad[56];memcpy(bad,golden,56);bad[i]^=0x40;
  if(i<2)refused(bad,56,NVM_SERVICE_VERSION);
  else if(i<4)refused(bad,56,NVM_SERVICE_CATALOG);
  else if(i<8)refused(bad,56,NVM_SERVICE_COUNT);
  else if(i<16)refused(bad,56,NVM_SERVICE_RESERVED);
  else if((i-16)%8<4)refused(bad,56,NVM_SERVICE_ORDINAL);
  else {CHECK(nvm_service_bindings_decode(bad,56,&decoded)==NVM_SERVICE_OK);size=0;CHECK(nvm_service_bindings_encode(&decoded,output,56,&size)==NVM_SERVICE_OK);CHECK(!memcmp(output,bad,56));}
 }
 for(size_t i=0;i<5;i++){
  NvmServiceBindings bad=value;bad.imports[i]=UINT32_MAX;encode_refused(&bad,56,NVM_SERVICE_INDEX);
  uint8_t bytes[56];memcpy(bytes,golden,56);memset(bytes+20+8*i,255,4);refused(bytes,56,NVM_SERVICE_INDEX);
  for(size_t j=0;j<i;j++){bad=value;bad.imports[i]=bad.imports[j];encode_refused(&bad,56,NVM_SERVICE_INDEX);memcpy(bytes,golden,56);memcpy(bytes+20+8*i,bytes+20+8*j,4);refused(bytes,56,NVM_SERVICE_INDEX);}
 }
 {uint8_t swapped[56];memcpy(swapped,golden,56);swapped[0]=0;swapped[1]=1;refused(swapped,56,NVM_SERVICE_VERSION);}
 {union {NvmServiceBindings value;uint8_t bytes[56];} same;
  same.value=value;size=0;CHECK(nvm_service_bindings_encode(&same.value,same.bytes,56,&size)==NVM_SERVICE_OK);CHECK(!memcmp(same.bytes,golden,56));
  CHECK(nvm_service_bindings_decode(same.bytes,56,&same.value)==NVM_SERVICE_OK);CHECK(!memcmp(&same.value,&value,sizeof value));}
 {uint8_t local[56];memcpy(local,golden,56);CHECK(nvm_service_bindings_decode(local,56,&decoded)==NVM_SERVICE_OK);memset(local,0,56);CHECK(!memcmp(&decoded,&value,sizeof value));}
 printf("PASS %u raw service codec checks; no module or service admission\n",checks);return 0;
}
