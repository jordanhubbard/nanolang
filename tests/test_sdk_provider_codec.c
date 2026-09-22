/* I exercise raw transport ownership, not module or execution admission. */
#ifdef NDEBUG
#error "I require assertions in my raw SDK codec controls."
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
static int fail_allocation, allocation_calls;
static void *codec_malloc(size_t n) {
    allocation_calls++;
    if(fail_allocation)return NULL;
    return malloc(n);
}
static void *codec_calloc(size_t n,size_t w) {
    allocation_calls++;
    if(fail_allocation)return NULL;
    return calloc(n,w);
}
#define malloc codec_malloc
#define calloc codec_calloc
#include "../src/nanoisa/sdk_provider_codec.c"
#undef malloc
#undef calloc
static void refuse(const uint8_t *data,size_t bytes,NvmSdkResult expected) {
    NvmSdkProviderTransport *sentinel=(NvmSdkProviderTransport *)(uintptr_t)1;
    int before=allocation_calls;
    assert(nvm_sdk_provider_decode(data,bytes,NVM_SDK_PROVIDER_MAX_BYTES,&sentinel)==expected);
    assert(sentinel==(NvmSdkProviderTransport *)(uintptr_t)1);
    assert(allocation_calls==before);
}
int main(void) {
    NvmSdkNominalRow nominal[]={
        {10,20,NVM_SDK_NOMINAL_RECORD,0,0,1},
        {11,20,NVM_SDK_NOMINAL_RECORD,1,1,1},
        {10,21,NVM_SDK_NOMINAL_OPAQUE,UINT32_MAX,2,0},
        {10,22,NVM_SDK_NOMINAL_ENUM,2,2,0}};
    NvmSdkProviderRow provider[]={{30,31,32,33,34,35}};
    NvmSdkSignatureRow signature[]={{2,0,2,2,1}};
    NvmSdkBindingRow binding[]={
        {NVM_SDK_BIND_IMPORT,0,UINT32_MAX,0,0},
        {NVM_SDK_BIND_FUNCTION,4,UINT32_MAX,0,UINT32_MAX},
        {NVM_SDK_BIND_FIELD,0,1,3,UINT32_MAX}};
    uint32_t refs[]={0,1,2};
    NvmSdkProviderRows rows={nominal,provider,signature,binding,refs,4,1,1,3,3};
    uint8_t *data=NULL;size_t size=0;
    assert(nvm_sdk_provider_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&data,&size)==NVM_SDK_OK);
    assert(size==32+4*32+32+24+3*24+3*4);
    assert(get32(data)==1 && get32(data+8)==4 && get32(data+24)==3);
    NvmSdkProviderTransport *plan=NULL;
    assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_OK);
    /* Borrowed row mutation and released encoded storage cannot alter my copy. */
    memset(nominal,0,sizeof nominal);memset(data,0,size);free(data);data=NULL;
    uint32_t counts[5];assert(nvm_sdk_provider_counts(plan,counts));assert(counts[0]==4&&counts[3]==3);
    NvmSdkNominalRow a,b;assert(nvm_sdk_provider_nominal(plan,0,&a));assert(nvm_sdk_provider_nominal(plan,1,&b));
    assert(a.owner==10&&b.owner==11&&a.name==b.name&&a.layout!=b.layout);
    assert(nvm_sdk_provider_nominal(plan,3,&a)&&a.kind==NVM_SDK_NOMINAL_ENUM&&a.layout==2);
    a.owner=987;assert(!nvm_sdk_provider_nominal(plan,4,&a)&&a.owner==987);
    NvmSdkProviderRow p;assert(nvm_sdk_provider_requirement(plan,0,&p)&&p.generation_digest==34);
    NvmSdkSignatureRow s;assert(nvm_sdk_provider_signature(plan,0,&s)&&s.parameter_count==2&&s.result_count==1);
    NvmSdkBindingRow bind;assert(nvm_sdk_provider_binding(plan,2,&bind)&&bind.slot==1&&bind.detail==3);
    uint32_t ref=99;assert(nvm_sdk_provider_reference(plan,2,&ref)&&ref==2);
    assert(!nvm_sdk_provider_reference(plan,3,&ref)&&ref==2);
    /* Reconstruct the logical rows through public by-value accessors. */
    for(uint32_t i=0;i<4;i++)assert(nvm_sdk_provider_nominal(plan,i,&nominal[i]));
    assert(nvm_sdk_provider_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&data,&size)==NVM_SDK_OK);
    assert(size==plan->size&&!memcmp(data,plan->data,size));
    nvm_sdk_provider_transport_free(plan);plan=NULL;
    for(size_t n=0;n<size;n++)refuse(data,n,NVM_SDK_INVALID);
    uint8_t *bad=malloc(size+1);assert(bad);memcpy(bad,data,size);bad[size]=0;
    refuse(bad,size+1,NVM_SDK_INVALID);
    const size_t reserved[]={4,28,32+24,32+28,32+4*32+24,32+4*32+32+20,32+4*32+32+24+20};
    for(size_t i=0;i<sizeof reserved/sizeof *reserved;i++) {
        memcpy(bad,data,size);bad[reserved[i]]=1;refuse(bad,size,NVM_SDK_INVALID);
    }
    memcpy(bad,data,size);put32(bad+8,4097);refuse(bad,size,NVM_SDK_LIMIT);
    memcpy(bad,data,size);put32(bad+32+16,UINT32_MAX);refuse(bad,size,NVM_SDK_INVALID);
    free(bad);
    uint8_t *old=(uint8_t *)(uintptr_t)1;size_t old_size=777;
    assert(nvm_sdk_provider_encode(&rows,0,&old,&old_size)==NVM_SDK_LIMIT);
    assert(old==(uint8_t *)(uintptr_t)1&&old_size==777);
    plan=(NvmSdkProviderTransport *)(uintptr_t)1;
    assert(nvm_sdk_provider_decode(data,size,sizeof(NvmSdkProviderTransport)+size-1,&plan)==NVM_SDK_LIMIT);
    assert(plan==(NvmSdkProviderTransport *)(uintptr_t)1);
    plan=NULL;
    assert(nvm_sdk_provider_decode(data,size,sizeof(NvmSdkProviderTransport)+size,&plan)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(plan);
    fail_allocation=1;
    assert(nvm_sdk_provider_encode(&rows,NVM_SDK_PROVIDER_MAX_BYTES,&old,&old_size)==NVM_SDK_MEMORY);
    assert(old==(uint8_t *)(uintptr_t)1&&old_size==777);
    plan=(NvmSdkProviderTransport *)(uintptr_t)1;
    assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_MEMORY);
    assert(plan==(NvmSdkProviderTransport *)(uintptr_t)1);
    fail_allocation=0;plan=NULL;
    assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(plan);free(data);
    NvmSdkProviderRows empty={0};data=NULL;size=0;
    assert(nvm_sdk_provider_encode(&empty,NVM_SDK_PROVIDER_MAX_BYTES,&data,&size)==NVM_SDK_OK&&size==32);
    plan=NULL;assert(nvm_sdk_provider_decode(data,size,NVM_SDK_PROVIDER_MAX_BYTES,&plan)==NVM_SDK_OK);
    nvm_sdk_provider_transport_free(plan);free(data);
    puts("I passed raw SDK transport ownership, refusal and allocation controls.");return 0;
}
