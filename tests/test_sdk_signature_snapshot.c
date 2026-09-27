/* I preserve coarse table indices and owned bytes, without provider admission. */
#ifdef NDEBUG
#error "I require assertions in my SDK signature snapshot controls."
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
static int allocation_count, fail_at, persistent;

static int fail_now(void) {
    allocation_count++;
    return fail_at && (allocation_count==fail_at || (persistent&&allocation_count>=fail_at));
}
static void *snapshot_malloc(size_t n) { if(fail_now())return NULL;return malloc(n); }
static void *snapshot_calloc(size_t n,size_t w) { if(fail_now())return NULL;return calloc(n,w); }
#define malloc snapshot_malloc
#define calloc snapshot_calloc
#include "../src/nanoisa/sdk_signature_snapshot.c"
#undef malloc
#undef calloc
int main(void) {
    uint8_t *tags=malloc(6);assert(tags);
    tags[0]=TAG_OPAQUE;tags[1]=TAG_OPAQUE;tags[2]=TAG_ENUM;tags[3]=TAG_INT;tags[4]=TAG_BOOL;tags[5]=TAG_STRING;
    NvmV2Signature signatures[]={
        {1,1,tags,tags+1},{1,1,tags,tags+1},
        {1,1,tags+2,tags+3},{1,1,tags+4,tags+5}}; /* Only row3 is unused. */
    NvmV2Function functions[2]={{0},{0}};functions[0].signature_idx=1;functions[1].signature_idx=0;
    NvmV2Import imports[1]={{0}};imports[0].signature_idx=1;
    NvmV2Callback callbacks[1]={{0}};callbacks[0].signature_idx=NVM_V2_NO_INDEX;
    NvmV2Link links[1]={{0}};links[0].signature_idx=2;
    NvmV2Module m={0};m.signatures=(NvmV2Signatures){signatures,4};
    m.functions=(NvmV2Functions){functions,2};m.imports=(NvmV2Imports){imports,1};
    m.callbacks=(NvmV2Callbacks){callbacks,1};m.links=(NvmV2Links){links,1};
    NvmSdkSignatureSnapshot *p=NULL;allocation_count=0;
    assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
    int sites=allocation_count;assert(sites==7);size_t bytes=nvm_sdk_signature_snapshot_bytes(p);
    nvm_sdk_signature_snapshot_free(p);
    for(int mode=0;mode<2;mode++)for(int at=1;at<=sites;at++) {
        allocation_count=0;fail_at=at;persistent=mode;p=(NvmSdkSignatureSnapshot *)(uintptr_t)1;
        assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_MEMORY);
        assert(p==(NvmSdkSignatureSnapshot *)(uintptr_t)1);
        fail_at=0;allocation_count=0;p=NULL;
        assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
        assert(allocation_count==sites);nvm_sdk_signature_snapshot_free(p);
    }
    p=(NvmSdkSignatureSnapshot *)(uintptr_t)1;allocation_count=0;
    assert(nvm_sdk_signature_snapshot_prepare(&m,bytes-1,&p)==NVM_SDK_LIMIT);
    assert(p==(NvmSdkSignatureSnapshot *)(uintptr_t)1&&!allocation_count);
    p=NULL;assert(nvm_sdk_signature_snapshot_prepare(&m,bytes,&p)==NVM_SDK_OK);
    nvm_sdk_signature_snapshot_free(p);
    functions[0].signature_idx=NVM_V2_NO_INDEX;allocation_count=0;p=NULL;
    assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_INVALID&&!p&&!allocation_count);
    functions[0].signature_idx=1;
    tags[2]=TAG_COUNT;
    assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_INVALID&&!p&&!allocation_count);
    tags[2]=TAG_ENUM;
    uint32_t old_count=m.functions.count;m.functions.count=UINT32_MAX;
    assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_LIMIT&&!p&&!allocation_count);
    m.functions.count=old_count;
    assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
    memset(tags,0,6);free(tags);memset(signatures,0,sizeof signatures);memset(functions,0,sizeof functions);
    memset(imports,0,sizeof imports);memset(callbacks,0,sizeof callbacks);memset(links,0,sizeof links);
    const NvmV2Signatures *copy=nvm_sdk_signature_snapshot_rows(p);assert(copy&&copy->count==4);
    assert(copy->items[0].param_tags[0]==TAG_OPAQUE&&copy->items[1].param_tags[0]==TAG_OPAQUE);
    assert(copy->items[2].param_tags[0]==TAG_ENUM&&copy->items[2].result_tags[0]==TAG_INT);
    assert(copy->items[3].param_tags[0]==TAG_BOOL&&copy->items[3].result_tags[0]==TAG_STRING);
    uint32_t index=98;assert(nvm_sdk_signature_snapshot_index(p,NVM_SDK_SIGNATURE_FUNCTION,0,&index)&&index==1);
    assert(nvm_sdk_signature_snapshot_index(p,NVM_SDK_SIGNATURE_FUNCTION,1,&index)&&index==0);
    index=98;
    assert(!nvm_sdk_signature_snapshot_index(p,NVM_SDK_SIGNATURE_FUNCTION,2,&index)&&index==98);
    assert(nvm_sdk_signature_snapshot_index(p,NVM_SDK_SIGNATURE_IMPORT,0,&index)&&index==1);
    assert(nvm_sdk_signature_snapshot_index(p,NVM_SDK_SIGNATURE_CALLBACK,0,&index)&&index==NVM_V2_NO_INDEX);
    assert(nvm_sdk_signature_snapshot_index(p,NVM_SDK_SIGNATURE_LINK,0,&index)&&index==2);
    assert(!nvm_sdk_signature_snapshot_index(p,4,0,&index)&&index==2);
    nvm_sdk_signature_snapshot_free(p);
    m=(NvmV2Module){0};p=NULL;
    assert(nvm_sdk_signature_snapshot_prepare(&m,NVM_SDK_GENERATION_MAX_BYTES,&p)==NVM_SDK_OK);
    assert(nvm_sdk_signature_snapshot_rows(p)->count==0);nvm_sdk_signature_snapshot_free(p);
    puts("I preserved canonical signature indices and complete independent owned snapshots.");return 0;
}
