#include "sdk_signature_snapshot.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>
struct NvmSdkSignatureSnapshot {
    NvmV2Signatures signatures;
    uint32_t counts[4];
    uint32_t *indices[4];
    uint8_t *tags;
    size_t bytes;
};
static bool add_bytes(size_t *used,size_t count,size_t width,size_t limit) {
    if(*used>limit || (width && count>(limit-*used)/width))return false;
    *used+=count*width;return true;
}
static bool charge(uint32_t *work,uint32_t amount) {
    if(amount>NVM_SDK_GENERATION_MAX_WORK-*work)return false;
    *work+=amount;return true;
}
static uint32_t selected(const NvmV2Module *m,unsigned kind,uint32_t i) {
    switch(kind) {
        case NVM_SDK_SIGNATURE_FUNCTION:return m->functions.items[i].signature_idx;
        case NVM_SDK_SIGNATURE_IMPORT:return m->imports.items[i].signature_idx;
        case NVM_SDK_SIGNATURE_CALLBACK:return m->callbacks.items[i].signature_idx;
        default:return m->links.items[i].signature_idx;
    }
}
void nvm_sdk_signature_snapshot_free(NvmSdkSignatureSnapshot *p) {
    if(!p)return;
    free(p->signatures.items);free(p->tags);
    for(unsigned k=0;k<4;k++)free(p->indices[k]);
    free(p);
}
NvmSdkResult nvm_sdk_signature_snapshot_prepare(const NvmV2Module *m,
    size_t limit,NvmSdkSignatureSnapshot **out) {
    if(!m||!out||(m->signatures.count&&!m->signatures.items)||
       (m->functions.count&&!m->functions.items)||(m->imports.count&&!m->imports.items)||
       (m->callbacks.count&&!m->callbacks.items)||(m->links.count&&!m->links.items))return NVM_SDK_INVALID;
    if(limit>NVM_SDK_GENERATION_MAX_BYTES)limit=NVM_SDK_GENERATION_MAX_BYTES;
    uint32_t counts[]={m->functions.count,m->imports.count,m->callbacks.count,m->links.count};
    uint32_t work=0;size_t bytes=sizeof(NvmSdkSignatureSnapshot),tags=0;
    if(!add_bytes(&bytes,m->signatures.count,sizeof(NvmV2Signature),limit)||
       !charge(&work,m->signatures.count)||!charge(&work,m->signatures.count))return NVM_SDK_LIMIT;
    for(unsigned k=0;k<4;k++) {
        if(!add_bytes(&bytes,counts[k],sizeof(uint32_t),limit)||!charge(&work,counts[k])||!charge(&work,counts[k]))return NVM_SDK_LIMIT;
        for(uint32_t i=0;i<counts[k];i++) {
            uint32_t index=selected(m,k,i);
            if(index>=m->signatures.count && !(k>=NVM_SDK_SIGNATURE_CALLBACK&&index==NVM_V2_NO_INDEX))return NVM_SDK_INVALID;
        }
    }
    for(uint32_t i=0;i<m->signatures.count;i++) {
        const NvmV2Signature *s=&m->signatures.items[i];
        uint32_t n=(uint32_t)s->param_count+s->result_count;
        if(!charge(&work,n)||!charge(&work,n)||!add_bytes(&bytes,n,1,limit))return NVM_SDK_LIMIT;
        if((s->param_count&&!s->param_tags)||(s->result_count&&!s->result_tags))return NVM_SDK_INVALID;
        for(uint16_t j=0;j<s->param_count;j++)if(s->param_tags[j]>=TAG_COUNT)return NVM_SDK_INVALID;
        for(uint16_t j=0;j<s->result_count;j++)if(s->result_tags[j]>=TAG_COUNT)return NVM_SDK_INVALID;
        tags+=n;
    }
    NvmSdkSignatureSnapshot *p=calloc(1,sizeof *p);if(!p)return NVM_SDK_MEMORY;
    p->bytes=bytes;p->signatures.count=m->signatures.count;memcpy(p->counts,counts,sizeof counts);
    if(m->signatures.count) {
        p->signatures.items=calloc(m->signatures.count,sizeof *p->signatures.items);
        if(!p->signatures.items)goto memory;
    }
    if(tags){p->tags=malloc(tags);if(!p->tags)goto memory;}
    for(unsigned k=0;k<4;k++)if(counts[k]) {
        p->indices[k]=malloc((size_t)counts[k]*sizeof *p->indices[k]);
        if(!p->indices[k])goto memory;
        for(uint32_t i=0;i<counts[k];i++)p->indices[k][i]=selected(m,k,i);
    }
    size_t at=0;
    for(uint32_t i=0;i<m->signatures.count;i++) {
        const NvmV2Signature *s=&m->signatures.items[i];NvmV2Signature *d=&p->signatures.items[i];
        d->param_count=s->param_count;d->result_count=s->result_count;
        if(s->param_count){memcpy(p->tags+at,s->param_tags,s->param_count);d->param_tags=p->tags+at;at+=s->param_count;}
        if(s->result_count){memcpy(p->tags+at,s->result_tags,s->result_count);d->result_tags=p->tags+at;at+=s->result_count;}
    }
    *out=p;return NVM_SDK_OK;
memory:
    nvm_sdk_signature_snapshot_free(p);return NVM_SDK_MEMORY;
}
const NvmV2Signatures *nvm_sdk_signature_snapshot_rows(const NvmSdkSignatureSnapshot *p) {
    return p?&p->signatures:NULL;
}
size_t nvm_sdk_signature_snapshot_bytes(const NvmSdkSignatureSnapshot *p) {
    return p?p->bytes:0;
}
bool nvm_sdk_signature_snapshot_index(const NvmSdkSignatureSnapshot *p,
    unsigned kind,uint32_t subject,uint32_t *out) {
    if(!p||!out||kind>=4||subject>=p->counts[kind])return false;
    *out=p->indices[kind][subject];return true;
}
