#include "local_bindings.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>
static const char key[]="nano.local.v1";
static uint32_t read_word(const uint8_t *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static void write_word(uint8_t *p,uint32_t v) {
    for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));
}
static bool named(const NvmModule *m,const NvmMetadataEntry *e) {
    return e->key_idx<m->string_count && m->string_lengths[e->key_idx]==sizeof key-1 &&
        !memcmp(m->strings[e->key_idx],key,sizeof key-1);
}
static bool boundary(const NvmModule *m,uint32_t fn,uint32_t pc) {
    const NvmFunctionEntry *f=&m->functions[fn];
    if(f->code_offset>m->code_size || f->code_length>m->code_size-f->code_offset ||
       pc>f->code_length || (f->code_length && !m->code))return false;
    uint32_t at=0;
    while(at<pc) {
        DecodedInstruction instruction;
        uint32_t n=isa_decode(m->code+f->code_offset+at,f->code_length-at,&instruction);
        if(!n || n>pc-at)return false;
        at+=n;
    }
    return at==pc;
}
static bool valid(const NvmModule *m,const NvmLocalBinding *b) {
    return m && (!m->function_count || m->functions) && b->name && b->name_size && b->name_size<=UINT32_MAX-16 &&
        b->function<m->function_count && b->slot<m->functions[b->function].local_count &&
        b->begin<=b->end && boundary(m,b->function,b->begin) && boundary(m,b->function,b->end);
}
static bool decode(const NvmModule *m,const NvmMetadataEntry *e,NvmLocalBinding *b) {
    if(e->value_idx>=m->string_count || m->string_lengths[e->value_idx]<=16)return false;
    const uint8_t *p=(const uint8_t *)m->strings[e->value_idx];
    if(p[6] || p[7])return false;
    *b=(NvmLocalBinding){.function=read_word(p),.slot=(uint16_t)(p[4]|((uint16_t)p[5]<<8)),
        .begin=read_word(p+8),.end=read_word(p+12),.name=p+16,.name_size=m->string_lengths[e->value_idx]-16};
    return true;
}
static bool overlap(const NvmLocalBinding *a,const NvmLocalBinding *b) {
    return a->function==b->function && a->slot==b->slot && a->begin<a->end && b->begin<b->end &&
        a->begin<b->end && b->begin<a->end;
}
NvmLocalNamesStatus nvm_local_names_validate(const NvmModule *m) {
    if(!m || (m->function_count && !m->functions) || !nvm_metadata_valid(m))return NVM_LOCAL_NAMES_INVALID;
    bool found=false;
    for(uint32_t i=0;i<m->metadata_count;i++) {
        if(!named(m,&m->metadata[i]))continue;
        found=true; NvmLocalBinding a;
        if(!decode(m,&m->metadata[i],&a) || !valid(m,&a))return NVM_LOCAL_NAMES_INVALID;
        for(uint32_t j=0;j<i;j++) {
            if(!named(m,&m->metadata[j]))continue;
            NvmLocalBinding b;
            if(!decode(m,&m->metadata[j],&b) || overlap(&a,&b))return NVM_LOCAL_NAMES_INVALID;
        }
    }
    return found?NVM_LOCAL_NAMES_VALID:NVM_LOCAL_NAMES_ABSENT;
}
NvmLocalNamesStatus nvm_local_name_at(const NvmModule *m,uint32_t fn,uint16_t slot,
                                     uint32_t pc,NvmLocalBinding *out) {
    NvmLocalNamesStatus status=nvm_local_names_validate(m);
    if(status!=NVM_LOCAL_NAMES_VALID)return status;
    for(uint32_t i=0;i<m->metadata_count;i++) {
        NvmLocalBinding b;
        if(named(m,&m->metadata[i]) && decode(m,&m->metadata[i],&b) &&
           b.function==fn && b.slot==slot && b.begin<=pc && pc<b.end) {
            if(out)*out=b;
            return NVM_LOCAL_NAMES_VALID;
        }
    }
    return NVM_LOCAL_NAMES_ABSENT;
}
bool nvm_add_local_binding(NvmModule *m,const NvmLocalBinding *b) {
    if(!m || !b || !nvm_metadata_valid(m) || !valid(m,b))return false;
    for(uint32_t i=0;i<m->metadata_count;i++) {
        if(!named(m,&m->metadata[i]))continue;
        NvmLocalBinding prior;
        if(!decode(m,&m->metadata[i],&prior) || overlap(b,&prior))return false;
    }
    uint32_t size=b->name_size+16;
    uint8_t *value=malloc(size); if(!value)return false;
    write_word(value,b->function);value[4]=(uint8_t)b->slot;value[5]=(uint8_t)(b->slot>>8);
    value[6]=value[7]=0;write_word(value+8,b->begin);write_word(value+12,b->end);
    memcpy(value+16,b->name,b->name_size);
    uint32_t k=nvm_add_string(m,key,sizeof key-1),v=nvm_add_string(m,(const char *)value,size);
    free(value);
    return k!=UINT32_MAX && v!=UINT32_MAX && nvm_add_metadata(m,k,v);
}
