#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static unsigned fail_at,calls;
static void *debug_realloc(void *p,size_t n) { if(fail_at && ++calls==fail_at)return NULL;return realloc(p,n); }
#define realloc debug_realloc
#include "../../src/nanoisa/nvm_format.c"
#include "../../src/nanoisa/nvm_v2_convert.c"
#undef realloc
int main(void) {
    NvmModule *m=nvm_module_new();assert(m);
    unsigned count=m->debug_capacity+1;
    for(unsigned i=0;i<count-1;i++)assert(nvm_add_debug_entry(m,i,i+1,0));
    calls=0;fail_at=1;assert(!nvm_add_debug_entry(m,0,0,0));fail_at=0;
    assert(m->debug_count==count-1 && m->debug_entries[count-2].source_line==count-1);
    assert(nvm_add_debug_entry(m,0,0,0));
    NvmV2Module v;assert(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
    calls=0;fail_at=1;NvmModule *out=(void *)1;
    assert(nvm_v2_to_nvm_module(&v,&out)!=NVM_V2_OK && out==NULL);fail_at=0;
    assert(nvm_v2_to_nvm_module(&v,&out)==NVM_V2_OK && out->debug_count==count);
    nvm_module_free(out);nvm_v2_module_free(&v);
    uint32_t length;uint8_t *wire=nvm_serialize(m,&length);assert(wire);
    calls=0;fail_at=1;out=nvm_deserialize(wire,length);assert(!out);fail_at=0;
    out=nvm_deserialize(wire,length);assert(out && out->debug_count==count);
    nvm_module_free(out);free(wire);nvm_module_free(m);
    puts("I passed DEBUG append, v1 and v2 allocation failure cleanup.");return 0;
}
