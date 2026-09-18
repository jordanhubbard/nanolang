/* I exercise every metadata bridge allocation boundary and release partial state. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned fail_at, calls;
static void *metadata_malloc(size_t n) { if (fail_at && ++calls == fail_at) return NULL; return malloc(n); }
static void *metadata_calloc(size_t n,size_t z) { if (fail_at && ++calls == fail_at) return NULL; return calloc(n,z); }
#define malloc metadata_malloc
#define calloc metadata_calloc
#include "../../src/nanoisa/nvm_format.c"
#include "../../src/nanoisa/nvm_v2_convert.c"
#undef malloc
#undef calloc
int main(void) {
    NvmModule *m=nvm_module_new(); assert(m);
    uint32_t k=nvm_add_string(m,"example.fact",12),v=nvm_add_string(m,"a\0b",3);
    for(unsigned i=0;i<12;i++) assert(nvm_add_metadata(m,k,v));
    unsigned checks=0; bool succeeded=false;
    for(unsigned attempt=1;attempt<128;attempt++) {
        calls=0;fail_at=attempt;NvmV2Module out;
        NvmV2Result r=nvm_v2_from_nvm_module(m,&out);fail_at=0;++checks;
        assert(m->metadata_count==12 && m->metadata[11].value_idx==v);
        if(r==NVM_V2_OK) { assert(out.metadata.count==12);nvm_v2_module_free(&out);succeeded=true;break; }
        assert(r==NVM_V2_ERR_TRUNCATED);
    }
    assert(succeeded);NvmV2Module wire;assert(nvm_v2_from_nvm_module(m,&wire)==NVM_V2_OK);
    succeeded=false;
    for(unsigned attempt=1;attempt<128;attempt++) {
        calls=0;fail_at=attempt;NvmModule *out=(void *)1;
        NvmV2Result r=nvm_v2_to_nvm_module(&wire,&out);fail_at=0;++checks;
        if(r==NVM_V2_OK) { assert(out && out->metadata_count==12);nvm_module_free(out);succeeded=true;break; }
        assert(out==NULL);
    }
    assert(succeeded);nvm_v2_module_free(&wire);nvm_module_free(m);
    printf("I passed %u metadata conversion allocation boundaries.\n",checks);return 0;
}
