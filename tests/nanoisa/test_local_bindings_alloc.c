#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static unsigned fail_at,calls;
static void *names_malloc(size_t n) { if(fail_at && ++calls==fail_at)return NULL;return malloc(n); }
static void *names_calloc(size_t n,size_t z) { if(fail_at && ++calls==fail_at)return NULL;return calloc(n,z); }
#define malloc names_malloc
#define calloc names_calloc
#include "../../src/nanoisa/nvm_format.c"
#include "../../src/nanoisa/local_bindings.c"
#undef malloc
#undef calloc
int main(void) {
    unsigned checks=0;
    for(unsigned attempt=1;attempt<30;attempt++) {
        NvmModule *m=nvm_module_new();assert(m);
        uint32_t name=nvm_add_string(m,"main",4);
        NvmFunctionEntry f={.name_idx=name,.local_count=1,.code_length=1,.result_tag=TAG_VOID};
        assert(nvm_add_function(m,&f)==0);
        uint8_t ret=OP_RET;assert(nvm_append_code(m,&ret,1)==0);
        NvmLocalBinding b={.function=0,.slot=0,.begin=0,.end=1,.name=(const uint8_t *)"x",.name_size=1};
        calls=0;fail_at=attempt;bool ok=nvm_add_local_binding(m,&b);fail_at=0;++checks;
        if(ok){assert(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_VALID);nvm_module_free(m);break;}
        assert(m->metadata_count==0 && m->code_size==1 && m->code[0]==OP_RET);
        assert(nvm_add_local_binding(m,&b));
        assert(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_VALID);nvm_module_free(m);
    }
    printf("I passed %u local-name allocation boundaries.\n",checks);return 0;
}
