#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static unsigned fail_at,calls;
static void *marker_malloc(size_t n){if(fail_at && ++calls==fail_at)return NULL;return malloc(n);}
static void *marker_calloc(size_t n,size_t z){if(fail_at && ++calls==fail_at)return NULL;return calloc(n,z);}
#define malloc marker_malloc
#define calloc marker_calloc
#include "../../src/nanoisa/assembler.c"
#undef malloc
#undef calloc
int main(void) {
    const char *source=".function main 0 1 0 int 1\n.local_begin 0 \"name\"\n"
        "PUSH_I64 2\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nRET\n.end\n.entry main\n";
    bool passed=false;unsigned checks=0;
    for(unsigned attempt=1;attempt<200;attempt++) {
        AsmResult r;calls=0;fail_at=attempt;NvmModule *m=asm_assemble(source,&r);fail_at=0;++checks;
        if(m){assert(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_VALID);nvm_module_free(m);passed=true;break;}
        assert(r.error!=ASM_OK);
        m=asm_assemble(source,&r);assert(m);nvm_module_free(m);
    }
    assert(passed);printf("I passed %u local-marker allocation boundaries.\n",checks);return 0;
}
