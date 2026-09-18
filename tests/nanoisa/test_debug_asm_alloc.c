/* I check the assembler's publication boundary when a debug append fails. */
#include "nvm_format.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
static unsigned fail_at,calls;
static bool debug_append(NvmModule *m,uint32_t offset,uint32_t line,uint32_t column) {
    if(fail_at && ++calls==fail_at)return false;
    return nvm_add_debug_entry(m,offset,line,column);
}
#define nvm_add_debug_entry debug_append
#include "../../src/nanoisa/assembler.c"
#undef nvm_add_debug_entry
int main(void) {
    const char *source=".debug 0 1 1\n.debug 0 2 0\n.function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.entry main\n";
    for(unsigned i=1;i<=2;i++) {
        AsmResult r;calls=0;fail_at=i;
        NvmModule *m=asm_assemble(source,&r);fail_at=0;
        assert(!m && r.error==ASM_ERR_MEMORY);
        m=asm_assemble(source,&r);assert(m && m->debug_count==2);
        nvm_module_free(m);
    }
    puts("I passed both assembler DEBUG append failure paths.");return 0;
}
