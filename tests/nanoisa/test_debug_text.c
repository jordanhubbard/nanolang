#include "assembler.h"
#include "disassembler.h"
#include "nvm_v2_sections.h"
#include "verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do { ++checks; assert(x); } while (0)
static const char *program = ".function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.entry main\n";
static void roundtrip(NvmModule *m) {
    NanoisaErr error; uint32_t n,k;
    uint8_t *wire=nanoisa_save_bytes(m,&n,&error); CHECK(wire);
    NvmModule *loaded=nanoisa_load_bytes(wire,n,&error); CHECK(loaded);
    char *text=disasm_module_styled(loaded,DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult r; NvmModule *again=asm_assemble(text,&r); CHECK(again);
    uint8_t *encoded=nanoisa_save_bytes(again,&k,&error); CHECK(encoded);
    CHECK(n==k && !memcmp(wire,encoded,n));
    CHECK(m->debug_count==again->debug_count);
    if(m->debug_count) CHECK(!memcmp(m->debug_entries,again->debug_entries,m->debug_count*sizeof(NvmDebugEntry)));
    CHECK(nvm_verify(m).ok==nvm_verify(again).ok);
    free(wire);free(encoded);free(text);nvm_module_free(loaded);nvm_module_free(again);
}
int main(void) {
    AsmResult r; NvmModule *m=asm_assemble(program,&r); CHECK(m);
    CHECK(nvm_add_debug_entry(m,9,17,0));
    CHECK(nvm_add_debug_entry(m,0,1,1));
    CHECK(nvm_add_debug_entry(m,0,2,0));
    CHECK(nvm_add_debug_entry(m,UINT32_MAX,UINT32_MAX,UINT32_MAX));
    uint32_t key=nvm_add_string(m,"unknown.debug.fact",18), value=nvm_add_string(m,"a\0\xff",3);
    CHECK(nvm_add_metadata(m,key,value)); roundtrip(m);
    nvm_strip_debug_info(m); CHECK(m->debug_count==0 && m->metadata_count==1); roundtrip(m);
    nvm_module_free(m);
    m=asm_assemble(".flag debug_info\n.function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.entry main\n",&r);CHECK(m);
    roundtrip(m);NvmV2Module v;CHECK(nvm_v2_from_nvm_module(m,&v)==NVM_V2_OK);
    CHECK(v.has_debug && !v.debug.count);nvm_v2_module_free(&v);
    CHECK(nvm_add_debug_entry(m,0,5,0));uint32_t n;uint8_t *old=nvm_serialize(m,&n);CHECK(old);
    NvmModule *legacy=nvm_deserialize(old,n);CHECK(legacy && legacy->debug_count==1 && legacy->debug_entries[0].source_line==5);
    nvm_module_free(legacy);free(old);nvm_module_free(m);
    const char *bad[]={".debug -1 2 3\n",".debug 4294967296 2 3\n",".debug 1 2\n",".debug 1 2 3 trailing\n", ".function main 0 0\n.debug 0 1 1\n.end\n"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++){m=asm_assemble(bad[i],&r);CHECK(!m && r.error!=ASM_OK);}
    m=asm_assemble(".debug 0 0 0\n.debug 9 3 4\n.function helper 0 0 0 int 1\nPUSH_I64 2\nRET\n.end\n.function main 0 0 0 int 1\nCALL 0\nRET\n.end\n.entry main\n",&r);
    CHECK(m);roundtrip(m);nvm_module_free(m);
    printf("I passed %u DEBUG text checks.\n",checks);return 0;
}
