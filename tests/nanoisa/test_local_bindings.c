#include "local_bindings.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do {++checks; assert(x);} while(0)
static void roundtrip(NvmModule *m) {
    NanoisaErr err;uint32_t n;
    uint8_t *wire=nanoisa_save_bytes(m,&n,&err);CHECK(wire);
    NvmModule *b=nanoisa_load_bytes(wire,n,&err);CHECK(b);
    char *text=disasm_module_styled(b,DISASM_STYLE_CANONICAL);CHECK(text);
    AsmResult ar;NvmModule *c=asm_assemble(text,&ar);CHECK(c);
    uint32_t k;uint8_t *again=nanoisa_save_bytes(c,&k,&err);CHECK(again);
    CHECK(m->code_size==c->code_size && !memcmp(m->code,c->code,m->code_size));
    CHECK(b->metadata_count==c->metadata_count);
    for(uint32_t i=0;i<b->metadata_count;i++) {
        const NvmMetadataEntry *x=&b->metadata[i],*y=&c->metadata[i];
        CHECK(b->string_lengths[x->key_idx]==c->string_lengths[y->key_idx]);
        CHECK(!memcmp(b->strings[x->key_idx],c->strings[y->key_idx],b->string_lengths[x->key_idx]));
        CHECK(b->string_lengths[x->value_idx]==c->string_lengths[y->value_idx]);
        CHECK(!memcmp(b->strings[x->value_idx],c->strings[y->value_idx],b->string_lengths[x->value_idx]));
    }
    char *canonical=disasm_module_styled(c,DISASM_STYLE_CANONICAL);CHECK(canonical);
    NvmModule *d=asm_assemble(canonical,&ar);CHECK(d);
    uint32_t final_size;uint8_t *final=nanoisa_save_bytes(d,&final_size,&err);CHECK(final);
    CHECK(k==final_size && !memcmp(again,final,k));
    free(final);free(canonical);nvm_module_free(d);
    CHECK(nvm_local_names_validate(m)==nvm_local_names_validate(c));
    free(wire);free(again);free(text);nvm_module_free(b);nvm_module_free(c);
}
int main(int argc,char **argv) {
    if(argc==2) {
        NanoisaErr err;NvmModule *m=nanoisa_load_file(argv[1],&err);CHECK(m);
        CHECK(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_VALID);
        for(uint32_t f=0;f<m->function_count;f++)for(uint16_t s=0;s<m->functions[f].local_count;s++) {
            uint32_t last=UINT32_MAX;
            for(uint32_t pc=0;pc<m->functions[f].code_length;pc++) {
                NvmLocalBinding b;
                if(nvm_local_name_at(m,f,s,pc,&b)==NVM_LOCAL_NAMES_VALID && b.begin!=last) {
                    printf("%s %u %u %u %.*s\n",nvm_get_string(m,m->functions[f].name_idx),s,b.begin,b.end,(int)b.name_size,b.name);
                    last=b.begin;
                }
            }
        }
        roundtrip(m);nvm_module_free(m);return 0;
    }
    NvmModule incomplete={.function_count=1};
    CHECK(nvm_local_names_validate(&incomplete)==NVM_LOCAL_NAMES_INVALID);
    const char *source=".function main 0 1 0 int 1\n"
        "PUSH_I64 40\nSTORE_LOCAL 0\n.local_begin 0 \"first\"\n"
        "LOAD_LOCAL 0\nPOP\n.local_end 0\nPUSH_I64 2\nSTORE_LOCAL 0\n"
        ".local_begin 0 \"second\"\nLOAD_LOCAL 0\nRET\n.end\n.entry main\n";
    AsmResult ar;NvmModule *m=asm_assemble(source,&ar);CHECK(m);
    CHECK(nvm_verify(m).ok && nvm_local_names_validate(m)==NVM_LOCAL_NAMES_VALID);
    NvmLocalBinding b={.slot=99};
    CHECK(nvm_local_name_at(m,0,0,11,&b)==NVM_LOCAL_NAMES_ABSENT && b.slot==99);
    CHECK(nvm_local_name_at(m,0,0,12,&b)==NVM_LOCAL_NAMES_VALID && b.begin==12 && b.end==16);
    CHECK(b.name_size==5 && !memcmp(b.name,"first",5));
    CHECK(nvm_local_name_at(m,0,0,15,&b)==NVM_LOCAL_NAMES_VALID);
    CHECK(nvm_local_name_at(m,0,0,16,&b)==NVM_LOCAL_NAMES_ABSENT);
    CHECK(nvm_local_name_at(m,0,0,28,&b)==NVM_LOCAL_NAMES_VALID && b.end==32);
    CHECK(b.name_size==6 && !memcmp(b.name,"second",6));
    CHECK(nvm_local_name_at(m,0,0,32,&b)==NVM_LOCAL_NAMES_ABSENT);
    CHECK(!nvm_add_local_binding(m,&b)); /* I refuse the overlapping interval. */
    b.begin=33;b.end=33;CHECK(!nvm_add_local_binding(m,&b));
    b.begin=1;b.end=2;CHECK(!nvm_add_local_binding(m,&b));
    b.begin=32;b.end=32;CHECK(nvm_add_local_binding(m,&b));
    CHECK(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_VALID);roundtrip(m);
    uint32_t n=99;CHECK(!nvm_serialize(m,&n) && n==0);
    /* Raw optional metadata remains transportable even when names are unusable. */
    CHECK(nvm_add_metadata(m,m->metadata[0].key_idx,m->metadata[0].value_idx));
    CHECK(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_INVALID && nvm_verify(m).ok);
    b.slot=99;CHECK(nvm_local_name_at(m,0,0,12,&b)==NVM_LOCAL_NAMES_INVALID && b.slot==99);
    roundtrip(m);nvm_module_free(m);
    const char *bad[]={
        ".local_begin 0 \"outside\"\n",
        ".function f 0 1 0 void 0\n.local_end 0\nRET\n.end\n",
        ".function f 0 1 0 void 0\n.local_begin 1 \"x\"\nRET\n.end\n",
        ".function f 0 1 0 void 0\n.local_begin 0 \"\"\nRET\n.end\n",
        ".function f 0 1 0 void 0\n.local_begin 0 \"a\"\n.local_begin 0 \"b\"\nRET\n.end\n"};
    for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){m=asm_assemble(bad[i],&ar);CHECK(!m);}
    m=asm_assemble(".function f 0 0 0 void 0\nRET\n.end\n",&ar);CHECK(m);
    CHECK(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_ABSENT);
    uint32_t k=nvm_add_string(m,"nano.local.v2",13),v=nvm_add_string(m,"future",6);
    CHECK(nvm_add_metadata(m,k,v));CHECK(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_ABSENT);roundtrip(m);
    k=nvm_add_string(m,"nano.local.v1",13);CHECK(nvm_add_metadata(m,k,v));
    CHECK(nvm_local_names_validate(m)==NVM_LOCAL_NAMES_INVALID && nvm_verify(m).ok);roundtrip(m);
    nvm_module_free(m);printf("I passed %u lexical local-name checks.\n",checks);return 0;
}
