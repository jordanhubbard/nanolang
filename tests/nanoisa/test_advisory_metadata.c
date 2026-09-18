/* I preserve advisory facts without granting execution authority. */
#include "nvm_format.h"
#include "nvm_v2_sections.h"
#include "assembler.h"
#include "disassembler.h"
#include "verifier.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "nvm2c.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
#define CHECK(x) do { ++checks; assert(x); } while (0)
static void word(uint8_t *p, uint32_t n) { for (unsigned i=0;i<4;i++) p[i]=(uint8_t)(n>>(8*i)); }
static void slot(uint8_t *p,uint8_t tag,uint8_t mode,uint32_t layout) {
    p[0]=tag; p[1]=mode; word(p+4,layout);
}
#include "multi_caller_fixture.h"
static uint8_t *wire(NvmModule *m, size_t *size) {
    NvmV2Module v;
    CHECK(nvm_v2_from_nvm_module(m, &v) == NVM_V2_OK);
    CHECK(nvm_v2_module_serialize(&v, NULL, 0, size) == NVM_V2_OK);
    uint8_t *p = malloc(*size); CHECK(p);
    CHECK(nvm_v2_module_serialize(&v, p, *size, size) == NVM_V2_OK);
    nvm_v2_module_free(&v); return p;
}
static NvmModule *copy(NvmModule *m) {
    size_t size; uint8_t *p = wire(m, &size);
    NvmV2Module v; CHECK(nvm_v2_module_deserialize(p, size, &v) == NVM_V2_OK);
    NvmModule *out = NULL; CHECK(nvm_v2_to_nvm_module(&v, &out) == NVM_V2_OK);
    nvm_v2_module_free(&v); free(p); return out;
}
static NvmModule *ordinary(void) {
    AsmResult r;
    NvmModule *m = asm_assemble(".function main 0 0 0 int 1\nPUSH_I64 42\nRET\n.end\n.entry main\n", &r);
    CHECK(m); return m;
}
static void compare_wire(NvmModule *a, NvmModule *b) {
    size_t na, nb; uint8_t *pa = wire(a, &na), *pb = wire(b, &nb);
    CHECK(na == nb && memcmp(pa, pb, na) == 0); free(pa); free(pb);
}
int main(int argc, char **argv) {
    NvmModule *m = ordinary();
    uint32_t legacy_size; uint8_t *legacy = nvm_serialize(m, &legacy_size);
    CHECK(legacy && legacy_size); free(legacy);
    uint32_t k = nvm_add_string(m, "example.fact", 12);
    const char exact[] = {'a', '\0', '"', '\\', '\n', (char)0xff};
    uint32_t v = nvm_add_string(m, exact, sizeof exact);
    uint32_t empty = nvm_add_string(m, "", 0);
    uint32_t source = nvm_add_string(m, "nano.source_file", 16);
    uint32_t first = nvm_add_string(m, "first.nano", 10);
    uint32_t last = nvm_add_string(m, "last.nano", 9);
    CHECK(nvm_add_metadata(m, k, v)); CHECK(nvm_add_metadata(m, k, empty));
    CHECK(nvm_add_metadata(m, source, first)); CHECK(nvm_add_metadata(m, source, last));
    CHECK(m->source_file_idx == last && nvm_metadata_valid(m));
    CHECK(!nvm_add_metadata(m, UINT32_MAX, v));
    CHECK(!nvm_add_metadata(m, k, UINT32_MAX));
    legacy_size = 99; CHECK(!nvm_serialize(m, &legacy_size) && legacy_size == 0);
    CHECK(nvm_verify(m).ok);
    NvmModule *b = copy(m); compare_wire(m, b);
    CHECK(b->metadata_count == 4 && b->source_file_idx == last);
    CHECK(b->metadata[0].key_idx == k && b->metadata[1].key_idx == k);
    CHECK(b->string_lengths[v] == sizeof exact && memcmp(b->strings[v], exact, sizeof exact) == 0);
    char *text = disasm_module_styled(b, DISASM_STYLE_CANONICAL); CHECK(text);
    AsmResult r; NvmModule *c = asm_assemble(text, &r); CHECK(c);
    compare_wire(b, c); CHECK(nvm_verify(c).ok); free(text);
    /* I free the source and decoded wire before using the independent module. */
    nvm_module_free(m); nvm_module_free(b);
    CHECK(c->metadata_count == 4 && c->strings[last][0] == 'l');
    c->source_file_idx = first;
    NvmV2Module bad; CHECK(nvm_v2_from_nvm_module(c, &bad) == NVM_V2_ERR_INDEX_RANGE);
    c->source_file_idx = last;
    nvm_strip_debug_info(c);
    CHECK(c->metadata_count == 2 && c->source_file_idx == 0);
    b = copy(c); compare_wire(c, b); nvm_module_free(b); nvm_module_free(c);

    /* I canonicalize a legacy source key once, without growing the pool again. */
    m = ordinary(); source = nvm_add_string(m, "nano.source_file", 16);
    m->source_file_idx = nvm_add_string(m, "legacy.nano", 11);
    b = copy(m); CHECK(b->metadata_count == 1 && b->metadata[0].key_idx == source);
    CHECK(b->string_count == m->string_count); c = copy(b); compare_wire(b, c);
    CHECK(c->string_count == b->string_count);
    nvm_module_free(m); nvm_module_free(b); nvm_module_free(c);

    /* I retain an explicit empty source value even at string index zero. */
    m = nvm_module_new(); CHECK(m);
    empty = nvm_add_string(m, "", 0); CHECK(empty == 0);
    source = nvm_add_string(m, "nano.source_file", 16);
    CHECK(nvm_add_metadata(m, source, empty));
    b = copy(m); CHECK(b->metadata_count == 1 && b->source_file_idx == 0);
    compare_wire(m, b); nvm_module_free(m); nvm_module_free(b);

    /* I carry advisory entries beside authoritative layouts/ownership unchanged. */
    uint8_t modes[2]={2,2};
    m=multi_fixture(TWO_ROOTS TWO_BORROWS "CALL_REF 1 0\nPOP\n" TWO_SUM, WRITE_BOTH,2,modes);
    k=nvm_add_string(m,"example.advisory",16); v=nvm_add_string(m,"untrusted claim",15);
    CHECK(nvm_add_metadata(m,k,v)); b=copy(m); compare_wire(m,b);
    CHECK(nvm_verify(b).ok);
    CHECK(b->ownership_size==m->ownership_size && !memcmp(b->ownership_data,m->ownership_data,m->ownership_size));
    CHECK(b->layout_size==m->layout_size && !memcmp(b->layout_data,m->layout_data,m->layout_size));
    text=disasm_module_styled(b,DISASM_STYLE_CANONICAL); CHECK(text);
    c=asm_assemble(text,&r); CHECK(c); compare_wire(b,c); free(text);
    nvm_module_free(m); nvm_module_free(b); nvm_module_free(c);

    m=asm_assemble(".entry 0\n.function main 0 2 0 int 1\n"
        "PUSH_I64 17\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 25\nADD\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n",&r); CHECK(m);
    uint32_t fields[]={1,1,2,0,0,28,2,12,28,0,1,0,0,0,1,0,12,1,0,0,0,0};
    m->passive_size=sizeof fields; m->passive_data=malloc(sizeof fields); CHECK(m->passive_data);
    for(size_t i=0;i<sizeof fields/sizeof *fields;i++) word(m->passive_data+i*4,fields[i]);
    k=nvm_add_string(m,"example.purity",14); v=nvm_add_string(m,"unknown",7);
    CHECK(nvm_add_metadata(m,k,v)); CHECK(nvm_verify(m).ok); b=copy(m); compare_wire(m,b);
    CHECK(nvm_verify(b).ok && b->passive_size==m->passive_size && !memcmp(b->passive_data,m->passive_data,m->passive_size));
    nvm_module_free(m); nvm_module_free(b);

    /* I emit an ordinary artifact for actual CLI/native controls. */
    if(argc==2) {
        m=ordinary(); k=nvm_add_string(m,"example.fact",12); v=nvm_add_string(m,"advisory",8);
        CHECK(nvm_add_metadata(m,k,v));
        char path[4096],error[256]; size_t length; uint8_t *bytes=wire(m,&length);
        snprintf(path,sizeof path,"%s.nvm",argv[1]); FILE *f=fopen(path,"wb"); CHECK(f);
        CHECK(fwrite(bytes,1,length,f)==length); CHECK(!fclose(f)); free(bytes);
        char *native=nvm2c_emit(m,error,sizeof error); CHECK(native);
        snprintf(path,sizeof path,"%s.c",argv[1]); f=fopen(path,"w"); CHECK(f);
        CHECK(fwrite(native,1,strlen(native),f)==strlen(native)); CHECK(!fclose(f)); free(native);
        nvm_module_free(m);
    }

    const char *bad_text[] = {".metadata 0 0\n", ".string \"k\"\n.metadata -1 0\n",
        ".string \"k\"\n.metadata 0 1\n", ".string \"k\"\n.metadata 0 0 extra\n",
        ".string \"k\"\n.function f 0 0 0 void 0\n.metadata 0 0\nRET\n.end\n"};
    for (size_t i = 0; i < sizeof bad_text / sizeof *bad_text; ++i) {
        m = asm_assemble(bad_text[i], &r); CHECK(!m);
    }
    printf("I passed %u advisory metadata checks.\n", checks); return 0;
}
