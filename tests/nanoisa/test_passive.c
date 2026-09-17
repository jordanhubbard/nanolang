#include "assembler.h"
#include "disassembler.h"
#include "passive.h"
#include "verifier.h"
#include "nvm_v2_sections.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int passed, failed;
#define CHECK(x) do { if (x) ++passed; else { ++failed; fprintf(stderr,"Failed line %d: %s\n",__LINE__,#x); } } while(0)
static void put(uint8_t *p, uint32_t v) {
    for (unsigned i=0;i<4;++i) p[i]=(uint8_t)(v>>(i*8));
}
static NvmModule *fixture(void) {
    AsmResult r;
    NvmModule *m=asm_assemble(".entry 0\n.function main 0 2 0 int 1\n"
      "PUSH_I64 17\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 25\nADD\nSTORE_LOCAL 0\n"
      "LOAD_LOCAL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n",&r);
    CHECK(m != NULL);
    if (!m) return NULL;
    /* Source node0 depends on later source node1: execution order is1,0. */
    uint32_t fields[]={1,1, 2,0,0,28,2, 12,28,0,1,0,0,0,1, 0,12,1,0,0,0,0};
    m->passive_size=sizeof fields;
    m->passive_data=malloc(sizeof fields);
    for (size_t i=0;i<sizeof fields/sizeof fields[0];++i) put(m->passive_data+i*4,fields[i]);
    return m;
}
static void attach(NvmModule *m, const uint32_t *fields, size_t size) {
    free(m->passive_data); m->passive_data=malloc(size); m->passive_size=(uint32_t)size;
    for(size_t i=0;i<size/4;++i) put(m->passive_data+i*4,fields[i]);
}
static void independent_and_input(void) {
    AsmResult r;
    NvmModule *m=asm_assemble(".entry 0\n.function main 0 2 0 int 1\n"
      "PUSH_I64 17\nSTORE_LOCAL 0\nPUSH_I64 25\nSTORE_LOCAL 1\n"
      "LOAD_LOCAL 0\nLOAD_LOCAL 1\nADD\nRET\n.end\n",&r);
    CHECK(m!=NULL); if(!m) return;
    const uint32_t independent[]={1,1,1,0,0,24,2, 0,12,0,0,0,0,0, 12,24,1,0,0,0,0};
    attach(m,independent,sizeof independent);CHECK(nvm_verify(m).ok);
    /* A declared edge not actually read is invalid even with a valid graph. */
    const uint32_t extra_edge[]={1,1,2,0,0,24,2, 0,12,0,0,0,0,0, 12,24,1,1,0,0,0,0};
    attach(m,extra_edge,sizeof extra_edge);CHECK(!nvm_verify(m).ok);
    const uint32_t cycle[]={1,1,2,0,0,24,2, 0,12,0,1,0,0,0,1, 12,24,1,1,0,0,0,0};
    attach(m,cycle,sizeof cycle);CHECK(!nvm_verify(m).ok);nvm_module_free(m);
    m=asm_assemble(".function helper 1 2 0 int 1\n"
      "LOAD_LOCAL 0\nPUSH_I64 1\nADD\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nRET\n.end\n"
      ".parameters 0 int\n",&r);
    CHECK(m!=NULL);if(!m)return;
    const uint32_t input[]={1,1,1,0,0,16,1, 0,16,1,0,1,0,0,0};
    /* A declared tag alone is not a closed proof of the caller value. */
    attach(m,input,sizeof input);CHECK(!nvm_verify(m).ok);
    m->function_param_types[0][0]=TAG_VOID;CHECK(!nvm_verify(m).ok);
    m->function_param_types[0][0]=TAG_ARRAY;CHECK(!nvm_verify(m).ok);
    m->function_param_types[0][0]=TAG_INT;
    /* A node cannot replace its input, and cannot omit a real read. */
    put(m->passive_data+9*4,0);CHECK(!nvm_verify(m).ok);put(m->passive_data+9*4,1);
    const uint32_t missing_read[]={1,1,1,0,0,16,1, 0,16,1,0,0,0,0};
    attach(m,missing_read,sizeof missing_read);CHECK(!nvm_verify(m).ok);
    nvm_module_free(m);
}
static void text_roundtrip(const NvmModule *m, const uint8_t *bytes, size_t size) {
    char *text = disasm_module_styled(m, DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text, ".passive ") != NULL);
    if (!text) return;
    AsmResult result;
    NvmModule *copy = asm_assemble(text, &result);
    CHECK(copy != NULL);
    if (copy) {
        CHECK(copy->passive_size == m->passive_size &&
              memcmp(copy->passive_data, m->passive_data, m->passive_size) == 0);
        CHECK(copy->code_size == m->code_size && memcmp(copy->code, m->code, m->code_size) == 0);
        NvmV2Module converted;
        CHECK(nvm_v2_from_nvm_module(copy, &converted) == NVM_V2_OK);
        size_t needed = 0;
        CHECK(nvm_v2_module_serialize(&converted, NULL, 0, &needed) == NVM_V2_OK);
        uint8_t *output = malloc(needed);
        CHECK(output != NULL);
        if (output) {
            CHECK(nvm_v2_module_serialize(&converted, output, needed, NULL) == NVM_V2_OK);
            CHECK(needed == size && memcmp(output, bytes, size) == 0);
            free(output);
        }
        nvm_v2_module_free(&converted);
        nvm_module_free(copy);
    }
    /* Replacing a valid version with an unsupported one cannot erase the claim. */
    char *claim = strstr(text, ".passive \"");
    if (claim) {
        claim[10] = '2';
        copy = asm_assemble(text, &result);
        CHECK(!copy && result.error == ASM_ERR_VERIFY);
        nvm_module_free(copy);
    }
    free(text);
    const char *invalid[] = {".passive \"\"\n", ".passive \"0\"\n",
        ".passive \"zz\"\n", ".passive \"00\" trailing\n",
        ".function main 0 0 0 void 0\n.passive \"00\"\nRET\n.end\n"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        copy = asm_assemble(invalid[i], &result);
        CHECK(!copy && result.error == ASM_ERR_SYNTAX);
        nvm_module_free(copy);
    }
}
int main(int argc,char **argv) {
    independent_and_input();
    NvmModule *m=fixture(); if(!m) return 1;
    CHECK(nvm_verify(m).ok);
    uint32_t legacy_size=99; uint8_t *legacy=nvm_serialize(m,&legacy_size);
    CHECK(legacy==NULL && legacy_size==0); free(legacy);
    NvmV2Module v2;
    CHECK(nvm_v2_from_nvm_module(m,&v2)==NVM_V2_OK);
    size_t size=0; CHECK(nvm_v2_module_serialize(&v2,NULL,0,&size)==NVM_V2_OK);
    uint8_t *bytes=malloc(size),*again=malloc(size);
    CHECK(nvm_v2_module_serialize(&v2,bytes,size,NULL)==NVM_V2_OK);
    text_roundtrip(m,bytes,size);
    NvmV2Header header; CHECK(nvm_v2_read_header(bytes,size,&header)==NVM_V2_OK);
    CHECK((header.feature_bits & NVM_V2_FEATURE_PASSIVE)!=0);
    NvmV2Module decoded; CHECK(nvm_v2_module_deserialize(bytes,size,&decoded)==NVM_V2_OK);
    NvmModule *copy=NULL; CHECK(nvm_v2_to_nvm_module(&decoded,&copy)==NVM_V2_OK);
    CHECK(copy && nvm_verify(copy).ok && copy->passive_size==m->passive_size &&
          memcmp(copy->passive_data,m->passive_data,m->passive_size)==0);
    CHECK(nvm_v2_module_serialize(&decoded,again,size,NULL)==NVM_V2_OK);
    CHECK(memcmp(bytes,again,size)==0);
    if (argc>1) {
        FILE *file=fopen(argv[1],"wb"); CHECK(file!=NULL);
        if(file) { CHECK(fwrite(bytes,1,size,file)==size); CHECK(fclose(file)==0); }
    }
    nvm_module_free(copy); nvm_v2_module_free(&decoded);
    /* Every prefix is incomplete, including one with a complete first node. */
    uint32_t original_size=m->passive_size;
    for(uint32_t i=0;i<original_size;++i) { m->passive_size=i; CHECK(!nvm_passive_valid(m)); }
    m->passive_size=original_size;
    const uint32_t mutations[][2]={
      {0,2},{1,0},{1,UINT32_MAX},{2,3},{2,1},{3,1},{4,1},{5,27},{5,29},
      {6,0},{6,UINT32_MAX},{7,13},{8,27},{9,1},{10,0},{11,1},{12,1},{13,1},
      {14,0},{14,2},{15,1},{16,13},{17,0},{18,1},{19,1},{20,1},{21,1}};
    for(size_t i=0;i<sizeof mutations/sizeof mutations[0];++i) {
        uint8_t old[4]; uint32_t offset=mutations[i][0]*4;
        memcpy(old,m->passive_data+offset,4);put(m->passive_data+offset,mutations[i][1]);
        CHECK(!nvm_verify(m).ok);
        NvmV2Module bad; CHECK(nvm_v2_from_nvm_module(m,&bad)!=NVM_V2_OK);
        /* Even a direct v2 producer cannot publish the malformed claim. */
        CHECK(nvm_v2_module_serialize(&v2,NULL,0,NULL)!=NVM_V2_OK);
        memcpy(m->passive_data+offset,old,4);
    }
    /* A syntactically valid but effectful node is not eligible. */
    uint8_t old=m->code[24]; m->code[24]=OP_PRINTLN;
    CHECK(!nvm_verify(m).ok);m->code[24]=old;
    /* Container checksum is recomputed so semantic metadata checks must reject. */
    for(uint32_t i=0;i<header.section_count;++i) {
        NvmV2SectionEntry e; CHECK(nvm_v2_read_section(bytes,size,&header,i,&e)==NVM_V2_OK);
        if(e.type==NVM_V2_SECTION_PASSIVE) {
            put(bytes+e.offset+12*4,1);
            header.checksum=nvm_crc32(bytes+header.header_size,(uint32_t)(size-header.header_size));
            nvm_v2_write_header(bytes,&header);
            NvmV2Module bad; CHECK(nvm_v2_module_deserialize(bytes,size,&bad)!=NVM_V2_OK);
        }
    }
    nvm_v2_module_free(&v2);nvm_module_free(m);free(bytes);free(again);
    printf("passive: %d passed, %d failed\n",passed,failed);
    return failed ? 1:0;
}
