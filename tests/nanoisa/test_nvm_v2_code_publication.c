/* I inspect corrected conversion results only; no bytecode executes here. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "nvm_format.h"
#include "isa.h"
static unsigned checks,live,attempts,failure_attempt,hits;
static void *allocations[128],*initial_code;
static size_t requested_code;
static int fail_code;
#define CHECK(x) do{checks++;assert(x);}while(0)
static void remember(void *p){if(!p)return;CHECK(live<128);allocations[live++]=p;}
static unsigned locate(void *p){for(unsigned i=0;i<live;i++)if(allocations[i]==p)return i;CHECK(0);return 0;}
void *code_test_malloc(size_t n){attempts++;void *p=malloc(n);remember(p);return p;}
void *code_test_calloc(size_t n,size_t width){attempts++;void *p=calloc(n,width);remember(p);if(n==4096 && width==1){CHECK(!initial_code);initial_code=p;}return p;}
void *code_test_realloc(void *p,size_t n){
    attempts++;unsigned at=locate(p);
    if(p==initial_code){CHECK(n==requested_code && n>4096);if(fail_code){fail_code=0;hits++;failure_attempt=attempts;return NULL;}}
    int is_code=p==initial_code;void *next=realloc(p,n);if(next){allocations[at]=next;if(is_code)initial_code=next;}return next;
}
void code_test_free(void *p){if(!p)return;unsigned at=locate(p);if(p==initial_code)initial_code=NULL;allocations[at]=allocations[--live];free(p);}
static NvmV2Module input(uint8_t *bytes,uint64_t size){NvmV2Module m={0};m.code=bytes;m.code_size=size;m.entry_point=NVM_V2_NO_ENTRY_POINT;return m;}
static void exact_copy(uint32_t size){
    uint8_t *bytes=malloc(size?size:1);CHECK(bytes);for(uint32_t i=0;i<size;i++)bytes[i]=(uint8_t)(i*37u);
    NvmV2Module m=input(bytes,size);NvmModule sentinel={0},*out=&sentinel;requested_code=size;hits=0;unsigned before=attempts;
    CHECK(nvm_v2_to_nvm_module(&m,&out)==NVM_V2_OK && out && out!=&sentinel);CHECK(out->code_size==size && out->code_capacity>=size && out->code!=bytes);CHECK(!memcmp(out->code,bytes,size));CHECK(m.code==bytes && m.code_size==size);
    if(size>4096){CHECK(out->code_capacity==size);}CHECK(attempts>before && !hits);free(bytes);for(uint32_t i=0;i<size;i++)CHECK(out->code[i]==(uint8_t)(i*37u));nvm_module_free(out);CHECK(!live && !initial_code);
}
static void transient_code_failure(void){
    enum{SIZE=8193};uint8_t bytes[SIZE],saved[SIZE];for(unsigned i=0;i<SIZE;i++)bytes[i]=(uint8_t)(i*19u);memcpy(saved,bytes,SIZE);NvmV2Module m=input(bytes,SIZE),copy=m;NvmModule sentinel={0},*out=&sentinel;
    requested_code=SIZE;fail_code=1;hits=0;failure_attempt=0;
    CHECK(nvm_v2_to_nvm_module(&m,&out)==NVM_V2_ERR_TRUNCATED && !out);CHECK(hits==1 && !fail_code && failure_attempt==attempts);CHECK(!live && !initial_code && !memcmp(bytes,saved,SIZE) && !memcmp(&m,&copy,sizeof m));
    out=&sentinel;CHECK(nvm_v2_to_nvm_module(&m,&out)==NVM_V2_OK && out && out!=&sentinel);CHECK(out->code_size==SIZE && !memcmp(out->code,bytes,SIZE));nvm_module_free(out);CHECK(!live && !initial_code);
}
static void refusals(void){
    NvmModule sentinel={0},*out=&sentinel;CHECK(nvm_v2_to_nvm_module(NULL,&out)==NVM_V2_ERR_INDEX_RANGE && out==&sentinel);
    NvmV2Module m=input(NULL,1);CHECK(nvm_v2_to_nvm_module(&m,NULL)==NVM_V2_ERR_INDEX_RANGE);CHECK(nvm_v2_to_nvm_module(&m,&out)==NVM_V2_ERR_INDEX_RANGE && !out && !live);
    uint8_t one=0;m=input(&one,(uint64_t)UINT32_MAX+1);out=&sentinel;CHECK(nvm_v2_to_nvm_module(&m,&out)==NVM_V2_ERR_INDEX_RANGE && !out && !live && !one);
    /* I retain the earlier constant-pool error before the CODE-stage refusal. */
    NvmV2Constant bad={0};bad.tag=TAG_INT;m=input(NULL,1);m.constants.items=&bad;m.constants.count=1;out=&sentinel;CHECK(nvm_v2_to_nvm_module(&m,&out)==NVM_V2_ERR_SECTION_TYPE && !out && !live);
}
int main(void){exact_copy(0);exact_copy(4095);exact_copy(4096);exact_copy(4097);exact_copy(8193);transient_code_failure();refusals();exact_copy(8193);printf("%u checked CODE publication controls passed\n",checks);return 0;}
