#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static size_t live,pack_size,hits;
static int fail_pack;
static FILE *output;
static void *tracked_malloc(size_t n){void *p=malloc(n);if(p)live++;return p;}
static void *tracked_calloc(size_t n,size_t size){void *p=calloc(n,size);if(p)live++;return p;}
static void tracked_free(void *p){if(p){assert(live);live--;free(p);}}
static void *pack_alloc(size_t n,size_t size){
    if(fail_pack && ftell(output)==2){
        assert(n==1 && size==pack_size);hits++;fail_pack=0;
        fprintf(stderr,"native post-append pack count=%zu bytes=%zu\n",n,size);return NULL;
    }
    return tracked_calloc(n,size);
}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#define NOWN_ALLOC pack_alloc
#undef stdout
#define stdout output
#define NVM2C_NO_MAIN
#include "GENERATED"
#undef malloc
#undef calloc
#undef free
static void run(int fault){
    output=tmpfile();assert(output);fail_pack=fault;hits=0;int64_t value=-91;
    int status=nvm_owned_entry(&value);assert(!live);assert(!fflush(output));
    long length=ftell(output);rewind(output);char bytes[32]={0};assert(length>=0 && length<(long)sizeof bytes);
    assert(fread(bytes,1,(size_t)length,output)==(size_t)length);assert(!fclose(output));output=NULL;
    fprintf(stderr,"native mutation fault=%d status=%d hits=%zu value=%lld prefix=%s",fault,status,hits,(long long)value,bytes);
    assert(fault?(hits==1 && status==1 && value==-91):(hits==0 && status==0 && value==0));
    assert(!strcmp(bytes,fault?"1\n":"1\n7\n1\n"));
}
int main(void){pack_size=sizeof(nown_record)+3*sizeof(nown_value);run(0);if(PACK_CASE){run(1);run(0);}return 0;}
