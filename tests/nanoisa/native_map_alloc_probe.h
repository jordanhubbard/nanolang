/* Test-only independent allocation oracle around generated C. */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
static struct {void *pointer;size_t bytes;} ntest_allocations[4096];
static size_t ntest_live,ntest_bytes,ntest_calls,ntest_fail=SIZE_MAX;
static int ntest_persistent;
static int ntest_denied(void){size_t i=ntest_calls++;return i==ntest_fail||(ntest_persistent&&i>ntest_fail);}
static size_t ntest_find(void *p){for(size_t i=0;i<ntest_live;i++)if(ntest_allocations[i].pointer==p)return i;fputs("test allocation identity mismatch\n",stderr);abort();}
static void ntest_record(void *p,size_t n){if(!p)return;if(ntest_live>=4096)abort();ntest_allocations[ntest_live].pointer=p;ntest_allocations[ntest_live++].bytes=n;ntest_bytes+=n;}
static void *ntest_malloc(size_t n){if(ntest_denied())return NULL;void *p=malloc(n);ntest_record(p,n);return p;}
static void *ntest_calloc(size_t n,size_t m){if(ntest_denied()||(m&&n>SIZE_MAX/m))return NULL;void *p=calloc(n,m);ntest_record(p,n*m);return p;}
static void ntest_free(void *p){if(!p)return;size_t i=ntest_find(p);ntest_bytes-=ntest_allocations[i].bytes;ntest_allocations[i]=ntest_allocations[--ntest_live];free(p);}
static void *ntest_realloc(void *p,size_t n){
    if(ntest_denied())return NULL;
    if(!p){void *q=malloc(n);ntest_record(q,n);return q;}
    size_t i=ntest_find(p),old=ntest_allocations[i].bytes;
    if(!n){ntest_free(p);return NULL;}
    void *q=realloc(p,n);if(!q)return NULL;
    ntest_allocations[i].pointer=q;ntest_allocations[i].bytes=n;ntest_bytes=ntest_bytes-old+n;return q;
}
#define malloc ntest_malloc
#define calloc ntest_calloc
#define realloc ntest_realloc
#define free ntest_free
