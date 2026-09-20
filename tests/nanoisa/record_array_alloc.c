/* I measure requested allocation payloads, excluding allocator metadata. */
#include "record_array_alloc.h"
#include <stdio.h>
#include <stdint.h>
static struct {void *p;size_t bytes;} slots[32768];
size_t ra_live,ra_bytes,ra_peak,ra_calls,ra_fail=SIZE_MAX;
int ra_persistent;
static void require(int ok){if(!ok){fputs("I reject an unmatched query allocation\n",stderr);abort();}}
static int denied(void){size_t i=ra_calls++;return i==ra_fail||(ra_persistent&&i>ra_fail);}
static size_t locate(void *p){size_t i=0;while(i<ra_live&&slots[i].p!=p)i++;require(i<ra_live);return i;}
static void remember(void *p,size_t n){if(!p)return;require(ra_live<32768);slots[ra_live].p=p;slots[ra_live++].bytes=n;ra_bytes+=n;if(ra_peak<ra_bytes)ra_peak=ra_bytes;}
void *ra_test_malloc(size_t n){if(denied())return NULL;void *p=malloc(n);remember(p,n);return p;}
void *ra_test_calloc(size_t n,size_t w){if(denied()||(w&&n>SIZE_MAX/w))return NULL;void *p=calloc(n,w);remember(p,n*w);return p;}
void ra_test_free(void *p){if(!p)return;size_t i=locate(p);ra_bytes-=slots[i].bytes;slots[i]=slots[--ra_live];free(p);}
void *ra_test_realloc(void *p,size_t n){
    if(denied())return NULL;
    if(!p){void *q=malloc(n);remember(q,n);return q;}
    size_t i=locate(p),old=slots[i].bytes;
    if(!n){ra_test_free(p);return NULL;}
    void *q=realloc(p,n);if(!q)return NULL;
    slots[i].p=q;slots[i].bytes=n;ra_bytes=ra_bytes-old+n;if(ra_peak<ra_bytes)ra_peak=ra_bytes;return q;
}
