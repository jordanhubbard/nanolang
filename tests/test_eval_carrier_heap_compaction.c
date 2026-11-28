/* I preserve colliding identities while retiring probe-chain holes in place. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail carrier compaction check %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
static size_t table_allocations,metadata;
static void *counted_calloc(size_t n,size_t bytes) { void *p=calloc(n,bytes);if(p){table_allocations++;metadata++;}return p; }
static void *counted_malloc(size_t bytes) { void *p=malloc(bytes);if(p)metadata++;return p; }
static void counted_free(void *p) { if(p){CHECK(metadata);metadata--;}free(p); }
#define calloc counted_calloc
#define malloc counted_malloc
#define free counted_free
#include "../src/eval/carrier_heap.c"
#undef calloc
#undef malloc
#undef free
static size_t destroyed;
typedef struct {size_t id,retired;} Payload;
static Payload payloads[8192];
static void retire(void *p,void *context) { Payload *v=p;CHECK(context==&destroyed && !v->retired);v->retired=1;destroyed++; }
static void track(NlCarrierHeap *h,Payload *p) { p->retired=0;CHECK(nl_carrier_heap_track(h,p,NULL,retire,&destroyed));CHECK(nl_carrier_heap_context(h,p)==&destroyed); }
int main(void) {
    NlCarrierHeap *h=nl_carrier_heap_new();CHECK(h);size_t initial_allocations=table_allocations;
    Payload *collision[8];size_t n=0;
    for(size_t i=0;i<8192 && n<8;i++)if((carrier_hash(&payloads[i])&31)==31)collision[n++]=&payloads[i];
    CHECK(n==8);
    for(size_t round=0;round<64;round++) {
        for(size_t i=0;i<8;i++)track(h,collision[i]);
        CHECK(nl_carrier_heap_forget(h,collision[2]));CHECK(nl_carrier_heap_forget(h,collision[0]));
        for(size_t i=0;i<8;i++)CHECK(nl_carrier_heap_contains(h,collision[i])==(i!=0 && i!=2));
        track(h,collision[2]);track(h,collision[0]);
        const void *roots[]={collision[1],collision[5]};CHECK(nl_carrier_heap_collect(h,roots,2));
        for(size_t i=0;i<8;i++)CHECK(nl_carrier_heap_contains(h,collision[i])==(i==1 || i==5));
        CHECK(nl_carrier_heap_stats(h).live==2);CHECK(nl_carrier_heap_collect(h,NULL,0));CHECK(nl_carrier_heap_stats(h).live==0);
    }
    for(size_t round=0;round<1024;round++) {
        for(size_t i=0;i<8;i++)track(h,&payloads[(round*13+i*17)%8192]);
        CHECK(nl_carrier_heap_clear(h));CHECK(nl_carrier_heap_stats(h).live==0);
        CHECK(table_allocations==initial_allocations);
    }
    NlCarrierHeapStats s=nl_carrier_heap_stats(h);CHECK(s.registered==64*10+1024*8);CHECK(s.transferred==128);CHECK(s.destroyed==destroyed);CHECK(s.registered==s.destroyed+s.transferred);
    CHECK(nl_carrier_heap_free(h));CHECK(metadata==0);
    puts("I preserve wrapped collisions, forget/readoption, moved sweep entries and changing-key churn without another table allocation; all metadata retires.");return 0;
}
