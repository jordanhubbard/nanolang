/* Direct ordinary adapters only; no VM dispatch or admission. */
#include "../nanoisa/record_array_alloc.h"
#include "../../src/nanovm/record_map_heap_private.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"ordinary heap check %u: %s\n",__LINE__,#x);abort();}checks++; } while(0)
static unsigned checks;
int g_argc;char **g_argv;
static void run(bool persistent) {
    VmHeap h;vm_heap_init(&h);CHECK(h.intern_buckets&&!h.private_record_dag);
    VmHashMap *m=NULL;CHECK(vm_rm_ordinary_new(&h,&m)==VM_RM_HEAP_OK);
    NanoValue keys[13];
    for(unsigned i=0;i<13;i++) {
        char bytes[3]={'k',0,(char)i};
        VmString *s=vm_string_new(&h,bytes,3);CHECK(s);keys[i]=val_string(s);
    }
    for(unsigned i=0;i<12;i++)CHECK(vm_rm_ordinary_set(&h,m,keys[i],val_int(i))==VM_RM_HEAP_OK);
    /* Real collector buffering, with one actual retained owner left. */
    vm_retain(&h,val_hashmap(m));vm_release(&h,val_hashmap(m));
    CHECK(m->header.buffered&&m->header.colour==VM_GC_PURPLE&&h.cycle_count);
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK(vm_rm_heap_import_validate(m,&w)==VM_RM_HEAP_OK);
    VmString *original=keys[0].as.string;size_t size=sizeof *original+original->length+1;
    VmString *equal=ra_test_malloc(size);CHECK(equal);memcpy(equal,original,size);
    equal->header.ref_count=UINT32_MAX;equal->intern_next=NULL;
    h.stats.allocated+=size;h.stats.num_objects++;h.stats.allocation_calls++;
    size_t calls=ra_calls;VmHMEntry *entries=m->entries;
    CHECK(vm_rm_ordinary_set(&h,m,val_string(equal),val_int(99))==VM_RM_HEAP_OK);
    CHECK(ra_calls==calls&&equal->header.ref_count==UINT32_MAX&&original->header.ref_count==2);
    bool stored=false;for(uint32_t i=0;i<m->bucket_count;i++)
        if(m->entries[i].state&&m->entries[i].key.as.string==original)stored=true;
    CHECK(stored);equal->header.ref_count=1;vm_release(&h,val_string(equal));
    keys[12].as.string->header.ref_count=UINT32_MAX;calls=ra_calls;
    CHECK(vm_rm_ordinary_set(&h,m,keys[12],val_int(12))==VM_RM_HEAP_RETAIN_OVERFLOW&&ra_calls==calls);
    keys[12].as.string->header.ref_count=1;
    size_t allocated=h.stats.allocated,freed=h.stats.freed,live=ra_live,bytes=ra_bytes;
    ra_fail=ra_calls;ra_persistent=persistent;
    CHECK(vm_rm_ordinary_set(&h,m,keys[12],val_int(12))==VM_RM_HEAP_MEMORY);
    ra_fail=SIZE_MAX;ra_persistent=0;
    CHECK(m->entries==entries&&m->count==12&&m->bucket_count==16);
    CHECK(h.stats.allocated==allocated&&h.stats.freed==freed&&ra_live==live&&ra_bytes==bytes);
    CHECK(vm_rm_ordinary_set(&h,m,keys[12],val_int(12))==VM_RM_HEAP_OK);
    CHECK(m->header.buffered&&m->header.colour==VM_GC_PURPLE&&m->bucket_count==32);
    for(unsigned i=0;i<13;i++) {
        NanoValue out=val_void();bool found=false;
        CHECK(vm_rm_ordinary_get(&h,m,keys[i],&out)==VM_RM_HEAP_OK&&out.tag==TAG_INT&&out.as.i64==(i?i:99));
        CHECK(vm_rm_ordinary_has(&h,m,keys[i],&found)==VM_RM_HEAP_OK&&found);
    }
    uint32_t count=0;CHECK(vm_rm_ordinary_len(&h,m,&count)==VM_RM_HEAP_OK&&count==13);
    NanoValue absent=val_string(vm_string_new(&h,"absent",6));CHECK(absent.as.string);
    NanoValue missing=val_int(81);bool present=true;
    CHECK(vm_rm_ordinary_get(&h,m,absent,&missing)==VM_RM_HEAP_OK&&missing.tag==TAG_VOID);
    CHECK(vm_rm_ordinary_has(&h,m,absent,&present)==VM_RM_HEAP_OK&&!present);
    vm_release(&h,absent);
    for(unsigned bad=0;bad<6;bad++) {
        VmHeapHeader mh=m->header,kh=keys[0].as.string->header;
        switch(bad) {
        case 0:m->header.colour=VM_GC_GRAY;break;
        case 1:m->header.colour=VM_GC_BLACK;break; /* buffered must be PURPLE */
        case 2:m->header.buffered=2;break;
        case 3:keys[0].as.string->header.colour=VM_GC_PURPLE;break;
        case 4:keys[0].as.string->header.buffered=1;break;
        default:keys[0].as.string->header.ref_count=0;break;
        }
        missing=val_int(81);present=true;
        CHECK(vm_rm_ordinary_get(&h,m,keys[0],&missing)==VM_RM_HEAP_INVALID&&missing.as.i64==81);
        CHECK(vm_rm_ordinary_has(&h,m,keys[0],&present)==VM_RM_HEAP_INVALID&&present);
        CHECK(vm_rm_ordinary_set(&h,m,keys[0],val_int(1))==VM_RM_HEAP_INVALID);
        m->header=mh;keys[0].as.string->header=kh;
        CHECK(vm_rm_ordinary_get(&h,m,keys[0],&missing)==VM_RM_HEAP_OK&&missing.as.i64==99);
    }
    h.private_record_dag=true;NanoValue out=val_int(71);bool found=false;
    VmHashMap *sentinel=m;
    CHECK(vm_rm_ordinary_new(&h,&sentinel)==VM_RM_HEAP_INVALID&&sentinel==m);
    CHECK(vm_rm_ordinary_set(&h,m,keys[0],val_int(1))==VM_RM_HEAP_INVALID);
    CHECK(vm_rm_ordinary_get(&h,m,keys[0],&out)==VM_RM_HEAP_INVALID&&out.as.i64==71);
    CHECK(vm_rm_ordinary_has(&h,m,keys[0],&found)==VM_RM_HEAP_INVALID&&!found);
    CHECK(vm_rm_ordinary_len(&h,m,&count)==VM_RM_HEAP_INVALID&&count==13);
    h.private_record_dag=false;
    for(unsigned i=0;i<13;i++)vm_release(&h,keys[i]);
    vm_release(&h,val_hashmap(m));vm_gc_collect_cycles(&h);
    CHECK(!h.stats.num_objects&&h.stats.allocated==h.stats.freed);
    vm_heap_destroy(&h);CHECK(!ra_live&&!ra_bytes);
}
static void constructor_prefixes(void) {
    for(unsigned persistent=0;persistent<2;persistent++)for(unsigned offset=0;offset<2;offset++) {
        VmHeap h;vm_heap_init(&h);CHECK(h.intern_buckets);
        VmHashMap *prior=NULL;CHECK(vm_rm_ordinary_new(&h,&prior)==VM_RM_HEAP_OK);
        VmHashMap *out=prior;size_t live=ra_live,bytes=ra_bytes;
        size_t allocated=h.stats.allocated,freed=h.stats.freed;
        ra_fail=ra_calls+offset;ra_persistent=(int)persistent;
        CHECK(vm_rm_ordinary_new(&h,&out)==VM_RM_HEAP_MEMORY&&out==prior);
        ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(ra_live==live&&ra_bytes==bytes&&h.stats.allocated==allocated&&h.stats.freed==freed);
        CHECK(vm_rm_ordinary_new(&h,&out)==VM_RM_HEAP_OK&&out!=prior);
        vm_release(&h,val_hashmap(out));vm_release(&h,val_hashmap(prior));
        vm_heap_destroy(&h);CHECK(!ra_live&&!ra_bytes);
    }
}
static void unbuffered_purple(void) {
    VmHeap h;vm_heap_init(&h);CHECK(h.intern_buckets&&!h.cycle_capacity);
    VmHashMap *m=NULL;CHECK(vm_rm_ordinary_new(&h,&m)==VM_RM_HEAP_OK);
    vm_retain(&h,val_hashmap(m));ra_fail=ra_calls;
    vm_release(&h,val_hashmap(m));ra_fail=SIZE_MAX;
    CHECK(m->header.ref_count==1&&!m->header.buffered&&m->header.colour==VM_GC_PURPLE);
    NanoValue key=val_string(vm_string_new(&h,"key",3));CHECK(key.as.string);
    CHECK(vm_rm_ordinary_set(&h,m,key,val_int(7))==VM_RM_HEAP_OK);
    NanoValue out=val_void();CHECK(vm_rm_ordinary_get(&h,m,key,&out)==VM_RM_HEAP_OK&&out.as.i64==7);
    vm_release(&h,key);vm_release(&h,val_hashmap(m));vm_heap_destroy(&h);
    CHECK(!ra_live&&!ra_bytes);
}
static void collisions(void) {
    VmHeap h;vm_heap_init(&h);CHECK(h.intern_buckets);
    VmHashMap *m=NULL;CHECK(vm_rm_ordinary_new(&h,&m)==VM_RM_HEAP_OK);
    unsigned kept=0;
    for(unsigned candidate=0;kept<20&&candidate<100000;candidate++) {
        char text[32];int n=snprintf(text,sizeof text,"collision-%u",candidate);
        NanoValue key=val_string(vm_string_new(&h,text,(uint32_t)n));CHECK(key.as.string);
        if(key.as.string->hash%128==0) {
            CHECK(vm_rm_ordinary_set(&h,m,key,val_int(kept))==VM_RM_HEAP_OK);kept++;
        }
        vm_release(&h,key);
    }
    CHECK(kept==20&&m->count==20);
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK(vm_rm_heap_import_validate(m,&w)==VM_RM_HEAP_OK);
    for(uint32_t i=0;i<m->bucket_count;i++)if(m->entries[i].state) {
        NanoValue out=val_void();
        CHECK(vm_rm_ordinary_get(&h,m,m->entries[i].key,&out)==VM_RM_HEAP_OK);
        CHECK(out.tag==TAG_INT&&out.as.i64==m->entries[i].value.as.i64);
    }
    vm_release(&h,val_hashmap(m));vm_gc_collect_cycles(&h);vm_heap_destroy(&h);
    CHECK(!ra_live&&!ra_bytes);
}
int main(void) {
    run(false);run(true);constructor_prefixes();unbuffered_purple();collisions();
    printf("I checked %u ordinary heap assertions; no VM dispatch or admission.\n",checks);
    return 0;
}
