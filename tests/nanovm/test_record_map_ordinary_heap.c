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
static void deletion_bounds(VmHashMap *m,bool private_heap,NanoValue key) {
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};size_t calls=ra_calls;
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_OK);
    uint64_t exact=w.used;CHECK(exact>0&&ra_calls==calls);
    w=(VmRmObservationWork){0,exact,false};
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_OK&&w.used==exact);
    w=(VmRmObservationWork){0,exact-1,false};
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_LIMIT&&w.limited);
    w=(VmRmObservationWork){0,exact,false};
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_OK);
    if(private_heap) {
        NanoValue out=val_int(71);const VmString *k=key.as.string;
        w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
        CHECK(vm_rm_heap_observe_lookup(m,k->data,k->length,&w,&out)==VM_RM_HEAP_OK);
        NanoValue expected=out;exact=w.used;
        w=(VmRmObservationWork){0,exact,false};out=val_int(71);
        CHECK(vm_rm_heap_observe_lookup(m,k->data,k->length,&w,&out)==VM_RM_HEAP_OK&&w.used==exact);
        CHECK(val_equal(out,expected));
        w=(VmRmObservationWork){0,exact-1,false};out=val_int(71);
        CHECK(vm_rm_heap_observe_lookup(m,k->data,k->length,&w,&out)==VM_RM_HEAP_LIMIT&&w.limited);
        CHECK(out.tag==TAG_INT&&out.as.i64==71);
        w=(VmRmObservationWork){0,exact,false};
        CHECK(vm_rm_heap_observe_lookup(m,k->data,k->length,&w,&out)==VM_RM_HEAP_OK&&val_equal(out,expected));
    }
    CHECK(ra_calls==calls);
}
/* Physical direct controls, not selected opcode execution. */
static void deletion(bool private_heap) {
    VmHeap h;vm_heap_init(&h);CHECK(h.intern_buckets);
    h.private_record_dag=private_heap;
    VmHashMap *m=NULL;
    CHECK((private_heap?vm_rm_heap_new(&h,&m):vm_rm_ordinary_new(&h,&m))==VM_RM_HEAP_OK);
    NanoValue keys[17];unsigned kept=0;
    /* All keys start at the last bucket: actual wraparound collision chain. */
    for(unsigned i=0;kept<17;i++) {
        char text[32];int n=snprintf(text,sizeof text,"delete-%u",i);
        text[n++]=0;text[n++]='x'; /* Counted-NUL key domain is VM-only. */
        VmString *key=vm_string_new(&h,text,(uint32_t)n);CHECK(key);
        if(key->hash%16==15)keys[kept++]=val_string(key);
        else vm_release(&h,val_string(key));
    }
    /* Build a fully occupied legal table without invoking growth. Every edge
     * has its actual retain; full charged validation establishes topology. */
    for(unsigned i=0;i<16;i++) {
        uint32_t at=keys[i].as.string->hash%m->bucket_count;
        while(m->entries[at].state)at=(at+1)%m->bucket_count;
        vm_retain(&h,keys[i]);m->entries[at].key=keys[i];
        m->entries[at].value=val_int(i);m->entries[at].state=1;m->count++;
    }
    if(!private_heap){vm_retain(&h,val_hashmap(m));vm_release(&h,val_hashmap(m));CHECK(m->header.buffered);}
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_OK);
    deletion_bounds(m,private_heap,keys[0]);
    size_t calls=ra_calls;
    CHECK((private_heap?vm_rm_heap_delete(&h,m,keys[16]):vm_rm_ordinary_delete(&h,m,keys[16]))==VM_RM_HEAP_OK);
    CHECK(m->count==16&&!m->tombstone_count&&ra_calls==calls);
    for(unsigned i=0;i<16;i++) {
        CHECK((private_heap?vm_rm_heap_delete(&h,m,keys[i]):vm_rm_ordinary_delete(&h,m,keys[i]))==VM_RM_HEAP_OK);
        CHECK(keys[i].as.string->header.ref_count==1&&m->count==15-i&&m->tombstone_count==i+1);
        CHECK(ra_calls==calls);
        w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
        CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_OK);
        for(unsigned j=i+1;j<16;j++)CHECK(vm_hashmap_has(m,keys[j]));
        if(i==0) {
            deletion_bounds(m,private_heap,keys[0]);
            deletion_bounds(m,private_heap,keys[15]);
            /* Duplicate equal live keys after a tombstone must not pass. */
            VmHMEntry saved=m->entries[1];m->entries[1]=m->entries[0];
            w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
            CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_INVALID);
            m->entries[1]=saved;
        }
    }
    CHECK((private_heap?vm_rm_heap_delete(&h,m,keys[16]):vm_rm_ordinary_delete(&h,m,keys[16]))==VM_RM_HEAP_OK);
    CHECK(!m->count&&m->tombstone_count==16&&ra_calls==calls);
    deletion_bounds(m,private_heap,keys[16]);
    NanoValue missing=val_int(7);
    CHECK((private_heap?vm_rm_heap_get(m,keys[16],&missing):vm_rm_ordinary_get(&h,m,keys[16],&missing))==VM_RM_HEAP_OK&&missing.tag==TAG_VOID);
    /* A subsequent growth failure preserves the already committed deletion. */
    VmHMEntry *entries=m->entries;size_t live=ra_live,bytes=ra_bytes;
    for(unsigned persistent=0;persistent<2;persistent++) {
        ra_fail=ra_calls;ra_persistent=persistent;
        CHECK((private_heap?vm_rm_heap_set(&h,m,keys[0],val_int(9)):vm_rm_ordinary_set(&h,m,keys[0],val_int(9)))==VM_RM_HEAP_MEMORY);
        ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(m->entries==entries&&!m->count&&m->tombstone_count==16&&ra_live==live&&ra_bytes==bytes);
    }
    CHECK((private_heap?vm_rm_heap_set(&h,m,keys[0],val_int(9)):vm_rm_ordinary_set(&h,m,keys[0],val_int(9)))==VM_RM_HEAP_OK);
    CHECK(m->count==1&&!m->tombstone_count&&m->bucket_count==32);
    CHECK((private_heap?vm_rm_heap_delete(&h,m,keys[0]):vm_rm_ordinary_delete(&h,m,keys[0]))==VM_RM_HEAP_OK);
    CHECK(m->tombstone_count==1);
    /* Sparse reinsert reuses the tombstone without growth or extra allocation. */
    calls=ra_calls;
    CHECK((private_heap?vm_rm_heap_set(&h,m,keys[0],val_int(11)):vm_rm_ordinary_set(&h,m,keys[0],val_int(11)))==VM_RM_HEAP_OK);
    CHECK(m->count==1&&!m->tombstone_count&&ra_calls==calls);
    VmString *stored=keys[0].as.string;size_t key_bytes=sizeof *stored+stored->length+1;
    VmString *equal=ra_test_malloc(key_bytes);CHECK(equal);memcpy(equal,stored,key_bytes);
    equal->header.ref_count=UINT32_MAX;equal->intern_next=NULL;
    h.stats.allocated+=key_bytes;h.stats.num_objects++;h.stats.allocation_calls++;
    calls=ra_calls;
    CHECK((private_heap?vm_rm_heap_delete(&h,m,val_string(equal)):vm_rm_ordinary_delete(&h,m,val_string(equal)))==VM_RM_HEAP_OK);
    CHECK(ra_calls==calls&&stored->header.ref_count==1&&equal->header.ref_count==UINT32_MAX);
    equal->header.ref_count=1;vm_release(&h,val_string(equal));
    CHECK((private_heap?vm_rm_heap_delete(&h,m,keys[0]):vm_rm_ordinary_delete(&h,m,keys[0]))==VM_RM_HEAP_OK);
    uint32_t at=keys[0].as.string->hash%m->bucket_count;CHECK(m->entries[at].state==2);
    m->entries[at].value=val_int(7);w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_INVALID);
    m->entries[at].value=val_void();m->tombstone_count=0;
    w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK((private_heap?vm_rm_heap_validate(m,&w):vm_rm_heap_import_validate(m,&w))==VM_RM_HEAP_INVALID);
    m->tombstone_count=1;
    for(unsigned i=0;i<17;i++)vm_release(&h,keys[i]);
    vm_release(&h,val_hashmap(m));if(!private_heap)vm_gc_collect_cycles(&h);
    vm_heap_destroy(&h);CHECK(!ra_live&&!ra_bytes);
}
int main(void) {
    run(false);run(true);constructor_prefixes();unbuffered_purple();collisions();
    deletion(false);deletion(true);
    printf("I checked %u ordinary heap assertions; no VM dispatch or admission.\n",checks);
    return 0;
}
