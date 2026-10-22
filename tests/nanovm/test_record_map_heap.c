/* Direct private heap qualification, never selected VM execution. */
#include "../nanoisa/record_array_alloc.h"
#include "../../src/nanovm/record_map_heap_private.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"heap check %u: %s\n",__LINE__,#x);abort();}checks++; } while(0)
static unsigned checks;
int g_argc;char **g_argv;
extern bool test_shared_slot(const VmHashMap *,NanoValue,uint32_t *,bool *);
static void init(VmHeap *h) {
    vm_heap_init(h);CHECK(h->intern_buckets);h->private_record_dag=true;
}
static void finish(VmHeap *h) {
    CHECK(!h->stats.num_objects);CHECK(h->stats.allocated==h->stats.freed);
    vm_heap_destroy(h);CHECK(!ra_live&&!ra_bytes);
}
static VmHashMap *map(VmHeap *h) {
    VmHashMap *m=NULL;CHECK(vm_rm_heap_new(h,&m)==VM_RM_HEAP_OK);return m;
}
static NanoValue string(VmHeap *h,const char *s,uint32_t n) {
    VmString *p=vm_string_new(h,s,n);CHECK(p);return val_string(p);
}
static uint64_t validate(const VmHashMap *m) {
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK(vm_rm_heap_validate(m,&w)==VM_RM_HEAP_OK);
    for(uint32_t i=0;i<m->bucket_count;i++)if(m->entries[i].state) {
        uint32_t slot=UINT32_MAX;bool found=false;
        CHECK(test_shared_slot(m,m->entries[i].key,&slot,&found)&&found&&slot==i);
    }
    return w.used;
}
static void exact_work(const VmHashMap *m) {
    uint64_t cost=validate(m);size_t calls=ra_calls;
    VmRmObservationWork w={0,cost,false};CHECK(vm_rm_heap_validate(m,&w)==VM_RM_HEAP_OK&&w.used==cost);
    w=(VmRmObservationWork){0,cost-1,false};CHECK(vm_rm_heap_validate(m,&w)==VM_RM_HEAP_LIMIT&&w.limited);
    w=(VmRmObservationWork){UINT64_MAX,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK(vm_rm_heap_validate(m,&w)==VM_RM_HEAP_LIMIT&&w.used==UINT64_MAX);
    w=(VmRmObservationWork){0,UINT64_MAX,false};CHECK(vm_rm_heap_validate(m,&w)==VM_RM_HEAP_INVALID);
    CHECK(ra_calls==calls);
}
/* This allocation deliberately bypasses interning, with its own counted owner.
 * Unlink on release safely finds no table entry; no existing object is cloned
 * as an additional owner without corresponding allocation/stat accounting. */
static NanoValue independent_equal(VmHeap *h,NanoValue original) {
    VmString *a=original.as.string;size_t bytes=sizeof *a+a->length+1;
    VmString *b=ra_test_malloc(bytes);CHECK(b&&b!=a);memcpy(b,a,bytes);
    b->header.ref_count=1;b->header.buffered=0;b->intern_next=NULL;
    h->stats.allocated+=bytes;h->stats.num_objects++;h->stats.allocation_calls++;
    return val_string(b);
}
static void byte_keys(void) {
    VmHeap h;init(&h);VmHashMap *m=map(&h);
    NanoValue keys[]={string(&h,"",0),string(&h,"a\0x",3),string(&h,"a\0y",3),string(&h,"a",1)};
    for(unsigned i=0;i<4;i++)CHECK(vm_rm_heap_set(&h,m,keys[i],val_int(i))==VM_RM_HEAP_OK);
    for(unsigned i=0;i<4;i++) {
        NanoValue out=val_bool(false);bool found=false;
        CHECK(vm_rm_heap_get(m,keys[i],&out)==VM_RM_HEAP_OK&&out.tag==TAG_INT&&out.as.i64==i);
        CHECK(vm_rm_heap_has(m,keys[i],&found)==VM_RM_HEAP_OK&&found);
    }
    NanoValue absent=string(&h,"absent",6),out=val_int(0);bool found=true;uint32_t count=0;
    CHECK(vm_rm_heap_get(m,absent,&out)==VM_RM_HEAP_OK&&out.tag==TAG_VOID);
    CHECK(vm_rm_heap_has(m,absent,&found)==VM_RM_HEAP_OK&&!found);
    CHECK(vm_rm_heap_len(m,&count)==VM_RM_HEAP_OK&&count==4);exact_work(m);
    out=val_int(27);CHECK(vm_rm_heap_get(m,val_int(1),&out)==VM_RM_HEAP_INVALID&&out.as.i64==27);
    for(unsigned i=0;i<4;i++)vm_release(&h,keys[i]);
    vm_release(&h,absent);
    vm_release(&h,val_hashmap(m));finish(&h);
}
static void threshold(bool persistent) {
    VmHeap h;init(&h);VmHashMap *m=map(&h);NanoValue keys[13];
    for(unsigned i=0;i<13;i++){char s[24];int n=snprintf(s,sizeof s,"key-%u",i);keys[i]=string(&h,s,(uint32_t)n);}
    for(unsigned i=0;i<12;i++)CHECK(vm_rm_heap_set(&h,m,keys[i],val_int(i))==VM_RM_HEAP_OK);
    CHECK(m->count==12&&m->bucket_count==16);VmHMEntry *prior=m->entries;
    NanoValue duplicate=independent_equal(&h,keys[0]);uint32_t intern_count=h.intern_count;
    size_t calls=ra_calls;ra_fail=ra_calls;ra_persistent=persistent;
    duplicate.as.string->header.ref_count=UINT32_MAX;
    CHECK(vm_rm_heap_set(&h,m,duplicate,val_int(99))==VM_RM_HEAP_OK&&ra_calls==calls);
    CHECK(duplicate.as.string->header.ref_count==UINT32_MAX&&keys[0].as.string->header.ref_count==2);
    duplicate.as.string->header.ref_count=1;
    keys[12].as.string->header.ref_count=UINT32_MAX;
    CHECK(vm_rm_heap_set(&h,m,keys[12],val_int(12))==VM_RM_HEAP_RETAIN_OVERFLOW&&ra_calls==calls);
    keys[12].as.string->header.ref_count=1;
    size_t allocated=h.stats.allocated,freed=h.stats.freed,bytes=ra_bytes,live=ra_live;
    CHECK(vm_rm_heap_set(&h,m,keys[12],val_int(12))==VM_RM_HEAP_MEMORY);
    ra_fail=SIZE_MAX;ra_persistent=0;
    CHECK(m->entries==prior&&m->count==12&&m->bucket_count==16);
    CHECK(h.stats.allocated==allocated&&h.stats.freed==freed&&ra_bytes==bytes&&ra_live==live);
    for(unsigned i=0;i<12;i++){NanoValue out;CHECK(vm_rm_heap_get(m,keys[i],&out)==VM_RM_HEAP_OK&&out.as.i64==(i?i:99));}
    CHECK(keys[12].as.string->header.ref_count==1);
    CHECK(vm_rm_heap_set(&h,m,keys[12],val_int(12))==VM_RM_HEAP_OK);
    CHECK(m->bucket_count==32&&m->count==13&&h.stats.allocated==allocated+16*sizeof(VmHMEntry));
    CHECK(keys[12].as.string->header.ref_count==2);exact_work(m);
    vm_release(&h,duplicate);CHECK(h.intern_count==intern_count);
    for(unsigned i=0;i<13;i++)vm_release(&h,keys[i]);
    vm_release(&h,val_hashmap(m));finish(&h);
}
static void new_prefixes(void) {
    for(unsigned persistent=0;persistent<2;persistent++)for(unsigned offset=0;offset<2;offset++) {
        VmHeap h;init(&h);VmHashMap *prior=map(&h),*out=prior;
        size_t bytes=ra_bytes,live=ra_live,allocated=h.stats.allocated,freed=h.stats.freed;
        ra_fail=ra_calls+offset;ra_persistent=(int)persistent;
        CHECK(vm_rm_heap_new(&h,&out)==VM_RM_HEAP_MEMORY&&out==prior);
        ra_fail=SIZE_MAX;ra_persistent=0;
        CHECK(ra_bytes==bytes&&ra_live==live&&h.stats.allocated==allocated&&h.stats.freed==freed);
        CHECK(vm_rm_heap_new(&h,&out)==VM_RM_HEAP_OK&&out!=prior);
        vm_release(&h,val_hashmap(out));validate(prior);vm_release(&h,val_hashmap(prior));finish(&h);
    }
}
static void malformed(void) {
    VmHeap h;init(&h);VmHashMap *m=map(&h);NanoValue key=string(&h,"key",3);
    CHECK(vm_rm_heap_set(&h,m,key,val_int(7))==VM_RM_HEAP_OK);
    uint32_t slot=0;while(!m->entries[slot].state)slot++;
    VmHMEntry saved=m->entries[slot];uint32_t other=(slot+1)%m->bucket_count;
    for(unsigned variant=0;variant<9;variant++) {
        switch(variant) {
        case 0:m->entries[other]=saved;m->count++;break; /* equal duplicate */
        case 1:m->entries[other]=saved;memset(&m->entries[slot],0,sizeof saved);break; /* unreachable */
        case 2:m->entries[slot].state=2;break;
        case 3:m->count++;break;
        case 4:m->tombstone_count=1;break;
        case 5:m->val_type=TAG_BOOL;break;
        case 6:m->entries[slot].value=val_bool(true);break;
        case 7:key.as.string->hash++;break;
        case 8:key.as.string->header.obj_type=TAG_INT;break;
        }
        VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};size_t calls=ra_calls;
        CHECK(vm_rm_heap_validate(m,&w)==VM_RM_HEAP_INVALID&&ra_calls==calls);
        m->entries[slot]=saved;memset(&m->entries[other],0,sizeof saved);m->count=1;m->tombstone_count=0;m->val_type=TAG_INT;
        if(variant==7)key.as.string->hash--;
        key.as.string->header.obj_type=TAG_STRING;
        validate(m);
    }
    vm_release(&h,key);vm_release(&h,val_hashmap(m));finish(&h);
}
static void deep_shared(void) {
    VmHeap h;init(&h);NanoValue key=string(&h,"shared",6);VmHashMap *a=map(&h),*b=map(&h);
    CHECK(vm_rm_heap_set(&h,a,key,val_int(1))==VM_RM_HEAP_OK);
    CHECK(vm_rm_heap_set(&h,b,key,val_int(2))==VM_RM_HEAP_OK);
    CHECK(key.as.string->header.ref_count==3);vm_release(&h,key);
    NanoValue value=val_hashmap(a);
    for(unsigned i=0;i<256;i++){VmStruct *s=vm_struct_new(&h,i,1);CHECK(s);s->fields[0]=value;value=val_struct(s);}
    vm_retain(&h,value);size_t calls=ra_calls;vm_release(&h,value);CHECK(ra_calls==calls);
    vm_release(&h,value);CHECK(ra_calls==calls&&key.as.string->header.ref_count==1);
    vm_release(&h,val_hashmap(b));CHECK(ra_calls==calls);finish(&h);
}
static void growth_and_collisions(void) {
    VmHeap h;init(&h);VmHashMap *m=map(&h);NanoValue keys[64];
    unsigned accepted=0,trial=0;
    while(accepted<64) {
        char name[32];int n=snprintf(name,sizeof name,"collision-%u",trial++);
        NanoValue key=string(&h,name,(uint32_t)n);
        if(key.as.string->hash%128!=0){vm_release(&h,key);continue;}
        keys[accepted]=key;
        CHECK(vm_rm_heap_set(&h,m,key,val_int(accepted))==VM_RM_HEAP_OK);accepted++;
    }
    CHECK(m->bucket_count==128&&m->count==64);
    for(unsigned i=0;i<64;i++)CHECK(m->entries[i].state==1); /* real long probe chain */
    exact_work(m);
    for(unsigned i=0;i<64;i++){NanoValue out;CHECK(vm_rm_heap_get(m,keys[i],&out)==VM_RM_HEAP_OK&&out.as.i64==i);}
    size_t allocated=h.stats.allocated;
    for(unsigned i=0;i<64;i++)vm_release(&h,keys[i]);
    vm_release(&h,val_hashmap(m));
    CHECK(h.stats.freed==allocated);finish(&h);
}
static NanoValue record_chain(VmHeap *h,NanoValue value,unsigned depth,bool *complete) {
    *complete=false;
    for(unsigned i=0;i<depth;i++) {
        VmStruct *s=vm_struct_new(h,i,1);if(!s)return value;
        s->fields[0]=value;value=val_struct(s); /* consuming transfer */
    }
    *complete=true;return value;
}
static void partial_chain(void) {
    unsigned prefixes=0;
    for(unsigned persistent=0;persistent<2;persistent++)for(unsigned offset=0;offset<16;offset++) {
        VmHeap h;init(&h);VmHashMap *m=map(&h);NanoValue key=string(&h,"partial",7);
        CHECK(vm_rm_heap_set(&h,m,key,val_int(1))==VM_RM_HEAP_OK);vm_release(&h,key);
        ra_fail=ra_calls+offset;ra_persistent=(int)persistent;bool complete;
        NanoValue root=record_chain(&h,val_hashmap(m),8,&complete);CHECK(!complete);
        ra_fail=SIZE_MAX;ra_persistent=0;vm_release(&h,root);finish(&h);prefixes++;
        init(&h);m=map(&h);root=record_chain(&h,val_hashmap(m),8,&complete);CHECK(complete);
        vm_release(&h,root);finish(&h);
    }
    printf("I checked %u partial record/map construction prefixes with independent retry.\n",prefixes);
}
static void maximum_array_neighbor(void) {
    VmHeap h;init(&h);NanoValue value=string(&h,"terminal",8);
    for(unsigned i=0;i<3;i++) {
        VmArray *a=vm_array_new(&h,i?TAG_ARRAY:TAG_STRING,1);CHECK(a);
        CHECK(vm_array_push(&h,a,value));vm_release(&h,value);value=val_array(a);
    }
    bool complete;value=record_chain(&h,value,256,&complete);CHECK(complete);
    size_t calls=ra_calls;vm_release(&h,value);CHECK(ra_calls==calls);finish(&h);
}
static void legacy_stats_and_cycle(void) {
    VmHeap h;init(&h);h.private_record_dag=false;
    VmHashMap *m=vm_hashmap_new(&h,TAG_STRING,TAG_INT);CHECK(m);
    for(unsigned i=0;i<64;i++) {
        char text[32];int n=snprintf(text,sizeof text,"key-%u",i);
        NanoValue key=string(&h,text,(uint32_t)n);
        CHECK(vm_hashmap_set(&h,m,key,val_int(i)));vm_release(&h,key);
    }
    CHECK(m->bucket_count==128&&m->count==64);
    printf("I measured corrected legacy growth allocated=%zu; ",h.stats.allocated);
    vm_release(&h,val_hashmap(m));vm_gc_collect_cycles(&h);
    printf("freed=%zu, objects=%zu.\n",h.stats.freed,h.stats.num_objects);
    CHECK(h.stats.allocated==h.stats.freed);finish(&h);
    init(&h);h.private_record_dag=false;
    m=vm_hashmap_new(&h,TAG_INT,TAG_HASHMAP);CHECK(m);
    CHECK(vm_hashmap_set(&h,m,val_int(1),val_hashmap(m)));
    vm_release(&h,val_hashmap(m));CHECK(h.stats.num_objects==1);
    CHECK(vm_gc_collect_cycles(&h)==1);finish(&h);
}
int main(void) {
    byte_keys();new_prefixes();threshold(false);threshold(true);malformed();deep_shared();
    growth_and_collisions();partial_chain();maximum_array_neighbor();legacy_stats_and_cycle();
    printf("I checked %u private heap assertions; no VM dispatch or admission. Heap size %zu.\n",checks,sizeof(VmHeap));
    return 0;
}
