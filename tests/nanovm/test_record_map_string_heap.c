/* Fixed STRING-value physical ownership only, without VM enrollment. */
#include "../nanoisa/record_array_alloc.h"
#include "../../src/nanovm/record_map_heap_private.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"string heap check %u: %s\n",__LINE__,#x);abort();}checks++; } while(0)
static unsigned checks;
int g_argc;char **g_argv;
static bool ordinary;
static VmRmHeapStatus make(VmHeap *h,VmHashMap **m) {
    return ordinary?vm_rm_ordinary_ss_new(h,m):vm_rm_heap_ss_new(h,m);
}
static VmRmHeapStatus set(VmHeap *h,VmHashMap *m,NanoValue k,NanoValue v) {
    return ordinary?vm_rm_ordinary_ss_set(h,m,k,v):vm_rm_heap_ss_set(h,m,k,v);
}
static VmRmHeapStatus get(VmHeap *h,VmHashMap *m,NanoValue k,NanoValue *v) {
    return ordinary?vm_rm_ordinary_ss_get_owned(h,m,k,v):vm_rm_heap_ss_get_owned(h,m,k,v);
}
static VmRmHeapStatus del(VmHeap *h,VmHashMap *m,NanoValue k) {
    return ordinary?vm_rm_ordinary_ss_delete(h,m,k):vm_rm_heap_ss_delete(h,m,k);
}
static VmRmHeapStatus has(VmHeap *h,VmHashMap *m,NanoValue k,bool *v) {
    return ordinary?vm_rm_ordinary_ss_has(h,m,k,v):vm_rm_heap_ss_has(m,k,v);
}
static VmRmHeapStatus validate(VmHashMap *m,VmRmObservationWork *w) {
    return ordinary?vm_rm_heap_ss_import_validate(m,w):vm_rm_heap_ss_validate(m,w);
}
static NanoValue string(VmHeap *h,const char *s,uint32_t n) {
    VmString *v=vm_string_new(h,s,n);CHECK(v);return val_string(v);
}
static void init(VmHeap *h) {
    vm_heap_init(h);CHECK(h->intern_buckets);h->private_record_dag=!ordinary;
}
static void finish(VmHeap *h) {
    if(ordinary)vm_gc_collect_cycles(h);
    vm_heap_destroy(h);CHECK(!ra_live&&!ra_bytes);
}
static void ownership(void) {
    VmHeap h;init(&h);VmHashMap *m=NULL;CHECK(make(&h,&m)==VM_RM_HEAP_OK);
    NanoValue key=string(&h,"k\0x",3),v=string(&h,"v\0y",3),out=val_int(71);
    CHECK(set(&h,m,key,v)==VM_RM_HEAP_OK);
    CHECK(key.as.string->header.ref_count==2&&v.as.string->header.ref_count==2);
    if(ordinary) {
        vm_retain(&h,val_hashmap(m));vm_release(&h,val_hashmap(m));
        CHECK(m->header.buffered&&m->header.colour==VM_GC_PURPLE);
    }
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};
    CHECK(validate(m,&w)==VM_RM_HEAP_OK);uint64_t exact=w.used;
    w=(VmRmObservationWork){0,exact,false};CHECK(validate(m,&w)==VM_RM_HEAP_OK&&w.used==exact);
    w=(VmRmObservationWork){0,exact-1,false};CHECK(validate(m,&w)==VM_RM_HEAP_LIMIT&&w.limited);
    uint32_t refs=v.as.string->header.ref_count;
    v.as.string->header.ref_count=UINT32_MAX;
    CHECK(get(&h,m,key,&out)==VM_RM_HEAP_RETAIN_OVERFLOW&&out.tag==TAG_INT&&out.as.i64==71);
    CHECK(v.as.string->header.ref_count==UINT32_MAX);
    CHECK(set(&h,m,key,v)==VM_RM_HEAP_RETAIN_OVERFLOW);
    v.as.string->header.ref_count=refs;
    CHECK(get(&h,m,key,&out)==VM_RM_HEAP_OK&&out.tag==TAG_STRING&&out.as.string==v.as.string);
    CHECK(v.as.string->header.ref_count==refs+1);vm_release(&h,out);
    /* Independent equal-content query key, not intern deduplication. */
    size_t size=sizeof(VmString)+key.as.string->length+1;
    VmString *equal=ra_test_malloc(size);CHECK(equal);memcpy(equal,key.as.string,size);
    equal->header.ref_count=UINT32_MAX;equal->intern_next=NULL;
    h.stats.allocated+=size;h.stats.num_objects++;h.stats.allocation_calls++;
    CHECK(set(&h,m,val_string(equal),v)==VM_RM_HEAP_OK);
    CHECK(equal->header.ref_count==UINT32_MAX&&key.as.string->header.ref_count==2);
    equal->header.ref_count=1;vm_release(&h,val_string(equal));
    /* Value corruption is charged and refused without changing the graph. */
    uint32_t hash=v.as.string->hash;v.as.string->hash^=1;
    w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};CHECK(validate(m,&w)==VM_RM_HEAP_INVALID);
    v.as.string->hash=hash;
    char terminal=v.as.string->data[v.as.string->length];v.as.string->data[v.as.string->length]='x';
    w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};CHECK(validate(m,&w)==VM_RM_HEAP_INVALID);
    v.as.string->data[v.as.string->length]=terminal;
    /* Drop caller value, obtain one owned edge, then destroy last map owner. */
    vm_release(&h,v);CHECK(get(&h,m,key,&out)==VM_RM_HEAP_OK);
    vm_release(&h,val_hashmap(m));if(ordinary)vm_gc_collect_cycles(&h);
    CHECK(out.tag==TAG_STRING&&out.as.string->length==3&&!memcmp(out.as.string->data,"v\0y",3));
    vm_release(&h,key);vm_release(&h,out);finish(&h);
}
static void same_edges(void) {
    VmHeap h;init(&h);VmHashMap *m=NULL;CHECK(make(&h,&m)==VM_RM_HEAP_OK);
    NanoValue s=string(&h,"same",4),out=val_int(1);
    s.as.string->header.ref_count=UINT32_MAX-1;size_t calls=ra_calls;
    CHECK(set(&h,m,s,s)==VM_RM_HEAP_RETAIN_OVERFLOW&&ra_calls==calls&&m->count==0);
    s.as.string->header.ref_count=UINT32_MAX-2;
    CHECK(set(&h,m,s,s)==VM_RM_HEAP_OK&&s.as.string->header.ref_count==UINT32_MAX);
    CHECK(get(&h,m,s,&out)==VM_RM_HEAP_RETAIN_OVERFLOW&&out.tag==TAG_INT);
    s.as.string->header.ref_count=3;
    CHECK(set(&h,m,s,s)==VM_RM_HEAP_OK&&s.as.string->header.ref_count==3);
    CHECK(get(&h,m,s,&out)==VM_RM_HEAP_OK&&out.as.string==s.as.string&&s.as.string->header.ref_count==4);
    CHECK(del(&h,m,s)==VM_RM_HEAP_OK&&s.as.string->header.ref_count==2&&m->tombstone_count==1);
    CHECK(del(&h,m,s)==VM_RM_HEAP_OK&&s.as.string->header.ref_count==2);
    bool present=true;NanoValue absent=val_int(3);
    CHECK(has(&h,m,s,&present)==VM_RM_HEAP_OK&&!present);
    CHECK(get(&h,m,s,&absent)==VM_RM_HEAP_OK&&absent.tag==TAG_VOID);
    CHECK(set(&h,m,s,s)==VM_RM_HEAP_OK&&s.as.string->header.ref_count==4);
    vm_release(&h,s);vm_release(&h,out); /* map owns the last two edges */
    vm_release(&h,val_hashmap(m));finish(&h);
}
static void allocation(bool persistent) {
    for(unsigned offset=0;offset<2;offset++) {
        VmHeap h;init(&h);VmHashMap *m=(VmHashMap *)(uintptr_t)1;
        size_t live=ra_live,bytes=ra_bytes;ra_fail=ra_calls+offset;ra_persistent=persistent;
        CHECK(make(&h,&m)==VM_RM_HEAP_MEMORY&&m==(VmHashMap *)(uintptr_t)1);
        ra_fail=SIZE_MAX;ra_persistent=0;CHECK(ra_live==live&&ra_bytes==bytes);
        CHECK(make(&h,&m)==VM_RM_HEAP_OK);vm_release(&h,val_hashmap(m));finish(&h);
    }
    VmHeap h;init(&h);VmHashMap *m=NULL;CHECK(make(&h,&m)==VM_RM_HEAP_OK);
    NanoValue keys[13],value=string(&h,"shared",6);
    for(unsigned i=0;i<13;i++) { char k[2]={'k',(char)i};keys[i]=string(&h,k,2); }
    for(unsigned i=0;i<12;i++)CHECK(set(&h,m,keys[i],value)==VM_RM_HEAP_OK);
    CHECK(del(&h,m,keys[0])==VM_RM_HEAP_OK);
    size_t live=ra_live,bytes=ra_bytes;VmHMEntry *table=m->entries;uint32_t refs=value.as.string->header.ref_count;
    ra_fail=ra_calls;ra_persistent=persistent;
    CHECK(set(&h,m,keys[12],value)==VM_RM_HEAP_MEMORY);
    ra_fail=SIZE_MAX;ra_persistent=0;
    CHECK(m->entries==table&&m->count==11&&m->tombstone_count==1&&value.as.string->header.ref_count==refs);
    CHECK(ra_live==live&&ra_bytes==bytes);
    CHECK(set(&h,m,keys[12],value)==VM_RM_HEAP_OK);
    for(unsigned i=0;i<13;i++)vm_release(&h,keys[i]);
    vm_release(&h,value);vm_release(&h,val_hashmap(m));finish(&h);
}
/* Independent physical fixture: populate all 16 valid slots without changing
 * the production setter's load-factor policy, then retire every stored edge. */
extern bool test_shared_slot(const VmHashMap *,NanoValue,uint32_t *,bool *);
static void full_tables(void) {
    VmHeap h;init(&h);VmHashMap *m=NULL;CHECK(make(&h,&m)==VM_RM_HEAP_OK);
    NanoValue keys[16],value=string(&h,"full-value",10),missing=string(&h,"not-present",11);
    for(unsigned i=0;i<16;i++) {
        char bytes[2]={'f',(char)i};keys[i]=string(&h,bytes,2);
        uint32_t slot;bool found;CHECK(test_shared_slot(m,keys[i],&slot,&found)&&!found);
        vm_retain(&h,keys[i]);vm_retain(&h,value);
        m->entries[slot]=(VmHMEntry){.key=keys[i],.value=value,.state=1};m->count++;
    }
    CHECK(m->bucket_count==16&&m->count==16);
    VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};CHECK(validate(m,&w)==VM_RM_HEAP_OK);
    NanoValue out=val_int(9);bool present=true;
    CHECK(get(&h,m,missing,&out)==VM_RM_HEAP_OK&&out.tag==TAG_VOID);
    CHECK(has(&h,m,missing,&present)==VM_RM_HEAP_OK&&!present);
    CHECK(del(&h,m,missing)==VM_RM_HEAP_OK&&m->count==16);
    CHECK(set(&h,m,missing,value)==VM_RM_HEAP_INVALID&&m->count==16);
    for(unsigned i=0;i<16;i++)CHECK(del(&h,m,keys[i])==VM_RM_HEAP_OK);
    CHECK(m->count==0&&m->tombstone_count==16);
    w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};CHECK(validate(m,&w)==VM_RM_HEAP_OK);
    uint64_t exact=w.used;w=(VmRmObservationWork){0,exact-1,false};CHECK(validate(m,&w)==VM_RM_HEAP_LIMIT);
    out=val_int(9);present=true;
    CHECK(get(&h,m,missing,&out)==VM_RM_HEAP_OK&&out.tag==TAG_VOID);
    CHECK(has(&h,m,missing,&present)==VM_RM_HEAP_OK&&!present);
    CHECK(del(&h,m,missing)==VM_RM_HEAP_OK);
    CHECK(set(&h,m,missing,value)==VM_RM_HEAP_OK&&m->count==1&&m->tombstone_count==0);
    for(unsigned i=0;i<16;i++)vm_release(&h,keys[i]);
    vm_release(&h,value);vm_release(&h,missing);vm_release(&h,val_hashmap(m));finish(&h);
}
static void collisions_and_corruption(void) {
    VmHeap h;init(&h);VmHashMap *m=NULL;CHECK(make(&h,&m)==VM_RM_HEAP_OK);
    NanoValue keys[3],v=string(&h,"collision-value",15);unsigned n=0;
    for(uint32_t candidate=0;n<3&&candidate<4096;candidate++) {
        char bytes[4]={(char)candidate,(char)(candidate>>8),'c',0};
        NanoValue k=string(&h,bytes,4);
        if(k.as.string->hash%16==15)keys[n++]=k;else vm_release(&h,k);
    }
    CHECK(n==3);
    for(unsigned i=0;i<3;i++)CHECK(set(&h,m,keys[i],v)==VM_RM_HEAP_OK);
    uint32_t slots[3];bool found;
    for(unsigned i=0;i<3;i++)CHECK(test_shared_slot(m,keys[i],&slots[i],&found)&&found);
    CHECK(slots[0]==15&&slots[1]==0&&slots[2]==1);
    CHECK(del(&h,m,keys[0])==VM_RM_HEAP_OK&&m->tombstone_count==1);
    for(unsigned bad=0;bad<10;bad++) {
        VmHeapHeader mh=m->header,vh=v.as.string->header;
        uint32_t count=m->count,tombs=m->tombstone_count;
        VmHMEntry saved=m->entries[slots[1]],tomb=m->entries[slots[0]];
        switch(bad) {
        case 0:m->count=m->bucket_count+1;break;
        case 1:m->tombstone_count=m->bucket_count;break;
        case 2:m->header.ref_count=0;break;
        case 3:m->header.buffered=2;break;
        case 4:v.as.string->header.colour=VM_GC_PURPLE;break;
        case 5:v.as.string->header.buffered=1;break;
        case 6:v.as.string->header.ref_count=0;break;
        case 7:m->entries[slots[1]].key=keys[2];break; /* duplicate after tomb */
        case 8:m->entries[slots[0]].value=v;break; /* tomb owns no edges */
        default:m->entries[slots[1]].state=0;m->entries[slots[1]].key=val_void();m->entries[slots[1]].value=val_void();break;
        }
        VmRmObservationWork w={0,VM_RM_OBSERVATION_WORK_MAX,false};
        CHECK(validate(m,&w)==VM_RM_HEAP_INVALID);
        /* Restore all ownership counters and payload before any release. */
        m->header=mh;v.as.string->header=vh;m->count=count;m->tombstone_count=tombs;
        m->entries[slots[1]]=saved;m->entries[slots[0]]=tomb;
        w=(VmRmObservationWork){0,VM_RM_OBSERVATION_WORK_MAX,false};CHECK(validate(m,&w)==VM_RM_HEAP_OK);
        NanoValue out=val_int(99);CHECK(get(&h,m,keys[2],&out)==VM_RM_HEAP_OK&&out.as.string==v.as.string);vm_release(&h,out);
    }
    if(ordinary) {
        VmHeapHeader saved=m->header;
        for(unsigned bad=0;bad<2;bad++) {
            m->header.colour=bad?VM_GC_BLACK:VM_GC_GRAY;m->header.buffered=bad?1:0;
            NanoValue out=val_int(99);uint32_t refs=v.as.string->header.ref_count;
            CHECK(get(&h,m,keys[2],&out)==VM_RM_HEAP_INVALID&&out.tag==TAG_INT&&out.as.i64==99);
            CHECK(v.as.string->header.ref_count==refs);
            m->header=saved;
        }
    }
    CHECK(set(&h,m,keys[0],v)==VM_RM_HEAP_OK&&m->tombstone_count==0);
    CHECK(test_shared_slot(m,keys[0],&slots[0],&found)&&found&&slots[0]==15);
    for(unsigned i=0;i<3;i++)vm_release(&h,keys[i]);
    vm_release(&h,v);vm_release(&h,val_hashmap(m));finish(&h);
}
int main(void) {
    for(unsigned mode=0;mode<2;mode++) {
        ordinary=mode!=0;ownership();same_edges();full_tables();collisions_and_corruption();allocation(false);allocation(true);
    }
    printf("I checked %u STRING-value heap assertions; no VM dispatch or admission.\n",checks);
    return 0;
}
