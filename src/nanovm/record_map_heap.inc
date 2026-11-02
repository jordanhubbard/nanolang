/* Included by heap.c only, after the shared hash/probe/resize helpers. */
static bool vm_rm_map_header(const VmHashMap *m) {
    return m && m->header.obj_type==TAG_HASHMAP && m->header.ref_count && !m->header.buffered &&
        m->key_type==TAG_STRING && m->val_type==TAG_INT && m->entries &&
        m->bucket_count>=HM_INITIAL_BUCKETS && !(m->bucket_count&(m->bucket_count-1)) &&
        sizeof(VmHMEntry)<=SIZE_MAX/(size_t)m->bucket_count &&
        m->count<=m->bucket_count && m->tombstone_count<=m->bucket_count-m->count;
}
static bool vm_rm_string(NanoValue key) {
    return key.tag==TAG_STRING && key.as.string && key.as.string->header.obj_type==TAG_STRING &&
        key.as.string->header.ref_count && !key.as.string->header.buffered &&
        (uint64_t)key.as.string->length+sizeof(VmString)+1<=SIZE_MAX;
}
/* Borrowed rooted inputs and established topology are mandatory. A full live
 * table has no insertion slot; an absent DELETE is nevertheless a no-op. */
static VmRmHeapStatus vm_rm_delete_validated(VmHeap *heap,VmHashMap *m,NanoValue key) {
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found) || !found)return VM_RM_HEAP_OK;
    VmHMEntry *entry=&m->entries[slot];
    if(entry->state!=1 || !vm_rm_string(entry->key) || entry->value.tag!=TAG_INT || !m->count)
        return VM_RM_HEAP_INVALID;
    NanoValue stored=entry->key;
    entry->key=val_void();entry->value=val_void();entry->state=2;
    m->count--;m->tombstone_count++;
    /* No key reads follow release. Caller still owns the query key edge. */
    vm_release(heap,stored);return VM_RM_HEAP_OK;
}
#ifdef NANO_RECORD_MAP_PRIVATE_HEAP
VmRmHeapStatus vm_rm_heap_new(VmHeap *heap,VmHashMap **out) {
    if(!heap || !out || heap->gc_running)return VM_RM_HEAP_INVALID;
    VmHashMap *m=vm_hashmap_new(heap,TAG_STRING,TAG_INT);
    if(!m)return VM_RM_HEAP_MEMORY;
    *out=m;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_set(VmHeap *heap,VmHashMap *m,NanoValue key,NanoValue value) {
    if(!heap || heap->gc_running || !vm_rm_map_header(m) || !vm_rm_string(key) || value.tag!=TAG_INT)
        return VM_RM_HEAP_INVALID;
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    if(found) {
        if(m->entries[slot].state!=1 || m->entries[slot].value.tag!=TAG_INT)return VM_RM_HEAP_INVALID;
        /* Replacement borrows even a maximum-refcount key. Both values are
         * scalars, so there is no retain/release or allocation to perform. */
        m->entries[slot].value=value;return VM_RM_HEAP_OK;
    }
    if(key.as.string->header.ref_count==UINT32_MAX)return VM_RM_HEAP_RETAIN_OVERFLOW;
    if(m->entries[slot].state!=0 &&
       (m->entries[slot].state!=2 || !m->tombstone_count ||
        m->entries[slot].key.tag!=TAG_VOID || m->entries[slot].value.tag!=TAG_VOID))return VM_RM_HEAP_INVALID;
    uint64_t occupied=(uint64_t)m->count+m->tombstone_count+1;
    if((double)occupied/(double)m->bucket_count>HM_LOAD_FACTOR) {
        if(m->bucket_count>UINT32_MAX/2 || !hm_resize(heap,m,m->bucket_count*2))return VM_RM_HEAP_MEMORY;
        /* Rehash invalidates slot; topology induction guarantees a new empty
         * slot. An impossible violation is not mislabeled allocator failure. */
        if(!hm_find_slot(m,key,&slot,&found) || found || m->entries[slot].state!=0)
            return VM_RM_HEAP_INVALID;
    }
    /* Nothing fallible follows this exact one-edge retain. */
    vm_retain(heap,key);
    if(m->entries[slot].state==2)m->tombstone_count--;
    m->entries[slot].key=key;m->entries[slot].value=value;
    m->entries[slot].state=1;m->count++;
    return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_delete(VmHeap *heap,VmHashMap *m,NanoValue key) {
    if(!heap || heap->gc_running || !vm_rm_map_header(m) || !vm_rm_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_delete_validated(heap,m,key);
}
VmRmHeapStatus vm_rm_heap_get(const VmHashMap *m,NanoValue key,NanoValue *out) {
    if(!out || !vm_rm_map_header(m) || !vm_rm_string(key))return VM_RM_HEAP_INVALID;
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    NanoValue result=val_void();
    if(found) {
        if(m->entries[slot].state!=1 || m->entries[slot].value.tag!=TAG_INT)return VM_RM_HEAP_INVALID;
        result=m->entries[slot].value;
    }
    *out=result;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_has(const VmHashMap *m,NanoValue key,bool *out) {
    if(!out || !vm_rm_map_header(m) || !vm_rm_string(key))return VM_RM_HEAP_INVALID;
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    if(found && (m->entries[slot].state!=1 || m->entries[slot].value.tag!=TAG_INT))return VM_RM_HEAP_INVALID;
    *out=found;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_len(const VmHashMap *m,uint32_t *out) {
    if(!out || !vm_rm_map_header(m))return VM_RM_HEAP_INVALID;
    *out=m->count;return VM_RM_HEAP_OK;
}
#endif
static bool vm_rm_observation_charge(VmRmObservationWork *w,uint64_t amount) {
    if(w->limited || w->used>w->limit || amount>w->limit-w->used) {
        w->limited=true;return false;
    }
    w->used+=amount;return true;
}
/* Internal ordinary header variant is physical validation, not imported-object
 * authority. The module-taking import query separately checks GC membership. */
static bool vm_rm_import_header(const VmHashMap *m) {
    return m && m->header.obj_type==TAG_HASHMAP && m->header.ref_count && m->header.buffered<=1 &&
        m->key_type==TAG_STRING && m->val_type==TAG_INT && m->entries &&
        m->bucket_count>=HM_INITIAL_BUCKETS && !(m->bucket_count&(m->bucket_count-1)) &&
        sizeof(VmHMEntry)<=SIZE_MAX/(size_t)m->bucket_count &&
        m->count<=m->bucket_count && m->tombstone_count<=m->bucket_count-m->count;
}
/* Internal adapters for a closed ordinary invocation. Complete topology and
 * collector membership must already have been validated for imported objects;
 * these physical guards do not confer execution authority. */
static bool vm_rm_ordinary_header(const VmHashMap *m) {
    return vm_rm_import_header(m) &&
        (m->header.colour==VM_GC_BLACK || m->header.colour==VM_GC_PURPLE) &&
        (!m->header.buffered || m->header.colour==VM_GC_PURPLE);
}
static bool vm_rm_ordinary_string(NanoValue key) {
    return vm_rm_string(key) && key.as.string->header.colour==VM_GC_BLACK;
}
VmRmHeapStatus vm_rm_ordinary_new(VmHeap *heap,VmHashMap **out) {
    if(!heap || !out || heap->gc_running || vm_rm_private_dag(heap))return VM_RM_HEAP_INVALID;
    VmHashMap *m=vm_hashmap_new(heap,TAG_STRING,TAG_INT);
    if(!m)return VM_RM_HEAP_MEMORY;
    *out=m;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_ordinary_set(VmHeap *heap,VmHashMap *m,NanoValue key,NanoValue value) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !vm_rm_ordinary_header(m) || !vm_rm_ordinary_string(key) || value.tag!=TAG_INT)
        return VM_RM_HEAP_INVALID;
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    if(found) {
        if(m->entries[slot].state!=1 || m->entries[slot].value.tag!=TAG_INT)return VM_RM_HEAP_INVALID;
        /* Replacement borrows even a maximum-refcount key. Both values are
         * scalars, so there is no retain/release or allocation to perform. */
        m->entries[slot].value=value;return VM_RM_HEAP_OK;
    }
    if(key.as.string->header.ref_count==UINT32_MAX)return VM_RM_HEAP_RETAIN_OVERFLOW;
    if(m->entries[slot].state!=0 &&
       (m->entries[slot].state!=2 || !m->tombstone_count ||
        m->entries[slot].key.tag!=TAG_VOID || m->entries[slot].value.tag!=TAG_VOID))return VM_RM_HEAP_INVALID;
    uint64_t occupied=(uint64_t)m->count+m->tombstone_count+1;
    if((double)occupied/(double)m->bucket_count>HM_LOAD_FACTOR) {
        if(m->bucket_count>UINT32_MAX/2 || !hm_resize(heap,m,m->bucket_count*2))return VM_RM_HEAP_MEMORY;
        /* Rehash invalidates slot; topology induction guarantees a new empty
         * slot. An impossible violation is not mislabeled allocator failure. */
        if(!hm_find_slot(m,key,&slot,&found) || found || m->entries[slot].state!=0)
            return VM_RM_HEAP_INVALID;
    }
    /* Nothing fallible follows this exact one-edge retain. */
    vm_retain(heap,key);
    if(m->entries[slot].state==2)m->tombstone_count--;
    m->entries[slot].key=key;m->entries[slot].value=value;
    m->entries[slot].state=1;m->count++;
    return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_ordinary_delete(VmHeap *heap,VmHashMap *m,NanoValue key) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) ||
       !vm_rm_ordinary_header(m) || !vm_rm_ordinary_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_delete_validated(heap,m,key);
}
VmRmHeapStatus vm_rm_ordinary_get(const VmHeap *heap,const VmHashMap *m,NanoValue key,NanoValue *out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out || !vm_rm_ordinary_header(m) || !vm_rm_ordinary_string(key))return VM_RM_HEAP_INVALID;
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    NanoValue result=val_void();
    if(found) {
        if(m->entries[slot].state!=1 || m->entries[slot].value.tag!=TAG_INT)return VM_RM_HEAP_INVALID;
        result=m->entries[slot].value;
    }
    *out=result;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_ordinary_has(const VmHeap *heap,const VmHashMap *m,NanoValue key,bool *out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out || !vm_rm_ordinary_header(m) || !vm_rm_ordinary_string(key))return VM_RM_HEAP_INVALID;
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    if(found && (m->entries[slot].state!=1 || m->entries[slot].value.tag!=TAG_INT))return VM_RM_HEAP_INVALID;
    *out=found;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_ordinary_len(const VmHeap *heap,const VmHashMap *m,uint32_t *out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out || !vm_rm_ordinary_header(m))return VM_RM_HEAP_INVALID;
    *out=m->count;return VM_RM_HEAP_OK;
}
/* Header charge/check already completed. Only nonzero tombstone_count uses
 * this bounded scan; zero-tombstone validation keeps its old work schedule. */
static VmRmHeapStatus vm_rm_validate_tombstones(const VmHashMap *m,VmRmObservationWork *w,bool ordinary) {
    uint32_t count=0,tombstones=0;
    for(uint32_t i=0;i<m->bucket_count;i++) {
        if(!vm_rm_observation_charge(w,8))return VM_RM_HEAP_LIMIT;
        const VmHMEntry *entry=&m->entries[i];
        if(!entry->state)continue;
        if(entry->state==2) {
            if(entry->key.tag!=TAG_VOID || entry->value.tag!=TAG_VOID)return VM_RM_HEAP_INVALID;
            tombstones++;continue;
        }
        if(entry->state!=1 || !vm_rm_string(entry->key) || entry->value.tag!=TAG_INT)return VM_RM_HEAP_INVALID;
        count++;
        const VmString *key=entry->key.as.string;
        if(!vm_rm_observation_charge(w,(uint64_t)key->length+1))return VM_RM_HEAP_LIMIT;
        if(ordinary && (key->header.colour!=VM_GC_BLACK || key->data[key->length]!=0))return VM_RM_HEAP_INVALID;
        if(fnv1a(key->data,key->length)!=key->hash)return VM_RM_HEAP_INVALID;
        uint32_t index=key->hash%m->bucket_count;bool located=false;
        for(uint32_t probes=0;probes<m->bucket_count;probes++) {
            if(!vm_rm_observation_charge(w,4))return VM_RM_HEAP_LIMIT;
            const VmHMEntry *candidate=&m->entries[index];
            if(candidate->state==0)return VM_RM_HEAP_INVALID;
            if(candidate->state==2) {
                if(candidate->key.tag!=TAG_VOID || candidate->value.tag!=TAG_VOID)return VM_RM_HEAP_INVALID;
                index=(index+1)%m->bucket_count;continue;
            }
            if(candidate->state!=1 || !vm_rm_string(candidate->key))return VM_RM_HEAP_INVALID;
            uint64_t comparison=candidate->key.as.string->length==key->length?(uint64_t)key->length+1:1;
            if(!vm_rm_observation_charge(w,comparison))return VM_RM_HEAP_LIMIT;
            if(val_equal(candidate->key,entry->key)) {
                if(index!=i)return VM_RM_HEAP_INVALID;
                located=true;break;
            }
            index=(index+1)%m->bucket_count;
        }
        if(!located)return VM_RM_HEAP_INVALID;
    }
    return count==m->count && tombstones==m->tombstone_count?VM_RM_HEAP_OK:VM_RM_HEAP_INVALID;
}
static VmRmHeapStatus vm_rm_validate_using(const VmHashMap *m,VmRmObservationWork *w,bool ordinary) {
    if(!w || w->limit>VM_RM_OBSERVATION_WORK_MAX)return VM_RM_HEAP_INVALID;
    if(!vm_rm_observation_charge(w,16))return VM_RM_HEAP_LIMIT;
    if(!(ordinary?vm_rm_import_header(m):vm_rm_map_header(m)))return VM_RM_HEAP_INVALID;
    if(m->tombstone_count!=0)return vm_rm_validate_tombstones(m,w,ordinary);
    uint32_t count=0;
    for(uint32_t i=0;i<m->bucket_count;i++) {
        if(!vm_rm_observation_charge(w,8))return VM_RM_HEAP_LIMIT;
        const VmHMEntry *entry=&m->entries[i];
        if(!entry->state)continue;
        if(entry->state!=1 || !vm_rm_string(entry->key) || entry->value.tag!=TAG_INT)return VM_RM_HEAP_INVALID;
        count++;
        const VmString *key=entry->key.as.string;
        if(!vm_rm_observation_charge(w,(uint64_t)key->length+1))return VM_RM_HEAP_LIMIT;
        if(ordinary && (key->header.colour!=VM_GC_BLACK || key->data[key->length]!=0))return VM_RM_HEAP_INVALID;
        if(fnv1a(key->data,key->length)!=key->hash)return VM_RM_HEAP_INVALID;
        uint32_t index=key->hash%m->bucket_count;bool located=false;
        /* This is the charged read-only equivalent of hm_find_slot for the
         * no-tombstone profile. It never invokes an uncharged full lookup. */
        for(uint32_t probes=0;probes<m->bucket_count;probes++) {
            if(!vm_rm_observation_charge(w,4))return VM_RM_HEAP_LIMIT;
            const VmHMEntry *candidate=&m->entries[index];
            if(candidate->state==0)return VM_RM_HEAP_INVALID;
            if(candidate->state!=1 || !vm_rm_string(candidate->key))return VM_RM_HEAP_INVALID;
            uint64_t comparison=candidate->key.as.string->length==key->length?(uint64_t)key->length+1:1;
            if(!vm_rm_observation_charge(w,comparison))return VM_RM_HEAP_LIMIT;
            if(val_equal(candidate->key,entry->key)) {
                if(index!=i)return VM_RM_HEAP_INVALID;
                located=true;break;
            }
            index=(index+1)%m->bucket_count;
        }
        if(!located)return VM_RM_HEAP_INVALID;
    }
    return count==m->count?VM_RM_HEAP_OK:VM_RM_HEAP_INVALID;
}

#ifdef NANO_RECORD_MAP_PRIVATE_HEAP
VmRmHeapStatus vm_rm_heap_validate(const VmHashMap *m,VmRmObservationWork *w) {
    return vm_rm_validate_using(m,w,false);
}
#endif
VmRmHeapStatus vm_rm_heap_import_validate(const VmHashMap *m,VmRmObservationWork *w) {
    return vm_rm_validate_using(m,w,true);
}
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
static VmRmHeapStatus vm_rm_validated_byte_probe(const VmHashMap *m,const void *bytes,uint32_t length,
                                                VmRmObservationWork *w,NanoValue *out) {
    if(!vm_rm_observation_charge(w,(uint64_t)length+1))return VM_RM_HEAP_LIMIT;
    uint32_t hash=fnv1a(bytes,length),index=hash%m->bucket_count;
    for(uint32_t probe=0;probe<m->bucket_count;probe++) {
        if(!vm_rm_observation_charge(w,4))return VM_RM_HEAP_LIMIT;
        const VmHMEntry *entry=&m->entries[index];
        if(!entry->state){*out=val_void();return VM_RM_HEAP_OK;}
        if(entry->state==2){index=(index+1)%m->bucket_count;continue;}
        const VmString *key=entry->key.as.string;
        if(!vm_rm_observation_charge(w,key->length==length?(uint64_t)length+1:1))return VM_RM_HEAP_LIMIT;
        if(key->length==length && (!length || !memcmp(key->data,bytes,length))) {
            *out=entry->value;return VM_RM_HEAP_OK;
        }
        index=(index+1)%m->bucket_count;
    }
    *out=val_void();return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_observe_lookup(const VmHashMap *m,const void *bytes,uint32_t length,
                                        VmRmObservationWork *w,NanoValue *out) {
    if(!out || (length&&!bytes) || !w || w->used>w->limit || w->limit>VM_RM_OBSERVATION_WORK_MAX)
        return VM_RM_HEAP_INVALID;
    if(w->limited)return VM_RM_HEAP_LIMIT;
    VmRmHeapStatus status=vm_rm_heap_validate(m,w);
    if(status!=VM_RM_HEAP_OK)return status;
    return vm_rm_validated_byte_probe(m,bytes,length,w,out);
}
#endif

/* STRING->STRING physical helpers are deliberately separate from the frozen
 * INT path. Managed heap/string storage and exclusive rooted inputs are C
 * preconditions; these guards cannot establish arbitrary pointer safety. */
static bool vm_rm_ss_header(const VmHashMap *m,bool ordinary) {
    return m && m->header.obj_type==TAG_HASHMAP && m->header.ref_count &&
        (ordinary ? (m->header.buffered<=1 &&
          (m->header.colour==VM_GC_BLACK || m->header.colour==VM_GC_PURPLE) &&
          (!m->header.buffered || m->header.colour==VM_GC_PURPLE)) : !m->header.buffered) &&
        m->key_type==TAG_STRING && m->val_type==TAG_STRING && m->entries &&
        m->bucket_count>=HM_INITIAL_BUCKETS && !(m->bucket_count&(m->bucket_count-1)) &&
        sizeof(VmHMEntry)<=SIZE_MAX/(size_t)m->bucket_count &&
        m->count<=m->bucket_count && m->tombstone_count<=m->bucket_count-m->count;
}
static bool vm_rm_ss_string(NanoValue value) {
    return vm_rm_ordinary_string(value);
}
static VmRmHeapStatus vm_rm_ss_set_validated(VmHeap *heap,VmHashMap *m,NanoValue key,NanoValue value) {
    uint32_t slot;bool found;
    /* A validated externally full table can have no insertion slot. This
     * selected boundary is distinct from absent GET/HAS/DELETE semantics. */
    if(!hm_find_slot(m,key,&slot,&found))return VM_RM_HEAP_INVALID;
    VmHMEntry *entry=&m->entries[slot];
    if(found) {
        if(entry->state!=1 || !vm_rm_ss_string(entry->key) || !vm_rm_ss_string(entry->value))return VM_RM_HEAP_INVALID;
        if(value.as.string->header.ref_count==UINT32_MAX)return VM_RM_HEAP_RETAIN_OVERFLOW;
        NanoValue old=entry->value;
        vm_retain(heap,value);entry->value=value;vm_release(heap,old);
        return VM_RM_HEAP_OK;
    }
    uint32_t needed=key.as.string==value.as.string?2:1;
    if(key.as.string->header.ref_count>UINT32_MAX-needed ||
       value.as.string->header.ref_count>UINT32_MAX-needed)return VM_RM_HEAP_RETAIN_OVERFLOW;
    if((entry->state!=0 && entry->state!=2) ||
       entry->key.tag!=TAG_VOID || entry->value.tag!=TAG_VOID ||
       (entry->state==2 && !m->tombstone_count))return VM_RM_HEAP_INVALID;
    uint64_t occupied=(uint64_t)m->count+m->tombstone_count+1;
    if((double)occupied/(double)m->bucket_count>HM_LOAD_FACTOR) {
        if(m->bucket_count>UINT32_MAX/2 || !hm_resize(heap,m,m->bucket_count*2))return VM_RM_HEAP_MEMORY;
        if(!hm_find_slot(m,key,&slot,&found) || found || m->entries[slot].state!=0)return VM_RM_HEAP_INVALID;
        entry=&m->entries[slot];
    }
    /* Both temporary headroom obligations precede growth; nothing fallible
     * follows these retains, even when key and value are the same object. */
    vm_retain(heap,key);vm_retain(heap,value);
    if(entry->state==2)m->tombstone_count--;
    entry->key=key;entry->value=value;entry->state=1;m->count++;
    return VM_RM_HEAP_OK;
}
static VmRmHeapStatus vm_rm_ss_get_validated(VmHeap *heap,const VmHashMap *m,NanoValue key,NanoValue *out) {
    uint32_t slot;bool found;
    NanoValue result=val_void();
    if(hm_find_slot(m,key,&slot,&found) && found) {
        const VmHMEntry *entry=&m->entries[slot];
        if(entry->state!=1 || !vm_rm_ss_string(entry->value))return VM_RM_HEAP_INVALID;
        result=entry->value;
        if(result.as.string->header.ref_count==UINT32_MAX)return VM_RM_HEAP_RETAIN_OVERFLOW;
        vm_retain(heap,result);
    }
    *out=result;return VM_RM_HEAP_OK;
}
static VmRmHeapStatus vm_rm_ss_has_validated(const VmHashMap *m,NanoValue key,bool *out) {
    uint32_t slot;bool found=false;
    if(!hm_find_slot(m,key,&slot,&found))found=false;
    if(found && (m->entries[slot].state!=1 || !vm_rm_ss_string(m->entries[slot].value)))return VM_RM_HEAP_INVALID;
    *out=found;return VM_RM_HEAP_OK;
}
static VmRmHeapStatus vm_rm_ss_delete_validated(VmHeap *heap,VmHashMap *m,NanoValue key) {
    uint32_t slot;bool found;
    if(!hm_find_slot(m,key,&slot,&found) || !found)return VM_RM_HEAP_OK;
    VmHMEntry *entry=&m->entries[slot];
    if(entry->state!=1 || !vm_rm_ss_string(entry->key) || !vm_rm_ss_string(entry->value) || !m->count)return VM_RM_HEAP_INVALID;
    NanoValue stored_key=entry->key,stored_value=entry->value;
    entry->key=val_void();entry->value=val_void();entry->state=2;
    m->count--;m->tombstone_count++;
    vm_release(heap,stored_key);vm_release(heap,stored_value);
    return VM_RM_HEAP_OK;
}
static VmRmHeapStatus vm_rm_ss_validate_using(const VmHashMap *m,VmRmObservationWork *w,bool ordinary) {
    if(!w || w->used>w->limit || w->limit>VM_RM_OBSERVATION_WORK_MAX)return VM_RM_HEAP_INVALID;
    if(!vm_rm_observation_charge(w,16))return VM_RM_HEAP_LIMIT;
    if(!vm_rm_ss_header(m,ordinary))return VM_RM_HEAP_INVALID;
    uint32_t count=0,tombstones=0;
    for(uint32_t i=0;i<m->bucket_count;i++) {
        if(!vm_rm_observation_charge(w,8))return VM_RM_HEAP_LIMIT;
        const VmHMEntry *entry=&m->entries[i];
        if(entry->state!=1) {
            if(entry->state>2 || entry->key.tag!=TAG_VOID || entry->value.tag!=TAG_VOID)return VM_RM_HEAP_INVALID;
            if(entry->state==2)tombstones++;
            continue;
        }
        if(!vm_rm_ss_string(entry->key))return VM_RM_HEAP_INVALID;
        /* Every stored value occurrence is charged, including key/value and
         * cross-bucket aliases. Header and byte charges precede their scans. */
        if(!vm_rm_observation_charge(w,16))return VM_RM_HEAP_LIMIT;
        if(!vm_rm_ss_string(entry->value))return VM_RM_HEAP_INVALID;
        const VmString *value=entry->value.as.string,*key=entry->key.as.string;
        if(!vm_rm_observation_charge(w,(uint64_t)value->length+1))return VM_RM_HEAP_LIMIT;
        if(value->data[value->length]!=0 || fnv1a(value->data,value->length)!=value->hash)return VM_RM_HEAP_INVALID;
        if(!vm_rm_observation_charge(w,(uint64_t)key->length+1))return VM_RM_HEAP_LIMIT;
        if(key->data[key->length]!=0 || fnv1a(key->data,key->length)!=key->hash)return VM_RM_HEAP_INVALID;
        count++;
        uint32_t index=key->hash%m->bucket_count;bool located=false;
        for(uint32_t probes=0;probes<m->bucket_count;probes++) {
            if(!vm_rm_observation_charge(w,4))return VM_RM_HEAP_LIMIT;
            const VmHMEntry *candidate=&m->entries[index];
            if(candidate->state==0)return VM_RM_HEAP_INVALID;
            if(candidate->state==2) {
                if(candidate->key.tag!=TAG_VOID || candidate->value.tag!=TAG_VOID)return VM_RM_HEAP_INVALID;
                index=(index+1)%m->bucket_count;continue;
            }
            if(candidate->state!=1 || !vm_rm_ss_string(candidate->key))return VM_RM_HEAP_INVALID;
            uint64_t comparison=candidate->key.as.string->length==key->length?(uint64_t)key->length+1:1;
            if(!vm_rm_observation_charge(w,comparison))return VM_RM_HEAP_LIMIT;
            if(val_equal(candidate->key,entry->key)) {
                if(index!=i)return VM_RM_HEAP_INVALID;
                located=true;break;
            }
            index=(index+1)%m->bucket_count;
        }
        if(!located)return VM_RM_HEAP_INVALID;
    }
    return count==m->count && tombstones==m->tombstone_count?VM_RM_HEAP_OK:VM_RM_HEAP_INVALID;
}

#ifdef NANO_RECORD_MAP_PRIVATE_HEAP
VmRmHeapStatus vm_rm_heap_ss_new(VmHeap *heap,VmHashMap **out) {
    if(!heap || heap->gc_running || !out)return VM_RM_HEAP_INVALID;
    VmHashMap *m=vm_hashmap_new(heap,TAG_STRING,TAG_STRING);
    if(!m)return VM_RM_HEAP_MEMORY;
    *out=m;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_ss_set(VmHeap *heap,VmHashMap *m,NanoValue key,NanoValue value) {
    if(!heap || heap->gc_running || !vm_rm_ss_header(m,false) || !vm_rm_ss_string(key) || !vm_rm_ss_string(value))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_set_validated(heap,m,key,value);
}
VmRmHeapStatus vm_rm_heap_ss_get_owned(VmHeap *heap,const VmHashMap *m,NanoValue key,NanoValue *out) {
    if(!heap || heap->gc_running || !out || !vm_rm_ss_header(m,false) || !vm_rm_ss_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_get_validated(heap,m,key,out);
}
VmRmHeapStatus vm_rm_heap_ss_delete(VmHeap *heap,VmHashMap *m,NanoValue key) {
    if(!heap || heap->gc_running || !vm_rm_ss_header(m,false) || !vm_rm_ss_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_delete_validated(heap,m,key);
}
VmRmHeapStatus vm_rm_heap_ss_has(const VmHashMap *m,NanoValue key,bool *out) {
    if(!out || !vm_rm_ss_header(m,false) || !vm_rm_ss_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_has_validated(m,key,out);
}
VmRmHeapStatus vm_rm_heap_ss_len(const VmHashMap *m,uint32_t *out) {
    if(!out || !vm_rm_ss_header(m,false))return VM_RM_HEAP_INVALID;
    *out=m->count;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_ss_validate(const VmHashMap *m,VmRmObservationWork *w) {
    return vm_rm_ss_validate_using(m,w,false);
}
#endif
VmRmHeapStatus vm_rm_ordinary_ss_new(VmHeap *heap,VmHashMap **out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out)return VM_RM_HEAP_INVALID;
    VmHashMap *m=vm_hashmap_new(heap,TAG_STRING,TAG_STRING);
    if(!m)return VM_RM_HEAP_MEMORY;
    *out=m;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_ordinary_ss_set(VmHeap *heap,VmHashMap *m,NanoValue key,NanoValue value) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !vm_rm_ss_header(m,true) || !vm_rm_ss_string(key) || !vm_rm_ss_string(value))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_set_validated(heap,m,key,value);
}
VmRmHeapStatus vm_rm_ordinary_ss_get_owned(VmHeap *heap,const VmHashMap *m,NanoValue key,NanoValue *out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out || !vm_rm_ss_header(m,true) || !vm_rm_ss_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_get_validated(heap,m,key,out);
}
VmRmHeapStatus vm_rm_ordinary_ss_delete(VmHeap *heap,VmHashMap *m,NanoValue key) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !vm_rm_ss_header(m,true) || !vm_rm_ss_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_delete_validated(heap,m,key);
}
VmRmHeapStatus vm_rm_ordinary_ss_has(const VmHeap *heap,const VmHashMap *m,NanoValue key,bool *out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out || !vm_rm_ss_header(m,true) || !vm_rm_ss_string(key))return VM_RM_HEAP_INVALID;
    return vm_rm_ss_has_validated(m,key,out);
}
VmRmHeapStatus vm_rm_ordinary_ss_len(const VmHeap *heap,const VmHashMap *m,uint32_t *out) {
    if(!heap || heap->gc_running || vm_rm_private_dag(heap) || !out || !vm_rm_ss_header(m,true))return VM_RM_HEAP_INVALID;
    *out=m->count;return VM_RM_HEAP_OK;
}
VmRmHeapStatus vm_rm_heap_ss_import_validate(const VmHashMap *m,VmRmObservationWork *w) {
    return vm_rm_ss_validate_using(m,w,true);
}

#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
/* Borrowed result remains internal to one read-only query. No retain or trust
 * flag escapes; the complete SS topology/byte validation always precedes probe. */
VmRmHeapStatus vm_rm_heap_ss_observe_lookup(const VmHashMap *m,const void *bytes,uint32_t length,
    VmRmObservationWork *w,NanoValue *out) {
    if(!out || (length&&!bytes) || !w || w->used>w->limit || w->limit>VM_RM_OBSERVATION_WORK_MAX)
        return VM_RM_HEAP_INVALID;
    if(w->limited)return VM_RM_HEAP_LIMIT;
    VmRmHeapStatus status=vm_rm_heap_ss_validate(m,w);
    if(status!=VM_RM_HEAP_OK)return status;
    return vm_rm_validated_byte_probe(m,bytes,length,w,out);
}
#endif
