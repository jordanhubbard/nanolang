/* Read-only copies; no guest callbacks, allocations, retains or identity IDs. */
static VmRecordMapObservationStatus vm_rmo_arguments(const VmRecordMapPrivate *p,
    const uint32_t *left,uint16_t ln,const uint32_t *right,uint16_t rn,
    const void *bytes,size_t bytes_n,VmRmObservationWork *work,void *out,size_t out_n) {
    if(!p||!work||ln>VM_RECORD_MAP_PATH_MAX||rn>VM_RECORD_MAP_PATH_MAX)return VM_RM_OBS_INVALID;
    size_t lb=(size_t)ln*sizeof *left,rb=(size_t)rn*sizeof *right;
    if(!vm_rmo_disjoint(out,out_n,work,sizeof *work)||
       !vm_rmo_disjoint(out,out_n,p,sizeof *p)||!vm_rmo_disjoint(work,sizeof *work,p,sizeof *p)||
       !vm_rmo_disjoint(out,out_n,left,lb)||!vm_rmo_disjoint(work,sizeof *work,left,lb)||
       !vm_rmo_disjoint(out,out_n,right,rb)||!vm_rmo_disjoint(work,sizeof *work,right,rb)||
       !vm_rmo_disjoint(out,out_n,bytes,bytes_n)||!vm_rmo_disjoint(work,sizeof *work,bytes,bytes_n))return VM_RM_OBS_INVALID;
    if(work->used>work->limit||work->limit>VM_RM_OBSERVATION_WORK_MAX)return VM_RM_OBS_INVALID;
    return work->limited?VM_RM_OBS_LIMIT:VM_RM_OBS_OK;
}
static bool vm_rmo_charge(VmRmObservationWork *w,uint64_t amount) {
    if(w->limited||w->used>w->limit||amount>w->limit-w->used){w->limited=true;return false;}
    w->used+=amount;return true;
}
static VmRecordMapObservationStatus vm_rmo_heap_status(VmRmHeapStatus s) {
    return s==VM_RM_HEAP_OK?VM_RM_OBS_OK:s==VM_RM_HEAP_LIMIT?VM_RM_OBS_LIMIT:VM_RM_OBS_INVALID;
}
typedef struct {
    uint8_t tag,depth,leaf,key,value;
    uint32_t layout;
} VmRmObservedExpected;
static bool vm_rmo_shell(const VmRecordMapPrivate *p,NanoValue value,VmRmObservedExpected expected) {
    if(expected.tag!=UINT8_MAX&&value.tag!=expected.tag)return false;
    if(value.tag==TAG_VOID||value.tag==TAG_INT||value.tag==TAG_FLOAT||value.tag==TAG_BOOL||value.tag==TAG_U8)return true;
    if(!val_is_heap_obj(value)||!value.as.obj)return false;
    const VmHeapHeader *h=value.as.obj;
    if(h->obj_type!=value.tag||!h->ref_count||h->buffered)return false;
    if(value.tag==TAG_STRING) {
        const VmString *s=value.as.string;
        return (uint64_t)s->length+sizeof *s+1<=SIZE_MAX&&s->data[s->length]=='\0';
    }
    if(value.tag==TAG_STRUCT) {
        const VmStruct *s=value.as.sval;
        return s->def_idx<p->layout_count&&(!s->field_count||s->fields)&&!s->field_names&&
            p->layouts[s->def_idx].declaration.kind==NVM_V2_LAYOUT_STRUCT&&
            s->field_count==p->layouts[s->def_idx].declaration.fields&&
            (expected.layout==UINT32_MAX||s->def_idx==expected.layout);
    }
    if(value.tag==TAG_ARRAY) {
        const VmArray *a=value.as.array;
        if(a->length>a->capacity||a->unboxed!=vm_array_type_unboxable(a->elem_type)||
           (a->capacity&&!(a->unboxed?a->packed!=NULL:a->elements!=NULL))||
           !(vm_rm_leaf(a->elem_type)||a->elem_type==TAG_ARRAY))return false;
        size_t width=a->unboxed?vm_array_elem_size(a->elem_type):sizeof(NanoValue);
        if(!width||a->capacity>SIZE_MAX/width)return false;
        return !expected.depth || (expected.depth<=3&&a->elem_type==(expected.depth==1?expected.leaf:TAG_ARRAY));
    }
    if(value.tag==TAG_HASHMAP) {
        uint32_t length;
        return vm_rm_heap_len(value.as.hashmap,&length)==VM_RM_HEAP_OK&&
            (!expected.key||value.as.hashmap->key_type==expected.key)&&
            (!expected.value||value.as.hashmap->val_type==expected.value);
    }
    return false;
}
static VmRecordMapObservationStatus vm_rmo_walk(const VmRecordMapPrivate *p,VmRecordMapRoot root,
    uint32_t global,const uint32_t *path,uint16_t count,VmRmObservationWork *w,bool validate_map,NanoValue *out) {
    if(!vm_rmo_charge(w,16))return VM_RM_OBS_LIMIT;
    if(!p->prepared||p->busy||!pthread_equal(p->vm.owner_thread,pthread_self())||p->vm.frame_count||p->vm.stack_size||p->vm.activation_floor||p->vm.heap.gc_running||
       p->vm.references.active||p->vm.callee_references.active||p->vm.reference_generation||
       p->vm.callbacks||p->vm.handler_count||p->vm.linked_module_count||p->vm.verified||
       p->vm.module!=nvm_record_map_image_borrow(p->image)||p->vm.root_module!=p->vm.module)return VM_RM_OBS_INVALID;
    NanoValue value;
    if(root==VM_RM_RESULT){if(global||!p->has_result)return VM_RM_OBS_INVALID;value=p->result;}
    else if(root==VM_RM_GLOBAL){if(global>=p->vm.global_count)return VM_RM_OBS_INVALID;value=p->vm.globals[global];}
    else return VM_RM_OBS_INVALID;
    VmRmObservedExpected expected={.tag=UINT8_MAX,.layout=UINT32_MAX};unsigned generic_arrays=0;
    for(uint16_t i=0;;i++) {
        if(!vm_rmo_charge(w,16))return VM_RM_OBS_LIMIT;
        if(!vm_rmo_shell(p,value,expected))return VM_RM_OBS_INVALID;
        if(value.tag==TAG_ARRAY){if(++generic_arrays>3)return VM_RM_OBS_INVALID;}else generic_arrays=0;
        if(i==count)break;
        if(!vm_rmo_charge(w,8))return VM_RM_OBS_LIMIT;
        uint32_t index=path[i];
        if(value.tag==TAG_STRUCT) {
            const VmStruct *s=value.as.sval;if(index>=s->field_count)return VM_RM_OBS_INVALID;
            uint32_t start=p->layouts[s->def_idx].field_start;
            if(start>p->field_count||index>=p->field_count-start)return VM_RM_OBS_INVALID;
            const VmRecordMapRuntimeField *f=&p->fields[start+index];
            expected=(VmRmObservedExpected){.tag=f->declaration.type_tag,.layout=UINT32_MAX};
            if(expected.tag==TAG_STRUCT)expected.layout=f->declaration.nested_idx;
            if(expected.tag==TAG_ARRAY){expected.depth=f->array_depth;expected.leaf=f->nested_leaf?f->nested_leaf:f->element;}
            if(expected.tag==TAG_HASHMAP){expected.key=f->key;expected.value=f->value;}
            value=s->fields[index];
        } else if(value.tag==TAG_ARRAY) {
            VmArray *a=value.as.array;if(index>=a->length)return VM_RM_OBS_INVALID;
            uint8_t depth=expected.depth,leaf=expected.leaf;
            expected=(VmRmObservedExpected){.tag=a->elem_type,.layout=UINT32_MAX};
            if(depth>1){expected.depth=(uint8_t)(depth-1);expected.leaf=leaf;}
            value=vm_array_get(a,index);
        } else return VM_RM_OBS_INVALID;
    }
    if(value.tag==TAG_HASHMAP&&validate_map) {
        VmRecordMapObservationStatus s=vm_rmo_heap_status(vm_rm_heap_validate(value.as.hashmap,w));
        if(s!=VM_RM_OBS_OK)return s;
    }
    *out=value;return VM_RM_OBS_OK;
}
static VmRecordMapObservation vm_rmo_copy(const VmRecordMapPrivate *p,NanoValue value) {
    VmRecordMapObservation out={.epoch=p->epoch,.tag=value.tag,.layout=UINT32_MAX};
    if(value.tag==TAG_BOOL)out.scalar_bits=value.as.boolean?1:0;
    else if(value.tag==TAG_U8)out.scalar_bits=value.as.u8;
    else if(value.tag==TAG_INT||value.tag==TAG_FLOAT)memcpy(&out.scalar_bits,&value.as,8);
    else if(value.tag==TAG_STRING)out.length=value.as.string->length;
    else if(value.tag==TAG_ARRAY){out.length=value.as.array->length;out.element_tag=value.as.array->elem_type;}
    else if(value.tag==TAG_STRUCT){out.length=value.as.sval->field_count;out.layout=value.as.sval->def_idx;}
    else if(value.tag==TAG_HASHMAP){out.length=value.as.hashmap->count;out.key_tag=value.as.hashmap->key_type;out.value_tag=value.as.hashmap->val_type;}
    return out;
}
VmRecordMapObservationStatus vm_record_map_private_observe(const VmRecordMapPrivate *p,VmRecordMapRoot root,
    uint32_t global,const uint32_t *path,uint16_t count,VmRmObservationWork *w,VmRecordMapObservation *out) {
    VmRecordMapObservationStatus s=vm_rmo_arguments(p,path,count,NULL,0,NULL,0,w,out,sizeof *out);if(s!=VM_RM_OBS_OK)return s;
    NanoValue value;s=vm_rmo_walk(p,root,global,path,count,w,true,&value);if(s!=VM_RM_OBS_OK)return s;
    if(!vm_rmo_charge(w,1))return VM_RM_OBS_LIMIT;
    *out=vm_rmo_copy(p,value);return VM_RM_OBS_OK;
}
VmRecordMapObservationStatus vm_record_map_private_lookup(const VmRecordMapPrivate *p,VmRecordMapRoot root,
    uint32_t global,const uint32_t *path,uint16_t count,const void *key,uint32_t length,
    VmRmObservationWork *w,VmRecordMapObservation *out) {
    VmRecordMapObservationStatus s=vm_rmo_arguments(p,path,count,NULL,0,key,length,w,out,sizeof *out);if(s!=VM_RM_OBS_OK)return s;
    NanoValue value,result;s=vm_rmo_walk(p,root,global,path,count,w,false,&value);if(s!=VM_RM_OBS_OK)return s;
    if(value.tag!=TAG_HASHMAP)return VM_RM_OBS_INVALID;
    s=vm_rmo_heap_status(vm_rm_heap_observe_lookup(value.as.hashmap,key,length,w,&result));if(s!=VM_RM_OBS_OK)return s;
    if(!vm_rmo_charge(w,1))return VM_RM_OBS_LIMIT;
    *out=vm_rmo_copy(p,result);return VM_RM_OBS_OK;
}
VmRecordMapObservationStatus vm_record_map_private_string(const VmRecordMapPrivate *p,VmRecordMapRoot root,
    uint32_t global,const uint32_t *path,uint16_t count,uint32_t start,uint32_t length,VmRmObservationWork *w,void *out) {
    VmRecordMapObservationStatus s=vm_rmo_arguments(p,path,count,NULL,0,NULL,0,w,out,length);if(s!=VM_RM_OBS_OK)return s;
    NanoValue value;s=vm_rmo_walk(p,root,global,path,count,w,true,&value);if(s!=VM_RM_OBS_OK)return s;
    if(value.tag!=TAG_STRING||start>value.as.string->length||length>value.as.string->length-start)return VM_RM_OBS_INVALID;
    /* Both byte work and the final commit are prepaid: no LIMIT after copy. */
    if(!vm_rmo_charge(w,(uint64_t)length+1)||!vm_rmo_charge(w,1))return VM_RM_OBS_LIMIT;
    if(length)memcpy(out,value.as.string->data+start,length);
    return VM_RM_OBS_OK;
}
VmRecordMapObservationStatus vm_record_map_private_same_identity(const VmRecordMapPrivate *p,
    VmRecordMapRoot lr,uint32_t lg,const uint32_t *lp,uint16_t ln,
    VmRecordMapRoot rr,uint32_t rg,const uint32_t *rp,uint16_t rn,VmRmObservationWork *w,VmRecordMapIdentityComparison *out) {
    VmRecordMapObservationStatus s=vm_rmo_arguments(p,lp,ln,rp,rn,NULL,0,w,out,sizeof *out);if(s!=VM_RM_OBS_OK)return s;
    NanoValue left,right;s=vm_rmo_walk(p,lr,lg,lp,ln,w,true,&left);if(s!=VM_RM_OBS_OK)return s;
    s=vm_rmo_walk(p,rr,rg,rp,rn,w,true,&right);if(s!=VM_RM_OBS_OK)return s;
    if(!val_is_heap_obj(left)||!val_is_heap_obj(right))return VM_RM_OBS_INVALID;
    if(!vm_rmo_charge(w,1))return VM_RM_OBS_LIMIT;
    VmRecordMapIdentityComparison result={.epoch=p->epoch,.same=left.tag==right.tag&&left.as.obj==right.as.obj};
    if(!vm_rmo_charge(w,1))return VM_RM_OBS_LIMIT;
    *out=result;return VM_RM_OBS_OK;
}
