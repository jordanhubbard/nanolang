/* Standalone read-only query. No selector or execution handler calls this. */
#include "record_map_import_private.h"
#include "../nanoisa/managed_record_map_boundary.h"
typedef struct {
    const void *object;
    uint32_t obligation;
    uint8_t colour,member,reserved[2];
} VmRmImportIdentity;
typedef struct { const void *object;uint32_t obligation,child; } VmRmImportFrame;
typedef struct {
    VmRmImportIdentity *table;
    uint32_t capacity,count;
    uint64_t bytes,work,objects,collectors,edges;
    VmRmImportFrame frames[259];
} VmRmImportState;
_Static_assert(sizeof(VmRmImportIdentity)==16,"import identity ledger");
_Static_assert(sizeof(VmRmImportFrame)==16,"import frame ledger");
_Static_assert(sizeof(VmRmImportState)==4200,"import state ledger");
_Static_assert(sizeof(VmRmImportReport)==40,"import report ledger");
static VmRmImportStatus vm_rmi_charge(VmRmImportState *s,uint64_t bytes,uint64_t work) {
    if(s->bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES || s->work>NVM_RECORD_ARRAY_EXECUTION_STEPS ||
       bytes>NVM_RECORD_ARRAY_EXECUTION_BYTES-s->bytes || work>NVM_RECORD_ARRAY_EXECUTION_STEPS-s->work)
        return VM_RM_IMPORT_LIMIT;
    s->bytes+=bytes;s->work+=work;return VM_RM_IMPORT_OK;
}
static uint32_t vm_rmi_hash(const void *p,uint32_t capacity) {
    uintptr_t n=(uintptr_t)p;n^=n>>17;n*=UINT64_C(0xed5ad4bb);n^=n>>11;
    return (uint32_t)n&(capacity-1);
}
static VmRmImportStatus vm_rmi_slot(VmRmImportState *s,const void *object,VmRmImportIdentity **out) {
    if(!object)return VM_RM_IMPORT_INVALID;
    uint32_t slot=0;
    if(s->capacity) {
        slot=vm_rmi_hash(object,s->capacity);
        for(uint32_t probe=0;probe<s->capacity;probe++) {
            if(vm_rmi_charge(s,0,2)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
            VmRmImportIdentity *row=&s->table[slot];
            if(row->object==object){*out=row;return VM_RM_IMPORT_OK;}
            if(!row->object)break;
            slot=(slot+1)&(s->capacity-1);
        }
    }
    if(!s->capacity || s->count>=s->capacity/2) {
        if(s->capacity>UINT32_MAX/2)return VM_RM_IMPORT_LIMIT;
        uint32_t capacity=s->capacity?s->capacity*2:16;
        uint64_t bytes=(uint64_t)capacity*sizeof *s->table;
        if(bytes>SIZE_MAX || vm_rmi_charge(s,bytes,bytes+s->capacity)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        VmRmImportIdentity *table=calloc(capacity,sizeof *table);
        if(!table)return VM_RM_IMPORT_MEMORY;
        for(uint32_t i=0;i<s->capacity;i++)if(s->table[i].object) {
            uint32_t at=vm_rmi_hash(s->table[i].object,capacity);
            for(;;) {
                if(vm_rmi_charge(s,0,2)!=VM_RM_IMPORT_OK){free(table);return VM_RM_IMPORT_LIMIT;}
                if(!table[at].object)break;
                at=(at+1)&(capacity-1);
            }
            table[at]=s->table[i];
        }
        free(s->table);s->table=table;s->capacity=capacity;
        slot=vm_rmi_hash(object,capacity);
        for(;;) {
            if(vm_rmi_charge(s,0,2)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
            if(!s->table[slot].object)break;
            slot=(slot+1)&(capacity-1);
        }
    }
    VmRmImportIdentity *row=&s->table[slot];
    row->object=object;row->obligation=UINT32_MAX;s->count++;*out=row;return VM_RM_IMPORT_OK;
}
static bool vm_rmi_container(uint8_t tag) {
    return tag==TAG_ARRAY || tag==TAG_STRUCT || tag==TAG_UNION || tag==TAG_TUPLE || tag==TAG_HASHMAP || tag==TAG_CLOSURE;
}
static VmRmImportStatus vm_rmi_membership(VmRmImportState *s,const VmHeap *heap) {
    if(heap->gc_running || heap->cycle_count>heap->cycle_capacity ||
       (!!heap->cycle_capacity!=!!heap->cycle_buf) ||
       (uint64_t)heap->cycle_capacity*sizeof *heap->cycle_buf>SIZE_MAX)return VM_RM_IMPORT_INVALID;
    if(vm_rmi_charge(s,0,(uint64_t)heap->cycle_count*8u)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    for(uint32_t i=0;i<heap->cycle_count;i++) {
        const VmHeapHeader *h=heap->cycle_buf[i];
        if(!h || !vm_rmi_container(h->obj_type) || h->buffered!=1 ||
           h->colour!=(h->ref_count?VM_GC_PURPLE:VM_GC_BLACK))return VM_RM_IMPORT_INVALID;
        VmRmImportIdentity *row;VmRmImportStatus status=vm_rmi_slot(s,h,&row);
        if(status!=VM_RM_IMPORT_OK)return status;
        if(row->member)return VM_RM_IMPORT_INVALID;
        row->member=1;s->collectors++;
    }
    return VM_RM_IMPORT_OK;
}
static VmRmImportStatus vm_rmi_string(VmRmImportState *s,const VmString *str) {
    if(!str || str->header.obj_type!=TAG_STRING || !str->header.ref_count ||
       str->header.buffered || str->header.colour!=VM_GC_BLACK ||
       (uint64_t)str->length+sizeof *str+1>SIZE_MAX)return VM_RM_IMPORT_INVALID;
    if(vm_rmi_charge(s,0,(uint64_t)str->length+16u)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    uint32_t hash=UINT32_C(2166136261);
    for(uint32_t i=0;i<str->length;i++){hash^=(uint8_t)str->data[i];hash*=UINT32_C(16777619);}
    return str->data[str->length]==0 && hash==str->hash?VM_RM_IMPORT_OK:VM_RM_IMPORT_INVALID;
}
/* Each occurrence checks descriptors before a completed identity can be reused. */
static VmRmImportStatus vm_rmi_enter(VmRmImportState *s,const NvmRecordMapExecutionPlan *plan,
    uint32_t origins,NanoValue value,NvmRecordValueOrigins expected,uint32_t nominal,uint32_t *depth) {
    if(vm_rmi_charge(s,0,16)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    if(expected.unknown || value.tag>=16 || !(expected.tags&(uint16_t)(1u<<value.tag)))return VM_RM_IMPORT_UNRESOLVED;
    if(value.tag==TAG_VOID || value.tag==TAG_INT || value.tag==TAG_FLOAT || value.tag==TAG_BOOL || value.tag==TAG_U8)
        return VM_RM_IMPORT_OK;
    if(value.tag==TAG_STRING)return vm_rmi_string(s,value.as.string);
    uint8_t kind;
    if(value.tag==TAG_ARRAY)kind=NVM_HEAP_ORIGIN_ARRAY;
    else if(value.tag==TAG_STRUCT)kind=NVM_HEAP_ORIGIN_RECORD;
    else if(value.tag==TAG_HASHMAP)kind=NVM_RECORD_MAP_ORIGIN;
    else return VM_RM_IMPORT_UNRESOLVED;
    const VmHeapHeader *h=value.as.obj;
    if(!h || h->obj_type!=value.tag || !h->ref_count || h->buffered>1 ||
       (h->colour!=VM_GC_BLACK && h->colour!=VM_GC_PURPLE) ||
       (h->buffered && h->colour!=VM_GC_PURPLE))return VM_RM_IMPORT_INVALID;
    if(origins>64 || vm_rmi_charge(s,0,64)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    uint32_t selected=UINT32_MAX;NvmRecordHeapOrigin origin={0};
    for(uint32_t i=0;i<origins;i++)if(expected.origins&(UINT64_C(1)<<i)) {
        NvmRecordHeapOrigin row;
        if(vm_rmi_charge(s,0,4)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        if(!nvm_record_map_execution_origin(plan,i,&row))return VM_RM_IMPORT_INVALID;
        if(row.kind==kind){if(selected!=UINT32_MAX)return VM_RM_IMPORT_UNRESOLVED;selected=i;origin=row;}
    }
    if(selected==UINT32_MAX)return VM_RM_IMPORT_UNRESOLVED;
    if(value.tag==TAG_STRUCT) {
        const VmStruct *record=value.as.sval;NvmRecordArrayExecutionDescriptor desc;NvmDeclarationLayout layout;
        if(vm_rmi_charge(s,0,8)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        if(!nvm_record_map_execution_descriptor(plan,origin.record_ordinal,&desc) ||
           !nvm_record_map_execution_layout(plan,origin.layout_index,&layout) ||
           desc.ordinal!=origin.record_ordinal || desc.layout!=origin.layout_index ||
           record->def_idx!=origin.record_ordinal || (nominal!=UINT32_MAX && nominal!=origin.layout_index) ||
           layout.kind!=NVM_V2_LAYOUT_STRUCT || record->field_count!=origin.field_count ||
           record->field_count!=layout.fields || record->field_names ||
           (record->field_count&&!record->fields))return VM_RM_IMPORT_UNRESOLVED;
    } else if(value.tag==TAG_ARRAY) {
        const VmArray *array=value.as.array;uint16_t required;
        if(vm_rmi_charge(s,0,4)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        if(!nvm_record_map_execution_required_elements(plan,selected,&required))return VM_RM_IMPORT_INVALID;
        if(array->elem_type!=origin.declared_tag || array->unboxed!=origin.packed ||
           array->unboxed!=vm_array_type_unboxable(array->elem_type) || array->length>array->capacity)
            return VM_RM_IMPORT_UNRESOLVED;
        size_t width=array->unboxed?vm_array_elem_size(array->elem_type):sizeof(NanoValue);
        if(!width || array->capacity>SIZE_MAX/width ||
           (array->capacity && !(array->unboxed?array->packed!=NULL:array->elements!=NULL)))return VM_RM_IMPORT_INVALID;
        if(required) {
            if(vm_rmi_charge(s,0,16)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
            uint8_t rank=nvm_record_array_requirement_depth(required);
            uint8_t leaf=rank==1?array->elem_type:nvm_record_array_requirement_leaf(required);
            if(!rank || nvm_record_array_requirement(rank,leaf)!=required ||
               array->elem_type!=(rank==1?leaf:TAG_ARRAY))return VM_RM_IMPORT_UNRESOLVED;
        }
    } else {
        uint8_t key,val;
        if(vm_rmi_charge(s,0,NVM_RECORD_MAP_ORIGIN_QUERY_WORK)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        if(!nvm_record_map_execution_map_origin(plan,selected,&key,&val) || key!=TAG_STRING || val!=TAG_INT)
            return VM_RM_IMPORT_UNRESOLVED;
    }
    VmRmImportIdentity *identity;VmRmImportStatus status=vm_rmi_slot(s,h,&identity);
    if(status!=VM_RM_IMPORT_OK)return status;
    if(identity->member!=h->buffered)return VM_RM_IMPORT_INVALID;
    if(identity->obligation!=UINT32_MAX && identity->obligation!=selected)return VM_RM_IMPORT_UNRESOLVED;
    if(identity->colour==1)return VM_RM_IMPORT_UNRESOLVED;
    if(identity->colour==2)return VM_RM_IMPORT_OK;
    identity->obligation=selected;identity->colour=1;s->objects++;
    if(value.tag==TAG_HASHMAP) {
        VmRmObservationWork work={.used=s->work,.limit=NVM_RECORD_ARRAY_EXECUTION_STEPS};
        VmRmHeapStatus checked=vm_rm_heap_import_validate(value.as.hashmap,&work);s->work=work.used;
        if(checked!=VM_RM_HEAP_OK)return checked==VM_RM_HEAP_LIMIT?VM_RM_IMPORT_LIMIT:VM_RM_IMPORT_INVALID;
        identity->colour=2;return VM_RM_IMPORT_OK;
    }
    if(*depth>=259)return VM_RM_IMPORT_LIMIT;
    s->frames[(*depth)++]=(VmRmImportFrame){h,selected,0};return VM_RM_IMPORT_OK;
}
static VmRmImportStatus vm_rmi_graph(VmRmImportState *s,const NvmRecordMapExecutionPlan *plan,
    uint32_t origins,NanoValue value,NvmRecordValueOrigins expected) {
    uint32_t depth=0;VmRmImportStatus status=vm_rmi_enter(s,plan,origins,value,expected,UINT32_MAX,&depth);
    if(status!=VM_RM_IMPORT_OK)return status;
    while(depth) {
        VmRmImportFrame *frame=&s->frames[depth-1];NvmRecordHeapOrigin origin;
        if(vm_rmi_charge(s,0,4)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        if(!nvm_record_map_execution_origin(plan,frame->obligation,&origin))return VM_RM_IMPORT_INVALID;
        const VmHeapHeader *h=frame->object;
        uint32_t count=h->obj_type==TAG_STRUCT?((const VmStruct *)h)->field_count:((const VmArray *)h)->length;
        if(frame->child==count) {
            VmRmImportIdentity *identity;status=vm_rmi_slot(s,h,&identity);
            if(status!=VM_RM_IMPORT_OK)return status;
            identity->colour=2;depth--;continue;
        }
        if(vm_rmi_charge(s,0,8)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
        uint32_t child=frame->child++;s->edges++;
        NanoValue next;NvmRecordValueOrigins facts={0};uint32_t nominal=UINT32_MAX;
        if(h->obj_type==TAG_STRUCT) {
            NvmV2LayoutField declaration;
            if(vm_rmi_charge(s,0,8)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
            if(child>UINT16_MAX || !nvm_record_map_execution_field(plan,origin.layout_index,(uint16_t)child,&declaration) ||
               !nvm_record_map_execution_field_value(plan,origin.field_start+child,&facts))return VM_RM_IMPORT_INVALID;
            next=((const VmStruct *)h)->fields[child];
            if(next.tag!=declaration.type_tag)return VM_RM_IMPORT_UNRESOLVED;
            if(next.tag==TAG_STRUCT)nominal=declaration.nested_idx;
        } else {
            /* Packed arrays physically store converted declared scalar values;
             * their proof intentionally has no boxed child-origin summary. */
            next=vm_array_get((VmArray *)h,child);
            facts=origin.children;
            if(origin.packed)facts=(NvmRecordValueOrigins){.tags=(uint16_t)(1u<<origin.declared_tag)};
        }
        status=vm_rmi_enter(s,plan,origins,next,facts,nominal,&depth);
        if(status!=VM_RM_IMPORT_OK)return status;
    }
    return VM_RM_IMPORT_OK;
}
static VmRmImportStatus vm_rmi_check_reserved(const NvmRecordMapBoundaryPlan *boundary,const VmState *vm,
    VmRmImportState *state,VmRmImportReport *report,uint64_t extra_bytes,uint64_t extra_work) {
    NvmRecordMapBoundaryCounts boundary_counts;NvmRecordArrayOriginCounts origins;
    const NvmRecordMapExecutionPlan *plan=nvm_record_map_boundary_execution(boundary);
    if(!plan || !nvm_record_map_boundary_counts(boundary,&boundary_counts))return VM_RM_IMPORT_INVALID;
    state->bytes=boundary_counts.peak_bytes_reserved;state->work=boundary_counts.work_reserved;
    /* Positive conservative pre-reservation only; the ordinary query supplies0/0. */
    if(vm_rmi_charge(state,extra_bytes,extra_work)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    /* Three origin rows can be live across graph->enter/filter; all other copied query
     * rows are reused in bounded helper stack frames, never guest recursion. */
    uint64_t scratch=sizeof *state+sizeof *report+3*sizeof(NvmRecordHeapOrigin)+2*sizeof(NvmRecordValueOrigins)+2*sizeof(NanoValue)+
        sizeof(NvmDeclarationLayout)+sizeof(NvmV2LayoutField)+sizeof(NvmRecordArrayExecutionDescriptor)+
        sizeof(NvmRecordMapBoundaryCounts)+sizeof(NvmRecordArrayOriginCounts)+sizeof(VmRmObservationWork);
    VmRmImportStatus status=vm_rmi_charge(state,scratch,scratch+32+8);
    if(status!=VM_RM_IMPORT_OK)return status;
    if(!nvm_record_map_execution_origin_counts(plan,&origins) || origins.origins>64)return VM_RM_IMPORT_INVALID;
    if(!pthread_equal(vm->owner_thread,pthread_self()) || vm->frame_count || vm->stack_size ||
       vm->activation_floor || vm->references.active || vm->callee_references.active || vm->handler_count ||
       vm->heap.gc_running || vm_rm_private_dag(&vm->heap) || vm->global_count>vm->global_capacity ||
       vm->global_capacity>VM_MAX_GLOBALS || (!!vm->global_capacity!=!!vm->globals) ||
       (uint64_t)vm->global_capacity*sizeof *vm->globals>SIZE_MAX)return VM_RM_IMPORT_INVALID;
    if(!vm_rmo_disjoint(report,sizeof *report,vm->globals,(size_t)vm->global_capacity*sizeof *vm->globals))
        return VM_RM_IMPORT_INVALID;
    status=vm_rmi_membership(state,&vm->heap);if(status!=VM_RM_IMPORT_OK)return status;
    uint32_t count=vm->global_capacity>boundary_counts.globals?vm->global_capacity:boundary_counts.globals;
    if(vm_rmi_charge(state,0,(uint64_t)count*8u)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    for(uint32_t i=0;i<count;i++) {
        NanoValue value=i<vm->global_capacity?vm->globals[i]:val_void();
        NvmRecordValueOrigins facts={.tags=(uint16_t)(1u<<TAG_VOID)};
        if(i<boundary_counts.globals) {
            if(vm_rmi_charge(state,0,4)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
            if(!nvm_record_map_boundary_global(boundary,i,&facts))return VM_RM_IMPORT_INVALID;
        }
        status=vm_rmi_graph(state,plan,origins.origins,value,facts);
        if(status!=VM_RM_IMPORT_OK)return status;
    }
    if(vm_rmi_charge(state,0,sizeof *report)!=VM_RM_IMPORT_OK)return VM_RM_IMPORT_LIMIT;
    *report=(VmRmImportReport){state->bytes,state->work,state->objects,state->collectors,state->edges};
    return VM_RM_IMPORT_OK;
}
#ifdef NANO_RECORD_MAP_PRIVATE_RUNTIME
VmRmImportStatus vm_record_map_import_check(const NvmModule *module,const VmState *vm,VmRmImportReport *out) {
    if(!module || !vm || !out || vm->module!=module || vm->root_module!=module ||
       !vm_rmo_disjoint(out,sizeof *out,vm,sizeof *vm) ||
       !vm_rmo_disjoint(out,sizeof *out,module,sizeof *module))return VM_RM_IMPORT_INVALID;
    NvmRecordMapBoundaryPlan *boundary=NULL;
    NvmArrayEligibilityResult result=nvm_prepare_record_map_boundary(module,&boundary);
    if(result.status!=NVM_ARRAY_ELIGIBLE) {
        return result.status==NVM_ARRAY_MEMORY?VM_RM_IMPORT_MEMORY:
            result.status==NVM_ARRAY_LIMIT?VM_RM_IMPORT_LIMIT:
            result.status==NVM_ARRAY_UNRESOLVED?VM_RM_IMPORT_UNRESOLVED:VM_RM_IMPORT_INVALID;
    }
    VmRmImportState state={0};
    VmRmImportStatus status=vm_rmi_check_reserved(boundary,vm,&state,out,0,0);
    free(state.table);nvm_record_map_boundary_free(boundary);return status;
}

#endif
