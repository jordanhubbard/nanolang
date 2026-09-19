/* I reuse retained byte preflight and ownership transport parsing, never their
 * executable authority. The existing mixed descriptor query stays unchanged. */
#include "owned_array_layouts.h"
#include "service_bindings_module.h"
struct NvmOwnedArrayLayouts {
    NvmV2Layouts layouts;
    NvmOwnedArrayLayoutCounts counts;
    NvmOwnedArrayLayoutFact facts[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t source_to_global[NVM_RECORD_PLAN_MAX_LAYOUTS];
    uint32_t managed_to_global[NVM_RECORD_PLAN_MAX_LAYOUTS];
    NvmOwnedArrayLeafPath *paths;
    uint8_t *layout_bytes, *ownership;
    uint32_t layout_size, ownership_size;
};
void nvm_owned_array_layouts_free(NvmOwnedArrayLayouts *v) {
    if(!v)return;
    nvm_v2_layouts_free(&v->layouts);
    free(v->paths);free(v->layout_bytes);free(v->ownership);free(v);
}
static NvmRecordPlanResult oa_classify(NvmOwnedArrayLayouts *v,const uint8_t *flags) {
    for(uint32_t i=0;i<v->layouts.count;i++) {
        const NvmV2Layout *layout=&v->layouts.items[i];
        NvmOwnedArrayLayoutFact *fact=&v->facts[i];
        *fact=(NvmOwnedArrayLayoutFact){.global_layout=i,.source_record=NVM_V2_NO_INDEX,
            .managed_record=NVM_V2_NO_INDEX,.field_count=layout->field_count,.flags=flags[i]};
        if(layout->kind!=NVM_V2_LAYOUT_STRUCT) {
            if(flags[i])return record_result(NVM_RECORD_UNRESOLVED,i,0,"I retain passive non-record declarations without owner authority.");
            continue;
        }
        if(!(flags[i]&NVM_LAYOUT_COMPLETE))
            return record_result(NVM_RECORD_UNRESOLVED,i,0,"I require explicit complete record declarations.");
        bool owner=(flags[i]&NVM_LAYOUT_RESOURCE)!=0;
        fact->source_record=v->counts.source_records;
        v->source_to_global[v->counts.source_records++]=i;
        for(uint16_t j=0;j<layout->field_count;j++) {
            const NvmV2LayoutField *field=&layout->fields[j];
            uint32_t paths=owner?1:0;
            if(field->type_tag==TAG_STRUCT) {
                uint32_t child=field->nested_idx;
                if(child>=i || v->layouts.items[child].kind!=NVM_V2_LAYOUT_STRUCT ||
                   !(flags[child]&NVM_LAYOUT_COMPLETE) ||
                   owner!=((flags[child]&NVM_LAYOUT_RESOURCE)!=0))
                    return record_result(NVM_RECORD_UNRESOLVED,i,j,"I require prior children in the same owner or ordinary category.");
                const NvmOwnedArrayLayoutFact *nested=&v->facts[child];
                fact->has_array|=nested->has_array;fact->has_string|=nested->has_string;
                if(owner) {
                    if(nested->owner_depth>=NVM_OWNED_ARRAY_MAX_PATH_DEPTH)
                        return record_result(NVM_RECORD_LIMIT,i,j,"I reached my owner nesting limit.");
                    uint8_t depth=(uint8_t)(nested->owner_depth+1);
                    if(depth>fact->owner_depth)fact->owner_depth=depth;
                    paths=nested->path_count;
                }
            } else {
                bool scalar=field->type_tag==TAG_INT || field->type_tag==TAG_BOOL ||
                            field->type_tag==TAG_U8 || (!owner && field->type_tag==TAG_FLOAT);
                if(field->nested_idx!=NVM_V2_NO_INDEX ||
                   (!scalar && field->type_tag!=TAG_STRING && field->type_tag!=TAG_ARRAY))
                    return record_result(NVM_RECORD_UNRESOLVED,i,j,"I require exact scalar, STRING or pending ARRAY leaves.");
                fact->has_array|=field->type_tag==TAG_ARRAY;
                fact->has_string|=field->type_tag==TAG_STRING;
            }
            if(paths>NVM_OWNED_ARRAY_MAX_PATHS-fact->path_count)
                return record_result(NVM_RECORD_LIMIT,i,j,"I reached my expanded owner leaf-path budget.");
            fact->path_count+=paths;
        }
        fact->whole_root_borrow_unsuitable=owner && (fact->has_array || fact->has_string);
        fact->path_start=v->counts.leaf_paths;
        if(fact->path_count>NVM_OWNED_ARRAY_MAX_PATHS-v->counts.leaf_paths)
            return record_result(NVM_RECORD_LIMIT,i,0,"I reached my total owner leaf-path budget.");
        v->counts.leaf_paths+=fact->path_count;
        /* Pending ordinary ARRAY rows receive no compact managed identity until
         * a separate origin proof. Owner rows never receive one. */
        if(!owner && !fact->has_array) {
            fact->managed_record=v->counts.managed_records;
            v->managed_to_global[v->counts.managed_records++]=i;
        }
    }
    return record_result(NVM_RECORD_DESCRIBED,0,0,"I described pending owner leaves without element proof.");
}
static NvmRecordPlanResult oa_paths(NvmOwnedArrayLayouts *v) {
    uint32_t total=v->counts.leaf_paths;
    if(total && sizeof(*v->paths)>SIZE_MAX/total)
        return record_result(NVM_RECORD_LIMIT,0,0,"I cannot represent my owner leaf-path storage.");
    if(total) {
        v->paths=calloc(total,sizeof(*v->paths));
        if(!v->paths)return record_result(NVM_RECORD_MEMORY,0,0,"I could not allocate my bounded owner leaf paths.");
    }
    for(uint32_t i=0;i<v->layouts.count;i++) {
        const NvmOwnedArrayLayoutFact *fact=&v->facts[i];
        if(!(fact->flags&NVM_LAYOUT_RESOURCE))continue;
        const NvmV2Layout *layout=&v->layouts.items[i];uint32_t next=fact->path_start;
        for(uint16_t field=0;field<layout->field_count;field++) {
            const NvmV2LayoutField *f=&layout->fields[field];
            if(f->type_tag==TAG_STRUCT) {
                const NvmOwnedArrayLayoutFact *child=&v->facts[f->nested_idx];
                for(uint32_t n=0;n<child->path_count;n++) {
                    const NvmOwnedArrayLeafPath *from=&v->paths[child->path_start+n];
                    if(from->length>=NVM_OWNED_ARRAY_MAX_PATH_DEPTH)
                        return record_result(NVM_RECORD_LIMIT,i,field,"I reached my retained leaf-path length limit.");
                    NvmOwnedArrayLeafPath *to=&v->paths[next++];
                    to->root_layout=i;to->terminal_layout=from->terminal_layout;
                    to->tag=from->tag;to->length=(uint16_t)(from->length+1);to->fields[0]=field;
                    memcpy(to->fields+1,from->fields,(size_t)from->length*sizeof(uint16_t));
                }
            } else {
                NvmOwnedArrayLeafPath *to=&v->paths[next++];
                to->root_layout=to->terminal_layout=i;to->tag=f->type_tag;
                to->length=1;to->fields[0]=field;
            }
        }
        if(next!=fact->path_start+fact->path_count)
            return record_result(NVM_RECORD_INVALID,i,0,"I require exact expanded leaf-path counts.");
    }
    return record_result(NVM_RECORD_DESCRIBED,0,0,"I retained bounded owner leaf paths.");
}
NvmRecordPlanResult nvm_describe_owned_array_layouts(const NvmModule *m,NvmOwnedArrayLayouts **out) {
    if(!m || !out || (!m->layout_data!=!m->layout_size) || (!m->ownership_data!=!m->ownership_size))
        return record_result(NVM_RECORD_INVALID,0,0,"I require consistent owner-array descriptor inputs.");
    if(nvm_service_bindings_present(m))
        return record_result(NVM_RECORD_UNRESOLVED,0,0,"I refuse service claims before owner-array description.");
    if(!m->layout_size || !m->ownership_size)
        return record_result(NVM_RECORD_UNRESOLVED,0,0,"I require retained layouts and explicit ownership facts.");
    if(m->ownership_size>NVM_MIXED_MAX_OWNERSHIP_BYTES)
        return record_result(NVM_RECORD_LIMIT,0,0,"I reached my ownership transport budget.");
    bool forward=false;NvmRecordPlanResult r=record_preflight(m,&forward);
    if(r.status!=NVM_RECORD_DESCRIBED)return r;
    if(forward)return record_result(NVM_RECORD_UNRESOLVED,0,0,"I retain prior-only owner-array layout dependencies.");
    NvmV2Cursor cursor;uint32_t count;
    nvm_v2_cursor_init(&cursor,m->layout_data,m->layout_size);
    if(nvm_v2_u32(&cursor,&count)!=NVM_V2_OK)
        return record_result(NVM_RECORD_INVALID,0,0,"I require a complete layout count.");
    uint8_t flags[NVM_RECORD_PLAN_MAX_LAYOUTS]={0};
    if(!mixed_ownership_read(m,count,NULL,flags))
        return record_result(NVM_RECORD_INVALID,0,0,"I require exact ownership transport before allocation.");
    NvmOwnedArrayLayouts *v=calloc(1,sizeof(*v));
    if(!v)return record_result(NVM_RECORD_MEMORY,0,0,"I could not allocate my owner-array descriptor view.");
    NvmV2Result decoded=nvm_v2_layouts_decode(m->layout_data,m->layout_size,&v->layouts);
    if(decoded!=NVM_V2_OK) {
        r=record_result(decoded==NVM_V2_ERR_TRUNCATED?NVM_RECORD_MEMORY:NVM_RECORD_INVALID,
                        0,0,"I could not materialize preflighted owner-array layouts.");goto fail;
    }
    v->counts.layouts=count;r=oa_classify(v,flags);
    if(r.status!=NVM_RECORD_DESCRIBED)goto fail;
    /* I retain the existing whole-root reference restriction, including ARRAY
     * and STRING siblings outside any selected scalar path. */
    if(!mixed_ownership_read(m,count,&v->layouts,flags)) {
        r=record_result(NVM_RECORD_UNRESOLVED,0,0,"I retain exact scalar-only borrowed roots.");goto fail;
    }
    r=oa_paths(v);if(r.status!=NVM_RECORD_DESCRIBED)goto fail;
    v->layout_bytes=malloc(m->layout_size);v->ownership=malloc(m->ownership_size);
    if(!v->layout_bytes || !v->ownership) {
        r=record_result(NVM_RECORD_MEMORY,0,0,"I could not retain original descriptor transport.");goto fail;
    }
    memcpy(v->layout_bytes,m->layout_data,m->layout_size);v->layout_size=m->layout_size;
    memcpy(v->ownership,m->ownership_data,m->ownership_size);v->ownership_size=m->ownership_size;
    *out=v;return record_result(NVM_RECORD_DESCRIBED,0,0,"I described pending owner ARRAY fields without runtime admission.");
fail:
    nvm_owned_array_layouts_free(v);return r;
}
bool nvm_owned_array_layout_counts(const NvmOwnedArrayLayouts *v,NvmOwnedArrayLayoutCounts *out) {
    if(!v || !out)return false;
    *out=v->counts;return true;
}
bool nvm_owned_array_layout_fact(const NvmOwnedArrayLayouts *v,uint32_t global,NvmOwnedArrayLayoutFact *out) {
    if(!v || !out || global>=v->counts.layouts)return false;
    *out=v->facts[global];return true;
}
bool nvm_owned_array_source_layout(const NvmOwnedArrayLayouts *v,uint32_t source,uint32_t *out) {
    if(!v || !out || source>=v->counts.source_records)return false;
    *out=v->source_to_global[source];return true;
}
bool nvm_owned_array_managed_layout(const NvmOwnedArrayLayouts *v,uint32_t managed,uint32_t *out) {
    if(!v || !out || managed>=v->counts.managed_records)return false;
    *out=v->managed_to_global[managed];return true;
}
bool nvm_owned_array_leaf_path(const NvmOwnedArrayLayouts *v,uint32_t path,NvmOwnedArrayLeafPath *out) {
    if(!v || !out || path>=v->counts.leaf_paths)return false;
    *out=v->paths[path];return true;
}
bool nvm_owned_array_layout_field(const NvmOwnedArrayLayouts *v,uint32_t global,uint16_t field,NvmV2LayoutField *out) {
    if(!v || !out || global>=v->layouts.count || field>=v->layouts.items[global].field_count)return false;
    *out=v->layouts.items[global].fields[field];return true;
}
bool nvm_owned_array_transport(const NvmOwnedArrayLayouts *v,NvmOwnedArrayTransport *out) {
    if(!v || !out)return false;
    *out=(NvmOwnedArrayTransport){v->layout_bytes,v->ownership,v->layout_size,v->ownership_size};return true;
}
