#include "affine_state.h"
#include "isa.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct { uint8_t tag, mode; uint32_t layout; } Slot;
typedef struct {
    uint32_t users;
    NvmV2Layouts layouts;
    uint8_t *flags;
    Slot result, *locals;
    uint16_t count, params;
    const NvmModule *module;
    uint32_t function;
} Facts;
typedef struct {
    bool live;
    uint32_t region, parent;
    NvmReferencePlace place;
} Reference;
typedef struct { uint64_t invocation; uint16_t local; uint32_t layout; } CallerOrigin;
struct NvmAffineState {
    Facts *facts;
    bool *live;
    Reference *refs;
    uint32_t ref_count, region;
    bool caller_bound;
    uint64_t invocation;
    uint16_t origin_count;
    CallerOrigin origins[NVM_AFFINE_MAX_PARAMETERS];
};
static bool scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_U8 || tag == TAG_FLOAT || tag == TAG_BOOL;
}
static bool resource(const Facts *f, Slot slot) {
    return slot.layout != NVM_V2_NO_INDEX &&
           (f->flags[slot.layout] & NVM_LAYOUT_RESOURCE);
}
static bool union_value(const Facts *f, Slot slot) {
    return slot.tag == TAG_UNION && slot.mode == 0 &&
           slot.layout < f->layouts.count && f->flags[slot.layout] == 0 &&
           f->layouts.items[slot.layout].kind == NVM_V2_LAYOUT_UNION;
}
static bool supported(const Facts *f, Slot slot) {
    return scalar(slot.tag) ||
           (slot.tag == TAG_STRING && slot.layout == NVM_V2_NO_INDEX) ||
           (slot.tag == TAG_STRUCT && slot.layout != NVM_V2_NO_INDEX) ||
           union_value(f,slot);
}
static bool same(Slot a, Slot b) {
    return a.tag == b.tag && a.layout == b.layout && a.mode == b.mode;
}
static bool slot_read(NvmV2Cursor *c, Slot *s) {
    uint16_t reserved;
    return nvm_v2_u8(c,&s->tag)==NVM_V2_OK && nvm_v2_u8(c,&s->mode)==NVM_V2_OK &&
           nvm_v2_u16(c,&reserved)==NVM_V2_OK && nvm_v2_u32(c,&s->layout)==NVM_V2_OK;
}
static void facts_free(Facts *f) {
    if (!f || --f->users) return;
    nvm_v2_layouts_free(&f->layouts); free(f->flags); free(f->locals); free(f);
}
void nvm_affine_state_free(NvmAffineState *s) {
    if (!s) return;
    if (s->refs) for (uint32_t i=0;i<s->ref_count;i++) free((void*)s->refs[i].place.fields);
    free(s->refs); free(s->live); facts_free(s->facts); free(s);
}
NvmAffineState *nvm_affine_state_create(const NvmModule *m, uint32_t function,
                                       uint32_t references) {
    bool needs;
    if (!m || !m->ownership_size || function >= m->function_count ||
        nvm_ownership_contracts_validate(m,&needs)!=NVM_V2_OK) return NULL;
    NvmAffineState *s=calloc(1,sizeof(*s));
    if (!s) return NULL;
    s->facts=calloc(1,sizeof(*s->facts));
    if (!s->facts) { free(s); return NULL; }
    Facts *f=s->facts; f->users=1; f->module=m; f->function=function;
    s->invocation=1;
    if (nvm_v2_layouts_decode(m->layout_data,m->layout_size,&f->layouts)!=NVM_V2_OK) goto fail;
    NvmV2Cursor c; nvm_v2_cursor_init(&c,m->ownership_data,m->ownership_size);
    uint32_t ignored, count; const uint8_t *flags;
    if (nvm_v2_u32(&c,&ignored)!=NVM_V2_OK || nvm_v2_u32(&c,&count)!=NVM_V2_OK ||
        nvm_v2_take(&c,count,&flags)!=NVM_V2_OK) goto fail;
    f->flags=malloc(count ? count : 1);
    if (!f->flags) goto fail;
    memcpy(f->flags,flags,count);
    if (nvm_v2_align4(&c)!=NVM_V2_OK || nvm_v2_u32(&c,&ignored)!=NVM_V2_OK) goto fail;
    uint16_t params=0;
    for (uint32_t i=0;i<=function;i++) {
        uint16_t locals; Slot result;
        if (nvm_v2_u16(&c,&locals)!=NVM_V2_OK || nvm_v2_u16(&c,&params)!=NVM_V2_OK ||
            !slot_read(&c,&result)) goto fail;
        if (i==function) {
            if (result.tag != TAG_VOID && !supported(f,result)) goto fail;
            f->count=locals; f->params=params; f->result=result;
            f->locals=calloc(locals ? locals : 1,sizeof(*f->locals));
            if (!f->locals) goto fail;
        }
        for (uint16_t j=0;j<locals;j++) {
            Slot slot; if (!slot_read(&c,&slot)) goto fail;
            if (i==function) {
                /* Unknown/collection declarations cannot establish absence
                 * of an ownership obligation in this bounded analysis. */
                if (!supported(f,slot)) goto fail;
                f->locals[j]=slot;
            }
        }
    }
    s->ref_count=references;
    s->live=calloc(f->count ? f->count : 1,sizeof(*s->live));
    s->refs=calloc(references ? references : 1,sizeof(*s->refs));
    if (!s->live || !s->refs) goto fail;
    for (uint16_t i=0;i<params;i++) {
        Slot slot=f->locals[i]; s->live[i]=true;
        if (slot.mode) {
            if (i>=references) goto fail;
            s->refs[i]=(Reference){true,0,UINT32_MAX,
                {1,i,slot.layout,NULL,0,slot.layout,(NvmReferenceMode)slot.mode}};
        }
    }
    return s;
fail:
    nvm_affine_state_free(s); return NULL;
}
NvmAffineState *nvm_affine_state_clone(const NvmAffineState *s) {
    if (!s || s->facts->users==UINT32_MAX) return NULL;
    NvmAffineState *out=calloc(1,sizeof(*out));
    if (!out) return NULL;
    out->facts=s->facts; out->facts->users++;
    out->ref_count=s->ref_count; out->region=s->region;
    out->caller_bound=s->caller_bound; out->invocation=s->invocation;
    out->origin_count=s->origin_count;
    memcpy(out->origins,s->origins,sizeof(out->origins));
    out->live=malloc((s->facts->count ? s->facts->count : 1)*sizeof(*s->live));
    out->refs=calloc(s->ref_count ? s->ref_count : 1,sizeof(*s->refs));
    if (!out->live || !out->refs) goto fail;
    memcpy(out->live,s->live,s->facts->count*sizeof(*s->live));
    for (uint32_t i=0;i<s->ref_count;i++) {
        out->refs[i]=s->refs[i]; out->refs[i].place.fields=NULL;
        uint16_t count=s->refs[i].place.field_count;
        if (count) {
            uint16_t *path=malloc(count*sizeof(*path)); if (!path) goto fail;
            memcpy(path,s->refs[i].place.fields,count*sizeof(*path));
            out->refs[i].place.fields=path;
        }
    }
    return out;
fail:
    nvm_affine_state_free(out); return NULL;
}
static bool state_equal(const NvmAffineState *a, const NvmAffineState *b, bool meet_scalars) {
    if (!a || !b || a->facts!=b->facts || a->region!=b->region ||
        a->ref_count!=b->ref_count || a->caller_bound!=b->caller_bound ||
        a->invocation!=b->invocation || a->origin_count!=b->origin_count) return false;
    for (uint16_t i=0;i<a->facts->count;i++) {
        Slot slot=a->facts->locals[i];
        if (meet_scalars && !slot.mode && (scalar(slot.tag) || slot.tag==TAG_STRING || slot.tag==TAG_UNION)) continue;
        if (a->live[i]!=b->live[i]) return false;
    }
    for (uint16_t i=0;i<a->origin_count;i++)
        if(a->origins[i].invocation!=b->origins[i].invocation ||
           a->origins[i].local!=b->origins[i].local ||
           a->origins[i].layout!=b->origins[i].layout) return false;
    for (uint32_t i=0;i<a->ref_count;i++) {
        const Reference *x=&a->refs[i],*y=&b->refs[i];
        if (x->live!=y->live) return false;
        if (!x->live) continue;
        const NvmReferencePlace *p=&x->place,*q=&y->place;
        if (x->region!=y->region || x->parent!=y->parent || p->invocation!=q->invocation ||
            p->local!=q->local || p->root_layout!=q->root_layout || p->field_count!=q->field_count ||
            p->referent_layout!=q->referent_layout || p->mode!=q->mode ||
            (p->field_count && memcmp(p->fields,q->fields,p->field_count*sizeof(*p->fields)))) return false;
    }
    return true;
}
bool nvm_affine_state_equal(const NvmAffineState *a, const NvmAffineState *b) {
    return state_equal(a,b,false);
}
bool nvm_affine_state_meet_initialization(NvmAffineState *destination,
                                          const NvmAffineState *incoming,bool *changed) {
    if (!changed) return false;
    *changed=false;
    /* I validate every authoritative fact before changing any scalar proof. */
    if (!state_equal(destination,incoming,true)) return false;
    for (uint16_t i=0;i<destination->facts->count;i++) {
        Slot slot=destination->facts->locals[i];
        if (!slot.mode && (scalar(slot.tag) || slot.tag==TAG_STRING || slot.tag==TAG_UNION) && destination->live[i] && !incoming->live[i]) {
            destination->live[i]=false;*changed=true;
        }
    }
    return true;
}
static bool value_local(const NvmAffineState *s, uint16_t local) {
    return s && local<s->facts->count && !s->facts->locals[local].mode;
}
static bool resolve(const NvmAffineState *s,uint16_t local,const uint16_t *path,
                    uint16_t count,Slot *out) {
    if (!s || local>=s->facts->count || (count && !path)) return false;
    Slot slot=s->facts->locals[local];
    for (uint16_t i=0;i<count;i++) {
        if (slot.layout==NVM_V2_NO_INDEX) return false;
        const NvmV2Layout *layout=&s->facts->layouts.items[slot.layout];
        if (path[i]>=layout->field_count) return false;
        NvmV2LayoutField field=layout->fields[path[i]];
        slot=(Slot){field.type_tag,0,field.nested_idx};
    }
    *out=slot; return scalar(slot.tag) ||
        (slot.tag==TAG_STRING && slot.layout==NVM_V2_NO_INDEX) ||
        slot.layout!=NVM_V2_NO_INDEX;
}
bool nvm_affine_owner_access(const NvmAffineState *s,uint16_t local,
                              const uint16_t *path,uint16_t count,bool write) {
    Slot slot;
    if (!value_local(s,local) || !s->live[local] || !resolve(s,local,path,count,&slot)) return false;
    NvmReferencePlace place={s->invocation,local,s->facts->locals[local].layout,path,count,
                              slot.layout,NVM_REFERENCE_SHARED};
    for (uint32_t i=0;i<s->ref_count;i++) if (s->refs[i].live &&
        nvm_reference_owner_access_conflicts(&s->refs[i].place,&place,write)) return false;
    return true;
}
static bool destination(const NvmAffineState *s,uint16_t local) {
    if (!value_local(s,local)) return false;
    if (!s->live[local]) return true;
    if (union_value(s->facts,s->facts->locals[local])) return true;
    return !resource(s->facts,s->facts->locals[local]) &&
           nvm_affine_owner_access(s,local,NULL,0,true);
}
bool nvm_affine_scalar_define(NvmAffineState *s,uint16_t local) {
    if (!destination(s,local) || !scalar(s->facts->locals[local].tag)) return false;
    s->live[local]=true; return true;
}
bool nvm_affine_string_define(NvmAffineState *s,uint16_t local) {
    if (!destination(s,local) || s->facts->locals[local].tag!=TAG_STRING ||
        s->facts->locals[local].layout!=NVM_V2_NO_INDEX) return false;
    s->live[local]=true;return true;
}
bool nvm_affine_union_define(NvmAffineState *s,uint16_t local,uint32_t layout) {
    if (!destination(s,local) || s->facts->locals[local].layout!=layout ||
        !union_value(s->facts,s->facts->locals[local])) return false;
    s->live[local]=true;return true;
}
bool nvm_affine_move(NvmAffineState *s,uint16_t from,uint16_t to) {
    if (from==to || !destination(s,to) || !nvm_affine_owner_access(s,from,NULL,0,true) ||
        !same(s->facts->locals[from],s->facts->locals[to])) return false;
    s->live[from]=false; s->live[to]=true; return true;
}
static bool fields_check(const NvmAffineState *s,uint16_t root,const uint16_t *fields,
                          uint16_t count,bool unpack) {
    if (!value_local(s,root) || (count && !fields)) return false;
    Slot slot=s->facts->locals[root];
    if (slot.layout==NVM_V2_NO_INDEX) return false;
    const NvmV2Layout *layout=&s->facts->layouts.items[slot.layout];
    if (count!=layout->field_count) return false;
    if (unpack ? !nvm_affine_owner_access(s,root,NULL,0,true) : !destination(s,root)) return false;
    for (uint16_t i=0;i<count;i++) {
        uint16_t local=fields[i]; NvmV2LayoutField field=layout->fields[i];
        if (local==root || !value_local(s,local) ||
            !same(s->facts->locals[local],(Slot){field.type_tag,0,field.nested_idx})) return false;
        for (uint16_t j=0;j<i;j++) if (fields[j]==local &&
            (unpack || resource(s->facts,s->facts->locals[local]))) return false;
        if (unpack ? !destination(s,local) :
            !nvm_affine_owner_access(s,local,NULL,0,resource(s->facts,s->facts->locals[local]))) return false;
    }
    return true;
}
bool nvm_affine_pack(NvmAffineState *s,uint16_t to,const uint16_t *fields,uint16_t count) {
    if (!fields_check(s,to,fields,count,false)) return false;
    for (uint16_t i=0;i<count;i++) if (resource(s->facts,s->facts->locals[fields[i]])) s->live[fields[i]]=false;
    s->live[to]=true; return true;
}
bool nvm_affine_unpack(NvmAffineState *s,uint16_t from,const uint16_t *fields,uint16_t count) {
    if (!fields_check(s,from,fields,count,true)) return false;
    s->live[from]=false;
    for (uint16_t i=0;i<count;i++) s->live[fields[i]]=true;
    return true;
}
bool nvm_affine_region_begin(NvmAffineState *s) {
    if (!s || s->region==UINT32_MAX) return false;
    s->region++; return true;
}
bool nvm_affine_region_end(NvmAffineState *s) {
    if (!s || !s->region) return false;
    for (uint32_t i=0;i<s->ref_count;i++) if (s->refs[i].live && s->refs[i].region==s->region) {
        free((void*)s->refs[i].place.fields); memset(&s->refs[i],0,sizeof(s->refs[i]));
    }
    s->region--; return true;
}
static bool ancestor(const NvmAffineState *s,uint32_t possible,uint32_t child) {
    for (uint32_t n=0;n<s->ref_count && child!=UINT32_MAX;n++) {
        if (child==possible) return true;
        child=s->refs[child].parent;
    }
    return false;
}
static uint32_t authoritative_root(const NvmAffineState *s,const NvmReferencePlace *place) {
    if(place->invocation==s->invocation && place->local<s->facts->count)
        return s->facts->locals[place->local].layout;
    for(uint16_t i=0;i<s->origin_count;i++)
        if(s->origins[i].invocation==place->invocation && s->origins[i].local==place->local)
            return s->origins[i].layout;
    return NVM_V2_NO_INDEX;
}
static bool borrow_install(NvmAffineState *s,uint32_t id,NvmReferencePlace place,uint32_t parent) {
    if (!s || !s->region || id>=s->ref_count || s->refs[id].live ||
        !nvm_reference_place_valid(&s->facts->layouts,authoritative_root(s,&place),&place) ||
        !resource(s->facts,(Slot){TAG_STRUCT,0,place.referent_layout})) return false;
    for (uint32_t i=0;i<s->ref_count;i++) if (s->refs[i].live &&
        !ancestor(s,i,parent) && nvm_reference_holds_conflict(&s->refs[i].place,&place)) return false;
    uint16_t *path=NULL;
    if (place.field_count) {
        path=malloc(place.field_count*sizeof(*path)); if (!path) return false;
        memcpy(path,place.fields,place.field_count*sizeof(*path));
    }
    place.fields=path;
    s->refs[id]=(Reference){true,s->region,parent,place}; return true;
}
bool nvm_affine_borrow(NvmAffineState *s,uint32_t id,uint16_t root,
                        const uint16_t *path,uint16_t count,NvmReferenceMode mode) {
    Slot slot;
    if (!value_local(s,root) || !s->live[root] || !resolve(s,root,path,count,&slot)) return false;
    return borrow_install(s,id,(NvmReferencePlace){s->invocation,root,s->facts->locals[root].layout,
        path,count,slot.layout,mode},UINT32_MAX);
}
bool nvm_affine_reborrow(NvmAffineState *s,uint32_t id,uint32_t parent,NvmReferenceMode mode) {
    if (!s || parent>=s->ref_count || !s->refs[parent].live ||
        s->refs[parent].region>=s->region ||
        (mode==NVM_REFERENCE_EXCLUSIVE && s->refs[parent].place.mode!=NVM_REFERENCE_EXCLUSIVE)) return false;
    NvmReferencePlace place=s->refs[parent].place; place.mode=mode;
    return borrow_install(s,id,place,parent);
}
bool nvm_affine_reference_access(const NvmAffineState *s,uint32_t id,uint16_t field,bool write) {
    if (!s || id>=s->ref_count || !s->refs[id].live) return false;
    const Reference *ref=&s->refs[id];
    const NvmV2Layout *layout=&s->facts->layouts.items[ref->place.referent_layout];
    if (field>=layout->field_count || !scalar(layout->fields[field].type_tag) ||
        (write && ref->place.mode!=NVM_REFERENCE_EXCLUSIVE)) return false;
    for (uint32_t i=0;i<s->ref_count;i++) if (i!=id && s->refs[i].live &&
        ancestor(s,id,i) && (write || s->refs[i].place.mode==NVM_REFERENCE_EXCLUSIVE)) return false;
    return true;
}
bool nvm_affine_reference_field(const NvmAffineState *s,uint32_t id,uint16_t field,
                                  bool write,uint8_t *tag) {
    if (!tag || !nvm_affine_reference_access(s,id,field,write)) return false;
    *tag=s->facts->layouts.items[s->refs[id].place.referent_layout].fields[field].type_tag;
    return true;
}
bool nvm_affine_can_exit(const NvmAffineState *s,uint16_t result) {
    if (!s || s->region) return false;
    if (result==UINT16_MAX) { if (s->facts->result.tag!=TAG_VOID) return false; }
    else if (!value_local(s,result) || !s->live[result] ||
             !same(s->facts->locals[result],s->facts->result)) return false;
    for (uint16_t i=0;i<s->facts->count;i++) if (i!=result && s->live[i] &&
        !s->facts->locals[i].mode && resource(s->facts,s->facts->locals[i])) return false;
    return true;
}

bool nvm_affine_local_info(const NvmAffineState *s,uint16_t local,
                            uint8_t *tag,uint8_t *mode) {
    if (!s || local>=s->facts->count || !s->live[local] || !tag || !mode) return false;
    Slot slot=s->facts->locals[local];
    if (s->caller_bound && slot.mode) return false;
    if (!slot.mode && slot.tag!=TAG_UNION && !nvm_affine_owner_access(s,local,NULL,0,false)) return false;
    *tag=slot.tag; *mode=slot.mode; return true;
}
bool nvm_affine_scalar_field(const NvmAffineState *s,uint16_t local,
                              uint16_t field,uint8_t *tag) {
    if (!s || local>=s->facts->count || !s->live[local] || !tag) return false;
    Slot root=s->facts->locals[local];
    if (root.layout==NVM_V2_NO_INDEX) return false;
    const NvmV2Layout *layout=&s->facts->layouts.items[root.layout];
    if (field>=layout->field_count || !scalar(layout->fields[field].type_tag)) return false;
    bool allowed=root.mode ? nvm_affine_reference_access(s,local,field,false)
        : nvm_affine_owner_access(s,local,&field,1,false);
    if (!allowed) return false;
    *tag=layout->fields[field].type_tag; return true;
}
bool nvm_affine_string_field(const NvmAffineState *s,uint16_t local,
                              uint16_t field,uint8_t *tag) {
    if (!s || local>=s->facts->count || !s->live[local] || !tag) return false;
    Slot root=s->facts->locals[local];
    if (root.mode || root.layout==NVM_V2_NO_INDEX) return false;
    const NvmV2Layout *layout=&s->facts->layouts.items[root.layout];
    if (field>=layout->field_count || layout->fields[field].type_tag!=TAG_STRING ||
        layout->fields[field].nested_idx!=NVM_V2_NO_INDEX ||
        !nvm_affine_owner_access(s,local,&field,1,false)) return false;
    *tag=TAG_STRING;return true;
}
bool nvm_affine_can_exit_scalar(const NvmAffineState *s,uint8_t tag) {
    if (!s || s->region || (tag!=TAG_VOID && !scalar(tag)) ||
        s->facts->result.tag!=tag) return false;
    for (uint16_t i=0;i<s->facts->count;i++) if (s->live[i] &&
        !s->facts->locals[i].mode && resource(s->facts,s->facts->locals[i])) return false;
    return true;
}

bool nvm_affine_local_type(const NvmAffineState *s,uint16_t local,NvmAffineType *type) {
    if (!s || local>=s->facts->count || !type) return false;
    Slot slot=s->facts->locals[local];
    if (slot.mode) return false;
    *type=(NvmAffineType){slot.tag,slot.layout}; return true;
}
bool nvm_affine_take_local(NvmAffineState *s,uint16_t local,NvmAffineType *type) {
    NvmAffineType found;
    if (!type || !nvm_affine_local_type(s,local,&found) || found.tag!=TAG_STRUCT ||
        found.layout==NVM_V2_NO_INDEX || !nvm_affine_owner_access(s,local,NULL,0,true)) return false;
    s->live[local]=false; *type=found; return true;
}
bool nvm_affine_put_local(NvmAffineState *s,uint16_t local,NvmAffineType type) {
    NvmAffineType wanted;
    if (!nvm_affine_local_type(s,local,&wanted) || type.tag!=TAG_STRUCT ||
        type.layout==NVM_V2_NO_INDEX || wanted.tag!=type.tag || wanted.layout!=type.layout ||
        !destination(s,local)) return false;
    s->live[local]=true; return true;
}
bool nvm_affine_record_fields(const NvmAffineState *s,uint32_t layout,
                               NvmAffineType *fields,uint16_t capacity,uint16_t *count) {
    if (!s || !count || layout>=s->facts->layouts.count ||
        !(s->facts->flags[layout]&NVM_LAYOUT_COMPLETE)) return false;
    const NvmV2Layout *record=&s->facts->layouts.items[layout];
    if (record->kind!=NVM_V2_LAYOUT_STRUCT || record->field_count>capacity ||
        (record->field_count && !fields)) return false;
    for (uint16_t i=0;i<record->field_count;i++)
        fields[i]=(NvmAffineType){record->fields[i].type_tag,record->fields[i].nested_idx};
    *count=record->field_count; return true;
}
bool nvm_affine_union_fields(const NvmAffineState *s,uint32_t ordinal,uint32_t *layout,
                              NvmAffineType *fields,uint16_t capacity,uint16_t *count) {
    if (!s || !layout || !count || (capacity && !fields)) return false;
    uint32_t seen=0;
    for (uint32_t i=0;i<s->facts->layouts.count;i++) {
        const NvmV2Layout *candidate=&s->facts->layouts.items[i];
        if (candidate->kind!=NVM_V2_LAYOUT_UNION) continue;
        if (seen++!=ordinal) continue;
        if (s->facts->flags[i] || candidate->field_count>capacity) return false;
        for (uint16_t f=0;f<candidate->field_count;f++) {
            uint8_t tag=candidate->fields[f].type_tag;
            if ((tag!=TAG_INT && tag!=TAG_U8 && tag!=TAG_FLOAT &&
                 tag!=TAG_BOOL && tag!=TAG_STRING) ||
                candidate->fields[f].nested_idx!=NVM_V2_NO_INDEX) return false;
            fields[f]=(NvmAffineType){tag,NVM_V2_NO_INDEX};
        }
        *layout=i;*count=candidate->field_count;return true;
    }
    return false;
}
bool nvm_affine_can_exit_type(const NvmAffineState *s,NvmAffineType type) {
    if (!s || s->region || !same(s->facts->result,(Slot){type.tag,0,type.layout})) return false;
    for (uint16_t i=0;i<s->facts->count;i++) if (s->live[i] &&
        !s->facts->locals[i].mode && resource(s->facts,s->facts->locals[i])) return false;
    return true;
}

/* I validate every requested authority against all caller holds, then all
 * requested pairs. Only after every path copy succeeds do I publish bindings. */
bool nvm_affine_bind_caller(NvmAffineState *callee,const NvmAffineState *caller,
                             uint32_t reference) {
    if (!callee || !caller || callee->caller_bound || caller->caller_bound ||
        callee->facts->module!=caller->facts->module ||
        caller->facts->function!=0 || callee->facts->function!=1 || callee->region)
        return false;
    uint16_t count=callee->facts->params;
    if(!count || count>NVM_AFFINE_MAX_PARAMETERS || count>callee->ref_count ||
       reference>caller->ref_count || count>caller->ref_count-reference) return false;
    for(uint32_t i=count;i<callee->ref_count;i++) if(callee->refs[i].live) return false;
    for(uint16_t i=count;i<callee->facts->count;i++) if(callee->live[i]) return false;
    NvmReferencePlace places[NVM_AFFINE_MAX_PARAMETERS];
    for(uint16_t p=0;p<count;p++) {
        uint32_t source=reference+p;Slot param=callee->facts->locals[p];
        if(!caller->refs[source].live) return false;
        NvmReferencePlace place=caller->refs[source].place;
        if(param.tag!=TAG_STRUCT || !param.mode || param.layout!=place.referent_layout ||
           (param.mode==NVM_REFERENCE_EXCLUSIVE && place.mode!=NVM_REFERENCE_EXCLUSIVE)) return false;
        place.mode=(NvmReferenceMode)param.mode;
        for(uint32_t i=0;i<caller->ref_count;i++) if(caller->refs[i].live &&
            !ancestor(caller,i,source) && nvm_reference_holds_conflict(&caller->refs[i].place,&place)) return false;
        if(place.invocation!=caller->invocation || place.local>=caller->facts->count ||
           !nvm_reference_place_valid(&caller->facts->layouts,
               caller->facts->locals[place.local].layout,&place)) return false;
        for(uint16_t i=0;i<p;i++) if(nvm_reference_holds_conflict(&places[i],&place)) return false;
        places[p]=place;
    }
    uint16_t *paths[NVM_AFFINE_MAX_PARAMETERS]={0};
    for(uint16_t p=0;p<count;p++) if(places[p].field_count) {
        paths[p]=malloc(places[p].field_count*sizeof(*paths[p]));
        if(!paths[p]) {
            for(uint16_t i=0;i<p;i++) free(paths[i]);
            return false;
        }
        memcpy(paths[p],places[p].fields,places[p].field_count*sizeof(*paths[p]));
    }
    for(uint16_t p=0;p<count;p++) {
        free((void*)callee->refs[p].place.fields);places[p].fields=paths[p];
        callee->refs[p]=(Reference){true,0,UINT32_MAX,places[p]};
        callee->origins[p]=(CallerOrigin){places[p].invocation,places[p].local,
            caller->facts->locals[places[p].local].layout};
    }
    callee->origin_count=count;callee->caller_bound=true;callee->invocation=2;
    return true;
}
bool nvm_affine_parameter_at(const NvmAffineState *s,uint16_t parameter,
                               NvmAffineType *type,NvmReferenceMode *mode) {
    if(!s || !type || !mode || !s->facts->params ||
       s->facts->params>NVM_AFFINE_MAX_PARAMETERS || parameter>=s->facts->params ||
       parameter>=s->ref_count || !s->refs[parameter].live ||
       !s->facts->locals[parameter].mode) return false;
    Slot param=s->facts->locals[parameter];
    NvmReferencePlace place={1,parameter,param.layout,NULL,0,param.layout,(NvmReferenceMode)param.mode};
    if(!resource(s->facts,param) ||
       !nvm_reference_place_valid(&s->facts->layouts,param.layout,&place)) return false;
    *type=(NvmAffineType){param.tag,param.layout};*mode=(NvmReferenceMode)param.mode;
    return true;
}
/* I expand each reachable prior-index layout once. A larger parent index
 * propagates its maximum depth before I visit the child, including shared DAGs. */
static bool nested_result_tree(const Facts *facts,uint32_t root) {
    uint8_t *depth=calloc((size_t)root+1,sizeof(*depth));
    if (!depth) return false;
    depth[root]=1;
    for (uint32_t next=root+1;next>0;) {
        uint32_t index=--next;
        if (!depth[index]) continue;
        const NvmV2Layout *layout=&facts->layouts.items[index];
        if ((facts->flags[index]&(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE))!=
                (NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE) ||
            layout->kind!=NVM_V2_LAYOUT_STRUCT ||
            layout->field_count>NVM_AFFINE_MAX_RESULT_FIELDS) goto refused;
        for (uint16_t f=0;f<layout->field_count;f++) {
            const NvmV2LayoutField *field=&layout->fields[f];
            if (field->type_tag==TAG_STRUCT) {
                uint32_t child=field->nested_idx;
                if (child>=index || depth[index]>=NVM_AFFINE_MAX_RESULT_DEPTH)
                    goto refused;
                uint8_t child_depth=(uint8_t)(depth[index]+1);
                if (depth[child]<child_depth) depth[child]=child_depth;
            } else if ((field->type_tag!=TAG_INT && field->type_tag!=TAG_BOOL &&
                        field->type_tag!=TAG_U8 && field->type_tag!=TAG_STRING) || field->nested_idx!=NVM_V2_NO_INDEX)
                goto refused;
        }
    }
    free(depth);return true;
refused:
    free(depth);return false;
}
bool nvm_affine_value_result(const NvmAffineState *s,NvmAffineType *type,
                              uint16_t *field_count) {
    if (!s || !type || !field_count) return false;
    Slot result=s->facts->result;
    const NvmFunctionEntry *fn=&s->facts->module->functions[s->facts->function];
    if (result.mode || fn->result_tag!=result.tag ||
        fn->result_count!=(result.tag==TAG_VOID?0:1)) return false;
    uint16_t fields=0;
    if (result.tag==TAG_STRUCT) {
        if (result.layout>=s->facts->layouts.count ||
            (s->facts->flags[result.layout]&(NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE))!=
                (NVM_LAYOUT_COMPLETE|NVM_LAYOUT_RESOURCE)) return false;
        const NvmV2Layout *layout=&s->facts->layouts.items[result.layout];
        if (layout->kind!=NVM_V2_LAYOUT_STRUCT) return false;
        bool nested=false;
        for (uint16_t i=0;i<layout->field_count;i++) {
            const NvmV2LayoutField *field=&layout->fields[i];
            if (field->type_tag==TAG_STRUCT) nested=true;
            else if ((field->type_tag!=TAG_INT && field->type_tag!=TAG_BOOL && field->type_tag!=TAG_U8 && field->type_tag!=TAG_STRING) ||
                     field->nested_idx!=NVM_V2_NO_INDEX) return false;
        }
        /* I preserve the allocation-free scalar-leaf query. */
        if (nested && !nested_result_tree(s->facts,result.layout)) return false;
        fields=layout->field_count;
    } else if (result.tag==TAG_UNION) {
        if (!union_value(s->facts,result)) return false;
    } else if ((result.tag!=TAG_VOID && result.tag!=TAG_INT &&
                result.tag!=TAG_BOOL && result.tag!=TAG_U8) ||
               result.layout!=NVM_V2_NO_INDEX) return false;
    *type=(NvmAffineType){result.tag,result.layout};*field_count=fields;
    return true;
}

bool nvm_affine_value_parameters(const NvmAffineState *s,NvmAffineType *types,
                                  uint16_t capacity,uint16_t *count) {
    if (!s || !types || !count || s->facts->params>NVM_AFFINE_MAX_PARAMETERS ||
        s->facts->params>capacity || s->facts->params>s->facts->count) return false;
    for (uint16_t p=0;p<s->facts->params;p++) {
        Slot parameter=s->facts->locals[p];
        if (parameter.mode) return false;
        if (parameter.tag==TAG_STRUCT) {
            if (!resource(s->facts,parameter) ||
                !(s->facts->flags[parameter.layout]&NVM_LAYOUT_COMPLETE)) return false;
        } else if (parameter.tag==TAG_STRING) {
            if (parameter.layout!=NVM_V2_NO_INDEX) return false;
        } else if (parameter.tag==TAG_UNION) {
            if (!union_value(s->facts,parameter)) return false;
        } else if (parameter.tag!=TAG_INT && parameter.tag!=TAG_BOOL && parameter.tag!=TAG_U8)
            return false;
    }
    for (uint16_t p=0;p<s->facts->params;p++)
        types[p]=(NvmAffineType){s->facts->locals[p].tag,s->facts->locals[p].layout};
    *count=s->facts->params;
    return true;
}
bool nvm_affine_consuming_parameters(const NvmAffineState *s,NvmAffineType *types,
                                      uint16_t capacity,uint16_t *count) {
    NvmAffineType checked[NVM_AFFINE_MAX_PARAMETERS];uint16_t length=0;
    if (!types || !count || !nvm_affine_value_parameters(s,checked,NVM_AFFINE_MAX_PARAMETERS,&length) ||
        length>capacity) return false;
    bool owned=false;
    for (uint16_t p=0;p<length;p++) if (checked[p].tag==TAG_STRUCT) owned=true;
    if (!owned) return false;
    memcpy(types,checked,length*sizeof(*types));*count=length;return true;
}
bool nvm_affine_owned_parameter_type(const NvmAffineState *s,NvmAffineType *type) {
    if (!s || !type || s->facts->params!=1 || !s->facts->count ||
        s->facts->locals[0].mode || !resource(s->facts,s->facts->locals[0]) ||
        !(s->facts->flags[s->facts->locals[0].layout]&NVM_LAYOUT_COMPLETE)) return false;
    *type=(NvmAffineType){s->facts->locals[0].tag,s->facts->locals[0].layout};
    return type->tag==TAG_STRUCT;
}
bool nvm_affine_parameter_type(const NvmAffineState *s,NvmAffineType *type,
                                 NvmReferenceMode *mode) {
    return s && s->facts->params==1 && nvm_affine_parameter_at(s,0,type,mode);
}

/* I keep mixed checked facts construction private; this query grants no execution. */
#include "mixed_samples.inc"

/* I compose private owner ARRAY lifetimes without changing shared admission. */
#include "owned_array_authority.inc"
