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
    uint16_t count;
} Facts;
typedef struct {
    bool live;
    uint32_t region, parent;
    NvmReferencePlace place;
} Reference;
struct NvmAffineState {
    Facts *facts;
    bool *live;
    Reference *refs;
    uint32_t ref_count, region;
};
static bool scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_U8 || tag == TAG_FLOAT || tag == TAG_BOOL;
}
static bool resource(const Facts *f, Slot slot) {
    return slot.layout != NVM_V2_NO_INDEX &&
           (f->flags[slot.layout] & NVM_LAYOUT_RESOURCE);
}
static bool supported(Slot slot) {
    return scalar(slot.tag) ||
           (slot.tag == TAG_STRUCT && slot.layout != NVM_V2_NO_INDEX);
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
    Facts *f=s->facts; f->users=1;
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
            if (result.tag != TAG_VOID && !supported(result)) goto fail;
            f->count=locals; f->result=result;
            f->locals=calloc(locals ? locals : 1,sizeof(*f->locals));
            if (!f->locals) goto fail;
        }
        for (uint16_t j=0;j<locals;j++) {
            Slot slot; if (!slot_read(&c,&slot)) goto fail;
            if (i==function) {
                /* Unknown/collection declarations cannot establish absence
                 * of an ownership obligation in this bounded analysis. */
                if (!supported(slot)) goto fail;
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
bool nvm_affine_state_equal(const NvmAffineState *a, const NvmAffineState *b) {
    if (!a || !b || a->facts!=b->facts || a->region!=b->region ||
        a->ref_count!=b->ref_count || memcmp(a->live,b->live,a->facts->count*sizeof(*a->live))) return false;
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
    *out=slot; return scalar(slot.tag) || slot.layout!=NVM_V2_NO_INDEX;
}
bool nvm_affine_owner_access(const NvmAffineState *s,uint16_t local,
                              const uint16_t *path,uint16_t count,bool write) {
    Slot slot;
    if (!value_local(s,local) || !s->live[local] || !resolve(s,local,path,count,&slot)) return false;
    NvmReferencePlace place={1,local,s->facts->locals[local].layout,path,count,
                              slot.layout,NVM_REFERENCE_SHARED};
    for (uint32_t i=0;i<s->ref_count;i++) if (s->refs[i].live &&
        nvm_reference_owner_access_conflicts(&s->refs[i].place,&place,write)) return false;
    return true;
}
static bool destination(const NvmAffineState *s,uint16_t local) {
    if (!value_local(s,local)) return false;
    if (!s->live[local]) return true;
    return !resource(s->facts,s->facts->locals[local]) &&
           nvm_affine_owner_access(s,local,NULL,0,true);
}
bool nvm_affine_scalar_define(NvmAffineState *s,uint16_t local) {
    if (!destination(s,local) || !scalar(s->facts->locals[local].tag)) return false;
    s->live[local]=true; return true;
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
static bool borrow_install(NvmAffineState *s,uint32_t id,NvmReferencePlace place,uint32_t parent) {
    if (!s || !s->region || id>=s->ref_count || s->refs[id].live ||
        !nvm_reference_place_valid(&s->facts->layouts,s->facts->locals[place.local].layout,&place) ||
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
    return borrow_install(s,id,(NvmReferencePlace){1,root,s->facts->locals[root].layout,
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
    if (!slot.mode && !nvm_affine_owner_access(s,local,NULL,0,false)) return false;
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
bool nvm_affine_can_exit_type(const NvmAffineState *s,NvmAffineType type) {
    if (!s || s->region || !same(s->facts->result,(Slot){type.tag,0,type.layout})) return false;
    for (uint16_t i=0;i<s->facts->count;i++) if (s->live[i] &&
        !s->facts->locals[i].mode && resource(s->facts,s->facts->locals[i])) return false;
    return true;
}
