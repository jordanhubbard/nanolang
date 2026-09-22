#include "ownership_contracts.h"
#include "retained_layouts.h"
#include "reference_places.h"
#include "isa.h"
#include "preparation_budget.h"
#include "ownership_layouts_private.h"
#include <stdlib.h>

static bool scalar(uint8_t tag) {
    return tag == TAG_INT || tag == TAG_U8 || tag == TAG_FLOAT || tag == TAG_BOOL;
}

static bool ownership_version(uint32_t version) {
    return version == NVM_OWNERSHIP_VERSION ||
           version == NVM_OWNERSHIP_PATH_VERSION ||
           version == NVM_OWNERSHIP_EXTENSION_VERSION;
}

static NvmV2Result check_layouts(const NvmV2Layouts *layouts, const uint8_t *flags,
                                bool *needs, bool array_fields, bool typed) {
    bool resource_table = false;
    for (uint32_t i = 0; i < layouts->count; i++)
        if (flags[i] & NVM_LAYOUT_RESOURCE) resource_table = true;
    /* I preserve the old codec precondition for every resource-bearing table,
     * including edges in disconnected UNKNOWN declarations. */
    if (resource_table) for (uint32_t i = 0; i < layouts->count; i++)
        for (uint16_t j = 0; j < layouts->items[i].field_count; j++) {
            uint32_t child = layouts->items[i].fields[j].nested_idx;
            if (child != NVM_V2_NO_INDEX && child >= i) return NVM_V2_ERR_SECTION_TYPE;
        }
    for (uint32_t i = 0; i < layouts->count; i++) {
        unsigned flag = flags[i];
        if (flag & ~(NVM_LAYOUT_COMPLETE | NVM_LAYOUT_RESOURCE))
            return NVM_V2_ERR_RESERVED_FLAGS;
        if ((flag & NVM_LAYOUT_RESOURCE) && !(flag & NVM_LAYOUT_COMPLETE))
            return NVM_V2_ERR_SECTION_TYPE;
        if (!(flag & NVM_LAYOUT_COMPLETE)) continue;
        const NvmV2Layout *layout = &layouts->items[i];
        if (typed) {
            if(flag & NVM_LAYOUT_RESOURCE)return NVM_V2_ERR_SECTION_TYPE;
            continue; /* The typed structural decoder checked every field. */
        }
        if (layout->kind != NVM_V2_LAYOUT_STRUCT) return NVM_V2_ERR_SECTION_TYPE;
        for (uint16_t j = 0; j < layout->field_count; j++) {
            const NvmV2LayoutField *field = &layout->fields[j];
            if (scalar(field->type_tag) || field->type_tag == TAG_STRING) {
                if (field->nested_idx != NVM_V2_NO_INDEX) return NVM_V2_ERR_SECTION_TYPE;
            } else if (array_fields && field->type_tag == TAG_ARRAY && !(flag & NVM_LAYOUT_RESOURCE)) {
                if (field->nested_idx != NVM_V2_NO_INDEX) return NVM_V2_ERR_SECTION_TYPE;
            } else if (field->type_tag == TAG_STRUCT && field->nested_idx < layouts->count &&
                       (!resource_table || field->nested_idx < i)) {
                unsigned nested = flags[field->nested_idx];
                if (!(nested & NVM_LAYOUT_COMPLETE)) return NVM_V2_ERR_SECTION_TYPE;
                if ((nested & NVM_LAYOUT_RESOURCE) && !(flag & NVM_LAYOUT_RESOURCE))
                    return NVM_V2_ERR_SECTION_TYPE;
            } else return NVM_V2_ERR_SECTION_TYPE;
        }
        /* STRING leaves retain value roots; executable and borrowed profiles
         * independently constrain these complete resource declarations. */
        if (flag & NVM_LAYOUT_RESOURCE) *needs = true;
    }
    return NVM_V2_OK;
}

static NvmV2Result descriptor(NvmV2Cursor *cursor, const NvmV2Layouts *layouts,
                               const uint8_t *flags, uint32_t version, bool parameter,
                               int signature_tag, bool *needs, bool *ambiguous, bool typed, bool *foreign_unresolved) {
    uint8_t tag, mode;
    uint16_t reserved;
    uint32_t layout;
    NvmV2Result result;
    if ((result = nvm_v2_u8(cursor, &tag)) != NVM_V2_OK ||
        (result = nvm_v2_u8(cursor, &mode)) != NVM_V2_OK ||
        (result = nvm_v2_u16(cursor, &reserved)) != NVM_V2_OK ||
        (result = nvm_v2_u32(cursor, &layout)) != NVM_V2_OK) return result;
    if (reserved) return NVM_V2_ERR_RESERVED_FLAGS;
    if (tag >= TAG_COUNT || mode > NVM_REFERENCE_EXCLUSIVE ||
        (!parameter && mode) || (signature_tag >= 0 && tag != signature_tag))
        return NVM_V2_ERR_SECTION_TYPE;
    if(typed && nvm_ownership_nominal_layout_kind(tag)>=0 && layout==NVM_V2_NO_INDEX)
        return NVM_V2_ERR_SECTION_TYPE;
    if(typed && foreign_unresolved && (tag==TAG_FUNCTION || tag==TAG_OPAQUE))
        *foreign_unresolved=true;
    if (layout != NVM_V2_NO_INDEX) {
        if (layout >= layouts->count) return NVM_V2_ERR_INDEX_RANGE;
        bool record=tag==TAG_STRUCT && (flags[layout]&NVM_LAYOUT_COMPLETE) &&
                    layouts->items[layout].kind==NVM_V2_LAYOUT_STRUCT;
        bool union_value=version==NVM_OWNERSHIP_EXTENSION_VERSION && tag==TAG_UNION &&
                         !flags[layout] && layouts->items[layout].kind==NVM_V2_LAYOUT_UNION;
        bool typed_value=typed && nvm_ownership_nominal_layout_kind(tag)>=0 &&
            layouts->items[layout].kind==(uint8_t)nvm_ownership_nominal_layout_kind(tag);
        if (!record && !union_value && !typed_value)
            return NVM_V2_ERR_SECTION_TYPE;
        if (union_value) *needs=true;
    }
    if (mode) {
        if (layout == NVM_V2_NO_INDEX || !(flags[layout] & NVM_LAYOUT_RESOURCE))
            return NVM_V2_ERR_SECTION_TYPE;
        NvmReferencePlace place = {1, 0, layout, NULL, 0, layout, (NvmReferenceMode)mode};
        if (!nvm_reference_place_valid(layouts, layout, &place)) {
            if (ambiguous) *ambiguous=true;
            return NVM_V2_ERR_SECTION_TYPE;
        }
        *needs = true;
    }
    return NVM_V2_OK;
}

/* I share exact path-table transport validation with runtime lookup. */
static NvmV2Result paths_read(NvmV2Cursor *cursor,uint32_t wanted,
                               uint16_t *fields,uint16_t capacity,uint16_t *length,
                               bool terminal) {
    uint32_t count;NvmV2Result result;
    uint16_t selected[NVM_OWNERSHIP_MAX_PATH_DEPTH],selected_count=0;
    if ((result=nvm_v2_u32(cursor,&count))!=NVM_V2_OK) return result;
    if (count>NVM_OWNERSHIP_MAX_PATHS) return NVM_V2_ERR_INDEX_RANGE;
    for (uint32_t i=0;i<count;i++) {
        uint16_t n,reserved;
        if ((result=nvm_v2_u16(cursor,&n))!=NVM_V2_OK ||
            (result=nvm_v2_u16(cursor,&reserved))!=NVM_V2_OK) return result;
        if (reserved) return NVM_V2_ERR_RESERVED_FLAGS;
        if (!n || n>NVM_OWNERSHIP_MAX_PATH_DEPTH) return NVM_V2_ERR_INDEX_RANGE;
        for (uint16_t j=0;j<n;j++) {
            uint16_t field;
            if ((result=nvm_v2_u16(cursor,&field))!=NVM_V2_OK) return result;
            if (i==wanted) selected[j]=field;
        }
        if (i==wanted) selected_count=n;
        if ((result=nvm_v2_align4(cursor))!=NVM_V2_OK) return result;
    }
    if (terminal && cursor->pos!=cursor->size) return NVM_V2_ERR_SECTION_RANGE;
    if (wanted!=NVM_V2_NO_INDEX) {
        if (!selected_count || selected_count>capacity || !fields || !length)
            return NVM_V2_ERR_INDEX_RANGE;
        for (uint16_t i=0;i<selected_count;i++) fields[i]=selected[i];
        *length=selected_count;
    }
    return NVM_V2_OK;
}

typedef struct {
    const uint8_t *data;
    uint32_t size;
} NvmOwnershipExtensionView;

typedef struct {
    NvmOwnershipExtensionView union_variants;
    NvmOwnershipExtensionView array_fields;
    uint16_t array_revision;
} NvmOwnershipExtensions;

/* I frame all version-3 extensions before a feature query may inspect one.
 * Kinds are mandatory-understanding, ordered and unique. */
static NvmV2Result extensions_read(NvmV2Cursor *cursor,NvmOwnershipExtensions *out,bool typed) {
    uint32_t count;NvmV2Result result;
    NvmOwnershipExtensions found={0};uint16_t prior=0;
    if ((result=nvm_v2_u32(cursor,&count))!=NVM_V2_OK) return result;
    if (!count || count>NVM_OWNERSHIP_MAX_EXTENSIONS) return NVM_V2_ERR_INDEX_RANGE;
    for (uint32_t i=0;i<count;i++) {
        uint16_t kind,revision;uint32_t bytes;const uint8_t *payload;
        if ((result=nvm_v2_u16(cursor,&kind))!=NVM_V2_OK ||
            (result=nvm_v2_u16(cursor,&revision))!=NVM_V2_OK ||
            (result=nvm_v2_u32(cursor,&bytes))!=NVM_V2_OK ||
            (result=nvm_v2_take(cursor,bytes,&payload))!=NVM_V2_OK ||
            (result=nvm_v2_align4(cursor))!=NVM_V2_OK) return result;
        if (!kind || kind<=prior) return NVM_V2_ERR_SECTION_TYPE;
        if (revision!=NVM_OWNERSHIP_EXTENSION_REVISION_1 &&
            !(typed && kind==NVM_OWNERSHIP_EXTENSION_ARRAY_FIELDS && revision==2))
            return NVM_V2_ERR_FORMAT_VERSION;
        NvmOwnershipExtensionView view={payload,bytes};
        if (kind==NVM_OWNERSHIP_EXTENSION_UNION_VARIANTS)
            found.union_variants=view;
        else if (kind==NVM_OWNERSHIP_EXTENSION_ARRAY_FIELDS) {
            found.array_fields=view;found.array_revision=revision;
        }
        else return NVM_V2_ERR_FORMAT_VERSION;
        prior=kind;
    }
    if (cursor->pos!=cursor->size) return NVM_V2_ERR_SECTION_RANGE;
    if (out) *out=found;
    return NVM_V2_OK;
}

/* Version 3 bounds the byte-identical version-2 path suffix before its shared
 * extension records. The path subcursor, not the outer payload, is terminal. */
static NvmV2Result extension_suffix_read(NvmV2Cursor *cursor,uint32_t wanted,
                                         uint16_t *fields,uint16_t capacity,
                                         uint16_t *length,NvmOwnershipExtensions *extensions,bool typed) {
    uint32_t bytes;const uint8_t *data;NvmV2Result result;
    if ((result=nvm_v2_u32(cursor,&bytes))!=NVM_V2_OK) return result;
    if (bytes<4 || bytes%4) return NVM_V2_ERR_SECTION_RANGE;
    if ((result=nvm_v2_take(cursor,bytes,&data))!=NVM_V2_OK) return result;
    NvmV2Cursor paths;nvm_v2_cursor_init(&paths,data,bytes);
    if ((result=paths_read(&paths,wanted,fields,capacity,length,true))!=NVM_V2_OK)
        return result;
    return extensions_read(cursor,extensions,typed);
}

NvmV2Result nvm_ownership_path(const NvmModule *module,uint32_t index,
                               uint16_t *fields,uint16_t capacity,uint16_t *count) {
    if (!module || !module->ownership_data || index==NVM_V2_NO_INDEX)
        return NVM_V2_ERR_INDEX_RANGE;
    bool needs=false;NvmV2Result result=nvm_ownership_contracts_validate(module,&needs);
    if (result!=NVM_V2_OK) return result;
    NvmV2Cursor cursor;
    nvm_v2_cursor_init(&cursor,module->ownership_data,module->ownership_size);
    uint32_t version,layouts,functions;const uint8_t *ignored;
    if ((result=nvm_v2_u32(&cursor,&version))!=NVM_V2_OK) return result;
    if (version!=NVM_OWNERSHIP_PATH_VERSION && version!=NVM_OWNERSHIP_EXTENSION_VERSION)
        return NVM_V2_ERR_FORMAT_VERSION;
    if ((result=nvm_v2_u32(&cursor,&layouts))!=NVM_V2_OK ||
        (result=nvm_v2_take(&cursor,layouts,&ignored))!=NVM_V2_OK ||
        (result=nvm_v2_align4(&cursor))!=NVM_V2_OK ||
        (result=nvm_v2_u32(&cursor,&functions))!=NVM_V2_OK) return result;
    if (functions!=module->function_count) return NVM_V2_ERR_INDEX_RANGE;
    for (uint32_t i=0;i<functions;i++) {
        uint16_t locals,params;
        if ((result=nvm_v2_u16(&cursor,&locals))!=NVM_V2_OK ||
            (result=nvm_v2_u16(&cursor,&params))!=NVM_V2_OK) return result;
        if (params>locals) return NVM_V2_ERR_INDEX_RANGE;
        if ((result=nvm_v2_take(&cursor,((size_t)locals+1)*8,&ignored))!=NVM_V2_OK) return result;
    }
    uint16_t selected[NVM_OWNERSHIP_MAX_PATH_DEPTH],selected_count=0;
    if (version==NVM_OWNERSHIP_PATH_VERSION)
        result=paths_read(&cursor,index,selected,NVM_OWNERSHIP_MAX_PATH_DEPTH,
                          &selected_count,true);
    else
        result=extension_suffix_read(&cursor,index,selected,NVM_OWNERSHIP_MAX_PATH_DEPTH,
                                     &selected_count,NULL,false);
    if (result!=NVM_V2_OK) return result;
    if (!fields || !count || selected_count>capacity) return NVM_V2_ERR_INDEX_RANGE;
    for (uint16_t i=0;i<selected_count;i++) fields[i]=selected[i];
    *count=selected_count;
    return NVM_V2_OK;
}

/* I borrow exact module facts without rebuilding signatures or string tables. */
typedef struct {
    const NvmModule *legacy;
    const NvmV2Module *v2;
    uint32_t function_count, union_count;
    const uint8_t *ownership_data;
    size_t ownership_size;
} OwnershipModuleFacts;
typedef struct {
    uint16_t locals, params, results;
    uint8_t result_tag;
    const uint8_t *parameter_tags;
} OwnershipFunctionFacts;
static OwnershipModuleFacts ownership_legacy_facts(const NvmModule *m) {
    return (OwnershipModuleFacts){m,NULL,m->function_count,m->union_count,
                                 m->ownership_data,m->ownership_size};
}
static bool ownership_name_valid(const OwnershipModuleFacts *m,uint32_t i) {
    if(m->legacy)return i<m->legacy->string_count;
    if(i>=m->v2->constants.count || !m->v2->constants.items)return false;
    const NvmV2Constant *c=&m->v2->constants.items[i];
    return c->tag==TAG_STRING && (!c->length || c->payload);
}
static bool ownership_function_facts(const OwnershipModuleFacts *m,uint32_t i,
                                     OwnershipFunctionFacts *out) {
    if(i>=m->function_count)return false;
    if(m->legacy) {
        if(!m->legacy->functions)return false;
        const NvmFunctionEntry *f=&m->legacy->functions[i];
        *out=(OwnershipFunctionFacts){f->local_count,f->arity,f->result_count,
            f->result_tag,m->legacy->function_param_types?m->legacy->function_param_types[i]:NULL};
        return true;
    }
    if(!m->v2->functions.items || !m->v2->signatures.items)return false;
    const NvmV2Function *f=&m->v2->functions.items[i];
    if(f->signature_idx>=m->v2->signatures.count)return false;
    const NvmV2Signature *sig=&m->v2->signatures.items[f->signature_idx];
    if((sig->param_count&&!sig->param_tags)||(sig->result_count&&!sig->result_tags))return false;
    *out=(OwnershipFunctionFacts){f->local_count,sig->param_count,sig->result_count,
        sig->result_count?sig->result_tags[0]:TAG_VOID,sig->param_tags};
    return true;
}

/* I collect only checked numeric facts during one complete declaration read. */
typedef struct {
    NvmOwnershipExtensionView view;
    NvmUnionVariantFact *rows;
    uint32_t capacity, count;
    uint32_t *starts;
    uint16_t *variants;
    bool ambiguous, typed;
} OwnershipProjection;
static NvmV2Result union_facts_read(NvmV2Cursor *cursor,const OwnershipModuleFacts *module,
                                    const NvmV2Layouts *layouts,
                                    uint32_t wanted_union,uint16_t wanted_variant,
                                    NvmUnionVariantFact *selected, OwnershipProjection *projection) {
    uint32_t count;NvmV2Result result;
    NvmUnionVariantFact found={0};bool have=false;
    if ((result=nvm_v2_u32(cursor,&count))!=NVM_V2_OK) return result;
    if (count!=module->union_count || count>NVM_OWNERSHIP_MAX_UNIONS)
        return NVM_V2_ERR_INDEX_RANGE;
    uint32_t ordinal=0, copied=0;
    for (uint32_t i=0;i<count;i++) {
        uint32_t layout;uint16_t variants,reserved;
        if ((result=nvm_v2_u32(cursor,&layout))!=NVM_V2_OK ||
            (result=nvm_v2_u16(cursor,&variants))!=NVM_V2_OK ||
            (result=nvm_v2_u16(cursor,&reserved))!=NVM_V2_OK) return result;
        while (ordinal<layouts->count && layouts->items[ordinal].kind!=NVM_V2_LAYOUT_UNION)
            ordinal++;
        if (reserved || !variants || variants>NVM_OWNERSHIP_MAX_VARIANTS ||
            ordinal>=layouts->count || layout!=ordinal ||
            layouts->items[layout].kind!=NVM_V2_LAYOUT_UNION ||
            layouts->items[layout].name_idx==NVM_V2_NO_INDEX)
            return NVM_V2_ERR_SECTION_TYPE;
        const NvmV2Layout *shape=&layouts->items[layout];
        if (projection && projection->rows) {
            if (copied>projection->capacity || variants>projection->capacity-copied)
                return NVM_V2_ERR_INDEX_RANGE;
            projection->starts[i]=copied;
            projection->variants[i]=variants;
        }
        size_t variants_at=cursor->pos;
        uint32_t next=0;
        for (uint16_t v=0;v<variants;v++) {
            uint32_t name;uint16_t offset,fields;
            if ((result=nvm_v2_u32(cursor,&name))!=NVM_V2_OK ||
                (result=nvm_v2_u16(cursor,&offset))!=NVM_V2_OK ||
                (result=nvm_v2_u16(cursor,&fields))!=NVM_V2_OK) return result;
            if (!ownership_name_valid(module,name) || offset!=next ||
                fields>shape->field_count-offset)
                return NVM_V2_ERR_INDEX_RANGE;
            for (uint16_t prior=0;prior<v;prior++) {
                size_t prior_at=variants_at+(size_t)prior*8;
                if (prior_at>cursor->size || cursor->size-prior_at<4)
                    return NVM_V2_ERR_SECTION_RANGE;
                uint32_t prior_name=(uint32_t)cursor->base[prior_at] |
                    ((uint32_t)cursor->base[prior_at+1]<<8) |
                    ((uint32_t)cursor->base[prior_at+2]<<16) |
                    ((uint32_t)cursor->base[prior_at+3]<<24);
                if (prior_name==name) return NVM_V2_ERR_SECTION_TYPE;
            }
            for (uint16_t f=0;f<fields;f++) {
                const NvmV2LayoutField *field=&shape->fields[offset+f];
                if ((!(projection && projection->typed) &&
                    ((!scalar(field->type_tag) && field->type_tag!=TAG_STRING) ||
                     field->nested_idx!=NVM_V2_NO_INDEX)) ||
                    field->name_idx==NVM_V2_NO_INDEX)
                    return NVM_V2_ERR_SECTION_TYPE;
            }
            if (projection && projection->rows)
                projection->rows[copied]=(NvmUnionVariantFact){layout,name,offset,fields};
            copied++;
            if (i==wanted_union && v==wanted_variant) {
                found=(NvmUnionVariantFact){layout,name,offset,fields};have=true;
            }
            next=(uint32_t)offset+fields;
        }
        if (next!=shape->field_count) return NVM_V2_ERR_INDEX_RANGE;
        ordinal++;
    }
    while (ordinal<layouts->count) {
        if (layouts->items[ordinal].kind==NVM_V2_LAYOUT_UNION)
            return NVM_V2_ERR_INDEX_RANGE;
        ordinal++;
    }
    if (cursor->pos!=cursor->size) return NVM_V2_ERR_SECTION_RANGE;
    if (wanted_union!=NVM_V2_NO_INDEX && !have) return NVM_V2_ERR_INDEX_RANGE;
    if (have && selected) *selected=found;
    if (projection) projection->count=copied;
    return NVM_V2_OK;
}

#include "ownership_array_fields.inc"
static NvmV2Result ownership_validate_facts(const OwnershipModuleFacts *module,
        const NvmV2Layouts *layouts,bool *requires_verifier,
        NvmOrdinaryArrayAuthority *private_plan, OwnershipProjection *projection) {
    NvmV2Result result;
    bool ambiguous=false;
    NvmV2Cursor cursor;
    nvm_v2_cursor_init(&cursor, module->ownership_data, module->ownership_size);
    uint32_t version, count;
    const uint8_t *flags;
    bool needs = false;
    if ((result = nvm_v2_u32(&cursor, &version)) != NVM_V2_OK ||
        (result = nvm_v2_u32(&cursor, &count)) != NVM_V2_OK) goto done;
    if (!ownership_version(version)) { result = NVM_V2_ERR_FORMAT_VERSION; goto done; }
    if (count != layouts->count) { result = NVM_V2_ERR_INDEX_RANGE; goto done; }
    if ((result = nvm_v2_take(&cursor, count, &flags)) != NVM_V2_OK ||
        (result = nvm_v2_align4(&cursor)) != NVM_V2_OK ||
        (result = check_layouts(layouts, flags, &needs, private_plan!=NULL, private_plan && private_plan->typed)) != NVM_V2_OK ||
        (result = nvm_v2_u32(&cursor, &count)) != NVM_V2_OK) goto done;
    if (private_plan) for (uint32_t i=0;i<layouts->count;i++) private_plan->flags[i]=flags[i];
    if (count != module->function_count) {
        result = NVM_V2_ERR_INDEX_RANGE; goto done;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint16_t locals, params;
        OwnershipFunctionFacts function;
        if(!ownership_function_facts(module,i,&function)) { result=NVM_V2_ERR_INDEX_RANGE;goto done; }
        if ((result = nvm_v2_u16(&cursor, &locals)) != NVM_V2_OK ||
            (result = nvm_v2_u16(&cursor, &params)) != NVM_V2_OK) goto done;
        if (locals != function.locals || params != function.params || params > locals ||
            function.results > 1) { result = NVM_V2_ERR_INDEX_RANGE; goto done; }
        int return_tag = function.results ? function.result_tag : TAG_VOID;
        ambiguous=false;
        result=descriptor(&cursor,layouts,flags,version,false,return_tag,&needs,private_plan?&ambiguous:NULL,private_plan && private_plan->typed,private_plan?&private_plan->foreign_unresolved:NULL);
        if (result!=NVM_V2_OK) {
            if (!projection || !ambiguous) goto done;
            projection->ambiguous=true;result=NVM_V2_OK;
        }
        for (uint16_t local = 0; local < locals; local++) {
            int tag = -1;
            if (local < params) tag = function.parameter_tags ? function.parameter_tags[local] : TAG_VOID;
            ambiguous=false;
            result=descriptor(&cursor,layouts,flags,version,local<params,tag,&needs,private_plan?&ambiguous:NULL,private_plan && private_plan->typed,private_plan?&private_plan->foreign_unresolved:NULL);
            if (result!=NVM_V2_OK) {
                if (!projection || !ambiguous) goto done;
                projection->ambiguous=true;result=NVM_V2_OK;
            }
        }
    }
    NvmOwnershipExtensions extensions={0};
    if (version==NVM_OWNERSHIP_PATH_VERSION) {
        if ((result=paths_read(&cursor,NVM_V2_NO_INDEX,NULL,0,NULL,true))!=NVM_V2_OK)
            goto done;
    } else if (version==NVM_OWNERSHIP_EXTENSION_VERSION) {
        if ((result=extension_suffix_read(&cursor,NVM_V2_NO_INDEX,NULL,0,NULL,
                                          &extensions,private_plan && private_plan->typed))!=NVM_V2_OK) goto done;
        if (!!extensions.union_variants.data != !!module->union_count) {
            result=NVM_V2_ERR_SECTION_TYPE;goto done;
        }
        if (extensions.union_variants.data) {
            NvmV2Cursor unions;nvm_v2_cursor_init(&unions,extensions.union_variants.data,
                                                  extensions.union_variants.size);
            if ((result=union_facts_read(&unions,module,layouts,NVM_V2_NO_INDEX,0,NULL,projection))
                !=NVM_V2_OK) goto done;
            if (projection) projection->view=extensions.union_variants;
            needs=true;
        }
        if (extensions.array_fields.data) {
            if (!private_plan) { result=NVM_V2_ERR_FORMAT_VERSION;goto done; }
            if ((result=oaa_array_fields(extensions.array_fields,private_plan,extensions.array_revision))!=NVM_V2_OK) goto done;
        }
    }
    if (cursor.pos != cursor.size) { result = NVM_V2_ERR_SECTION_RANGE; goto done; }
    *requires_verifier = needs;
done:
    if (private_plan && ambiguous && !projection) private_plan->failure=NVM_OAA_UNKNOWN;
    return result;
}

static NvmV2Result ownership_validate_owned(const NvmModule *module,
        const NvmV2Layouts *layouts,bool *requires_verifier,
        NvmOrdinaryArrayAuthority *private_plan, OwnershipProjection *projection) {
    OwnershipModuleFacts facts=ownership_legacy_facts(module);
    return ownership_validate_facts(&facts,layouts,requires_verifier,private_plan,projection);
}

NvmV2Result nvm_ownership_contracts_validate(const NvmModule *module,
                                            bool *requires_verifier) {
    if (!module || !requires_verifier) return NVM_V2_ERR_INDEX_RANGE;
    *requires_verifier = false;
    if (!module->ownership_size)
        return module->ownership_data ? NVM_V2_ERR_SECTION_RANGE : NVM_V2_OK;
    if (!module->ownership_data || !module->layout_size ||
        !nvm_retained_layouts_valid(module)) return NVM_V2_ERR_SECTION_TYPE;
    NvmV2Layouts layouts = {0};
    NvmV2Result result = nvm_v2_layouts_decode(module->layout_data,
                                             module->layout_size, &layouts);
    if (result != NVM_V2_OK) return result;
    result=ownership_validate_owned(module,&layouts,requires_verifier,NULL,NULL);
    nvm_v2_layouts_free(&layouts);
    return result;
}

NvmV2Result nvm_ownership_union_variant(const NvmModule *module,
                                        uint32_t union_ordinal,
                                        uint16_t variant,
                                        NvmUnionVariantFact *out) {
    if (!module || !out || !module->ownership_data || !module->layout_data)
        return NVM_V2_ERR_INDEX_RANGE;
    bool needs=false;
    NvmV2Result result=nvm_ownership_contracts_validate(module,&needs);
    if (result!=NVM_V2_OK) return result;
    NvmV2Layouts layouts={0};
    result=nvm_v2_layouts_decode(module->layout_data,module->layout_size,&layouts);
    if (result!=NVM_V2_OK) return result;
    NvmV2Cursor cursor;
    nvm_v2_cursor_init(&cursor,module->ownership_data,module->ownership_size);
    uint32_t version,count;const uint8_t *ignored;
    if ((result=nvm_v2_u32(&cursor,&version))!=NVM_V2_OK ||
        version!=NVM_OWNERSHIP_EXTENSION_VERSION ||
        (result=nvm_v2_u32(&cursor,&count))!=NVM_V2_OK ||
        (result=nvm_v2_take(&cursor,count,&ignored))!=NVM_V2_OK ||
        (result=nvm_v2_align4(&cursor))!=NVM_V2_OK ||
        (result=nvm_v2_u32(&cursor,&count))!=NVM_V2_OK) {
        if (result==NVM_V2_OK) result=NVM_V2_ERR_FORMAT_VERSION;
        goto done_query;
    }
    for (uint32_t i=0;i<count;i++) {
        uint16_t locals,params;
        if ((result=nvm_v2_u16(&cursor,&locals))!=NVM_V2_OK ||
            (result=nvm_v2_u16(&cursor,&params))!=NVM_V2_OK ||
            (result=nvm_v2_take(&cursor,((size_t)locals+1)*8,&ignored))!=NVM_V2_OK)
            goto done_query;
    }
    NvmOwnershipExtensions extensions={0};
    if ((result=extension_suffix_read(&cursor,NVM_V2_NO_INDEX,NULL,0,NULL,&extensions,false))
        !=NVM_V2_OK) goto done_query;
    if (!extensions.union_variants.data) { result=NVM_V2_ERR_FORMAT_VERSION;goto done_query; }
    NvmV2Cursor unions;nvm_v2_cursor_init(&unions,extensions.union_variants.data,
                                          extensions.union_variants.size);
    NvmUnionVariantFact selected;
    OwnershipModuleFacts facts=ownership_legacy_facts(module);
    result=union_facts_read(&unions,&facts,&layouts,union_ordinal,variant,&selected,NULL);
    if (result==NVM_V2_OK) *out=selected;
done_query:
    nvm_v2_layouts_free(&layouts);
    return result;
}

NvmV2Result nvm_ownership_layout_authority(const NvmModule *module, uint32_t layout,
                                          NvmLayoutAuthority *out) {
    if (!module || !out || layout == NVM_V2_NO_INDEX) return NVM_V2_ERR_INDEX_RANGE;
    bool needs;
    NvmV2Result result = nvm_ownership_contracts_validate(module, &needs);
    if (result != NVM_V2_OK) return result;
    NvmLayoutAuthority authority = NVM_LAYOUT_AUTHORITY_UNKNOWN;
    if (module->ownership_size) {
        NvmV2Cursor cursor;
        uint32_t version, count;
        const uint8_t *flags;
        nvm_v2_cursor_init(&cursor, module->ownership_data, module->ownership_size);
        if ((result = nvm_v2_u32(&cursor, &version)) != NVM_V2_OK ||
            (result = nvm_v2_u32(&cursor, &count)) != NVM_V2_OK ||
            (result = nvm_v2_take(&cursor, count, &flags)) != NVM_V2_OK) return result;
        if (layout >= count) return NVM_V2_ERR_INDEX_RANGE;
        if (flags[layout] & NVM_LAYOUT_COMPLETE)
            authority = flags[layout] & NVM_LAYOUT_RESOURCE ?
                NVM_LAYOUT_AUTHORITY_RESOURCE : NVM_LAYOUT_AUTHORITY_ORDINARY;
    }
    *out = authority;
    return NVM_V2_OK;
}

NvmV2Result nvm_ownership_layout_authorities(const NvmModule *module, uint32_t count,
                                            NvmLayoutAuthority *out) {
    if (!module || (count && !out)) return NVM_V2_ERR_INDEX_RANGE;
    bool needs;
    NvmV2Result result = nvm_ownership_contracts_validate(module, &needs);
    if (result != NVM_V2_OK) return result;
    const uint8_t *flags = NULL;
    if (module->ownership_size) {
        NvmV2Cursor cursor;
        uint32_t version, declared_count;
        nvm_v2_cursor_init(&cursor, module->ownership_data, module->ownership_size);
        if ((result = nvm_v2_u32(&cursor, &version)) != NVM_V2_OK ||
            (result = nvm_v2_u32(&cursor, &declared_count)) != NVM_V2_OK) return result;
        if (declared_count != count) return NVM_V2_ERR_INDEX_RANGE;
        if ((result = nvm_v2_take(&cursor, count, &flags)) != NVM_V2_OK) return result;
    }
    for (uint32_t i = 0; i < count; i++)
        out[i] = flags && (flags[i] & NVM_LAYOUT_COMPLETE) ?
            (flags[i] & NVM_LAYOUT_RESOURCE ? NVM_LAYOUT_AUTHORITY_RESOURCE :
             NVM_LAYOUT_AUTHORITY_ORDINARY) : NVM_LAYOUT_AUTHORITY_UNKNOWN;
    return NVM_V2_OK;
}

#include "ordinary_array_authority.inc"

#include "ownership_declaration_projection.inc"
