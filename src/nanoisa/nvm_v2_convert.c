/*
 * The NvmModule <-> v2 bridge.
 *
 * This is what lets v2 be adopted without rewriting every producer at once:
 * codegen keeps building the v1 in-memory module it already builds, and this
 * converts it. The interesting direction is v1 -> v2, because that is where
 * the structural differences surface.
 *
 * Three of them matter:
 *
 *  - v1's string pool becomes CONSTANTS entries tagged TAG_STRING, carrying
 *    the stored length rather than strlen, so an embedded zero survives.
 *  - v1 repeats a call shape at every function and import. v2 names each
 *    distinct shape once in SIGNATURES and references it by index, so this
 *    must deduplicate exactly -- if two identically-shaped callables get
 *    different indices, comparing signature indices stops meaning "same type",
 *    which is the property the verifier is meant to gain from v2.
 *  - Legacy producers can omit function parameter types. I preserve typed
 *    producers' declarations and retain TAG_VOID placeholders only where
 *    types are unknown. I derive max_stack when verification succeeds.
 *
 * Constant payloads and import tag arrays alias the source NvmModule, so it
 * must outlive the NvmV2Module the bridge produces.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "nvm_format.h"
#include "isa.h"
#include "verifier.h"
#include "passive.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"

/* v1 keeps the source filename as a string-pool index outside every table. v2
 * has no such field, so it travels as a metadata pair under this key -- which
 * is what METADATA is for. Losing it silently would be a worse answer than
 * spending one constant on it. */
static const char SOURCE_FILE_KEY[] = "nano.source_file";

/* ── v1 -> v2 ───────────────────────────────────────────────────────────── */

/* Append `sig` to `sigs`, or return the index of an identical entry already
 * there. `pool`/`pool_used` supply storage for tag arrays; a duplicate rewinds
 * the pool so the bytes it would have used are not stranded. */
static uint32_t intern_signature(NvmV2Signatures *sigs, const NvmV2Signature *sig,
                                 size_t *pool_used, size_t pool_mark) {
    for (uint32_t i = 0; i < sigs->count; i++) {
        if (nvm_v2_signature_equal(&sigs->items[i], sig)) {
            *pool_used = pool_mark;
            return i;
        }
    }
    sigs->items[sigs->count] = *sig;
    return sigs->count++;
}

NvmV2Result nvm_v2_from_nvm_module(const NvmModule *mod, NvmV2Module *out) {
    if (!mod || !out) return NVM_V2_ERR_INDEX_RANGE;
    memset(out, 0, sizeof *out);
    out->isa_version = NVM_V2_ISA_VERSION;
    if (!nvm_metadata_valid(mod) || !nvm_callback_contracts_valid(mod) || !nvm_passive_valid(mod) ||
        !nvm_retained_layouts_valid(mod)) return NVM_V2_ERR_INDEX_RANGE;
    bool needs_ownership = false;
    NvmV2Result ownership = nvm_ownership_contracts_validate(mod, &needs_ownership);
    if (ownership != NVM_V2_OK) return ownership;
    out->ownership_data = mod->ownership_data;
    out->ownership_size = mod->ownership_size;
    out->passive_data = mod->passive_data;
    out->passive_size = mod->passive_size;

    const uint32_t n_fn = mod->function_count;
    const uint32_t n_im = mod->import_count;
    const uint32_t n_cb = mod->callback_contract_count;
    const uint32_t n_lk = mod->module_ref_count;
    const uint32_t n_db = mod->debug_count;
    const bool has_source = mod->source_file_idx != 0 &&
                            mod->source_file_idx < mod->string_count;

    bool explicit_source = false;
    uint32_t key_idx = NVM_V2_NO_INDEX;
    for (uint32_t i = 0; i < mod->metadata_count; ++i)
        if (nvm_metadata_source_key(mod, mod->metadata[i].key_idx)) explicit_source = true;
    const bool synthesize_source = has_source && !explicit_source;
    if (synthesize_source)
        for (uint32_t i = 0; i < mod->string_count; ++i)
            if (nvm_metadata_source_key(mod, i)) { key_idx = i; break; }
    bool append_key = synthesize_source && key_idx == NVM_V2_NO_INDEX;
    if (append_key && mod->string_count == UINT32_MAX) return NVM_V2_ERR_INDEX_RANGE;
    uint32_t n_ck = mod->string_count + (append_key ? 1u : 0u);
    NvmV2Constant *ck = n_ck ? calloc(n_ck, sizeof *ck) : NULL;
    if (n_ck && !ck) return NVM_V2_ERR_TRUNCATED;
    for (uint32_t i = 0; i < mod->string_count; i++) {
        ck[i].tag = TAG_STRING;
        ck[i].length = mod->string_lengths ? mod->string_lengths[i] : 0;
        ck[i].payload = (const uint8_t *)mod->strings[i];
    }
    if (append_key) {
        key_idx = mod->string_count;
        ck[key_idx].tag = TAG_STRING;
        ck[key_idx].length = (uint32_t)(sizeof SOURCE_FILE_KEY - 1);
        ck[key_idx].payload = (const uint8_t *)SOURCE_FILE_KEY;
    }
    out->constants.items = ck;
    out->constants.count = n_ck;

    /* SIGNATURES: at most one per function and one per import before dedup.
     * The tag pool is sized for that worst case in one block, so the arrays
     * the signatures point at have a single owner. */
    size_t pool_cap = 0;
    for (uint32_t i = 0; i < n_fn; i++)
        pool_cap += (size_t)mod->functions[i].arity + mod->functions[i].result_count;
    for (uint32_t i = 0; i < n_im; i++)
        pool_cap += (size_t)mod->imports[i].param_count + 1u;
    for (uint32_t i = 0; i < n_cb; i++)
        pool_cap += (size_t)mod->callback_contracts[i].param_count + 1u;

    uint8_t *pool = pool_cap ? calloc(pool_cap, 1) : NULL;
    if (pool_cap && !pool) goto oom;
    out->owned_tags = pool;
    size_t pool_used = 0;

    if ((uint64_t)n_fn + n_im + n_cb > UINT32_MAX) goto oom;
    uint32_t sig_cap = n_fn + n_im + n_cb;
    NvmV2Signature *sigs = sig_cap ? calloc(sig_cap, sizeof *sigs) : NULL;
    if (sig_cap && !sigs) goto oom;
    out->signatures.items = sigs;
    out->signatures.count = 0;

    NvmV2Function *fns = n_fn ? calloc(n_fn, sizeof *fns) : NULL;
    if (n_fn && !fns) goto oom;
    out->functions.items = fns;
    out->functions.count = n_fn;

    for (uint32_t i = 0; i < n_fn; i++) {
        const NvmFunctionEntry *f = &mod->functions[i];
        size_t mark = pool_used;

        /* I retain producer-declared tags. Interning can rewind this pool,
         * so absent declarations must overwrite reused bytes with TAG_VOID. */
        const uint8_t *ptags = f->arity ? pool + pool_used : NULL;
        if (f->arity && mod->function_param_types && mod->function_param_types[i])
            memcpy(pool + pool_used, mod->function_param_types[i], f->arity);
        else if (f->arity)
            memset(pool + pool_used, TAG_VOID, f->arity);
        pool_used += f->arity;

        const uint8_t *rtags = NULL;
        if (f->result_count) {
            rtags = pool + pool_used;
            memset(pool + pool_used, f->result_tag, f->result_count);
            pool_used += f->result_count;
        }

        NvmV2Signature sig = { f->arity, f->result_count, ptags, rtags };
        fns[i].signature_idx = intern_signature(&out->signatures, &sig,
                                                &pool_used, mark);
        fns[i].name_idx      = f->name_idx;
        fns[i].code_offset   = f->code_offset;
        fns[i].code_length   = f->code_length;
        fns[i].local_count   = f->local_count;
        fns[i].upvalue_count = f->upvalue_count;
        /* The producer computes the depth and the loader confirms it, rather
         * than the loader recomputing it: that is cheaper at load, and a
         * mismatch then means the producer and the verifier disagree, which is
         * worth failing on. A function the verifier rejects has no honest
         * depth, so it keeps 0 -- "not declared" -- and the confirming side
         * treats 0 as nothing to check. The module still has to pass
         * nvm_verify before it runs either way. */
        uint16_t depth = 0;
        if (nvm_verify_function_max_stack(mod, i, &depth).ok)
            fns[i].max_stack = depth;
    }

    NvmV2Import *ims = n_im ? calloc(n_im, sizeof *ims) : NULL;
    if (n_im && !ims) goto oom;
    out->imports.items = ims;
    out->imports.count = n_im;

    for (uint32_t i = 0; i < n_im; i++) {
        const NvmImportEntry *im = &mod->imports[i];
        size_t mark = pool_used;

        const uint8_t *ptags = NULL;
        if (im->param_count) {
            ptags = pool + pool_used;
            if (mod->import_param_types && mod->import_param_types[i])
                memcpy(pool + pool_used, mod->import_param_types[i], im->param_count);
            else
                memset(pool + pool_used, TAG_VOID, im->param_count);
            pool_used += im->param_count;
        }

        /* A v1 import returns zero or one value; TAG_VOID means zero. */
        uint16_t rcount = (im->return_type == TAG_VOID) ? 0 : 1;
        const uint8_t *rtags = NULL;
        if (rcount) {
            rtags = pool + pool_used;
            pool[pool_used++] = im->return_type;
        }

        NvmV2Signature sig = { im->param_count, rcount, ptags, rtags };
        ims[i].signature_idx   = intern_signature(&out->signatures, &sig,
                                                  &pool_used, mark);
        ims[i].module_name_idx = im->module_name_idx;
        ims[i].symbol_name_idx = im->function_name_idx;
        ims[i].kind            = mod->imports[i].kind;
    }

    out->callbacks.items = n_cb ? calloc(n_cb, sizeof(*out->callbacks.items)) : NULL;
    if (n_cb && !out->callbacks.items) goto oom;
    out->callbacks.count = n_cb;
    for (uint32_t i = 0; i < n_cb; i++) {
        const NvmCallbackContract *c = &mod->callback_contracts[i];
        NvmV2Callback *wire = &out->callbacks.items[i];
        wire->import_idx = c->import_idx;
        wire->adapter_name_idx = c->adapter_name_idx;
        wire->parameter_idx = c->parameter_idx;
        wire->abi_version = c->abi_version;
        wire->execution = c->execution;
        wire->signature_idx = NVM_V2_NO_INDEX;
        if (c->parameter_idx != NVM_CALLBACK_NO_PARAMETER) {
            size_t mark = pool_used;
            const uint8_t *params = c->param_count ? pool + pool_used : NULL;
            if (c->param_count) memcpy(pool + pool_used, c->param_tags, c->param_count);
            pool_used += c->param_count;
            uint16_t result_count = c->return_tag != TAG_VOID;
            const uint8_t *results = result_count ? pool + pool_used : NULL;
            if (result_count) pool[pool_used++] = c->return_tag;
            NvmV2Signature signature = {c->param_count, result_count, params, results};
            wire->signature_idx = intern_signature(&out->signatures, &signature, &pool_used, mark);
        }
    }

    /* LINKS: a v1 module ref names a dependency, not a symbol or a call shape,
     * so both of those stay absent rather than being invented. */
    NvmV2Link *lks = n_lk ? calloc(n_lk, sizeof *lks) : NULL;
    if (n_lk && !lks) goto oom;
    for (uint32_t i = 0; i < n_lk; i++) {
        lks[i].module_name_idx = mod->module_refs[i].module_name_idx;
        lks[i].symbol_name_idx = NVM_V2_NO_INDEX;
        lks[i].signature_idx   = NVM_V2_NO_INDEX;
        lks[i].flags           = 0;
    }
    out->links.items = lks;
    out->links.count = n_lk;

    NvmV2DebugEntry *dbs = n_db ? calloc(n_db, sizeof *dbs) : NULL;
    if (n_db && !dbs) goto oom;
    for (uint32_t i = 0; i < n_db; i++) {
        dbs[i].bytecode_offset = mod->debug_entries[i].bytecode_offset;
        dbs[i].source_line     = mod->debug_entries[i].source_line;
        dbs[i].source_col      = mod->debug_entries[i].source_col;
    }
    out->debug.items = dbs;
    out->debug.count = n_db;
    out->has_debug   = n_db > 0 || (mod->header.flags & NVM_FLAG_DEBUG_INFO) != 0;

    if (synthesize_source && mod->metadata_count == UINT32_MAX) goto oom;
    uint32_t metadata_count = mod->metadata_count + (synthesize_source ? 1u : 0u);
    if (metadata_count) {
        NvmV2MetadataEntry *md = calloc(metadata_count, sizeof *md);
        if (!md) goto oom;
        for (uint32_t i = 0; i < mod->metadata_count; ++i)
            md[i] = (NvmV2MetadataEntry){mod->metadata[i].key_idx, mod->metadata[i].value_idx};
        if (synthesize_source)
            md[mod->metadata_count] = (NvmV2MetadataEntry){key_idx, mod->source_file_idx};
        out->metadata.items = md;
        out->metadata.count = metadata_count;
    }

    /* v1 records only how many structs, enums and unions a module defines --
     * the verifier bounds AGG_* operands against those counts. v2 has no
     * count field because it has the layouts themselves, so the counts travel
     * as that many field-less layouts of each kind. They carry no shape
     * because v1 has none to give; a v2-native producer emits real ones. */
    uint32_t n_lay = mod->struct_count + mod->enum_count + mod->union_count;
    if (mod->layout_size) {
        NvmV2Result retained = nvm_v2_layouts_decode(mod->layout_data,
                                                    mod->layout_size, &out->layouts);
        if (retained != NVM_V2_OK) { nvm_v2_module_free(out); return retained; }
        out->extra_features |= NVM_V2_FEATURE_RETAINED_LAYOUTS;
    } else if (n_lay) {
        NvmV2Layout *lay = calloc(n_lay, sizeof *lay);
        if (!lay) goto oom;
        uint32_t k = 0;
        for (uint32_t i = 0; i < mod->struct_count; i++, k++) {
            lay[k].kind = NVM_V2_LAYOUT_STRUCT; lay[k].name_idx = NVM_V2_NO_INDEX;
        }
        for (uint32_t i = 0; i < mod->enum_count; i++, k++) {
            lay[k].kind = NVM_V2_LAYOUT_ENUM; lay[k].name_idx = NVM_V2_NO_INDEX;
        }
        for (uint32_t i = 0; i < mod->union_count; i++, k++) {
            lay[k].kind = NVM_V2_LAYOUT_UNION; lay[k].name_idx = NVM_V2_NO_INDEX;
        }
        out->layouts.items = lay;
        out->layouts.count = n_lay;
    }

    /* NEEDS_EXTERN is not always derivable: the assembler lets a module
     * declare it with no import table at all (`.flag needs_extern`). The
     * feature bit carries it either way. */
    if (mod->header.flags & NVM_FLAG_NEEDS_EXTERN)
        out->extra_features |= NVM_V2_FEATURE_FFI;

    out->code      = mod->code;
    out->code_size = mod->code_size;

    /* v1 leaves entry_point at 0 when there is no main and marks the absence
     * with a flag, so the flag is the source of truth. Reading the field alone
     * would make every main-less module claim function 0 as its entry. */
    out->entry_point = ((mod->header.flags & NVM_FLAG_HAS_MAIN) &&
                        mod->header.entry_point < n_fn)
                         ? mod->header.entry_point
                         : NVM_V2_NO_ENTRY_POINT;

    return NVM_V2_OK;

oom:
    nvm_v2_module_free(out);
    return NVM_V2_ERR_TRUNCATED;
}

/* ── v2 -> v1 ───────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_to_nvm_module(const NvmV2Module *m, NvmModule **out) {
    if (!m || !out) return NVM_V2_ERR_INDEX_RANGE;
    *out = NULL;

    NvmModule *mod = nvm_module_new();
    if (!mod) return NVM_V2_ERR_TRUNCATED;

    /* The constant pool must map one-to-one onto the v1 string pool, in order,
     * or every recorded index shifts. A non-string constant has no v1
     * representation at all, so it is refused rather than dropped. */
    for (uint32_t i = 0; i < m->constants.count; i++) {
        const NvmV2Constant *c = &m->constants.items[i];
        if (c->tag != TAG_STRING) {
            nvm_module_free(mod);
            return NVM_V2_ERR_SECTION_TYPE;
        }
        uint32_t idx = nvm_add_string(mod, (const char *)c->payload, c->length);
        if (idx != i) {   /* a duplicate would have collapsed and shifted the rest */
            nvm_module_free(mod);
            return NVM_V2_ERR_INDEX_RANGE;
        }
    }

    for (uint32_t i = 0; i < m->metadata.count; i++) {
        const NvmV2MetadataEntry *e = &m->metadata.items[i];
        if (!nvm_add_metadata(mod, e->key_idx, e->value_idx)) {
            nvm_module_free(mod);
            return NVM_V2_ERR_INDEX_RANGE;
        }
    }

    if (m->code_size) {
        if (m->code_size > UINT32_MAX) { nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE; }
        nvm_append_code(mod, m->code, (uint32_t)m->code_size);
    }

    for (uint32_t i = 0; i < m->functions.count; i++) {
        const NvmV2Function *f = &m->functions.items[i];
        const NvmV2Signature *s = &m->signatures.items[f->signature_idx];
        /* v1 stores offsets and lengths as u32. A module that outgrew that is
         * simply not expressible as v1, which is one of the reasons v2 widened
         * them; say so rather than truncating. */
        if (f->code_offset > UINT32_MAX || f->code_length > UINT32_MAX ||
            s->result_count > UINT8_MAX) {
            nvm_module_free(mod);
            return NVM_V2_ERR_INDEX_RANGE;
        }
        NvmFunctionEntry e;
        memset(&e, 0, sizeof e);
        e.name_idx      = f->name_idx;
        e.arity         = s->param_count;
        e.code_offset   = (uint32_t)f->code_offset;
        e.code_length   = (uint32_t)f->code_length;
        e.local_count   = f->local_count;
        e.upvalue_count = f->upvalue_count;
        e.result_count  = (uint8_t)s->result_count;
        e.result_tag    = s->result_count ? s->result_tags[0] : TAG_VOID;
        uint32_t index = nvm_add_function(mod, &e);
        if (index == UINT32_MAX ||
            !nvm_set_function_param_types(mod, index, s->param_tags, s->param_count)) {
            nvm_module_free(mod);
            return NVM_V2_ERR_TRUNCATED;
        }
    }

    for (uint32_t i = 0; i < m->imports.count; i++) {
        const NvmV2Import *im = &m->imports.items[i];
        const NvmV2Signature *s = &m->signatures.items[im->signature_idx];
        uint8_t ret = s->result_count ? s->result_tags[0] : TAG_VOID;
        uint32_t index = nvm_add_import(mod, im->module_name_idx, im->symbol_name_idx,
                                       s->param_count, ret, s->param_tags);
        if (index == UINT32_MAX) {
            nvm_module_free(mod);
            return NVM_V2_ERR_TRUNCATED;
        }
        mod->imports[index].kind = im->kind;
    }

    for (uint32_t i = 0; i < m->callbacks.count; i++) {
        const NvmV2Callback *wire = &m->callbacks.items[i];
        NvmCallbackContract c = {0};
        c.import_idx = wire->import_idx;
        c.adapter_name_idx = wire->adapter_name_idx;
        c.parameter_idx = wire->parameter_idx;
        c.abi_version = wire->abi_version;
        c.execution = wire->execution;
        if (c.parameter_idx != NVM_CALLBACK_NO_PARAMETER) {
            if (wire->signature_idx >= m->signatures.count) { nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE; }
            const NvmV2Signature *s = &m->signatures.items[wire->signature_idx];
            if (s->param_count > NANO_MAX_FFI_ARGS || s->result_count > 1) {
                nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE;
            }
            c.param_count = s->param_count;
            c.return_tag = s->result_count ? s->result_tags[0] : TAG_VOID;
            if (s->param_count) memcpy(c.param_tags, s->param_tags, s->param_count);
        } else if (wire->signature_idx != NVM_V2_NO_INDEX) {
            nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE;
        }
        if (!nvm_add_callback_contract(mod, &c)) {
            nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE;
        }
    }
    if (!nvm_callback_contracts_valid(mod)) { nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE; }

    for (uint32_t i = 0; i < m->links.count; i++)
        nvm_add_module_ref(mod, m->links.items[i].module_name_idx);

    for (uint32_t i = 0; i < m->debug.count; i++) {
        const NvmV2DebugEntry *d = &m->debug.items[i];
        if (d->bytecode_offset > UINT32_MAX) {
            nvm_module_free(mod);
            return NVM_V2_ERR_INDEX_RANGE;
        }
        if (!nvm_add_debug_entry(mod, (uint32_t)d->bytecode_offset,
                                 d->source_line, d->source_col)) {
            nvm_module_free(mod);
            return NVM_V2_ERR_INDEX_RANGE;
        }
    }

    for (uint32_t i = 0; i < m->layouts.count; i++) {
        switch (m->layouts.items[i].kind) {
        case NVM_V2_LAYOUT_STRUCT: mod->struct_count++; break;
        case NVM_V2_LAYOUT_ENUM:   mod->enum_count++;   break;
        case NVM_V2_LAYOUT_UNION:  mod->union_count++;  break;
        default: break;   /* a tuple layout has no v1 counterpart */
        }
    }

    if ((m->extra_features & NVM_V2_FEATURE_RETAINED_LAYOUTS) ||
        nvm_layouts_have_facts(&m->layouts)) {
        NvmV2Result retained = nvm_retain_layouts(mod, &m->layouts);
        if (retained != NVM_V2_OK) { nvm_module_free(mod); return retained; }
    }

    /* Every v1 header flag restates something v2 encodes structurally, so all
     * three are derived rather than carried. Deriving them is also what keeps
     * them from disagreeing with the module they describe. */
    mod->header.flags = 0;
    if (m->entry_point != NVM_V2_NO_ENTRY_POINT) {
        mod->header.flags |= NVM_FLAG_HAS_MAIN;
        mod->header.entry_point = m->entry_point;
    } else {
        mod->header.entry_point = 0;
    }
    if (m->imports.count || (m->extra_features & NVM_V2_FEATURE_FFI))
        mod->header.flags |= NVM_FLAG_NEEDS_EXTERN;
    if (m->has_debug)     mod->header.flags |= NVM_FLAG_DEBUG_INFO;

    if (m->passive_size) {
        if (!m->passive_data) { nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE; }
        mod->passive_data = malloc(m->passive_size);
        if (!mod->passive_data) { nvm_module_free(mod); return NVM_V2_ERR_TRUNCATED; }
        memcpy(mod->passive_data, m->passive_data, m->passive_size);
        mod->passive_size = m->passive_size;
    }
    if (m->ownership_size) {
        if (!m->ownership_data) { nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE; }
        mod->ownership_data = malloc(m->ownership_size);
        if (!mod->ownership_data) { nvm_module_free(mod); return NVM_V2_ERR_TRUNCATED; }
        memcpy(mod->ownership_data, m->ownership_data, m->ownership_size);
        mod->ownership_size = m->ownership_size;
    }
    bool needs_ownership = false;
    NvmV2Result ownership = nvm_ownership_contracts_validate(mod, &needs_ownership);
    if (ownership != NVM_V2_OK) { nvm_module_free(mod); return ownership; }
    if (!nvm_passive_valid(mod)) { nvm_module_free(mod); return NVM_V2_ERR_INDEX_RANGE; }
    *out = mod;
    return NVM_V2_OK;
}
