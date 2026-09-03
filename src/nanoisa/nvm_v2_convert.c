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
 *  - v1 records neither function parameter types nor max_stack. Rather than
 *    guess, the bridge emits TAG_VOID placeholder parameter tags and a
 *    max_stack of 0; a v2-native producer supplies the real values.
 *
 * Constant payloads and import tag arrays alias the source NvmModule, so it
 * must outlive the NvmV2Module the bridge produces.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "nvm_format.h"
#include "isa.h"

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

    const uint32_t n_fn = mod->function_count;
    const uint32_t n_im = mod->import_count;
    const uint32_t n_lk = mod->module_ref_count;
    const uint32_t n_db = mod->debug_count;
    const bool has_source = mod->source_file_idx != 0 &&
                            mod->source_file_idx < mod->string_count;

    /* CONSTANTS: every v1 string, plus the metadata key when it is needed. */
    uint32_t n_ck = mod->string_count + (has_source ? 1u : 0u);
    NvmV2Constant *ck = n_ck ? calloc(n_ck, sizeof *ck) : NULL;
    if (n_ck && !ck) return NVM_V2_ERR_TRUNCATED;
    for (uint32_t i = 0; i < mod->string_count; i++) {
        ck[i].tag = TAG_STRING;
        ck[i].length = mod->string_lengths ? mod->string_lengths[i] : 0;
        ck[i].payload = (const uint8_t *)mod->strings[i];
    }
    uint32_t key_idx = NVM_V2_NO_INDEX;
    if (has_source) {
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

    uint8_t *pool = pool_cap ? calloc(pool_cap, 1) : NULL;
    if (pool_cap && !pool) goto oom;
    out->owned_tags = pool;
    size_t pool_used = 0;

    uint32_t sig_cap = n_fn + n_im;
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

        /* Parameter tags are TAG_VOID placeholders: v1 does not record them,
         * and the pool is already zeroed, so this only reserves the bytes. */
        const uint8_t *ptags = f->arity ? pool + pool_used : NULL;
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
        fns[i].max_stack     = 0;   /* Task 13 fills this from the verifier */
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
        ims[i].kind            = NVM_V2_IMPORT_FFI;
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
    out->has_debug   = n_db > 0;

    if (has_source) {
        NvmV2MetadataEntry *md = calloc(1, sizeof *md);
        if (!md) goto oom;
        md[0].key_idx   = key_idx;
        md[0].value_idx = mod->source_file_idx;
        out->metadata.items = md;
        out->metadata.count = 1;
    }

    /* v1 records only how many structs, enums and unions a module defines --
     * the verifier bounds AGG_* operands against those counts. v2 has no
     * count field because it has the layouts themselves, so the counts travel
     * as that many field-less layouts of each kind. They carry no shape
     * because v1 has none to give; a v2-native producer emits real ones. */
    uint32_t n_lay = mod->struct_count + mod->enum_count + mod->union_count;
    if (n_lay) {
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
    uint32_t source_file_idx = 0;
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
        const NvmV2Constant *k = &m->constants.items[e->key_idx];
        if (k->length == sizeof SOURCE_FILE_KEY - 1 &&
            memcmp(k->payload, SOURCE_FILE_KEY, k->length) == 0)
            source_file_idx = e->value_idx;
    }
    mod->source_file_idx = source_file_idx;

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
        nvm_add_function(mod, &e);
    }

    for (uint32_t i = 0; i < m->imports.count; i++) {
        const NvmV2Import *im = &m->imports.items[i];
        const NvmV2Signature *s = &m->signatures.items[im->signature_idx];
        uint8_t ret = s->result_count ? s->result_tags[0] : TAG_VOID;
        nvm_add_import(mod, im->module_name_idx, im->symbol_name_idx,
                       s->param_count, ret, s->param_tags);
    }

    for (uint32_t i = 0; i < m->links.count; i++)
        nvm_add_module_ref(mod, m->links.items[i].module_name_idx);

    for (uint32_t i = 0; i < m->debug.count; i++) {
        const NvmV2DebugEntry *d = &m->debug.items[i];
        if (d->bytecode_offset > UINT32_MAX) {
            nvm_module_free(mod);
            return NVM_V2_ERR_INDEX_RANGE;
        }
        nvm_add_debug_entry(mod, (uint32_t)d->bytecode_offset,
                            d->source_line, d->source_col);
    }

    for (uint32_t i = 0; i < m->layouts.count; i++) {
        switch (m->layouts.items[i].kind) {
        case NVM_V2_LAYOUT_STRUCT: mod->struct_count++; break;
        case NVM_V2_LAYOUT_ENUM:   mod->enum_count++;   break;
        case NVM_V2_LAYOUT_UNION:  mod->union_count++;  break;
        default: break;   /* a tuple layout has no v1 counterpart */
        }
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
    if (m->imports.count) mod->header.flags |= NVM_FLAG_NEEDS_EXTERN;
    if (m->debug.count)   mod->header.flags |= NVM_FLAG_DEBUG_INFO;

    *out = mod;
    return NVM_V2_OK;
}
