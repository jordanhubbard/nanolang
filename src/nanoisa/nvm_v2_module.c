/*
 * Whole-module v2 serialization, deserialization, and cross-section validation.
 *
 * The section codecs each validate what they can see: their own counts, their
 * own reserved bytes, their own tags. None of them can check that an index in
 * one section names something that exists in another, because none of them can
 * see another. That check belongs here, and it is the substance of this file.
 *
 * Layout: header, then the directory in ascending section-type order, then the
 * payloads in the same order, then the checksum patched over the header.
 * Ascending order makes the directory canonical -- two producers emitting the
 * same module emit the same bytes -- and lets validation be a single forward
 * pass.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "retained_layouts.h"
#include "service_bindings_module.h"
#include "isa.h"
#include "nvm_format.h"   /* nvm_crc32 */

/* One row per section we may emit, in ascending type order. */
typedef struct {
    uint32_t type;
    size_t   size;
    size_t   offset;   /* filled during layout */
    bool     present;
} SectionPlan;

#define PLAN_SLOTS 15

static size_t align4(size_t n) { return (n + 3u) & ~(size_t)3u; }

/* Feature bits a module's contents REQUIRE. The header must set at least
 * these: a module carrying a link table but not advertising LINKED would make
 * a reader that trusted the bits skip work it has to do.
 *
 * The rule is a floor rather than an equality because a capability can be
 * needed without leaving a trace in a table. A module can declare that it
 * needs the FFI without listing an import -- v1's assembler spells that
 * `.flag needs_extern` -- and refusing that would make a legal module
 * unrepresentable. What is not allowed is carrying the table and denying the
 * capability. */
static uint32_t required_features(const NvmV2Module *m) {
    uint32_t f = 0;
    if (m->links.count) f |= NVM_V2_FEATURE_LINKED;
    if (m->imports.count) f |= NVM_V2_FEATURE_FFI;
    if (m->callbacks.count) f |= NVM_V2_FEATURE_CALLBACKS;
    if (m->passive_size) f |= NVM_V2_FEATURE_PASSIVE;
    if (m->ownership_size) f |= NVM_V2_FEATURE_OWNERSHIP;
    if (m->capture_size) f |= NVM_V2_FEATURE_CAPTURE_BINDINGS;
    if (m->service_size) f |= NVM_V2_FEATURE_SERVICE_BINDINGS;
    if (nvm_layouts_have_facts(&m->layouts)) f |= NVM_V2_FEATURE_RETAINED_LAYOUTS;
    if (m->has_debug) f |= NVM_V2_FEATURE_DEBUG;
    for (uint32_t i = 0; i < m->imports.count; i++)
        if (m->imports.items[i].kind == NVM_V2_IMPORT_COPROCESS)
            f |= NVM_V2_FEATURE_COPROCESS;
    return f;
}

static size_t build_plan(const NvmV2Module *m, SectionPlan *plan) {
    size_t n = 0;
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_METADATA,
                               nvm_v2_metadata_encoded_size(&m->metadata), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_CONSTANTS,
                               nvm_v2_constants_encoded_size(&m->constants), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_SIGNATURES,
                               nvm_v2_signatures_encoded_size(&m->signatures), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_LAYOUTS,
                               nvm_v2_layouts_encoded_size(&m->layouts), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_FUNCTIONS,
                               nvm_v2_functions_encoded_size(&m->functions), 0, true };
    /* CODE is opaque bytes. A module with no code still gets the section, so
     * the required-section set does not vary with content. */
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_CODE, (size_t)m->code_size, 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_GLOBALS,
                               nvm_v2_globals_encoded_size(&m->globals), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_IMPORTS,
                               nvm_v2_imports_encoded_size(&m->imports), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_LINKS,
                               nvm_v2_links_encoded_size(&m->links), 0, true };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_DEBUG,
                               nvm_v2_debug_encoded_size(&m->debug), 0,
                               m->has_debug };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_CALLBACKS,
                               nvm_v2_callbacks_encoded_size(&m->callbacks), 0,
                               m->callbacks.count != 0 };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_PASSIVE, m->passive_size, 0, m->passive_size != 0 };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_OWNERSHIP, m->ownership_size, 0, m->ownership_size != 0 };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_SERVICE_BINDINGS, m->service_size, 0, m->service_size != 0 };
    plan[n++] = (SectionPlan){ NVM_V2_SECTION_CAPTURE_BINDINGS, m->capture_size, 0, m->capture_size != 0 };
    return n;
}

static NvmV2Result validate_cross_section(const NvmV2Module *m, uint32_t declared_features);

NvmV2Result nvm_v2_module_serialize(const NvmV2Module *m,
                                    uint8_t *out, size_t capacity,
                                    size_t *out_size) {
    if (!m) return NVM_V2_ERR_INDEX_RANGE;
    NvmV2Result service = nvm_v2_service_bindings_validate(m);
    if (service != NVM_V2_OK) return service;
    if ((m->passive_size && !m->passive_data) ||
        (!m->passive_size && (m->extra_features & NVM_V2_FEATURE_PASSIVE)))
        return NVM_V2_ERR_FEATURE_MISMATCH;
    if ((m->ownership_size && !m->ownership_data) ||
        (!m->ownership_size && (m->extra_features & NVM_V2_FEATURE_OWNERSHIP)))
        return NVM_V2_ERR_FEATURE_MISMATCH;
    if ((!m->capture_data != !m->capture_size) ||
        (!m->capture_size && (m->extra_features & NVM_V2_FEATURE_CAPTURE_BINDINGS)))
        return NVM_V2_ERR_FEATURE_MISMATCH;
    if (m->capture_size || m->passive_size || m->ownership_size || nvm_v2_service_bindings_present(m)) {
        NvmV2Result result = validate_cross_section(m, required_features(m) | m->extra_features);
        if (result != NVM_V2_OK) return result;
        NvmModule *checked = NULL;
        result = nvm_v2_to_nvm_module(m, &checked);
        nvm_module_free(checked);
        if (result != NVM_V2_OK) return result;
    }
    SectionPlan plan[PLAN_SLOTS];
    size_t slots = build_plan(m, plan);

    uint32_t section_count = 0;
    for (size_t i = 0; i < slots; i++) if (plan[i].present) section_count++;

    size_t dir_bytes = (size_t)section_count * NVM_V2_SECTION_ENTRY_SIZE;
    size_t cursor = NVM_V2_HEADER_SIZE + dir_bytes;

    /* Payloads are 4-byte aligned so no section starts mid-word; the container
     * does not require it, but it keeps offsets readable in a hex dump and
     * costs at most three bytes per section. */
    for (size_t i = 0; i < slots; i++) {
        if (!plan[i].present) continue;
        cursor = align4(cursor);
        plan[i].offset = cursor;
        cursor += plan[i].size;
    }
    size_t total = cursor;

    if (out_size) *out_size = total;
    if (!out) return NVM_V2_OK;              /* sizing pass */
    if (capacity < total) return NVM_V2_ERR_TRUNCATED;

    memset(out, 0, total);

    NvmV2Header h;
    memset(&h, 0, sizeof h);
    h.magic[0] = NVM_V2_MAGIC_0; h.magic[1] = NVM_V2_MAGIC_1;
    h.magic[2] = NVM_V2_MAGIC_2; h.magic[3] = NVM_V2_MAGIC_3;
    h.format_version = NVM_V2_FORMAT_VERSION;
    h.isa_version    = m->isa_version ? m->isa_version : NVM_V2_ISA_VERSION;
    h.feature_bits   = required_features(m) | m->extra_features;
    h.total_size     = total;
    h.header_size    = NVM_V2_HEADER_SIZE;
    h.section_count  = section_count;
    h.entry_point    = m->entry_point;
    h.flags          = 0;
    h.checksum       = 0;
    nvm_v2_write_header(out, &h);

    size_t dir = NVM_V2_HEADER_SIZE;
    for (size_t i = 0; i < slots; i++) {
        if (!plan[i].present) continue;
        NvmV2SectionEntry e = { plan[i].type, 0, plan[i].offset, plan[i].size };
        nvm_v2_write_section(out + dir, &e);
        dir += NVM_V2_SECTION_ENTRY_SIZE;
    }

    for (size_t i = 0; i < slots; i++) {
        if (!plan[i].present || plan[i].size == 0) continue;
        uint8_t *p = out + plan[i].offset;
        size_t   z = plan[i].size;
        switch (plan[i].type) {
        case NVM_V2_SECTION_METADATA:   nvm_v2_metadata_encode(&m->metadata, p, z); break;
        case NVM_V2_SECTION_CONSTANTS:  nvm_v2_constants_encode(&m->constants, p, z); break;
        case NVM_V2_SECTION_SIGNATURES: nvm_v2_signatures_encode(&m->signatures, p, z); break;
        case NVM_V2_SECTION_LAYOUTS:    nvm_v2_layouts_encode(&m->layouts, p, z); break;
        case NVM_V2_SECTION_FUNCTIONS:  nvm_v2_functions_encode(&m->functions, p, z); break;
        case NVM_V2_SECTION_CODE:       if (m->code) memcpy(p, m->code, z); break;
        case NVM_V2_SECTION_GLOBALS:    nvm_v2_globals_encode(&m->globals, p, z); break;
        case NVM_V2_SECTION_IMPORTS:    nvm_v2_imports_encode(&m->imports, p, z); break;
        case NVM_V2_SECTION_LINKS:      nvm_v2_links_encode(&m->links, p, z); break;
        case NVM_V2_SECTION_DEBUG:      nvm_v2_debug_encode(&m->debug, p, z); break;
        case NVM_V2_SECTION_CALLBACKS:  nvm_v2_callbacks_encode(&m->callbacks, p, z); break;
        case NVM_V2_SECTION_PASSIVE: memcpy(p, m->passive_data, z); break;
        case NVM_V2_SECTION_OWNERSHIP: memcpy(p, m->ownership_data, z); break;
        case NVM_V2_SECTION_CAPTURE_BINDINGS: memcpy(p, m->capture_data, z); break;
        case NVM_V2_SECTION_SERVICE_BINDINGS: memcpy(p, m->service_data, z); break;
        default: break;
        }
    }

    /* Checksum last: it covers everything after the header, so it can only be
     * computed once the payloads are in place. */
    h.checksum = nvm_crc32(out + NVM_V2_HEADER_SIZE,
                           (uint32_t)(total - NVM_V2_HEADER_SIZE));
    nvm_v2_write_header(out, &h);
    return NVM_V2_OK;
}

/* ── Cross-section validation ───────────────────────────────────────────── */

static bool index_ok(uint32_t idx, uint32_t count, bool sentinel_allowed) {
    if (sentinel_allowed && idx == NVM_V2_NO_INDEX) return true;
    return idx < count;
}

static NvmV2Result validate_callbacks(const NvmV2Module *m) {
    uint32_t begin = 0;
    while (begin < m->callbacks.count) {
        const NvmV2Callback *first = &m->callbacks.items[begin];
        if (first->import_idx >= m->imports.count) return NVM_V2_ERR_INDEX_RANGE;
        const NvmV2Signature *import = &m->signatures.items[m->imports.items[first->import_idx].signature_idx];
        if (import->param_count > NANO_MAX_FFI_ARGS) return NVM_V2_ERR_INDEX_RANGE;
        uint32_t expected = 0, actual = 0;
        for (uint16_t p = 0; p < import->param_count; p++)
            if (import->param_tags[p] == TAG_FUNCTION || import->param_tags[p] == TAG_CLOSURE)
                expected |= 1u << p;
        uint32_t end = begin;
        while (end < m->callbacks.count && m->callbacks.items[end].import_idx == first->import_idx) {
            const NvmV2Callback *c = &m->callbacks.items[end];
            if (c->adapter_name_idx >= m->constants.count) return NVM_V2_ERR_INDEX_RANGE;
            const NvmV2Constant *name = &m->constants.items[c->adapter_name_idx];
            if (name->tag != TAG_STRING || !name->length || !name->payload ||
                memchr(name->payload, 0, name->length) ||
                c->execution != first->execution || c->adapter_name_idx != first->adapter_name_idx)
                return NVM_V2_ERR_SECTION_TYPE;
            if (c->parameter_idx == NVM_CALLBACK_NO_PARAMETER) {
                if (expected || end != begin || c->signature_idx != NVM_V2_NO_INDEX)
                    return NVM_V2_ERR_INDEX_RANGE;
            } else {
                if (c->parameter_idx >= import->param_count ||
                    !(expected & (1u << c->parameter_idx)) || c->signature_idx >= m->signatures.count)
                    return NVM_V2_ERR_INDEX_RANGE;
                const NvmV2Signature *signature = &m->signatures.items[c->signature_idx];
                if (signature->result_count > 1 ||
                    (signature->result_count && signature->result_tags[0] == TAG_VOID) ||
                    !nvm_callback_shape_valid(signature->param_tags, signature->param_count,
                        signature->result_count ? signature->result_tags[0] : TAG_VOID))
                    return NVM_V2_ERR_SECTION_TYPE;
                actual |= 1u << c->parameter_idx;
            }
            end++;
        }
        if (expected != actual) return NVM_V2_ERR_INDEX_RANGE;
        begin = end;
    }
    return NVM_V2_OK;
}

static NvmV2Result validate_cross_section(const NvmV2Module *m,
                                          uint32_t declared_features) {
    NvmV2Result service = nvm_v2_service_bindings_validate(m);
    if (service != NVM_V2_OK) return service;
    if (((declared_features & NVM_V2_FEATURE_SERVICE_BINDINGS) != 0) !=
        (m->service_size != 0)) return NVM_V2_ERR_FEATURE_MISMATCH;
    const uint32_t nc = m->constants.count;
    const uint32_t ns = m->signatures.count;
    const uint32_t nl = m->layouts.count;

    for (uint32_t i = 0; i < m->functions.count; i++) {
        const NvmV2Function *f = &m->functions.items[i];
        if (!index_ok(f->name_idx, nc, false)) return NVM_V2_ERR_INDEX_RANGE;
        if (!index_ok(f->signature_idx, ns, false)) return NVM_V2_ERR_INDEX_RANGE;
        /* Subtraction form: bound the offset first so the length comparison
         * cannot overflow. An addition here is exactly the wrap v1 shipped. */
        if (f->code_offset > m->code_size) return NVM_V2_ERR_INDEX_RANGE;
        if (f->code_length > m->code_size - f->code_offset)
            return NVM_V2_ERR_INDEX_RANGE;
    }

    for (uint32_t i = 0; i < m->globals.count; i++) {
        const NvmV2Global *g = &m->globals.items[i];
        if (!index_ok(g->name_idx, nc, false)) return NVM_V2_ERR_INDEX_RANGE;
        if (!index_ok(g->init_idx, nc, true)) return NVM_V2_ERR_INDEX_RANGE;
    }

    for (uint32_t i = 0; i < m->imports.count; i++) {
        const NvmV2Import *im = &m->imports.items[i];
        if (im->kind > NVM_V2_IMPORT_KIND_MAX) return NVM_V2_ERR_SECTION_TYPE;
        if (!index_ok(im->module_name_idx, nc, false)) return NVM_V2_ERR_INDEX_RANGE;
        if (!index_ok(im->symbol_name_idx, nc, false)) return NVM_V2_ERR_INDEX_RANGE;
        if (!index_ok(im->signature_idx, ns, false)) return NVM_V2_ERR_INDEX_RANGE;
        if (im->kind == NVM_V2_IMPORT_ARTIFACT ||
            im->kind == NVM_V2_IMPORT_DECLARED_SCALAR_ARTIFACT) {
            const NvmV2Constant *path = &m->constants.items[im->module_name_idx];
            if (path->tag != TAG_STRING || !path->length || !path->payload ||
                path->payload[0] != '/' || memchr(path->payload, 0, path->length))
                return NVM_V2_ERR_SECTION_TYPE;
        }
        if (im->kind == NVM_V2_IMPORT_DECLARED_SCALAR_ARTIFACT) {
            const NvmV2Constant *symbol = &m->constants.items[im->symbol_name_idx];
            const NvmV2Signature *sig = &m->signatures.items[im->signature_idx];
            if (symbol->tag != TAG_STRING || !symbol->length || !symbol->payload ||
                memchr(symbol->payload, 0, symbol->length) || sig->result_count > 1 ||
                (sig->result_count && !sig->result_tags) ||
                !nvm_declared_scalar_shape_valid(sig->param_tags, sig->param_count,
                    sig->result_count ? sig->result_tags[0] : TAG_VOID))
                return NVM_V2_ERR_SECTION_TYPE;
        }
    }

    NvmV2Result callbacks_valid = validate_callbacks(m);
    if (callbacks_valid != NVM_V2_OK) return callbacks_valid;

    for (uint32_t i = 0; i < m->links.count; i++) {
        const NvmV2Link *lk = &m->links.items[i];
        if (!index_ok(lk->module_name_idx, nc, false)) return NVM_V2_ERR_INDEX_RANGE;
        /* A link may name a whole module rather than one symbol in it -- a
         * plain dependency edge has no symbol and no call shape -- so both of
         * these accept the sentinel. An out-of-range value is still rejected;
         * only "absent" is legal. */
        if (!index_ok(lk->symbol_name_idx, nc, true)) return NVM_V2_ERR_INDEX_RANGE;
        if (!index_ok(lk->signature_idx, ns, true)) return NVM_V2_ERR_INDEX_RANGE;
    }

    for (uint32_t i = 0; i < m->layouts.count; i++) {
        const NvmV2Layout *l = &m->layouts.items[i];
        if (!index_ok(l->name_idx, nc, true)) return NVM_V2_ERR_INDEX_RANGE;
        for (uint16_t f = 0; f < l->field_count; f++) {
            if (!index_ok(l->fields[f].name_idx, nc, true))
                return NVM_V2_ERR_INDEX_RANGE;
            /* I retain the codec's acyclic exact indices without reordering. */
            if (!index_ok(l->fields[f].nested_idx, nl, true))
                return NVM_V2_ERR_INDEX_RANGE;
        }
    }

    for (uint32_t i = 0; i < m->metadata.count; i++) {
        if (!index_ok(m->metadata.items[i].key_idx, nc, false))
            return NVM_V2_ERR_INDEX_RANGE;
        if (!index_ok(m->metadata.items[i].value_idx, nc, false))
            return NVM_V2_ERR_INDEX_RANGE;
    }

    if (!index_ok(m->entry_point, m->functions.count, false) &&
        m->entry_point != NVM_V2_NO_ENTRY_POINT)
        return NVM_V2_ERR_INDEX_RANGE;

    uint32_t required = required_features(m);
    if (((declared_features & NVM_V2_FEATURE_CALLBACKS) != 0) != (m->callbacks.count != 0))
        return NVM_V2_ERR_FEATURE_MISMATCH;
    if ((declared_features & required) != required)
        return NVM_V2_ERR_FEATURE_MISMATCH;

    return NVM_V2_OK;
}

NvmV2Result nvm_v2_module_deserialize(const uint8_t *data, size_t size,
                                      NvmV2Module *out) {
    memset(out, 0, sizeof *out);

    NvmV2Result r = nvm_v2_validate(data, size);
    if (r != NVM_V2_OK) return r;

    NvmV2Header h;
    r = nvm_v2_read_header(data, size, &h);
    if (r != NVM_V2_OK) return r;

    out->isa_version = h.isa_version;
    out->entry_point = h.entry_point;
    /* Kept whole: bits the tables already require are recomputed on the way
     * out, so re-serializing this module produces the same header. */
    out->extra_features = h.feature_bits;

    for (uint32_t i = 0; i < h.section_count; i++) {
        NvmV2SectionEntry e;
        r = nvm_v2_read_section(data, size, &h, i, &e);
        if (r != NVM_V2_OK) goto fail;

        const uint8_t *p = data + e.offset;
        size_t z = (size_t)e.size;
        switch (e.type) {
        case NVM_V2_SECTION_METADATA:   r = nvm_v2_metadata_decode(p, z, &out->metadata); break;
        case NVM_V2_SECTION_CONSTANTS:  r = nvm_v2_constants_decode(p, z, &out->constants); break;
        case NVM_V2_SECTION_SIGNATURES: r = nvm_v2_signatures_decode(p, z, &out->signatures); break;
        case NVM_V2_SECTION_LAYOUTS:    r = nvm_v2_layouts_decode(p, z, &out->layouts); break;
        case NVM_V2_SECTION_FUNCTIONS:  r = nvm_v2_functions_decode(p, z, &out->functions); break;
        case NVM_V2_SECTION_CODE:       out->code = z ? p : NULL; out->code_size = e.size; break;
        case NVM_V2_SECTION_GLOBALS:    r = nvm_v2_globals_decode(p, z, &out->globals); break;
        case NVM_V2_SECTION_IMPORTS:    r = nvm_v2_imports_decode(p, z, &out->imports); break;
        case NVM_V2_SECTION_LINKS:      r = nvm_v2_links_decode(p, z, &out->links); break;
        case NVM_V2_SECTION_CALLBACKS:  r = nvm_v2_callbacks_decode(p, z, &out->callbacks); break;
        case NVM_V2_SECTION_SERVICE_BINDINGS:
            if (z != NVM_SERVICE_BINDING_BYTES && z != NVM_FILE_NOMINAL_BYTES) {
                r = NVM_V2_ERR_SECTION_RANGE; break;
            }
            out->service_data = p; out->service_size = (uint32_t)z; break;
        case NVM_V2_SECTION_CAPTURE_BINDINGS:
            if (!z || z > UINT32_MAX) { r = NVM_V2_ERR_SECTION_RANGE; break; }
            out->capture_data = p; out->capture_size = (uint32_t)z; break;
        case NVM_V2_SECTION_OWNERSHIP:
            if (!z || z > UINT32_MAX) { r = NVM_V2_ERR_SECTION_RANGE; break; }
            out->ownership_data = p; out->ownership_size = (uint32_t)z; break;
        case NVM_V2_SECTION_PASSIVE:
            if (!z || z > UINT32_MAX) { r = NVM_V2_ERR_SECTION_RANGE; break; }
            out->passive_data = p; out->passive_size = (uint32_t)z; break;
        case NVM_V2_SECTION_DEBUG:
            r = nvm_v2_debug_decode(p, z, &out->debug);
            out->has_debug = true;
            break;
        default: r = NVM_V2_ERR_SECTION_TYPE; break;
        }
        if (r != NVM_V2_OK) goto fail;
    }

    r = validate_cross_section(out, h.feature_bits);
    if (r != NVM_V2_OK) goto fail;
    if (((h.feature_bits & NVM_V2_FEATURE_PASSIVE) != 0) != (out->passive_size != 0)) {
        r = NVM_V2_ERR_FEATURE_MISMATCH; goto fail;
    }
    if (((h.feature_bits & NVM_V2_FEATURE_OWNERSHIP) != 0) != (out->ownership_size != 0)) {
        r = NVM_V2_ERR_FEATURE_MISMATCH; goto fail;
    }
    if (((h.feature_bits & NVM_V2_FEATURE_CAPTURE_BINDINGS) != 0) != (out->capture_size != 0)) {
        r = NVM_V2_ERR_FEATURE_MISMATCH; goto fail;
    }
    if (out->capture_size || out->passive_size || out->ownership_size || out->service_size) {
        NvmModule *checked = NULL;
        r = nvm_v2_to_nvm_module(out, &checked);
        nvm_module_free(checked);
        if (r != NVM_V2_OK) goto fail;
    }

    return NVM_V2_OK;

fail:
    nvm_v2_module_free(out);
    return r;
}

void nvm_v2_module_free(NvmV2Module *m) {
    if (!m) return;
    nvm_v2_metadata_free(&m->metadata);
    nvm_v2_constants_free(&m->constants);
    nvm_v2_signatures_free(&m->signatures);
    nvm_v2_layouts_free(&m->layouts);
    nvm_v2_functions_free(&m->functions);
    nvm_v2_globals_free(&m->globals);
    nvm_v2_imports_free(&m->imports);
    nvm_v2_callbacks_free(&m->callbacks);
    nvm_v2_links_free(&m->links);
    nvm_v2_debug_free(&m->debug);
    free(m->owned_tags);
    m->owned_tags = NULL;
    m->capture_data = NULL;
    m->capture_size = 0;
    m->code = NULL;
    m->code_size = 0;
    m->service_data = NULL;
    m->service_size = 0;
    m->passive_data = NULL;
    m->passive_size = 0;
    m->has_debug = false;
}
