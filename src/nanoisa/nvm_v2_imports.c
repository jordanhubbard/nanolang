/*
 * v2 IMPORTS, LINKS, METADATA and DEBUG sections.
 *
 * Four fixed-width record tables kept together because they share the same
 * decode shape: a count, a bound check against the remaining bytes, then a
 * fixed stride. Splitting them would have meant four near-identical files.
 *
 * IMPORTS and LINKS have the same layout and different meanings -- an import
 * is a foreign function, a link is a call into another NanoISA module. Neither
 * carries parameter counts or type tags: those live in SIGNATURES, which is
 * what removes v1's variable-length import tail.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"

#define IMPORT_ENTRY_BYTES   16
#define LINK_ENTRY_BYTES     16
#define METADATA_ENTRY_BYTES 8
#define DEBUG_ENTRY_BYTES    16

static void wr32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;         p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void wr64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (i * 8));
}

/* Shared prologue: read the count and bound it against what remains, so an
 * impossible count is refused on the arithmetic rather than by failing to
 * allocate for it. Returns the count via `out_count`. */
static NvmV2Result read_count(NvmV2Cursor *c, size_t size, size_t stride,
                              uint32_t *out_count) {
    NvmV2Result r = nvm_v2_u32(c, out_count);
    if (r != NVM_V2_OK) return r;
    if (*out_count == 0) return NVM_V2_OK;
    if ((size_t)*out_count > (size - c->pos) / stride) return NVM_V2_ERR_TRUNCATED;
    return NVM_V2_OK;
}

/* ── IMPORTS ────────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_imports_decode(const uint8_t *data, size_t size,
                                  NvmV2Imports *out) {
    out->items = NULL; out->count = 0;
    NvmV2Cursor c; nvm_v2_cursor_init(&c, data, size);
    uint32_t count;
    NvmV2Result r = read_count(&c, size, IMPORT_ENTRY_BYTES, &count);
    if (r != NVM_V2_OK || count == 0) return r;

    NvmV2Import *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        uint8_t kind, p0, p1, p2;
        if ((r = nvm_v2_u32(&c, &items[i].module_name_idx)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].symbol_name_idx)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].signature_idx))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &kind)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &p0))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &p1))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &p2))   != NVM_V2_OK) goto fail;
        if (kind > NVM_V2_IMPORT_KIND_MAX) { r = NVM_V2_ERR_SECTION_TYPE; goto fail; }
        if (p0 || p1 || p2) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }
        items[i].kind = kind;
    }
    out->items = items; out->count = count;
    return NVM_V2_OK;
fail:
    free(items);
    return r;
}

void nvm_v2_imports_free(NvmV2Imports *i) {
    if (!i) return;
    free(i->items); i->items = NULL; i->count = 0;
}

size_t nvm_v2_imports_encoded_size(const NvmV2Imports *i) {
    return 4 + (size_t)i->count * IMPORT_ENTRY_BYTES;
}

NvmV2Result nvm_v2_imports_encode(const NvmV2Imports *i,
                                  uint8_t *out, size_t size) {
    size_t need = nvm_v2_imports_encoded_size(i);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);
    size_t p = 0;
    wr32(out + p, i->count); p += 4;
    for (uint32_t k = 0; k < i->count; k++) {
        wr32(out + p, i->items[k].module_name_idx); p += 4;
        wr32(out + p, i->items[k].symbol_name_idx); p += 4;
        wr32(out + p, i->items[k].signature_idx);   p += 4;
        out[p] = i->items[k].kind;
        p += 4;   /* kind plus three reserved bytes, already zero */
    }
    return NVM_V2_OK;
}

/* ── LINKS ──────────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_links_decode(const uint8_t *data, size_t size,
                                NvmV2Links *out) {
    out->items = NULL; out->count = 0;
    NvmV2Cursor c; nvm_v2_cursor_init(&c, data, size);
    uint32_t count;
    NvmV2Result r = read_count(&c, size, LINK_ENTRY_BYTES, &count);
    if (r != NVM_V2_OK || count == 0) return r;

    NvmV2Link *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        if ((r = nvm_v2_u32(&c, &items[i].module_name_idx)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].symbol_name_idx)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].signature_idx))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].flags))           != NVM_V2_OK) goto fail;
        /* Fail closed on a flag this reader does not implement, as the header
         * does for feature bits. */
        if (items[i].flags & ~(uint32_t)NVM_V2_LINK_KNOWN_FLAGS) {
            r = NVM_V2_ERR_RESERVED_FLAGS; goto fail;
        }
    }
    out->items = items; out->count = count;
    return NVM_V2_OK;
fail:
    free(items);
    return r;
}

void nvm_v2_links_free(NvmV2Links *l) {
    if (!l) return;
    free(l->items); l->items = NULL; l->count = 0;
}

size_t nvm_v2_links_encoded_size(const NvmV2Links *l) {
    return 4 + (size_t)l->count * LINK_ENTRY_BYTES;
}

NvmV2Result nvm_v2_links_encode(const NvmV2Links *l, uint8_t *out, size_t size) {
    size_t need = nvm_v2_links_encoded_size(l);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);
    size_t p = 0;
    wr32(out + p, l->count); p += 4;
    for (uint32_t k = 0; k < l->count; k++) {
        wr32(out + p, l->items[k].module_name_idx); p += 4;
        wr32(out + p, l->items[k].symbol_name_idx); p += 4;
        wr32(out + p, l->items[k].signature_idx);   p += 4;
        wr32(out + p, l->items[k].flags);           p += 4;
    }
    return NVM_V2_OK;
}

/* ── METADATA ───────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_metadata_decode(const uint8_t *data, size_t size,
                                   NvmV2Metadata *out) {
    out->items = NULL; out->count = 0;
    NvmV2Cursor c; nvm_v2_cursor_init(&c, data, size);
    uint32_t count;
    NvmV2Result r = read_count(&c, size, METADATA_ENTRY_BYTES, &count);
    if (r != NVM_V2_OK || count == 0) return r;

    NvmV2MetadataEntry *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        if ((r = nvm_v2_u32(&c, &items[i].key_idx))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].value_idx)) != NVM_V2_OK) goto fail;
    }
    out->items = items; out->count = count;
    return NVM_V2_OK;
fail:
    free(items);
    return r;
}

void nvm_v2_metadata_free(NvmV2Metadata *m) {
    if (!m) return;
    free(m->items); m->items = NULL; m->count = 0;
}

size_t nvm_v2_metadata_encoded_size(const NvmV2Metadata *m) {
    return 4 + (size_t)m->count * METADATA_ENTRY_BYTES;
}

NvmV2Result nvm_v2_metadata_encode(const NvmV2Metadata *m,
                                   uint8_t *out, size_t size) {
    size_t need = nvm_v2_metadata_encoded_size(m);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);
    size_t p = 0;
    wr32(out + p, m->count); p += 4;
    for (uint32_t k = 0; k < m->count; k++) {
        wr32(out + p, m->items[k].key_idx);   p += 4;
        wr32(out + p, m->items[k].value_idx); p += 4;
    }
    return NVM_V2_OK;
}

/* ── DEBUG ──────────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_debug_decode(const uint8_t *data, size_t size,
                                NvmV2Debug *out) {
    out->items = NULL; out->count = 0;
    NvmV2Cursor c; nvm_v2_cursor_init(&c, data, size);
    uint32_t count;
    NvmV2Result r = read_count(&c, size, DEBUG_ENTRY_BYTES, &count);
    if (r != NVM_V2_OK || count == 0) return r;

    NvmV2DebugEntry *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        if ((r = nvm_v2_u64(&c, &items[i].bytecode_offset)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].source_line))     != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].source_col))      != NVM_V2_OK) goto fail;
    }
    out->items = items; out->count = count;
    return NVM_V2_OK;
fail:
    free(items);
    return r;
}

void nvm_v2_debug_free(NvmV2Debug *d) {
    if (!d) return;
    free(d->items); d->items = NULL; d->count = 0;
}

size_t nvm_v2_debug_encoded_size(const NvmV2Debug *d) {
    return 4 + (size_t)d->count * DEBUG_ENTRY_BYTES;
}

NvmV2Result nvm_v2_debug_encode(const NvmV2Debug *d, uint8_t *out, size_t size) {
    size_t need = nvm_v2_debug_encoded_size(d);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);
    size_t p = 0;
    wr32(out + p, d->count); p += 4;
    for (uint32_t k = 0; k < d->count; k++) {
        wr64(out + p, d->items[k].bytecode_offset); p += 8;
        wr32(out + p, d->items[k].source_line);     p += 4;
        wr32(out + p, d->items[k].source_col);      p += 4;
    }
    return NVM_V2_OK;
}
