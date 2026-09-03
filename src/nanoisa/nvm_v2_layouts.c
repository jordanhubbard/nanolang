/*
 * v2 LAYOUTS section: the on-disk referent for AGG_PACK, AGG_GET, AGG_SET and
 * AGG_TAG.
 *
 * The table is closed: every nested layout index refers to a lower-numbered
 * entry. That is enforced here at decode, which is what lets everything
 * downstream walk a layout without a visited set and without a depth limit --
 * a forward or self reference cannot exist in a table that decoded.
 *
 * Fields are copied rather than aliased, unlike CONSTANTS and SIGNATURES: a
 * field is three fixed-width values, so a caller reading one wants a struct
 * rather than an offset into the buffer, and the array is small.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "isa.h"

/* kind, reserved, field_count, name_idx. */
#define ENTRY_HEADER_BYTES 8
/* type_tag, three reserved, nested_idx, name_idx. */
#define FIELD_BYTES        12

static void free_fields(NvmV2Layout *items, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) free(items[i].fields);
}

NvmV2Result nvm_v2_layouts_decode(const uint8_t *data, size_t size,
                                  NvmV2Layouts *out) {
    out->items = NULL;
    out->count = 0;

    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, data, size);

    uint32_t count;
    NvmV2Result r = nvm_v2_u32(&c, &count);
    if (r != NVM_V2_OK) return r;
    if (count == 0) return NVM_V2_OK;

    if ((size_t)count > (size - c.pos) / ENTRY_HEADER_BYTES)
        return NVM_V2_ERR_TRUNCATED;

    NvmV2Layout *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    uint32_t built = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t kind, pad;
        uint16_t field_count;
        uint32_t name_idx;

        if ((r = nvm_v2_u8(&c, &kind))         != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &pad))          != NVM_V2_OK) goto fail;
        if (pad) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }
        if (kind > NVM_V2_LAYOUT_KIND_MAX) { r = NVM_V2_ERR_SECTION_TYPE; goto fail; }
        if ((r = nvm_v2_u16(&c, &field_count)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &name_idx))    != NVM_V2_OK) goto fail;

        items[i].kind = kind;
        items[i].field_count = field_count;
        items[i].name_idx = name_idx;
        items[i].fields = NULL;
        built = i + 1;

        if (field_count == 0) continue;

        /* Bound the field array against what remains before allocating. */
        if ((size_t)field_count > (size - c.pos) / FIELD_BYTES) {
            r = NVM_V2_ERR_TRUNCATED;
            goto fail;
        }
        NvmV2LayoutField *fields = calloc(field_count, sizeof *fields);
        if (!fields) { r = NVM_V2_ERR_TRUNCATED; goto fail; }
        items[i].fields = fields;

        for (uint16_t f = 0; f < field_count; f++) {
            uint8_t tag, p0, p1, p2;
            uint32_t nested, fname;
            if ((r = nvm_v2_u8(&c, &tag)) != NVM_V2_OK) goto fail;
            if ((r = nvm_v2_u8(&c, &p0))  != NVM_V2_OK) goto fail;
            if ((r = nvm_v2_u8(&c, &p1))  != NVM_V2_OK) goto fail;
            if ((r = nvm_v2_u8(&c, &p2))  != NVM_V2_OK) goto fail;
            if (p0 || p1 || p2) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }
            if (tag >= TAG_COUNT) { r = NVM_V2_ERR_SECTION_TYPE; goto fail; }
            if ((r = nvm_v2_u32(&c, &nested)) != NVM_V2_OK) goto fail;
            if ((r = nvm_v2_u32(&c, &fname))  != NVM_V2_OK) goto fail;

            /* The closure property. A nested layout must already have been
             * decoded, so it must be strictly lower-numbered than this one.
             * `nested >= i` covers both a forward reference and a layout
             * nesting itself. */
            if (nested != NVM_V2_NO_INDEX && nested >= i) {
                r = NVM_V2_ERR_INDEX_RANGE;
                goto fail;
            }

            fields[f].type_tag   = tag;
            fields[f].nested_idx = nested;
            fields[f].name_idx   = fname;
        }
    }

    out->items = items;
    out->count = count;
    return NVM_V2_OK;

fail:
    free_fields(items, built);
    free(items);
    return r;
}

void nvm_v2_layouts_free(NvmV2Layouts *l) {
    if (!l || !l->items) { if (l) { l->items = NULL; l->count = 0; } return; }
    free_fields(l->items, l->count);
    free(l->items);
    l->items = NULL;
    l->count = 0;
}

size_t nvm_v2_layouts_encoded_size(const NvmV2Layouts *l) {
    size_t n = 4;   /* count */
    for (uint32_t i = 0; i < l->count; i++)
        n += ENTRY_HEADER_BYTES + (size_t)l->items[i].field_count * FIELD_BYTES;
    return n;
}

static void wr32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;         p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

NvmV2Result nvm_v2_layouts_encode(const NvmV2Layouts *l,
                                  uint8_t *out, size_t size) {
    size_t need = nvm_v2_layouts_encoded_size(l);
    if (size < need) return NVM_V2_ERR_TRUNCATED;

    /* Zero first: the decoder rejects nonzero reserved bytes, so this is what
     * makes them correct without writing each one. */
    memset(out, 0, need);

    size_t p = 0;
    wr32(out + p, l->count); p += 4;

    for (uint32_t i = 0; i < l->count; i++) {
        const NvmV2Layout *e = &l->items[i];
        out[p] = e->kind;
        p += 2;                              /* kind plus one reserved byte */
        out[p++] = (uint8_t)e->field_count;
        out[p++] = (uint8_t)(e->field_count >> 8);
        wr32(out + p, e->name_idx); p += 4;

        for (uint16_t f = 0; f < e->field_count; f++) {
            out[p] = e->fields[f].type_tag;
            p += 4;                          /* tag plus three reserved bytes */
            wr32(out + p, e->fields[f].nested_idx); p += 4;
            wr32(out + p, e->fields[f].name_idx);   p += 4;
        }
    }
    return NVM_V2_OK;
}
