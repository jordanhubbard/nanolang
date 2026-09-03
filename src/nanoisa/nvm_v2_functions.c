/*
 * v2 FUNCTIONS and GLOBALS sections.
 *
 * Both are fixed-width record tables with no variable-length tail, which is
 * itself the point: v1's function entries carried arity and result shape
 * inline and its import entries carried a variable-length type tail. Both now
 * reference SIGNATURES by index instead, so these records are a fixed stride
 * and a decoder can bound the whole table from the count alone.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "isa.h"

#define FN_ENTRY_BYTES 32
#define GL_ENTRY_BYTES 12

static void wr16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static void wr32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;         p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void wr64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (i * 8));
}

/* ── FUNCTIONS ──────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_functions_decode(const uint8_t *data, size_t size,
                                    NvmV2Functions *out) {
    out->items = NULL;
    out->count = 0;

    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, data, size);

    uint32_t count;
    NvmV2Result r = nvm_v2_u32(&c, &count);
    if (r != NVM_V2_OK) return r;
    if (count == 0) return NVM_V2_OK;

    if ((size_t)count > (size - c.pos) / FN_ENTRY_BYTES)
        return NVM_V2_ERR_TRUNCATED;

    NvmV2Function *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        uint16_t flags;
        if ((r = nvm_v2_u32(&c, &items[i].name_idx))      != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].signature_idx)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u64(&c, &items[i].code_offset))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u64(&c, &items[i].code_length))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u16(&c, &items[i].local_count))   != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u16(&c, &items[i].upvalue_count)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u16(&c, &items[i].max_stack))     != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u16(&c, &flags))                  != NVM_V2_OK) goto fail;
        if (flags != 0) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }
    }

    out->items = items;
    out->count = count;
    return NVM_V2_OK;

fail:
    free(items);
    return r;
}

void nvm_v2_functions_free(NvmV2Functions *f) {
    if (!f) return;
    free(f->items);
    f->items = NULL;
    f->count = 0;
}

size_t nvm_v2_functions_encoded_size(const NvmV2Functions *f) {
    return 4 + (size_t)f->count * FN_ENTRY_BYTES;
}

NvmV2Result nvm_v2_functions_encode(const NvmV2Functions *f,
                                    uint8_t *out, size_t size) {
    size_t need = nvm_v2_functions_encoded_size(f);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);   /* zeroes the reserved flags field of every entry */

    size_t p = 0;
    wr32(out + p, f->count); p += 4;
    for (uint32_t i = 0; i < f->count; i++) {
        const NvmV2Function *e = &f->items[i];
        wr32(out + p, e->name_idx);      p += 4;
        wr32(out + p, e->signature_idx); p += 4;
        wr64(out + p, e->code_offset);   p += 8;
        wr64(out + p, e->code_length);   p += 8;
        wr16(out + p, e->local_count);   p += 2;
        wr16(out + p, e->upvalue_count); p += 2;
        wr16(out + p, e->max_stack);     p += 2;
        p += 2;                          /* reserved flags, already zero */
    }
    return NVM_V2_OK;
}

/* ── GLOBALS ────────────────────────────────────────────────────────────── */

NvmV2Result nvm_v2_globals_decode(const uint8_t *data, size_t size,
                                  NvmV2Globals *out) {
    out->items = NULL;
    out->count = 0;

    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, data, size);

    uint32_t count;
    NvmV2Result r = nvm_v2_u32(&c, &count);
    if (r != NVM_V2_OK) return r;
    if (count == 0) return NVM_V2_OK;

    if ((size_t)count > (size - c.pos) / GL_ENTRY_BYTES)
        return NVM_V2_ERR_TRUNCATED;

    NvmV2Global *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        uint8_t tag, flags;
        uint16_t pad;
        if ((r = nvm_v2_u32(&c, &items[i].name_idx)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &tag))                != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &flags))              != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u16(&c, &pad))               != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u32(&c, &items[i].init_idx)) != NVM_V2_OK) goto fail;

        if (tag >= TAG_COUNT) { r = NVM_V2_ERR_SECTION_TYPE; goto fail; }
        /* Fail closed on a flag bit this reader does not know, for the same
         * reason the header rejects unknown feature bits. */
        if (flags & ~(uint8_t)NVM_V2_GLOBAL_KNOWN_FLAGS) {
            r = NVM_V2_ERR_RESERVED_FLAGS; goto fail;
        }
        if (pad != 0) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }

        items[i].type_tag = tag;
        items[i].flags = flags;
    }

    out->items = items;
    out->count = count;
    return NVM_V2_OK;

fail:
    free(items);
    return r;
}

void nvm_v2_globals_free(NvmV2Globals *g) {
    if (!g) return;
    free(g->items);
    g->items = NULL;
    g->count = 0;
}

size_t nvm_v2_globals_encoded_size(const NvmV2Globals *g) {
    return 4 + (size_t)g->count * GL_ENTRY_BYTES;
}

NvmV2Result nvm_v2_globals_encode(const NvmV2Globals *g,
                                  uint8_t *out, size_t size) {
    size_t need = nvm_v2_globals_encoded_size(g);
    if (size < need) return NVM_V2_ERR_TRUNCATED;
    memset(out, 0, need);   /* zeroes each entry's reserved _pad */

    size_t p = 0;
    wr32(out + p, g->count); p += 4;
    for (uint32_t i = 0; i < g->count; i++) {
        const NvmV2Global *e = &g->items[i];
        wr32(out + p, e->name_idx); p += 4;
        out[p++] = e->type_tag;
        out[p++] = e->flags;
        p += 2;                      /* reserved pad, already zero */
        wr32(out + p, e->init_idx); p += 4;
    }
    return NVM_V2_OK;
}
