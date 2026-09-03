/*
 * v2 CONSTANTS section: a typed constant pool.
 *
 * Replaces v1's string pool. Each entry carries an explicit byte length, so a
 * string keeps its bytes verbatim and an embedded zero survives a round trip.
 * v1 stored strings the pool could only measure with strlen, which truncated
 * silently at the first zero.
 *
 * Payloads alias the decoded buffer rather than being copied: the caller
 * already holds the module bytes, and copying every constant would double the
 * cost of loading for no benefit. The header says so, and the module loader
 * keeps the buffer alive for the module's lifetime.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "isa.h"

/* Bytes an entry's payload occupies once padded to a 4-byte boundary. */
static size_t padded(size_t n) { return (n + 3u) & ~(size_t)3u; }

/* Fixed part of an entry: tag, three reserved bytes, length. */
#define ENTRY_HEADER_BYTES 8

NvmV2Result nvm_v2_constants_decode(const uint8_t *data, size_t size,
                                    NvmV2Constants *out) {
    out->items = NULL;
    out->count = 0;

    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, data, size);

    uint32_t count;
    NvmV2Result r = nvm_v2_u32(&c, &count);
    if (r != NVM_V2_OK) return r;
    if (count == 0) return NVM_V2_OK;

    /* Every entry needs at least ENTRY_HEADER_BYTES, so a count larger than
     * the remaining bytes can supply is malformed. Checked before allocating,
     * so a four-byte section claiming four billion entries is refused on the
     * arithmetic rather than by failing to allocate for them. */
    if ((size_t)count > (size - c.pos) / ENTRY_HEADER_BYTES)
        return NVM_V2_ERR_TRUNCATED;

    NvmV2Constant *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        uint8_t tag, pad0, pad1, pad2;
        uint32_t length;

        if ((r = nvm_v2_u8(&c, &tag))  != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &pad0)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &pad1)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u8(&c, &pad2)) != NVM_V2_OK) goto fail;
        if (pad0 || pad1 || pad2) { r = NVM_V2_ERR_RESERVED_FLAGS; goto fail; }
        if (tag >= TAG_COUNT)     { r = NVM_V2_ERR_SECTION_TYPE;   goto fail; }

        if ((r = nvm_v2_u32(&c, &length)) != NVM_V2_OK) goto fail;

        const uint8_t *payload = NULL;
        if ((r = nvm_v2_take(&c, length, &payload)) != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_align4(&c)) != NVM_V2_OK) goto fail;

        items[i].tag     = tag;
        items[i].length  = length;
        items[i].payload = payload;
    }

    out->items = items;
    out->count = count;
    return NVM_V2_OK;

fail:
    free(items);
    return r;
}

void nvm_v2_constants_free(NvmV2Constants *c) {
    if (!c) return;
    free(c->items);
    c->items = NULL;
    c->count = 0;
}

size_t nvm_v2_constants_encoded_size(const NvmV2Constants *c) {
    size_t n = 4;   /* count */
    for (uint32_t i = 0; i < c->count; i++)
        n += ENTRY_HEADER_BYTES + padded(c->items[i].length);
    return n;
}

NvmV2Result nvm_v2_constants_encode(const NvmV2Constants *c,
                                    uint8_t *out, size_t size) {
    size_t need = nvm_v2_constants_encoded_size(c);
    if (size < need) return NVM_V2_ERR_TRUNCATED;

    /* Zero first so every reserved byte and every payload pad is zero without
     * writing them individually. The decoder rejects nonzero padding, so this
     * is load-bearing rather than tidiness. */
    memset(out, 0, need);

    size_t p = 0;
    out[p++] = (uint8_t)c->count;
    out[p++] = (uint8_t)(c->count >> 8);
    out[p++] = (uint8_t)(c->count >> 16);
    out[p++] = (uint8_t)(c->count >> 24);

    for (uint32_t i = 0; i < c->count; i++) {
        uint32_t len = c->items[i].length;
        out[p] = c->items[i].tag;
        p += 4;                       /* tag plus three reserved zero bytes */
        out[p++] = (uint8_t)len;
        out[p++] = (uint8_t)(len >> 8);
        out[p++] = (uint8_t)(len >> 16);
        out[p++] = (uint8_t)(len >> 24);
        if (len) memcpy(out + p, c->items[i].payload, len);
        p += padded(len);
    }
    return NVM_V2_OK;
}
