/*
 * v2 SIGNATURES section: the single source of truth for call shapes.
 *
 * Functions, imports, links and indirect call sites reference a signature by
 * index rather than each carrying their own shape. That is what lets
 * verification compare indices instead of re-deriving a shape from three
 * different encodings, and it is why v2 import entries have no variable-length
 * type tail the way v1's did.
 *
 * The index comparison is only sound if producers deduplicate, so
 * nvm_v2_signature_equal lives here beside the codec rather than in the
 * converter that uses it.
 */

#include <stdlib.h>
#include <string.h>
#include "nvm_v2_sections.h"
#include "isa.h"

static size_t padded(size_t n) { return (n + 3u) & ~(size_t)3u; }

/* param_count and result_count. */
#define ENTRY_HEADER_BYTES 4

bool nvm_v2_signature_equal(const NvmV2Signature *a, const NvmV2Signature *b) {
    if (a->param_count != b->param_count) return false;
    if (a->result_count != b->result_count) return false;
    /* memcmp of zero bytes is defined and returns 0, but the pointers may be
     * NULL for an empty array, which memcmp does not permit. */
    if (a->param_count &&
        memcmp(a->param_tags, b->param_tags, a->param_count) != 0) return false;
    if (a->result_count &&
        memcmp(a->result_tags, b->result_tags, a->result_count) != 0) return false;
    return true;
}

/* Read `n` tag bytes, rejecting any that is not a valid NanoValueTag, then
 * consume the padding to the next 4-byte boundary. */
static NvmV2Result take_tags(NvmV2Cursor *c, uint16_t n, const uint8_t **out) {
    const uint8_t *p = NULL;
    NvmV2Result r = nvm_v2_take(c, n, &p);
    if (r != NVM_V2_OK) return r;
    for (uint16_t i = 0; i < n; i++)
        if (p[i] >= TAG_COUNT) return NVM_V2_ERR_SECTION_TYPE;
    r = nvm_v2_align4(c);
    if (r != NVM_V2_OK) return r;
    *out = n ? p : NULL;
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_signatures_decode(const uint8_t *data, size_t size,
                                     NvmV2Signatures *out) {
    out->items = NULL;
    out->count = 0;

    NvmV2Cursor c;
    nvm_v2_cursor_init(&c, data, size);

    uint32_t count;
    NvmV2Result r = nvm_v2_u32(&c, &count);
    if (r != NVM_V2_OK) return r;
    if (count == 0) return NVM_V2_OK;

    /* Refuse an impossible count on the arithmetic rather than by failing to
     * allocate for it. */
    if ((size_t)count > (size - c.pos) / ENTRY_HEADER_BYTES)
        return NVM_V2_ERR_TRUNCATED;

    NvmV2Signature *items = calloc(count, sizeof *items);
    if (!items) return NVM_V2_ERR_TRUNCATED;

    for (uint32_t i = 0; i < count; i++) {
        uint16_t params, results;
        if ((r = nvm_v2_u16(&c, &params))  != NVM_V2_OK) goto fail;
        if ((r = nvm_v2_u16(&c, &results)) != NVM_V2_OK) goto fail;

        const uint8_t *ptags = NULL, *rtags = NULL;
        if ((r = take_tags(&c, params, &ptags))  != NVM_V2_OK) goto fail;
        if ((r = take_tags(&c, results, &rtags)) != NVM_V2_OK) goto fail;

        items[i].param_count  = params;
        items[i].result_count = results;
        items[i].param_tags   = ptags;
        items[i].result_tags  = rtags;
    }

    out->items = items;
    out->count = count;
    return NVM_V2_OK;

fail:
    free(items);
    return r;
}

void nvm_v2_signatures_free(NvmV2Signatures *s) {
    if (!s) return;
    free(s->items);
    s->items = NULL;
    s->count = 0;
}

size_t nvm_v2_signatures_encoded_size(const NvmV2Signatures *s) {
    size_t n = 4;   /* count */
    for (uint32_t i = 0; i < s->count; i++)
        n += ENTRY_HEADER_BYTES
           + padded(s->items[i].param_count)
           + padded(s->items[i].result_count);
    return n;
}

NvmV2Result nvm_v2_signatures_encode(const NvmV2Signatures *s,
                                     uint8_t *out, size_t size) {
    size_t need = nvm_v2_signatures_encoded_size(s);
    if (size < need) return NVM_V2_ERR_TRUNCATED;

    /* Zeroing first is what makes every tag-array pad byte zero; the decoder
     * rejects nonzero padding, so this is load-bearing. */
    memset(out, 0, need);

    size_t p = 0;
    out[p++] = (uint8_t)s->count;
    out[p++] = (uint8_t)(s->count >> 8);
    out[p++] = (uint8_t)(s->count >> 16);
    out[p++] = (uint8_t)(s->count >> 24);

    for (uint32_t i = 0; i < s->count; i++) {
        uint16_t params  = s->items[i].param_count;
        uint16_t results = s->items[i].result_count;
        out[p++] = (uint8_t)params;
        out[p++] = (uint8_t)(params >> 8);
        out[p++] = (uint8_t)results;
        out[p++] = (uint8_t)(results >> 8);
        if (params)  memcpy(out + p, s->items[i].param_tags, params);
        p += padded(params);
        if (results) memcpy(out + p, s->items[i].result_tags, results);
        p += padded(results);
    }
    return NVM_V2_OK;
}
