/*
 * Bounds-checked cursor for decoding v2 section payloads.
 *
 * Every bound here is written by subtraction. `pos + n > size` can wrap when
 * `n` comes from a module being decoded -- and it always does -- which is how
 * a crafted length slips past an addition-form check. The invariant
 * `pos <= size` is maintained by every operation, so `n > size - pos` cannot
 * overflow.
 */

#include "nvm_v2_sections.h"

void nvm_v2_cursor_init(NvmV2Cursor *c, const uint8_t *base, size_t size) {
    c->base = base;
    c->size = size;
    c->pos  = 0;
}

bool nvm_v2_cursor_exhausted(const NvmV2Cursor *c) {
    return c->pos >= c->size;
}

NvmV2Result nvm_v2_take(NvmV2Cursor *c, size_t n, const uint8_t **out) {
    if (n > c->size - c->pos) return NVM_V2_ERR_TRUNCATED;
    if (out) *out = c->base + c->pos;
    c->pos += n;
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_u8(NvmV2Cursor *c, uint8_t *out) {
    const uint8_t *p;
    NvmV2Result r = nvm_v2_take(c, 1, &p);
    if (r == NVM_V2_OK) *out = p[0];
    return r;
}

NvmV2Result nvm_v2_u16(NvmV2Cursor *c, uint16_t *out) {
    const uint8_t *p;
    NvmV2Result r = nvm_v2_take(c, 2, &p);
    if (r == NVM_V2_OK) *out = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
    return r;
}

NvmV2Result nvm_v2_u32(NvmV2Cursor *c, uint32_t *out) {
    const uint8_t *p;
    NvmV2Result r = nvm_v2_take(c, 4, &p);
    if (r == NVM_V2_OK)
        *out = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
             | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return r;
}

NvmV2Result nvm_v2_u64(NvmV2Cursor *c, uint64_t *out) {
    const uint8_t *p;
    NvmV2Result r = nvm_v2_take(c, 8, &p);
    if (r != NVM_V2_OK) return r;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
    *out = v;
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_align4(NvmV2Cursor *c) {
    while (c->pos % 4 != 0) {
        uint8_t pad;
        NvmV2Result r = nvm_v2_u8(c, &pad);
        if (r != NVM_V2_OK) return r;   /* section ended mid-padding */
        if (pad != 0) return NVM_V2_ERR_SECTION_RANGE;
    }
    return NVM_V2_OK;
}
