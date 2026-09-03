/*
 * NanoISA v2 section payloads: shared decoding cursor and per-section codecs.
 *
 * Implements the section encodings in
 * docs/superpowers/specs/2026-09-01-nanoisa-v2-module-format.md, per the plan
 * in docs/superpowers/plans/2026-09-02-nanoisa-v2-module-format.md.
 *
 * The container (header and section directory) is nvm_format_v2.h; this is
 * what lives inside the sections it locates.
 */

#ifndef NANOISA_NVM_V2_SECTIONS_H
#define NANOISA_NVM_V2_SECTIONS_H

#include "nvm_format_v2.h"

/* ── Cursor ──────────────────────────────────────────────────────────────
 * Every section decoder walks a byte range and must never read past it.
 * Writing that check once, here, removes the most likely bug from eight
 * decoders and keeps the subtraction-form bound in a single place.
 */

typedef struct {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
} NvmV2Cursor;

void        nvm_v2_cursor_init(NvmV2Cursor *c, const uint8_t *base, size_t size);
bool        nvm_v2_cursor_exhausted(const NvmV2Cursor *c);

/* Advance by `n` bytes, yielding a pointer to them. Fails rather than
 * clamping: a short read is a malformed section, not a partial one. */
NvmV2Result nvm_v2_take(NvmV2Cursor *c, size_t n, const uint8_t **out);

NvmV2Result nvm_v2_u8 (NvmV2Cursor *c, uint8_t  *out);
NvmV2Result nvm_v2_u16(NvmV2Cursor *c, uint16_t *out);
NvmV2Result nvm_v2_u32(NvmV2Cursor *c, uint32_t *out);
NvmV2Result nvm_v2_u64(NvmV2Cursor *c, uint64_t *out);

/* Skip to the next 4-byte boundary, requiring the padding be zero. Arbitrary
 * filler would let two different byte strings decode identically, which breaks
 * the lossless round-tripping the disassembler depends on. */
NvmV2Result nvm_v2_align4(NvmV2Cursor *c);

#endif /* NANOISA_NVM_V2_SECTIONS_H */
