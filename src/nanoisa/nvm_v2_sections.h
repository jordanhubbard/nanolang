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

/* ── CONSTANTS ───────────────────────────────────────────────────────────
 * Replaces v1's string pool with a typed constant pool.
 *
 *   count       u32
 *   per entry:
 *     tag       u8    NanoValueTag
 *     _pad      u8[3] must be zero
 *     length    u32   payload byte length
 *     payload   u8[length], zero-padded to a 4-byte boundary
 *
 * The explicit length is the point: strings keep their bytes verbatim, so an
 * embedded zero survives a round trip. This is the serialized half of the
 * stored-string-length work -- a strlen-based pool truncates silently.
 */

typedef struct {
    uint8_t        tag;      /* NanoValueTag */
    uint32_t       length;   /* payload bytes */
    const uint8_t *payload;  /* aliases the decoded buffer; not owned */
} NvmV2Constant;

typedef struct {
    NvmV2Constant *items;
    uint32_t       count;
} NvmV2Constants;

/* Decodes in place: items[].payload points into `data`, which must outlive
 * `out`. Frees nothing on failure beyond its own working state. */
NvmV2Result nvm_v2_constants_decode(const uint8_t *data, size_t size,
                                    NvmV2Constants *out);
void        nvm_v2_constants_free(NvmV2Constants *c);
size_t      nvm_v2_constants_encoded_size(const NvmV2Constants *c);
NvmV2Result nvm_v2_constants_encode(const NvmV2Constants *c,
                                    uint8_t *out, size_t size);

/* ── SIGNATURES ──────────────────────────────────────────────────────────
 * The single source of truth for call shapes.
 *
 *   count          u32
 *   per entry:
 *     param_count  u16
 *     result_count u16
 *     param_tags   u8[param_count],  padded to 4
 *     result_tags  u8[result_count], padded to 4
 *
 * Functions, imports, links and indirect call sites all reference a signature
 * by index rather than each carrying their own shape. That is what lets
 * verification compare signature indices instead of re-deriving a shape from
 * three different encodings, and it is why v2 import entries have no
 * variable-length type tail the way v1's did.
 *
 * Producers must deduplicate: two identically-shaped callables share an entry,
 * or index comparison stops being a valid equality test.
 */

typedef struct {
    uint16_t       param_count;
    uint16_t       result_count;
    const uint8_t *param_tags;   /* aliases the decoded buffer; not owned */
    const uint8_t *result_tags;  /* aliases the decoded buffer; not owned */
} NvmV2Signature;

typedef struct {
    NvmV2Signature *items;
    uint32_t        count;
} NvmV2Signatures;

/* Decodes in place: the tag arrays point into `data`, which must outlive
 * `out`. */
NvmV2Result nvm_v2_signatures_decode(const uint8_t *data, size_t size,
                                     NvmV2Signatures *out);
void        nvm_v2_signatures_free(NvmV2Signatures *s);
size_t      nvm_v2_signatures_encoded_size(const NvmV2Signatures *s);
NvmV2Result nvm_v2_signatures_encode(const NvmV2Signatures *s,
                                     uint8_t *out, size_t size);

/* True when two signatures describe the same call shape. Producers use this to
 * deduplicate before assigning indices. */
bool nvm_v2_signature_equal(const NvmV2Signature *a, const NvmV2Signature *b);

/* ── LAYOUTS ─────────────────────────────────────────────────────────────
 * Gives AGG_PACK, AGG_GET, AGG_SET and AGG_TAG an on-disk referent.
 *
 *   count           u32
 *   per entry:
 *     kind          u8    NvmV2LayoutKind
 *     _pad          u8    must be zero
 *     field_count   u16
 *     name_idx      u32   CONSTANTS index, or NVM_V2_NO_INDEX
 *     per field:
 *       type_tag    u8
 *       _pad        u8[3] must be zero
 *       nested_idx  u32   layout index, or NVM_V2_NO_INDEX when scalar
 *       name_idx    u32   CONSTANTS index, or NVM_V2_NO_INDEX
 *
 * A layout is closed: every nested index refers to a LOWER-numbered layout.
 * That makes the table acyclic by construction, so a decoder can validate it
 * in one forward pass and nothing walking it can recurse forever. A forward or
 * self reference is rejected rather than merely unusual.
 */

#define NVM_V2_NO_INDEX 0xFFFFFFFFu

typedef enum {
    NVM_V2_LAYOUT_STRUCT = 0,
    NVM_V2_LAYOUT_TUPLE  = 1,
    NVM_V2_LAYOUT_UNION  = 2,
    NVM_V2_LAYOUT_ENUM   = 3
} NvmV2LayoutKind;

#define NVM_V2_LAYOUT_KIND_MAX NVM_V2_LAYOUT_ENUM

typedef struct {
    uint8_t  type_tag;
    uint32_t nested_idx;  /* lower-numbered layout, or NVM_V2_NO_INDEX */
    uint32_t name_idx;    /* CONSTANTS index, or NVM_V2_NO_INDEX */
} NvmV2LayoutField;

typedef struct {
    uint8_t           kind;
    uint16_t          field_count;
    uint32_t          name_idx;
    NvmV2LayoutField *fields;   /* owned; freed by nvm_v2_layouts_free */
} NvmV2Layout;

typedef struct {
    NvmV2Layout *items;
    uint32_t     count;
} NvmV2Layouts;

NvmV2Result nvm_v2_layouts_decode(const uint8_t *data, size_t size,
                                  NvmV2Layouts *out);
void        nvm_v2_layouts_free(NvmV2Layouts *l);
size_t      nvm_v2_layouts_encoded_size(const NvmV2Layouts *l);
NvmV2Result nvm_v2_layouts_encode(const NvmV2Layouts *l,
                                  uint8_t *out, size_t size);

/* ── FUNCTIONS ───────────────────────────────────────────────────────────
 *   count             u32
 *   per entry (32 bytes):
 *     name_idx        u32   CONSTANTS index
 *     signature_idx   u32   SIGNATURES index
 *     code_offset     u64   byte offset into CODE
 *     code_length     u64
 *     local_count     u16
 *     upvalue_count   u16
 *     max_stack       u16   verifier-proven maximum operand depth
 *     flags           u16   reserved, must be zero
 *
 * arity, result_tag and result_count are deliberately absent: they live in
 * SIGNATURES, referenced by signature_idx. max_stack is new, and is what lets
 * the verifier discharge the maximum-operand-depth obligation statically
 * rather than leaning on a runtime stack limit.
 */

typedef struct {
    uint32_t name_idx;
    uint32_t signature_idx;
    uint64_t code_offset;
    uint64_t code_length;
    uint16_t local_count;
    uint16_t upvalue_count;
    uint16_t max_stack;
} NvmV2Function;

typedef struct {
    NvmV2Function *items;
    uint32_t       count;
} NvmV2Functions;

NvmV2Result nvm_v2_functions_decode(const uint8_t *data, size_t size,
                                    NvmV2Functions *out);
void        nvm_v2_functions_free(NvmV2Functions *f);
size_t      nvm_v2_functions_encoded_size(const NvmV2Functions *f);
NvmV2Result nvm_v2_functions_encode(const NvmV2Functions *f,
                                    uint8_t *out, size_t size);

/* ── GLOBALS ─────────────────────────────────────────────────────────────
 *   count           u32
 *   per entry (12 bytes):
 *     name_idx      u32   CONSTANTS index
 *     type_tag      u8
 *     flags         u8    bit 0 = mutable; other bits reserved, must be zero
 *     _pad          u16   must be zero
 *     init_idx      u32   CONSTANTS index, or NVM_V2_NO_INDEX
 *
 * This section is what lets a VM size its globals from the module's own
 * declarations rather than embedding a fixed ceiling, and gives the verifier a
 * real bound to check LOAD_GLOBAL and STORE_GLOBAL operands against.
 */

#define NVM_V2_GLOBAL_MUTABLE 0x01u
#define NVM_V2_GLOBAL_KNOWN_FLAGS 0x01u

typedef struct {
    uint32_t name_idx;
    uint8_t  type_tag;
    uint8_t  flags;
    uint32_t init_idx;   /* CONSTANTS index, or NVM_V2_NO_INDEX */
} NvmV2Global;

typedef struct {
    NvmV2Global *items;
    uint32_t     count;
} NvmV2Globals;

NvmV2Result nvm_v2_globals_decode(const uint8_t *data, size_t size,
                                  NvmV2Globals *out);
void        nvm_v2_globals_free(NvmV2Globals *g);
size_t      nvm_v2_globals_encoded_size(const NvmV2Globals *g);
NvmV2Result nvm_v2_globals_encode(const NvmV2Globals *g,
                                  uint8_t *out, size_t size);

#endif /* NANOISA_NVM_V2_SECTIONS_H */
