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
#include "nvm_format.h"   /* NvmModule, for the v1 bridge declared at the end */

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

/* ── IMPORTS and LINKS ───────────────────────────────────────────────────
 * Same shape, different meaning. An import is a foreign function reached
 * through the FFI or the co-process; a link is a call into another NanoISA
 * module resolved at link time.
 *
 *   IMPORTS                          LINKS
 *   count             u32            count             u32
 *   per entry (16 bytes):            per entry (16 bytes):
 *     module_name_idx u32              module_name_idx u32
 *     symbol_name_idx u32              symbol_name_idx u32
 *     signature_idx   u32              signature_idx   u32
 *     kind            u8               flags           u32  bit 0 = weak
 *     _pad            u8[3]
 *
 * Parameter counts and type tags are deliberately absent from both: they live
 * in SIGNATURES. That is what removes v1's variable-length import tail.
 */

typedef enum {
    NVM_V2_IMPORT_FFI       = 0,
    NVM_V2_IMPORT_COPROCESS = 1
} NvmV2ImportKind;

#define NVM_V2_IMPORT_KIND_MAX NVM_V2_IMPORT_COPROCESS

/* A weak link may resolve to nothing. Encoded and validated now; nothing
 * consumes it until the 4.4 capability work. */
#define NVM_V2_LINK_WEAK        0x01u
#define NVM_V2_LINK_KNOWN_FLAGS 0x01u

typedef struct {
    uint32_t module_name_idx;
    uint32_t symbol_name_idx;
    uint32_t signature_idx;
    uint8_t  kind;
} NvmV2Import;

typedef struct { NvmV2Import *items; uint32_t count; } NvmV2Imports;

typedef struct {
    uint32_t module_name_idx;
    uint32_t symbol_name_idx;
    uint32_t signature_idx;
    uint32_t flags;
} NvmV2Link;

typedef struct { NvmV2Link *items; uint32_t count; } NvmV2Links;

NvmV2Result nvm_v2_imports_decode(const uint8_t *data, size_t size, NvmV2Imports *out);
void        nvm_v2_imports_free(NvmV2Imports *i);
size_t      nvm_v2_imports_encoded_size(const NvmV2Imports *i);
NvmV2Result nvm_v2_imports_encode(const NvmV2Imports *i, uint8_t *out, size_t size);

NvmV2Result nvm_v2_links_decode(const uint8_t *data, size_t size, NvmV2Links *out);
void        nvm_v2_links_free(NvmV2Links *l);
size_t      nvm_v2_links_encoded_size(const NvmV2Links *l);
NvmV2Result nvm_v2_links_encode(const NvmV2Links *l, uint8_t *out, size_t size);

/* ── METADATA and DEBUG ──────────────────────────────────────────────────
 *   METADATA                         DEBUG
 *   count           u32              count             u32
 *   per entry (8):                   per entry (16):
 *     key_idx       u32                bytecode_offset u64  widened from v1
 *     value_idx     u32                source_line     u32
 *                                      source_col      u32  1-based, 0=unknown
 *
 * METADATA is free-form key/value into CONSTANTS, so adding a key is not a
 * format change.
 */

typedef struct { uint32_t key_idx, value_idx; } NvmV2MetadataEntry;
typedef struct { NvmV2MetadataEntry *items; uint32_t count; } NvmV2Metadata;

typedef struct { uint64_t bytecode_offset; uint32_t source_line, source_col; } NvmV2DebugEntry;
typedef struct { NvmV2DebugEntry *items; uint32_t count; } NvmV2Debug;

NvmV2Result nvm_v2_metadata_decode(const uint8_t *data, size_t size, NvmV2Metadata *out);
void        nvm_v2_metadata_free(NvmV2Metadata *m);
size_t      nvm_v2_metadata_encoded_size(const NvmV2Metadata *m);
NvmV2Result nvm_v2_metadata_encode(const NvmV2Metadata *m, uint8_t *out, size_t size);

NvmV2Result nvm_v2_debug_decode(const uint8_t *data, size_t size, NvmV2Debug *out);
void        nvm_v2_debug_free(NvmV2Debug *d);
size_t      nvm_v2_debug_encoded_size(const NvmV2Debug *d);
NvmV2Result nvm_v2_debug_encode(const NvmV2Debug *d, uint8_t *out, size_t size);

/* ── Whole module ────────────────────────────────────────────────────────
 * Every section, plus the CODE byte range the directory locates.
 *
 * Serialization writes the header, then the directory, then payloads in
 * ascending section-type order, then patches the checksum. Deserialization
 * validates the container first and decodes only what the directory names.
 *
 * Cross-section validation lives here rather than in the codecs: a codec sees
 * one section and cannot check an index into another.
 */

typedef struct {
    uint16_t        isa_version;
    uint32_t        entry_point;   /* FUNCTIONS index, or NVM_V2_NO_ENTRY_POINT */
    NvmV2Metadata   metadata;
    NvmV2Constants  constants;
    NvmV2Signatures signatures;
    NvmV2Layouts    layouts;
    NvmV2Functions  functions;
    NvmV2Globals    globals;
    NvmV2Imports    imports;
    NvmV2Links      links;
    NvmV2Debug      debug;
    const uint8_t  *code;          /* aliases the module buffer when decoded */
    uint64_t        code_size;
    bool            has_debug;     /* DEBUG present, even if empty */

    /* Signature tag arrays alias the buffer when a module is decoded, so
     * nothing owns them. A producer that synthesizes signatures has nowhere
     * to put the bytes, so it parks them here in one block and
     * nvm_v2_module_free releases it. NULL for a decoded module. */
    uint8_t        *owned_tags;
} NvmV2Module;

/* Serialize into a caller-provided buffer. Returns the byte length via
 * `out_size`; pass out=NULL to size only. */
NvmV2Result nvm_v2_module_serialize(const NvmV2Module *m,
                                    uint8_t *out, size_t capacity,
                                    size_t *out_size);

/* Validate the container, decode every section, then check cross-section
 * indices. The buffer must outlive `out`. */
NvmV2Result nvm_v2_module_deserialize(const uint8_t *data, size_t size,
                                      NvmV2Module *out);

void nvm_v2_module_free(NvmV2Module *m);

/* ── The v1 bridge ──────────────────────────────────────────────────────────
 *
 * These let v2 be adopted without rewriting every producer at once. Declared
 * here rather than in a header of their own so a caller needs one include.
 *
 * `nvm_v2_from_nvm_module` builds an owning NvmV2Module: free it with
 * nvm_v2_module_free. `nvm_v2_to_nvm_module` allocates an NvmModule: free it
 * with nvm_module_free.
 *
 * A v1 module does not record function parameter types or max_stack. The
 * bridge emits TAG_VOID placeholder parameter tags and a max_stack of 0
 * rather than guessing; a v2-native producer supplies the real values.
 */
NvmV2Result nvm_v2_from_nvm_module(const NvmModule *mod, NvmV2Module *out);
NvmV2Result nvm_v2_to_nvm_module(const NvmV2Module *m, NvmModule **out);

#endif /* NANOISA_NVM_V2_SECTIONS_H */
