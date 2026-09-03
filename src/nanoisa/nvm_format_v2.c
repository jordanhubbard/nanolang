/*
 * NanoISA v2 container: header and section directory.
 *
 * Every bounds check here is written with subtraction rather than addition.
 * `offset + size > total` can wrap and let a crafted range slip through; the
 * subtraction form cannot, because the operands are ordered first. The v1
 * loader learned this the hard way (see #160), so v2 starts that way.
 */

#include "nvm_format_v2.h"
#include "nvm_format.h"   /* nvm_crc32 */

/* ── Little-endian accessors ─────────────────────────────────────────────
 * Byte-wise on purpose: the wire format is little-endian regardless of host
 * byte order, and this avoids both unaligned loads and an endianness #ifdef.
 */

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
    return v;
}

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

/* ── Field offsets ──────────────────────────────────────────────────────── */

#define OFF_MAGIC          0
#define OFF_FORMAT_VERSION 4
#define OFF_ISA_VERSION    6
#define OFF_FEATURE_BITS   8
#define OFF_TOTAL_SIZE     12
#define OFF_HEADER_SIZE    20
#define OFF_SECTION_COUNT  24
#define OFF_ENTRY_POINT    28
#define OFF_FLAGS          32
#define OFF_CHECKSUM       36

#define SEC_OFF_TYPE   0
#define SEC_OFF_FLAGS  4
#define SEC_OFF_OFFSET 8
#define SEC_OFF_SIZE   16

const char *nvm_v2_result_name(NvmV2Result r) {
    switch (r) {
    case NVM_V2_OK:                   return "ok";
    case NVM_V2_ERR_TRUNCATED:        return "truncated";
    case NVM_V2_ERR_MAGIC:            return "bad-magic";
    case NVM_V2_ERR_FORMAT_VERSION:   return "unsupported-format-version";
    case NVM_V2_ERR_HEADER_SIZE:      return "bad-header-size";
    case NVM_V2_ERR_UNKNOWN_FEATURE:  return "unknown-feature-bit";
    case NVM_V2_ERR_TOTAL_SIZE:       return "total-size-mismatch";
    case NVM_V2_ERR_RESERVED_FLAGS:   return "reserved-flags-set";
    case NVM_V2_ERR_SECTION_TYPE:     return "unknown-section-type";
    case NVM_V2_ERR_SECTION_RANGE:    return "section-out-of-range";
    case NVM_V2_ERR_SECTION_OVERLAP:  return "section-overlap";
    case NVM_V2_ERR_SECTION_DUPLICATE:return "duplicate-section";
    case NVM_V2_ERR_CHECKSUM:         return "checksum-mismatch";
    case NVM_V2_ERR_INDEX_RANGE:      return "index-out-of-range";
    }
    return "unknown";
}

NvmV2Result nvm_v2_read_header(const uint8_t *data, size_t size,
                               NvmV2Header *out) {
    if (!data || !out) return NVM_V2_ERR_TRUNCATED;
    if (size < NVM_V2_HEADER_SIZE) return NVM_V2_ERR_TRUNCATED;

    NvmV2Header h;
    for (int i = 0; i < 4; i++) h.magic[i] = data[OFF_MAGIC + i];
    h.format_version = rd16(data + OFF_FORMAT_VERSION);
    h.isa_version    = rd16(data + OFF_ISA_VERSION);
    h.feature_bits   = rd32(data + OFF_FEATURE_BITS);
    h.total_size     = rd64(data + OFF_TOTAL_SIZE);
    h.header_size    = rd32(data + OFF_HEADER_SIZE);
    h.section_count  = rd32(data + OFF_SECTION_COUNT);
    h.entry_point    = rd32(data + OFF_ENTRY_POINT);
    h.flags          = rd32(data + OFF_FLAGS);
    h.checksum       = rd32(data + OFF_CHECKSUM);

    if (h.magic[0] != NVM_V2_MAGIC_0 || h.magic[1] != NVM_V2_MAGIC_1 ||
        h.magic[2] != NVM_V2_MAGIC_2 || h.magic[3] != NVM_V2_MAGIC_3)
        return NVM_V2_ERR_MAGIC;

    if (h.format_version != NVM_V2_FORMAT_VERSION)
        return NVM_V2_ERR_FORMAT_VERSION;

    /* header_size is exact rather than a minimum: a reader that accepted a
     * larger one would be claiming to understand a header it has not seen. */
    if (h.header_size != NVM_V2_HEADER_SIZE)
        return NVM_V2_ERR_HEADER_SIZE;

    /* Fail closed on capabilities we do not implement. */
    if (h.feature_bits & ~(uint32_t)NVM_V2_FEATURE_KNOWN_MASK)
        return NVM_V2_ERR_UNKNOWN_FEATURE;

    if (h.total_size != (uint64_t)size)
        return NVM_V2_ERR_TOTAL_SIZE;

    if (h.flags != 0)
        return NVM_V2_ERR_RESERVED_FLAGS;

    *out = h;
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_read_section(const uint8_t *data, size_t size,
                                const NvmV2Header *header, uint32_t index,
                                NvmV2SectionEntry *out) {
    if (!data || !header || !out) return NVM_V2_ERR_TRUNCATED;
    if (index >= header->section_count) return NVM_V2_ERR_TRUNCATED;

    /* The directory must fit between the header and the end of the file.
     * Computed in 64-bit and compared by subtraction so a large section_count
     * cannot wrap the multiplication into a small number. */
    uint64_t dir_bytes = (uint64_t)header->section_count
                       * (uint64_t)NVM_V2_SECTION_ENTRY_SIZE;
    if (header->section_count != 0 &&
        dir_bytes / header->section_count != NVM_V2_SECTION_ENTRY_SIZE)
        return NVM_V2_ERR_TRUNCATED;
    if ((uint64_t)size < NVM_V2_HEADER_SIZE) return NVM_V2_ERR_TRUNCATED;
    if (dir_bytes > (uint64_t)size - NVM_V2_HEADER_SIZE)
        return NVM_V2_ERR_TRUNCATED;

    const uint8_t *p = data + NVM_V2_HEADER_SIZE
                     + (size_t)index * NVM_V2_SECTION_ENTRY_SIZE;
    out->type   = rd32(p + SEC_OFF_TYPE);
    out->flags  = rd32(p + SEC_OFF_FLAGS);
    out->offset = rd64(p + SEC_OFF_OFFSET);
    out->size   = rd64(p + SEC_OFF_SIZE);
    return NVM_V2_OK;
}

NvmV2Result nvm_v2_validate(const uint8_t *data, size_t size) {
    NvmV2Header h;
    NvmV2Result r = nvm_v2_read_header(data, size, &h);
    if (r != NVM_V2_OK) return r;

    uint64_t dir_bytes = (uint64_t)h.section_count
                       * (uint64_t)NVM_V2_SECTION_ENTRY_SIZE;
    if (h.section_count != 0 &&
        dir_bytes / h.section_count != NVM_V2_SECTION_ENTRY_SIZE)
        return NVM_V2_ERR_TRUNCATED;
    if (dir_bytes > h.total_size - NVM_V2_HEADER_SIZE)
        return NVM_V2_ERR_TRUNCATED;

    /* Nothing may land on the header or the directory that describes it. */
    const uint64_t payload_floor = (uint64_t)NVM_V2_HEADER_SIZE + dir_bytes;

    uint64_t seen_types = 0;   /* bit per section type; all types are singletons */

    for (uint32_t i = 0; i < h.section_count; i++) {
        NvmV2SectionEntry e;
        r = nvm_v2_read_section(data, size, &h, i, &e);
        if (r != NVM_V2_OK) return r;

        if (e.flags != 0) return NVM_V2_ERR_RESERVED_FLAGS;

        if (e.type < NVM_V2_SECTION_TYPE_MIN || e.type > NVM_V2_SECTION_TYPE_MAX)
            return NVM_V2_ERR_SECTION_TYPE;

        uint64_t bit = (uint64_t)1 << e.type;
        if (seen_types & bit) return NVM_V2_ERR_SECTION_DUPLICATE;
        seen_types |= bit;

        /* Subtraction form throughout: offset is bounded first, so the size
         * comparison cannot overflow. */
        if (e.offset > h.total_size) return NVM_V2_ERR_SECTION_RANGE;
        if (e.size > h.total_size - e.offset) return NVM_V2_ERR_SECTION_RANGE;
        if (e.size != 0 && e.offset < payload_floor)
            return NVM_V2_ERR_SECTION_OVERLAP;
    }

    /* Pairwise overlap. section_count is small and bounded by the file size,
     * so the quadratic scan is not worth an allocation to avoid. */
    for (uint32_t i = 0; i < h.section_count; i++) {
        NvmV2SectionEntry a;
        if (nvm_v2_read_section(data, size, &h, i, &a) != NVM_V2_OK)
            return NVM_V2_ERR_TRUNCATED;
        if (a.size == 0) continue;
        for (uint32_t j = i + 1; j < h.section_count; j++) {
            NvmV2SectionEntry b;
            if (nvm_v2_read_section(data, size, &h, j, &b) != NVM_V2_OK)
                return NVM_V2_ERR_TRUNCATED;
            if (b.size == 0) continue;
            /* Disjoint iff one ends at or before the other begins. */
            bool disjoint = (a.offset >= b.offset && a.offset - b.offset >= b.size)
                         || (b.offset >= a.offset && b.offset - a.offset >= a.size);
            if (!disjoint) return NVM_V2_ERR_SECTION_OVERLAP;
        }
    }

    /* Checksum last: a structurally broken directory should report what is
     * wrong with it, not merely that some byte changed. The structural error
     * is the one a person can act on. */
    uint32_t actual = nvm_crc32(data + NVM_V2_HEADER_SIZE,
                                (uint32_t)(h.total_size - NVM_V2_HEADER_SIZE));
    if (actual != h.checksum) return NVM_V2_ERR_CHECKSUM;

    return NVM_V2_OK;
}

void nvm_v2_write_header(uint8_t *out, const NvmV2Header *header) {
    if (!out || !header) return;
    for (int i = 0; i < 4; i++) out[OFF_MAGIC + i] = header->magic[i];
    wr16(out + OFF_FORMAT_VERSION, header->format_version);
    wr16(out + OFF_ISA_VERSION,    header->isa_version);
    wr32(out + OFF_FEATURE_BITS,   header->feature_bits);
    wr64(out + OFF_TOTAL_SIZE,     header->total_size);
    wr32(out + OFF_HEADER_SIZE,    header->header_size);
    wr32(out + OFF_SECTION_COUNT,  header->section_count);
    wr32(out + OFF_ENTRY_POINT,    header->entry_point);
    wr32(out + OFF_FLAGS,          header->flags);
    wr32(out + OFF_CHECKSUM,       header->checksum);
}

void nvm_v2_write_section(uint8_t *out, const NvmV2SectionEntry *entry) {
    if (!out || !entry) return;
    wr32(out + SEC_OFF_TYPE,   entry->type);
    wr32(out + SEC_OFF_FLAGS,  entry->flags);
    wr64(out + SEC_OFF_OFFSET, entry->offset);
    wr64(out + SEC_OFF_SIZE,   entry->size);
}
