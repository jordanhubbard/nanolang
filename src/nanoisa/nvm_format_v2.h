/*
 * NanoISA v2 module container: header and section directory.
 *
 * Specified in docs/superpowers/specs/2026-09-01-nanoisa-v2-module-format.md.
 * This file covers only the container -- the header and the section directory
 * that locates everything else. Section payload encodings land separately.
 *
 * v2 exists alongside v1 (nvm_format.h) rather than replacing it in place. The
 * two are distinguished by magic[3], so a v1 reader handed a v2 module rejects
 * it rather than misreading it.
 *
 * All integers are little-endian on the wire, independent of host byte order.
 */

#ifndef NANOISA_NVM_FORMAT_V2_H
#define NANOISA_NVM_FORMAT_V2_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Magic: "NVM\x02". magic[3] is the format lineage; v1 uses 0x01. */
#define NVM_V2_MAGIC_0 'N'
#define NVM_V2_MAGIC_1 'V'
#define NVM_V2_MAGIC_2 'M'
#define NVM_V2_MAGIC_3 0x02

/* Layout version within the v2 lineage. Reset to 1: magic[3] already
 * distinguishes v2 from v1, so this counts v2's own revisions. */
#define NVM_V2_FORMAT_VERSION 1

/* Instruction-set version the code was assembled against. Versioned
 * separately from the layout: the container can change without the
 * semantics changing, and vice versa. */
#define NVM_V2_ISA_VERSION 2

#define NVM_V2_HEADER_SIZE         40
#define NVM_V2_SECTION_ENTRY_SIZE  24

/* Feature bits. A reader rejects a module setting a bit it does not know:
 * failing loudly at load beats failing subtly during execution. */
#define NVM_V2_FEATURE_LINKED    (1u << 0)  /* LINKS present and non-empty */
#define NVM_V2_FEATURE_FFI       (1u << 1)  /* IMPORTS non-empty */
#define NVM_V2_FEATURE_COPROCESS (1u << 2)  /* requires the co-process host */
#define NVM_V2_FEATURE_DEBUG     (1u << 3)  /* DEBUG section present */
#define NVM_V2_FEATURE_CLOSURES  (1u << 4)  /* constructs heap closures */
#define NVM_V2_FEATURE_KNOWN_MASK 0x0000001Fu

/* Section types. Renumbered from v1: the clean break makes v1 numbering
 * irrelevant, and reusing it would invite confusion between the two. */
typedef enum {
    NVM_V2_SECTION_METADATA   = 0x01,
    NVM_V2_SECTION_CONSTANTS  = 0x02,
    NVM_V2_SECTION_SIGNATURES = 0x03,
    NVM_V2_SECTION_LAYOUTS    = 0x04,
    NVM_V2_SECTION_FUNCTIONS  = 0x05,
    NVM_V2_SECTION_CODE       = 0x06,
    NVM_V2_SECTION_GLOBALS    = 0x07,
    NVM_V2_SECTION_IMPORTS    = 0x08,
    NVM_V2_SECTION_LINKS      = 0x09,
    NVM_V2_SECTION_DEBUG      = 0x0A
} NvmV2SectionType;

#define NVM_V2_SECTION_TYPE_MIN NVM_V2_SECTION_METADATA
#define NVM_V2_SECTION_TYPE_MAX NVM_V2_SECTION_DEBUG

/* No entry_point. 0xFFFFFFFF means the module has none. */
#define NVM_V2_NO_ENTRY_POINT 0xFFFFFFFFu

typedef struct {
    uint8_t  magic[4];
    uint16_t format_version;
    uint16_t isa_version;
    uint32_t feature_bits;
    uint64_t total_size;    /* byte length of the whole file */
    uint32_t header_size;   /* NVM_V2_HEADER_SIZE; lets the header grow */
    uint32_t section_count;
    uint32_t entry_point;   /* function index, or NVM_V2_NO_ENTRY_POINT */
    uint32_t flags;         /* reserved; must be zero */
    uint32_t checksum;      /* CRC32 of [header_size, total_size) */
} NvmV2Header;

typedef struct {
    uint32_t type;
    uint32_t flags;   /* reserved; must be zero */
    uint64_t offset;  /* from start of file */
    uint64_t size;
} NvmV2SectionEntry;

typedef enum {
    NVM_V2_OK = 0,
    NVM_V2_ERR_TRUNCATED,        /* buffer smaller than the header claims */
    NVM_V2_ERR_MAGIC,
    NVM_V2_ERR_FORMAT_VERSION,
    NVM_V2_ERR_HEADER_SIZE,
    NVM_V2_ERR_UNKNOWN_FEATURE,
    NVM_V2_ERR_TOTAL_SIZE,
    NVM_V2_ERR_RESERVED_FLAGS,
    NVM_V2_ERR_SECTION_TYPE,
    NVM_V2_ERR_SECTION_RANGE,    /* section escapes the file */
    NVM_V2_ERR_SECTION_OVERLAP,
    NVM_V2_ERR_SECTION_DUPLICATE,
    NVM_V2_ERR_CHECKSUM,
    NVM_V2_ERR_INDEX_RANGE,  /* an index into another table is out of range */
    NVM_V2_ERR_FEATURE_MISMATCH /* feature bits disagree with the sections present */
} NvmV2Result;

/* Human-readable name for a result, for diagnostics. Never NULL. */
const char *nvm_v2_result_name(NvmV2Result r);

/* Read and validate the fixed header. Does not look at the directory. */
NvmV2Result nvm_v2_read_header(const uint8_t *data, size_t size,
                               NvmV2Header *out);

/* Read one directory entry by index. The header must already have validated. */
NvmV2Result nvm_v2_read_section(const uint8_t *data, size_t size,
                                const NvmV2Header *header, uint32_t index,
                                NvmV2SectionEntry *out);

/* Validate the whole container: header, then every directory entry, then the
 * directory as a set (no duplicates, no overlaps, nothing escaping the file). */
NvmV2Result nvm_v2_validate(const uint8_t *data, size_t size);

/* Serialize a header. `out` must have room for NVM_V2_HEADER_SIZE bytes. */
void nvm_v2_write_header(uint8_t *out, const NvmV2Header *header);

/* Serialize a directory entry. `out` needs NVM_V2_SECTION_ENTRY_SIZE bytes. */
void nvm_v2_write_section(uint8_t *out, const NvmV2SectionEntry *entry);

#endif /* NANOISA_NVM_FORMAT_V2_H */
