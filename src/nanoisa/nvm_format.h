/*
 * NVM Binary Format - .nvm file format for NanoISA bytecode
 *
 * Layout:
 *   [Header: 32 bytes]
 *   [Section Directory: 12 bytes * section_count]
 *   [Section Data...]
 */

#ifndef NANOISA_NVM_FORMAT_H
#define NANOISA_NVM_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Magic bytes: "NVM\x01" */
#define NVM_MAGIC_0 'N'
#define NVM_MAGIC_1 'V'
#define NVM_MAGIC_2 'M'
#define NVM_MAGIC_3 0x01

#define NVM_FORMAT_VERSION 1

/* Header flags */
#define NVM_FLAG_HAS_MAIN    (1 << 0)
#define NVM_FLAG_NEEDS_EXTERN (1 << 1)
#define NVM_FLAG_DEBUG_INFO  (1 << 2)

/* Section types */
typedef enum {
    NVM_SECTION_CODE      = 0x0001,  /* Bytecode instructions */
    NVM_SECTION_STRINGS   = 0x0002,  /* String constant pool */
    NVM_SECTION_FUNCTIONS = 0x0003,  /* Function table */
    NVM_SECTION_STRUCTS   = 0x0004,  /* Struct definitions */
    NVM_SECTION_ENUMS     = 0x0005,  /* Enum definitions */
    NVM_SECTION_UNIONS    = 0x0006,  /* Union definitions */
    NVM_SECTION_GLOBALS   = 0x0007,  /* Global variable declarations */
    NVM_SECTION_IMPORTS   = 0x0008,  /* Extern function stubs */
    NVM_SECTION_DEBUG     = 0x0009,  /* Source maps */
    NVM_SECTION_METADATA  = 0x000A,  /* Module name, version */
    NVM_SECTION_MODULE_REFS = 0x000B /* Referenced module names for linking */
} NvmSectionType;

/* Module Reference Entry (serialized in MODULE_REFS section) */
typedef struct {
    uint32_t module_name_idx;  /* String pool index of referenced module name */
} NvmModuleRefEntry;

#define NVM_MODULE_REF_ENTRY_SIZE 4

/* ========================================================================
 * Header (32 bytes)
 * ======================================================================== */

#define NVM_HEADER_SIZE 32

typedef struct {
    uint8_t  magic[4];          /* "NVM\x01" */
    uint32_t format_version;    /* Format version */
    uint32_t flags;             /* Bitfield: has_main, needs_extern, debug_info */
    uint32_t entry_point;       /* Function table index of main */
    uint32_t section_count;     /* Number of sections */
    uint32_t string_pool_offset;/* Byte offset to string pool section */
    uint32_t string_pool_length;/* Byte length of string pool section */
    uint32_t checksum;          /* CRC32 of everything after the header */
} NvmHeader;

/* ========================================================================
 * Section Directory Entry (12 bytes each)
 * ======================================================================== */

#define NVM_SECTION_ENTRY_SIZE 12

typedef struct {
    uint32_t type;    /* NvmSectionType */
    uint32_t offset;  /* Byte offset from start of file */
    uint32_t size;    /* Byte size of section data */
} NvmSectionEntry;

/* ========================================================================
 * Function Table Entry (serialized in FUNCTIONS section)
 * ======================================================================== */

typedef struct {
    uint32_t name_idx;       /* String pool index for function name */
    uint16_t arity;          /* Number of parameters */
    uint32_t code_offset;    /* Byte offset into CODE section */
    uint32_t code_length;    /* Byte length of function's bytecode */
    uint16_t local_count;    /* Number of local variables (including params) */
    uint16_t upvalue_count;  /* Number of upvalue captures */
} NvmFunctionEntry;

#define NVM_FUNCTION_ENTRY_SIZE 18

/* ========================================================================
 * String Pool Entry (serialized in STRINGS section)
 * Format: [length: u32] [utf8 bytes: length]
 * ======================================================================== */

/* No struct needed - variable length, read with helpers */

/* ========================================================================
 * Debug Info Entry (serialized in DEBUG section)
 * Format: [bytecode_offset: u32] [source_line: u32]
 * ======================================================================== */

typedef struct {
    uint32_t bytecode_offset;
    uint32_t source_line;
} NvmDebugEntry;

#define NVM_DEBUG_ENTRY_SIZE 8

/* ========================================================================
 * Import Entry (serialized in IMPORTS section)
 * ======================================================================== */

typedef struct {
    uint32_t module_name_idx;   /* String pool index */
    uint32_t function_name_idx; /* String pool index */
    uint16_t param_count;
    uint8_t  return_type;       /* NanoValueTag */
    /* Followed by param_count bytes of param type tags */
} NvmImportEntry;

#define NVM_IMPORT_ENTRY_BASE_SIZE 11

/* ========================================================================
 * In-Memory NVM Module
 * Used as the intermediate representation for building and loading.
 * ======================================================================== */

#define NVM_MAX_SECTIONS   16
#define NVM_MAX_STRINGS   4096
#define NVM_MAX_FUNCTIONS  512

typedef struct {
    NvmHeader header;
    NvmSectionEntry sections[NVM_MAX_SECTIONS];
    uint32_t section_count;

    /* String pool */
    char **strings;
    uint32_t *string_lengths;
    uint32_t string_count;
    uint32_t string_capacity;

    /* Function table */
    NvmFunctionEntry *functions;
    uint32_t function_count;
    uint32_t function_capacity;

    /* Code section (raw bytecode) */
    uint8_t *code;
    uint32_t code_size;
    uint32_t code_capacity;

    /* Debug info */
    NvmDebugEntry *debug_entries;
    uint32_t debug_count;
    uint32_t debug_capacity;

    /* Import table */
    NvmImportEntry *imports;
    uint8_t **import_param_types; /* param type arrays, one per import */
    uint32_t import_count;
    uint32_t import_capacity;
} NvmModule;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Create a new empty module */
NvmModule *nvm_module_new(void);

/* Free a module and all its data */
void nvm_module_free(NvmModule *mod);

/* Add a string to the string pool. Returns the string index.
 * Deduplicates: returns existing index if string already present. */
uint32_t nvm_add_string(NvmModule *mod, const char *str, uint32_t length);

/* Add a function entry. Returns the function index. */
uint32_t nvm_add_function(NvmModule *mod, const NvmFunctionEntry *entry);

/* Append bytecode to the code section. Returns the byte offset where it was written. */
uint32_t nvm_append_code(NvmModule *mod, const uint8_t *code, uint32_t size);

/* Add a debug entry (bytecode offset -> source line) */
void nvm_add_debug_entry(NvmModule *mod, uint32_t bytecode_offset, uint32_t source_line);

/* Serialize module to a byte buffer. Caller must free returned buffer.
 * Sets *out_size to the total size. Returns NULL on error. */
uint8_t *nvm_serialize(const NvmModule *mod, uint32_t *out_size);

/* Deserialize a byte buffer into a module. Returns NULL on error. */
NvmModule *nvm_deserialize(const uint8_t *data, uint32_t size);

/* Validate an NVM header. Returns true if valid. */
bool nvm_validate_header(const NvmHeader *header);

/* Compute CRC32 over a byte range */
uint32_t nvm_crc32(const uint8_t *data, uint32_t size);

/* Add an import entry. Returns the import table index. */
uint32_t nvm_add_import(NvmModule *mod, uint32_t module_name_idx,
                        uint32_t function_name_idx, uint16_t param_count,
                        uint8_t return_type, const uint8_t *param_types);

/* Get a string from the module by index. Returns NULL if out of range. */
const char *nvm_get_string(const NvmModule *mod, uint32_t index);

#endif /* NANOISA_NVM_FORMAT_H */
