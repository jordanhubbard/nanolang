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

#define NVM_FORMAT_VERSION 2

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
    uint8_t result_tag;      /* NanoValueTag when result_count is nonzero */
    uint8_t result_count;    /* Number of homogeneous results */
} NvmFunctionEntry;

#define NVM_FUNCTION_ENTRY_SIZE 20

/* ========================================================================
 * String Pool Entry (serialized in STRINGS section)
 * Format: [length: u32] [utf8 bytes: length]
 * ======================================================================== */

/* No struct needed - variable length, read with helpers */

/* ========================================================================
 * Debug Info Entry (serialized in DEBUG section)
 * Format: [bytecode_offset: u32] [source_line: u32] [source_col: u32]
 * ======================================================================== */

typedef struct {
    uint32_t bytecode_offset;
    uint32_t source_line;
    uint32_t source_col;    /* 1-based column, 0 = unknown */
} NvmDebugEntry;

#define NVM_DEBUG_ENTRY_SIZE 12

/* ========================================================================
 * Foreign-call argument limit
 *
 * A single limit is shared by every path that crosses the native boundary:
 * imported (extern) calls, the OP_CALL_EXTERN trap that carries their
 * arguments, the direct in-process FFI dispatch, and the co-process
 * (out-of-process) FFI dispatch. Keeping one constant guarantees that a
 * module accepted by the verifier can actually be marshaled and dispatched
 * by every backend rather than being silently truncated by whichever path
 * happens to have the smallest hand-rolled array.
 *
 * Calls above the small dispatch-table range use typed libffi dispatch.
 * ======================================================================== */

#define NANO_MAX_FFI_ARGS 16

#define NVM_CALLBACK_ABI_RETAINED_V1 1u
#define NVM_CALLBACK_NO_PARAMETER UINT16_MAX
typedef enum {
    NVM_FOREIGN_OWNER_THREAD = 0,
    NVM_FOREIGN_WORKER_THREAD = 1
} NvmForeignExecution;

/* I store callback shapes inline in my execution module; the v2 wire form
 * references SIGNATURES. A NO_PARAMETER record specifies a wait/release
 * adapter's execution policy without claiming a callback argument. */
typedef struct {
    uint32_t import_idx;
    uint32_t adapter_name_idx;
    uint16_t parameter_idx;
    uint8_t abi_version;
    uint8_t execution;
    uint16_t param_count;
    uint8_t return_tag;
    uint8_t param_tags[NANO_MAX_FFI_ARGS];
} NvmCallbackContract;

bool nvm_callback_shape_valid(const uint8_t *tags, uint16_t count, uint8_t result);

/* ========================================================================
 * Import Entry (serialized in IMPORTS section)
 * ======================================================================== */

typedef enum {
    NVM_IMPORT_FFI = 0,
    NVM_IMPORT_COPROCESS = 1,
    NVM_IMPORT_ARTIFACT = 2
} NvmImportKind;

typedef struct {
    uint32_t module_name_idx;   /* Logical module, or absolute artifact path */
    uint32_t function_name_idx; /* String pool index */
    uint16_t param_count;
    uint8_t  return_type;       /* NanoValueTag */
    uint8_t  kind;              /* NvmImportKind; nonzero requires v2 wire format */
    /* Followed by param_count bytes of param type tags */
} NvmImportEntry;

#define NVM_IMPORT_ENTRY_BASE_SIZE 11

/* ========================================================================
 * Typed Call Descriptor (runtime-only, not serialized)
 *
 * Imports are resolved once — the module name is loaded, the function symbol
 * is looked up through the shared FFI loader, and the typed signature is
 * precomputed — then cached here keyed by import index. Subsequent FFI calls
 * reuse the cached descriptor instead of re-resolving the symbol and
 * recomputing the signature classification on every invocation.
 * ======================================================================== */

typedef enum {
    NVM_CALL_UNRESOLVED = 0, /* not yet resolved */
    NVM_CALL_RESOLVED,       /* func_ptr valid, ready to dispatch */
    NVM_CALL_FAILED          /* resolution attempted and failed */
} NvmCallResolution;

typedef struct {
    NvmCallResolution state;  /* resolution status of this import */
    void *func_ptr;           /* resolved native function pointer */
    const char *func_name;    /* interned function name (module string pool) */
    const char *module_name;  /* interned module name (module string pool) */
    const uint8_t *param_types; /* param type tags, or NULL */
    uint16_t param_count;     /* declared parameter count */
    uint8_t return_type;      /* NanoValueTag of the return value */
    bool all_float;           /* true when return + all params are TAG_FLOAT */
    void (*string_release)(const char *); /* optional provider cleanup, after copying */
} NvmCallDescriptor;

/* ========================================================================
 * In-Memory NVM Module
 * Used as the intermediate representation for building and loading.
 * ======================================================================== */

/* I preserve duplicate advisory keys in their declared order. */
typedef struct { uint32_t key_idx, value_idx; } NvmMetadataEntry;

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
    uint8_t **function_param_types; /* Owned tags; NULL entry means unknown. */
    uint32_t function_count;
    uint32_t function_capacity;

    /* Code section (raw bytecode) */
    uint8_t *code;
    uint32_t code_size;
    uint32_t code_capacity;

    /* Owned versioned function-mode/root ownership declarations. */
    uint8_t *ownership_data;
    uint32_t ownership_size;

    /* Owned canonical v2 LAYOUTS bytes; indices and names remain exact. */
    uint8_t *layout_data;
    uint32_t layout_size;

    /* Owned canonical passive-eligibility payload; absent means no claim. */
    uint8_t *passive_data;
    uint32_t passive_size;

    NvmMetadataEntry *metadata; /* Owned entries, indices borrow my string pool. */
    uint32_t metadata_count;
    uint32_t metadata_capacity;

    /* Debug info */
    NvmDebugEntry *debug_entries;
    uint32_t debug_count;
    uint32_t debug_capacity;

    /* Source file: string pool index of the source filename (0 = unknown) */
    uint32_t source_file_idx;

    /* Import table */
    NvmImportEntry *imports;
    uint8_t **import_param_types; /* param type arrays, one per import */
    NvmCallbackContract *callback_contracts;
    uint32_t callback_contract_count;
    uint32_t callback_contract_capacity;
    uint32_t import_count;
    uint32_t import_capacity;

    /* Ordered separately linked module dependencies */
    NvmModuleRefEntry *module_refs;
    uint32_t module_ref_count;
    uint32_t module_ref_capacity;    /* Resolved typed call descriptors — runtime-only, lazily populated on the
     * first FFI call for each import; never serialized. NULL until allocated. */
    NvmCallDescriptor *call_descriptors;
    uint32_t call_descriptor_count;

    /* Type definition counts — populated by codegen, used by verifier */
    uint32_t struct_count;
    uint32_t enum_count;
    uint32_t union_count;
} NvmModule;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Create a new empty module */
NvmModule *nvm_module_new(void);

/* Free a module and all its data */
void nvm_module_free(NvmModule *mod);

/* I return a deduplicated string index, or UINT32_MAX on allocation/input
 * failure. Existing entries remain usable after failed growth. */
uint32_t nvm_add_string(NvmModule *mod, const char *str, uint32_t length);

/* I append advisory entries atomically; the last source key updates my view. */
bool nvm_add_metadata(NvmModule *mod, uint32_t key, uint32_t value);
bool nvm_metadata_valid(const NvmModule *mod);
bool nvm_metadata_source_key(const NvmModule *mod, uint32_t key);

/* Add a function entry. Returns the function index. */
/* I return UINT32_MAX on failure; entry may borrow an existing table entry. */
uint32_t nvm_add_function(NvmModule *mod, const NvmFunctionEntry *entry);
/* I copy exact-arity tags transactionally; failure leaves the old tags intact.
 * TAG_VOID explicitly means unknown, not a callback-compatible scalar. */
bool nvm_set_function_param_types(NvmModule *mod, uint32_t index,
                                  const uint8_t *tags, uint16_t count);
/* I append in (import_idx, parameter_idx) order and fail transactionally. */
bool nvm_add_callback_contract(NvmModule *mod, const NvmCallbackContract *contract);
bool nvm_callback_contracts_valid(const NvmModule *mod);

/* Append bytecode to the code section. Returns the byte offset where it was written. */
uint32_t nvm_append_code(NvmModule *mod, const uint8_t *code, uint32_t size);

/* Add a debug entry (bytecode offset -> source line + column).
 * source_col is 1-based; pass 0 if unknown. I return false without appending
 * if allocation or representable capacity is exhausted. */
bool nvm_add_debug_entry(NvmModule *mod, uint32_t bytecode_offset,
                         uint32_t source_line, uint32_t source_col);

/* Remove all debug info from a module for production builds.
 * Clears debug entries, source file index, and debug-info header flag. */
void nvm_strip_debug_info(NvmModule *mod);

/* Serialize module to a byte buffer. Caller must free returned buffer.
 * Sets *out_size to the total size. Returns NULL on error. */
uint8_t *nvm_serialize(const NvmModule *mod, uint32_t *out_size);

/* Deserialize a byte buffer into a module. Returns NULL on error. */
NvmModule *nvm_deserialize(const uint8_t *data, uint32_t size);

/* Validate an NVM header. Returns true if valid. */
bool nvm_validate_header(const NvmHeader *header);

/* Compute CRC32 over a byte range */
uint32_t nvm_crc32(const uint8_t *data, uint32_t size);

/* I return the import index, or UINT32_MAX on allocation failure; existing
 * entries and their parameter arrays remain usable after failed growth. */
uint32_t nvm_add_import(NvmModule *mod, uint32_t module_name_idx,
                        uint32_t function_name_idx, uint16_t param_count,
                        uint8_t return_type, const uint8_t *param_types);

/* Add an ordered separately linked module dependency. OP_CALL_MODULE uses the
 * returned index as its module operand. */
uint32_t nvm_add_module_ref(NvmModule *mod, uint32_t module_name_idx);

/* Get a string from the module by index. Returns NULL if out of range. */
const char *nvm_get_string(const NvmModule *mod, uint32_t index);
/* Stored byte length of a string constant. Callers that reconstruct string
 * values must use this instead of strlen() so embedded zero bytes survive.
 * Returns 0 for an out-of-range index. */
uint32_t nvm_get_string_len(const NvmModule *mod, uint32_t index);

/* Get the byte length of a string by index. Returns 0 if out of range.
 * Strings may contain embedded NUL bytes, so callers that need lossless
 * handling of binary strings must use this length rather than strlen(). */
uint32_t nvm_get_string_len(const NvmModule *mod, uint32_t index);

/* Find the latest function with this name. Returns UINT32_MAX if absent. */
uint32_t nvm_find_function(const NvmModule *mod, const char *name);

/* Release any cached typed call descriptors for a module. Safe to call when
 * no descriptors are allocated. After this returns, the next FFI call will
 * resolve imports from scratch again. */
void nvm_call_descriptors_reset(NvmModule *mod);

#endif /* NANOISA_NVM_FORMAT_H */
