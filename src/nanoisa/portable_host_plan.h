#ifndef NANOISA_PORTABLE_HOST_PLAN_H
#define NANOISA_PORTABLE_HOST_PLAN_H

#include "nvm_format.h"

#define NVM_PORTABLE_READ_NO_INDEX UINT32_MAX
#define NVM_PORTABLE_READ_MAX_IMPORTS 64u
#define NVM_PORTABLE_READ_MAX_FUNCTIONS 256u
#define NVM_PORTABLE_READ_MAX_INSTRUCTIONS 65536u
#define NVM_PORTABLE_READ_MAX_BYTES (16u * 1024u * 1024u)
#define NVM_PORTABLE_READ_SYMBOL_CAPACITY 32u

typedef struct NvmPortableReadPlan NvmPortableReadPlan;
typedef enum {
    NVM_PORTABLE_READ_PREPARED = 0,
    NVM_PORTABLE_READ_NOT_SELECTED,
    NVM_PORTABLE_READ_INVALID,
    NVM_PORTABLE_READ_UNSUPPORTED,
    NVM_PORTABLE_READ_LIMIT,
    NVM_PORTABLE_READ_MEMORY
} NvmPortableReadStatus;

typedef struct {
    NvmPortableReadStatus status;
    uint32_t function_index;
    uint32_t pc; /* Absolute CODE offset, or NO_INDEX. */
    uint32_t import_index;
    const char *message; /* Static storage; never borrows the input. */
} NvmPortableReadResult;

typedef struct {
    uint32_t entry_function;
    uint32_t function_count;
    uint32_t import_count;
    uint32_t instruction_count;
    size_t module_bytes;
    size_t allocation_bound;
} NvmPortableReadCounts;

typedef enum { NVM_PORTABLE_HOST_READ_TEXT = 1 } NvmPortableHostOperation;
typedef enum { NVM_PORTABLE_HOST_BORROW_ROOTED_ARGUMENT = 1 } NvmPortableHostArgument;
typedef enum { NVM_PORTABLE_HOST_COPY_MANAGED_RESULT = 1 } NvmPortableHostResult;
typedef struct {
    uint32_t import_index;
    uint32_t namespace_string_index;
    uint32_t symbol_string_index;
    NvmPortableHostOperation operation;
    uint32_t revision;
    /* Catalog obligations for a later adapter, not proved runtime behavior. */
    NvmPortableHostArgument argument_ownership;
    NvmPortableHostResult result_ownership;
    uint16_t parameter_count;
    uint8_t parameter_tag;
    uint8_t result_count;
    uint8_t result_tag;
    uint8_t import_kind;
    uint32_t namespace_length;
    uint32_t symbol_length;
    char namespace_bytes[1]; /* The exact empty namespace, NUL terminated. */
    char symbol_bytes[NVM_PORTABLE_READ_SYMBOL_CAPACITY];
} NvmPortableReadImport;

/* I describe declarations and the common structural/stack envelope only.
 * I grant no host, operand-type, lifetime, profile or execution authority.
 * The caller supplies a stable, readable in-memory module for this call.
 * Only PREPARED writes *out. The owned plan survives input destruction.
 * Shared verifier errors (including reported allocation failures) are INVALID;
 * its advisory type analysis supplies no proof fact. My own OOM is MEMORY.
 * NOT_SELECTED means a fully checked supported envelope has no imports.
 * Locations are NO_INDEX when a delegated error has no structured location. */
NvmPortableReadResult nvm_portable_read_plan(const NvmModule *module,
                                           NvmPortableReadPlan **out);
void nvm_portable_read_plan_free(NvmPortableReadPlan *plan);
/* I copy complete facts; false leaves the caller's output unchanged. */
bool nvm_portable_read_plan_counts(const NvmPortableReadPlan *plan,
                                  NvmPortableReadCounts *out);
bool nvm_portable_read_plan_import(const NvmPortableReadPlan *plan, uint32_t index,
                                  NvmPortableReadImport *out);

#endif
