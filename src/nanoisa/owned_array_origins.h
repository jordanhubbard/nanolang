#ifndef NANOISA_OWNED_ARRAY_ORIGINS_H
#define NANOISA_OWNED_ARRAY_ORIGINS_H
#include "owned_array_layouts.h"
#define NVM_OWNER_ORIGIN_WORDS 5u
#define NVM_OWNER_ORIGIN_FORMALS 256u
/* Private origin facts only; I grant no scalar/affine/runtime authority. */
typedef struct NvmOwnedArrayOrigins NvmOwnedArrayOrigins;
typedef enum { NVM_OWNER_ORIGIN_PROVED, NVM_OWNER_ORIGIN_UNRESOLVED,
    NVM_OWNER_ORIGIN_INVALID, NVM_OWNER_ORIGIN_LIMIT, NVM_OWNER_ORIGIN_MEMORY
} NvmOwnerOriginStatus;
typedef struct { NvmOwnerOriginStatus status; uint32_t function, pc; const char *message; } NvmOwnerOriginResult;
typedef struct { uint64_t words[NVM_OWNER_ORIGIN_WORDS]; } NvmOwnerOriginSet;
typedef struct { uint32_t function, pc; uint8_t opcode; } NvmOwnerOriginSite;
typedef struct { uint16_t ordinal; NvmOwnedArrayLeafPath path; NvmOwnerOriginSet origins; } NvmOwnerOriginPath;
typedef struct {
    uint32_t formal_count, result_count;
    bool reachable, conditional;
    NvmOwnerOriginSet required;
} NvmOwnerOriginSummary;
typedef struct { uint32_t functions, sites, obligations; } NvmOwnerOriginCounts;
typedef struct {
    uint32_t function, pc;
    uint16_t actual_tags, required_tags, read_tags;
} NvmOwnerOriginObligation;
/* All failures preserve outputs. I borrow immutable module bytes during this
 * call only. Helper summaries use function-local formals; entry is closed. */
NvmOwnerOriginResult nvm_analyze_owned_array_origins(const NvmModule *, NvmOwnedArrayOrigins **);
void nvm_owned_array_origins_free(NvmOwnedArrayOrigins *);
bool nvm_owned_array_origin_counts(const NvmOwnedArrayOrigins *, NvmOwnerOriginCounts *);
bool nvm_owned_array_origin_site(const NvmOwnedArrayOrigins *, uint32_t, NvmOwnerOriginSite *);
bool nvm_owned_array_origin_summary(const NvmOwnedArrayOrigins *, uint32_t, NvmOwnerOriginSummary *);
bool nvm_owned_array_origin_input(const NvmOwnedArrayOrigins *, uint32_t, uint32_t, NvmOwnerOriginPath *);
bool nvm_owned_array_origin_result(const NvmOwnedArrayOrigins *, uint32_t, uint32_t, NvmOwnerOriginPath *);
bool nvm_owned_array_origin_obligation(const NvmOwnedArrayOrigins *, uint32_t, NvmOwnerOriginObligation *);
#endif
