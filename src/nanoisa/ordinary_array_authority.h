#ifndef NANOISA_ORDINARY_ARRAY_AUTHORITY_H
#define NANOISA_ORDINARY_ARRAY_AUTHORITY_H
#include "nvm_format.h"
/* I describe copied declaration facts only. No executable admission follows.
 * Input remains valid/immutable during the call; output storage is disjoint.
 * Every unsuccessful query/getter preserves caller output. */
typedef struct NvmOrdinaryArrayAuthority NvmOrdinaryArrayAuthority;
typedef enum { NVM_OAA_DESCRIBED, NVM_OAA_UNKNOWN, NVM_OAA_INVALID,
               NVM_OAA_LIMIT, NVM_OAA_MEMORY } NvmOrdinaryArrayStatus;
typedef struct { NvmOrdinaryArrayStatus status; uint32_t layout, field;
                 const char *message; } NvmOrdinaryArrayResult;
typedef struct { uint32_t layouts, types, bindings; } NvmOrdinaryArrayCounts;
typedef struct { uint8_t tag; uint32_t referent; } NvmOrdinaryArrayType;
typedef struct { uint32_t layout; uint16_t field; uint32_t element_type; } NvmOrdinaryArrayBinding;
NvmOrdinaryArrayResult nvm_describe_ordinary_array_authority(const NvmModule *, NvmOrdinaryArrayAuthority **);
void nvm_ordinary_array_authority_free(NvmOrdinaryArrayAuthority *);
bool nvm_ordinary_array_authority_counts(const NvmOrdinaryArrayAuthority *, NvmOrdinaryArrayCounts *);
bool nvm_ordinary_array_authority_type(const NvmOrdinaryArrayAuthority *, uint32_t, NvmOrdinaryArrayType *);
bool nvm_ordinary_array_authority_binding(const NvmOrdinaryArrayAuthority *, uint32_t, NvmOrdinaryArrayBinding *);
#endif
