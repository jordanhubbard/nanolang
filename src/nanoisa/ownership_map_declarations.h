#ifndef NANOISA_OWNERSHIP_MAP_DECLARATIONS_H
#define NANOISA_OWNERSHIP_MAP_DECLARATIONS_H
#include "nvm_format.h"
/* I own copied numeric declarations, never executable authority. Unsuccessful
 * preparation/getters preserve output. Immutable getters allocate nothing;
 * cumulative proof callers precharge these worst-case query costs. */
#define NVM_MAP_FIELD_QUERY_WORK 20u
#define NVM_MAP_SIMPLE_QUERY_WORK 1u
typedef struct NvmMapDeclarationPlan NvmMapDeclarationPlan;
typedef enum { NVM_MAP_ABSENT, NVM_MAP_DESCRIBED, NVM_MAP_INVALID,
               NVM_MAP_LIMIT, NVM_MAP_MEMORY } NvmMapDeclarationStatus;
typedef struct { NvmMapDeclarationStatus status; const char *message; } NvmMapDeclarationResult;
typedef struct { uint32_t scalars,pairs,bindings; } NvmMapDeclarationCounts;
typedef struct { uint32_t key_type,value_type; } NvmMapDeclarationPair;
typedef struct { uint32_t layout; uint16_t field; uint32_t pair; } NvmMapDeclarationBinding;
NvmMapDeclarationResult nvm_prepare_map_declarations(const NvmModule *,NvmMapDeclarationPlan **);
void nvm_map_declarations_free(NvmMapDeclarationPlan *);
bool nvm_map_declarations_counts(const NvmMapDeclarationPlan *,NvmMapDeclarationCounts *);
bool nvm_map_declarations_scalar(const NvmMapDeclarationPlan *,uint32_t,uint8_t *);
bool nvm_map_declarations_pair(const NvmMapDeclarationPlan *,uint32_t,NvmMapDeclarationPair *);
bool nvm_map_declarations_binding(const NvmMapDeclarationPlan *,uint32_t,NvmMapDeclarationBinding *);
bool nvm_map_declarations_field(const NvmMapDeclarationPlan *,uint32_t,uint16_t,NvmMapDeclarationPair *);
#endif
