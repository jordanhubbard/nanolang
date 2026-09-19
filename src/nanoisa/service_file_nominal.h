#ifndef NANOISA_SERVICE_FILE_NOMINAL_H
#define NANOISA_SERVICE_FILE_NOMINAL_H
#include "service_bindings.h"
#include "nvm_format.h"

/* Private v2 only: no change to v1 codec or module/execution admission. */
#define NVM_FILE_NOMINAL_VERSION 2u
#define NVM_FILE_NOMINAL_TYPES 8u
#define NVM_FILE_NOMINAL_BYTES 120u
#define NVM_FILE_NOMINAL_MAX_LAYOUTS 65536u

typedef struct {
    uint32_t imports[NVM_SERVICE_BINDING_COUNT];
    uint32_t layouts[NVM_FILE_NOMINAL_TYPES];
} NvmFileNominalBindings;
NvmServiceResult nvm_file_nominal_check(const NvmFileNominalBindings *);
/* Failure preserves output; buffers may overlap; no input is retained. */
NvmServiceResult nvm_file_nominal_decode(const uint8_t *,size_t,NvmFileNominalBindings *);
/* Null bytes selects checked size-only mode. Size storage must be disjoint
 * from input and bytes; input/bytes may overlap. Failure preserves both outputs. */
NvmServiceResult nvm_file_nominal_encode(const NvmFileNominalBindings *,uint8_t *,size_t,size_t *);

typedef enum {
    NVM_FILE_NOMINAL_DESCRIBED, NVM_FILE_NOMINAL_INVALID,
    NVM_FILE_NOMINAL_LIMIT, NVM_FILE_NOMINAL_MEMORY
} NvmFileNominalStatus;
typedef enum {
    NVM_FILE_CATEGORY_UNKNOWN, NVM_FILE_CATEGORY_FILE,
    NVM_FILE_CATEGORY_OPEN_RESULT, NVM_FILE_CATEGORY_SCALAR_RESULT,
    NVM_FILE_CATEGORY_RECORD
} NvmFileCategory;
typedef struct {
    uint32_t global_index, catalog_ordinal, source_ordinal;
    uint8_t layout_kind, ownership_flags;
    NvmFileCategory category;
} NvmFileNominalLayout;
typedef struct NvmFileNominalPlan NvmFileNominalPlan;
/* Caller supplies valid immutable in-memory module arrays for this call.
 * I check complete v2 service, layout, import and ownership-v1 declarations,
 * not code or host authority. Success publishes an independently owned map;
 * failure preserves *out. Shared ownership validation is not relaxed. */
NvmFileNominalStatus nvm_file_nominal_plan(const NvmModule *,NvmFileNominalPlan **);
void nvm_file_nominal_plan_free(NvmFileNominalPlan *);
uint32_t nvm_file_nominal_layout_count(const NvmFileNominalPlan *);
/* Failure preserves output. UNKNOWN rows have no catalog/runtime authority. */
bool nvm_file_nominal_layout(const NvmFileNominalPlan *,uint32_t,NvmFileNominalLayout *);
bool nvm_file_nominal_type(const NvmFileNominalPlan *,uint32_t,NvmFileNominalLayout *);
bool nvm_file_nominal_source(const NvmFileNominalPlan *,uint8_t,uint32_t,NvmFileNominalLayout *);
bool nvm_file_nominal_import(const NvmFileNominalPlan *,uint32_t,uint32_t *);
#endif
