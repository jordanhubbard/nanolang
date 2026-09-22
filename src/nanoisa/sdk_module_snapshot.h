/* I own complete private module storage without granting execution authority. */
#ifndef NANOISA_SDK_MODULE_SNAPSHOT_H
#define NANOISA_SDK_MODULE_SNAPSHOT_H
#include "sdk_signature_snapshot.h"
typedef struct NvmSdkModuleSnapshot NvmSdkModuleSnapshot;
/* Source storage must remain readable and immutable throughout preparation.
 * Failure preserves *out and the source. Success transfers an independent
 * snapshot; *out must not already own storage. limit covers every allocation. */
NvmSdkResult nvm_sdk_module_snapshot_prepare(const NvmV2Module *, size_t limit,
                                           NvmSdkModuleSnapshot **out);
void nvm_sdk_module_snapshot_free(NvmSdkModuleSnapshot *);
/* Contractually read-only, including nested pointers. Never pass this view to
 * nvm_v2_module_free or mutate its tables. No old module/cache lifetime changes. */
const NvmV2Module *nvm_sdk_module_snapshot_view(const NvmSdkModuleSnapshot *);
size_t nvm_sdk_module_snapshot_bytes(const NvmSdkModuleSnapshot *);
#endif
