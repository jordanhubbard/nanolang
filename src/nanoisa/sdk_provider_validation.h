#ifndef NANOISA_SDK_PROVIDER_VALIDATION_H
#define NANOISA_SDK_PROVIDER_VALIDATION_H
#include "sdk_module_snapshot.h"
#include "ownership_declaration_projection.h"
/* I own a description generation only. No image is loaded and no runtime call
 * descriptor, old module or cache is modified. Every accessor is contractually
 * read-only, including nested pointers whose C types do not enforce const.
 * Provider manifest/image identity and logical lifetime hooks remain unresolved;
 * no description generation is executable. Success requires an unpublished
 * output slot; every refusal preserves both output and caller budget. */
typedef struct NvmSdkDescription NvmSdkDescription;
NvmSdkResult nvm_sdk_description_prepare(const NvmV2Module *,const uint8_t *,size_t,
    NvmPreparationBudget *,NvmSdkDescription **);
void nvm_sdk_description_free(NvmSdkDescription *);
const NvmV2Module *nvm_sdk_description_module(const NvmSdkDescription *);
/* I borrow these read-only opaque views for the description lifetime. I expose
 * the same retained graph and row readers, not reconstructed source visibility. */
const NvmOwnershipDeclarationPlan *nvm_sdk_description_declarations(const NvmSdkDescription *);
const NvmSdkProviderTransport *nvm_sdk_description_provider(const NvmSdkDescription *);
/* I compare complete shared type indices in an already validated description.
 * A successful unequal comparison publishes false and charges its actual work.
 * Invalid indices, allocation failure and exhausted budget preserve both output
 * and budget. This query issues no source-selection or executable authority. */
NvmSdkResult nvm_sdk_description_types_equal(const NvmSdkDescription *,uint32_t,uint32_t,
    NvmPreparationBudget *,bool *);
#endif
