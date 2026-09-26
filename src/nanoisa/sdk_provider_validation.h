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
#endif
