#ifndef NANOISA_SERVICE_CLASSIFICATION_PRIVATE_H
#define NANOISA_SERVICE_CLASSIFICATION_PRIVATE_H
#include "service_bindings_module.h"
#include "owned_array_admission.h"
/* I bind one service query to one read-only synchronous classification call
 * tree. I am not an admission proof and must not survive into dispatch. */
typedef struct {
    const NvmModule *module;
    bool pending;
} NvmServiceClassification;
NvmServiceClassification nvm_service_classify(const NvmModule *module);
/* NULL or a different module always performs a fresh service query. */
bool nvm_service_pending_classified(const NvmModule *module,
                                    const NvmServiceClassification *facts);
NvmOwnedArrayRoute nvm_owned_array_route_classified(
    const NvmModule *module, const NvmServiceClassification *facts);
bool nvm_mixed_samples_candidate_classified(
    const NvmModule *module, const NvmServiceClassification *facts);
#endif
