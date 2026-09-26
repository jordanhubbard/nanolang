/* I transport exact SDK declarations; this codec grants no execution authority. */
#ifndef NANOISA_SDK_PROVIDER_CODEC_H
#define NANOISA_SDK_PROVIDER_CODEC_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "preparation_budget.h"
#define NVM_SDK_PROVIDER_REVISION 1u
#define NVM_SDK_PROVIDER_HEADER_BYTES 32u
#define NVM_SDK_PROVIDER_MAX_ROWS 4096u
#define NVM_SDK_PROVIDER_MAX_REFERENCES 65536u
#define NVM_SDK_PROVIDER_MAX_BINDINGS 65536u
#define NVM_SDK_PROVIDER_MAX_BYTES ((size_t)16u * 1024u * 1024u)
#define NVM_SDK_PROVIDER_NO_INDEX UINT32_MAX
/* Indices refer to my existing module strings/layouts/signatures and shared
 * ARRAY_FIELDS type pool. I do not introduce a second recursive type system. */
enum { NVM_SDK_NOMINAL_RECORD, NVM_SDK_NOMINAL_UNION, NVM_SDK_NOMINAL_OPAQUE, NVM_SDK_NOMINAL_ENUM };
enum { NVM_SDK_BIND_IMPORT, NVM_SDK_BIND_FUNCTION, NVM_SDK_BIND_FIELD };
typedef struct {
    uint32_t owner, name, kind, layout, argument_first, argument_count;
} NvmSdkNominalRow;
typedef struct {
    uint32_t module, abi, target, artifact_digest, generation_digest, library;
} NvmSdkProviderRow;
typedef struct {
    uint32_t coarse_signature, parameter_first, parameter_count, result_first, result_count;
} NvmSdkSignatureRow;
typedef struct {
    uint32_t kind, subject, slot, detail, provider;
} NvmSdkBindingRow;
typedef struct {
    const NvmSdkNominalRow *nominals;
    const NvmSdkProviderRow *providers;
    const NvmSdkSignatureRow *signatures;
    const NvmSdkBindingRow *bindings;
    const uint32_t *references;
    uint32_t nominal_count, provider_count, signature_count, binding_count, reference_count;
} NvmSdkProviderRows;
typedef enum { NVM_SDK_OK, NVM_SDK_INVALID, NVM_SDK_LIMIT, NVM_SDK_MEMORY } NvmSdkResult;
/* I own immutable canonical bytes. Accessors decode by value, never exposing
 * mutable interior pointers. Free only plans returned successfully by decode. */
typedef struct NvmSdkProviderTransport NvmSdkProviderTransport;
/* I expose revision2 only through explicit private lifetime transport entries.
 * These rows describe policies, not validated ownership or executable authority. */
enum { NVM_SDK_VALUE, NVM_SDK_BORROW_CALL, NVM_SDK_BORROW_MUTABLE_CALL,
       NVM_SDK_SNAPSHOT_RESULT, NVM_SDK_BORROW_ARGUMENT_RESULT,
       NVM_SDK_OPAQUE_PIN, NVM_SDK_OPAQUE_PROVIDER_LIFETIME,
       NVM_SDK_CALLBACK_CALL, NVM_SDK_CALLBACK_RETAINED };
typedef struct {
    uint32_t parameter_first, parameter_count, result_first, result_count, execution;
} NvmSdkCallPolicy;
typedef struct {
    uint32_t type, mode, owner_argument, hook_set, child_first, child_count, callback_profile;
} NvmSdkLifetimeNode;
typedef struct {
    NvmSdkProviderRows declarations;
    const uint32_t *binding_policies; /* one per binding; nonimport entries zero */
    const NvmSdkCallPolicy *policies;
    const NvmSdkLifetimeNode *nodes;
    uint32_t policy_count, node_count;
} NvmSdkLifetimeRows;
NvmSdkResult nvm_sdk_provider_lifetime_decode_budget(const uint8_t *,size_t,
    NvmPreparationBudget *,NvmSdkProviderTransport **);
NvmSdkResult nvm_sdk_provider_lifetime_encode(const NvmSdkLifetimeRows *,size_t,
    uint8_t **,size_t *);
bool nvm_sdk_provider_lifetime_counts(const NvmSdkProviderTransport *,uint32_t counts[2]);
bool nvm_sdk_provider_call_policy(const NvmSdkProviderTransport *,uint32_t,NvmSdkCallPolicy *);
bool nvm_sdk_provider_lifetime_node(const NvmSdkProviderTransport *,uint32_t,NvmSdkLifetimeNode *);
bool nvm_sdk_provider_binding_policy(const NvmSdkProviderTransport *,uint32_t,uint32_t *);
/* Failure preserves *out. Success transfers a new plan; *out must not already
 * own a live plan. limit includes plan and payload, bounded by MAX_BYTES. */
NvmSdkResult nvm_sdk_provider_decode(const uint8_t *, size_t, size_t,
                                    NvmSdkProviderTransport **out);
/* I charge the same generation budget, publishing it only with success. */
NvmSdkResult nvm_sdk_provider_decode_budget(const uint8_t *,size_t,
    NvmPreparationBudget *,NvmSdkProviderTransport **);
void nvm_sdk_provider_transport_free(NvmSdkProviderTransport *);
/* I copy borrowed row arrays into malloc-owned bytes. Failure preserves both
 * outputs. Outputs must not alias inputs or each other. Raw success validates
 * wire shape only; module indices/ABI/graph/visibility require a separate plan. */
NvmSdkResult nvm_sdk_provider_encode(const NvmSdkProviderRows *, size_t,
                                    uint8_t **data, size_t *size);
bool nvm_sdk_provider_counts(const NvmSdkProviderTransport *, uint32_t counts[5]);
bool nvm_sdk_provider_nominal(const NvmSdkProviderTransport *, uint32_t, NvmSdkNominalRow *);
bool nvm_sdk_provider_requirement(const NvmSdkProviderTransport *, uint32_t, NvmSdkProviderRow *);
bool nvm_sdk_provider_signature(const NvmSdkProviderTransport *, uint32_t, NvmSdkSignatureRow *);
bool nvm_sdk_provider_binding(const NvmSdkProviderTransport *, uint32_t, NvmSdkBindingRow *);
bool nvm_sdk_provider_reference(const NvmSdkProviderTransport *, uint32_t, uint32_t *);
#endif
