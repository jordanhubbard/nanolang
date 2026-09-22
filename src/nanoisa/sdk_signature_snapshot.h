/* I retain existing coarse signature indices, not a second semantic type graph. */
#ifndef NANOISA_SDK_SIGNATURE_SNAPSHOT_H
#define NANOISA_SDK_SIGNATURE_SNAPSHOT_H
#include "sdk_provider_codec.h"
#include "nvm_v2_sections.h"
#define NVM_SDK_GENERATION_MAX_BYTES ((size_t)32u * 1024u * 1024u)
#define NVM_SDK_GENERATION_MAX_WORK 1048576u
typedef struct NvmSdkSignatureSnapshot NvmSdkSignatureSnapshot;
/* source and all its tables must remain readable/immutable during this call.
 * I copy every signature, including unused ones, and exact function/import/
 * callback/link selectors. Failure preserves *out, success transfers ownership.
 * This is structural transport, not complete module or execution validation. */
NvmSdkResult nvm_sdk_signature_snapshot_prepare(const NvmV2Module *source,
    size_t byte_limit, NvmSdkSignatureSnapshot **out);
void nvm_sdk_signature_snapshot_free(NvmSdkSignatureSnapshot *);
/* I expose contractually read-only owned rows, live until snapshot_free.
 * Nested pointer fields use existing mutable C types: callers must not mutate
 * them. This is not a deep-const or module-admission guarantee. No old-module alias
 * survives preparation. bytes reports all owned snapshot allocations. */
const NvmV2Signatures *nvm_sdk_signature_snapshot_rows(const NvmSdkSignatureSnapshot *);
size_t nvm_sdk_signature_snapshot_bytes(const NvmSdkSignatureSnapshot *);
enum { NVM_SDK_SIGNATURE_FUNCTION, NVM_SDK_SIGNATURE_IMPORT,
       NVM_SDK_SIGNATURE_CALLBACK, NVM_SDK_SIGNATURE_LINK };
bool nvm_sdk_signature_snapshot_index(const NvmSdkSignatureSnapshot *,
    unsigned kind, uint32_t subject, uint32_t *out);
#endif
