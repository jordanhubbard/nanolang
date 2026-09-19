/* I live inside verifier.c. No caller can request the internal structure
 * delegation or pass a prebuilt origin/lifetime certificate to preparation. */
#include "owned_array_authority_internal.h"
struct NvmOwnedArrayPlan { NvmOwnerLifetimeFacts *facts; };
void nvm_owned_array_plan_free(NvmOwnedArrayPlan *p) {
    if(!p)return;
    nvm_owned_array_lifetime_facts_free(p->facts);free(p);
}
NvmOwnerAuthorityResult nvm_prepare_owned_array_authority(const NvmModule *module,NvmOwnedArrayPlan **out) {
    if(!out)return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_INVALID,0,0,"I require an output for private owner preparation."};
    /* The fresh analysis checks service presence before every other profile. */
    NvmOwnerLifetimeFacts *facts=NULL;
    NvmOwnerAuthorityResult result=nvm_analyze_owned_array_lifetimes(module,&facts);
    if(result.status!=NVM_OWNER_AUTH_PREPARED)return result;
    /* This static path neither dispatches candidates nor calls public verifiers.
     * Only the already completed independent ownership analysis is delegated. */
    NvmVerifyResult structure=verify_structure_checked(module,true,NULL,true);
    if(!structure.ok || !nvm_metadata_valid(module)) {
        nvm_owned_array_lifetime_facts_free(facts);
        return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_INVALID,0,0,"I require complete common structure and metadata for owner preparation."};
    }
    NvmOwnedArrayPlan *p=calloc(1,sizeof *p);
    if(!p){nvm_owned_array_lifetime_facts_free(facts);return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_MEMORY,0,0,"I could not allocate the private owner plan."};}
    p->facts=facts;*out=p;
    return (NvmOwnerAuthorityResult){NVM_OWNER_AUTH_PREPARED,0,0,"I prepared private origin, lifetime, scalar and structural facts; I grant no runtime permission."};
}
bool nvm_owned_array_plan_counts(const NvmOwnedArrayPlan *p,NvmOwnerAuthorityCounts *out) {
    if(!p || !out)return false;
    *out=p->facts->counts;return true;
}
bool nvm_owned_array_plan_signature(const NvmOwnedArrayPlan *p,uint32_t f,NvmOwnerSignature *out) {
    if(!p || !out || f>=p->facts->counts.functions)return false;
    *out=p->facts->signatures[f];return true;
}
bool nvm_owned_array_plan_local(const NvmOwnedArrayPlan *p,uint32_t f,uint16_t local,NvmOwnerDeclaration *out) {
    if(!p || !out || f>=p->facts->counts.functions || local>=p->facts->signatures[f].local_count)return false;
    *out=p->facts->locals[f][local];return true;
}
bool nvm_owned_array_plan_layout(const NvmOwnedArrayPlan *p,uint32_t global,NvmOwnedArrayLayoutFact *out) {
    return p && nvm_owned_array_layout_fact(p->facts->layouts,global,out);
}
bool nvm_owned_array_plan_source(const NvmOwnedArrayPlan *p,uint32_t source,uint32_t *out) {
    return p && nvm_owned_array_source_layout(p->facts->layouts,source,out);
}
bool nvm_owned_array_plan_transport(const NvmOwnedArrayPlan *p,NvmOwnedArrayTransport *out) {
    return p && nvm_owned_array_transport(p->facts->layouts,out);
}
bool nvm_owned_array_plan_obligation(const NvmOwnedArrayPlan *p,uint32_t n,NvmOwnerOriginObligation *out) {
    if(!p || !out || n>=p->facts->counts.instructions)return false;
    *out=p->facts->obligations[n];return true;
}
