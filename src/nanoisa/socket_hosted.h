#ifndef NANOISA_SOCKET_HOSTED_H
#define NANOISA_SOCKET_HOSTED_H
#include "socket_body.h"
#define NVM_SOCKET_HOSTED_INPUT_BYTES (16u*1024u*1024u)
#define NVM_SOCKET_HOSTED_BYTES (64u*1024u*1024u)
typedef struct NvmSocketHostedPlan NvmSocketHostedPlan;
typedef struct {
    uint32_t entry, initializer, functions, features;
    uint64_t vm_value_slots, native_value_slots;
    uint16_t frames;
    uint32_t reference_slots, region_slots;
    size_t allocation_bound;
} NvmSocketHostedStartup;
typedef struct {
    NvmSocketCodeFunction code;
    uint16_t declared_stack, operand_peak, locals, staging_slots, frames;
    uint32_t reference_slots, region_slots;
    uint64_t vm_value_slots, native_value_slots;
} NvmSocketHostedFunction;
/* Private serialized-v2 startup/storage facts only. No host/service operation,
 * public admission or concrete carrier ABI certification. Input stays immutable
 * during the call; output is disjoint. Reports own all facts; failure preserves
 * output. Calls require the underlying private query's external serialization.
 * Value-slot upper bounds exclude bytes inside a runtime carrier/host resource;
 * a later matched runtime must separately check those extents and cleanup. */
NvmSocketFlowStatus nvm_socket_hosted_prepare(const uint8_t *,size_t,NvmSocketHostedPlan **);
void nvm_socket_hosted_free(NvmSocketHostedPlan *);
bool nvm_socket_hosted_startup(const NvmSocketHostedPlan *,NvmSocketHostedStartup *);
bool nvm_socket_hosted_function(const NvmSocketHostedPlan *,uint32_t,NvmSocketHostedFunction *);
bool nvm_socket_hosted_local(const NvmSocketHostedPlan *,uint32_t,uint16_t,NvmSocketFlowDeclaration *);
bool nvm_socket_hosted_instruction(const NvmSocketHostedPlan *,uint32_t,uint16_t,
                                NvmSocketCodeInstruction *,NvmSocketBodyInstruction *);
/* Read-only exact nominal/catalog maps owned by this same hosted plan. */
bool nvm_socket_hosted_type(const NvmSocketHostedPlan *,uint32_t,NvmSocketNominalLayout *);
bool nvm_socket_hosted_import(const NvmSocketHostedPlan *,uint32_t,uint32_t *);
#endif
