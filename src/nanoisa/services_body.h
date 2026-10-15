#ifndef NANOISA_SERVICES_BODY_H
#define NANOISA_SERVICES_BODY_H
#include "services_code.h"
/* Private acyclic logical body facts, not hosted or runtime authority. Input
 * must remain immutable during analysis. Successful reports borrow nothing.
 * Output storage is disjoint from inputs/report and other output objects.
 * Calls require external serialization like preparation. Failure preserves outputs. */
typedef struct NvmServicesBodyReport NvmServicesBodyReport;
typedef enum {
    NVM_SERVICES_BODY_CLEANUP_NORMAL, NVM_SERVICES_BODY_CLEANUP_DROP_LOCAL,
    NVM_SERVICES_BODY_CLEANUP_DROP_STACK, NVM_SERVICES_BODY_CLEANUP_ASSERT,
    NVM_SERVICES_BODY_CLEANUP_RETURN, NVM_SERVICES_BODY_CLEANUP_CALL,
    NVM_SERVICES_BODY_CLEANUP_SERVICE
} NvmServicesBodyCleanup;
typedef struct {
    bool reachable, exit_checked, refinement, has_obligation;
    uint16_t input_stack, output_stack, cleanup_local;
    NvmServicesBodyCleanup cleanup;
    uint32_t discharged_checks, pending_checks;
    NvmServicesFlowObligation obligation; /* Original mask retained verbatim. */
} NvmServicesBodyInstruction;
NvmServicesFlowStatus nvm_services_body_analyze(const NvmModule *, NvmServicesBodyReport **out);
void nvm_services_body_free(NvmServicesBodyReport *);
uint32_t nvm_services_body_function_count(const NvmServicesBodyReport *);
bool nvm_services_body_function(const NvmServicesBodyReport *, uint32_t, NvmServicesCodeFunction *);
bool nvm_services_body_local(const NvmServicesBodyReport *, uint32_t, uint16_t, NvmServicesFlowDeclaration *);
bool nvm_services_body_instruction(const NvmServicesBodyReport *, uint32_t, uint16_t,
                              NvmServicesCodeInstruction *, NvmServicesBodyInstruction *);
#endif
