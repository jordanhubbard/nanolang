#ifndef NANOISA_SOCKET_BODY_H
#define NANOISA_SOCKET_BODY_H
#include "socket_code.h"
/* Private acyclic logical body facts, not hosted or runtime authority. Input
 * must remain immutable during analysis. Successful reports borrow nothing.
 * Output storage is disjoint from inputs/report and other output objects.
 * Calls require external serialization like preparation. Failure preserves outputs. */
typedef struct NvmSocketBodyReport NvmSocketBodyReport;
typedef enum {
    NVM_SOCKET_BODY_CLEANUP_NORMAL, NVM_SOCKET_BODY_CLEANUP_DROP_LOCAL,
    NVM_SOCKET_BODY_CLEANUP_DROP_STACK, NVM_SOCKET_BODY_CLEANUP_ASSERT,
    NVM_SOCKET_BODY_CLEANUP_RETURN, NVM_SOCKET_BODY_CLEANUP_CALL,
    NVM_SOCKET_BODY_CLEANUP_SERVICE
} NvmSocketBodyCleanup;
typedef struct {
    bool reachable, exit_checked, refinement, has_obligation;
    uint16_t input_stack, output_stack, cleanup_local;
    NvmSocketBodyCleanup cleanup;
    uint32_t discharged_checks, pending_checks;
    NvmSocketFlowObligation obligation; /* Original mask retained verbatim. */
} NvmSocketBodyInstruction;
NvmSocketFlowStatus nvm_socket_body_analyze(const NvmModule *, NvmSocketBodyReport **out);
void nvm_socket_body_free(NvmSocketBodyReport *);
uint32_t nvm_socket_body_function_count(const NvmSocketBodyReport *);
bool nvm_socket_body_function(const NvmSocketBodyReport *, uint32_t, NvmSocketCodeFunction *);
bool nvm_socket_body_local(const NvmSocketBodyReport *, uint32_t, uint16_t, NvmSocketFlowDeclaration *);
bool nvm_socket_body_instruction(const NvmSocketBodyReport *, uint32_t, uint16_t,
                              NvmSocketCodeInstruction *, NvmSocketBodyInstruction *);
#endif
