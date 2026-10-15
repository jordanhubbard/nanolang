#ifndef NANOISA_WEBSOCKET_BODY_H
#define NANOISA_WEBSOCKET_BODY_H
#include "websocket_code.h"
/* Private acyclic logical body facts, not hosted or runtime authority. Input
 * must remain immutable during analysis. Successful reports borrow nothing.
 * Output storage is disjoint from inputs/report and other output objects.
 * Calls require external serialization like preparation. Failure preserves outputs. */
typedef struct NvmWebSocketBodyReport NvmWebSocketBodyReport;
typedef enum {
    NVM_WEBSOCKET_BODY_CLEANUP_NORMAL, NVM_WEBSOCKET_BODY_CLEANUP_DROP_LOCAL,
    NVM_WEBSOCKET_BODY_CLEANUP_DROP_STACK, NVM_WEBSOCKET_BODY_CLEANUP_ASSERT,
    NVM_WEBSOCKET_BODY_CLEANUP_RETURN, NVM_WEBSOCKET_BODY_CLEANUP_CALL,
    NVM_WEBSOCKET_BODY_CLEANUP_SERVICE
} NvmWebSocketBodyCleanup;
typedef struct {
    bool reachable, exit_checked, refinement, has_obligation;
    uint16_t input_stack, output_stack, cleanup_local;
    NvmWebSocketBodyCleanup cleanup;
    uint32_t discharged_checks, pending_checks;
    NvmWebSocketFlowObligation obligation; /* Original mask retained verbatim. */
} NvmWebSocketBodyInstruction;
NvmWebSocketFlowStatus nvm_websocket_body_analyze(const NvmModule *, NvmWebSocketBodyReport **out);
void nvm_websocket_body_free(NvmWebSocketBodyReport *);
uint32_t nvm_websocket_body_function_count(const NvmWebSocketBodyReport *);
bool nvm_websocket_body_function(const NvmWebSocketBodyReport *, uint32_t, NvmWebSocketCodeFunction *);
bool nvm_websocket_body_local(const NvmWebSocketBodyReport *, uint32_t, uint16_t, NvmWebSocketFlowDeclaration *);
bool nvm_websocket_body_instruction(const NvmWebSocketBodyReport *, uint32_t, uint16_t,
                              NvmWebSocketCodeInstruction *, NvmWebSocketBodyInstruction *);
#endif
