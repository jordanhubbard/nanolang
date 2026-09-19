#ifndef NANOISA_FILE_BODY_H
#define NANOISA_FILE_BODY_H
#include "file_code.h"
/* Private acyclic logical body facts, not hosted or runtime authority. Input
 * must remain immutable during analysis. Successful reports borrow nothing.
 * Output storage is disjoint from inputs/report and other output objects.
 * Calls require external serialization like preparation. Failure preserves outputs. */
typedef struct NvmFileBodyReport NvmFileBodyReport;
typedef enum {
    NVM_FILE_BODY_CLEANUP_NORMAL, NVM_FILE_BODY_CLEANUP_DROP_LOCAL,
    NVM_FILE_BODY_CLEANUP_DROP_STACK, NVM_FILE_BODY_CLEANUP_ASSERT,
    NVM_FILE_BODY_CLEANUP_RETURN, NVM_FILE_BODY_CLEANUP_CALL,
    NVM_FILE_BODY_CLEANUP_SERVICE
} NvmFileBodyCleanup;
typedef struct {
    bool reachable, exit_checked, refinement, has_obligation;
    uint16_t input_stack, output_stack, cleanup_local;
    NvmFileBodyCleanup cleanup;
    uint32_t discharged_checks, pending_checks;
    NvmFileFlowObligation obligation; /* Original mask retained verbatim. */
} NvmFileBodyInstruction;
NvmFileFlowStatus nvm_file_body_analyze(const NvmModule *, NvmFileBodyReport **out);
void nvm_file_body_free(NvmFileBodyReport *);
uint32_t nvm_file_body_function_count(const NvmFileBodyReport *);
bool nvm_file_body_function(const NvmFileBodyReport *, uint32_t, NvmFileCodeFunction *);
bool nvm_file_body_local(const NvmFileBodyReport *, uint32_t, uint16_t, NvmFileFlowDeclaration *);
bool nvm_file_body_instruction(const NvmFileBodyReport *, uint32_t, uint16_t,
                              NvmFileCodeInstruction *, NvmFileBodyInstruction *);
#endif
