#ifndef NVM2C_CALLABLES_H
#define NVM2C_CALLABLES_H

#include "nvm_format.h"
#include "nvm2c_shape.h"

/* I compute possible module-local targets before native representation
 * inference. This analysis does not admit opcodes, signatures or ownership. */
typedef struct {
    NvmShapeGraph shapes;
    NvmShapeId **callees;
    uint32_t function_count;
    uint32_t *code_lengths;
    char error[256];
} NvmCallableAnalysis;

/* I require a fresh output, or one cleared with nvm_callable_destroy.
 * I retain failure details and partial storage for the caller to destroy. */
int nvm_callable_analyze(const NvmModule *module, NvmCallableAnalysis *out);
void nvm_callable_destroy(NvmCallableAnalysis *analysis);
/* I return zero for an unresolved callee or an instruction that is not a call.
 * A nonzero shape is a FUNCTION whose targets remain module indices. */
NvmShapeId nvm_callable_at(NvmCallableAnalysis *analysis, uint32_t function,
                          uint32_t byte_offset);

#endif
