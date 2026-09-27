#ifndef NANOISA_NVM2LLVM_RECORD_ARRAY_PRIVATE_H
#define NANOISA_NVM2LLVM_RECORD_ARRAY_PRIVATE_H
#include "nvm2c_record_array_private.h"
#ifdef NANO_RECORD_ARRAY_GENERATED_PRIVATE
/* I emit direct native or wasm32 LLVM text; success owns *text until free().
 * Failure preserves disjoint text/length/cost outputs. No public selection. */
NvmArrayEligibilityResult nvm2llvm_record_array_private(const NvmModule *, bool wasm32,
    char **text,size_t *length,NvmRecordArrayGeneratedCost *);
#endif
#endif
