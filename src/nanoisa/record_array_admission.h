#ifndef NANOISA_RECORD_ARRAY_ADMISSION_H
#define NANOISA_RECORD_ARRAY_ADMISSION_H
#include "managed_record_array_execution.h"
#define NVM_RECORD_ARRAY_ADMISSION_FUNCTIONS 256u
/* I classify bounded bytes only; a candidate is never an admission proof. */
bool nvm_record_array_candidate(const NvmModule *);
/* I prepare the original immutable module synchronously. Callers discard this
 * owned plan before returning; it is not reusable authority for another call. */
NvmArrayEligibilityResult nvm_record_array_admit(const NvmModule *,NvmRecordArrayExecutionPlan **);
#endif
