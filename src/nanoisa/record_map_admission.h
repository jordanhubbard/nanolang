#ifndef NANOISA_RECORD_MAP_ADMISSION_H
#define NANOISA_RECORD_MAP_ADMISSION_H
#include "managed_record_map_execution.h"
/* DEFER preserves the old validator/diagnostic for framing where no map identity
 * was established. It is NOT proof of valid absence and grants no admission.
 * DEFER may only run old FULL validators, which reject bare HASHMAP fields and
 * every kind5 extension. Never use DEFER to directly select legacy execution;
 * future changes to those rejecting validators require this contract review. */
typedef enum { NVM_RECORD_MAP_ABSENT, NVM_RECORD_MAP_PRESENT,
    NVM_RECORD_MAP_INVALID, NVM_RECORD_MAP_LIMIT, NVM_RECORD_MAP_DEFER } NvmRecordMapRoute;
NvmRecordMapRoute nvm_record_map_route(const NvmModule *);
/* Wire-table presence only, for atomic staging/structural decode, never proof. */
bool nvm_record_map_wire_selected(const NvmV2Module *);
static inline bool nvm_record_map_selected(NvmRecordMapRoute r) {
    return r==NVM_RECORD_MAP_PRESENT || r==NVM_RECORD_MAP_INVALID || r==NVM_RECORD_MAP_LIMIT;
}
/* Fresh full module proof plus public VM root signatures, not native target
 * eligibility. Failure preserves output. No imported-state execution token. */
NvmArrayEligibilityResult nvm_record_map_admit(const NvmModule *,NvmRecordMapExecutionPlan **);
#endif
