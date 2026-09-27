#ifndef NANOISA_NVM2C_RECORD_ARRAY_PRIVATE_H
#define NANOISA_NVM2C_RECORD_ARRAY_PRIVATE_H
#include "managed_record_array_execution.h"
/* Private transactional source emission. Success owns *text until free(); all
 * failures leave both output arguments untouched. No public route selection. */
#ifdef NANO_RECORD_ARRAY_GENERATED_PRIVATE
typedef struct {
    uint64_t plan_bytes, plan_steps, consumer_bytes, consumer_steps;
} NvmRecordArrayGeneratedCost;
NvmArrayEligibilityResult nvm2c_record_array_private(const NvmModule *,
    char **text,size_t *length,NvmRecordArrayGeneratedCost *);
#endif
#endif
