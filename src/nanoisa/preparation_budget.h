/* I charge one preparation across owned copies, scratch storage and passes. */
#ifndef NANOISA_PREPARATION_BUDGET_H
#define NANOISA_PREPARATION_BUDGET_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#define NVM_PREPARATION_MAX_BYTES ((size_t)32u * 1024u * 1024u)
#define NVM_PREPARATION_MAX_STEPS 1048576u
typedef struct { size_t bytes; uint32_t steps; } NvmPreparationBudget;
static inline bool nvm_preparation_budget_valid(const NvmPreparationBudget *budget) {
    return budget && budget->bytes<=NVM_PREPARATION_MAX_BYTES &&
        budget->steps<=NVM_PREPARATION_MAX_STEPS;
}
/* Callers stage a local copy and commit it only with the prepared output.
 * Budget storage must be disjoint from input and output storage.
 * Consumed allocation bytes are cumulative, including freed scratch. */
static inline bool nvm_preparation_charge(NvmPreparationBudget *budget,
                                           size_t bytes,uint32_t steps) {
    if(!budget || bytes>budget->bytes || steps>budget->steps)return false;
    budget->bytes-=bytes;budget->steps-=steps;return true;
}
#endif
