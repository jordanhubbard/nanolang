#ifndef NANOISA_RECORD_ARRAY_STRUCTURE_PRIVATE_H
#define NANOISA_RECORD_ARRAY_STRUCTURE_PRIVATE_H
#include "managed_array_shapes.h"
#include "ownership_declaration_projection.h"
#include "verifier.h"
#include "../nanovm/vm_decode.h"
#include <stdlib.h>
#define NVM_RA_BYTES (UINT64_C(64)*1024*1024)
#define NVM_RA_STEPS UINT64_C(16777216)
/* Call-local accounting, never module trust. Reservations conservatively include
 * allocation capacity and temporary overlap, not only retained object sizes. */
typedef struct {
    uint64_t bytes, peak, work;
    bool limited, memory;
} NvmRecordArrayBudget;
static inline bool nvm_ra_bytes(NvmRecordArrayBudget *b,uint64_t n) {
    if(!b)return true;
    if(b->limited || b->memory)return false;
    if(b->bytes>NVM_RA_BYTES || n>NVM_RA_BYTES-b->bytes){b->limited=true;return false;}
    b->bytes+=n;if(b->peak<b->bytes)b->peak=b->bytes;return true;
}
static inline bool nvm_ra_steps(NvmRecordArrayBudget *b,uint64_t n) {
    if(!b)return true;
    if(b->limited || b->memory)return false;
    if(b->work>NVM_RA_STEPS || n>NVM_RA_STEPS-b->work){b->limited=true;return false;}
    b->work+=n;return true;
}
static inline void *nvm_ra_alloc(NvmRecordArrayBudget *b,size_t n,size_t width) {
    if(width && n>SIZE_MAX/width){if(b)b->limited=true;return NULL;}
    if(!nvm_ra_bytes(b,(uint64_t)n*width))return NULL;
    void *p=calloc(n,width);if(!p && n && width && b)b->memory=true;return p;
}
/* Internal owned staging only. No constructor accepts a caller's declaration
 * plan. Preparation validates this original module and makes its own plan. */
typedef struct {
    NvmOwnershipDeclarationPlan *declarations;
    uint32_t functions;
    VmDecodedFunction decoded[256];
    uint16_t stacks[256];
} NvmRecordArrayStructure;
bool nvm_record_array_opcode_supported(uint8_t);
NvmArrayEligibilityResult nvm_record_array_structure_prepare(
    const NvmModule *,NvmRecordArrayBudget *,NvmRecordArrayStructure **);
void nvm_record_array_structure_free(NvmRecordArrayStructure *);
NvmVerifyResult nvm_verify_function_types_record_array(
    const NvmModule *,uint32_t,const VmDecodedFunction *,uint16_t,NvmRecordArrayBudget *);
#endif
