#ifndef NANOISA_LOCAL_BINDINGS_H
#define NANOISA_LOCAL_BINDINGS_H
#include "nvm_format.h"
/* I describe lexical names, never executable authority. Views borrow the module. */
typedef struct {
    uint32_t function, begin, end;
    uint16_t slot;
    const uint8_t *name;
    uint32_t name_size;
} NvmLocalBinding;
typedef enum { NVM_LOCAL_NAMES_ABSENT, NVM_LOCAL_NAMES_VALID, NVM_LOCAL_NAMES_INVALID } NvmLocalNamesStatus;
NvmLocalNamesStatus nvm_local_names_validate(const NvmModule *module);
NvmLocalNamesStatus nvm_local_name_at(const NvmModule *module, uint32_t function,
                                     uint16_t slot, uint32_t pc, NvmLocalBinding *out);
bool nvm_add_local_binding(NvmModule *module, const NvmLocalBinding *binding);
#endif
