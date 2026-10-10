#ifndef NVM2LLVM_H
#define NVM2LLVM_H
#include "nvm_format.h"
#include <stdio.h>
/* I validate the complete scalar profile before writing any IR. */
typedef enum { NVM_LLVM_NATIVE = 0, NVM_LLVM_WASM32 = 1 } NvmLlvmTarget;
/* I opt into exact read-text declarations. A linked adapter and explicit host
 * binding remain required at execution; this API grants no filesystem access. */
int nvm2llvm_emit_portable_read_target(const NvmModule *, FILE *, char *, size_t, const char *, NvmLlvmTarget);
int nvm2llvm_emit_target(const NvmModule *, FILE *, char *, size_t, const char *, NvmLlvmTarget);
int nvm2llvm_emit(const NvmModule *module, FILE *output, char *error, size_t size);
/* I allow main or a nano_ identifier for freestanding target entry points. */
int nvm2llvm_emit_entry(const NvmModule *module, FILE *output, char *error, size_t size, const char *entry);
#endif
