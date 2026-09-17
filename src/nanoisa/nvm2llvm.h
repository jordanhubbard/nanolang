#ifndef NVM2LLVM_H
#define NVM2LLVM_H
#include "nvm_format.h"
#include <stdio.h>
/* I validate the complete scalar profile before writing any IR. */
int nvm2llvm_emit(const NvmModule *module, FILE *output, char *error, size_t size);
/* I allow main or a nano_ identifier for freestanding target entry points. */
int nvm2llvm_emit_entry(const NvmModule *module, FILE *output, char *error, size_t size, const char *entry);
#endif
