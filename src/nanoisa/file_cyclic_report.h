#ifndef NANOISA_FILE_CYCLIC_REPORT_H
#define NANOISA_FILE_CYCLIC_REPORT_H
#include "file_runtime.h"
/* I retain the qualified native C value layout; this is not a wire encoding. */
#define NVM_FILE_CYCLIC_RUNTIME_REVISION 1u
#define NVM_FILE_CYCLIC_FUEL_MAX UINT64_C(1000000)
#define NVM_FILE_CYCLIC_FUEL_DEFAULT UINT64_C(100000)
typedef struct { uint32_t revision; uint64_t instruction_limit; } NvmFileCyclicOptions;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmFileCyclicExecutionReport;
#endif
