#ifndef NANOISA_SERVICES_INDIRECT_REPORT_H
#define NANOISA_SERVICES_INDIRECT_REPORT_H
#include "services_runtime.h"
/* I require explicit fuel and publish a scalar only after clean destruction. */
#define NVM_SERVICES_INDIRECT_RUNTIME_REVISION 1u
#define NVM_SERVICES_INDIRECT_FUEL_MAX UINT64_C(1000000)
typedef struct { uint32_t revision; uint64_t instruction_limit; } NvmServicesIndirectOptions;
typedef struct {
    uint32_t revision;
    NvmServicesRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmServicesIndirectExecutionReport;
#endif
