#ifndef NANOISA_SOCKET_INDIRECT_REPORT_H
#define NANOISA_SOCKET_INDIRECT_REPORT_H
#include "socket_runtime.h"
/* I require explicit fuel and publish a scalar only after clean destruction. */
#define NVM_SOCKET_INDIRECT_RUNTIME_REVISION 1u
#define NVM_SOCKET_INDIRECT_FUEL_MAX UINT64_C(1000000)
typedef struct { uint32_t revision; uint64_t instruction_limit; } NvmSocketIndirectOptions;
typedef struct {
    uint32_t revision;
    NvmSocketRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmSocketIndirectExecutionReport;
#endif
