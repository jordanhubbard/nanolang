#ifndef NANOISA_FILE_CYCLIC_PUBLIC_INTERNAL_H
#define NANOISA_FILE_CYCLIC_PUBLIC_INTERNAL_H
#include "file_cyclic_public.h"
#include "file_public_internal.h"
/* I own no context or gate here. BUSY callers pass NULL without reading options. */
static inline NvmFileCyclicExecutionReport nvm_file_cyclic_public_refused(
    NvmFileRuntimeStatus status, const NvmFileCyclicOptions *options) {
    NvmFileCyclicExecutionReport report = {0};
    report.revision = NVM_FILE_CYCLIC_RUNTIME_REVISION;
    report.instruction_limit = options ? options->instruction_limit : 0;
    report.runtime = nvm_file_public_refused(status);
    return report;
}
static inline bool nvm_file_cyclic_public_options(const NvmFileCyclicOptions *options) {
    return options && options->revision == NVM_FILE_CYCLIC_RUNTIME_REVISION &&
        options->instruction_limit <= NVM_FILE_CYCLIC_FUEL_MAX;
}
#endif
