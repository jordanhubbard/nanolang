#ifndef NANOISA_FILE_INDIRECT_PUBLIC_INTERNAL_H
#define NANOISA_FILE_INDIRECT_PUBLIC_INTERNAL_H
#include "file_indirect_public.h"
#include "file_public_internal.h"
/* I own no context or gate here. BUSY callers pass NULL without reading options. */
static inline NvmFileIndirectExecutionReport nvm_file_indirect_public_refused(
    NvmFileRuntimeStatus status, const NvmFileIndirectOptions *options) {
    NvmFileIndirectExecutionReport report = {0};
    report.revision = NVM_FILE_INDIRECT_RUNTIME_REVISION;
    report.instruction_limit = options ? options->instruction_limit : 0;
    report.runtime = nvm_file_public_refused(status);
    return report;
}
static inline bool nvm_file_indirect_public_options(const NvmFileIndirectOptions *options) {
    return options && options->revision == NVM_FILE_INDIRECT_RUNTIME_REVISION &&
        options->instruction_limit <= NVM_FILE_INDIRECT_FUEL_MAX;
}
#endif
