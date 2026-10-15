#include "file_runtime.h"
#include "file_runtime_frames.h"
#include "file_cyclic_runtime.h"
#include "file_indirect_runtime.h"
#include "../nsi_file_values_internal.h"
#include "nvm_v2_sections.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "file_runtime_config.h"
#include "service_runtime.inc"
#if defined(NVM_FILE_NATIVE_PRIVATE) || defined(NVM_FILE_PUBLIC_ENGINE)
#include "file_native_abi.h"
bool nvm_file_runtime_native_abi(uint32_t revision,size_t view,size_t frame,size_t report) {
    return revision==NVM_FILE_NATIVE_ABI && view==sizeof(NvmFileRuntimeView) &&
        frame==sizeof(NvmFileRuntimeFrameView) && report==sizeof(NvmFileRuntimeReport);
}
#endif
