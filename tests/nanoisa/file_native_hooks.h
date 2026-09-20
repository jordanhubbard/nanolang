#ifndef FILE_NATIVE_TEST_HOOKS_H
#define FILE_NATIVE_TEST_HOOKS_H
#include "../../src/nanoisa/file_runtime_frames.h"
NvmFileRuntimeStatus file_native_begin(NvmFileRuntime *);
NvmFileRuntimeStatus file_native_call(NvmFileRuntime *);
NvmFileRuntimeStatus file_native_return(NvmFileRuntime *);
NvmFileRuntimeStatus file_native_service(NvmFileRuntime *,uint32_t,uint32_t,uint32_t,uint32_t);
#endif
