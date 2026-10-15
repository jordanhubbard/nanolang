#ifndef FILE_INDIRECT_DISPATCH_TEST_HOOKS_H
#define FILE_INDIRECT_DISPATCH_TEST_HOOKS_H
#include "../../src/nanoisa/file_indirect_runtime.h"
NvmFileRuntimeStatus dispatch_begin(NvmFileRuntime *);
NvmFileRuntimeStatus dispatch_call(NvmFileRuntime *);
NvmFileRuntimeStatus dispatch_return(NvmFileRuntime *);
NvmFileRuntimeStatus dispatch_service(NvmFileRuntime *,uint32_t,uint32_t,uint32_t,uint32_t);
NvmFileIndirectExecutionReport dispatch_destroy(NvmFileRuntime **,NvmFileRuntimeView *);
#endif
