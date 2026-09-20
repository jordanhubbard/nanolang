#ifndef FILE_CYCLIC_DISPATCH_TEST_HOOKS_H
#define FILE_CYCLIC_DISPATCH_TEST_HOOKS_H
#include "../../src/nanoisa/file_cyclic_runtime.h"
NvmFileRuntimeStatus dispatch_begin(NvmFileRuntime *);
NvmFileRuntimeStatus dispatch_call(NvmFileRuntime *);
NvmFileRuntimeStatus dispatch_return(NvmFileRuntime *);
NvmFileRuntimeStatus dispatch_service(NvmFileRuntime *,uint32_t,uint32_t,uint32_t,uint32_t);
NvmFileCyclicExecutionReport dispatch_destroy(NvmFileRuntime **,NvmFileRuntimeView *);
#endif
