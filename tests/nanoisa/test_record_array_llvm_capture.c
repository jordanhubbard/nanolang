/* I select only the emitter; I retain every original VM program and observation. */
#include "../../src/nanoisa/nvm2llvm_record_array_private.h"
#ifndef RECORD_LLVM_WASM32
#define RECORD_LLVM_WASM32 0
#endif
static NvmArrayEligibilityResult capture_llvm_emit(const NvmModule *module,
    char **text,size_t *length,NvmRecordArrayGeneratedCost *cost) {
    return nvm2llvm_record_array_private(module,RECORD_LLVM_WASM32!=0,text,length,cost);
}
#if RECORD_LLVM_WASM32
#define RECORD_LLVM_WASM_OBSERVER 1
#endif
#define RECORD_GENERATED_EMIT capture_llvm_emit
#define RECORD_GENERATED_SUFFIX "ll"
#include "test_record_array_generated_capture.c"
