#ifndef NANOVIRT_SHADOW_RUNNER_H
#define NANOVIRT_SHADOW_RUNNER_H
#include "nanolang.h"
#include "module_builder.h"
#include "nanoisa/nvm_format.h"
typedef struct {
    char *artifact;
    ModuleBuildMetadata *metadata;
} FfiBinding;
void free_ffi_bindings(FfiBinding *bindings, int count);
bool bind_ffi_imports(NvmModule *module, ModuleList *modules,
    FfiBinding *bindings, const char *input, Environment *env);
bool build_ffi_modules(ModuleList *modules, FfiBinding *bindings);
bool check_shadows(ASTNode *program, Environment *env, ModuleList *modules,
    const char *input, FfiBinding *bindings, bool include_imports);
/* 0: no retained foreign policy; 1: passed; -1: failed. */
int check_callback_shadows(ASTNode *program, Environment *env, ModuleList *modules,
    const char *input, bool include_imports);
#endif
