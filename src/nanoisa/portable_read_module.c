#include "portable_read_module.h"
#include "portable_read_managed.h"
#ifdef __wasm32__
#include "portable_read_wasm.h"
#endif

/* I link to the exact managed runtime packaged in the generated module. */
extern NmsRuntime *nms_module_host_runtime(void);
extern uint32_t nms_module_status(void);
extern void nms_module_fail(uint32_t);
static NprStatus npr_module_error;
#ifndef __wasm32__
static NprHostBinding npr_module_binding;
NprStatus npr_module_bind(const NprHostBinding *binding) {
    NmsRuntime *runtime=nms_module_host_runtime();
    if(runtime->active || runtime->disposed)return NPR_INVALID;
    if(binding && (!binding->read || !binding->context))return NPR_INVALID;
    npr_module_binding=binding?*binding:(NprHostBinding){0};
    return NPR_OK;
}
#endif
uint32_t npr_module_host_status(void) { return npr_module_error; }
void npr_module_reset(void) {
    if(!nms_module_host_runtime()->active)npr_module_error=NPR_OK;
}
/* I retain the declared Wasm import even when the program never calls it. */
#ifdef __wasm32__
__attribute__((used, retain))
#endif
uint64_t npr_module_read_text(uint64_t argument) {
    if(nms_module_status()!=NMS_OK)return 0;
    NmsRuntime *runtime=nms_module_host_runtime();
#ifdef __wasm32__
    NprManagedResult result=npr_wasm_read_managed(runtime,argument);
#else
    NprManagedResult result=npr_read_managed(runtime,argument,&npr_module_binding);
#endif
    if(result.host_status!=NPR_OK && npr_module_error==NPR_OK)
        npr_module_error=result.host_status;
    nms_module_fail(result.managed_status);
    if(result.host_status!=NPR_OK)nms_module_fail(NMS_STATE);
    return result.value;
}
