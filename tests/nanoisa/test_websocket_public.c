/* I reuse the complete bytecode lifecycle through my public boundaries. */
#include "../../src/nanoisa/websocket_indirect_public.h"
#include <assert.h>
static NvmWebSocketRuntimeStatus public_emit(const uint8_t *bytes,size_t size,char **out,char *error,size_t capacity) {
    return nvm2c_emit_websocket_indirect_bytes(bytes,size,"test",out,error,capacity);
}
static NvmWebSocketIndirectExecutionReport public_vm(const uint8_t *bytes,size_t size,
    const NvmWebSocketIndirectOptions *options,NvmWebSocketRuntimeView *out,const NlWsTransportPolicy *policy) {
    NvmWebSocketHostGrant *grant=NULL;
    if(policy){NvmWebSocketHostPolicy input={1,policy->allow_network,policy->allow_lookup,policy->max_timeout_ms,policy->resolver_helper};
        assert(nvm_websocket_host_grant_create(&input,&grant)==NVM_WEBSOCKET_HOST_OK);}
    NvmWebSocketScalar scalar={TAG_INT,12345};
    NvmWebSocketIndirectExecutionReport result=nvm_websocket_execute_indirect_bytes(grant,bytes,size,options,&scalar);
    if(result.runtime.status==NVM_WEBSOCKET_RUNTIME_OK){out->fields=1;out->values[0]=scalar.value;}
    else assert(scalar.tag==TAG_INT && scalar.value==12345);
    assert(nvm_websocket_host_grant_destroy(&grant)==NVM_WEBSOCKET_HOST_OK);
    return result;
}
#define nvm2c_websocket_indirect_private_emit public_emit
#define nvm_websocket_vm_indirect_execute public_vm
#include "test_websocket_dispatch.c"
