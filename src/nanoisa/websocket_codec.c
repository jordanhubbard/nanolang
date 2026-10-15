#include "websocket_codec.h"
#include "service_codec_internal.h"
#include "service_websocket_nominal.h"
static NvmV2Result websocket_module(const NvmModule *m) {
    NvmWebSocketNominalPlan *plan=NULL;
    NvmWebSocketNominalStatus status=nvm_websocket_nominal_plan(m,&plan);
    nvm_websocket_nominal_plan_free(plan);
    if(status==NVM_WEBSOCKET_NOMINAL_DESCRIBED)return NVM_V2_OK;
    if(status==NVM_WEBSOCKET_NOMINAL_MEMORY)return NVM_V2_ERR_TRUNCATED;
    return status==NVM_WEBSOCKET_NOMINAL_LIMIT?NVM_V2_ERR_INDEX_RANGE:NVM_V2_ERR_SECTION_TYPE;
}
static NvmV2Result websocket_wire(const NvmV2Module *m) {
    return nvm_private_nominal_wire(m,NVM_WEBSOCKET_NOMINAL_METHODS,NVM_WEBSOCKET_NOMINAL_TYPES,websocket_module);
}
static const NvmPrivateServiceCodec websocket_codec={websocket_module,websocket_wire,NVM_WEBSOCKET_NOMINAL_BYTES};
NvmV2Result nvm_websocket_from_module(const NvmModule *m,NvmV2Module *out){return nvm_private_from_module(m,out,&websocket_codec);}
NvmV2Result nvm_websocket_to_module(const NvmV2Module *m,NvmModule **out){return nvm_private_to_module(m,out,&websocket_codec);}
NvmV2Result nvm_websocket_serialize(const NvmV2Module *m,uint8_t *out,size_t capacity,size_t *size){return nvm_private_serialize(m,out,capacity,size,&websocket_codec);}
NvmV2Result nvm_websocket_deserialize(const uint8_t *data,size_t size,NvmV2Module *out){return nvm_private_deserialize(data,size,out,&websocket_codec);}
