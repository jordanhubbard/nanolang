#ifndef NANOISA_WEBSOCKET_CODEC_H
#define NANOISA_WEBSOCKET_CODEC_H
#include "nvm_v2_sections.h"
#include "nvm_format.h"
/* I transport only the exact WebSocket nominal catalog. These private native
 * entrypoints do not register source, verifier or VM execution admission.
 * Bridge storage ownership matches the ordinary codec. */
NvmV2Result nvm_websocket_from_module(const NvmModule *,NvmV2Module *);
NvmV2Result nvm_websocket_to_module(const NvmV2Module *,NvmModule **);
NvmV2Result nvm_websocket_serialize(const NvmV2Module *,uint8_t *,size_t,size_t *);
NvmV2Result nvm_websocket_deserialize(const uint8_t *,size_t,NvmV2Module *);
#endif
