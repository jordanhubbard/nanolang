#ifndef NANOISA_SERVICE_CODEC_INTERNAL_H
#define NANOISA_SERVICE_CODEC_INTERNAL_H
#include "service_bindings_module.h"
/* I permit a trusted native adapter to select its exact metadata validators.
 * Success must establish complete nominal ownership metadata. These callbacks
 * grant no execution authority. Ordinary codec entrypoints always pass NULL. */
typedef struct {
    NvmV2Result (*module)(const NvmModule *);
    NvmV2Result (*wire)(const NvmV2Module *);
    uint32_t service_bytes;
} NvmPrivateServiceCodec;
static inline NvmV2Result nvm_private_service_module(const NvmModule *m,const NvmPrivateServiceCodec *c) {
    return c?(c->module && c->wire && c->service_bytes?c->module(m):NVM_V2_ERR_SECTION_TYPE):nvm_service_bindings_validate(m);
}
static inline NvmV2Result nvm_private_service_wire(const NvmV2Module *m,const NvmPrivateServiceCodec *c) {
    return c?(c->module && c->wire && c->service_bytes?c->wire(m):NVM_V2_ERR_SECTION_TYPE):nvm_v2_service_bindings_validate(m);
}
NvmV2Result nvm_private_nominal_wire(const NvmV2Module *,uint32_t minimum_imports,
    NvmV2Result (*validate)(const NvmModule *));
NvmV2Result nvm_private_from_module(const NvmModule *,NvmV2Module *,const NvmPrivateServiceCodec *);
NvmV2Result nvm_private_to_module(const NvmV2Module *,NvmModule **,const NvmPrivateServiceCodec *);
NvmV2Result nvm_private_serialize(const NvmV2Module *,uint8_t *,size_t,size_t *,const NvmPrivateServiceCodec *);
NvmV2Result nvm_private_deserialize(const uint8_t *,size_t,NvmV2Module *,const NvmPrivateServiceCodec *);
#endif
