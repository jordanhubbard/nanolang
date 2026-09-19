#ifndef NANOISA_SERVICE_BINDINGS_H
#define NANOISA_SERVICE_BINDINGS_H
#include <stddef.h>
#include <stdint.h>

/* Private raw codec only. I neither recognize a public module feature nor grant
 * import/host authority here. Catalog1 is the immutable PR814 file contract. */
#define NVM_SERVICE_BINDING_VERSION 1u
#define NVM_SERVICE_BINDING_CATALOG_FILE 1u
#define NVM_SERVICE_BINDING_COUNT 5u
#define NVM_SERVICE_BINDING_BYTES 56u

typedef struct { uint32_t imports[NVM_SERVICE_BINDING_COUNT]; } NvmServiceBindings;
typedef enum {
    NVM_SERVICE_OK = 0, NVM_SERVICE_ARGUMENT, NVM_SERVICE_SIZE,
    NVM_SERVICE_VERSION, NVM_SERVICE_CATALOG, NVM_SERVICE_COUNT,
    NVM_SERVICE_RESERVED, NVM_SERVICE_ORDINAL, NVM_SERVICE_INDEX
} NvmServiceResult;

/* I check distinct indices and the reserved no-index value, not table bounds.
 * A later module validator must enforce the exact five-entry import table and
 * category signatures/identities. No successful raw query admits execution. */
NvmServiceResult nvm_service_bindings_check(const NvmServiceBindings *);
/* I leave the output unchanged on every failure; input memory is not retained. */
NvmServiceResult nvm_service_bindings_decode(const uint8_t *, size_t, NvmServiceBindings *);
/* Null bytes selects size-only mode. A size pointer is required in both modes.
 * Failure leaves bytes and size unchanged. Size storage must be disjoint from
 * the value and byte output; input value/byte output may overlap. */
NvmServiceResult nvm_service_bindings_encode(const NvmServiceBindings *, uint8_t *, size_t, size_t *);
#endif
