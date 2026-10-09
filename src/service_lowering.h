#ifndef NL_SERVICE_LOWERING_H
#define NL_SERVICE_LOWERING_H
#include "service_ownership.h"
#include "nanoisa/nvm_format.h"
/* I lower the complete checked graph, including unselected shadows. I grant
 * no host access and publish no filesystem output. Selection identifies a
 * parsed function or shadow in this exact graph; NULL selects root main.
 * All input storage remains immutable and live during this serialized call.
 * Success transfers one independently owned module; failure preserves *out. */
typedef struct {
    unsigned status; /* 0 lowered, 1 invalid, 2 unsupported, 3 limit, 4 memory */
    int line, column;
    const char *diagnostic;
} NlServiceLoweringResult;
NlServiceLoweringResult nl_service_lower(const NlServiceNamespace *,
    const NlServiceBodyCheck *, const NlServiceOwnershipCheck *,
    const ASTNode *selection, NvmModule **out);
/* I derive exact stack bounds from the complete cyclic File analysis before
 * serialization. Consumer admission remains a separate required check. */
NlServiceLoweringResult nl_service_serialize(const NvmModule *, uint8_t **out, size_t *size);
#endif
