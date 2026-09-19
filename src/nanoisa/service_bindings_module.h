#ifndef NANOISA_SERVICE_BINDINGS_MODULE_H
#define NANOISA_SERVICE_BINDINGS_MODULE_H
#include "service_bindings.h"
#include "nvm_v2_sections.h"
#include "../nsi_file_plan.h"

/* Presence includes malformed partial claims. Executing/dropping consumers
 * refuse any claim; they do not use successful transport validation as proof. */
bool nvm_service_bindings_present(const NvmModule *);
bool nvm_v2_service_bindings_present(const NvmV2Module *);
/* Valid in-memory containers must supply storage for their declared tables.
 * These queries validate this catalog's own pointers/counts and crossrefs. */
NvmV2Result nvm_service_bindings_validate(const NvmModule *);
NvmV2Result nvm_v2_service_bindings_validate(const NvmV2Module *);
/* A valid owned plan is required. Failure preserves every module field.
 * Identical valid reattachment is a no-op; a different payload is refused.
 * Memory failure follows the existing bridge's ERR_TRUNCATED convention. */
NvmV2Result nvm_service_bindings_attach(NvmModule *, const NlFilePlan *,
                                       const NvmServiceBindings *);
#endif
