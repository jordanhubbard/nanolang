#ifndef NL_SERVICE_SHADOWS_H
#define NL_SERVICE_SHADOWS_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* I accept independently emitted bytes, never frontend facts. Callers retain
 * immutable storage until return and select the complete required suite.
 * This supervisor is serialized and is not a security sandbox. */
typedef struct {
    const uint8_t *bytes;
    size_t size;
    const char *origin, *name;
} NlServiceShadow;
typedef enum {
    NL_SERVICE_SHADOW_OK, NL_SERVICE_SHADOW_INVALID, NL_SERVICE_SHADOW_DENIED,
    NL_SERVICE_SHADOW_FAILED, NL_SERVICE_SHADOW_TIMEOUT, NL_SERVICE_SHADOW_SYSTEM
} NlServiceShadowStatus;
typedef struct {
    NlServiceShadowStatus status;
    size_t completed;
} NlServiceShadowReport;
/* I create log_path exclusively, mode 0600, within the caller's private staging
 * directory. Existing paths are never replaced. The caller owns log cleanup.
 * Names and origins are hex encoded in durable selection/start/done records.
 * One 10-second default deadline covers the entire suite; the shared bounded
 * NANO_SHADOW_TIMEOUT_SECONDS override applies. A grant is never persisted. */
NlServiceShadowReport nl_service_run_shadows(const NlServiceShadow *, size_t,
    bool allow_temporary_files, const char *log_path);
/* Catalog 1 selects File; catalog 2 selects TCP. The selected public consumer
 * independently checks every module; the boolean grants only this catalog. */
NlServiceShadowReport nl_service_run_catalog_shadows(const NlServiceShadow *, size_t,
    unsigned catalog, bool allowed, const char *log_path);
#endif
