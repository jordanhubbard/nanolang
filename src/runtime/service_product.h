#ifndef NL_SERVICE_PRODUCT_H
#define NL_SERVICE_PRODUCT_H
#include "service_shadows.h"
typedef struct {
    const char *output, *root, *compiler, *cflags, *ldflags;
    bool emit_nvm, allow_temporary_files, run, allow_tcp_connections;
} NlServiceProductOptions;
/* I consume already lowered bytes and selected shadows. My caller checks all
 * source/companion aliases before calling. I never mutate those inputs or keep
 * a grant. I publish only after verification, shadows and staging cleanup. */
int nl_service_publish(const uint8_t *, size_t, const NlServiceShadow *, size_t,
                       const NlServiceProductOptions *);
#endif
