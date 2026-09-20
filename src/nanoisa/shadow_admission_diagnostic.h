#ifndef NANO_SHADOW_ADMISSION_DIAGNOSTIC_H
#define NANO_SHADOW_ADMISSION_DIAGNOSTIC_H
/* I forward unchanged queries; nested inclusive timing is diagnostic only. */
#ifdef NANO_SHADOW_PROGRESS_DIAGNOSTIC
#include <stdint.h>
enum { SHADOW_SERVICE, SHADOW_OWNER, SHADOW_MIXED, SHADOW_CONTRACTS,
       SHADOW_TRANSFERS, SHADOW_LEAF_COUNT };
uint64_t shadow_query_begin(void);
void shadow_query_end(unsigned kind, uint64_t token);
#define SHADOW_QUERY(type, name, params, args, kind) \
    static type name##_observed_body params; \
    type name params { \
        uint64_t token = shadow_query_begin(); \
        type result = name##_observed_body args; \
        shadow_query_end(kind, token); \
        return result; \
    } \
    static type name##_observed_body params
#else
#define SHADOW_QUERY(type, name, params, args, kind) type name params
#endif
#endif
