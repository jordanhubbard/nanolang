#ifndef NL_SERVICE_SOURCE_CATALOG_H
#define NL_SERVICE_SOURCE_CATALOG_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
/* I identify exact source catalogs, not runtime grants or serialized tags. */
/* I reserve WebSocket identity for strict snapshots; source queries are pending. */
enum { NL_SOURCE_CATALOG_NONE = 0, NL_SOURCE_CATALOG_FILE = 1,
       NL_SOURCE_CATALOG_SOCKET = 2, NL_SOURCE_CATALOG_WEBSOCKET = 3 };
int64_t nl_service_source_catalog_id(const char *interface_id);
/* Kind 1 counts types; kind 2 counts methods. Invalid queries return -1. */
int64_t nl_service_source_catalog_count(int64_t catalog, int64_t kind);
/* I retain File's field numbering. Unknown selectors return empty/-1. */
const char *nl_service_source_catalog_string(int64_t catalog, int64_t kind,
                                            int64_t ordinal, int64_t field, int64_t member);
int64_t nl_service_source_catalog_number(int64_t catalog, int64_t kind,
                                        int64_t ordinal, int64_t field, int64_t member);
/* I render an allocation-free complete immutable view, including its identity.
 * Size includes NUL. Failure preserves both outputs. Storage is disjoint. */
bool nl_service_source_catalog_view(int64_t catalog, char *out, size_t capacity, size_t *needed);
#endif
