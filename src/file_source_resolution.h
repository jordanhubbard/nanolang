#ifndef NL_FILE_SOURCE_RESOLUTION_H
#define NL_FILE_SOURCE_RESOLUTION_H
#include "file_companion_snapshot.h"
typedef struct NlFileResolution NlFileResolution;
typedef enum { NL_FILE_RESOLUTION_NONE,NL_FILE_RESOLUTION_PREPARED,
 NL_FILE_RESOLUTION_INVALID,NL_FILE_RESOLUTION_LIMIT,NL_FILE_RESOLUTION_MEMORY,
 NL_FILE_RESOLUTION_UNRESOLVED,NL_FILE_RESOLUTION_IO } NlFileResolutionStatus;
typedef struct {
 NlFileResolutionStatus status;
 size_t modules,services,ordinary,visibility;
 /* Retained graph storage includes fixed tables, copied paths/source/qualifiers. */
 uint64_t input_bytes,source_bytes,plan_bytes,work;
 NlFileCompanionReport companion;
} NlFileResolutionReport;
typedef struct {
 NlFileSourceText origin,qualifier,name,target_origin,target_name;
 uint32_t id,target,kind,ordinal;
 bool exported,service;
} NlFileVisibility;
/* I use the actual parser and import resolver before module preparation.
 * This API is descriptive only. NONE leaves output unchanged; PREPARED owns an
 * immutable resolution and does not permit lowering/execution. Other failures
 * preserve output. Source providers/ancestors are stable for this invocation.
 * Existing parser/lexer failure may be ambiguous; I do not diagnose all as OOM.
 * My input/work limits do not claim to measure legacy parser allocator storage. */
NlFileResolutionReport nl_file_source_resolve(const char *,NlFileResolution **);
void nl_file_source_resolution_free(NlFileResolution *);
size_t nl_file_source_resolution_origins(const NlFileResolution *);
bool nl_file_source_resolution_origin(const NlFileResolution *,size_t,NlFileSourceText *,NlFileSourceText *);
size_t nl_file_source_resolution_visibility(const NlFileResolution *);
bool nl_file_source_resolution_row(const NlFileResolution *,size_t,NlFileVisibility *);
bool nl_file_source_resolution_snapshot(const NlFileResolution *,size_t,NlFileCompanionView *);
size_t nl_file_source_resolution_plan_rows(const NlFileResolution *);
bool nl_file_source_resolution_plan_row(const NlFileResolution *,size_t,NlFileSourceRow *);
/* Successful views borrow owning immutable storage until free. Output fields
 * are valid disjoint storage, also disjoint from the owning resolution. */
#endif
