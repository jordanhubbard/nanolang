#ifndef NL_FILE_SOURCE_PLAN_H
#define NL_FILE_SOURCE_PLAN_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
/* Descriptive only: the caller supplies complete checked namespace facts and
 * an already validated NSI catalog view. No source parsing or host authority. */
#define NL_FILE_SOURCE_REQUESTS 16u
#define NL_FILE_SOURCE_BINDINGS 13u
#define NL_FILE_SOURCE_ALIASES 64u
#define NL_FILE_SOURCE_ORDINARY 256u
#define NL_FILE_SOURCE_TEXT_BUDGET 1048576u
#define NL_FILE_SOURCE_NO_INDEX UINT32_MAX
typedef enum { NL_FILE_SOURCE_OK,NL_FILE_SOURCE_INVALID,NL_FILE_SOURCE_LIMIT,
 NL_FILE_SOURCE_MEMORY,NL_FILE_SOURCE_UNRESOLVED } NlFileSourceStatus;
typedef struct { const char *data;size_t size; } NlFileSourceText;
typedef struct { uint32_t id,kind,ordinal;NlFileSourceText name; } NlFileSourceBinding;
typedef struct {
 NlFileSourceText module,interface_id,catalog_view;
 uint32_t catalog_version,line,column;
 const NlFileSourceBinding *bindings;size_t binding_count;
} NlFileSourceRequest;
typedef struct { NlFileSourceText module,name;uint32_t id,target; } NlFileSourceAlias;
typedef struct { NlFileSourceText module,name;uint32_t id; } NlFileSourceOrdinary;
typedef struct NlFileSourcePlan NlFileSourcePlan;
typedef struct {
 NlFileSourceText module,name;uint32_t id,target,request,kind,ordinal,category;
 uint32_t input_mode,result_ordinal,global_layout,import_index,line,column;
} NlFileSourceRow;
/* kind:0 type,1 method. category:1 File,2 affine OpenResult,3 scalar record,
 * 4 scalar Result,5 method. Future wire indices are always NO_INDEX.
 * All input spans/arrays are valid immutable storage during the call. Output
 * storage is disjoint from input/plan. Failure preserves *out. */
NlFileSourceStatus nl_file_source_plan_build(const NlFileSourceRequest *,size_t,
 const NlFileSourceAlias *,size_t,const NlFileSourceOrdinary *,size_t,NlFileSourcePlan **out);
void nl_file_source_plan_free(NlFileSourcePlan *);
size_t nl_file_source_plan_count(const NlFileSourcePlan *);
size_t nl_file_source_plan_bytes(const NlFileSourcePlan *);
bool nl_file_source_plan_row(const NlFileSourcePlan *,size_t,NlFileSourceRow *out);
/* Copied rows borrow immutable strings from the live owning plan. */
/* Data-only immutable catalog access: kind0 interface,1 types,2 methods.
 * Invalid queries return empty string/-1. Fields are documented in .c. */
const char *nl_file_source_catalog_string(int64_t,int64_t,int64_t,int64_t);
int64_t nl_file_source_catalog_number(int64_t,int64_t,int64_t,int64_t);
/* Allocation-free full descriptive catalog rendering. Size includes NUL.
 * Size-only is out=NULL/capacity=0. Failure preserves out and *needed; output
 * and needed must be disjoint. No descriptor/document validation is implied. */
bool nl_file_source_catalog_view(char *out,size_t capacity,size_t *needed);
/* Explicit formatter buffers only: CatalogText plus largest nested text buffer.
 * No allocation; excludes scalar locals, compiler call frames and libc. */
size_t nl_file_source_catalog_buffer_bytes(void);
#endif
