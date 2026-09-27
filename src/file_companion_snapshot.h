#ifndef NL_FILE_COMPANION_SNAPSHOT_H
#define NL_FILE_COMPANION_SNAPSHOT_H
#include "nanoisa/file_source_plan.h"
#include <stdint.h>
#define NL_FILE_COMPANION_PATH 4095u
#define NL_FILE_COMPANION_BYTES (UINT64_C(64)*1024*1024)
#define NL_FILE_COMPANION_WORK (UINT64_C(1)<<41)
#define NL_FILE_COMPANION_RETRIES 64u
typedef struct NlFileCompanionSet NlFileCompanionSet;
typedef enum {
 NL_FILE_COMPANION_OK, NL_FILE_COMPANION_INVALID, NL_FILE_COMPANION_LIMIT,
 NL_FILE_COMPANION_MEMORY, NL_FILE_COMPANION_UNRESOLVED, NL_FILE_COMPANION_IO
} NlFileCompanionStatus;
typedef enum {
 NL_FILE_COMPANION_INPUT, NL_FILE_COMPANION_STORAGE, NL_FILE_COMPANION_PARENT,
 NL_FILE_COMPANION_OPEN, NL_FILE_COMPANION_STAT, NL_FILE_COMPANION_READ,
 NL_FILE_COMPANION_CLOSE, NL_FILE_COMPANION_DOCUMENT, NL_FILE_COMPANION_DONE
} NlFileCompanionStage;
typedef struct {
 NlFileSourceText module_path, companion_path, interface_id;
 uint32_t catalog_version, line, column;
} NlFileCompanionRequest;
typedef struct {
 NlFileCompanionStatus status;
 NlFileCompanionStage stage;
 uint32_t request;
 int system_error, close_error;
 /* Heap only; explicit automatic buffers below are separate, not total stack. */
 uint64_t peak_heap_bytes_reserved, work_reserved;
 uint64_t catalog_buffer_bytes, parent_buffer_bytes;
} NlFileCompanionReport;
typedef struct {
 NlFileCompanionRequest request;
 NlFileSourceText original_document, canonical_document, generated_source, catalog_view;
} NlFileCompanionView;
/* I read compiler inputs only; I grant no service/FFI/source authority.
 * The caller supplies canonical declaring-module paths from its real graph,
 * stable ancestors and readable immutable counted storage during this call.
 * Relative companions contain no empty/dot/dot-dot components. Final symlinks
 * and nonregular files refuse. Input providers remain stable during each read;
 * retained successful bytes are independent of subsequent changes/deletion.
 * cJSON hooks remain stable as required by nl_file_binding_prepare.
 * Output is disjoint from inputs. Failure preserves *out; report is by value.
 * No thread-safety or hostile ancestor replacement guarantee is added. */
NlFileCompanionReport nl_file_companions_prepare(const NlFileCompanionRequest *,size_t,
                                                NlFileCompanionSet **out);
void nl_file_companions_free(NlFileCompanionSet *);
size_t nl_file_companions_count(const NlFileCompanionSet *);
/* Successful views borrow counted immutable data until set destruction.
 * Invalid index/null output returns false without changing output. */
bool nl_file_companions_view(const NlFileCompanionSet *,size_t,NlFileCompanionView *);
#endif
