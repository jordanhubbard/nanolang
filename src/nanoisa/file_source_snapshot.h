#ifndef NL_FILE_SOURCE_SNAPSHOT_H
#define NL_FILE_SOURCE_SNAPSHOT_H
#include "nsi_file_binding.h"
#define NL_FILE_SOURCE_SNAPSHOT_LIMIT 16u
#define NL_FILE_SOURCE_SNAPSHOT_BUDGET (64u * 1024u * 1024u)
typedef struct NlFileSourceSnapshots NlFileSourceSnapshots;
/* I own immutable source companions, not service authority. Origins are
 * retained absolute source paths. Ancestors remain stable during acquisition.
 * Counted input spans are readable and disjoint from output/context storage.
 * Failures preserve output arguments and previously acquired snapshots. */
NlFileBindingStatus nl_file_source_snapshots_new(NlFileSourceSnapshots **out);
void nl_file_source_snapshots_free(NlFileSourceSnapshots *);
NlFileBindingStatus nl_file_source_snapshot_open(NlFileSourceSnapshots *,
 const char *origin,size_t origin_size,const char *relative,size_t relative_size,size_t *index);
size_t nl_file_source_snapshot_count(const NlFileSourceSnapshots *);
size_t nl_file_source_snapshot_storage(const NlFileSourceSnapshots *);
/* I bound project-requested heap, including failed admitted attempts. Stack,
 * allocator overhead and libc internals are outside this bound. */
size_t nl_file_source_snapshot_peak_bound(const NlFileSourceSnapshots *);
/* I borrow counted views until the owning context is destroyed. Invalid
 * queries return NULL and preserve size. Kind: 0 path, 1 exact read bytes,
 * 2 canonical strict interface, 3 generated binding source. */
const unsigned char *nl_file_source_snapshot_bytes(const NlFileSourceSnapshots *,size_t index,unsigned kind,size_t *size);
#endif
