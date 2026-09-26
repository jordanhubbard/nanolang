#ifndef NL_FILE_SOURCE_INPUT_H
#define NL_FILE_SOURCE_INPUT_H
#include "file_companion_snapshot.h"
/* I copy bounded regular compiler input bytes, never parse or resolve them.
 * Failure preserves both outputs. Success owns a NUL-terminated buffer whose
 * complete counted input contains no NUL and is RFC3629 UTF-8. Free with free().
 * Stable source providers/ancestors, valid terminated path and disjoint outputs
 * are caller preconditions. I close once and retain secondary close failure. */
NlFileCompanionReport nl_file_source_input_read(const char *,size_t,char **,size_t *);
/* Existing or symlink metadata returns1, absent ENOENT returns0, errors -1. */
int64_t nl_file_source_metadata(const char *,int64_t);
#endif
