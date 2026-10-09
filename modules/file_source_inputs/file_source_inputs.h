#ifndef NL_FILE_SOURCE_INPUTS_H
#define NL_FILE_SOURCE_INPUTS_H
#include "nanoisa/file_source_snapshot.h"
#include <stdint.h>
NlFileSourceSnapshots *nl_source_inputs_new(void);
int64_t nl_source_inputs_valid(NlFileSourceSnapshots *);
int64_t nl_source_inputs_open(NlFileSourceSnapshots *, const char *, int64_t,
                            const char *, int64_t);
int64_t nl_source_inputs_count(NlFileSourceSnapshots *);
char *nl_source_inputs_text(NlFileSourceSnapshots *, int64_t, int64_t);
void nl_source_inputs_free(NlFileSourceSnapshots *);
#endif
