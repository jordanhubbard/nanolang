#ifndef EXAMPLES_IO_H
#define EXAMPLES_IO_H

#include <stdint.h>

/* Flush stdout (non-SDL alternative to sdl_helpers' nl_flush_stdout) */
void nl_examples_flush(void);

/* High-resolution timestamp in milliseconds (monotonic clock) */
int64_t nl_examples_timestamp_ms(void);

/* Return TMPDIR-aware temp directory as a string */
char *nl_examples_tmpdir(void);

#endif /* EXAMPLES_IO_H */
