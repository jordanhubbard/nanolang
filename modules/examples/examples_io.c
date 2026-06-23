#include "examples_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

/* Flush stdout - essential for JSON diagnostics (line-buffered output) */
void nl_examples_flush(void) {
    fflush(stdout);
}

/* High-resolution monotonic timestamp in milliseconds */
int64_t nl_examples_timestamp_ms(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t t = mach_absolute_time();
    return (int64_t)((t * info.numer / info.denom) / 1000000ULL);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
#endif
}

/* Return TMPDIR-aware temp directory (caller must not free - static or env pointer) */
char *nl_examples_tmpdir(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    /* Return a gc-compatible copy (nanolang strings are gc-managed) */
    size_t len = strlen(tmp);
    char *out = malloc(len + 1);
    if (!out) return "";
    memcpy(out, tmp, len + 1);
    return out;
}
