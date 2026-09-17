#ifndef NANOLANG_SHADOW_TIMEOUT_H
#define NANOLANG_SHADOW_TIMEOUT_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#define NL_SHADOW_TIMEOUT_DEFAULT_SECONDS 60
#else
#define NL_SHADOW_TIMEOUT_DEFAULT_SECONDS 10
#endif

/* Instrumented test jobs may request more time, but never unbounded execution. */
static inline int nl_shadow_timeout_seconds(int default_seconds) {
    const char *value = getenv("NANO_SHADOW_TIMEOUT_SECONDS");
    if (!value) return default_seconds;
    unsigned seconds = 0;
    const char *cursor = value;
    while (*cursor >= '0' && *cursor <= '9') {
        seconds = seconds * 10 + (unsigned)(*cursor++ - '0');
        if (seconds > 300) break;
    }
    if (cursor == value || *cursor || seconds == 0 || seconds > 300) {
        fprintf(stderr, "I require NANO_SHADOW_TIMEOUT_SECONDS to be an integer from 1 to 300.\n");
        return -1;
    }
    return (int)seconds;
}

#endif
