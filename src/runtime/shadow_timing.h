#ifndef NANOLANG_SHADOW_TIMING_H
#define NANOLANG_SHADOW_TIMING_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

/* I emit diagnostic metadata only; caller-supplied labels are fixed literals. */
static inline void nl_shadow_timing(const char *phase, int source, int item,
                                    int ordinal, int children, int status) {
    static int enabled = -1;
    static unsigned records = 0;
    static int stopped = 0;
    int saved_errno = errno;
    if (enabled < 0) {
        const char *value = getenv("NANO_SHADOW_TIMING");
        enabled = value && !strcmp(value, "1");
    }
    if (!enabled || stopped) { errno = saved_errno; return; }
    if (++records == 4096) { phase = "record_limit"; stopped = 1; }
    struct timespec wall = {0};
    struct rusage cpu = {0};
    int wall_ok = clock_gettime(CLOCK_MONOTONIC, &wall) == 0;
    int cpu_ok = getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &cpu) == 0;
    char line[512];
    int length = snprintf(line, sizeof line,
        "\nNANO_SHADOW_TIMING {\"phase\":\"%s\",\"source\":%d,\"item\":%d,"
        "\"ordinal\":%d,\"children\":%d,\"status\":%d,\"wall_ok\":%d,"
        "\"wall_sec\":%lld,\"wall_nsec\":%ld,\"cpu_ok\":%d,"
        "\"user_sec\":%lld,\"user_usec\":%ld,\"sys_sec\":%lld,\"sys_usec\":%ld}\n",
        phase, source, item, ordinal, children, status, wall_ok,
        (long long)wall.tv_sec, (long)wall.tv_nsec, cpu_ok,
        (long long)cpu.ru_utime.tv_sec, (long)cpu.ru_utime.tv_usec,
        (long long)cpu.ru_stime.tv_sec, (long)cpu.ru_stime.tv_usec);
    if (length < 0 || (size_t)length >= sizeof line) stopped = 1;
    else {
        size_t offset = 0;
        for (unsigned attempt = 0; offset < (size_t)length && attempt < 4; ++attempt) {
            ssize_t written = write(STDERR_FILENO, line + offset, (size_t)length - offset);
            if (written > 0) offset += (size_t)written;
            else if (written < 0 && errno == EINTR) continue;
            else break;
        }
        if (offset != (size_t)length) stopped = 1;
    }
    errno = saved_errno;
}
#endif
