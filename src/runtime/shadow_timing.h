#ifndef NANOLANG_SHADOW_TIMING_H
#define NANOLANG_SHADOW_TIMING_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <stdint.h>

static uint64_t nl_search_calls, nl_search_completed, nl_search_active;
static uint64_t nl_search_wall_ns, nl_search_cpu_ns, nl_search_invalid;
static int nl_search_overflow;

typedef struct {
    int active, clocks_ok;
    uint64_t wall_ns, cpu_ns;
} NlSearchTiming;

static inline int nl_shadow_timing_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("NANO_SHADOW_TIMING");
        enabled = value && !strcmp(value, "1");
    }
    return enabled;
}

static inline int nl_search_clock(clockid_t clock, uint64_t *out) {
    struct timespec value;
    if (clock_gettime(clock, &value) || value.tv_sec < 0 ||
        value.tv_nsec < 0 || value.tv_nsec >= 1000000000L ||
        (uint64_t)value.tv_sec > (UINT64_MAX - (uint64_t)value.tv_nsec) / 1000000000ULL)
        return 0;
    *out = (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
    return 1;
}

static inline void nl_search_add(uint64_t *target, uint64_t amount) {
    if (amount > UINT64_MAX - *target) nl_search_overflow = 1;
    else *target += amount;
}

static inline NlSearchTiming nl_search_begin(const char *registered_name) {
    int saved_errno = errno;
    NlSearchTiming stamp = {0};
    if (nl_shadow_timing_enabled() && registered_name &&
        !strcmp(registered_name, "transp_str_index_of")) {
        if (nl_search_active == UINT64_MAX) nl_search_overflow = 1;
        else {
            stamp.active = 1;
            ++nl_search_active;
            nl_search_add(&nl_search_calls, 1);
            stamp.clocks_ok = nl_search_clock(CLOCK_MONOTONIC, &stamp.wall_ns) &&
                nl_search_clock(CLOCK_PROCESS_CPUTIME_ID, &stamp.cpu_ns);
        }
    }
    errno = saved_errno;
    return stamp;
}

static inline void nl_search_end(NlSearchTiming stamp) {
    if (!stamp.active) return;
    int saved_errno = errno;
    uint64_t wall = 0, cpu = 0;
    int valid = stamp.clocks_ok && nl_search_clock(CLOCK_MONOTONIC, &wall) &&
        nl_search_clock(CLOCK_PROCESS_CPUTIME_ID, &cpu) &&
        wall >= stamp.wall_ns && cpu >= stamp.cpu_ns;
    --nl_search_active;
    nl_search_add(&nl_search_completed, 1);
    if (valid) {
        nl_search_add(&nl_search_wall_ns, wall - stamp.wall_ns);
        nl_search_add(&nl_search_cpu_ns, cpu - stamp.cpu_ns);
    } else nl_search_add(&nl_search_invalid, 1);
    errno = saved_errno;
}

/* I emit diagnostic metadata only; caller-supplied labels are fixed literals. */
static inline void nl_shadow_timing(const char *phase, int source, int item,
                                    int ordinal, int children, int status) {
    static unsigned records = 0;
    static int stopped = 0;
    int saved_errno = errno;
    if (!nl_shadow_timing_enabled() || stopped) { errno = saved_errno; return; }
    if (++records == 4096) { phase = "record_limit"; stopped = 1; }
    struct timespec wall = {0};
    struct rusage cpu = {0};
    int wall_ok = clock_gettime(CLOCK_MONOTONIC, &wall) == 0;
    int cpu_ok = getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &cpu) == 0;
    char line[1024];
    int length = snprintf(line, sizeof line,
        "\nNANO_SHADOW_TIMING {\"phase\":\"%s\",\"source\":%d,\"item\":%d,"
        "\"ordinal\":%d,\"children\":%d,\"status\":%d,\"wall_ok\":%d,"
        "\"wall_sec\":%lld,\"wall_nsec\":%ld,\"cpu_ok\":%d,"
        "\"user_sec\":%lld,\"user_usec\":%ld,\"sys_sec\":%lld,\"sys_usec\":%ld,"
        "\"search_calls\":%llu,\"search_completed\":%llu,\"search_active\":%llu,"
        "\"search_wall_ns\":%llu,\"search_cpu_ns\":%llu,\"search_invalid\":%llu,"
        "\"search_overflow\":%d}\n",
        phase, source, item, ordinal, children, status, wall_ok,
        (long long)wall.tv_sec, (long)wall.tv_nsec, cpu_ok,
        (long long)cpu.ru_utime.tv_sec, (long)cpu.ru_utime.tv_usec,
        (long long)cpu.ru_stime.tv_sec, (long)cpu.ru_stime.tv_usec,
        (unsigned long long)nl_search_calls, (unsigned long long)nl_search_completed,
        (unsigned long long)nl_search_active, (unsigned long long)nl_search_wall_ns,
        (unsigned long long)nl_search_cpu_ns, (unsigned long long)nl_search_invalid,
        nl_search_overflow);
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
