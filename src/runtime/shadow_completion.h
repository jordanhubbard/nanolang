#ifndef NANOLANG_SHADOW_COMPLETION_H
#define NANOLANG_SHADOW_COMPLETION_H
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <time.h>
#include <sys/wait.h>

/* I keep this private record between the two sides of one fork. */
typedef struct {
    uint32_t tag;
    struct timespec finished;
} NlShadowCompletion;
#define NL_SHADOW_COMPLETION_TAG UINT32_C(0x4e534331)
_Static_assert(sizeof(NlShadowCompletion) <= PIPE_BUF,
               "I require one atomic shadow completion record");

static inline bool nl_shadow_completion_valid(const NlShadowCompletion *record,
                                               size_t received,
                                               const struct timespec *start,
                                               int seconds) {
    if (!record || !start || received != sizeof(*record) || seconds <= 0 ||
        record->tag != NL_SHADOW_COMPLETION_TAG || start->tv_sec < 0 ||
        start->tv_nsec < 0 || start->tv_nsec >= 1000000000L ||
        record->finished.tv_sec < start->tv_sec ||
        record->finished.tv_nsec < 0 || record->finished.tv_nsec >= 1000000000L)
        return false;
    if (record->finished.tv_sec == start->tv_sec &&
        record->finished.tv_nsec < start->tv_nsec) return false;
    return true;
}

static inline bool nl_shadow_completion_ontime(const NlShadowCompletion *record,
                                               size_t received,
                                               const struct timespec *start,
                                               int seconds) {
    if (!nl_shadow_completion_valid(record, received, start, seconds)) return false;
    uintmax_t elapsed_seconds = (uintmax_t)record->finished.tv_sec -
                                (uintmax_t)start->tv_sec;
    return elapsed_seconds < (uintmax_t)seconds ||
           (elapsed_seconds == (uintmax_t)seconds &&
            record->finished.tv_nsec < start->tv_nsec);
}

static inline bool nl_shadow_completion_accept(const NlShadowCompletion *record,
                                               size_t received,
                                               const struct timespec *start,
                                               int seconds, bool clock_failed,
                                               bool wait_failed, int status) {
    return !clock_failed && !wait_failed && WIFEXITED(status) &&
           WEXITSTATUS(status) == 0 &&
           nl_shadow_completion_ontime(record, received, start, seconds);
}
#endif
