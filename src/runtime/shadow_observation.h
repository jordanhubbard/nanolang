#ifndef NANOLANG_SHADOW_OBSERVATION_H
#define NANOLANG_SHADOW_OBSERVATION_H

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* I observe fixed internal phase names only. My optional writes can perturb
 * timing; they neither change supervision decisions nor cancel either timer. */
static inline bool nl_shadow_observation_enabled(void) {
    return getenv("NANO_SHADOW_PHASE_TRACE") != NULL;
}

static inline void nl_shadow_observe_phase(bool enabled, const char *phase, int value) {
    if (!enabled) return;
    int saved_errno = errno;
    struct timespec now = {0};
    bool measured = clock_gettime(CLOCK_MONOTONIC, &now) == 0;
    fprintf(stderr, "{\"event\":\"cseed-shadow-phase\",\"phase\":\"%s\","
            "\"pid\":%ld,\"clock_ok\":%s,\"monotonic_seconds\":%lld,"
            "\"monotonic_nanoseconds\":%ld,\"value\":%d}\n",
            phase, (long)getpid(), measured ? "true" : "false",
            (long long)now.tv_sec, now.tv_nsec, value);
    fflush(stderr);
    errno = saved_errno;
}

static inline void nl_shadow_observe_status(bool enabled, pid_t child, pid_t waited,
        int status, bool timed_out, bool clock_failed, bool completed, int seconds,
        const struct timespec *start) {
    if (!enabled) return;
    int saved_errno = errno;
    struct timespec now = {0};
    bool measured = clock_gettime(CLOCK_MONOTONIC, &now) == 0;
    bool reaped = waited == child;
    fprintf(stderr, "\n{\"event\":\"cseed-shadow-phase\",\"phase\":\"supervisor-status\","
            "\"pid\":%ld,\"child_pid\":%ld,\"clock_ok\":%s,"
            "\"monotonic_seconds\":%lld,\"monotonic_nanoseconds\":%ld,"
            "\"wait_result\":%ld,\"wait_status\":%d,\"reaped\":%s,"
            "\"signal\":%d,\"exit_code\":%d,\"parent_timeout\":%s,"
            "\"clock_failure\":%s,\"completed\":%s,\"deadline_seconds\":%d,"
            "\"parent_start_clock_ok\":%s,\"parent_start_seconds\":%lld,"
            "\"parent_start_nanoseconds\":%ld}\n",
            (long)getpid(), (long)child, measured ? "true" : "false",
            (long long)now.tv_sec, now.tv_nsec, (long)waited, status,
            reaped ? "true" : "false", reaped && WIFSIGNALED(status) ? WTERMSIG(status) : 0,
            reaped && WIFEXITED(status) ? WEXITSTATUS(status) : -1,
            timed_out ? "true" : "false", clock_failed ? "true" : "false",
            completed ? "true" : "false", seconds, start ? "true" : "false",
            start ? (long long)start->tv_sec : 0, start ? start->tv_nsec : 0);
    fflush(stderr);
    errno = saved_errno;
}

#endif
