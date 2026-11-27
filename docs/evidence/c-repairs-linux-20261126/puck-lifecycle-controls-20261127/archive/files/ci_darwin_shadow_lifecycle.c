#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <mach/mach_time.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/proc.h>
#include <sys/proc_info.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* I observe an owned diagnostic child. I never signal or reap a monitored PID. */
static uint64_t now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) abort();
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static void delay_ms(unsigned milliseconds) {
    struct timespec left = {(time_t)(milliseconds / 1000),
                            (long)(milliseconds % 1000) * 1000000L};
    while (nanosleep(&left, &left) && errno == EINTR) {}
}

static bool transfer(int fd, void *bytes, size_t count, bool writing) {
    char *cursor = bytes;
    while (count) {
        ssize_t amount = writing ? write(fd, cursor, count) : read(fd, cursor, count);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) return false;
        cursor += amount;
        count -= (size_t)amount;
    }
    return true;
}

typedef struct Identity {
    uint64_t start_absolute;
    uint64_t birth_seconds;
    uint64_t birth_microseconds;
} Identity;

/* My two API reads are bracketed observations, not one atomic kernel snapshot. */
static int snapshot(FILE *output, pid_t pid, pid_t parent, Identity *identity) {
    struct rusage_info_v0 usage = {0};
    struct proc_bsdinfo bsd = {0};
    uint64_t before_ns = now_ns();
    uint64_t before_absolute = mach_absolute_time();
    errno = 0;
    int usage_result = proc_pid_rusage(pid, RUSAGE_INFO_V0, (rusage_info_t *)&usage);
    int usage_error = errno;
    errno = 0;
    int bsd_result = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bsd, sizeof bsd);
    int bsd_error = errno;
    uint64_t after_absolute = mach_absolute_time();
    uint64_t after_ns = now_ns();
    fprintf(output, "{\"event\":\"sample\",\"pid\":%d,\"before_ns\":%" PRIu64
        ",\"after_ns\":%" PRIu64 ",\"before_absolute\":%" PRIu64
        ",\"after_absolute\":%" PRIu64 ",\"usage_result\":%d,\"usage_errno\":%d,"
        "\"bsd_result\":%d,\"bsd_errno\":%d,\"parent\":%u,\"status\":%u,"
        "\"zombie\":%s,\"birth_seconds\":%" PRIu64 ",\"birth_microseconds\":%" PRIu64
        ",\"start_absolute\":%" PRIu64 ",\"exit_absolute\":%" PRIu64
        ",\"user_time\":%" PRIu64 ",\"system_time\":%" PRIu64 "}\n",
        pid, before_ns, after_ns, before_absolute, after_absolute,
        usage_result, usage_error, bsd_result, bsd_error, bsd.pbi_ppid, bsd.pbi_status,
        bsd.pbi_status == SZOMB ? "true" : "false",
        (uint64_t)bsd.pbi_start_tvsec, (uint64_t)bsd.pbi_start_tvusec,
        usage.ri_proc_start_abstime, usage.ri_proc_exit_abstime,
        usage.ri_user_time, usage.ri_system_time);
    if (usage_result || bsd_result != (int)sizeof bsd) return 1;
    if (bsd.pbi_ppid != (uint32_t)parent || !usage.ri_proc_start_abstime) return -1;
    if (!identity->start_absolute) {
        identity->start_absolute = usage.ri_proc_start_abstime;
        identity->birth_seconds = bsd.pbi_start_tvsec;
        identity->birth_microseconds = bsd.pbi_start_tvusec;
    } else if (identity->start_absolute != usage.ri_proc_start_abstime ||
        identity->birth_seconds != bsd.pbi_start_tvsec ||
        identity->birth_microseconds != bsd.pbi_start_tvusec) return -1;
    return 0;
}

/* I register before waiting for a flush trigger and leave shadow bodies idle. */
static int observe(pid_t pid, pid_t parent, unsigned milliseconds, FILE *output,
                   int trigger, int release) {
    struct mach_timebase_info timebase;
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || !timebase.denom) return 2;
    uint64_t clock_before = now_ns();
    uint64_t absolute = mach_absolute_time();
    uint64_t clock_after = now_ns();
    fprintf(output, "{\"event\":\"clock\",\"numer\":%u,\"denom\":%u,"
                    "\"before_ns\":%" PRIu64 ",\"absolute\":%" PRIu64
                    ",\"after_ns\":%" PRIu64 "}\n",
                    timebase.numer, timebase.denom, clock_before, absolute, clock_after);
    Identity identity = {0};
    if (snapshot(output, pid, parent, &identity)) return 2;
    int queue = kqueue();
    if (queue < 0) return 2;
    struct kevent change;
    EV_SET(&change, (uintptr_t)pid, EVFILT_PROC, EV_ADD | EV_ENABLE | EV_ONESHOT,
           NOTE_EXIT, 0, NULL);
    if (kevent(queue, &change, 1, NULL, 0, NULL)) { close(queue); return 2; }
    fprintf(output, "{\"event\":\"armed\",\"pid\":%d,\"monotonic_ns\":%" PRIu64 "}\n",
            pid, now_ns());
    fflush(output);
    if (release >= 0) {
        char go = 'g';
        if (!transfer(release, &go, 1, true)) { close(queue); return 2; }
    }
    char command;
    if (!transfer(trigger, &command, 1, false) || command != 'g') {
        fprintf(output, "{\"event\":\"cancelled-before-trigger\"}\n");
        close(queue); return 2;
    }
    uint64_t start = now_ns();
    fprintf(output, "{\"event\":\"trigger\",\"monotonic_ns\":%" PRIu64 "}\n", start);
    int result = 3;
    while (now_ns() - start < (uint64_t)milliseconds * UINT64_C(1000000)) {
        if (snapshot(output, pid, parent, &identity) < 0) { result = 2; break; }
        struct kevent event = {0};
        struct timespec timeout = {0, 1000000L};
        uint64_t before = now_ns();
        int count = kevent(queue, NULL, 0, &event, 1, &timeout);
        int saved = errno;
        uint64_t after = now_ns();
        if (count < 0 && saved == EINTR) continue;
        if (count < 0) { result = 2; break; }
        if (!count) continue;
        fprintf(output, "{\"event\":\"process-event\",\"before_ns\":%" PRIu64
            ",\"after_ns\":%" PRIu64 ",\"ident\":%" PRIuPTR ",\"filter\":%d,"
            "\"flags\":%u,\"fflags\":%u,\"data\":%" PRId64 "}\n",
            before, after, event.ident, (int)event.filter, (unsigned)event.flags, event.fflags,
            (int64_t)event.data);
        if (event.ident != (uintptr_t)pid || event.filter != EVFILT_PROC ||
            (event.flags & EV_ERROR) || !(event.fflags & NOTE_EXIT)) { result = 2; break; }
        (void)snapshot(output, pid, parent, &identity);
        result = 0;
        break;
    }
    fprintf(output, "{\"event\":\"end\",\"result\":%d,\"monotonic_ns\":%" PRIu64 "}\n",
            result, now_ns());
    fflush(output);
    close(queue);
    return result;
}

static int control(const char *kind, FILE *output) {
    bool blocked = !strcmp(kind, "blocked"), late = !strcmp(kind, "late");
    if (!blocked && !late && strcmp(kind, "success")) return 2;
    int marker[2], trigger[2], release[2];
    if (pipe(marker) || pipe(trigger) || pipe(release)) return 2;
    uint64_t origin = now_ns();
    pid_t child = fork();
    if (child < 0) return 2;
    if (!child) {
        close(marker[0]); close(trigger[0]); close(release[1]);
        char go;
        if (!transfer(release[0], &go, 1, false) || go != 'g') _exit(2);
        close(release[0]);
        delay_ms(20);
        uint64_t clocks[2];
        clocks[0] = now_ns();
        clocks[1] = mach_absolute_time();
        if (!transfer(marker[1], clocks, sizeof clocks, true)) _exit(2);
        close(marker[1]);
        go = 'g';
        if (!transfer(trigger[1], &go, 1, true)) _exit(2);
        close(trigger[1]);
        if (blocked) { for (;;) pause(); }
        if (late) delay_ms(150);
        _exit(0);
    }
    close(marker[1]); close(trigger[1]); close(release[0]);
    int observed = observe(child, getpid(), blocked ? 100 : 1000, output,
                           trigger[0], release[1]);
    close(trigger[0]); close(release[1]);
    if (blocked || observed) kill(child, SIGKILL);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    uint64_t clocks[2] = {0};
    bool marked = transfer(marker[0], clocks, sizeof clocks, false);
    close(marker[0]);
    bool passed = waited == child && marked &&
        (blocked ? observed == 3 && WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL :
                   observed == 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    fprintf(output, "{\"event\":\"control\",\"kind\":\"%s\",\"origin_ns\":%" PRIu64
        ",\"marker_ns\":%" PRIu64 ",\"marker_absolute\":%" PRIu64
        ",\"pid\":%d,\"wait_status\":%d,\"passed\":%s}\n", kind, origin,
        clocks[0], clocks[1], child, status, passed ? "true" : "false");
    fflush(output);
    return passed ? 0 : 2;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--control")) {
        FILE *output = fopen(argv[3], "w");
        if (!output) return 2;
        int result = control(argv[2], output);
        if (fclose(output)) return 2;
        return result;
    }
    if (argc != 5) return 2;
    char *end = NULL;
    long pid = strtol(argv[1], &end, 10);
    if (!end || *end || pid <= 1 || pid > INT32_MAX) return 2;
    long parent = strtol(argv[2], &end, 10);
    if (*end || parent <= 1 || parent > INT32_MAX) return 2;
    long limit = strtol(argv[3], &end, 10);
    if (*end || limit < 1 || limit > 30000) return 2;
    FILE *output = fopen(argv[4], "w");
    if (!output) return 2;
    int result = observe((pid_t)pid, (pid_t)parent, (unsigned)limit, output, STDIN_FILENO, -1);
    if (fclose(output)) return 2;
    return result;
}
