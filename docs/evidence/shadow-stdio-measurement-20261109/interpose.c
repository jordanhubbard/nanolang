/* I observe real descriptor calls; I preserve their arguments and return values. */
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mach-o/dyld.h>

static int monitored, child, reported;
static unsigned long long calls[4], nanos[4];
static int valid_clock = 1;
static uint64_t stamp(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) { valid_clock = 0; return 0; }
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}
static void report(const char *point, int status) {
    if (!monitored || reported) return;
    reported = 1;
    char text[2048];
    int length = snprintf(text, sizeof text,
        "{\"tool\":\"nanolang-shadow-stdio-probe\",\"point\":\"%s\",\"pid\":%d,\"child\":%d,\"status\":%d,\"valid_clock\":%d,"
        "\"open_devnull\":%llu,\"dup\":%llu,\"dup2\":%llu,\"close\":%llu,"
        "\"open_devnull_ns\":%llu,\"dup_ns\":%llu,\"dup2_ns\":%llu,\"close_ns\":%llu}\n",
        point, (int)getpid(), child, status, valid_clock,
        calls[0], calls[1], calls[2], calls[3], nanos[0], nanos[1], nanos[2], nanos[3]);
    if (length > 0 && (size_t)length < sizeof text) (void)write(STDERR_FILENO, text, (size_t)length);
}
__attribute__((constructor)) static void begin(void) {
    char path[4096]; uint32_t size = sizeof path;
    if (_NSGetExecutablePath(path, &size)) return;
    const char *name = strrchr(path, '/');
    monitored = strcmp(name ? name + 1 : path, "nanoc_c") == 0;
}
__attribute__((destructor)) static void end(void) { report("exit", 0); }
static int observed_open(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list args; va_start(args, flags); mode = (mode_t)va_arg(args, int); va_end(args); }
    int selected = monitored && path && strcmp(path, "/dev/null") == 0;
    int incoming_errno = errno;
    uint64_t start = selected ? stamp() : 0;
    errno = incoming_errno;
    int result = flags & O_CREAT ? open(path, flags, mode) : open(path, flags);
    int result_errno = errno;
    if (selected) { calls[0]++; nanos[0] += stamp() - start; }
    errno = result_errno;
    return result;
}
static int observed_dup(int fd) {
    int incoming_errno = errno;
    uint64_t start = monitored ? stamp() : 0;
    errno = incoming_errno;
    int result = dup(fd);
    int result_errno = errno;
    if (monitored) { calls[1]++; nanos[1] += stamp() - start; }
    errno = result_errno;
    return result;
}
static int observed_dup2(int from, int to) {
    int incoming_errno = errno;
    uint64_t start = monitored ? stamp() : 0;
    errno = incoming_errno;
    int result = dup2(from, to);
    int result_errno = errno;
    if (monitored) { calls[2]++; nanos[2] += stamp() - start; }
    errno = result_errno;
    return result;
}
static int observed_close(int fd) {
    int incoming_errno = errno;
    uint64_t start = monitored ? stamp() : 0;
    errno = incoming_errno;
    int result = close(fd);
    int result_errno = errno;
    if (monitored) { calls[3]++; nanos[3] += stamp() - start; }
    errno = result_errno;
    return result;
}
static pid_t observed_fork(void) {
    pid_t result = fork();
    if (monitored && result == 0) { memset(calls, 0, sizeof calls); memset(nanos, 0, sizeof nanos); child = 1; reported = 0; valid_clock = 1; }
    return result;
}
__attribute__((noreturn)) static void observed_exit(int status) { report("_exit", status); _exit(status); }
#define INTERPOSE(replacement, original) \
    __attribute__((used)) static struct { const void *replacement; const void *original; } \
    pair_##original __attribute__((section("__DATA,__interpose"))) = { (const void *)(replacement), (const void *)(original) }
INTERPOSE(observed_open, open);
INTERPOSE(observed_dup, dup);
INTERPOSE(observed_dup2, dup2);
INTERPOSE(observed_close, close);
INTERPOSE(observed_fork, fork);
INTERPOSE(observed_exit, _exit);
