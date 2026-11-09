#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static uint64_t monotonic_ns(void) {
    struct timespec value;
    assert(clock_gettime(CLOCK_MONOTONIC, &value) == 0);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

int main(void) {
    uint64_t start = monotonic_ns();
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        const size_t bytes = 64U * 1024U * 1024U;
        volatile unsigned char *memory = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(memory != MAP_FAILED);
        for (size_t i = 0; i < bytes; i += 4096) memory[i] = (unsigned char)i;
        memory[bytes - 1] = 7;
        assert(memory[bytes - 1] == 7);
        _exit(7);
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    assert(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 7);
    uint64_t end = monotonic_ns();
    assert(end >= start);
    printf("{\"pid\":%ld,\"child_pid\":%ld,\"wait_status\":%d,\"start_ns\":%" PRIu64
           ",\"end_ns\":%" PRIu64 "}\n", (long)getpid(), (long)child, status, start, end);
    return 0;
}
