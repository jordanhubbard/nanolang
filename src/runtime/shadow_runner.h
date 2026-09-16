#ifndef NANOLANG_SHADOW_RUNNER_H
#define NANOLANG_SHADOW_RUNNER_H

#include "shadow_timeout.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* I bound execution, not authority. Foreign calls retain the user's privileges. */
static inline int nl_run_shadow_entry(int (*entry)(void), int seconds) {
    seconds = nl_shadow_timeout_seconds(seconds);
    if (seconds < 1) return 1;
    int completion[2];
    if (pipe(completion) != 0) {
        fprintf(stderr, "I could not create the shadow completion channel\n");
        return 1;
    }
    if (fcntl(completion[0], F_SETFL, O_NONBLOCK) < 0 ||
        fcntl(completion[0], F_SETFD, FD_CLOEXEC) < 0 ||
        fcntl(completion[1], F_SETFD, FD_CLOEXEC) < 0) {
        close(completion[0]);
        close(completion[1]);
        fprintf(stderr, "I could not configure the shadow completion channel\n");
        return 1;
    }
    fflush(NULL);
    pid_t child = fork();
    if (child < 0) {
        close(completion[0]);
        close(completion[1]);
        fprintf(stderr, "I could not start shadow execution\n");
        return 1;
    }
    if (child == 0) {
        close(completion[0]);
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) _exit(1);
        int result = entry();
        unsigned char done = 1;
        if (result == 0 && write(completion[1], &done, 1) != 1) result = 1;
        close(completion[1]);
        fflush(NULL);
        _exit(result);
    }
    close(completion[1]);
    struct timespec start, now, pause = {0, 10000000};
    int status = 0;
    int clock_ok = clock_gettime(CLOCK_MONOTONIC, &start) == 0;
    for (;;) {
        pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR) {
            close(completion[0]);
            fprintf(stderr, "I could not wait for shadow execution\n");
            return 1;
        }
        int clock_failed = !clock_ok || clock_gettime(CLOCK_MONOTONIC, &now) != 0;
        int expired = !clock_failed && (now.tv_sec - start.tv_sec > seconds ||
            (now.tv_sec - start.tv_sec == seconds && now.tv_nsec >= start.tv_nsec));
        if (clock_failed || expired) {
            kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            close(completion[0]);
            if (clock_failed) fprintf(stderr, "I could not measure the shadow deadline\n");
            else fprintf(stderr, "I stopped shadow execution after %d seconds\n", seconds);
            return 1;
        }
        nanosleep(&pause, NULL);
    }
    unsigned char done = 0;
    int completed = read(completion[0], &done, 1) == 1 && done == 1;
    close(completion[0]);
    if (!completed || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "I rejected failed shadow execution\n");
        return 1;
    }
    return 0;
}

#endif
