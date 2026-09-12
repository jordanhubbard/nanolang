#ifndef NANOLANG_SHADOW_RUNNER_H
#define NANOLANG_SHADOW_RUNNER_H

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* I bound execution, not authority. Foreign calls retain the user's privileges. */
static inline int nl_run_shadow_entry(int (*entry)(void), int seconds) {
    fflush(NULL);
    pid_t child = fork();
    if (child < 0) {
        fprintf(stderr, "I could not start shadow execution\n");
        return 1;
    }
    if (child == 0) {
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) _exit(1);
        int result = entry();
        fflush(NULL);
        _exit(result);
    }
    struct timespec start, now, pause = {0, 10000000};
    int status = 0;
    int clock_ok = clock_gettime(CLOCK_MONOTONIC, &start) == 0;
    for (;;) {
        pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR) {
            fprintf(stderr, "I could not wait for shadow execution\n");
            return 1;
        }
        int clock_failed = !clock_ok || clock_gettime(CLOCK_MONOTONIC, &now) != 0;
        int expired = !clock_failed && (now.tv_sec - start.tv_sec > seconds ||
            (now.tv_sec - start.tv_sec == seconds && now.tv_nsec >= start.tv_nsec));
        if (clock_failed || expired) {
            kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            if (clock_failed) fprintf(stderr, "I could not measure the shadow deadline\n");
            else fprintf(stderr, "I stopped shadow execution after %d seconds\n", seconds);
            return 1;
        }
        nanosleep(&pause, NULL);
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "I rejected failed shadow execution\n");
        return 1;
    }
    return 0;
}

#endif
