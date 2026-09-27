#define _POSIX_C_SOURCE 200809L
#include "runtime/shadow_completion.h"
#include <assert.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>

static void actual_record_after_parent_delay(bool partial) {
    int channel[2];
    assert(pipe(channel) == 0);
    struct timespec start;
    assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(channel[0]);
        NlShadowCompletion done = {0};
        done.tag = NL_SHADOW_COMPLETION_TAG;
        if (clock_gettime(CLOCK_MONOTONIC, &done.finished) != 0) _exit(2);
        size_t count = sizeof(done) - (partial ? 1 : 0);
        if (write(channel[1], &done, count) != (ssize_t)count) _exit(3);
        close(channel[1]);
        _exit(0);
    }
    close(channel[1]);
    NlShadowCompletion done = {0};
    ssize_t received = read(channel[0], &done, sizeof(done));
    close(channel[0]);
    assert(received == (ssize_t)(sizeof(done) - (partial ? 1 : 0)));
    /* I delay only this fixture parent, after receiving the child's proof. */
    struct timespec now, pause = {0, 10000000};
    do {
        assert(clock_gettime(CLOCK_MONOTONIC, &now) == 0);
        if (now.tv_sec - start.tv_sec > 1 ||
            (now.tv_sec - start.tv_sec == 1 && now.tv_nsec >= start.tv_nsec)) break;
        nanosleep(&pause, NULL);
    } while (true);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    assert(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(nl_shadow_completion_accept(&done, (size_t)received, &start, 1,
                                       false, false, status) == !partial);
}

int main(void) {
    struct timespec start = {100, 500000000};
    NlShadowCompletion done = {NL_SHADOW_COMPLETION_TAG, {110, 499999999}};
    /* I accept a proven normal exit even when polling has reached its deadline. */
    assert(nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 0));
    done.finished.tv_nsec = 500000000;
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 0));
    done.finished.tv_nsec++;
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 0));
    done.finished = start;
    assert(nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 0));
    done.finished.tv_nsec--;
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 0));
    done.finished = (struct timespec){101, 0};
    assert(nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 0));
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, true, false, 0));
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, true, 0));
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, SIGKILL));
    assert(!nl_shadow_completion_accept(&done, sizeof(done), &start, 10, false, false, 1 << 8));
    assert(!nl_shadow_completion_accept(&done, 0, &start, 10, false, false, 0));
    assert(!nl_shadow_completion_accept(&done, sizeof(done) - 1, &start, 10, false, false, 0));
    assert(!nl_shadow_completion_accept(&done, sizeof(done) + 1, &start, 10, false, false, 0));
    done.tag++;
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    done.tag = NL_SHADOW_COMPLETION_TAG;
    done.finished.tv_nsec = -1;
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    done.finished.tv_nsec = 1000000000L;
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    done.finished = (struct timespec){99, 999999999};
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    done.finished = (struct timespec){110, 0};
    assert(nl_shadow_completion_ontime(&done, sizeof(done), &start, 10));
    done.finished = (struct timespec){111, 0};
    assert(!nl_shadow_completion_ontime(&done, sizeof(done), &start, 10));
    assert(!nl_shadow_completion_valid(NULL, sizeof(done), &start, 10));
    assert(!nl_shadow_completion_valid(&done, sizeof(done), NULL, 10));
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 0));
    start.tv_nsec = -1;
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    start.tv_nsec = 1000000000L;
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    start.tv_sec = -1; start.tv_nsec = 0;
    assert(!nl_shadow_completion_valid(&done, sizeof(done), &start, 10));
    actual_record_after_parent_delay(false);
    actual_record_after_parent_delay(true);
    return 0;
}
