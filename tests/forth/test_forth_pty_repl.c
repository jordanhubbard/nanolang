/*
 * test_forth_pty_repl.c — interpreter liveness on a PTY, the same shape
 * sdl_forth_ide uses: fork bin/forth --interactive with FORTH_RAW_IO=1.
 * bin/forth is the NanoISA session REPL (bin/nano_forth), not the old
 * NanoLang nl_forth_interpreter.
 *
 * The SDL IDE used to show "Forth interpreter exited" on the first frame
 * because the child called GNU/libedit readline() on the slave, got EOF,
 * and treated an empty line as bye. This test fails if that happens.
 */

#include "pty.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void drain_into(int64_t master_fd, char *acc, size_t acc_size, int timeout_ms) {
    int waited = 0;
    size_t used = strlen(acc);
    while (waited < timeout_ms) {
        const char *chunk = nl_pty_read(master_fd);
        if (chunk != NULL && chunk[0] != '\0') {
            size_t n = strlen(chunk);
            if (used + n >= acc_size) {
                n = acc_size - used - 1;
            }
            memcpy(acc + used, chunk, n);
            used += n;
            acc[used] = '\0';
        }
        usleep(20000);
        waited += 20;
    }
}

static int contains(const char *hay, const char *needle) {
    return hay != NULL && needle != NULL && strstr(hay, needle) != NULL;
}

static const char *forth_path(void) {
    const char *env = getenv("FORTH_BIN");
    if (env != NULL && env[0] != '\0') {
        return env;
    }
    return "bin/forth";
}

static void test_interactive_repl_stays_alive(void) {
    const char *test_name = "PTY Forth REPL stays alive and evaluates 1 2 +";
    const char *prog = forth_path();
    char acc[16384];
    int64_t master;
    int64_t pid;

    if (access(prog, X_OK) != 0) {
        FAIL(test_name, "bin/forth is not executable (build with make forth)");
        return;
    }

    master = nl_pty_open(24, 80);
    if (master < 0) {
        FAIL(test_name, "nl_pty_open failed");
        return;
    }

    pid = nl_pty_fork_exec(master, prog, "--interactive", "FORTH_RAW_IO", "1");
    if (pid < 0) {
        nl_pty_close(master);
        FAIL(test_name, "nl_pty_fork_exec failed");
        return;
    }

    memset(acc, 0, sizeof(acc));
    drain_into(master, acc, sizeof(acc), 1500);

    if (nl_pty_is_alive(pid) == 0) {
        nl_pty_close(master);
        FAIL(test_name, "child exited before the first prompt");
        fprintf(stderr, "PTY output:\n%s\n", acc);
        return;
    }

    if (!contains(acc, "forth>") && !contains(acc, "FIG Forth")) {
        nl_pty_close(master);
        FAIL(test_name, "banner/prompt never appeared");
        fprintf(stderr, "PTY output:\n%s\n", acc);
        return;
    }

    if (nl_pty_write(master, "1 2 + .\n") < 0) {
        nl_pty_close(master);
        FAIL(test_name, "write of 1 2 + . failed");
        return;
    }

    drain_into(master, acc, sizeof(acc), 1500);

    if (nl_pty_is_alive(pid) == 0) {
        nl_pty_close(master);
        FAIL(test_name, "child exited after evaluating a line");
        fprintf(stderr, "PTY output:\n%s\n", acc);
        return;
    }

    if (!contains(acc, "3")) {
        nl_pty_close(master);
        FAIL(test_name, "did not print 3 for 1 2 + .");
        fprintf(stderr, "PTY output:\n%s\n", acc);
        return;
    }

    if (nl_pty_write(master, "\n") < 0) {
        nl_pty_close(master);
        FAIL(test_name, "write of empty line failed");
        return;
    }

    drain_into(master, acc, sizeof(acc), 400);

    if (nl_pty_is_alive(pid) == 0) {
        nl_pty_close(master);
        FAIL(test_name, "empty line quit the REPL");
        fprintf(stderr, "PTY output:\n%s\n", acc);
        return;
    }

    (void)nl_pty_write(master, "bye\n");
    drain_into(master, acc, sizeof(acc), 800);
    nl_pty_close(master);
    PASS(test_name);
}

int main(void) {
    printf("Forth PTY REPL liveness\n");
    test_interactive_repl_stays_alive();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
