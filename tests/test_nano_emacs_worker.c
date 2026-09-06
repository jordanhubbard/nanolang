/*
 * Isolated SDL-editor walker: length-prefixed pipe to bin/nano_emacs_worker.
 * The frame must not dlopen the interpreter. C-x C-e stays walker eval.
 * Freeze is a nano_vm grandchild. I do not claim GNU Emacs compatibility.
 */

#include "../modules/nano_eval/nano_eval.h"
#include "../modules/nano_eval/nano_eval_ipc.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static const char *k_add_src =
    "fn add(a: int, b: int) -> int {\n"
    "    return (+ a b)\n"
    "}\n"
    "\n"
    "shadow add {\n"
    "    assert (== (add 1 2) 3)\n"
    "}\n";

static const char *k_ed_src =
    "fn shout() -> void {\n"
    "    (ed_message \"x\")\n"
    "}\n";

struct KillArg {
    pid_t pid;
};

static void *kill_later(void *arg) {
    struct KillArg *k = (struct KillArg *)arg;
    usleep(300000);
    if (k->pid > 1) {
        kill(k->pid, SIGKILL);
    }
    return NULL;
}

static void test_extract_defun(void) {
    const char *name = "extract_defun takes the fn at point";
    char out[4096];
    if (nano_eval_extract_defun(k_add_src, 20, out, sizeof(out)) != 0) {
        FAIL(name, "extract failed");
        return;
    }
    if (strstr(out, "fn add") == NULL || strstr(out, "shadow add") == NULL) {
        FAIL(name, out);
        return;
    }
    PASS(name);
}

static void test_reject_ed(void) {
    const char *name = "source_has_ed detects ed_message";
    if (!nano_eval_source_has_ed(k_ed_src) || nano_eval_source_has_ed(k_add_src)) {
        FAIL(name, "ed_ scan wrong");
        return;
    }
    PASS(name);
}

static void test_worker_eval(void) {
    const char *name = "worker eval (+ 1 2) is 3";
    int64_t session = nano_eval_create();
    const char *out;
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0 (build bin/nano_emacs_worker)");
        return;
    }
    if (nano_eval_worker_pid(session) <= 1) {
        FAIL(name, "worker pid missing");
        nano_eval_destroy(session);
        return;
    }
    out = nano_eval_string(session, "(+ 1 2)");
    if (out == NULL || strstr(out, "3") == NULL) {
        FAIL(name, out ? out : "NULL");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_defn_survives(void) {
    const char *name = "worker defn survives a second eval";
    int64_t session = nano_eval_create();
    const char *err;
    const char *out;
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    (void)nano_eval_string(session, "fn triple(n: int) -> int { return (* n 3) }");
    err = nano_eval_error(session);
    if (err != NULL && err[0] != '\0') {
        FAIL(name, err);
        nano_eval_destroy(session);
        return;
    }
    out = nano_eval_string(session, "(triple 7)");
    if (out == NULL || strstr(out, "21") == NULL) {
        FAIL(name, out ? out : "NULL");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_ed_message_queues(void) {
    const char *name = "worker ed_message queues a command";
    int64_t session = nano_eval_create();
    const char *err;
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    (void)nano_eval_string(session, "(ed_message \"hello from nano\")");
    err = nano_eval_error(session);
    if (err != NULL && err[0] != '\0') {
        FAIL(name, err);
        nano_eval_destroy(session);
        return;
    }
    if (nano_eval_cmd_count(session) < 1) {
        FAIL(name, "no queued command");
        nano_eval_destroy(session);
        return;
    }
    if (nano_eval_cmd_kind(session, 0) != NANO_EVAL_CMD_MESSAGE) {
        FAIL(name, "wrong kind");
        nano_eval_destroy(session);
        return;
    }
    if (strcmp(nano_eval_cmd_arg(session, 0), "hello from nano") != 0) {
        FAIL(name, nano_eval_cmd_arg(session, 0));
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_kill_mid_eval(void) {
    const char *name = "kill worker mid-eval; parent keeps buffers";
    int64_t session = nano_eval_create();
    pthread_t th;
    struct KillArg arg;
    const char *err;
    const char *buf;
    const char *out;
    int64_t pid;
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    nano_eval_bind_buffer(session, "kept-buffer", 0);
    pid = nano_eval_worker_pid(session);
    arg.pid = (pid_t)pid;
    if (pthread_create(&th, NULL, kill_later, &arg) != 0) {
        FAIL(name, "pthread_create");
        nano_eval_destroy(session);
        return;
    }
    (void)nano_eval_string(session, "__hang__");
    pthread_join(th, NULL);
    err = nano_eval_error(session);
    if (err == NULL || strstr(err, "crashed") == NULL) {
        FAIL(name, err ? err : "missing crash echo");
        nano_eval_destroy(session);
        return;
    }
    buf = nano_eval_buffer(session);
    if (buf == NULL || strcmp(buf, "kept-buffer") != 0) {
        FAIL(name, buf ? buf : "NULL buffer");
        nano_eval_destroy(session);
        return;
    }
    out = nano_eval_string(session, "(+ 2 3)");
    if (out == NULL || strstr(out, "5") == NULL) {
        FAIL(name, out ? out : "restart eval failed");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_freeze_pure(void) {
    const char *name = "freeze-defun of a pure fn";
    const char *out = nano_eval_freeze(k_add_src, 20);
    if (out == NULL) {
        FAIL(name, "NULL");
        return;
    }
    if (strstr(out, "cannot") != NULL || strstr(out, "refuse") != NULL ||
        strstr(out, "failed") != NULL || strstr(out, "timed out") != NULL) {
        FAIL(name, out);
        return;
    }
    PASS(name);
}

static void test_freeze_rejects_ed(void) {
    const char *name = "freeze-defun refuses ed_*";
    const char *out = nano_eval_freeze(k_ed_src, 10);
    if (out == NULL || strstr(out, "ed_*") == NULL) {
        FAIL(name, out ? out : "NULL");
        return;
    }
    PASS(name);
}

static void test_freeze_does_not_kill_frame(void) {
    const char *name = "freeze failure does not kill the frame";
    int64_t session;
    const char *out;
    (void)nano_eval_freeze(k_ed_src, 10);
    session = nano_eval_create();
    if (session == 0) {
        FAIL(name, "frame could not create a walker after freeze");
        return;
    }
    out = nano_eval_string(session, "(+ 8 1)");
    if (out == NULL || strstr(out, "9") == NULL) {
        FAIL(name, out ? out : "NULL");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

int main(void) {
    printf("nano_emacs_worker isolation tests\n");
    test_extract_defun();
    test_reject_ed();
    test_worker_eval();
    test_defn_survives();
    test_ed_message_queues();
    test_kill_mid_eval();
    test_freeze_pure();
    test_freeze_rejects_ed();
    test_freeze_does_not_kill_frame();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
