/*
 * test_nano_eval.c — persistent tree-walker session used by the SDL editor.
 *
 * C-x C-e must keep defn across evals and must queue ed_message instead of
 * touching SDL from inside the walker.
 */

#include "../modules/nano_eval/nano_eval.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_eval_add(void) {
    const char *name = "eval_string of (+ 1 2) is 3";
    int64_t session = nano_eval_create();
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    const char *out = nano_eval_string(session, "(+ 1 2)");
    if (out == NULL || strstr(out, "3") == NULL) {
        FAIL(name, out ? out : "NULL");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_defn_survives(void) {
    const char *name = "defn survives a second eval";
    int64_t session = nano_eval_create();
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    (void)nano_eval_string(session, "fn triple(n: int) -> int { return (* n 3) }");
    const char *err = nano_eval_error(session);
    if (err != NULL && err[0] != '\0') {
        FAIL(name, err);
        nano_eval_destroy(session);
        return;
    }
    const char *out = nano_eval_string(session, "(triple 7)");
    if (out == NULL || strstr(out, "21") == NULL) {
        FAIL(name, out ? out : "NULL");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_ed_message_queues(void) {
    const char *name = "ed_message queues a command";
    int64_t session = nano_eval_create();
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    (void)nano_eval_string(session, "(ed_message \"hello from nano\")");
    const char *err = nano_eval_error(session);
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
        FAIL(name, "command is not MESSAGE");
        nano_eval_destroy(session);
        return;
    }
    const char *arg = nano_eval_cmd_arg(session, 0);
    if (arg == NULL || strcmp(arg, "hello from nano") != 0) {
        FAIL(name, arg ? arg : "NULL arg");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_cmd_clear(session);
    if (nano_eval_cmd_count(session) != 0) {
        FAIL(name, "clear left commands");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

static void test_insert_updates_bound_buffer(void) {
    const char *name = "ed_insert mutates the bound buffer";
    int64_t session = nano_eval_create();
    if (session == 0) {
        FAIL(name, "nano_eval_create returned 0");
        return;
    }
    nano_eval_bind_buffer(session, "ab", 2);
    (void)nano_eval_string(session, "(ed_insert \"c\")");
    const char *err = nano_eval_error(session);
    if (err != NULL && err[0] != '\0') {
        FAIL(name, err);
        nano_eval_destroy(session);
        return;
    }
    const char *buf = nano_eval_buffer(session);
    if (buf == NULL || strcmp(buf, "abc") != 0) {
        FAIL(name, buf ? buf : "NULL");
        nano_eval_destroy(session);
        return;
    }
    if (nano_eval_point(session) != 3) {
        FAIL(name, "point was not advanced");
        nano_eval_destroy(session);
        return;
    }
    nano_eval_destroy(session);
    PASS(name);
}

int main(void) {
    printf("Nano eval session\n");
    test_eval_add();
    test_defn_survives();
    test_ed_message_queues();
    test_insert_updates_bound_buffer();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
