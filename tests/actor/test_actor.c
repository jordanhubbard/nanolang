/*
 * Nano Actor laboratory tests (4.6).
 */

#include "actor/actor.h"
#include "nanoisa/frontend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int g_argc = 0;
char **g_argv = NULL;

static int g_pass;
static int g_fail;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void expect_i64(const char *name, const char *src, int64_t want) {
    char err[NL_ACTOR_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_actor_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, err[0] ? err : "eval failed");
        return;
    }
    if (got != want) {
        char buf[128];
        snprintf(buf, sizeof buf, "got %lld want %lld",
                 (long long)got, (long long)want);
        FAIL(name, buf);
        return;
    }
    PASS(name);
}

static void expect_fail(const char *name, const char *src, const char *needle) {
    char err[NL_ACTOR_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (nl_actor_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, "accepted source that must fail closed");
        return;
    }
    if (needle && !strstr(err, needle)) {
        FAIL(name, err[0] ? err : "wrong error");
        return;
    }
    PASS(name);
}

static const char *k_echo =
    "message Ping of int\n"
    "message Pong of int\n"
    "message Boom\n"
    "message Down of int\n"
    "message Timeout\n"
    "actor Echo {\n"
    "  receive\n"
    "    | Ping n => reply (Pong n)\n"
    "    | Boom => crash\n"
    "}\n";

static const char *k_counter =
    "message Inc\n"
    "message Get\n"
    "message Count of int\n"
    "actor Counter {\n"
    "  state 0\n"
    "  receive\n"
    "    | Inc => become (state + 1)\n"
    "    | Get => reply (Count state)\n"
    "}\n";

static void test_ping(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  e = spawn Echo\n"
             "  send e (Ping 41)\n"
             "  recv\n"
             "    | Pong n => n\n"
             "}\n", k_echo);
    expect_i64("ping", src, 41);
}

static void test_mailbox_order(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  e = spawn Echo\n"
             "  send e (Ping 1)\n"
             "  send e (Ping 2)\n"
             "  send e (Ping 3)\n"
             "  a = recv | Pong n => n\n"
             "  a\n"
             "}\n", k_echo);
    expect_i64("mailbox order", src, 1);
}

static void test_become(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = spawn Counter\n"
             "  send c Inc\n"
             "  send c Inc\n"
             "  send c Get\n"
             "  recv | Count n => n\n"
             "}\n", k_counter);
    expect_i64("become", src, 2);
}

static void test_isolation(void) {
    char src[1536];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  a = spawn Echo\n"
             "  b = spawn Echo\n"
             "  send b Boom\n"
             "  send a (Ping 7)\n"
             "  recv | Pong n => n\n"
             "}\n", k_echo);
    expect_i64("crash containment", src, 7);
}

static void test_monitor(void) {
    char src[1536];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  e = spawn Echo\n"
             "  monitor e\n"
             "  send e Boom\n"
             "  recv | Down n => n\n"
             "}\n", k_echo);
    expect_i64("monitor", src, 1);
}

static void test_link(void) {
    char src[1536];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  a = spawn Echo\n"
             "  b = spawn Echo\n"
             "  link a\n"
             "  send b Boom\n"
             "  send a (Ping 1)\n"
             "  recv | Pong n => n\n"
             "}\n", k_echo);
    expect_i64("link does not kill unlinked", src, 1);
}

static void test_link_kill(void) {
    char src[1536];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  a = spawn Echo\n"
             "  link a\n"
             "  send a Boom\n"
             "  recv after 0\n"
             "    | Pong n => n\n"
             "    | Timeout => 0\n"
             "}\n", k_echo);
    expect_i64("link crash", src, 0);
}

static void test_supervise(void) {
    char src[2048];
    snprintf(src, sizeof src,
             "%s"
             "supervise one_for_one {\n"
             "  child echo = Echo\n"
             "}\n"
             "main {\n"
             "  e = spawn echo\n"
             "  send e Boom\n"
             "  send e (Ping 9)\n"
             "  recv | Pong n => n\n"
             "}\n", k_echo);
    expect_i64("supervise restart", src, 9);
}

static void test_timeout(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  e = spawn Echo\n"
             "  recv after 0\n"
             "    | Pong n => n\n"
             "    | Timeout => 3\n"
             "}\n", k_echo);
    expect_i64("deadline after 0", src, 3);
}

static void test_cancel(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  e = spawn Echo\n"
             "  cancel e\n"
             "  send e (Ping 1)\n"
             "  recv after 0\n"
             "    | Pong n => n\n"
             "    | Timeout => 0\n"
             "}\n", k_echo);
    expect_fail("cancel send", src, "cancel");
}

static void test_replace(void) {
    char src[2048];
    snprintf(src, sizeof src,
             "%s"
             "actor Echo2 {\n"
             "  receive\n"
             "    | Ping n => reply (Pong (n + 1))\n"
             "    | Boom => crash\n"
             "}\n"
             "main {\n"
             "  e = spawn Echo\n"
             "  replace e Echo2\n"
             "  send e (Ping 4)\n"
             "  recv | Pong n => n\n"
             "}\n", k_echo);
    expect_i64("hot replace", src, 5);
}

static void test_typed_mailbox(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "%s"
             "main {\n"
             "  c = spawn Counter\n"
             "  send c (Ping 1)\n"
             "  recv | Pong n => n\n"
             "}\n", k_echo, k_counter);
    expect_fail("typed mailbox", src, "mailbox");
}

static void test_exclusions(void) {
    expect_fail("remote spawn",
                "message Ping of int\n"
                "message Pong of int\n"
                "actor Echo { receive | Ping n => reply (Pong n) }\n"
                "main { e = spawn remote Echo\n  0 }\n",
                "remote");
    expect_fail("cap",
                "message Ping of int\n"
                "message Pong of int\n"
                "actor Echo { receive | Ping n => reply (Pong n) }\n"
                "main { e = spawn Echo\n  send e cap:x\n  0 }\n",
                "cap");
}

static void test_frontend(void) {
    char err[NL_ACTOR_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult acc;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_ACTOR);
    char src[1024];
    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(ACTOR).implemented is 0");
    } else {
        PASS("goal implemented");
    }
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  e = spawn Echo\n"
             "  send e (Ping 1)\n"
             "  recv | Pong n => n\n"
             "}\n", k_echo);
    err[0] = '\0';
    mod = nl_actor_compile(src, "echo.act", err, sizeof err);
    if (!mod) {
        FAIL("nl_actor_accept", err[0] ? err : "compile failed");
        return;
    }
    acc = nl_actor_accept(mod, "echo.act");
    nvm_module_free(mod);
    if (!acc.ok) {
        FAIL("nl_actor_accept", acc.error);
        return;
    }
    PASS("nl_actor_accept");
}

int main(void) {
    printf("\n[actor] Nano Actor laboratory...\n\n");
    test_ping();
    test_mailbox_order();
    test_become();
    test_isolation();
    test_monitor();
    test_link();
    test_link_kill();
    test_supervise();
    test_timeout();
    test_cancel();
    test_replace();
    test_typed_mailbox();
    test_exclusions();
    test_frontend();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
