/*
 * Nano Shell laboratory tests (4.6).
 */

#include "shell/shell.h"
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
    char err[NL_SH_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_shell_eval_i64(src, &got, err, sizeof err)) {
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
    char err[NL_SH_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (nl_shell_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, "expected failure");
        return;
    }
    if (!strstr(err, needle)) {
        FAIL(name, err[0] ? err : "missing needle");
        return;
    }
    PASS(name);
}

static void test_pipe(void) {
    expect_i64("typed pipe",
               "fn add a b = a + b\n"
               "main { 2 | add 3 }\n",
               5);
}

static void test_chain(void) {
    expect_i64("chained pipe",
               "fn add a b = a + b\n"
               "fn inc x = x + 1\n"
               "main { 2 | add 3 | inc }\n",
               6);
}

static void test_bind(void) {
    expect_i64("structured bind",
               "fn add a b = a + b\n"
               "main {\n"
               "  x = 2\n"
               "  y = 3\n"
               "  x | add y\n"
               "}\n",
               5);
}

static void test_parse(void) {
    expect_i64("parse adapter",
               "fn add a b = a + b\n"
               "main { parse \"3\" | add 1 }\n",
               4);
}

static void test_text_refused(void) {
    expect_fail("text is not a pipeline value",
                "fn add a b = a + b\n"
                "main { \"3\" | add 1 }\n",
                "text");
}

static void test_cancel(void) {
    expect_fail("cancel",
                "fn add a b = a + b\n"
                "main {\n"
                "  cancel\n"
                "  2 | add 3\n"
                "}\n",
                "cancel");
}

static void test_caps(void) {
    expect_fail("read without cap", "main { read 1 }\n", "cap:");
    expect_fail("read with cap still refuses files",
                "need files\nmain { read 1 }\n", "host file");
    expect_fail("run without cap", "main { run 1 }\n", "cap:");
    expect_fail("run with cap still refuses processes",
                "need proc\nmain { run 1 }\n", "host process");
    expect_fail("connect without cap", "main { connect 1 }\n", "cap:");
    expect_fail("connect with cap still refuses networks",
                "need net\nmain { connect 1 }\n", "host network");
    expect_fail("service without cap", "main { service 1 }\n", "cap:");
    expect_fail("service with cap still refuses services",
                "need service\nmain { service 1 }\n", "host service");
    expect_fail("stream without cap", "main { stream 1 }\n", "cap:");
    expect_fail("stream with cap still refuses host streams",
                "need stream\nmain { stream 1 }\n", "host stream");
    expect_fail("remote without cap", "main { remote 1 }\n", "cap:");
    expect_fail("remote with cap still refuses remote",
                "need remote\nmain { remote 1 }\n", "host remote");
}

static void test_arith(void) {
    expect_i64("host arith", "main { 2 + 3 }\n", 5);
}

static void test_frontend(void) {
    char err[NL_SH_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult acc;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_SHELL);
    const char *src = "fn add a b = a + b\nmain { 2 | add 3 }\n";
    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(SHELL).implemented is 0");
    } else {
        PASS("goal implemented");
    }
    err[0] = '\0';
    mod = nl_shell_compile(src, "add.sh", err, sizeof err);
    if (!mod) {
        FAIL("nl_shell_accept", err[0] ? err : "compile failed");
        return;
    }
    acc = nl_shell_accept(mod, "add.sh");
    nvm_module_free(mod);
    if (!acc.ok) {
        FAIL("nl_shell_accept", acc.error);
        return;
    }
    PASS("nl_shell_accept");
}

int main(void) {
    printf("\n[shell] Nano Shell laboratory...\n\n");
    test_pipe();
    test_chain();
    test_bind();
    test_parse();
    test_text_refused();
    test_cancel();
    test_caps();
    test_arith();
    test_frontend();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
