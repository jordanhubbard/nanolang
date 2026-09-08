/*
 * Nano Logic laboratory tests (4.6).
 */

#include "logic/logic.h"
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
    char err[NL_LG_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_logic_eval_i64(src, &got, err, sizeof err)) {
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
    char err[NL_LG_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (nl_logic_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, "expected failure");
        return;
    }
    if (!strstr(err, needle)) {
        FAIL(name, err[0] ? err : "missing needle");
        return;
    }
    PASS(name);
}

static void test_fact(void) {
    expect_i64("ground fact",
               "fact edge 1 2\n"
               "query edge 1 2\n",
               1);
    expect_i64("missing fact",
               "fact edge 1 2\n"
               "query edge 1 3\n",
               0);
}

static void test_path(void) {
    expect_i64("transitive least fixed-point",
               "fact edge 1 2\n"
               "fact edge 2 3\n"
               "rule path x y :- edge x y\n"
               "rule path x z :- edge x y, path y z\n"
               "query path 1 3\n",
               1);
    expect_i64("no spurious path",
               "fact edge 1 2\n"
               "fact edge 2 3\n"
               "rule path x y :- edge x y\n"
               "rule path x z :- edge x y, path y z\n"
               "query path 3 1\n",
               0);
}

static void test_policy(void) {
    expect_i64("restricted policy query",
               "fact grant 7\n"
               "rule allow x :- grant x\n"
               "query allow 7\n",
               1);
    expect_i64("policy deny",
               "fact grant 7\n"
               "rule allow x :- grant x\n"
               "query allow 8\n",
               0);
}

static void test_unary(void) {
    expect_i64("unary fact",
               "fact live 1\n"
               "query live 1\n",
               1);
}

static void test_open_refused(void) {
    expect_fail("open query",
                "fact edge 1 2\n"
                "query edge 1 y\n",
                "ground");
}

static void test_frontend(void) {
    char err[NL_LG_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult acc;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_LOGIC);
    const char *src = "fact edge 1 2\nquery edge 1 2\n";
    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(LOGIC).implemented is 0");
    } else if (g->out_of_scope && strstr(g->out_of_scope, "choice")) {
        PASS("goal implemented");
        PASS("choice points stay out of scope");
    } else {
        PASS("goal implemented");
        FAIL("choice points stay out of scope", "goal did not exclude choice points");
    }
    err[0] = '\0';
    mod = nl_logic_compile(src, "edge.dl", err, sizeof err);
    if (!mod) {
        FAIL("nl_logic_accept", err[0] ? err : "compile failed");
        return;
    }
    acc = nl_logic_accept(mod, "edge.dl");
    nvm_module_free(mod);
    if (!acc.ok) {
        FAIL("nl_logic_accept", acc.error);
        return;
    }
    PASS("nl_logic_accept");
}

int main(void) {
    printf("\n[logic] Nano Logic laboratory...\n\n");
    test_fact();
    test_path();
    test_policy();
    test_unary();
    test_open_refused();
    test_frontend();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
