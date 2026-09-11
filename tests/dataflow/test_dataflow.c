/*
 * Nano Dataflow laboratory tests (4.6).
 */

#include "dataflow/dataflow.h"
#include "nanoisa/frontend.h"
#include "nanoisa/nvm_format.h"

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
    char err[NL_DF_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_dataflow_eval_i64(src, &got, err, sizeof err)) {
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
    char err[NL_DF_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (nl_dataflow_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, "accepted source that must fail closed");
        return;
    }
    if (needle && !strstr(err, needle)) {
        FAIL(name, err[0] ? err : "wrong error");
        return;
    }
    PASS(name);
}

static const char *k_add =
    "node add a b = a + b\n"
    "node inc x = x + 1\n"
    "node id x = x\n";

static void test_add(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "graph {\n"
             "  a = in\n"
             "  b = in\n"
             "  s = add a b\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  feed a 2\n"
             "  feed b 3\n"
             "  drain\n"
             "}\n", k_add);
    expect_i64("add", src, 5);
}

static void test_pipeline(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "graph {\n"
             "  a = in\n"
             "  b = in\n"
             "  s = add a b\n"
             "  t = inc s\n"
             "  out t\n"
             "}\n"
             "main {\n"
             "  feed a 2\n"
             "  feed b 3\n"
             "  drain\n"
             "}\n", k_add);
    expect_i64("pipeline", src, 6);
}

static void test_backpressure(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "buffer 1\n"
             "graph {\n"
             "  a = in\n"
             "  s = id a\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  feed a 1\n"
             "  feed a 2\n"
             "  drain\n"
             "}\n", k_add);
    expect_fail("backpressure", src, "buffer");
}

static void test_determinism(void) {
    char src[1024];
    char err0[NL_DF_ERR_SIZE], err1[NL_DF_ERR_SIZE];
    int64_t a = 0, b = 0;
    snprintf(src, sizeof src,
             "%s"
             "graph {\n"
             "  p = in\n"
             "  q = in\n"
             "  u = inc p\n"
             "  v = inc q\n"
             "  s = add u v\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  feed p 1\n"
             "  feed q 1\n"
             "  drain\n"
             "}\n", k_add);
    err0[0] = err1[0] = '\0';
    if (!nl_dataflow_eval_i64_sched(src, 0, &a, err0, sizeof err0)) {
        FAIL("parallel determinism", err0[0] ? err0 : "fifo failed");
        return;
    }
    if (!nl_dataflow_eval_i64_sched(src, 1, &b, err1, sizeof err1)) {
        FAIL("parallel determinism", err1[0] ? err1 : "reverse failed");
        return;
    }
    if (a != 4 || b != 4) {
        FAIL("parallel determinism", "scheduler order changed the result");
        return;
    }
    PASS("parallel determinism");
}

static void test_retry(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "node flaky x retry 1 = if attempt == 0 then fail else x\n"
             "graph {\n"
             "  a = in\n"
             "  s = flaky a\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  feed a 7\n"
             "  drain\n"
             "}\n");
    expect_i64("retry", src, 7);
}

static void test_cancel(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "graph {\n"
             "  a = in\n"
             "  s = id a\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  cancel s\n"
             "  feed a 1\n"
             "  drain\n"
             "}\n", k_add);
    expect_fail("cancel", src, "cancel");
}

static void test_replay(void) {
    char src[1024];
    char e1[NL_DF_ERR_SIZE], e2[NL_DF_ERR_SIZE];
    int64_t a = 0, b = 0;
    snprintf(src, sizeof src,
             "%s"
             "graph {\n"
             "  a = in\n"
             "  b = in\n"
             "  s = add a b\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  feed a 2\n"
             "  feed b 3\n"
             "  drain\n"
             "}\n", k_add);
    if (!nl_dataflow_eval_i64(src, &a, e1, sizeof e1)
        || !nl_dataflow_eval_i64(src, &b, e2, sizeof e2) || a != 5 || b != 5) {
        FAIL("replay", "second run did not reproduce the first");
        return;
    }
    PASS("replay");
}

static void test_provenance(void) {
    char err[NL_DF_ERR_SIZE];
    NvmModule *mod;
    uint32_t i;
    int saw_a = 0, saw_b = 0;
    const char *src =
        "node add a b = a + b\n"
        "graph {\n"
        "  a = in\n"
        "  b = in\n"
        "  s = add a b\n"
        "  out s\n"
        "}\n"
        "main {\n"
        "  feed a 2\n"
        "  feed b 3\n"
        "  drain\n"
        "}\n";
    err[0] = '\0';
    mod = nl_dataflow_compile(src, "add.df", err, sizeof err);
    if (!mod) {
        FAIL("provenance journal", err[0] ? err : "compile failed");
        return;
    }
    for (i = 0; i < mod->string_count; i++) {
        const char *s = nvm_get_string(mod, i);
        if (s && strcmp(s, "feed a 2") == 0) saw_a = 1;
        if (s && strcmp(s, "feed b 3") == 0) saw_b = 1;
    }
    nvm_module_free(mod);
    if (!saw_a || !saw_b) {
        FAIL("provenance journal", "missing interned feed records");
        return;
    }
    PASS("provenance journal");
}

static void test_bulk(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "node sum4 a b c d = ((a + b) + c) + d\n"
             "graph {\n"
             "  a = in\n"
             "  b = in\n"
             "  c = in\n"
             "  d = in\n"
             "  s = sum4 a b c d\n"
             "  out s\n"
             "}\n"
             "main {\n"
             "  feed a 10\n"
             "  feed b 20\n"
             "  feed c 30\n"
             "  feed d 40\n"
             "  drain\n"
             "}\n");
    expect_i64("bulk copy", src, 100);
}

static void test_exclusions(void) {
    expect_fail("remote place",
                "node id x = x\n"
                "graph {\n"
                "  a = in\n"
                "  s = id a\n"
                "  place s remote\n"
                "  out s\n"
                "}\n"
                "main { feed a 1\n  drain }\n",
                "remote");
    expect_fail("unknown effect",
                "node id x effect Warp = x\n"
                "graph { a = in\n  s = id a\n  out s }\n"
                "main { feed a 1\n  drain }\n",
                "effect");
}

static void test_frontend(void) {
    char err[NL_DF_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult acc;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_DATAFLOW);
    const char *src =
        "node add a b = a + b\n"
        "graph {\n"
        "  a = in\n"
        "  b = in\n"
        "  s = add a b\n"
        "  out s\n"
        "}\n"
        "main {\n"
        "  feed a 1\n"
        "  feed b 1\n"
        "  drain\n"
        "}\n";
    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(DATAFLOW).implemented is 0");
    } else {
        PASS("goal implemented");
    }
    err[0] = '\0';
    mod = nl_dataflow_compile(src, "add.df", err, sizeof err);
    if (!mod) {
        FAIL("nl_dataflow_accept", err[0] ? err : "compile failed");
        return;
    }
    acc = nl_dataflow_accept(mod, "add.df");
    nvm_module_free(mod);
    if (!acc.ok) {
        FAIL("nl_dataflow_accept", acc.error);
        return;
    }
    PASS("nl_dataflow_accept");
}

int main(void) {
    printf("\n[dataflow] Nano Dataflow laboratory...\n\n");
    test_add();
    test_pipeline();
    test_backpressure();
    test_determinism();
    test_retry();
    test_cancel();
    test_replay();
    test_provenance();
    test_bulk();
    test_exclusions();
    test_frontend();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
