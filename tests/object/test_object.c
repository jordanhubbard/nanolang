/*
 * Nano Object laboratory tests (4.6).
 */

#include "object/object.h"
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
    char err[NL_OBJ_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_object_eval_i64(src, &got, err, sizeof err)) {
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

static const char *k_counter =
    "class Counter {\n"
    "  n\n"
    "  method inc { n = n + 1 }\n"
    "  method get { n }\n"
    "  method plus x { n + x }\n"
    "}\n";

static void test_inc(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  send c inc\n"
             "  send c inc\n"
             "  send c get\n"
             "}\n", k_counter);
    expect_i64("inc", src, 2);
}

static void test_plus(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  send c plus 4\n"
             "}\n", k_counter);
    expect_i64("plus", src, 4);
}

static void test_identity(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "class Node {\n"
             "  next\n"
             "  method setnext x { next = x }\n"
             "  method getnext { next }\n"
             "}\n"
             "main {\n"
             "  a = new Node\n"
             "  b = new Node\n"
             "  send a setnext b\n"
             "  send a getnext\n"
             "}\n");
    expect_i64("mutable graph", src, 2);
}

static void test_ic(void) {
    char src[1024];
    char err[NL_OBJ_ERR_SIZE];
    int64_t got = 0;
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  send c inc\n"
             "  send c inc\n"
             "  send c get\n"
             "}\n", k_counter);
    if (!nl_object_eval_i64(src, &got, err, sizeof err) || got != 2) {
        FAIL("inline cache", err[0] ? err : "eval failed");
        return;
    }
    if (nl_object_last_ic_misses() < 1 || nl_object_last_ic_hits() < 1) {
        FAIL("inline cache", "expected a miss then a hit without a cache opcode");
        return;
    }
    PASS("inline cache");
}

static void test_replace(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  send c inc\n"
             "  replace Counter inc { n = n + 2 }\n"
             "  send c inc\n"
             "  send c get\n"
             "}\n", k_counter);
    expect_i64("live replace", src, 3);
}

static void test_extend(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  extend Counter m\n"
             "  slots c\n"
             "}\n", k_counter);
    expect_i64("layout evolution", src, 2);
}

static void test_reflect(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  classof c\n"
             "}\n", k_counter);
    expect_i64("classof", src, 0);
}

static void test_handle(void) {
    char src[1024];
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  h = handle Counter inc\n"
             "  sendvia h c\n"
             "  sendvia h c\n"
             "  send c get\n"
             "}\n", k_counter);
    expect_i64("callable handle", src, 2);
}

static void test_image(void) {
    char src[1024];
    char err[NL_OBJ_ERR_SIZE];
    char img[NL_OBJ_IMAGE_SIZE];
    int64_t got = 0;
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  send c inc\n"
             "  send c get\n"
             "}\n", k_counter);
    if (!nl_object_eval_i64(src, &got, err, sizeof err) || got != 1) {
        FAIL("image snapshot", err[0] ? err : "eval failed");
        return;
    }
    nl_object_last_image(img, sizeof img);
    if (!strstr(img, "Counter") || !strstr(img, "n=1")) {
        FAIL("image snapshot", img[0] ? img : "empty image");
        return;
    }
    PASS("image snapshot");
}

static void test_frontend(void) {
    char err[NL_OBJ_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult acc;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_OBJECT);
    char src[1024];
    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(OBJECT).implemented is 0");
    } else {
        PASS("goal implemented");
    }
    snprintf(src, sizeof src,
             "%s"
             "main {\n"
             "  c = new Counter\n"
             "  send c get\n"
             "}\n", k_counter);
    err[0] = '\0';
    mod = nl_object_compile(src, "counter.obj", err, sizeof err);
    if (!mod) {
        FAIL("nl_object_accept", err[0] ? err : "compile failed");
        return;
    }
    acc = nl_object_accept(mod, "counter.obj");
    nvm_module_free(mod);
    if (!acc.ok) {
        FAIL("nl_object_accept", acc.error);
        return;
    }
    PASS("nl_object_accept");
}

int main(void) {
    printf("\n[object] Nano Object laboratory...\n\n");
    test_inc();
    test_plus();
    test_identity();
    test_ic();
    test_replace();
    test_extend();
    test_reflect();
    test_handle();
    test_image();
    test_frontend();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
