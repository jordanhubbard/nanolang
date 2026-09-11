/*
 * Nano Scheme laboratory tests (4.6).
 */

#include "scheme/scheme.h"
#include "nanoisa/frontend.h"
#include "nanovm/value.h"

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
    char err[NL_SCHEME_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_scheme_eval_i64(src, &got, err, sizeof err)) {
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
    char err[NL_SCHEME_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (nl_scheme_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, "accepted source that must fail closed");
        return;
    }
    if (needle && !strstr(err, needle)) {
        FAIL(name, err[0] ? err : "wrong error");
        return;
    }
    PASS(name);
}

static void test_arithmetic(void) {
    expect_i64("(+ 1 2)", "(+ 1 2)", 3);
    expect_i64("(+)", "(+)", 0);
    expect_i64("(+ 4)", "(+ 4)", 4);
    expect_i64("(+ 1 2 3)", "(+ 1 2 3)", 6);
    expect_i64("(- 10 3)", "(- 10 3)", 7);
    expect_i64("(- 5)", "(- 5)", -5);
    expect_i64("(* 6 7)", "(* 6 7)", 42);
    expect_i64("(/ 20 4)", "(/ 20 4)", 5);
}

static void test_if_and_truth(void) {
    expect_i64("(if #t 1 0)", "(if #t 1 0)", 1);
    expect_i64("(if #f 1 0)", "(if #f 1 0)", 0);
    expect_i64("zero is truthy", "(if 0 1 0)", 1);
    expect_i64("(not #f)", "(if (not #f) 1 0)", 1);
    expect_i64("(not 0)", "(if (not 0) 1 0)", 0);
}

static void test_define_and_lexical(void) {
    expect_i64("named add",
               "(define (add a b) (+ a b)) (add 2 3)", 5);
    expect_i64("lambda apply",
               "((lambda (x) (+ x 1)) 4)", 5);
    expect_i64("let desugars",
               "(let ((x 2) (y 3)) (+ x y))", 5);
    expect_i64("lexical shadow",
               "(define (f x) (let ((x 10)) x)) (f 1)", 10);
}

static void test_closures(void) {
    expect_i64("make-add closure",
               "(define (make-add n) (lambda (x) (+ x n)))\n"
               "((make-add 10) 3)", 13);
    expect_i64("nested capture",
               "(define (outer a)\n"
               "  (lambda (b)\n"
               "    (lambda (c) (+ a (+ b c)))))\n"
               "(((outer 1) 2) 3)", 6);
    expect_i64("first-class procedure",
               "(define (apply1 f x) (f x))\n"
               "(define (inc n) (+ n 1))\n"
               "(apply1 inc 41)", 42);
}

static void test_pairs(void) {
    expect_i64("car cons", "(car (cons 1 2))", 1);
    expect_i64("cdr cons", "(cdr (cons 1 2))", 2);
    expect_i64("null?", "(if (null? '()) 1 0)", 1);
    expect_i64("pair?", "(if (pair? (cons 1 '())) 1 0)", 1);
    expect_i64("quoted list car", "(car (quote (9 8 7)))", 9);
    expect_i64("recursive list tail build",
               "(define (list-n n acc)\n"
               "  (if (= n 0) acc (list-n (- n 1) (cons n acc))))\n"
               "(car (list-n 1000 '()))", 1);
}

static void test_tail_calls(void) {
    char err[NL_SCHEME_ERR_SIZE];
    NvmModule *mod;
    NanoValue v;
    uint32_t depth = 0;
    VmResult r;
    const char *src =
        "(define (sum n acc)\n"
        "  (if (= n 0) acc (sum (- n 1) (+ acc n))))\n"
        "(sum 10000 0)";

    err[0] = '\0';
    mod = nl_scheme_compile(src, "tail_sum.scm", err, sizeof err);
    if (!mod) {
        FAIL("tail compile", err[0] ? err : "compile failed");
        return;
    }
    memset(&v, 0, sizeof v);
    r = nl_scheme_execute(mod, &v, &depth, err, sizeof err);
    if (r != VM_OK) {
        FAIL("tail execute", err[0] ? err : "vm error");
        nvm_module_free(mod);
        return;
    }
    if (v.tag != TAG_INT || v.as.i64 != 50005000) {
        FAIL("tail result", "sum 10000 is not 50005000");
        nvm_module_free(mod);
        return;
    }
    PASS("tail result 50005000");
    if (depth > 3) {
        char buf[64];
        snprintf(buf, sizeof buf, "frame depth %u (want <= 3)", depth);
        FAIL("constant frame depth", buf);
    } else {
        PASS("constant frame depth");
    }
    nvm_module_free(mod);
}

static void test_frontend_accept(void) {
    char err[NL_SCHEME_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult r;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_SCHEME);

    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(SCHEME).implemented is 0");
    } else {
        PASS("goal implemented");
    }

    err[0] = '\0';
    mod = nl_scheme_compile("(+ 1 2)", "add.scm", err, sizeof err);
    if (!mod) {
        FAIL("accept compile", err[0] ? err : "compile failed");
        return;
    }
    r = nl_scheme_accept(mod, "add.scm");
    if (!r.ok) {
        FAIL("nl_scheme_accept", r.error);
    } else {
        PASS("nl_scheme_accept");
    }
    nvm_module_free(mod);
}

static void test_session_live_code(void) {
    char err[NL_SCHEME_ERR_SIZE];
    NlScheme *s = nl_scheme_open();
    int64_t v = 0;

    if (!s) {
        FAIL("session open", "nl_scheme_open returned NULL");
        return;
    }
    err[0] = '\0';
    if (!nl_scheme_eval_i64_session(s, "(define (f x) (+ x 1))", &v, err, sizeof err)) {
        FAIL("session define", err[0] ? err : "define failed");
        nl_scheme_close(s);
        return;
    }
    PASS("session define");
    if (!nl_scheme_eval_i64_session(s, "(f 10)", &v, err, sizeof err) || v != 11) {
        FAIL("session call", err[0] ? err : "f did not return 11");
        nl_scheme_close(s);
        return;
    }
    PASS("session call");
    if (!nl_scheme_eval_i64_session(s, "(define (f x) (+ x 100))", &v, err, sizeof err)) {
        FAIL("session redefine", err[0] ? err : "redefine failed");
        nl_scheme_close(s);
        return;
    }
    if (!nl_scheme_eval_i64_session(s, "(f 10)", &v, err, sizeof err) || v != 110) {
        FAIL("live code publication", err[0] ? err : "redefined f still old");
        nl_scheme_close(s);
        return;
    }
    PASS("live code publication");
    nl_scheme_close(s);
}

static void test_exclusions(void) {
    expect_fail("call/cc refused", "(call/cc (lambda (k) (k 1)))", "continuation");
    expect_fail("set! refused", "(define (f x) (set! x 2) x) (f 1)", "set!");
    expect_fail("unknown form", "(unbound-xyz 1)", NULL);
}

static void test_dynamic_if(void) {
    expect_i64("if joins int and uses chosen branch",
               "(if (= 1 1) 42 0)", 42);
}

int main(void) {
    printf("\n[scheme] Nano Scheme laboratory...\n\n");
    test_arithmetic();
    test_if_and_truth();
    test_define_and_lexical();
    test_closures();
    test_pairs();
    test_tail_calls();
    test_frontend_accept();
    test_session_live_code();
    test_exclusions();
    test_dynamic_if();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
