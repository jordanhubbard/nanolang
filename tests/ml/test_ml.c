/*
 * Nano ML laboratory tests (4.6).
 */

#include "ml/ml.h"
#include "nanoisa/assembler.h"
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
    char err[NL_ML_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (!nl_ml_eval_i64(src, &got, err, sizeof err)) {
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
    char err[NL_ML_ERR_SIZE];
    int64_t got = 0;
    err[0] = '\0';
    if (nl_ml_eval_i64(src, &got, err, sizeof err)) {
        FAIL(name, "accepted source that must fail closed");
        return;
    }
    if (needle && !strstr(err, needle)) {
        FAIL(name, err[0] ? err : "wrong error");
        return;
    }
    PASS(name);
}

static void expect_type(const char *name, const char *src, const char *id,
                        const char *want) {
    char err[NL_ML_ERR_SIZE];
    char got[128];
    err[0] = '\0';
    got[0] = '\0';
    if (!nl_ml_type_of(src, id, got, sizeof got, err, sizeof err)) {
        FAIL(name, err[0] ? err : "type_of failed");
        return;
    }
    if (strcmp(got, want) != 0) {
        char buf[192];
        snprintf(buf, sizeof buf, "got '%s' want '%s'", got, want);
        FAIL(name, buf);
        return;
    }
    PASS(name);
}

static void test_core(void) {
    expect_i64("1 + 2", "1 + 2", 3);
    expect_i64("if", "if true then 1 else 0", 1);
    expect_i64("if false", "if false then 1 else 0", 0);
    expect_i64("fun add", "fun add a b = a + b\nadd 2 3", 5);
    expect_i64("fn apply", "(fn x => x + 1) 4", 5);
    expect_i64("let", "let val x = 2 in x + 3 end", 5);
    expect_i64("lexical", "let val x = 1 in let val x = 10 in x end end", 10);
}

static void test_hof_and_poly(void) {
    expect_i64("id int", "fun id x = x\nid 41", 41);
    expect_i64("make-add",
               "fun makeAdd n = fn x => x + n\n(makeAdd 10) 3", 13);
    expect_i64("apply",
               "fun apply f x = f x\nfun inc n = n + 1\napply inc 41", 42);
    expect_type("id scheme", "fun id x = x\nid 1", "id", "a -> a");
}

static void test_tuples_and_adt(void) {
    expect_i64("tuple fst",
               "fun fst p = case p of (a, b) => a\nfst (7, 8)", 7);
    expect_i64("option some",
               "datatype option = None | Some of int\n"
               "fun get x = case x of None => 0 | Some n => n\n"
               "get (Some 9)", 9);
    expect_i64("option none",
               "datatype option = None | Some of int\n"
               "fun get x = case x of None => 0 | Some n => n\n"
               "get None", 0);
    expect_fail("inexhaustive",
                "datatype option = None | Some of int\n"
                "fun get x = case x of None => 0\n"
                "get None",
                "exhaustive");
}

static void test_signature(void) {
    expect_i64("signature add",
               "signature ADD = sig\n"
               "  val add : int -> int -> int\n"
               "end\n"
               "fun add a b = a + b\n"
               "add 4 5", 9);
    expect_fail("signature mismatch",
                "signature ADD = sig\n"
                "  val add : int -> int -> int\n"
                "end\n"
                "fun add a = a\n"
                "add 1",
                "signature");
}

static void test_exclusions(void) {
    expect_fail("ref", "val x = ref 1\n1", "ref");
    expect_fail("exception", "exception E\n1", "exception");
    expect_fail("assign", "val x = 1\nx := 2", ":=");
}

static void test_frontend(void) {
    char err[NL_ML_ERR_SIZE];
    NvmModule *mod;
    NlFrontendResult acc;
    const NlFrontendGoal *g = nl_frontend_goal(NL_FE_ML);
    if (!g || !g->implemented) {
        FAIL("goal implemented", "nl_frontend_goal(ML).implemented is 0");
    } else {
        PASS("goal implemented");
    }
    err[0] = '\0';
    mod = nl_ml_compile("fun add a b = a + b\nadd 1 2", "add.sml", err, sizeof err);
    if (!mod) {
        FAIL("nl_ml_accept", err[0] ? err : "compile failed");
        return;
    }
    acc = nl_ml_accept(mod, "add.sml");
    nvm_module_free(mod);
    if (!acc.ok) {
        FAIL("nl_ml_accept", acc.error);
        return;
    }
    PASS("nl_ml_accept");
}

static void test_type_metadata(void) {
    char err[NL_ML_ERR_SIZE];
    NvmModule *mod;
    uint32_t i;
    int found = 0;
    err[0] = '\0';
    mod = nl_ml_compile("fun id x = x\nid 1", "id.sml", err, sizeof err);
    if (!mod) {
        FAIL("type metadata compile", err[0] ? err : "compile failed");
        return;
    }
    for (i = 0; i < mod->string_count; i++) {
        const char *s = nvm_get_string(mod, i);
        if (s && strcmp(s, "id : a -> a") == 0) {
            found = 1;
            break;
        }
    }
    nvm_module_free(mod);
    if (!found) {
        FAIL("type metadata", "missing id : a -> a in the string pool");
        return;
    }
    PASS("type metadata");
}

static void test_shared_aggregate(void) {
    char err[NL_ML_ERR_SIZE];
    NvmModule *ml, *nano;
    NlFrontendFacts fml, fnano;
    NlFrontendResult r;
    AsmResult ar;
    const NvmModule *table[1];
    uint32_t idx;
    const char *caller =
        ".module_ref \"ml\"\n"
        ".function main 0 2 0 int 1\n"
        "  PUSH_I64 7\n"
        "  PUSH_I64 8\n"
        "  TUPLE_NEW 2\n"
        "  CALL_MODULE 0 0 1 1\n"
        "  RET\n"
        ".end\n";

    err[0] = '\0';
    ml = nl_ml_compile("fun fst p = case p of (a, b) => a\nfst (1, 2)",
                       "fst.sml", err, sizeof err);
    if (!ml) {
        FAIL("shared compile", err[0] ? err : "compile failed");
        return;
    }
    idx = nl_ml_fun_index("fun fst p = case p of (a, b) => a\nfst (1, 2)",
                          "fst", err, sizeof err);
    if (idx != 0) {
        char buf[64];
        snprintf(buf, sizeof buf, "fst index %u want 0", idx);
        FAIL("shared fun index", buf);
        nvm_module_free(ml);
        return;
    }
    memset(&ar, 0, sizeof ar);
    nano = asm_assemble(caller, &ar);
    if (!nano) {
        FAIL("shared nanolang assemble", ar.message);
        nvm_module_free(ml);
        return;
    }
    nano->header.flags |= NVM_FLAG_DEBUG_INFO;
    nvm_add_debug_entry(nano, 0, 1, 1);
    memset(&fml, 0, sizeof fml);
    fml.language = NL_FE_ML;
    fml.source_path = "fst.sml";
    fml.purity = 1;
    fml.exhaustiveness = 1;
    fml.affine_use = -1;
    fml.diagnostics_shared = 1;
    memset(&fnano, 0, sizeof fnano);
    fnano.language = NL_FE_NANOLANG;
    fnano.source_path = "caller.nano";
    fnano.purity = -1;
    fnano.exhaustiveness = -1;
    fnano.affine_use = -1;
    fnano.diagnostics_shared = 1;
    table[0] = ml;
    r = nl_ml_accept(ml, "fst.sml");
    if (!r.ok) {
        FAIL("shared ml accept", r.error);
        nvm_module_free(ml);
        nvm_module_free(nano);
        return;
    }
    r = nl_frontend_accept_linked(nano, &fnano, table, 1);
    nvm_module_free(ml);
    nvm_module_free(nano);
    if (!r.ok) {
        FAIL("shared linked", r.error);
        return;
    }
    PASS("shared aggregate NanoLang+ML");
}

int main(void) {
    printf("\n[ml] Nano ML laboratory...\n\n");
    test_core();
    test_hof_and_poly();
    test_tuples_and_adt();
    test_signature();
    test_exclusions();
    test_frontend();
    test_type_metadata();
    test_shared_aggregate();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
