#include "nsi.h"
#include "nsi_gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static char *gen_to_buf(int (*fn)(const NlNsi *, FILE *), const NlNsi *nsi) {
    char *buf = NULL;
    size_t n = 0;
    FILE *fp = open_memstream(&buf, &n);
    if (!fp) return NULL;
    if (fn(nsi, fp) != 0) {
        fclose(fp);
        free(buf);
        return NULL;
    }
    fclose(fp);
    return buf;
}

static void test_gen_nanolang_and_forth(void) {
    const char *test_name = "nsi_gen: NanoLang and Forth from the same NSI";
    NlNsi *nsi = nl_nsi_load_path("schema/nsi/examples/log.nsi.json");
    char *nano;
    char *forth;
    if (!nsi) { FAIL(test_name, "load"); return; }
    nano = gen_to_buf(nl_nsi_gen_nanolang, nsi);
    forth = gen_to_buf(nl_nsi_gen_forth, nsi);
    if (!nano || !forth) { FAIL(test_name, "gen"); free(nano); free(forth); nl_nsi_free(nsi); return; }
    if (!strstr(nano, "fn nsi_nanolang_log_write") || !strstr(nano, "shadow nsi_nanolang_log_write"))
        { FAIL(test_name, "nanolang"); free(nano); free(forth); nl_nsi_free(nsi); return; }
    if (!strstr(forth, ": nsi-nanolang-log-write") || !strstr(forth, "nsi:nanolang/log"))
        { FAIL(test_name, "forth"); free(nano); free(forth); nl_nsi_free(nsi); return; }
    PASS(test_name);
    free(nano);
    free(forth);
    nl_nsi_free(nsi);
}

static void test_gen_dispatch_and_imports(void) {
    const char *test_name = "nsi_gen: dispatch ids and NanoISA imports";
    NlNsi *nsi = nl_nsi_load_path("schema/nsi/examples/log.nsi.json");
    char *disp;
    char *imp;
    char *docs;
    if (!nsi) { FAIL(test_name, "load"); return; }
    disp = gen_to_buf(nl_nsi_gen_dispatch, nsi);
    imp = gen_to_buf(nl_nsi_gen_nanoisa_imports, nsi);
    docs = gen_to_buf(nl_nsi_gen_docs, nsi);
    if (!disp || !imp || !docs) { FAIL(test_name, "gen"); free(disp); free(imp); free(docs); nl_nsi_free(nsi); return; }
    if (!strstr(disp, "nsi:nanolang/log#write"))
        { FAIL(test_name, "dispatch"); free(disp); free(imp); free(docs); nl_nsi_free(nsi); return; }
    if (!strstr(imp, "IMPORT nsi:nanolang/log nsi:nanolang/log#write") ||
        !strstr(imp, "TRAP cap cap:nanolang/log.write"))
        { FAIL(test_name, "imports"); free(disp); free(imp); free(docs); nl_nsi_free(nsi); return; }
    if (!strstr(docs, "nsi:nanolang/log#write"))
        { FAIL(test_name, "docs"); free(disp); free(imp); free(docs); nl_nsi_free(nsi); return; }
    PASS(test_name);
    free(disp);
    free(imp);
    free(docs);
    nl_nsi_free(nsi);
}

static void test_gen_languages_and_wire(void) {
    const char *test_name = "nsi_gen: C/C++/Python/Rust/NanoVM share method ids";
    NlNsi *nsi = nl_nsi_load_path("schema/nsi/examples/log.nsi.json");
    char *py;
    char *rs;
    char *cxx;
    char *idx;
    char *ser;
    char *val;
    char *compat;
    if (!nsi) { FAIL(test_name, "load"); return; }
    py = gen_to_buf(nl_nsi_gen_python, nsi);
    rs = gen_to_buf(nl_nsi_gen_rust, nsi);
    cxx = gen_to_buf(nl_nsi_gen_cxx, nsi);
    idx = gen_to_buf(nl_nsi_gen_language_index, nsi);
    ser = gen_to_buf(nl_nsi_gen_serialize, nsi);
    val = gen_to_buf(nl_nsi_gen_validate, nsi);
    compat = gen_to_buf(nl_nsi_gen_compat_tests, nsi);
    if (!py || !rs || !cxx || !idx || !ser || !val || !compat) {
        FAIL(test_name, "gen");
        free(py); free(rs); free(cxx); free(idx); free(ser); free(val); free(compat);
        nl_nsi_free(nsi);
        return;
    }
    if (!strstr(py, "nsi:nanolang/log#write") || !strstr(rs, "nsi:nanolang/log#write") ||
        !strstr(cxx, "nsi:nanolang/log#write") || !strstr(idx, "Python nsi:nanolang/log#write") ||
        !strstr(idx, "remote nsi:nanolang/log#write") || !strstr(idx, "NanoVM nsi:nanolang/log#write")) {
        FAIL(test_name, "ids");
        free(py); free(rs); free(cxx); free(idx); free(ser); free(val); free(compat);
        nl_nsi_free(nsi);
        return;
    }
    if (!strstr(ser, "\"frame\":\"request\"") || !strstr(ser, "nsi:nanolang/log#write") ||
        !strstr(val, "nsi:nanolang/log#write") || !strstr(compat, "nl_nsi_compat")) {
        FAIL(test_name, "wire");
        free(py); free(rs); free(cxx); free(idx); free(ser); free(val); free(compat);
        nl_nsi_free(nsi);
        return;
    }
    PASS(test_name);
    free(py); free(rs); free(cxx); free(idx); free(ser); free(val); free(compat);
    nl_nsi_free(nsi);
}

int main(void) {
    test_gen_nanolang_and_forth();
    test_gen_dispatch_and_imports();
    test_gen_languages_and_wire();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
