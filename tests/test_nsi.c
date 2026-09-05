#include "nsi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_load_log_example(void) {
    const char *test_name = "nsi: load log example and read stable ids";
    NlNsi *nsi = nl_nsi_load_path("schema/nsi/examples/log.nsi.json");
    if (!nsi) { FAIL(test_name, "load"); return; }
    if (nsi->version != 0) { FAIL(test_name, "version"); nl_nsi_free(nsi); return; }
    if (strcmp(nl_nsi_interface_id(nsi), "nsi:nanolang/log") != 0)
        { FAIL(test_name, "interface id"); nl_nsi_free(nsi); return; }
    if (nl_nsi_method_count(nsi) != 2)
        { FAIL(test_name, "method count"); nl_nsi_free(nsi); return; }
    if (strcmp(nl_nsi_method_id(nsi, 0), "nsi:nanolang/log#write") != 0)
        { FAIL(test_name, "method id"); nl_nsi_free(nsi); return; }
    if (nsi->type_count != 1 || strcmp(nsi->types[0].id, "nsi:nanolang/log#Level") != 0)
        { FAIL(test_name, "type id"); nl_nsi_free(nsi); return; }
    if (nsi->error_count != 1 || strcmp(nsi->errors[0].id, "nsi:nanolang/log#io") != 0)
        { FAIL(test_name, "error id"); nl_nsi_free(nsi); return; }
    if (nsi->capability_count != 1 ||
        strcmp(nsi->capabilities[0].id, "cap:nanolang/log.write") != 0)
        { FAIL(test_name, "capability id"); nl_nsi_free(nsi); return; }
    if (nl_nsi_param_count(nsi, 0) != 2)
        { FAIL(test_name, "param count"); nl_nsi_free(nsi); return; }
    if (nl_nsi_param_count(nsi, 1) != 0)
        { FAIL(test_name, "write_event params optional"); nl_nsi_free(nsi); return; }
    {
        const NlNsiParam *p = nl_nsi_param(nsi, 0, 1);
        if (!p || strcmp(p->id, "nsi:nanolang/log#write.message") != 0)
            { FAIL(test_name, "param id"); nl_nsi_free(nsi); return; }
        if (p->direction != NL_NSI_DIR_IN || p->ownership != NL_NSI_OWN_BORROW ||
            p->lifetime != NL_NSI_LIFE_CALL || p->mutability != NL_NSI_MUT_IMMUTABLE ||
            p->optional != 0 || p->streaming != NL_NSI_STREAM_NONE)
            { FAIL(test_name, "param contract"); nl_nsi_free(nsi); return; }
        if (strcmp(p->type_id, "nsi:core/string") != 0)
            { FAIL(test_name, "param type"); nl_nsi_free(nsi); return; }
    }
    PASS(test_name);
    nl_nsi_free(nsi);
}

static int write_tmp(const char *path, const char *body) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fputs(body, fp);
    fclose(fp);
    return 1;
}

static void test_reject_missing_interface_id(void) {
    const char *test_name = "nsi: reject missing interface id";
    const char *path = "/tmp/nl_nsi_no_id.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,\"interface\":{\"name\":\"log\"},"
        "\"methods\":[],\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_duplicate_method_id(void) {
    const char *test_name = "nsi: reject duplicate method id";
    const char *path = "/tmp/nl_nsi_dup.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":["
        "{\"id\":\"nsi:nanolang/log#write\",\"name\":\"write\"},"
        "{\"id\":\"nsi:nanolang/log#write\",\"name\":\"other\"}"
        "],\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_method_not_under_interface(void) {
    const char *test_name = "nsi: reject method id outside interface";
    const char *path = "/tmp/nl_nsi_orphan.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":[{\"id\":\"nsi:nanolang/fs#write\",\"name\":\"write\"}],"
        "\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_wrong_version(void) {
    const char *test_name = "nsi: reject unknown nsi_version";
    const char *path = "/tmp/nl_nsi_ver.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":1,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":[],\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_invalid_utf8(void) {
    const char *test_name = "nsi: reject invalid UTF-8";
    const char *path = "/tmp/nl_nsi_utf8.json";
    FILE *fp = fopen(path, "wb");
    NlNsi *nsi;
    if (!fp) { FAIL(test_name, "open"); return; }
    fputs("{\"nsi_version\":0,\"interface\":{\"id\":\"nsi:x\",\"name\":\"x\xff\"}}", fp);
    fclose(fp);
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_capability_without_cap_prefix(void) {
    const char *test_name = "nsi: reject capability without cap: prefix";
    const char *path = "/tmp/nl_nsi_cap.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":[],\"types\":[],\"errors\":[],"
        "\"capabilities\":[{\"id\":\"nsi:nanolang/log#write\",\"name\":\"write\"}]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_param_missing_direction(void) {
    const char *test_name = "nsi: reject param missing direction";
    const char *path = "/tmp/nl_nsi_param_dir.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":[{\"id\":\"nsi:nanolang/log#write\",\"name\":\"write\","
        "\"params\":[{\"id\":\"nsi:nanolang/log#write.p\",\"name\":\"p\","
        "\"type\":\"nsi:core/string\",\"ownership\":\"borrow\","
        "\"lifetime\":\"call\",\"mutability\":\"immutable\","
        "\"optional\":false,\"streaming\":\"none\"}]}],"
        "\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_unknown_ownership(void) {
    const char *test_name = "nsi: reject unknown ownership";
    const char *path = "/tmp/nl_nsi_own.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":[{\"id\":\"nsi:nanolang/log#write\",\"name\":\"write\","
        "\"params\":[{\"id\":\"nsi:nanolang/log#write.p\",\"name\":\"p\","
        "\"type\":\"nsi:core/string\",\"direction\":\"in\","
        "\"ownership\":\"shared\",\"lifetime\":\"call\","
        "\"mutability\":\"immutable\",\"optional\":false,"
        "\"streaming\":\"none\"}]}],"
        "\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

static void test_reject_stream_in_on_out_param(void) {
    const char *test_name = "nsi: reject streaming in on out param";
    const char *path = "/tmp/nl_nsi_stream.json";
    NlNsi *nsi;
    if (!write_tmp(path,
        "{\"nsi_version\":0,"
        "\"interface\":{\"id\":\"nsi:nanolang/log\",\"name\":\"log\"},"
        "\"methods\":[{\"id\":\"nsi:nanolang/log#write\",\"name\":\"write\","
        "\"params\":[{\"id\":\"nsi:nanolang/log#write.p\",\"name\":\"p\","
        "\"type\":\"nsi:core/string\",\"direction\":\"out\","
        "\"ownership\":\"transfer\",\"lifetime\":\"caller\","
        "\"mutability\":\"mutable\",\"optional\":false,"
        "\"streaming\":\"in\"}]}],"
        "\"types\":[],\"errors\":[],\"capabilities\":[]}"))
        { FAIL(test_name, "write"); return; }
    nsi = nl_nsi_load_path(path);
    unlink(path);
    if (nsi) { FAIL(test_name, "accepted"); nl_nsi_free(nsi); return; }
    PASS(test_name);
}

int main(void) {
    test_load_log_example();
    test_reject_missing_interface_id();
    test_reject_duplicate_method_id();
    test_reject_method_not_under_interface();
    test_reject_wrong_version();
    test_reject_invalid_utf8();
    test_reject_capability_without_cap_prefix();
    test_reject_param_missing_direction();
    test_reject_unknown_ownership();
    test_reject_stream_in_on_out_param();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
