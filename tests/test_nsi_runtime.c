#include "nsi.h"
#include "nsi_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static int client_log_write(NlNsiSession *s) {
    NlNsiCall c;
    NlNsiResult r;
    int ok;
    memset(&c, 0, sizeof(c));
    c.method_id = "nsi:nanolang/log#write";
    c.payload_json = "{\"level\":0,\"message\":\"hi\"}";
    c.capability = "cap:nanolang/log.write";
    c.auth = "tok";
    c.timeout_ms = 1000;
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_OK) {
        nl_nsi_result_free(&r);
        return 0;
    }
    ok = r.payload_json && strstr(r.payload_json, "\"ok\":true") != NULL;
    nl_nsi_result_free(&r);
    return ok;
}

static int client_vec_add(NlNsiSession *s) {
    NlNsiCall c;
    NlNsiResult r;
    int ok;
    memset(&c, 0, sizeof(c));
    c.method_id = "nsi:nanolang/vector2d#add";
    c.payload_json = "{\"x1\":1,\"y1\":2,\"x2\":3,\"y2\":4}";
    c.auth = "";
    c.timeout_ms = 1000;
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_OK) {
        nl_nsi_result_free(&r);
        return 0;
    }
    ok = r.payload_json && strstr(r.payload_json, "\"x\":4") && strstr(r.payload_json, "\"y\":6");
    nl_nsi_result_free(&r);
    return ok;
}

static NlNsiSession *open_log(NlNsiAdapterKind kind) {
    static NlNsi *nsi;
    const char *caps[] = { "cap:nanolang/log.write" };
    NlNsiSession *s;
    if (!nsi) nsi = nl_nsi_load_path("schema/nsi/examples/log.nsi.json");
    if (!nsi) return NULL;
    s = nl_nsi_session_open(nsi, kind, "schema/nsi/examples/log.nsi.json",
                            "tok", caps, 1, 1);
    if (!s) return NULL;
    if (nl_nsi_session_hello(s, nsi) != NL_NSI_OK) {
        nl_nsi_session_close(s);
        return NULL;
    }
    return s;
}

static NlNsiSession *open_vec(NlNsiAdapterKind kind) {
    static NlNsi *nsi;
    NlNsiSession *s;
    if (!nsi) nsi = nl_nsi_load_path("schema/nsi/modules/vector2d.nsi.json");
    if (!nsi) return NULL;
    s = nl_nsi_session_open(nsi, kind, "schema/nsi/modules/vector2d.nsi.json",
                            "", NULL, 0, 2);
    if (!s) return NULL;
    if (nl_nsi_session_hello(s, nsi) != NL_NSI_OK) {
        nl_nsi_session_close(s);
        return NULL;
    }
    return s;
}

static void test_three_adapters(void) {
    const char *test_name = "nsi_rt: one client vs inproc/mock/local";
    NlNsiSession *a = open_log(NL_NSI_ADAPTER_INPROC);
    NlNsiSession *b = open_log(NL_NSI_ADAPTER_MOCK);
    NlNsiSession *c = open_log(NL_NSI_ADAPTER_LOCAL);
    if (!a || !b || !c) { FAIL(test_name, "open"); nl_nsi_session_close(a); nl_nsi_session_close(b); nl_nsi_session_close(c); return; }
    if (!client_log_write(a) || !client_log_write(b) || !client_log_write(c)) {
        FAIL(test_name, "client");
        nl_nsi_session_close(a); nl_nsi_session_close(b); nl_nsi_session_close(c);
        return;
    }
    PASS(test_name);
    nl_nsi_session_close(a);
    nl_nsi_session_close(b);
    nl_nsi_session_close(c);
}

static void test_pure_identical(void) {
    const char *test_name = "nsi_rt: vector2d inproc and local match";
    NlNsiSession *a = open_vec(NL_NSI_ADAPTER_INPROC);
    NlNsiSession *b = open_vec(NL_NSI_ADAPTER_LOCAL);
    if (!a || !b) { FAIL(test_name, "open"); nl_nsi_session_close(a); nl_nsi_session_close(b); return; }
    if (!client_vec_add(a) || !client_vec_add(b)) { FAIL(test_name, "add"); nl_nsi_session_close(a); nl_nsi_session_close(b); return; }
    PASS(test_name);
    nl_nsi_session_close(a);
    nl_nsi_session_close(b);
}

static void test_replace_impl(void) {
    const char *test_name = "nsi_rt: replace impl without changing client";
    NlNsiSession *a = open_log(NL_NSI_ADAPTER_INPROC);
    NlNsiSession *b = open_log(NL_NSI_ADAPTER_MOCK);
    if (!a || !b) { FAIL(test_name, "open"); nl_nsi_session_close(a); nl_nsi_session_close(b); return; }
    if (!client_log_write(a) || !client_log_write(b)) { FAIL(test_name, "client"); nl_nsi_session_close(a); nl_nsi_session_close(b); return; }
    PASS(test_name);
    nl_nsi_session_close(a);
    nl_nsi_session_close(b);
}

static void test_auth_and_malformed(void) {
    const char *test_name = "nsi_rt: auth, malformed, extra keys fail closed";
    NlNsiSession *s = open_log(NL_NSI_ADAPTER_INPROC);
    NlNsiCall c;
    NlNsiResult r;
    if (!s) { FAIL(test_name, "open"); return; }
    memset(&c, 0, sizeof(c));
    c.method_id = "nsi:nanolang/log#write";
    c.payload_json = "{\"level\":0,\"message\":\"hi\"}";
    c.capability = "cap:nanolang/log.write";
    c.auth = "wrong";
    c.timeout_ms = 1000;
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_ERR_UNAUTHORIZED) {
        FAIL(test_name, "auth"); nl_nsi_result_free(&r); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r);
    c.auth = "tok";
    c.payload_json = "not-json";
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_ERR_MALFORMED) {
        FAIL(test_name, "malformed"); nl_nsi_result_free(&r); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r);
    c.payload_json = "{\"level\":0,\"message\":\"hi\",\"c_ptr\":1}";
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_ERR_MALFORMED) {
        FAIL(test_name, "extra"); nl_nsi_result_free(&r); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r);
    PASS(test_name);
    nl_nsi_session_close(s);
}

static void test_queue_cancel_deadline(void) {
    const char *test_name = "nsi_rt: backpressure, cancel, deadline";
    NlNsiSession *s = open_log(NL_NSI_ADAPTER_INPROC);
    NlNsiCall c;
    NlNsiResult r1;
    NlNsiResult r2;
    NlNsiResult r3;
    if (!s) { FAIL(test_name, "open"); return; }
    memset(&c, 0, sizeof(c));
    c.method_id = "nsi:nanolang/log#write";
    c.payload_json = "{\"level\":0,\"message\":\"hi\"}";
    c.capability = "cap:nanolang/log.write";
    c.auth = "tok";
    c.timeout_ms = 1000;
    c.async = 1;
    if (nl_nsi_invoke(s, &c, &r1) != NL_NSI_OK) {
        FAIL(test_name, "async1"); nl_nsi_result_free(&r1); nl_nsi_session_close(s); return;
    }
    if (nl_nsi_invoke(s, &c, &r2) != NL_NSI_ERR_BACKPRESSURE) {
        FAIL(test_name, "backpressure"); nl_nsi_result_free(&r1); nl_nsi_result_free(&r2); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r2);
    if (nl_nsi_cancel(s, r1.call_id) != NL_NSI_OK) {
        FAIL(test_name, "cancel"); nl_nsi_result_free(&r1); nl_nsi_session_close(s); return;
    }
    if (nl_nsi_take_async(s, &r3) != NL_NSI_ERR_CANCELLED) {
        FAIL(test_name, "take"); nl_nsi_result_free(&r1); nl_nsi_result_free(&r3); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r1);
    nl_nsi_result_free(&r3);
    c.async = 0;
    c.timeout_ms = 0;
    if (nl_nsi_invoke(s, &c, &r2) != NL_NSI_ERR_DEADLINE) {
        FAIL(test_name, "deadline"); nl_nsi_result_free(&r2); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r2);
    PASS(test_name);
    nl_nsi_session_close(s);
}

static void test_idemp_and_frames(void) {
    const char *test_name = "nsi_rt: idempotent write_event and frames";
    NlNsiSession *s = open_log(NL_NSI_ADAPTER_INPROC);
    NlNsiCall c;
    NlNsiResult r;
    char kind[32];
    char *frame;
    if (!s) { FAIL(test_name, "open"); return; }
    memset(&c, 0, sizeof(c));
    c.method_id = "nsi:nanolang/log#write_event";
    c.payload_json = "{}";
    c.capability = "cap:nanolang/log.write";
    c.auth = "tok";
    c.request_id = "req-1";
    c.timeout_ms = 1000;
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_OK) {
        FAIL(test_name, "first"); nl_nsi_result_free(&r); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r);
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_OK) {
        FAIL(test_name, "second"); nl_nsi_result_free(&r); nl_nsi_session_close(s); return;
    }
    nl_nsi_result_free(&r);
    frame = nl_nsi_frame_request("nsi:nanolang/log", "nsi:nanolang/log#write",
                                 "1", "{}", "req-1", "cap:nanolang/log.write",
                                 "tok", 1000, 0);
    if (!frame || nl_nsi_frame_kind(frame, kind, sizeof(kind)) != 0 || strcmp(kind, "request") != 0) {
        FAIL(test_name, "frame"); free(frame); nl_nsi_session_close(s); return;
    }
    free(frame);
    if (nl_nsi_frame_kind("{\"nsi_version\":0,\"frame\":\"cancel\"}", kind, sizeof(kind)) != 0 ||
        strcmp(kind, "cancel") != 0) {
        FAIL(test_name, "cancel-frame"); nl_nsi_session_close(s); return;
    }
    if (nl_nsi_frame_kind("not-json", kind, sizeof(kind)) == 0) {
        FAIL(test_name, "bad-frame"); nl_nsi_session_close(s); return;
    }
    if (nl_nsi_frame_kind("{\"nsi_version\":0,\"frame\":\"stream\"}", kind, sizeof(kind)) != 0 ||
        strcmp(kind, "stream") != 0 ||
        nl_nsi_frame_kind("{\"nsi_version\":0,\"frame\":\"hello\"}", kind, sizeof(kind)) != 0 ||
        strcmp(kind, "hello") != 0 ||
        nl_nsi_frame_kind("{\"nsi_version\":0,\"frame\":\"deadline\"}", kind, sizeof(kind)) != 0 ||
        strcmp(kind, "deadline") != 0 ||
        nl_nsi_frame_kind("{\"nsi_version\":0,\"frame\":\"error\"}", kind, sizeof(kind)) != 0 ||
        strcmp(kind, "error") != 0) {
        FAIL(test_name, "more-frames"); nl_nsi_session_close(s); return;
    }
    PASS(test_name);
    nl_nsi_session_close(s);
}

static void test_handles_and_schema(void) {
    const char *test_name = "nsi_rt: resource handle and schema evolution";
    NlNsi *fs = nl_nsi_load_path("schema/nsi/modules/filesystem.nsi.json");
    NlNsi *older;
    NlNsi *newer;
    const char *caps[] = { "cap:nanolang/filesystem.open" };
    NlNsiSession *s;
    NlNsiCall c;
    NlNsiResult r;
    const NlNsiHandle *h;
    const char *oldp = "/tmp/nl_nsi_old.json";
    const char *newp = "/tmp/nl_nsi_new.json";
    FILE *fp;
    if (!fs) { FAIL(test_name, "load-fs"); return; }
    s = nl_nsi_session_open(fs, NL_NSI_ADAPTER_INPROC, "schema/nsi/modules/filesystem.nsi.json",
                            "tok", caps, 1, 2);
    if (!s || nl_nsi_session_hello(s, fs) != NL_NSI_OK) {
        FAIL(test_name, "open"); nl_nsi_session_close(s); nl_nsi_free(fs); return;
    }
    memset(&c, 0, sizeof(c));
    c.method_id = "nsi:nanolang/filesystem#open";
    c.payload_json = "{\"path\":\"/tmp\"}";
    c.capability = "cap:nanolang/filesystem.open";
    c.auth = "tok";
    c.timeout_ms = 1000;
    if (nl_nsi_invoke(s, &c, &r) != NL_NSI_OK || nl_nsi_handle_count(s) != 1) {
        FAIL(test_name, "open-handle"); nl_nsi_result_free(&r); nl_nsi_session_close(s); nl_nsi_free(fs); return;
    }
    h = nl_nsi_handle_at(s, 0);
    if (!h || !h->live || h->generation == 0 ||
        strstr(h->type_id, "File") == NULL ||
        strcmp(h->service_id, "nsi:nanolang/filesystem") != 0 ||
        (h->rights & NL_NSI_RIGHT_READ) == 0 ||
        strstr(r.payload_json, "FILE") != NULL) {
        FAIL(test_name, "handle-fields"); nl_nsi_result_free(&r); nl_nsi_session_close(s); nl_nsi_free(fs); return;
    }
    nl_nsi_result_free(&r);
    nl_nsi_session_close(s);
    nl_nsi_free(fs);
    fp = fopen(oldp, "w");
    if (!fp) { FAIL(test_name, "write"); return; }
    fprintf(fp, "{\"nsi_version\":0,\"interface\":{\"id\":\"nsi:nanolang/c\",\"name\":\"c\"},"
                "\"methods\":[{\"id\":\"nsi:nanolang/c#ping\",\"name\":\"ping\",\"params\":[]}],"
                "\"types\":[],\"errors\":[],\"capabilities\":[]}");
    fclose(fp);
    fp = fopen(newp, "w");
    if (!fp) { FAIL(test_name, "write2"); unlink(oldp); return; }
    fprintf(fp, "{\"nsi_version\":0,\"interface\":{\"id\":\"nsi:nanolang/c\",\"name\":\"c\"},"
                "\"methods\":["
                "{\"id\":\"nsi:nanolang/c#ping\",\"name\":\"ping\",\"params\":[]},"
                "{\"id\":\"nsi:nanolang/c#pong\",\"name\":\"pong\",\"params\":[]}],"
                "\"types\":[],\"errors\":[],\"capabilities\":[]}");
    fclose(fp);
    older = nl_nsi_load_path(oldp);
    newer = nl_nsi_load_path(newp);
    unlink(oldp);
    unlink(newp);
    if (!older || !newer || nl_nsi_compat(older, newer) != NL_NSI_COMPAT_OK) {
        FAIL(test_name, "compat-ok"); nl_nsi_free(older); nl_nsi_free(newer); return;
    }
    s = nl_nsi_session_open(newer, NL_NSI_ADAPTER_INPROC, "", "", NULL, 0, 1);
    if (!s || nl_nsi_session_hello(s, older) != NL_NSI_OK) {
        FAIL(test_name, "hello-ok"); nl_nsi_session_close(s); nl_nsi_free(older); nl_nsi_free(newer); return;
    }
    nl_nsi_session_close(s);
    s = nl_nsi_session_open(older, NL_NSI_ADAPTER_INPROC, "", "", NULL, 0, 1);
    if (!s || nl_nsi_session_hello(s, newer) == NL_NSI_OK) {
        FAIL(test_name, "hello-breaking"); nl_nsi_session_close(s); nl_nsi_free(older); nl_nsi_free(newer); return;
    }
    nl_nsi_session_close(s);
    nl_nsi_free(older);
    nl_nsi_free(newer);
    PASS(test_name);
}

static void test_privilege_methods(void) {
    const char *test_name = "nsi_rt: process/net/audio/graphics/gpu/python";
    struct { const char *schema; const char *method; const char *payload; const char *cap; } cases[] = {
        { "schema/nsi/modules/process.nsi.json", "nsi:nanolang/process#spawn", "{\"argv\":\"true\"}", "cap:nanolang/process.spawn" },
        { "schema/nsi/modules/net.nsi.json", "nsi:nanolang/net#connect", "{\"endpoint\":\"127.0.0.1:1\"}", "cap:nanolang/net.connect" },
        { "schema/nsi/modules/audio.nsi.json", "nsi:nanolang/audio#write_frame", "{\"bytes\":\"AA\"}", "cap:nanolang/audio.write_frame" },
        { "schema/nsi/modules/graphics.nsi.json", "nsi:nanolang/graphics#present", "{\"surface\":0}", "cap:nanolang/graphics.present" },
        { "schema/nsi/modules/gpu.nsi.json", "nsi:nanolang/gpu#submit", "{\"queue\":0}", "cap:nanolang/gpu.submit" },
        { "schema/nsi/modules/python.nsi.json", "nsi:nanolang/python#eval", "{\"source\":\"1\"}", "cap:nanolang/python.eval" },
    };
    size_t i;
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        NlNsi *nsi = nl_nsi_load_path(cases[i].schema);
        const char *caps[1];
        NlNsiSession *s;
        NlNsiCall c;
        NlNsiResult r;
        caps[0] = cases[i].cap;
        if (!nsi) { FAIL(test_name, "load"); return; }
        s = nl_nsi_session_open(nsi, NL_NSI_ADAPTER_MOCK, cases[i].schema, "tok", caps, 1, 2);
        if (!s || nl_nsi_session_hello(s, nsi) != NL_NSI_OK) {
            FAIL(test_name, "open"); nl_nsi_session_close(s); nl_nsi_free(nsi); return;
        }
        memset(&c, 0, sizeof(c));
        c.method_id = cases[i].method;
        c.payload_json = cases[i].payload;
        c.capability = cases[i].cap;
        c.auth = "tok";
        c.timeout_ms = 1000;
        if (nl_nsi_invoke(s, &c, &r) != NL_NSI_OK || nl_nsi_handle_count(s) < 1) {
            FAIL(test_name, cases[i].method);
            nl_nsi_result_free(&r); nl_nsi_session_close(s); nl_nsi_free(nsi); return;
        }
        nl_nsi_result_free(&r);
        nl_nsi_session_close(s);
        nl_nsi_free(nsi);
    }
    PASS(test_name);
}

int main(void) {
    test_three_adapters();
    test_pure_identical();
    test_replace_impl();
    test_auth_and_malformed();
    test_queue_cancel_deadline();
    test_idemp_and_frames();
    test_handles_and_schema();
    test_privilege_methods();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
