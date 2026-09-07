#include "nsi_obs.h"
#include "nsi_cap.h"
#include "nsi_fabric.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while (0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while (0)

static void test_spans(void) {
    const char *test_name = "obs: one trace across nanovm, router, service, host";
    NlObs *o = nl_obs_create();
    char trace[NL_OBS_ID];
    int i;
    if (!o) { FAIL(test_name, "create"); return; }
    nl_obs_new_trace(o, trace, sizeof(trace));
    if (nl_obs_record_span(o, trace, "nanovm", "trap.print") != NL_OBS_OK ||
        nl_obs_record_span(o, trace, "router", "dispatch") != NL_OBS_OK ||
        nl_obs_record_span(o, trace, "service", "log.write") != NL_OBS_OK ||
        nl_obs_record_span(o, trace, "host", "posix") != NL_OBS_OK) {
        FAIL(test_name, "record");
        nl_obs_destroy(o);
        return;
    }
    if (nl_obs_span_count(o) != 4) { FAIL(test_name, "count"); nl_obs_destroy(o); return; }
    for (i = 0; i < 4; i++) {
        if (strcmp(nl_obs_span_trace(o, i), trace) != 0) {
            FAIL(test_name, "trace-mismatch");
            nl_obs_destroy(o);
            return;
        }
    }
    if (strcmp(nl_obs_span_boundary(o, 0), "nanovm") != 0 ||
        strcmp(nl_obs_span_boundary(o, 3), "host") != 0) {
        FAIL(test_name, "boundary");
        nl_obs_destroy(o);
        return;
    }
    PASS(test_name);
    nl_obs_destroy(o);
}

static void test_metrics_provenance(void) {
    const char *test_name = "obs: metrics and provenance fields";
    NlObs *o = nl_obs_create();
    char trace[NL_OBS_ID];
    nl_obs_new_trace(o, trace, sizeof(trace));
    if (nl_obs_emit(o, trace, "calls", 3) != NL_OBS_OK || nl_obs_metric_count(o) != 1) {
        FAIL(test_name, "metric");
        nl_obs_destroy(o);
        return;
    }
    if (nl_obs_provenance(o, "src/main.nano", "mod.nvm", "nsi:nanolang/log",
                          "inproc-log", "least-privilege", "logged") != NL_OBS_OK) {
        FAIL(test_name, "prov");
        nl_obs_destroy(o);
        return;
    }
    if (strcmp(nl_obs_prov_source(o), "src/main.nano") != 0 ||
        strcmp(nl_obs_prov_module(o), "mod.nvm") != 0 ||
        strcmp(nl_obs_prov_iface(o), "nsi:nanolang/log") != 0 ||
        strcmp(nl_obs_prov_impl(o), "inproc-log") != 0 ||
        strcmp(nl_obs_prov_policy(o), "least-privilege") != 0 ||
        strcmp(nl_obs_prov_output(o), "logged") != 0 ||
        nl_obs_audit_id(o)[0] == 0) {
        FAIL(test_name, "fields");
        nl_obs_destroy(o);
        return;
    }
    PASS(test_name);
    nl_obs_destroy(o);
}

static void test_locale_stable_audit(void) {
    const char *test_name = "obs: localized log does not change audit fields";
    NlObs *o = nl_obs_create();
    char audit_before[NL_OBS_ID];
    char trace_before[NL_OBS_ID];
    char trace[NL_OBS_ID];
    nl_obs_new_trace(o, trace, sizeof(trace));
    nl_obs_provenance(o, "a.nano", "a.nvm", "nsi:x", "impl", "policy", "out");
    snprintf(audit_before, sizeof(audit_before), "%s", nl_obs_audit_id(o));
    snprintf(trace_before, sizeof(trace_before), "%s", nl_obs_trace_id(o));
    nl_obs_localize_log(o, "ja", "エラー");
    nl_obs_localize_log(o, "en", "error");
    if (strcmp(nl_obs_audit_id(o), audit_before) != 0 ||
        strcmp(nl_obs_trace_id(o), trace_before) != 0 ||
        strcmp(nl_obs_prov_source(o), "a.nano") != 0 ||
        strcmp(nl_obs_localized(o), "error") != 0) {
        FAIL(test_name, "mutated");
        nl_obs_destroy(o);
        return;
    }
    PASS(test_name);
    nl_obs_destroy(o);
}

static void test_fabric_trace(void) {
    const char *test_name = "obs: fabric stores the assigned trace and audit ids";
    NlObs *o = nl_obs_create();
    NlHost host = nl_host_inproc();
    NlFabric *f = nl_fabric_create(&host);
    NlCap cap;
    char trace[NL_OBS_ID];
    char audit[NL_OBS_ID];
    char out[64];
    NlFailClass cls;
    if (!o || !f) { FAIL(test_name, "create"); nl_obs_destroy(o); nl_fabric_destroy(f); return; }
    if (nl_fabric_register_core_services(f) != 0 || nl_fabric_start(f) != 0) {
        FAIL(test_name, "start");
        nl_obs_destroy(o);
        nl_fabric_destroy(f);
        return;
    }
    nl_obs_new_trace(o, trace, sizeof(trace));
    nl_obs_new_audit(o, audit, sizeof(audit));
    memset(&cap, 0, sizeof(cap));
    if (nl_fabric_call(f, "log", "write", "hi", &cap, 10, "r1", 1, trace, audit,
                       out, sizeof(out), &cls) != NL_FAB_OK) {
        FAIL(test_name, "call");
        nl_obs_destroy(o);
        nl_fabric_destroy(f);
        return;
    }
    if (strcmp(nl_fabric_last_trace(f, "log"), trace) != 0 ||
        strcmp(nl_fabric_last_audit(f, "log"), audit) != 0) {
        FAIL(test_name, "stored");
        nl_obs_destroy(o);
        nl_fabric_destroy(f);
        return;
    }
    PASS(test_name);
    nl_obs_destroy(o);
    nl_fabric_destroy(f);
}

int main(void) {
    printf("NSI observability tests\n");
    test_spans();
    test_metrics_provenance();
    test_locale_stable_audit();
    test_fabric_trace();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
