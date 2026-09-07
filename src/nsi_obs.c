#include "nsi_obs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void bounded_copy(char *dest, size_t dest_size, const char *src) {
    size_t n;
    if (!dest || dest_size == 0) return;
    if (!src) { dest[0] = 0; return; }
    n = strlen(src);
    if (n >= dest_size) n = dest_size - 1;
    memcpy(dest, src, n);
    dest[n] = 0;
}

typedef struct {
    char trace[NL_OBS_ID];
    char boundary[16];
    char name[32];
} NlObsSpan;

typedef struct {
    char name[32];
    int64_t value;
    char trace[NL_OBS_ID];
} NlObsMetric;

struct NlObs {
    uint32_t next_trace;
    uint32_t next_audit;
    char trace_id[NL_OBS_ID];
    char audit_id[NL_OBS_ID];
    char source[80];
    char module[80];
    char iface[80];
    char impl[80];
    char policy[80];
    char output[80];
    char localized[160];
    char locale[16];
    NlObsSpan spans[NL_OBS_MAX_SPANS];
    int span_n;
    NlObsMetric metrics[NL_OBS_MAX_SPANS];
    int metric_n;
};

NlObs *nl_obs_create(void) {
    NlObs *o = calloc(1, sizeof(*o));
    if (o) o->next_trace = 1;
    return o;
}

void nl_obs_destroy(NlObs *o) {
    free(o);
}

void nl_obs_new_trace(NlObs *o, char *out, size_t n) {
    if (!o || !out || n == 0) return;
    o->next_trace++;
    snprintf(o->trace_id, sizeof(o->trace_id), "TRACE%04u", (unsigned)o->next_trace);
    bounded_copy(out, n, o->trace_id);
}

void nl_obs_new_audit(NlObs *o, char *out, size_t n) {
    if (!o || !out || n == 0) return;
    o->next_audit++;
    snprintf(o->audit_id, sizeof(o->audit_id), "AUDIT%04u", (unsigned)o->next_audit);
    bounded_copy(out, n, o->audit_id);
}

int nl_obs_record_span(NlObs *o, const char *trace_id, const char *boundary,
                       const char *name) {
    NlObsSpan *s;
    if (!o || !trace_id || !boundary || !name) return NL_OBS_ERR;
    if (strcmp(boundary, "nanovm") != 0 && strcmp(boundary, "router") != 0 &&
        strcmp(boundary, "service") != 0 && strcmp(boundary, "host") != 0)
        return NL_OBS_ERR;
    if (o->span_n >= NL_OBS_MAX_SPANS) return NL_OBS_ERR;
    s = &o->spans[o->span_n++];
    bounded_copy(s->trace, sizeof(s->trace), trace_id);
    bounded_copy(s->boundary, sizeof(s->boundary), boundary);
    bounded_copy(s->name, sizeof(s->name), name);
    bounded_copy(o->trace_id, sizeof(o->trace_id), trace_id);
    return NL_OBS_OK;
}

int nl_obs_span_count(const NlObs *o) {
    return o ? o->span_n : 0;
}

const char *nl_obs_span_boundary(const NlObs *o, int i) {
    if (!o || i < 0 || i >= o->span_n) return "";
    return o->spans[i].boundary;
}

const char *nl_obs_span_trace(const NlObs *o, int i) {
    if (!o || i < 0 || i >= o->span_n) return "";
    return o->spans[i].trace;
}

int nl_obs_emit(NlObs *o, const char *trace_id, const char *name, int64_t value) {
    NlObsMetric *m;
    if (!o || !name) return NL_OBS_ERR;
    if (o->metric_n >= NL_OBS_MAX_SPANS) return NL_OBS_ERR;
    m = &o->metrics[o->metric_n++];
    bounded_copy(m->name, sizeof(m->name), name);
    bounded_copy(m->trace, sizeof(m->trace), trace_id ? trace_id : "");
    m->value = value;
    return NL_OBS_OK;
}

int nl_obs_metric_count(const NlObs *o) {
    return o ? o->metric_n : 0;
}

int nl_obs_provenance(NlObs *o, const char *source, const char *module,
                      const char *iface, const char *impl,
                      const char *policy, const char *output) {
    if (!o) return NL_OBS_ERR;
    bounded_copy(o->source, sizeof(o->source), source);
    bounded_copy(o->module, sizeof(o->module), module);
    bounded_copy(o->iface, sizeof(o->iface), iface);
    bounded_copy(o->impl, sizeof(o->impl), impl);
    bounded_copy(o->policy, sizeof(o->policy), policy);
    bounded_copy(o->output, sizeof(o->output), output);
    if (!o->audit_id[0]) {
        char tmp[NL_OBS_ID];
        nl_obs_new_audit(o, tmp, sizeof(tmp));
    }
    return NL_OBS_OK;
}

const char *nl_obs_prov_source(const NlObs *o) { return o ? o->source : ""; }
const char *nl_obs_prov_module(const NlObs *o) { return o ? o->module : ""; }
const char *nl_obs_prov_iface(const NlObs *o) { return o ? o->iface : ""; }
const char *nl_obs_prov_impl(const NlObs *o) { return o ? o->impl : ""; }
const char *nl_obs_prov_policy(const NlObs *o) { return o ? o->policy : ""; }
const char *nl_obs_prov_output(const NlObs *o) { return o ? o->output : ""; }
const char *nl_obs_audit_id(const NlObs *o) { return o ? o->audit_id : ""; }
const char *nl_obs_trace_id(const NlObs *o) { return o ? o->trace_id : ""; }

int nl_obs_localize_log(NlObs *o, const char *locale, const char *msg) {
    if (!o) return NL_OBS_ERR;
    bounded_copy(o->locale, sizeof(o->locale), locale);
    bounded_copy(o->localized, sizeof(o->localized), msg);
    return NL_OBS_OK;
}

const char *nl_obs_localized(const NlObs *o) {
    return o ? o->localized : "";
}
