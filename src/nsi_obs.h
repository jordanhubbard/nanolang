#ifndef NL_NSI_OBS_H
#define NL_NSI_OBS_H

#include <stddef.h>
#include <stdint.h>

#define NL_OBS_OK 0
#define NL_OBS_ERR 1
#define NL_OBS_MAX_SPANS 32
#define NL_OBS_ID 64

typedef struct NlObs NlObs;

NlObs *nl_obs_create(void);
void nl_obs_destroy(NlObs *o);

void nl_obs_new_trace(NlObs *o, char *out, size_t n);
void nl_obs_new_audit(NlObs *o, char *out, size_t n);

int nl_obs_record_span(NlObs *o, const char *trace_id, const char *boundary,
                       const char *name);
int nl_obs_span_count(const NlObs *o);
const char *nl_obs_span_boundary(const NlObs *o, int i);
const char *nl_obs_span_trace(const NlObs *o, int i);

int nl_obs_emit(NlObs *o, const char *trace_id, const char *name, int64_t value);
int nl_obs_metric_count(const NlObs *o);

int nl_obs_provenance(NlObs *o, const char *source, const char *module,
                      const char *iface, const char *impl,
                      const char *policy, const char *output);
const char *nl_obs_prov_source(const NlObs *o);
const char *nl_obs_prov_module(const NlObs *o);
const char *nl_obs_prov_iface(const NlObs *o);
const char *nl_obs_prov_impl(const NlObs *o);
const char *nl_obs_prov_policy(const NlObs *o);
const char *nl_obs_prov_output(const NlObs *o);
const char *nl_obs_audit_id(const NlObs *o);
const char *nl_obs_trace_id(const NlObs *o);

int nl_obs_localize_log(NlObs *o, const char *locale, const char *msg);
const char *nl_obs_localized(const NlObs *o);

#endif
