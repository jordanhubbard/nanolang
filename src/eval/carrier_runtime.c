/* I connect explicit constructor provenance to a qualified interpreter session. */
#include "carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>

struct NlEvalCarrierSession {
    NlCarrierValues *values;
    unsigned depth;
};
static NlCarrierHeapStats last_retired;
static size_t retired_sessions;
static void ownership_failure(void) {
    fputs("I cannot account for interpreter carrier ownership.\n", stderr);
    exit(1);
}
bool env_carrier_enable(Environment *env) {
    if (!env) return false;
    if (env->carrier_session) return true;
    struct NlEvalCarrierSession *session = calloc(1, sizeof *session);
    if (!session) return false;
    session->values = nl_carrier_values_new();
    if (!session->values) { free(session); return false; }
    env->carrier_session = session; return true;
}
void env_carrier_enter(Environment *env) {
    if (env_carrier_enabled(env)) env->carrier_session->depth++;
}
void env_carrier_leave(Environment *env) {
    if (env_carrier_enabled(env)) {
        if (!env->carrier_session->depth) ownership_failure();
        env->carrier_session->depth--;
    }
}
void env_carrier_track(Environment *env, Value value, bool copies) {
    if (!env_carrier_enabled(env)) return;
    if (!nl_carrier_values_adopt(env->carrier_session->values, nl_carrier_value(value)))
        ownership_failure();
    if (!copies) return;
    Value *fields = NULL; int count = 0;
    if (value.type == VAL_STRUCT) { fields = value.as.struct_val->field_values; count = value.as.struct_val->field_count; }
    if (value.type == VAL_UNION) { fields = value.as.union_val->field_values; count = value.as.union_val->field_count; }
    for (int i = 0; i < count; ++i)
        if (fields[i].type == VAL_STRUCT && fields[i].as.struct_val)
            env_carrier_track(env, fields[i], true);
}
void env_carrier_track_binding(Environment *env, Value value) {
    if (env_carrier_enabled(env))
        env_carrier_track(env, value, env->carrier_session->depth > 0);
}
void env_carrier_track_list(Environment *env, DynArray *list) {
    if (env_carrier_enabled(env) &&
        !nl_carrier_values_adopt(env->carrier_session->values, nl_carrier_record_list(list)))
        ownership_failure();
}
bool env_carrier_owns(Environment *env, Value value) {
    return env_carrier_enabled(env) &&
        nl_carrier_values_contains(env->carrier_session->values, nl_carrier_value(value));
}
void env_carrier_export(Environment *env, Value value, bool list) {
    if (!env_carrier_enabled(env)) return;
    NlCarrierRef root = list ? nl_carrier_record_list((DynArray *)(intptr_t)value.as.int_val) : nl_carrier_value(value);
    if (!nl_carrier_values_transfer(env->carrier_session->values, &root, 1)) ownership_failure();
}
NlCarrierHeapStats env_carrier_stats(const Environment *env) {
    return nl_carrier_values_stats(env_carrier_enabled(env) ? env->carrier_session->values : NULL);
}
void env_carrier_close(Environment *env) {
    if (!env_carrier_enabled(env)) return;
    struct NlEvalCarrierSession *session = env->carrier_session;
    if (session->depth || !nl_carrier_values_clear(session->values)) ownership_failure();
    last_retired = nl_carrier_values_stats(session->values);
    if (!nl_carrier_values_free(session->values)) ownership_failure();
    free(session); env->carrier_session = NULL; retired_sessions++;
}
NlCarrierHeapStats env_carrier_last_retired_stats(void) { return last_retired; }
size_t env_carrier_retired_sessions(void) { return retired_sessions; }
