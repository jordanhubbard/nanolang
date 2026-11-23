#ifndef NANO_EVAL_CARRIER_RUNTIME_H
#define NANO_EVAL_CARRIER_RUNTIME_H
#include "carrier_values.h"

/* I keep qualification explicit until every runtime/host/GC root is adapted. */
bool env_carrier_enable(Environment *env);
static inline bool env_carrier_enabled(const Environment *env) {
    return env && env->carrier_session;
}
void env_carrier_enter(Environment *env);
void env_carrier_leave(Environment *env);
void env_carrier_track(Environment *env, Value value, bool fresh_record_copies);
void env_carrier_track_binding(Environment *env, Value value);
void env_carrier_track_list(Environment *env, DynArray *list);
bool env_carrier_owns(Environment *env, Value value);
void env_carrier_export(Environment *env, Value value, bool record_list);
void env_carrier_close(Environment *env);
NlCarrierHeapStats env_carrier_stats(const Environment *env);
NlCarrierHeapStats env_carrier_last_retired_stats(void);
size_t env_carrier_retired_sessions(void);
#endif
