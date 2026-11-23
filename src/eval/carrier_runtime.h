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
void env_carrier_leave_value(Environment *env, Value result, bool record_list);
/* My private collector requires complete valid external roots supplied by its
 * qualification caller. I do not enable collection through ordinary creation. */
bool env_carrier_enable_collection(Environment *env, const NlCarrierRef *host_roots, size_t count);
bool env_carrier_collect(Environment *env);
void env_carrier_hold(Environment *env, NlCarrierRef value);
void env_carrier_edges(Environment *env, Value owner, const NlCarrierRef *edges, size_t count);
typedef struct { size_t roots, strings; } NlCarrierScopeMark;
typedef struct { size_t registered, released, exported, live; } NlCarrierStringStats;
NlCarrierScopeMark env_carrier_root_mark(Environment *env);
void env_carrier_root_reset(Environment *env, NlCarrierScopeMark mark);
/* I accept only a fresh evaluator-owned STRING reference, never a borrowed read. */
void env_carrier_track_string(Environment *env, Value value);
void env_carrier_borrow_string(Environment *env, Value value);
/* A binding gains its own reference; the temporary remains scoped. */
bool env_carrier_retain_string(Environment *env, Value value);
NlCarrierStringStats env_carrier_string_stats(const Environment *env);
NlCarrierStringStats env_carrier_last_string_stats(void);
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
