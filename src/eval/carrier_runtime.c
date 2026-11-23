/* I connect explicit constructor provenance to a qualified interpreter session. */
#include "carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct NlEvalCarrierSession {
    NlCarrierValues *values;
    size_t depth;
    NlCarrierRef *roots, *host_roots;
    size_t root_count, root_capacity, host_count;
    size_t *frames, frame_capacity;
    bool collect;
};
static NlCarrierHeapStats last_retired;
static size_t retired_sessions;
static void ownership_failure(void) {
    fputs("I cannot account for interpreter carrier ownership.\n", stderr);
    exit(1);
}
static bool record_list_slot(Value value, int index, void *context) {
    Environment *env = context;
    char **names = NULL; Type *types = NULL; int count = 0;
    const char *name = NULL;
    if (value.type == VAL_STRUCT && value.as.struct_val && index < value.as.struct_val->field_count) {
        StructValue *r = value.as.struct_val;
        StructDef *def = env_get_struct(env, r->struct_name);
        if (!def) return false;
        names = def->field_names; types = def->field_types; count = def->field_count;
        name = r->field_names[index];
    } else if (value.type == VAL_UNION && value.as.union_val && index < value.as.union_val->field_count) {
        UnionValue *u = value.as.union_val;
        UnionDef *def = env_get_union(env, u->union_name);
        if (!def || u->variant_index < 0 || u->variant_index >= def->variant_count) return false;
        int variant = u->variant_index;
        names = def->variant_field_names[variant]; types = def->variant_field_types[variant];
        count = def->variant_field_counts[variant]; name = u->field_names[index];
    }
    if (!name || !names || !types) return false;
    for (int i = 0; i < count; ++i)
        if (strcmp(name, names[i]) == 0) return types[i] == TYPE_LIST_GENERIC;
    return false;
}
bool env_carrier_enable(Environment *env) {
    if (!env) return false;
    if (env->carrier_session) return true;
    struct NlEvalCarrierSession *session = calloc(1, sizeof *session);
    if (!session) return false;
    session->values = nl_carrier_values_new();
    if (!session->values) { free(session); return false; }
    nl_carrier_values_set_slots(session->values, record_list_slot, env);
    env->carrier_session = session; return true;
}
static void *reserve(void *data, size_t *capacity, size_t count, size_t width) {
    if (count <= *capacity) return data;
    size_t next = *capacity ? *capacity : 16;
    while (next < count) {
        if (next > SIZE_MAX / 2) ownership_failure();
        next *= 2;
    }
    if (next > SIZE_MAX / width) ownership_failure();
    void *result = realloc(data, next * width);
    if (!result) ownership_failure();
    *capacity = next; return result;
}
void env_carrier_hold(Environment *env, NlCarrierRef value) {
    if (!env_carrier_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (!s->depth) return;
    /* I never retain transient STRING/FUNCTION scalar storage as graph roots. */
    if (!value.record_list && value.value.type != VAL_STRUCT && value.value.type != VAL_UNION &&
        value.value.type != VAL_TUPLE && value.value.type != VAL_ARRAY) return;
    if (s->root_count == SIZE_MAX) ownership_failure();
    s->roots = reserve(s->roots, &s->root_capacity, s->root_count + 1, sizeof *s->roots);
    s->roots[s->root_count++] = value;
}
void env_carrier_edges(Environment *env, Value owner, const NlCarrierRef *edges, size_t count) {
    if (env_carrier_enabled(env) && !nl_carrier_values_edges(env->carrier_session->values,
                                                           nl_carrier_value(owner), edges, count))
        ownership_failure();
}
size_t env_carrier_root_mark(Environment *env) {
    return env_carrier_enabled(env) ? env->carrier_session->root_count : 0;
}
void env_carrier_root_reset(Environment *env, size_t mark) {
    if (!env_carrier_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (!s->depth || mark < s->frames[s->depth - 1] || mark > s->root_count) ownership_failure();
    s->root_count = mark;
}
void env_carrier_enter(Environment *env) {
    if (!env_carrier_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (s->depth == SIZE_MAX) ownership_failure();
    s->frames = reserve(s->frames, &s->frame_capacity, s->depth + 1, sizeof *s->frames);
    s->frames[s->depth++] = s->root_count;
}
void env_carrier_leave(Environment *env) {
    if (!env_carrier_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (!s->depth) ownership_failure();
    s->root_count = s->frames[--s->depth];
}
void env_carrier_leave_value(Environment *env, Value result, bool record_list) {
    env_carrier_leave(env);
    env_carrier_hold(env, (NlCarrierRef){result, record_list});
}
bool env_carrier_enable_collection(Environment *env, const NlCarrierRef *roots, size_t count) {
    if (!env_carrier_enabled(env) || (count && !roots) || count > SIZE_MAX / sizeof *roots) return false;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (s->depth) return false;
    NlCarrierRef *copy = count ? malloc(count * sizeof *copy) : NULL;
    if (count && !copy) return false;
    if (count) memcpy(copy, roots, count * sizeof *copy);
    free(s->host_roots); s->host_roots = copy; s->host_count = count; s->collect = true;
    return true;
}
bool env_carrier_collect(Environment *env) {
    if (!env_carrier_enabled(env) || !env->carrier_session->collect) return true;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (env->symbol_count < 0) return false;
    size_t count = s->root_count;
    if (s->host_count > SIZE_MAX - count) return false;
    count += s->host_count;
    if ((size_t)env->symbol_count > SIZE_MAX - count) return false;
    count += (size_t)env->symbol_count;
    if (count > SIZE_MAX / sizeof(NlCarrierRef)) return false;
    NlCarrierRef *roots = count ? malloc(count * sizeof *roots) : NULL;
    if (count && !roots) return false;
    size_t n = 0;
    for (size_t i = 0; i < s->root_count; ++i) roots[n++] = s->roots[i];
    for (size_t i = 0; i < s->host_count; ++i) roots[n++] = s->host_roots[i];
    for (int i = 0; i < env->symbol_count; ++i) {
        Symbol *symbol = &env->symbols[i];
        /* Checker-retained declarations have no runtime payload yet. */
        if (symbol->value.type == VAL_VOID) continue;
        roots[n++] = (NlCarrierRef){symbol->value, symbol->type == TYPE_LIST_GENERIC};
    }
    bool ok = nl_carrier_values_collect(s->values, roots, n);
    free(roots); return ok;
}
void env_carrier_track(Environment *env, Value value, bool copies) {
    if (!env_carrier_enabled(env)) return;
    if (!nl_carrier_values_adopt(env->carrier_session->values, nl_carrier_value(value)))
        ownership_failure();
    env_carrier_hold(env, nl_carrier_value(value));
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
    if (!env_carrier_enabled(env)) return;
    NlCarrierRef value = nl_carrier_record_list(list);
    if (!nl_carrier_values_adopt(env->carrier_session->values, value)) ownership_failure();
    env_carrier_hold(env, value);
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
    free(session->roots); free(session->frames); free(session->host_roots);
    free(session); env->carrier_session = NULL; retired_sessions++;
}
NlCarrierHeapStats env_carrier_last_retired_stats(void) { return last_retired; }
size_t env_carrier_retired_sessions(void) { return retired_sessions; }
