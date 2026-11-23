/* I connect explicit constructor provenance to a qualified interpreter session. */
#include "carrier_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

struct NlEvalCarrierSession {
    NlCarrierValues *values;
    size_t depth;
    NlCarrierRef *roots, *host_roots, *collection_roots;
    size_t root_count, root_capacity, host_count, collection_capacity;
    NlCarrierScopeMark *frames;
    size_t frame_capacity;
    char **strings;
    size_t string_count, string_capacity;
    NlCarrierStringStats string_stats;
    bool collect, collecting;
};
static NlCarrierHeapStats last_retired;
static NlCarrierStringStats last_strings;
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
bool env_carrier_enable_strings(Environment *env) {
    if (!env) return false;
    if (env->carrier_session) return true;
    env->carrier_session = calloc(1, sizeof *env->carrier_session);
    return env->carrier_session != NULL;
}
bool env_carrier_enabled(const Environment *env) {
    return env_carrier_scopes_enabled(env) && env->carrier_session->values;
}
bool env_carrier_enable(Environment *env) {
    bool existing = env_carrier_scopes_enabled(env);
    if (!env_carrier_enable_strings(env)) return false;
    struct NlEvalCarrierSession *session = env->carrier_session;
    if (session->values) return true;
    /* I do not change ownership beneath an active STRING-only frame. */
    if (session->depth) return false;
    session->values = nl_carrier_values_new();
    if (!session->values) {
        if (!existing) { free(session); env->carrier_session = NULL; }
        return false;
    }
    nl_carrier_values_set_slots(session->values, record_list_slot, env);
    return true;
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
static void track_string_reference(Environment *env, Value value, bool fresh) {
    if (!env_carrier_scopes_enabled(env) || value.type != VAL_STRING) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (!s->depth) return;
    char *pointer = value.as.string_val;
    if (!pointer || !gc_is_managed(pointer)) ownership_failure();
    for (size_t i = 0; i < s->string_count; ++i)
        if (fresh && s->strings[i] == pointer) ownership_failure();
    if (s->string_count == SIZE_MAX) ownership_failure();
    s->strings = reserve(s->strings, &s->string_capacity, s->string_count + 1, sizeof *s->strings);
    s->strings[s->string_count++] = pointer;
    s->string_stats.registered++; s->string_stats.live++;
}
void env_carrier_track_string(Environment *env, Value value) {
    track_string_reference(env, value, true);
}
void env_carrier_borrow_string(Environment *env, Value value) {
    if (!env_carrier_scopes_enabled(env) || !env->carrier_session->depth || value.type != VAL_STRING ||
        !value.as.string_val || !gc_is_managed(value.as.string_val)) return;
    gc_retain(value.as.string_val);
    track_string_reference(env, value, false);
}
bool env_carrier_retain_string(Environment *env, Value value) {
    if (!env_carrier_scopes_enabled(env) || value.type != VAL_STRING) return false;
    struct NlEvalCarrierSession *s = env->carrier_session;
    for (size_t i = 0; i < s->string_count; ++i)
        if (s->strings[i] == value.as.string_val) { gc_retain(value.as.string_val); return true; }
    return false;
}
NlCarrierScopeMark env_carrier_root_mark(Environment *env) {
    if (!env_carrier_scopes_enabled(env)) return (NlCarrierScopeMark){0};
    struct NlEvalCarrierSession *s = env->carrier_session;
    return (NlCarrierScopeMark){s->root_count, s->string_count};
}
static void retire_strings(struct NlEvalCarrierSession *s, size_t first, const char *result) {
    bool keep = false;
    for (size_t i = first; i < s->string_count; ++i) {
        char *pointer = s->strings[i];
        if (pointer == result && !keep) keep = true;
        else { gc_release(pointer); s->string_stats.released++; s->string_stats.live--; }
    }
    s->string_count = first;
    if (keep) {
        if (s->depth) s->strings[s->string_count++] = (char *)result;
        else { s->string_stats.exported++; s->string_stats.live--; }
    }
}
void env_carrier_root_reset(Environment *env, NlCarrierScopeMark mark) {
    if (!env_carrier_scopes_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (!s->depth || mark.roots < s->frames[s->depth - 1].roots || mark.roots > s->root_count ||
        mark.strings < s->frames[s->depth - 1].strings || mark.strings > s->string_count) ownership_failure();
    s->root_count = mark.roots;
    retire_strings(s, mark.strings, NULL);
}
void env_carrier_enter(Environment *env) {
    if (!env_carrier_scopes_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (s->depth == SIZE_MAX) ownership_failure();
    s->frames = reserve(s->frames, &s->frame_capacity, s->depth + 1, sizeof *s->frames);
    s->frames[s->depth++] = (NlCarrierScopeMark){s->root_count, s->string_count};
}
static void finish_frame(Environment *env, const Value *result) {
    if (!env_carrier_scopes_enabled(env)) return;
    struct NlEvalCarrierSession *s = env->carrier_session;
    if (!s->depth) ownership_failure();
    NlCarrierScopeMark mark = s->frames[--s->depth];
    s->root_count = mark.roots;
    retire_strings(s, mark.strings, result && result->type == VAL_STRING ? result->as.string_val : NULL);
}
void env_carrier_leave(Environment *env) { finish_frame(env, NULL); }
void env_carrier_leave_value(Environment *env, Value result, bool record_list) {
    finish_frame(env, &result);
    env_carrier_hold(env, (NlCarrierRef){result, record_list});
}
NlCarrierStringStats env_carrier_string_stats(const Environment *env) {
    return env_carrier_scopes_enabled(env) ? env->carrier_session->string_stats : (NlCarrierStringStats){0};
}
NlCarrierStringStats env_carrier_last_string_stats(void) { return last_strings; }
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
    if (s->collecting || env->symbol_count < 0) return false;
    size_t count = s->root_count;
    if (s->host_count > SIZE_MAX - count) return false;
    count += s->host_count;
    if ((size_t)env->symbol_count > SIZE_MAX - count) return false;
    count += (size_t)env->symbol_count;
    if (count > SIZE_MAX / sizeof(NlCarrierRef)) return false;
    /* I reuse metadata only; no copied payload root survives this collection. */
    if (count > s->collection_capacity) {
        size_t capacity = s->collection_capacity ? s->collection_capacity : 16;
        while (capacity < count) {
            if (capacity > SIZE_MAX / 2) return false;
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / sizeof(NlCarrierRef)) return false;
        NlCarrierRef *grown = realloc(s->collection_roots, capacity * sizeof *grown);
        if (!grown) return false;
        s->collection_roots = grown;
        s->collection_capacity = capacity;
    }
    NlCarrierRef *roots = s->collection_roots;
    size_t n = 0;
    for (size_t i = 0; i < s->root_count; ++i) roots[n++] = s->roots[i];
    for (size_t i = 0; i < s->host_count; ++i) roots[n++] = s->host_roots[i];
    for (int i = 0; i < env->symbol_count; ++i) {
        Symbol *symbol = &env->symbols[i];
        /* Checker-retained declarations have no runtime payload yet. */
        if (symbol->value.type == VAL_VOID) continue;
        roots[n++] = (NlCarrierRef){symbol->value, symbol->type == TYPE_LIST_GENERIC};
    }
    s->collecting = true;
    bool ok = nl_carrier_values_collect(s->values, roots, n);
    if (n) memset(roots, 0, n * sizeof *roots);
    s->collecting = false;
    return ok;
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
void env_carrier_track_legacy_list(Environment *env, List_int *list) {
    if (!env_carrier_enabled(env)) return;
    NlCarrierRef value = nl_carrier_legacy_record_list(list);
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
    if (!env_carrier_scopes_enabled(env)) return;
    struct NlEvalCarrierSession *session = env->carrier_session;
    if (session->depth || session->string_count || session->string_stats.live ||
        (session->values && !nl_carrier_values_clear(session->values))) ownership_failure();
    last_strings = session->string_stats;
    last_retired = nl_carrier_values_stats(session->values);
    if (!nl_carrier_values_free(session->values)) ownership_failure();
    free(session->roots); free(session->frames); free(session->host_roots); free(session->strings);
    free(session->collection_roots);
    free(session); env->carrier_session = NULL; retired_sessions++;
}
NlCarrierHeapStats env_carrier_last_retired_stats(void) { return last_retired; }
size_t env_carrier_retired_sessions(void) { return retired_sessions; }
