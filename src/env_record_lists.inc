/* I own record/tuple/string graphs independently of symbols. Other reference
 * leaves keep their existing borrowed representation. Results accumulate until
 * Environment teardown; this is not expression-bounded storage. */
#include "runtime/list_int.h"
#include <limits.h>
#include <stdint.h>

#ifdef NANO_EVALUATOR_LIFETIME_TIMING
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
static uint64_t lifetime_allocations, lifetime_clone_nodes, lifetime_clones_ns;
static uint64_t lifetime_retirements, lifetime_roots, lifetime_lookups, lifetime_visits, lifetime_lookup_ns;
static unsigned lifetime_markers;
static void lifetime_add(uint64_t *value, uint64_t amount) {
    if (*value > UINT64_MAX - amount) { fputs("I overflowed a diagnostic counter.\n", stderr); exit(92); }
    *value += amount;
}
static uint64_t lifetime_now(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) || now.tv_sec < 0 || now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
        (uint64_t)now.tv_sec > (UINT64_MAX - (uint64_t)now.tv_nsec) / UINT64_C(1000000000)) {
        fputs("I cannot read the diagnostic clock.\n", stderr); exit(92);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}
static void lifetime_elapsed(uint64_t *total, uint64_t start) {
    uint64_t end = lifetime_now();
    if (end < start) { fputs("I observed a diagnostic clock reversal.\n", stderr); exit(92); }
    lifetime_add(total, end - start);
}
void nano_evaluator_lifetime_marker(const char *phase, const char *name) {
    if (lifetime_markers++ >= 8192) { fputs("I exceeded my diagnostic marker bound.\n", stderr); exit(92); }
    int written = fprintf(stderr, "LIFETIME %s %.160s ns=%" PRIu64 " allocations=%" PRIu64
        " clone_nodes=%" PRIu64 " clone_ns=%" PRIu64 " retirements=%" PRIu64
        " roots=%" PRIu64 " lookups=%" PRIu64 " visits=%" PRIu64 " lookup_ns=%" PRIu64 "\n",
        phase, name, lifetime_now(), lifetime_allocations, lifetime_clone_nodes, lifetime_clones_ns,
        lifetime_retirements, lifetime_roots, lifetime_lookups, lifetime_visits, lifetime_lookup_ns);
    if (written < 0 || fflush(stderr)) exit(92);
}
#if defined(malloc) || defined(calloc) || defined(realloc) || defined(strdup)
#error I require unwrapped allocators for this separate timing build.
#endif
static void *lifetime_malloc(size_t n) { lifetime_add(&lifetime_allocations, 1); return malloc(n); }
static void *lifetime_calloc(size_t n, size_t width) { lifetime_add(&lifetime_allocations, 1); return calloc(n, width); }
static void *lifetime_realloc(void *p, size_t n) { lifetime_add(&lifetime_allocations, 1); return realloc(p, n); }
static char *lifetime_strdup(const char *p) { lifetime_add(&lifetime_allocations, 1); return strdup(p); }
#define malloc lifetime_malloc
#define calloc lifetime_calloc
#define realloc lifetime_realloc
#define strdup lifetime_strdup
#define LIFETIME_COUNT(counter) lifetime_add(&counter, 1)
#else
#define LIFETIME_COUNT(counter) ((void)0)
#endif

struct EnvRecordList {
    struct EnvRecordList *next;
    NominalIdentity element;
    bool live;
    List_int storage;
};
struct EnvRecordResult {
    struct EnvRecordResult *next;
    Value value;
};

/* I own only the record/tuple/string value graph. Other reference leaves keep
 * their original ownership contract and are never released by this pair. */
void env_discard_value_snapshot(Value owned) {
    if (owned.type == VAL_STRING) {
        if (gc_is_managed(owned.as.string_val)) gc_release(owned.as.string_val);
        else free(owned.as.string_val);
    } else if (owned.type == VAL_STRUCT) {
        StructValue *record = owned.as.struct_val;
        if (!record) return;
        for (int i = 0; i < record->field_count; ++i) {
            env_discard_value_snapshot(record->field_values[i]);
            free(record->field_names[i]);
        }
        free(record->field_names);
        free(record->field_values);
        free(record->struct_name);
        free(record);
    } else if (owned.type == VAL_TUPLE) {
        TupleValue *tuple = owned.as.tuple_val;
        if (!tuple) return;
        for (int i = 0; i < tuple->element_count; ++i)
            env_discard_value_snapshot(tuple->elements[i]);
        free(tuple->elements);
        free(tuple);
    }
}

void env_discard_record(StructValue *record) {
    Value owned = {0};
    owned.type = VAL_STRUCT;
    owned.as.struct_val = record;
    env_discard_value_snapshot(owned);
}

static bool clone_value_at(Value source, Value *out, unsigned depth) {
    LIFETIME_COUNT(lifetime_clone_nodes);
    if (depth >= 128) return false;
    Value copy = source;
    copy.is_return = copy.is_break = copy.is_continue = false;
    copy.return_target = NULL;
    if (source.type == VAL_STRING) {
        copy.as.string_val = strdup(source.as.string_val ? source.as.string_val : "");
        if (!copy.as.string_val) return false;
    } else if (source.type == VAL_STRUCT) {
        StructValue *src = source.as.struct_val;
        if (!src || !src->struct_name || src->field_count < 0 ||
            (src->field_count && (!src->field_names || !src->field_values)) ||
            (size_t)src->field_count > SIZE_MAX / sizeof(Value) ||
            (size_t)src->field_count > SIZE_MAX / sizeof(char *)) return false;
        StructValue *dst = calloc(1, sizeof(*dst));
        if (!dst) return false;
        copy.as.struct_val = dst;
        dst->struct_name = strdup(src->struct_name);
        size_t count = src->field_count ? (size_t)src->field_count : 1;
        dst->field_names = calloc(count, sizeof(*dst->field_names));
        dst->field_values = calloc(count, sizeof(*dst->field_values));
        if (!dst->struct_name || !dst->field_names || !dst->field_values) goto fail;
        for (int i = 0; i < src->field_count; ++i) {
            if (!src->field_names[i]) goto fail;
            dst->field_names[i] = strdup(src->field_names[i]);
            if (!dst->field_names[i]) goto fail;
            dst->field_values[i].type = VAL_VOID;
            dst->field_count = i + 1;
            if (!clone_value_at(src->field_values[i], &dst->field_values[i], depth + 1)) goto fail;
        }
    } else if (source.type == VAL_TUPLE) {
        TupleValue *src = source.as.tuple_val;
        if (!src || src->element_count < 0 || (src->element_count && !src->elements) ||
            (size_t)src->element_count > SIZE_MAX / sizeof(Value)) return false;
        TupleValue *dst = calloc(1, sizeof(*dst));
        if (!dst) return false;
        copy.as.tuple_val = dst;
        if (src->element_count) {
            dst->elements = calloc((size_t)src->element_count, sizeof(Value));
            if (!dst->elements) goto fail;
        }
        for (int i = 0; i < src->element_count; ++i) {
            if (!clone_value_at(src->elements[i], &dst->elements[i], depth + 1)) goto fail;
            ++dst->element_count;
        }
    }
    *out = copy;
    return true;
fail:
    env_discard_value_snapshot(copy);
    return false;
}

bool env_clone_value_snapshot(Value source, Value *out) {
#ifdef NANO_EVALUATOR_LIFETIME_TIMING
    uint64_t start = lifetime_now();
    bool result = out && clone_value_at(source, out, 0);
    lifetime_elapsed(&lifetime_clones_ns, start);
    return result;
#else
    return out && clone_value_at(source, out, 0);
#endif
}

bool env_clone_record(Value source, Value *out) {
    return source.type == VAL_STRUCT && env_clone_value_snapshot(source, out);
}

bool env_value_snapshot(Environment *env, Value source, Value *out) {
    if (!env || !out || (source.type != VAL_STRUCT && source.type != VAL_TUPLE && source.type != VAL_STRING)) return false;
    struct EnvRecordResult *entry = malloc(sizeof(*entry));
    if (!entry) return false;
    Value copy;
    if (!env_clone_value_snapshot(source, &copy)) { free(entry); return false; }
    entry->value = copy;
    entry->next = env->record_results;
    env->record_results = entry;
    LIFETIME_COUNT(lifetime_roots);
    *out = copy;
    return true;
}

bool env_record_snapshot(Environment *env, Value source, Value *out) {
    return source.type == VAL_STRUCT && env_value_snapshot(env, source, out);
}

#ifdef NANO_EVALUATOR_LIFETIME_TIMING
static bool lifetime_borrowed_body(Environment *env, Value value) {
#else
bool env_record_result_borrowed(Environment *env, Value value) {
#endif
    LIFETIME_COUNT(lifetime_lookups);
    if (!env || (value.type != VAL_STRUCT && value.type != VAL_TUPLE && value.type != VAL_STRING)) return false;
    for (struct EnvRecordResult *entry = env->record_results; entry; entry = entry->next) {
        LIFETIME_COUNT(lifetime_visits);
        if (entry->value.type != value.type) continue;
        if (value.type == VAL_STRING && entry->value.as.string_val == value.as.string_val) return true;
        if (value.type == VAL_STRUCT && entry->value.as.struct_val == value.as.struct_val) return true;
        if (value.type == VAL_TUPLE && entry->value.as.tuple_val == value.as.tuple_val) return true;
    }
    return false;
}

#ifdef NANO_EVALUATOR_LIFETIME_TIMING
bool env_record_result_borrowed(Environment *env, Value value) {
    uint64_t start = lifetime_now();
    bool result = lifetime_borrowed_body(env, value);
    lifetime_elapsed(&lifetime_lookup_ns, start);
    return result;
}
#endif

/* I transfer ownership only after registration succeeds. My caller clears its
 * symbol after true; false preserves its ownership. */
bool env_retire_value(Environment *env, Value owned) {
    LIFETIME_COUNT(lifetime_retirements);
    if (!env || (owned.type != VAL_STRUCT && owned.type != VAL_TUPLE) ||
        (owned.type == VAL_STRUCT ? !owned.as.struct_val : !owned.as.tuple_val) ||
        env_record_result_borrowed(env, owned)) return false;
    struct EnvRecordResult *entry = malloc(sizeof(*entry));
    if (!entry) return false;
    entry->value = owned;
    entry->next = env->record_results;
    env->record_results = entry;
    LIFETIME_COUNT(lifetime_roots);
    return true;
}

bool env_retire_record(Environment *env, Value owned) {
    return owned.type == VAL_STRUCT && env_retire_value(env, owned);
}

static struct EnvRecordList *record_list_find(Environment *env, Value handle) {
    if (!env || handle.type != VAL_INT) return NULL;
    for (struct EnvRecordList *entry = env->record_lists; entry; entry = entry->next)
        if (handle.as.int_val == (long long)(intptr_t)&entry->storage) return entry;
    return NULL;
}

bool env_record_list_identity(Environment *env, Value handle, NominalIdentity *out) {
    struct EnvRecordList *entry = record_list_find(env, handle);
    if (!entry || !entry->live || !out) return false;
    *out = entry->element;
    return true;
}

static void record_list_clear(struct EnvRecordList *entry) {
    for (int i = 0; i < entry->storage.length; ++i) {
        env_discard_record((StructValue *)(intptr_t)entry->storage.data[i]);
        entry->storage.data[i] = 0;
    }
    entry->storage.length = 0;
}

static bool record_list_grow(List_int *list) {
    if (list->length < list->capacity) return true;
    if (list->capacity == INT_MAX) return false;
    int capacity = list->capacity > INT_MAX / 2 ? INT_MAX : list->capacity * 2;
    if (capacity < 8) capacity = 8;
    if ((size_t)capacity > SIZE_MAX / sizeof(*list->data)) return false;
    int64_t *data = realloc(list->data, (size_t)capacity * sizeof(*data));
    if (!data) return false;
    list->data = data;
    list->capacity = capacity;
    return true;
}

/* I leave output and list state unchanged on failure. A result snapshot is
 * fully published in my arena before a non-failing slot removal. */
bool env_record_list_apply(Environment *env, NominalIdentity element,
                           const char *operation, const Value *args, int argc, Value *out) {
    const char *name = env_nominal_name(env, element);
    if (!env || !out || !operation || !name || element.kind != TYPE_STRUCT ||
        argc < 0 || (argc && !args) || env->structs[element.ordinal - 1].is_resource) return false;
    bool capacity_new = !strcmp(operation, "with_capacity");
    if (!strcmp(operation, "new") || capacity_new) {
        if (argc != (capacity_new ? 1 : 0)) return false;
        long long capacity = capacity_new ?
            (args[0].type == VAL_INT ? args[0].as.int_val : -1) : 8;
        if (capacity < 0 || capacity > INT_MAX ||
            (unsigned long long)capacity > SIZE_MAX / sizeof(int64_t)) return false;
        struct EnvRecordList *entry = calloc(1, sizeof(*entry));
        if (!entry) return false;
        if (capacity) {
            entry->storage.data = malloc((size_t)capacity * sizeof(int64_t));
            if (!entry->storage.data) { free(entry); return false; }
        }
        entry->storage.capacity = (int)capacity;
        entry->element = element;
        entry->live = true;
        entry->next = env->record_lists;
        env->record_lists = entry;
        *out = create_int((long long)(intptr_t)&entry->storage);
        return true;
    }
    if (argc < 1) return false;
    struct EnvRecordList *entry = record_list_find(env, args[0]);
    if (!entry || !entry->live || entry->element.kind != element.kind ||
        entry->element.ordinal != element.ordinal) return false;
    List_int *list = &entry->storage;
    bool push = !strcmp(operation, "push"), insert = !strcmp(operation, "insert");
    bool set = !strcmp(operation, "set"), get = !strcmp(operation, "get");
    bool remove = !strcmp(operation, "remove"), pop = !strcmp(operation, "pop");
    bool length = !strcmp(operation, "length"), capacity = !strcmp(operation, "capacity");
    bool empty = !strcmp(operation, "is_empty"), clear = !strcmp(operation, "clear");
    bool destroy = !strcmp(operation, "free");
    if (!(push || insert || set || get || remove || pop || length || capacity ||
          empty || clear || destroy)) return false;
    int needed = insert || set ? 3 : push || get || remove ? 2 : 1;
    if (argc != needed) return false;
    long long index = push ? list->length : pop ? (long long)list->length - 1 : 0;
    if (insert || set || get || remove) {
        if (args[1].type != VAL_INT) return false;
        index = args[1].as.int_val;
    }
    if ((push || insert || set || get || remove || pop) &&
        (index < 0 || index > list->length ||
         (!(push || insert) && index == list->length))) return false;
    Value result = create_void();
    result.return_target = NULL;
    if (push || insert || set) {
        Value source = args[push ? 1 : 2];
        if (source.type != VAL_STRUCT || !source.as.struct_val ||
            !source.as.struct_val->struct_name || strcmp(source.as.struct_val->struct_name, name)) return false;
        Value copy;
        if (!env_clone_record(source, &copy)) return false;
        if (!set && !record_list_grow(list)) { env_discard_record(copy.as.struct_val); return false; }
        if (set) env_discard_record((StructValue *)(intptr_t)list->data[index]);
        else {
            memmove(list->data + index + 1, list->data + index,
                    (size_t)(list->length - index) * sizeof(*list->data));
            ++list->length;
        }
        list->data[index] = (int64_t)(intptr_t)copy.as.struct_val;
    } else if (get || pop || remove) {
        Value source = create_void();
        source.type = VAL_STRUCT;
        source.as.struct_val = (StructValue *)(intptr_t)list->data[index];
        if (!env_record_snapshot(env, source, &result)) return false;
        if (!get) {
            env_discard_record(source.as.struct_val);
            memmove(list->data + index, list->data + index + 1,
                    (size_t)(list->length - index - 1) * sizeof(*list->data));
            list->data[--list->length] = 0;
        }
    } else if (length) result = create_int(list->length);
    else if (capacity) result = create_int(list->capacity);
    else if (empty) result = create_bool(list->length == 0);
    else if (clear || destroy) {
        record_list_clear(entry);
        if (destroy) {
            free(list->data);
            list->data = NULL;
            list->capacity = 0;
            entry->live = false;
        }
    }
    *out = result;
    return true;
}

static void env_record_storage_free(Environment *env) {
    while (env->record_lists) {
        struct EnvRecordList *entry = env->record_lists;
        env->record_lists = entry->next;
        record_list_clear(entry);
        free(entry->storage.data);
        free(entry);
    }
    while (env->record_results) {
        struct EnvRecordResult *entry = env->record_results;
        env->record_results = entry->next;
        env_discard_value_snapshot(entry->value);
        free(entry);
    }
}

#undef LIFETIME_COUNT
#ifdef NANO_EVALUATOR_LIFETIME_TIMING
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#endif
