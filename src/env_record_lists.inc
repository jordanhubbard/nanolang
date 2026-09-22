/* I own record/tuple/string graphs independently of symbols. Other reference
 * leaves keep their existing borrowed representation. Results accumulate until
 * Environment teardown; this is not expression-bounded storage. */
#include "runtime/list_int.h"
#include <limits.h>
#include <stdint.h>

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

/* My slots borrow arena entries. They never own or shorten a graph's lifetime. */
struct EnvRecordIndex {
    size_t capacity;
    size_t count;
    struct EnvRecordResult *slots[];
};

static const void *record_result_pointer(Value value) {
    if (value.type == VAL_STRING) return value.as.string_val;
    if (value.type == VAL_STRUCT) return value.as.struct_val;
    if (value.type == VAL_TUPLE) return value.as.tuple_val;
    if (value.type == VAL_FUNCTION) return value.as.function_val.function_name;
    return NULL;
}

static size_t record_result_hash(Value value) {
    uint64_t hash = (uint64_t)(uintptr_t)record_result_pointer(value);
    hash ^= (uint64_t)value.type * UINT64_C(0x9e3779b97f4a7c15);
    hash ^= hash >> 30;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash *= UINT64_C(0x94d049bb133111eb);
    hash ^= hash >> 31;
    return (size_t)hash;
}

static bool record_index_bytes(size_t capacity, size_t *out) {
    if (!out || capacity < 16 || (capacity & (capacity - 1)) ||
        capacity > (SIZE_MAX - sizeof(struct EnvRecordIndex)) / sizeof(struct EnvRecordResult *)) return false;
    *out = sizeof(struct EnvRecordIndex) + capacity * sizeof(struct EnvRecordResult *);
    return true;
}

/* I locate before writing. Hash equality alone grants no membership. */
static bool record_index_slot(const struct EnvRecordIndex *index, Value value,
                              size_t *slot, bool *found) {
    size_t bytes;
    const void *pointer = record_result_pointer(value);
    if (!index || !pointer || !slot || !found || !record_index_bytes(index->capacity, &bytes) ||
        index->count > index->capacity - index->capacity / 4) return false;
    size_t position = record_result_hash(value) & (index->capacity - 1);
    for (size_t probe = 0; probe < index->capacity; ++probe) {
        const struct EnvRecordResult *entry = index->slots[position];
        if (!entry || (entry->value.type == value.type && record_result_pointer(entry->value) == pointer)) {
            *slot = position;
            *found = entry != NULL;
            return true;
        }
        position = (position + 1) & (index->capacity - 1);
    }
    return false;
}

/* True transfers the complete entry; false leaves caller ownership and every
 * old index/list byte unchanged. No user callback occurs in this transaction. */
static bool record_result_publish(Environment *env, struct EnvRecordResult *entry) {
    if (!env || !entry || !record_result_pointer(entry->value)) return false;
    struct EnvRecordIndex *old = env->record_result_index;
    size_t count = old ? old->count : 0, capacity = old ? old->capacity : 16;
    size_t slot = 0, bytes;
    bool found = false;
    if (count == SIZE_MAX || (!old && env->record_results) ||
        !record_index_bytes(capacity, &bytes)) return false;
    if (old && (!record_index_slot(old, entry->value, &slot, &found) || found)) return false;
    bool grow = old && count + 1 > capacity - capacity / 4;
    if (grow) {
        if (capacity > SIZE_MAX / 2) return false;
        capacity *= 2;
        if (!record_index_bytes(capacity, &bytes)) return false;
    }
    struct EnvRecordIndex *table = old;
    if (!old || grow) {
        table = calloc(1, bytes);
        if (!table) return false;
        table->capacity = capacity;
        for (size_t i = 0; old && i < old->capacity; ++i) {
            struct EnvRecordResult *retained = old->slots[i];
            if (!retained) continue;
            if (!record_index_slot(table, retained->value, &slot, &found) || found) goto failed;
            table->slots[slot] = retained;
            ++table->count;
        }
        if (table->count != count ||
            !record_index_slot(table, entry->value, &slot, &found) || found) goto failed;
    }
    /* All fallible work is complete; the old graphs stay at their addresses. */
    entry->next = env->record_results;
    table->slots[slot] = entry;
    table->count = count + 1;
    env->record_results = entry;
    env->record_result_index = table;
    if (table != old) free(old);
    return true;
failed:
    free(table); /* This path only owns an unpublished replacement table. */
    return false;
}

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
            if (!record->field_name_storage) free(record->field_names[i]);
        }
        free(record->field_name_storage);
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
        size_t name_bytes = 0;
        for (int i = 0; i < src->field_count; ++i) {
            if (!src->field_names[i]) return false;
            size_t length = strlen(src->field_names[i]);
            if (length == SIZE_MAX || length + 1 > SIZE_MAX - name_bytes) return false;
            name_bytes += length + 1;
        }
        StructValue *dst = calloc(1, sizeof(*dst));
        if (!dst) return false;
        copy.as.struct_val = dst;
        dst->struct_name = strdup(src->struct_name);
        size_t count = src->field_count ? (size_t)src->field_count : 1;
        dst->field_names = calloc(count, sizeof(*dst->field_names));
        dst->field_values = calloc(count, sizeof(*dst->field_values));
        if (!dst->struct_name || !dst->field_names || !dst->field_values) goto fail;
        if (name_bytes) {
            dst->field_name_storage = malloc(name_bytes);
            if (!dst->field_name_storage) goto fail;
        }
        char *next_name = dst->field_name_storage;
        for (int i = 0; i < src->field_count; ++i) {
            size_t bytes = strlen(src->field_names[i]) + 1;
            dst->field_names[i] = next_name;
            memcpy(next_name, src->field_names[i], bytes);
            next_name += bytes;
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
    return out && clone_value_at(source, out, 0);
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
    if (!record_result_publish(env, entry)) {
        env_discard_value_snapshot(copy);
        free(entry);
        return false;
    }
    *out = copy;
    return true;
}

bool env_record_snapshot(Environment *env, Value source, Value *out) {
    return source.type == VAL_STRUCT && env_value_snapshot(env, source, out);
}

bool env_record_result_borrowed(Environment *env, Value value) {
    size_t slot;
    bool found;
    return env && record_index_slot(env->record_result_index, value, &slot, &found) && found;
}

/* I transfer ownership only after registration succeeds. My caller clears its
 * symbol after true; false preserves its ownership. */
bool env_retire_value(Environment *env, Value owned) {
    if (!env || (owned.type != VAL_STRUCT && owned.type != VAL_TUPLE &&
                 owned.type != VAL_FUNCTION) || !record_result_pointer(owned) ||
        env_record_result_borrowed(env, owned)) return false;
    struct EnvRecordResult *entry = malloc(sizeof(*entry));
    if (!entry) return false;
    entry->value = owned;
    if (!record_result_publish(env, entry)) { free(entry); return false; }
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
    struct EnvRecordIndex *index = env->record_result_index;
    env->record_result_index = NULL;
    free(index);
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
        /* Callable leaves in value graphs borrow these independently owned roots. */
        if (entry->value.type == VAL_FUNCTION) {
            free((char *)entry->value.as.function_val.function_name);
            free_function_signature(entry->value.as.function_val.signature);
        } else env_discard_value_snapshot(entry->value);
        free(entry);
    }
}
