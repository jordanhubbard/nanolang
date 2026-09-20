/* I own these records independently of symbols. Non-record reference fields
 * retain create_struct's borrowed representation. Results accumulate until
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
    StructValue *record;
};

void env_discard_record(StructValue *record) {
    if (!record) return;
    for (int i = 0; i < record->field_count; ++i) {
        Value field = record->field_values[i];
        if (field.type == VAL_STRUCT) env_discard_record(field.as.struct_val);
        else if (field.type == VAL_STRING && field.as.string_val) {
            if (gc_is_managed(field.as.string_val)) gc_release(field.as.string_val);
            else free(field.as.string_val);
        }
        free(record->field_names[i]);
    }
    free(record->field_names);
    free(record->field_values);
    free(record->struct_name);
    free(record);
}

static bool clone_record_at(const StructValue *src, StructValue **out, unsigned depth) {
    if (!src || !src->struct_name || depth >= 128 || src->field_count < 0 ||
        (src->field_count && (!src->field_names || !src->field_values)) ||
        (size_t)src->field_count > SIZE_MAX / sizeof(Value) ||
        (size_t)src->field_count > SIZE_MAX / sizeof(char *)) return false;
    StructValue *dst = calloc(1, sizeof(*dst));
    if (!dst) return false;
    dst->struct_name = strdup(src->struct_name);
    size_t count = src->field_count ? (size_t)src->field_count : 1;
    dst->field_names = calloc(count, sizeof(*dst->field_names));
    dst->field_values = calloc(count, sizeof(*dst->field_values));
    if (!dst->struct_name || !dst->field_names || !dst->field_values) goto fail;
    for (int i = 0; i < src->field_count; ++i) {
        if (!src->field_names[i]) goto fail;
        dst->field_names[i] = strdup(src->field_names[i]);
        if (!dst->field_names[i]) goto fail;
        dst->field_values[i] = create_void();
        dst->field_count = i + 1;
        Value field = src->field_values[i];
        if (field.type == VAL_STRING) {
            char *text = strdup(field.as.string_val ? field.as.string_val : "");
            if (!text) goto fail;
            field.as.string_val = text;
        } else if (field.type == VAL_STRUCT) {
            StructValue *nested = NULL;
            if (!clone_record_at(field.as.struct_val, &nested, depth + 1)) goto fail;
            field.as.struct_val = nested;
        }
        dst->field_values[i] = field;
    }
    *out = dst;
    return true;
fail:
    env_discard_record(dst);
    return false;
}

bool env_clone_record(Value source, Value *out) {
    if (!out || source.type != VAL_STRUCT) return false;
    StructValue *record = NULL;
    if (!clone_record_at(source.as.struct_val, &record, 0)) return false;
    Value result = {0};
    result.type = VAL_STRUCT;
    result.as.struct_val = record;
    *out = result;
    return true;
}

bool env_record_snapshot(Environment *env, Value source, Value *out) {
    if (!env || !out) return false;
    struct EnvRecordResult *entry = malloc(sizeof(*entry));
    if (!entry) return false;
    Value copy;
    if (!env_clone_record(source, &copy)) { free(entry); return false; }
    entry->record = copy.as.struct_val;
    entry->next = env->record_results;
    env->record_results = entry;
    *out = copy;
    return true;
}

bool env_record_result_borrowed(Environment *env, Value value) {
    if (!env || value.type != VAL_STRUCT) return false;
    for (struct EnvRecordResult *entry = env->record_results; entry; entry = entry->next)
        if (entry->record == value.as.struct_val) return true;
    return false;
}

/* I transfer ownership only after registration succeeds. My caller clears its
 * symbol after true; false preserves its ownership. */
bool env_retire_record(Environment *env, Value owned) {
    if (!env || owned.type != VAL_STRUCT || !owned.as.struct_val ||
        env_record_result_borrowed(env, owned)) return false;
    struct EnvRecordResult *entry = malloc(sizeof(*entry));
    if (!entry) return false;
    entry->record = owned.as.struct_val;
    entry->next = env->record_results;
    env->record_results = entry;
    return true;
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
        env_discard_record(entry->record);
        free(entry);
    }
}
