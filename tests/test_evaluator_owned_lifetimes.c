/* I test checked graph ownership separately from fatal legacy evaluator allocation. */
#define _POSIX_C_SOURCE 200809L
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#define _DARWIN_C_SOURCE
#include "../src/nanolang.h"
#include "../src/runtime/gc.h"
#include "../src/coroutine.h"
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks, attempts, failures, failure_at = SIZE_MAX, live;
static bool observing, transient;
static void *tracked[16384];
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "I failed %s at %d\n", #x, __LINE__); exit(90); } } while (0)
static bool fail_allocation(void) {
    if (!observing) return false;
    size_t current = attempts++;
    bool fail = transient ? current == failure_at : current >= failure_at;
    if (fail) ++failures;
    return fail;
}
static void remember(void *pointer) {
    if (!observing || !pointer) return;
    for (size_t i = 0; i < sizeof(tracked) / sizeof(*tracked); ++i)
        if (!tracked[i]) { tracked[i] = pointer; ++live; return; }
    CHECK(false);
}
static void forget(void *pointer) {
    if (!pointer) return;
    for (size_t i = 0; i < sizeof(tracked) / sizeof(*tracked); ++i)
        if (tracked[i] == pointer) { tracked[i] = NULL; CHECK(live); --live; return; }
}
static void *owned_malloc(size_t bytes) {
    if (fail_allocation()) return NULL;
    void *pointer = malloc(bytes); remember(pointer); return pointer;
}
static void *owned_calloc(size_t count, size_t bytes) {
    if (fail_allocation()) return NULL;
    void *pointer = calloc(count, bytes); remember(pointer); return pointer;
}
static void owned_free(void *pointer) { forget(pointer); free(pointer); }
static void *owned_realloc(void *pointer, size_t bytes) {
    if (fail_allocation()) return NULL;
    /* I retain the old allocation record if realloc fails. */
    size_t slot = SIZE_MAX;
    for (size_t i = 0; i < sizeof(tracked) / sizeof(*tracked); ++i)
        if (pointer && tracked[i] == pointer) { slot = i; break; }
    void *next = realloc(pointer, bytes);
    if (next) {
        if (slot != SIZE_MAX) tracked[slot] = next;
        else remember(next);
    }
    return next;
}
static char *owned_strdup(const char *text) {
    size_t bytes = strlen(text) + 1;
    char *copy = owned_malloc(bytes);
    if (copy) memcpy(copy, text, bytes);
    return copy;
}
void *lifetime_alloc_malloc(size_t bytes) { return owned_malloc(bytes); }
void *lifetime_alloc_calloc(size_t count, size_t bytes) { return owned_calloc(count, bytes); }
void *lifetime_alloc_realloc(void *p, size_t bytes) { return owned_realloc(p, bytes); }
char *lifetime_alloc_strdup(const char *text) { return owned_strdup(text); }
void lifetime_alloc_free(void *p) { owned_free(p); }
extern bool lifetime_task_clone(Value, Value *);
extern void lifetime_task_drop(Value);
extern bool lifetime_prepare_bundle(Environment *, Value, int *);
extern Value lifetime_stage_argument(Environment *, ASTNode *, const char *);
extern void lifetime_scope_release(Environment *, int, bool);
#define malloc owned_malloc
#define calloc owned_calloc
#define realloc owned_realloc
#define strdup owned_strdup
#define free owned_free
#include "../src/env.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static void begin(size_t index, bool once) {
    CHECK(!observing && live == 0);
    attempts = failures = 0; failure_at = index; transient = once; observing = true;
}
static void end(void) { observing = false; }
static Value integer(int n) { Value v = {0}; v.type = VAL_INT; v.as.int_val = n; return v; }
static Value text_value(char *s) { Value v = {0}; v.type = VAL_STRING; v.as.string_val = s; return v; }
static Value tuple_value(TupleValue *t) { Value v = {0}; v.type = VAL_TUPLE; v.as.tuple_val = t; return v; }
static Value record_value(StructValue *r) { Value v = {0}; v.type = VAL_STRUCT; v.as.struct_val = r; return v; }

static void graph_attempt(size_t at, bool once, size_t *count) {
    char payload[] = "kept";
    Value fields[] = {text_value(payload), integer(37)};
    fields[0].is_return = fields[0].is_break = fields[0].is_continue = true;
    fields[0].return_target = &fields[0];
    fields[1].is_return = fields[1].is_break = fields[1].is_continue = true;
    fields[1].return_target = &fields[1];
    char *names[] = {"text", "number"};
    StructValue record = {.struct_name = "Item", .field_names = names, .field_values = fields, .field_count = 2};
    Value nested[] = {record_value(&record), text_value(payload), integer(73)};
    nested[1].is_return = nested[1].is_break = nested[1].is_continue = true;
    nested[1].return_target = &nested[1];
    nested[2].is_return = nested[2].is_break = nested[2].is_continue = true;
    nested[2].return_target = &nested[2];
    TupleValue tuple = {.elements = nested, .element_count = 3};
    Value source = tuple_value(&tuple), output = integer(771), sentinel = output;
    begin(at, once);
    bool ok = env_clone_value_snapshot(source, &output);
    *count = attempts; end();
    if (at != SIZE_MAX) {
        CHECK(!ok && failures && !memcmp(&output, &sentinel, sizeof output));
        CHECK(!strcmp(payload, "kept") && record.field_values == fields);
    } else {
        CHECK(ok && output.type == VAL_TUPLE && output.as.tuple_val != &tuple);
        CHECK(output.as.tuple_val->elements[0].as.struct_val != &record);
        payload[0] = 'X'; fields[1] = integer(99);
        StructValue *copied = output.as.tuple_val->elements[0].as.struct_val;
        CHECK(!strcmp(copied->field_values[0].as.string_val, "kept"));
        CHECK(copied->field_values[0].as.string_val != fields[0].as.string_val);
        CHECK(!copied->field_values[0].is_return && !copied->field_values[0].is_break &&
              !copied->field_values[0].is_continue && !copied->field_values[0].return_target);
        CHECK(copied->field_values[1].as.int_val == 37);
        CHECK(!copied->field_values[1].is_return && !copied->field_values[1].is_break &&
              !copied->field_values[1].is_continue && !copied->field_values[1].return_target);
        CHECK(!strcmp(output.as.tuple_val->elements[1].as.string_val, "kept"));
        CHECK(output.as.tuple_val->elements[1].as.string_val != nested[1].as.string_val);
        CHECK(!output.as.tuple_val->elements[1].is_return && !output.as.tuple_val->elements[1].is_break &&
              !output.as.tuple_val->elements[1].is_continue && !output.as.tuple_val->elements[1].return_target);
        CHECK(output.as.tuple_val->elements[2].as.int_val == 73);
        CHECK(!output.as.tuple_val->elements[2].is_return && !output.as.tuple_val->elements[2].is_break &&
              !output.as.tuple_val->elements[2].is_continue && !output.as.tuple_val->elements[2].return_target);
        env_discard_value_snapshot(output);
    }
    CHECK(live == 0);
}
static void graph_controls(void) {
    size_t count; graph_attempt(SIZE_MAX, false, &count); CHECK(count > 8);
    for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
        size_t ignored; graph_attempt(i, once != 0, &ignored); graph_attempt(SIZE_MAX, false, &ignored);
    }
    TupleValue chain[128]; Value values[128];
    for (int i = 127; i >= 0; --i) {
        values[i] = i == 127 ? integer(1) : tuple_value(&chain[i + 1]);
        chain[i] = (TupleValue){.elements = &values[i], .element_count = 1};
    }
    Value out = integer(19); CHECK(!env_clone_value_snapshot(tuple_value(&chain[0]), &out));
    CHECK(out.type == VAL_INT && out.as.int_val == 19);
    CHECK(env_clone_value_snapshot(tuple_value(&chain[1]), &out)); env_discard_value_snapshot(out);
    TupleValue bad = {.element_count = -1}; out = integer(19);
    CHECK(!env_clone_value_snapshot(tuple_value(&bad), &out) && out.as.int_val == 19);
}

static void record_names_attempt(int fields_count, size_t at, bool once, size_t *count) {
    char *name = malloc(4097), *type_name = strdup("Independent");
    char **names = calloc(fields_count ? (size_t)fields_count : 1, sizeof *names);
    Value *values = calloc(fields_count ? (size_t)fields_count : 1, sizeof *values);
    CHECK(name && type_name && names && values);
    memset(name, 'n', 4096); name[4096] = 0;
    for (int i = 0; i < fields_count; ++i) { names[i] = name; values[i] = integer(i); }
    StructValue record = {.struct_name=type_name, .field_names=names,
                         .field_values=values, .field_count=fields_count};
    Value result = integer(919), sentinel;
    memcpy(&sentinel, &result, sizeof result);
    begin(at, once);
    bool ok = env_clone_record(record_value(&record), &result);
    *count = attempts; end();
    if (at != SIZE_MAX) {
        CHECK(!ok && failures && !memcmp(&result, &sentinel, sizeof result));
        CHECK(name[0] == 'n' && !strcmp(type_name, "Independent"));
    } else {
        CHECK(ok && result.as.struct_val != &record);
        StructValue *copied = result.as.struct_val;
        CHECK(copied->field_count == fields_count);
        CHECK((copied->field_name_storage != NULL) == (fields_count != 0));
        name[0] = 'x'; type_name[0] = 'X';
        for (int i = 0; i < fields_count; ++i) values[i] = integer(-1);
        free(name); free(type_name); free(names); free(values);
        name = type_name = NULL; names = NULL; values = NULL;
        CHECK(!strcmp(copied->struct_name, "Independent"));
        for (int i = 0; i < fields_count; ++i) {
            CHECK(strlen(copied->field_names[i]) == 4096 && copied->field_names[i][0] == 'n');
            CHECK(copied->field_values[i].as.int_val == i);
            if (i) CHECK(copied->field_names[i] != copied->field_names[i-1]);
        }
        Value sibling = integer(0);
        CHECK(env_clone_record(result, &sibling));
        if (fields_count) {
            CHECK(sibling.as.struct_val->field_name_storage != copied->field_name_storage);
            copied->field_names[0][0] = 'c';
            CHECK(sibling.as.struct_val->field_names[0][0] == 'n');
            if (fields_count > 1) CHECK(copied->field_names[1][0] == 'n');
        }
        env_discard_value_snapshot(result);
        CHECK(!strcmp(sibling.as.struct_val->struct_name, "Independent"));
        env_discard_value_snapshot(sibling);
    }
    free(name); free(type_name); free(names); free(values);
    CHECK(live == 0);
}
static void record_names_controls(void) {
    const int sizes[] = {0, 1, 77};
    for (size_t s = 0; s < sizeof sizes / sizeof *sizes; ++s) {
        size_t count; record_names_attempt(sizes[s], SIZE_MAX, false, &count);
        for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
            size_t ignored; record_names_attempt(sizes[s], i, once != 0, &ignored);
            record_names_attempt(sizes[s], SIZE_MAX, false, &ignored);
        }
    }
    /* I retain disposal of independently allocated names in legacy owned views. */
    StructValue *legacy = calloc(1, sizeof *legacy); CHECK(legacy);
    legacy->struct_name = strdup("Legacy");
    legacy->field_names = calloc(1, sizeof *legacy->field_names);
    legacy->field_values = calloc(1, sizeof *legacy->field_values);
    CHECK(legacy->struct_name && legacy->field_names && legacy->field_values);
    legacy->field_names[0] = strdup(""); CHECK(legacy->field_names[0]);
    legacy->field_values[0] = integer(7); legacy->field_count = 1;
    Value result = integer(0); CHECK(env_clone_record(record_value(legacy), &result));
    env_discard_record(legacy);
    CHECK(result.as.struct_val->field_names[0][0] == 0);
    env_discard_value_snapshot(result);
}

static void signature_attempt(size_t at, bool once, size_t *count) {
    char nominal[] = "Item"; char *names[] = {nominal}; Type tags[] = {TYPE_STRUCT};
    TypeInfo leaf = {.base_type = TYPE_STRUCT, .generic_name = nominal};
    TypeInfo *parameters[] = {&leaf};
    FunctionSignature nested = {.param_count = 1, .param_types = tags, .param_struct_names = names,
                               .param_type_info = parameters, .return_type = TYPE_STRUCT,
                               .return_struct_name = nominal, .return_type_info = &leaf};
    TypeInfo full = {.base_type = TYPE_ARRAY, .element_type = &leaf, .generic_name = "List",
        .type_params = parameters, .type_param_count = 1, .tuple_types = tags,
        .tuple_type_names = names, .tuple_element_count = 1, .opaque_type_name = "Opaque",
        .fn_sig = &nested, .is_open_row = true, .row_var_name = "r", .row_field_names = names,
        .row_field_types = tags, .row_field_type_names = names, .row_field_count = 1,
        .type_var_names = names, .type_var_count = 1};
    TypeInfo *full_parameters[] = {&full};
    FunctionSignature source = nested; source.param_type_info = full_parameters;
    source.return_fn_sig = &nested;
    FunctionSignature *out = &source;
    begin(at, once); bool ok = copy_function_signature_checked(&source, &out);
    *count = attempts; end();
    if (at != SIZE_MAX) CHECK(!ok && failures && out == &source);
    else {
        CHECK(ok && out != &source && out->param_type_info[0] != &full);
        nominal[0] = 'X';
        TypeInfo *copy = out->param_type_info[0];
        CHECK(copy->is_open_row && !strcmp(copy->row_var_name, "r"));
        CHECK(!strcmp(copy->element_type->generic_name, "Item"));
        CHECK(!strcmp(copy->type_params[0]->generic_name, "Item"));
        CHECK(!strcmp(copy->tuple_type_names[0], "Item") && copy->tuple_types[0] == TYPE_STRUCT);
        CHECK(!strcmp(copy->row_field_names[0], "Item") && copy->row_field_types[0] == TYPE_STRUCT);
        CHECK(!strcmp(copy->row_field_type_names[0], "Item") && !strcmp(copy->type_var_names[0], "Item"));
        CHECK(!strcmp(copy->opaque_type_name, "Opaque") && !strcmp(copy->fn_sig->return_struct_name, "Item"));
        CHECK(!strcmp(out->return_fn_sig->param_struct_names[0], "Item"));
        free_function_signature(out);
    }
    CHECK(live == 0);
}
static void signature_controls(void) {
    size_t count; signature_attempt(SIZE_MAX, false, &count); CHECK(count > 30);
    for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
        size_t ignored; signature_attempt(i, once != 0, &ignored); signature_attempt(SIZE_MAX, false, &ignored);
    }
    FunctionSignature source = {.param_count = -1}, *out = &source;
    CHECK(!copy_function_signature_checked(&source, &out) && out == &source);
    CHECK(copy_function_signature_checked(NULL, &out) && out == NULL);
    FunctionSignature chain[129] = {{0}};
    for (int i = 0; i < 128; ++i) chain[i].return_fn_sig = &chain[i + 1];
    out = &source; CHECK(!copy_function_signature_checked(chain, &out) && out == &source);
    CHECK(copy_function_signature_checked(chain + 1, &out)); free_function_signature(out);
}

static Environment *list_environment(NominalIdentity *identity) {
    Environment *env = create_environment(); CHECK(env);
    StructDef definition = {0}; definition.name = strdup("Item"); CHECK(definition.name);
    env_define_struct(env, definition);
    *identity = env_nominal_identity(env, "Item", NULL, TYPE_STRUCT); CHECK(identity->ordinal);
    return env;
}
static void list_attempt(const char *operation, size_t at, bool once, size_t *count) {
    NominalIdentity id; Environment *env = list_environment(&id);
    char old[] = "old", next[] = "next"; char *names[] = {"text"};
    Value old_fields[] = {text_value(old)}, new_fields[] = {text_value(next)};
    StructValue old_record = {.struct_name = "Item", .field_count = 1, .field_names = names, .field_values = old_fields};
    StructValue new_record = old_record; new_record.field_values = new_fields;
    Value capacity = integer(1), handle, ignored;
    CHECK(env_record_list_apply(env, id, "with_capacity", &capacity, 1, &handle));
    Value push[] = {handle, record_value(&old_record)};
    CHECK(env_record_list_apply(env, id, "push", push, 2, &ignored));
    struct EnvRecordList *entry = env->record_lists;
    int64_t *data_before = entry->storage.data; int64_t first_before = data_before[0];
    Value args[] = {handle, integer(0), record_value(&new_record)};
    int argc = 3;
    if (!strcmp(operation, "push")) { args[1] = args[2]; argc = 2; }
    else if (!strcmp(operation, "get") || !strcmp(operation, "remove")) argc = 2;
    else if (!strcmp(operation, "pop")) argc = 1;
    Value out = integer(441), sentinel = out;
    begin(at, once); bool ok = env_record_list_apply(env, id, operation, args, argc, &out);
    *count = attempts; end();
    if (at != SIZE_MAX) {
        CHECK(!ok && failures && !memcmp(&out, &sentinel, sizeof out));
        CHECK(entry->storage.data == data_before && entry->storage.length == 1 && entry->storage.capacity == 1);
        CHECK(entry->storage.data[0] == first_before && env->record_results == NULL);
    } else {
        CHECK(ok);
        if (out.type == VAL_STRUCT) CHECK(!strcmp(out.as.struct_val->field_values[0].as.string_val, "old"));
        if (!strcmp(operation, "push") || !strcmp(operation, "insert")) CHECK(entry->storage.length == 2);
        else if (!strcmp(operation, "pop") || !strcmp(operation, "remove")) CHECK(entry->storage.length == 0);
        else CHECK(entry->storage.length == 1);
        next[0] = 'X';
        if (!strcmp(operation, "set") || !strcmp(operation, "insert")) {
            StructValue *stored = (StructValue *)(intptr_t)entry->storage.data[0];
            CHECK(!strcmp(stored->field_values[0].as.string_val, "next"));
        }
        CHECK(env_record_list_apply(env, id, "clear", &handle, 1, &ignored));
        CHECK(entry->storage.length == 0);
        if (out.type == VAL_STRUCT) CHECK(!strcmp(out.as.struct_val->field_values[0].as.string_val, "old"));
    }
    free_environment(env); CHECK(live == 0);
}
static void list_controls(void) {
    const char *operations[] = {"push", "insert", "set", "get", "remove", "pop"};
    for (size_t op = 0; op < sizeof(operations) / sizeof(*operations); ++op) {
        size_t count; list_attempt(operations[op], SIZE_MAX, false, &count); CHECK(count > 0);
        for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
            size_t ignored; list_attempt(operations[op], i, once != 0, &ignored);
            list_attempt(operations[op], SIZE_MAX, false, &ignored);
        }
    }
    NominalIdentity id; Environment *env = list_environment(&id); Value handle, out = integer(5), zero = integer(0);
    CHECK(env_record_list_apply(env, id, "with_capacity", &zero, 1, &handle));
    Value args[] = {handle, integer(-1), integer(1)};
    CHECK(!env_record_list_apply(env, id, "insert", args, 3, &out) && out.as.int_val == 5);
    CHECK(!env_record_list_apply(env, id, "pop", &handle, 1, &out) && out.as.int_val == 5);
    CHECK(env_record_list_apply(env, id, "free", &handle, 1, &out));
    CHECK(!env_record_list_apply(env, id, "length", &handle, 1, &out));
    NominalIdentity unknown = {TYPE_UNKNOWN, 0}; CHECK(!env_record_list_identity(env, handle, &unknown));
    free_environment(env);
}

static void publication_controls(void) {
    NominalIdentity id; Environment *env = list_environment(&id);
    for (int once = 0; once < 2; ++once) for (size_t index = 0; index < 2; ++index) {
        Value out = integer(67); begin(index, once != 0);
        CHECK(!env_record_list_apply(env, id, "new", NULL, 0, &out)); end();
        CHECK(failures && out.type == VAL_INT && out.as.int_val == 67 && env->record_lists == NULL && !live);
    }
    for (int once = 0; once < 2; ++once) for (size_t index = 0; index < 3; ++index) {
        Value out = integer(67); begin(index, once != 0);
        CHECK(!env_value_snapshot(env, text_value("kept"), &out)); end();
        CHECK(failures && out.type == VAL_INT && out.as.int_val == 67 && env->record_results == NULL && !live);
    }
    TupleValue tuple = {0}; Value owned;
    CHECK(env_clone_value_snapshot(tuple_value(&tuple), &owned));
    for (int once = 0; once < 2; ++once) for (size_t index = 0; index < 2; ++index) {
        begin(index, once != 0); CHECK(!env_retire_value(env, owned)); end();
        CHECK(failures == 1 && !live && env->record_results == NULL && owned.as.tuple_val->element_count == 0);
    }
    CHECK(env_retire_value(env, owned)); CHECK(!env_retire_value(env, owned));
    free_environment(env);
}

/* I exercise the actual publication transaction, including a new table and a
 * 12-root growth boundary. Old graph addresses and every old slot must survive. */
static void index_attempt(bool retirement, size_t preload, size_t at, bool once, size_t *count) {
    Environment *env = create_environment(); CHECK(env);
    Value retained[12]; struct EnvRecordResult *entries[12];
    CHECK(preload <= 12);
    for (size_t i = 0; i < preload; ++i) {
        CHECK(env_value_snapshot(env, text_value("retained"), &retained[i]));
        entries[i] = env->record_results;
    }
    struct EnvRecordIndex *old_index = env->record_result_index;
    struct EnvRecordResult *old_head = env->record_results;
    uintptr_t old_address = (uintptr_t)old_index;
    size_t bytes = 0; unsigned char *old_bytes = NULL;
    if (old_index) {
        CHECK(old_index->capacity == 16 && old_index->count == preload);
        CHECK(record_index_bytes(old_index->capacity, &bytes));
        old_bytes = malloc(bytes); CHECK(old_bytes); memcpy(old_bytes, old_index, bytes);
    }
    TupleValue empty = {0}; Value owned = integer(0);
    if (retirement) CHECK(env_clone_value_snapshot(tuple_value(&empty), &owned));
    Value saved_owned, out = integer(883), sentinel;
    memcpy(&saved_owned, &owned, sizeof owned); memcpy(&sentinel, &out, sizeof out);
    begin(at, once);
    bool ok = retirement ? env_retire_value(env, owned) : env_value_snapshot(env, text_value("new"), &out);
    *count = attempts; end();
    if (at != SIZE_MAX) {
        CHECK(!ok && failures && !memcmp(&out, &sentinel, sizeof out));
        CHECK(!memcmp(&owned, &saved_owned, sizeof owned));
        CHECK(env->record_results == old_head && env->record_result_index == old_index);
        if (old_index) CHECK(!memcmp(old_bytes, old_index, bytes));
        CHECK(live == 0);
        if (retirement)
            CHECK(owned.as.tuple_val->element_count == 0 && !owned.as.tuple_val->elements);
    } else {
        CHECK(ok && failures == 0 && env->record_result_index->count == preload + 1);
        CHECK(env->record_result_index->capacity == (preload == 12 ? 32 : 16));
        if (preload == 12) CHECK((uintptr_t)env->record_result_index != old_address);
        if (preload && preload != 12) CHECK(env->record_result_index == old_index);
        CHECK(env->record_results->next == old_head);
        CHECK(env_record_result_borrowed(env, retirement ? owned : out));
        if (!retirement) CHECK(!strcmp(out.as.string_val, "new"));
    }
    for (size_t i = 0; i < preload; ++i) {
        CHECK(env_record_result_borrowed(env, retained[i]));
        CHECK(!strcmp(retained[i].as.string_val, "retained"));
        size_t slot; bool found;
        CHECK(record_index_slot(env->record_result_index, retained[i], &slot, &found) && found);
        CHECK(env->record_result_index->slots[slot] == entries[i]);
    }
    if (at != SIZE_MAX) {
        Value recovered = integer(883);
        CHECK(retirement ? env_retire_value(env, owned) : env_value_snapshot(env, text_value("new"), &recovered));
        CHECK(env->record_result_index->count == preload + 1);
        CHECK(env_record_result_borrowed(env, retirement ? owned : recovered));
        for (size_t i = 0; i < preload; ++i) CHECK(env_record_result_borrowed(env, retained[i]));
    }
    free(old_bytes); free_environment(env); CHECK(live == 0);
}
static void index_allocation_controls(void) {
    const size_t preloads[] = {0, 3, 12};
    for (int retirement = 0; retirement < 2; ++retirement)
        for (size_t p = 0; p < sizeof(preloads) / sizeof(*preloads); ++p) {
            size_t count;
            index_attempt(retirement != 0, preloads[p], SIZE_MAX, false, &count);
            CHECK(count == (size_t)(retirement ? 1 : 2) + (preloads[p] == 3 ? 0 : 1));
            for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
                size_t ignored;
                index_attempt(retirement != 0, preloads[p], i, once != 0, &ignored);
                index_attempt(retirement != 0, preloads[p], SIZE_MAX, false, &ignored);
            }
        }
}
static void index_identity_controls(void) {
    Environment *a = create_environment(), *b = create_environment(); CHECK(a && b);
    CHECK(!env_record_result_borrowed(NULL, integer(0)));
    CHECK(!env_record_result_borrowed(a, integer(0)));
    Value null_record = record_value(NULL); CHECK(!env_record_result_borrowed(a, null_record));
    char text[] = "same"; Value fields[] = {text_value(text)}; char *names[] = {"text"};
    StructValue record = {.struct_name = "Item", .field_names = names, .field_values = fields, .field_count = 1};
    Value elements[] = {record_value(&record)}; TupleValue tuple = {.elements = elements, .element_count = 1};
    Value string, second_string, record_root, tuple_root, other;
    begin(SIZE_MAX, false);
    CHECK(env_value_snapshot(a, text_value(text), &string));
    CHECK(env_value_snapshot(a, text_value(text), &second_string));
    CHECK(string.as.string_val != second_string.as.string_val);
    CHECK(env_value_snapshot(a, record_value(&record), &record_root));
    CHECK(env_value_snapshot(a, tuple_value(&tuple), &tuple_root));
    CHECK(env_value_snapshot(b, text_value(text), &other));
    end(); CHECK(failures == 0);
    CHECK(env_record_result_borrowed(a, string) && env_record_result_borrowed(a, second_string));
    CHECK(env_record_result_borrowed(a, record_root) && env_record_result_borrowed(a, tuple_root));
    CHECK(!env_record_result_borrowed(a, text_value(text)));
    CHECK(!env_record_result_borrowed(a, record_root.as.struct_val->field_values[0]));
    CHECK(!env_record_result_borrowed(a, tuple_root.as.tuple_val->elements[0]));
    CHECK(!env_record_result_borrowed(a, other) && !env_record_result_borrowed(b, string));
    /* A char pointer to an actual live object's representation is valid. The
     * query does not read it as text, and its different tag must still miss. */
    CHECK(!env_record_result_borrowed(a, text_value((char *)tuple_root.as.tuple_val)));
    size_t before = a->record_result_index->count;
    CHECK(!env_retire_value(a, tuple_root) && a->record_result_index->count == before);
    text[0] = 'X'; CHECK(!strcmp(string.as.string_val, "same"));
    free_environment(a);
    CHECK(env_record_result_borrowed(b, other) && !strcmp(other.as.string_val, "same"));
    free_environment(b); CHECK(live == 0);
}
static void index_collision_and_limits(void) {
    Environment *env = create_environment(); CHECK(env);
    Value pool[33]; TupleValue empty = {0}; size_t buckets[16][3], used[16] = {0}, selected = SIZE_MAX;
    /* Thirty-three live roots in sixteen buckets guarantee a bucket of three.
     * All addresses come from actual successful graph allocations. */
    begin(SIZE_MAX, false);
    for (size_t i = 0; i < 33; ++i) {
        CHECK(env_clone_value_snapshot(tuple_value(&empty), &pool[i]));
        size_t bucket = record_result_hash(pool[i]) & 15;
        if (used[bucket] < 3) buckets[bucket][used[bucket]++] = i;
        if (used[bucket] == 3) selected = bucket;
    }
    CHECK(selected != SIZE_MAX);
    size_t first = buckets[selected][0], second = buckets[selected][1], missing = buckets[selected][2];
    CHECK(env_retire_value(env, pool[first]) && env_retire_value(env, pool[second]));
    CHECK(env->record_result_index->capacity == 16 && env->record_result_index->count == 2);
    CHECK(env->record_result_index->slots[selected]->value.as.tuple_val == pool[first].as.tuple_val);
    CHECK(env->record_result_index->slots[(selected + 1) & 15]->value.as.tuple_val == pool[second].as.tuple_val);
    end(); CHECK(failures == 0);
    /* I forbid allocation through actual lookup and duplicate refusal while
     * already tracked roots remain live; begin() intentionally requires zero. */
    attempts = failures = 0; failure_at = 0; transient = false; observing = true;
    CHECK(env_record_result_borrowed(env, pool[first]) && env_record_result_borrowed(env, pool[second]));
    CHECK(!env_record_result_borrowed(env, pool[missing]));
    CHECK(!env_retire_value(env, pool[first]));
    end(); CHECK(attempts == 0 && failures == 0);
    size_t bytes = 71;
    CHECK(!record_index_bytes(0, &bytes) && bytes == 71);
    CHECK(!record_index_bytes(15, &bytes) && bytes == 71);
    CHECK(!record_index_bytes(24, &bytes) && bytes == 71);
    CHECK(!record_index_bytes(SIZE_MAX, &bytes) && bytes == 71);
    CHECK(!record_index_bytes(SIZE_MAX / 2 + 1, &bytes) && bytes == 71);
    CHECK(!record_index_bytes(16, NULL));
    CHECK(record_index_bytes(16, &bytes) && bytes == sizeof(struct EnvRecordIndex) + 16 * sizeof(struct EnvRecordResult *));
    struct EnvRecordIndex *index = env->record_result_index;
    struct EnvRecordResult candidate = {.next = env->record_results, .value = pool[missing]};
    struct EnvRecordResult saved; memcpy(&saved, &candidate, sizeof candidate);
    index->count = SIZE_MAX;
    CHECK(!record_result_publish(env, &candidate));
    CHECK(!memcmp(&candidate, &saved, sizeof candidate) && env->record_result_index == index && index->count == SIZE_MAX);
    index->count = 2;
    CHECK(env_record_result_borrowed(env, pool[first]) && !env_record_result_borrowed(env, pool[missing]));
    for (size_t i = 0; i < 33; ++i) if (i != first && i != second) env_discard_value_snapshot(pool[i]);
    free_environment(env); CHECK(live == 0);
}

static void provider_controls(void) {
    Environment *a = create_environment(), *b = create_environment(); CHECK(a && b);
    EnvEvaluationProvider *pa = env_provider_new(), *pb = env_provider_new(); CHECK(pa && pb);
    CHECK(env_register_provider(a, pa)); CHECK(pa->references == 2);
    CHECK(env_register_provider(a, pa)); CHECK(pa->references == 2);
    CHECK(env_register_provider(b, pa)); CHECK(pa->references == 3);
    CHECK(env_acquire_evaluation_lease(a)); CHECK(env_acquire_evaluation_lease(a));
    CHECK(!env_provider_close(pa) && pa->alive && pa->leases == 2);
    CHECK(env_register_provider(a, pb)); CHECK(pb->leases == 2);
    CHECK(env_acquire_evaluation_lease(b) && pa->leases == 3);
    env_release_evaluation_lease(a); env_release_evaluation_lease(b); env_release_evaluation_lease(a);
    CHECK(pa->leases == 0 && pb->leases == 0 && env_can_destroy(a));
    CHECK(env_provider_close(pb)); CHECK(!env_acquire_evaluation_lease(a));
    CHECK(a->evaluation_leases == 0 && pa->leases == 0); /* No partial first-provider increment. */
    CHECK(!env_register_provider(b, pb));
    env_provider_release(pb); CHECK(env_provider_close(pa)); env_provider_release(pa);
    free_environment(a); free_environment(b);
    a = create_environment(); pa = env_provider_new(); pb = env_provider_new(); CHECK(a && pa && pb);
    CHECK(env_register_provider(a, pa));
    pa->leases = SIZE_MAX; CHECK(!env_acquire_evaluation_lease(a) && a->evaluation_leases == 0);
    pa->leases = 0; a->evaluation_leases = SIZE_MAX;
    CHECK(!env_acquire_evaluation_lease(a));
    pb->leases = 1; CHECK(!env_register_provider(a, pb) && pb->references == 1 && pb->leases == 1);
    pb->leases = 0; a->evaluation_leases = 0; pb->references = SIZE_MAX;
    CHECK(!env_register_provider(a, pb)); pb->references = 1;
    CHECK(env_acquire_evaluation_lease(a));
    for (int once = 0; once < 2; ++once) {
        begin(0, once != 0); CHECK(!env_register_provider(a, pb)); end();
        CHECK(failures == 1 && pb->references == 1 && live == 0 && a->evaluation_providers->next == NULL);
        begin(0, once != 0); CHECK(env_provider_new() == NULL); end(); CHECK(failures == 1 && live == 0);
    }
    CHECK(env_register_provider(a, pb)); CHECK(pb->leases == 1 && pa->leases == 1);
    env_release_evaluation_lease(a); CHECK(pb->leases == 0 && pa->leases == 0);
    /* A dead later edge must not increment an earlier live edge. */
    CHECK(env_provider_close(pa)); CHECK(!env_acquire_evaluation_lease(a)); CHECK(pb->leases == 0);
    CHECK(env_provider_close(pb)); env_provider_release(pa); env_provider_release(pb); free_environment(a);
}

static unsigned callbacks, dropped_args, dropped_results;
static Environment *leased_env;
static bool scheduler_clone(Value input, Value *out) { return env_clone_value_snapshot(input, out); }
static void scheduler_drop(Value value) { ++dropped_results; env_discard_value_snapshot(value); }
static void scheduler_arg_drop(void *arg) {
    CHECK(arg == leased_env); ++dropped_args; env_release_evaluation_lease(leased_env);
}
static Value callback(void *arg, int id) {
    CHECK(arg == leased_env && !env_can_destroy(leased_env)); ++callbacks;
    Value value = text_value("callback"), owned;
    CHECK(env_clone_value_snapshot(value, &owned));
    if (callbacks == 1) {
        nano_coro_complete(value);
        CHECK(nano_coro_is_done(id)); CHECK(!nano_coro_release(id) && !nano_coro_cancel(id));
        CHECK(!env_can_destroy(leased_env)); nano_coro_error("ignored after done");
    }
    return owned;
}
static Value nested_callback(void *arg, int id) {
    (void)arg; CHECK(!nano_coro_release(id));
    CHECK(env_acquire_evaluation_lease(leased_env));
    int child = nano_coro_spawn_owned(callback, leased_env, scheduler_arg_drop, scheduler_drop, scheduler_clone);
    CHECK(child >= 0); Value value; CHECK(nano_coro_await_copy(child, &value));
    CHECK(nano_coro_release(child)); return value;
}
static Value error_callback(void *arg, int id) {
    CHECK(arg == leased_env && !env_can_destroy(leased_env));
    nano_coro_error("first error"); nano_coro_error("second error");
    CHECK(!nano_coro_release(id) && !nano_coro_cancel(id));
    nano_coro_complete(integer(1));
    return integer(2);
}
static Value complete_failure_callback(void *arg, int id) {
    CHECK(arg == leased_env);
    begin(0, false); nano_coro_complete(text_value("unpublished")); end();
    CHECK(failures == 1 && live == 0 && !nano_coro_release(id));
    return integer(3);
}
static Value cycle_callback(void *arg, int id) {
    CHECK(arg == leased_env);
    Value result = nano_coro_await_id(id);
    CHECK(result.type == VAL_VOID && !nano_coro_release(id));
    return integer(4);
}

static void scheduler_controls(void) {
    nano_scheduler_init(); leased_env = create_environment(); CHECK(leased_env);
    CHECK(env_acquire_evaluation_lease(leased_env));
    int id = nano_coro_spawn_owned(callback, leased_env, scheduler_arg_drop, scheduler_drop, scheduler_clone);
    CHECK(id >= 0 && !env_can_destroy(leased_env));
    Value copy; CHECK(nano_coro_await_copy(id, &copy));
    CHECK(callbacks == 1 && dropped_args == 1 && dropped_results == 1 && env_can_destroy(leased_env));
    CHECK(!strcmp(copy.as.string_val, "callback"));
    Value borrowed = nano_coro_result(id);
    CHECK(borrowed.type == VAL_STRING && borrowed.as.string_val != copy.as.string_val);
    CHECK(!strcmp(borrowed.as.string_val, "callback"));
    CHECK(nano_coro_release(id) && dropped_results == 2); CHECK(!nano_coro_release(id));
    CHECK(!strcmp(copy.as.string_val, "callback")); env_discard_value_snapshot(copy);
    CHECK(env_acquire_evaluation_lease(leased_env));
    id = nano_coro_spawn_owned(callback, leased_env, scheduler_arg_drop, scheduler_drop, scheduler_clone);
    CHECK(id >= 0 && nano_coro_cancel(id)); CHECK(callbacks == 1 && dropped_args == 2);
    CHECK(env_can_destroy(leased_env) && nano_coro_release(id));
    CHECK(env_acquire_evaluation_lease(leased_env));
    id = nano_coro_spawn_owned(nested_callback, leased_env, scheduler_arg_drop, scheduler_drop, scheduler_clone);
    CHECK(id >= 0 && nano_coro_await_copy(id, &copy));
    CHECK(callbacks == 2 && dropped_args == 4 && env_can_destroy(leased_env));
    CHECK(nano_coro_release(id)); env_discard_value_snapshot(copy);
    CHECK(env_acquire_evaluation_lease(leased_env));
    id = nano_coro_spawn_owned(callback, leased_env, scheduler_arg_drop, scheduler_drop, scheduler_clone);
    CHECK(id >= 0 && nano_scheduler_step());
    Value sentinel = integer(81); copy = sentinel;
    begin(0, false); CHECK(!nano_coro_result_copy(id, &copy)); end();
    CHECK(failures && !memcmp(&copy, &sentinel, sizeof copy) && live == 0);
    CHECK(nano_coro_result_copy(id, &copy)); CHECK(nano_coro_release(id));
    CHECK(!strcmp(copy.as.string_val, "callback")); env_discard_value_snapshot(copy);
    CoroFn failing[] = {error_callback, complete_failure_callback, cycle_callback};
    for (size_t i = 0; i < sizeof(failing) / sizeof(*failing); ++i) {
        CHECK(env_acquire_evaluation_lease(leased_env));
        id = nano_coro_spawn_owned(failing[i], leased_env, scheduler_arg_drop, scheduler_drop, scheduler_clone);
        CHECK(id >= 0 && nano_scheduler_step() && nano_coro_is_done(id) && env_can_destroy(leased_env));
        copy = integer(81); CHECK(!nano_coro_result_copy(id, &copy) && copy.as.int_val == 81);
        for (int slot = 0; slot < MAX_COROUTINES; ++slot) if (g_scheduler.coroutines[slot].id == id) {
            CHECK(g_scheduler.coroutines[slot].status == CORO_ERROR);
            if (!i) CHECK(!strcmp(g_scheduler.coroutines[slot].error_msg, "first error"));
        }
        CHECK(nano_coro_release(id));
    }
    CHECK(nano_scheduler_pending_count() == 0); free_environment(leased_env); leased_env = NULL;
}

static void task_allocation_attempt(size_t at, bool once, bool bundle, size_t *count) {
    TypeInfo leaf = {.base_type = TYPE_STRING}; TypeInfo *params[] = {&leaf};
    Type tags[] = {TYPE_STRING};
    FunctionSignature signature = {.param_count = 1, .param_types = tags, .param_type_info = params,
        .return_type = TYPE_STRING, .return_type_info = &leaf};
    Value function = {0}; function.type = VAL_FUNCTION;
    function.as.function_val.function_name = "queued_fixture_target";
    function.as.function_val.signature = &signature;
    Environment *env = create_environment(); CHECK(env);
    Value out = integer(53), sentinel = out; int id = -72;
    begin(at, once);
    bool ok = bundle ? lifetime_prepare_bundle(env, function, &id) : lifetime_task_clone(function, &out);
    *count = attempts; end();
    if (at != SIZE_MAX) {
        CHECK(!ok && failures && id == -72 && !memcmp(&out, &sentinel, sizeof out));
        CHECK(env_can_destroy(env) && live == 0);
    } else if (bundle) {
        CHECK(ok && id >= 0 && !env_can_destroy(env));
        CHECK(nano_coro_cancel(id) && env_can_destroy(env) && nano_coro_release(id));
    } else {
        CHECK(ok && out.type == VAL_FUNCTION && out.as.function_val.signature != &signature);
        lifetime_task_drop(out);
    }
    free_environment(env); CHECK(live == 0);
}
static void task_allocation_controls(void) {
    for (int bundle = 0; bundle < 2; ++bundle) {
        size_t count; task_allocation_attempt(SIZE_MAX, false, bundle != 0, &count); CHECK(count > 4);
        for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
            size_t ignored; task_allocation_attempt(i, once != 0, bundle != 0, &ignored);
            task_allocation_attempt(SIZE_MAX, false, bundle != 0, &ignored);
        }
    }
    Environment *env = create_environment(); CHECK(env);
    int reserved[MAX_COROUTINES];
    for (int i = 0; i < MAX_COROUTINES; ++i) {
        reserved[i] = nano_coro_spawn(callback, NULL); CHECK(reserved[i] >= 0);
    }
    int id = -72; begin(SIZE_MAX, false);
    CHECK(!lifetime_prepare_bundle(env, text_value("queued"), &id)); end();
    CHECK(id == -72 && env_can_destroy(env) && live == 0 && failures == 0);
    for (int i = 0; i < MAX_COROUTINES; ++i) CHECK(nano_coro_cancel(reserved[i]) && nano_coro_release(reserved[i]));
    int saved_count = g_scheduler.count; g_scheduler.count = INT_MAX;
    CHECK(!lifetime_prepare_bundle(env, text_value("queued"), &id) && env_can_destroy(env));
    g_scheduler.count = saved_count;
    free_environment(env);
}
static void borrowed_staging_controls(void) {
    Environment *env = create_environment(); CHECK(env);
    char *names[] = {"text"}; Value fields[] = {text_value("original")};
    StructValue record = {.struct_name = "Item", .field_count = 1, .field_names = names, .field_values = fields};
    env_define_var(env, "owner", TYPE_STRUCT, true, record_value(&record));
    Symbol *owner = env_get_var(env, "owner"); CHECK(owner && owner->value.type == VAL_STRUCT);
    Parameter parameter = {.name = "view", .type = TYPE_BORROW_SHARED};
    Function fn = {.name = "fixture_view", .param_count = 1, .params = &parameter, .return_type = TYPE_VOID};
    env_define_function(env, fn);
    ASTNode expression = {0}; expression.type = AST_IDENTIFIER; expression.as.identifier = "owner";
    Value shared = lifetime_stage_argument(env, &expression, "fixture_view");
    CHECK(shared.as.struct_val == owner->value.as.struct_val && env->record_results == NULL);
    parameter.type = TYPE_BORROW_MUT;
    Value exclusive = lifetime_stage_argument(env, &expression, "fixture_view");
    CHECK(exclusive.as.struct_val == shared.as.struct_val && env->record_results == NULL);
    parameter.type = TYPE_STRUCT;
    Value copied = lifetime_stage_argument(env, &expression, "fixture_view");
    CHECK(copied.as.struct_val != shared.as.struct_val && env_record_result_borrowed(env, copied));
    fields[0] = text_value("replacement"); env_set_var(env, "owner", record_value(&record));
    CHECK(!strcmp(copied.as.struct_val->field_values[0].as.string_val, "original"));
    free_environment(env);
}

extern void lifetime_cache_initialize(void);
extern EnvEvaluationProvider *lifetime_cache_provider(void);
static void cache_fatal_cleanup(void) {
    if (live) { fprintf(stderr, "I retained %zu allocations at failed cache initialization.\n", live); _Exit(90); }
    puts("I released every observed cache initialization allocation.");
}
static void cache_init_control(size_t at, bool once) {
    CHECK(!atexit(cache_fatal_cleanup));
    begin(at, once); lifetime_cache_initialize(); end();
    CHECK(at == SIZE_MAX && attempts == 4 && failures == 0);
    clear_module_cache(); CHECK(live == 0);
}
static void cache_registration_control(const char *path) {
    Environment *a = create_environment(), *b = create_environment(); CHECK(a && b);
    ASTNode *ast = load_module(path, a); CHECK(ast);
    EnvEvaluationProvider *provider = lifetime_cache_provider(); CHECK(provider && provider->references == 2);
    CHECK(env_acquire_evaluation_lease(a));
    for (int once = 0; once < 2; ++once) {
        begin(0, once != 0); CHECK(load_module(path, b) == NULL); end();
        CHECK(failures == 1 && live == 0 && b->evaluation_providers == NULL);
        CHECK(provider->references == 2 && provider->leases == 1 && provider->alive);
    }
    CHECK(load_module(path, b) == ast && provider->references == 3);
    CHECK(env_acquire_evaluation_lease(b) && provider->leases == 2);
    env_release_evaluation_lease(a); env_release_evaluation_lease(b);
    free_environment(a); free_environment(b); clear_module_cache();
}

/* I retain the existing numeric-link index across actual evaluator cleanup.
 * No allocation may hide a full rebuild during pop/lookup; slot insertion still
 * follows the normal same-file synchronization path. */
static void nominal_import_attempt(const char *importer, const char *owner,
                                   size_t failure, bool once, size_t *count) {
    Environment *env = create_environment(); CHECK(env);
    StructDef record = {0}; record.name = strdup("Item");
    record.module_name = (char *)owner;
    CHECK(record.name && (!owner || record.module_name)); env_define_struct(env, record);
    NominalIdentity identity = env_nominal_identity(env, "Item", owner, TYPE_STRUCT);
    CHECK(identity.ordinal);
    CHECK(env_register_nominal_import(env, "Existing", "Retained", identity));
    struct EnvNominalImport *previous = env->nominal_imports;
    begin(failure, once);
    bool ok = env_register_nominal_import(env, importer, "ImportedItem", identity);
    end(); *count = attempts;
    if (failure == SIZE_MAX) {
        CHECK(ok && env->nominal_imports != previous);
        CHECK(env_nominal_identity(env, "ImportedItem", importer, TYPE_STRUCT).ordinal == identity.ordinal);
        size_t retained = live;
        CHECK(!observing); attempts = failures = 0; failure_at = 0; transient = false; observing = true;
        CHECK(env_register_nominal_import(env, importer, "ImportedItem", identity));
        end(); CHECK(attempts == 0 && failures == 0 && live == retained);
    } else {
        CHECK(!ok && failures && env->nominal_imports == previous && live == 0);
        CHECK(!env_nominal_identity(env, "ImportedItem", importer, TYPE_STRUCT).ordinal);
        CHECK(env_nominal_identity(env, "Retained", "Existing", TYPE_STRUCT).ordinal == identity.ordinal);
        CHECK(env_register_nominal_import(env, importer, "ImportedItem", identity));
    }
    free_environment(env); CHECK(live == 0);
}
static void nominal_import_controls(void) {
    for (int importer = 0; importer < 2; ++importer) for (int owner = 0; owner < 2; ++owner) {
        size_t count; nominal_import_attempt(importer ? "Caller" : NULL, owner ? "Source" : NULL, SIZE_MAX, false, &count);
        CHECK(count == (size_t)(3 + importer + owner));
        for (int once = 0; once < 2; ++once) for (size_t i = 0; i < count; ++i) {
            size_t ignored;
            nominal_import_attempt(importer ? "Caller" : NULL, owner ? "Source" : NULL, i, once != 0, &ignored);
        }
    }
    Environment *env = create_environment(); CHECK(env);
    const char *owners[] = {"First", "Second", "Caller"};
    for (int i = 0; i < 3; ++i) {
        StructDef record = {0}; record.name = strdup("Item"); record.module_name = (char *)owners[i];
        CHECK(record.name && record.module_name); env_define_struct(env, record);
    }
    NominalIdentity first = env_nominal_identity(env, "Item", "First", TYPE_STRUCT);
    NominalIdentity second = env_nominal_identity(env, "Item", "Second", TYPE_STRUCT);
    CHECK(first.ordinal && second.ordinal && first.ordinal != second.ordinal);
    CHECK(env_register_nominal_import(env, "User", "Item", first));
    struct EnvNominalImport *before = env->nominal_imports;
    CHECK(!env_register_nominal_import(env, "User", "Item", second) && env->nominal_imports == before);
    CHECK(env_register_nominal_import(env, "Reverse", "Item", second));
    CHECK(!env_register_nominal_import(env, "Reverse", "Item", first));
    CHECK(env_register_nominal_import(env, "Caller", "Item", first));
    CHECK(env_nominal_identity(env, "Item", "Caller", TYPE_STRUCT).ordinal == 3);
    CHECK(!env_nominal_identity(env, "Item", "Unrelated", TYPE_STRUCT).ordinal);
    CHECK(!env_nominal_identity(env, "Item", "User", TYPE_ENUM).ordinal);
    char saved = env->structs[first.ordinal - 1].name[0];
    env->structs[first.ordinal - 1].name[0] = 'X';
    CHECK(!env_nominal_identity(env, "Item", "User", TYPE_STRUCT).ordinal);
    env->structs[first.ordinal - 1].name[0] = saved;
    CHECK(env_nominal_identity(env, "Item", "User", TYPE_STRUCT).ordinal == first.ordinal);
    CHECK(!env_register_nominal_import(env, "User", "Invalid", (NominalIdentity){TYPE_STRUCT, 999}));
    EnumDef enumeration = {0}; enumeration.name = strdup("Color"); enumeration.module_name = "First";
    env_define_enum(env, enumeration);
    UnionDef sum = {0}; sum.name = strdup("Choice"); sum.module_name = strdup("First"); env_define_union(env, sum);
    CHECK(env_register_nominal_import(env, "User", "Color", env_nominal_identity(env, "Color", "First", TYPE_ENUM)));
    CHECK(env_register_nominal_import(env, "User", "Choice", env_nominal_identity(env, "Choice", "First", TYPE_UNION)));
    CHECK(env_nominal_identity(env, "Color", "User", TYPE_ENUM).ordinal == 1);
    CHECK(env_nominal_identity(env, "Choice", "User", TYPE_UNION).ordinal == 1);
    NominalIdentity kinds[] = {first, env_nominal_identity(env, "Color", "First", TYPE_ENUM),
                              env_nominal_identity(env, "Choice", "First", TYPE_UNION)};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) if (a != b) {
        char binding[32];
        int written = snprintf(binding, sizeof binding, "shared_%d_%d", a, b);
        CHECK(written >= 0 && (size_t)written < sizeof binding);
        CHECK(env_register_nominal_import(env, "Kinds", binding, kinds[a]));
        struct EnvNominalImport *saved_row = env->nominal_imports;
        CHECK(!env_register_nominal_import(env, "Kinds", binding, kinds[b]));
        CHECK(env->nominal_imports == saved_row);
        CHECK(env_nominal_identity(env, binding, "Kinds", kinds[a].kind).ordinal == kinds[a].ordinal);
        CHECK(!env_nominal_identity(env, binding, "Kinds", kinds[b].kind).ordinal);
    }
    CHECK(env_register_nominal_import(env, "Caller", "Color", first));
    EnumDef local_enum = {0}; local_enum.name = strdup("LocalColor"); local_enum.module_name = "Local";
    env_define_enum(env, local_enum);
    CHECK(env_register_nominal_import(env, "Local", "LocalColor", first));
    CHECK(!env_nominal_identity(env, "LocalColor", "Local", TYPE_STRUCT).ordinal);
    CHECK(env_nominal_identity(env, "LocalColor", "Local", TYPE_ENUM).ordinal == 2);
    CHECK(env_register_nominal_import(env, "LocalRecord", "Item", kinds[2]));
    StructDef local_record = {0}; local_record.name = strdup("Item"); local_record.module_name = "LocalRecord";
    env_define_struct(env, local_record);
    CHECK(!env_nominal_identity(env, "Item", "LocalRecord", TYPE_UNION).ordinal);
    CHECK(env_nominal_identity(env, "Item", "LocalRecord", TYPE_STRUCT).ordinal == 4);
    Environment *other = create_environment(); CHECK(other);
    CHECK(!env_nominal_identity(other, "Item", "User", TYPE_STRUCT).ordinal);
    free_environment(other); free_environment(env);
}

static void string_binding_ownership_controls(void) {
    Environment *env = create_environment();
    Value fresh = text_value(strdup("original"));
    CHECK(fresh.as.string_val);
    env_define_var(env, "owner", TYPE_STRING, true, fresh);
    CHECK(env_get_var(env, "owner")->value.as.string_val == fresh.as.string_val);
    Value sentinel = create_int(812), out = sentinel;
    for (int once = 0; once < 2; ++once) {
        begin(0, once != 0);
        CHECK(!env_prepare_binding_string(env, TYPE_STRING, fresh, &out));
        end();
        CHECK(failures == 1 && live == 0);
        CHECK(out.type == VAL_INT && out.as.int_val == 812);
        CHECK(env_get_var(env, "owner")->value.as.string_val == fresh.as.string_val);
        CHECK(!strcmp(fresh.as.string_val, "original"));
    }
    begin(SIZE_MAX, false);
    CHECK(env_prepare_binding_string(env, TYPE_STRING, fresh, &out));
    end();
    CHECK(attempts == 1 && out.as.string_val != fresh.as.string_val);
    CHECK(!strcmp(out.as.string_val, "original"));
    env_discard_value_snapshot(out); CHECK(live == 0);
    int outer = env->symbol_count;
    env_define_var(env, "alias", TYPE_STRING, false, fresh);
    CHECK(env_get_var(env, "alias")->value.as.string_val != fresh.as.string_val);
    env_set_var(env, "owner", text_value("replacement"));
    CHECK(!strcmp(env_get_var(env, "alias")->value.as.string_val, "original"));
    lifetime_scope_release(env, outer, false);
    CHECK(!strcmp(env_get_var(env, "owner")->value.as.string_val, "replacement"));
    Value current = env_get_var(env, "owner")->value;
    env_define_var(env, "borrow", TYPE_BORROW_SHARED, false, current);
    CHECK(env_get_var(env, "borrow")->value.as.string_val == current.as.string_val);
    /* A value binding copied from a borrowed formal still needs its own owner. */
    env_define_var(env, "borrow_copy", TYPE_STRING, false, env_get_var(env, "borrow")->value);
    CHECK(env_get_var(env, "borrow_copy")->value.as.string_val != current.as.string_val);
    lifetime_scope_release(env, outer, true);
    CHECK(!strcmp(env_get_var(env, "owner")->value.as.string_val, "replacement"));
    free_environment(env);
}

static void evaluator_symbol_pop_controls(void) {
    Environment *env = create_environment();
    for (int i = 0; i < 128; ++i) {
        char name[32]; snprintf(name, sizeof name, "outer_%d", i);
        env_define_var(env, name, TYPE_INT, true, create_int(i));
    }
    int outer = env->symbol_count;
    for (int mode = 0; mode < 2; ++mode) {
        env_define_var(env, "outer_0", TYPE_INT, true, create_int(900));
        int middle = env->symbol_count;
        env_define_var(env, "temporary", TYPE_INT, true, create_int(901));
        CHECK(env_get_var(env, "outer_0")->value.as.int_val == 900);
        CHECK(env_get_var(env, "temporary")->value.as.int_val == 901);
        uintptr_t identity = (uintptr_t)env->symbol_index;
        CHECK(identity != 0);
        begin(SIZE_MAX, false);
        lifetime_scope_release(env, middle, mode != 0);
        lifetime_scope_release(env, outer, mode != 0);
        CHECK((uintptr_t)env->symbol_index == identity);
        CHECK(!env_get_var(env, "temporary"));
        CHECK(env_get_var(env, "outer_0")->value.as.int_val == 0);
        CHECK(attempts == 0 && failures == 0 && live == 0);
        end();
        env_define_var(env, "replacement", TYPE_INT, true, create_int(902));
        CHECK(env_get_var(env, "replacement")->value.as.int_val == 902);
        CHECK(env_get_var(env, "outer_127")->value.as.int_val == 127);
        lifetime_scope_release(env, outer, mode != 0);
        /* I also insert immediately after pop, without an intervening lookup. */
        env_define_var(env, "outer_0", TYPE_INT, true, create_int(903));
        CHECK(!env_get_var(env, "replacement"));
        CHECK(env_get_var(env, "outer_0")->value.as.int_val == 903);
        lifetime_scope_release(env, outer, mode != 0);
        CHECK(env_get_var(env, "outer_0")->value.as.int_val == 0);
    }
    lifetime_scope_release(env, 0, true);
    CHECK(!env_get_var(env, "outer_0"));
    env_define_var(env, "after_reset", TYPE_INT, false, create_int(904));
    CHECK(env_get_var(env, "after_reset")->value.as.int_val == 904);
    free_environment(env);
}

static Function *function_equivalent(Environment *env, const char *name) {
    Function *linear = env_get_function_with_index(env, name, false);
    Function *indexed = env_get_function(env, name);
    CHECK(indexed == linear);
    return indexed;
}

static void function_index_controls(void) {
    Environment *env = create_environment(); CHECK(env);
    char names[128][32];
    for (int i = 0; i < 128; ++i) {
        snprintf(names[i], sizeof names[i], "candidate_%d", i);
        env_define_function(env, (Function){.name=names[i], .module_name="Owner"});
        CHECK(!env->function_index);
        CHECK(function_equivalent(env, names[i]) == &env->functions[i]);
    }
    size_t collisions = 0;
    for (size_t b = 0; b < env->function_index->buckets; ++b) {
        int previous = -1;
        for (int i = env->function_index->heads[b]; i >= 0; i = env->function_index->next[i]) {
            CHECK(i > previous); if (previous >= 0) ++collisions; previous = i;
        }
    }
    CHECK(collisions > 0);
    for (int i = 0; i < 128; ++i) CHECK(function_equivalent(env, names[i]) == &env->functions[i]);
    CHECK(!function_equivalent(env, "absent_candidate"));
    CHECK(!function_equivalent(env, NULL));
    /* I force relocation independent of allocator address reuse. */
    Function *replacement = malloc((size_t)env->function_capacity * sizeof(*replacement)); CHECK(replacement);
    memcpy(replacement, env->functions, (size_t)env->function_count * sizeof(*replacement));
    free(env->functions); env->functions = replacement;
    CHECK(function_equivalent(env, names[127]) == &replacement[127]);
    env->function_count = 3;
    CHECK(!function_equivalent(env, names[127]));
    CHECK(function_equivalent(env, names[2]) == &replacement[2]);
    env_function_index_invalidate(env);
    env->functions[1] = (Function){.name="replacement", .module_name="Changed"};
    CHECK(!function_equivalent(env, names[1]));
    CHECK(function_equivalent(env, "replacement") == &env->functions[1]);
    env_define_function(env, (Function){.name="same", .module_name="First"});
    env_define_function(env, (Function){.name="same", .module_name="Second"});
    env_define_function(env, (Function){.name="same", .module_name="First"});
    env_define_function(env, (Function){0});
    env->current_module = "First"; CHECK(function_equivalent(env, "same") == &env->functions[3]);
    env->current_module = "Second"; CHECK(function_equivalent(env, "same") == &env->functions[4]);
    env->current_module = "Other"; CHECK(function_equivalent(env, "same") == &env->functions[3]);
    for (int owner = 0; owner < 2; ++owner) {
        env->current_module = owner ? "CallerB" : "CallerA";
        char **exports = malloc(sizeof(*exports)); CHECK(exports); exports[0] = strdup("same"); CHECK(exports[0]);
        env_register_namespace(env, "Alias", owner ? "Second" : "First", exports, 1, NULL, 0, NULL, 0, NULL, 0);
    }
    env->current_module = "CallerA"; CHECK(function_equivalent(env, "Alias.same") == &env->functions[3]);
    env->current_module = "CallerB"; CHECK(function_equivalent(env, "Alias.same") == &env->functions[4]);
    env->current_module = "Other"; CHECK(!function_equivalent(env, "Alias.same"));
    CHECK(!function_equivalent(env, "Alias.absent"));
    env_define_function(env, (Function){.name="array_push", .module_name="Other", .is_extern=true});
    CHECK(function_equivalent(env, "array_push") == &env->functions[7]);
    CHECK(!env_array_push_is_builtin(env, 1, 1));
    ASTNode body = {0};
    env_define_function(env, (Function){.name="array_push", .module_name="Other", .body=&body});
    CHECK(function_equivalent(env, "array_push") == &env->functions[7]);
    env->functions[8].is_extern = true;
    CHECK(function_equivalent(env, "array_push") == &env->functions[7]);
    env->current_module = "Unrelated";
    CHECK(env_function_is_builtin(function_equivalent(env, "array_push")));
    CHECK(env_array_push_is_builtin(env, 1, 1));
    env->current_module = "Other";
    CHECK(function_equivalent(env, "array_push") == &env->functions[7]);
    env_define_function(env, (Function){.name="str_length", .module_name="Other", .body=&body});
    CHECK(env_function_is_builtin(function_equivalent(env, "str_length")));
    /* I populate exact generated ordinals without invoking unrelated legacy allocators. */
    StructDef record = {0}; record.name = strdup("IndexItem"); CHECK(record.name); env_define_struct(env, record);
    env->generic_instances[0] = (GenericInstantiation){.list_element={TYPE_STRUCT,1}, .list_functions={11,0,0,0}};
    env->generic_instance_count = 1;
    env_define_function(env, (Function){.name="List_IndexItem_new", .is_extern=true});
    CHECK(env_generated_list_element(env, function_equivalent(env, "List_IndexItem_new")).ordinal == 1);
    env_define_function(env, (Function){.name="List_IndexItem_new", .module_name="Foreign", .body=&body});
    CHECK(function_equivalent(env, "List_IndexItem_new") == &env->functions[11]);
    env_define_function(env, (Function){.name="List_IndexItem_new", .module_name="Other", .body=&body});
    CHECK(function_equivalent(env, "List_IndexItem_new") == &env->functions[12]);
    env->generic_instances[0].list_functions[0] = 0;
    env->current_module = NULL;
    CHECK(function_equivalent(env, "List_IndexItem_new") == &env->functions[10]);
    env_function_index_invalidate(env);
    begin(SIZE_MAX, false);
    CHECK(function_equivalent(env, "same") == &env->functions[3]);
    size_t count = attempts; CHECK(count == 3 && live == 3 && !failures);
    env_function_index_invalidate(env); CHECK(live == 0); end();
    for (int once = 0; once < 2; ++once) for (size_t at = 0; at < count; ++at) {
        begin(at, once != 0);
        CHECK(function_equivalent(env, "same") == &env->functions[3]);
        CHECK(failures && !env->function_index && live == 0);
        if (!once) {
            for (int repeat = 0; repeat < 3; ++repeat) {
                CHECK(function_equivalent(env, "same") == &env->functions[3]);
                CHECK(!env->function_index && live == 0);
            }
        }
        failure_at = SIZE_MAX;
        CHECK(function_equivalent(env, "same") == &env->functions[3]);
        CHECK(env->function_index && live == 3);
        env_function_index_invalidate(env); CHECK(live == 0); end();
    }
    env->current_module = "Other";
    for (size_t at = 0; at < count; ++at) {
        begin(at, false);
        CHECK(function_equivalent(env, "array_push") == &env->functions[7]);
        CHECK(!env_array_push_is_builtin(env, 1, 1));
        CHECK(failures && !env->function_index && live == 0);
        end();
    }
    env->current_module = NULL;
    Environment *other = create_environment(); CHECK(other);
    env_define_function(other, (Function){.name="same"});
    CHECK(function_equivalent(other, "same") == &other->functions[0]);
    CHECK(function_equivalent(env, "same") == &env->functions[3]);
    CHECK(other->function_index != env->function_index);
    free_environment(other); free_environment(env);
    /* The index never borrows name storage for destruction. */
    env = create_environment(); CHECK(env);
    char *owned_name = strdup("temporary_function_name"); CHECK(owned_name);
    env_define_function(env, (Function){.name=owned_name});
    CHECK(function_equivalent(env, owned_name) == &env->functions[0]);
    free(owned_name); free_environment(env);
}

extern int lifetime_context_spawn(Environment *, Type, CoroFn, void *,
    CoroArgDropFn, CoroResultDropFn, CoroResultCloneFn);
static Environment *context_env;
static int context_id, context_args, context_results, context_runs, context_mode;
static bool context_fail_clone;
static Value context_graph, context_scalar_value;
static void context_arg_drop(void *arg) {
    CHECK(arg == context_env && !nano_coro_release(context_id));
    ++context_args;
}
static void context_result_drop(Value value) {
    CHECK(!nano_coro_release(context_id));
    if (value.type == VAL_STRUCT) CHECK(!env_can_destroy(context_env));
    ++context_results; env_discard_value_snapshot(value);
}
static bool context_clone(Value value, Value *out) {
    CHECK(!env_can_destroy(context_env) && !nano_coro_release(context_id));
    return !context_fail_clone && env_clone_value_snapshot(value, out);
}
static Value context_callback(void *arg, int id) {
    CHECK(arg == context_env && id == context_id && !env_can_destroy(context_env));
    CHECK(!nano_coro_release(id) && !nano_coro_cancel(id));
    ++context_runs;
    if (context_mode == 1) nano_coro_complete(context_graph);
    if (context_mode == 2) nano_coro_error("context error");
    Value copy; CHECK(env_clone_value_snapshot(context_graph, &copy)); return copy;
}
static Value context_scalar_callback(void *arg, int id) {
    CHECK(arg == context_env && id == context_id); return context_scalar_value;
}
static void contextual_task_controls(void) {
    for (size_t failure = 0; failure < 2; ++failure) {
        begin(failure, true);
        Environment *absent = create_environment();
        end(); CHECK(!absent && failures == 1 && live == 0);
    }
    context_env = create_environment(); CHECK(context_env);
    size_t references = context_env->task_identity->references;
    context_env->task_identity->references = SIZE_MAX;
    CHECK(!env_task_identity_retain(context_env->task_identity));
    context_env->task_identity->references = references;
    Value callable = create_function("held", NULL);
    CHECK(env_retire_value(context_env, callable));
    UnionValue borrowed_union = {.union_name = "Borrowed", .variant_name = "Empty"};
    Array borrowed_array = {.element_type = VAL_INT};
    Value union_leaf = {0}, array_leaf = {0};
    union_leaf.type = VAL_UNION; union_leaf.as.union_val = &borrowed_union;
    array_leaf.type = VAL_ARRAY; array_leaf.as.array_val = &borrowed_array;
    char *names[] = {"callback", "union", "array"};
    Value values[] = {callable, union_leaf, array_leaf};
    context_graph = create_struct("Graph", names, values, 3);
    for (context_mode = 0; context_mode < 3; ++context_mode) {
        context_args = context_results = context_runs = 0;
        context_id = lifetime_context_spawn(context_env, TYPE_STRUCT, context_callback,
            context_env, context_arg_drop, context_result_drop, context_clone);
        CHECK(context_id >= 0 && !env_can_destroy(context_env));
        CHECK(nano_scheduler_step() && context_runs == 1 && context_args == 1);
        if (context_mode == 2) {
            CHECK(env_can_destroy(context_env) && context_results == 1);
        } else {
            CHECK(!env_can_destroy(context_env));
            Value out = integer(89), before = out;
            context_fail_clone = true;
            CHECK(!nano_coro_result_copy(context_id, &out) && !memcmp(&out, &before, sizeof out));
            context_fail_clone = false;
            CHECK(nano_coro_result_copy(context_id, &out));
            Value *fields = out.as.struct_val->field_values;
            CHECK(!strcmp(fields[0].as.function_val.function_name, "held"));
            CHECK(fields[1].as.union_val == &borrowed_union && fields[2].as.array_val == &borrowed_array);
            env_discard_value_snapshot(out);
        }
        CHECK(nano_coro_release(context_id) && env_can_destroy(context_env));
        CHECK(!nano_coro_release(context_id));
    }
    context_args = 0;
    context_id = lifetime_context_spawn(context_env, TYPE_STRUCT, context_callback,
        context_env, context_arg_drop, context_result_drop, context_clone);
    CHECK(context_id >= 0 && nano_coro_cancel(context_id) && context_args == 1 && env_can_destroy(context_env));
    CHECK(nano_coro_release(context_id));
    const Type declared[] = {TYPE_INT, TYPE_U8, TYPE_ENUM, TYPE_BOOL, TYPE_FLOAT, TYPE_VOID,
        TYPE_OPAQUE, TYPE_UNKNOWN, TYPE_STRUCT, TYPE_BOOL};
    for (size_t i = 0; i < sizeof declared / sizeof *declared; ++i) {
        context_scalar_value = i == 3 ? create_bool(true) : i == 4 ? create_float(1.5) :
            i == 5 ? create_void() : integer(7);
        context_id = lifetime_context_spawn(context_env, declared[i], context_scalar_callback,
            context_env, context_arg_drop, context_result_drop, context_clone);
        CHECK(context_id >= 0 && nano_scheduler_step() && nano_coro_is_done(context_id));
        CHECK(env_can_destroy(context_env) == (i < 6));
        CHECK(nano_coro_release(context_id) && env_can_destroy(context_env));
    }
    env_discard_value_snapshot(context_graph);
    char *union_names[] = {"number"}; Value union_values[] = {integer(51)};
    CHECK(env_create_union(context_env,"TaskResult",0,"Held",union_names,union_values,1,&context_graph));
    context_mode=0; context_args=context_runs=context_results=0;
    context_id=lifetime_context_spawn(context_env,TYPE_UNION,context_callback,
        context_env,context_arg_drop,context_result_drop,context_clone);
    CHECK(context_id>=0 && nano_scheduler_step() && nano_coro_is_done(context_id));
    CHECK(context_args==1 && !env_can_destroy(context_env));
    Value union_copy=create_void(); CHECK(nano_coro_result_copy(context_id,&union_copy));
    CHECK(env_union_result_borrowed(context_env,union_copy));
    CHECK(union_copy.as.union_val==context_graph.as.union_val);
    CHECK(union_copy.as.union_val->field_values[0].as.int_val==51);
    env_discard_value_snapshot(union_copy); /* Union remains borrowed. */
    CHECK(nano_coro_release(context_id) && env_can_destroy(context_env));
    CHECK(context_graph.as.union_val->field_values[0].as.int_val==51);
    free_environment(context_env); context_env = NULL;
}
static void union_root_attempt(size_t at, bool once, size_t *count) {
    Environment *env = create_environment(); CHECK(env);
    Value child = create_void();
    CHECK(env_create_union(env, "Choice", 0, "Empty", NULL, NULL, 0, &child));
    struct EnvUnionRoot *before = env->union_roots;
    char payload[] = "retained";
    char *record_names[] = {"text"}; Value record_values[] = {text_value(payload)};
    StructValue record = {.struct_name="Item", .field_count=1,
        .field_names=record_names, .field_values=record_values};
    Value tuple_values[] = {record_value(&record), child};
    TupleValue tuple = {.element_count=2, .elements=tuple_values};
    Array borrowed = {.element_type=VAL_INT};
    Value array = create_void(); array.type=VAL_ARRAY; array.as.array_val=&borrowed;
    Value callable = create_void(); callable.type=VAL_FUNCTION;
    callable.as.function_val.function_name="borrowed";
    char *names[] = {"text", "record", "tuple", "child", "alias", "array", "callback"};
    Value values[] = {text_value(payload),record_value(&record),tuple_value(&tuple),child,child,array,callable};
    Value out=integer(991), sentinel=out;
    begin(at,once);
    bool ok=env_create_union(env,"Parent",1,"Held",names,values,7,&out);
    *count=attempts; end();
    if (at!=SIZE_MAX) {
        CHECK(!ok && failures>0 && !memcmp(&out,&sentinel,sizeof out));
        CHECK(env->union_roots==before && live==0);
        CHECK(env_create_union(env,"Parent",1,"Held",names,values,7,&out));
    } else CHECK(ok && failures==0);
    CHECK(env_union_result_borrowed(env,out) && env_union_result_borrowed(env,child));
    CHECK(!env_record_result_borrowed(env,out));
    Value *fields=out.as.union_val->field_values;
    payload[0]='X';
    CHECK(!strcmp(fields[0].as.string_val,"retained"));
    CHECK(!strcmp(fields[1].as.struct_val->field_values[0].as.string_val,"retained"));
    CHECK(!strcmp(fields[2].as.tuple_val->elements[0].as.struct_val->field_values[0].as.string_val,"retained"));
    CHECK(fields[3].as.union_val==child.as.union_val && fields[4].as.union_val==child.as.union_val);
    CHECK(fields[5].as.array_val==&borrowed && fields[6].as.function_val.function_name==callable.as.function_val.function_name);
    env_define_var(env,"first",TYPE_UNION,false,out);
    env_define_var(env,"second",TYPE_UNION,false,out);
    free_environment(env); CHECK(live==0);
}
static void union_root_controls(void) {
    size_t count; union_root_attempt(SIZE_MAX,false,&count); CHECK(count>10);
    for(int once=0;once<2;++once) for(size_t i=0;i<count;++i) {
        size_t ignored; union_root_attempt(i,once!=0,&ignored);
        union_root_attempt(SIZE_MAX,false,&ignored);
    }
    Environment *env=create_environment(); CHECK(env);
    Value raw=create_union("Raw",0,"Empty",NULL,NULL,0);
    CHECK(!env_union_result_borrowed(env,raw));
    Value sentinel=integer(77), out=sentinel;
    CHECK(!env_create_union(env,"Bad",0,"Bad",NULL,NULL,-1,&out));
    CHECK(!memcmp(&out,&sentinel,sizeof out) && !env->union_roots);
    free_environment(env);
    CHECK(!strcmp(raw.as.union_val->union_name,"Raw"));
    free(raw.as.union_val->union_name); free(raw.as.union_val->variant_name); free(raw.as.union_val);
}
int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "cache-init")) {
        cache_init_control(!strcmp(argv[2], "all") ? SIZE_MAX : (size_t)strtoul(argv[2], NULL, 10), atoi(argv[3]) != 0);
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "cache-registration")) {
        cache_registration_control(argv[2]); return 0;
    }
    CHECK(argc == 1);
    record_names_controls();
    union_root_controls(); function_index_controls(); nominal_import_controls(); string_binding_ownership_controls(); evaluator_symbol_pop_controls(); graph_controls(); signature_controls(); list_controls(); publication_controls(); index_allocation_controls(); index_identity_controls(); index_collision_and_limits(); provider_controls(); scheduler_controls(); contextual_task_controls(); task_allocation_controls(); borrowed_staging_controls();
    CHECK(live == 0 && !observing);
    printf("I passed %zu checked ownership assertions.\n", checks);
    return 0;
}
