/* I test checked graph ownership separately from fatal legacy evaluator allocation. */
#define _POSIX_C_SOURCE 200809L
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
    char *names[] = {"text", "number"};
    StructValue record = {.struct_name = "Item", .field_names = names, .field_values = fields, .field_count = 2};
    Value nested[] = {record_value(&record), text_value(payload)};
    TupleValue tuple = {.elements = nested, .element_count = 2};
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
        CHECK(copied->field_values[1].as.int_val == 37);
        CHECK(!strcmp(output.as.tuple_val->elements[1].as.string_val, "kept"));
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
    for (int once = 0; once < 2; ++once) for (size_t index = 0; index < 2; ++index) {
        Value out = integer(67); begin(index, once != 0);
        CHECK(!env_value_snapshot(env, text_value("kept"), &out)); end();
        CHECK(failures && out.type == VAL_INT && out.as.int_val == 67 && env->record_results == NULL && !live);
    }
    TupleValue tuple = {0}; Value owned;
    CHECK(env_clone_value_snapshot(tuple_value(&tuple), &owned));
    for (int once = 0; once < 2; ++once) {
        begin(0, once != 0); CHECK(!env_retire_value(env, owned)); end();
        CHECK(failures == 1 && !live && env->record_results == NULL && owned.as.tuple_val->element_count == 0);
    }
    CHECK(env_retire_value(env, owned)); CHECK(!env_retire_value(env, owned));
    free_environment(env);
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

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "cache-init")) {
        cache_init_control(!strcmp(argv[2], "all") ? SIZE_MAX : (size_t)strtoul(argv[2], NULL, 10), atoi(argv[3]) != 0);
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "cache-registration")) {
        cache_registration_control(argv[2]); return 0;
    }
    CHECK(argc == 1);
    graph_controls(); signature_controls(); list_controls(); publication_controls(); provider_controls(); scheduler_controls(); task_allocation_controls(); borrowed_staging_controls();
    CHECK(live == 0 && !observing);
    printf("I passed %zu checked ownership assertions.\n", checks);
    return 0;
}
