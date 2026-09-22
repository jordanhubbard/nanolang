/* I qualify actual evaluator collection owners, aliases and fatal rollback. */
#define main nano_full_eval_test_main
#include "test_eval.c"
#undef main
#include <assert.h>
#include <limits.h>
#include "../src/eval/eval_hashmap.h"
#include "struct_ownership_worker.h"

static Environment *collection_env;
static RunCtx partial_context;
static Value caller_array;
static Value caller_record;
static size_t allocation_count, failure_position;
static int failure_mode, failure_seen, published;
static unsigned argument_serial;

void *nano_test_collection_calloc(size_t count, size_t width) {
    ++allocation_count;
    if (failure_position && (allocation_count == failure_position ||
        (failure_mode && allocation_count > failure_position))) {
        failure_seen = 1;
        return NULL;
    }
    return calloc(count, width);
}

static Value text_value(const char *text) {
    Value value = create_void(); value.type = VAL_STRING;
    value.as.string_val = (char *)text; return value;
}

static Value invoke(const char *name, Value *values, int count, const char *map_type) {
    ASTNode call = {0}, args[3] = {{0}};
    ASTNode *pointers[3]; char names[3][48];
    assert(count >= 0 && count <= 3);
    for (int i = 0; i < count; ++i) {
        snprintf(names[i], sizeof names[i], "collection_arg_%u", argument_serial++);
        /* My fixture borrows all supplied values, including aliased string leaves. */
        env_define_var(collection_env, names[i], TYPE_BORROW_SHARED, false, values[i]);
        args[i].type = AST_IDENTIFIER; args[i].as.identifier = names[i]; pointers[i] = &args[i];
    }
    call.type = AST_CALL; call.as.call.name = (char *)name;
    call.as.call.args = pointers; call.as.call.arg_count = count;
    call.as.call.return_struct_type_name = (char *)map_type;
    return repl_eval_node(&call, collection_env);
}

static void cleanup_case(void) {
    if (!collection_env) return;
    assert(caller_array.type == VAL_ARRAY);
    assert(caller_array.as.array_val->length == 2);
    assert(((long long *)caller_array.as.array_val->data)[0] == 17);
    if (failure_seen) assert(!published);
    if (partial_context.env) run_ctx_free(&partial_context);
    else free_environment(collection_env);
    collection_env = NULL;
    /* My caller allocation survives Environment destruction. */
    assert(((long long *)caller_array.as.array_val->data)[1] == 29);
    free(caller_array.as.array_val->data); free(caller_array.as.array_val);
    caller_array = create_void();
    if (caller_record.type == VAL_STRUCT) env_discard_record(caller_record.as.struct_val);
    caller_record = create_void();
}

static size_t collection_case(unsigned kind, size_t position, int mode) {
    allocation_count = 0; failure_position = 0; failure_seen = 0; published = 0;
    if (kind == 5 || kind == 6) {
        const char *source = kind == 5 ?
            "fn next()->string { let extra:array<int> = (array_new 1 0) return \"later\" }\n"
            "fn build()->array<string> { return [\"first\", (next)] }\n"
            "fn main()->int{return 0}\n" :
            "struct CollectionItem { label:string }\n"
            "fn next()->CollectionItem { let extra:array<int> = (array_new 1 0) return CollectionItem { label: \"later\" } }\n"
            "fn build()->array<CollectionItem> { return [CollectionItem { label: \"first\" }, (next)] }\n"
            "fn main()->int{return 0}\n";
        assert(run_ctx_init(&partial_context, source));
        collection_env = partial_context.env;
    } else { collection_env = create_environment(); assert(collection_env); }
    caller_array = create_array(VAL_INT, 2, 2);
    ((long long *)caller_array.as.array_val->data)[0] = 17;
    ((long long *)caller_array.as.array_val->data)[1] = 29;
    caller_record = create_void();
    if (kind == 1) {
        char *fields[] = {"label"}; Value values[] = {text_value("kept")};
        caller_record = create_struct("CollectionItem", fields, values, 1);
    }
    allocation_count = 0; failure_position = position; failure_mode = mode;
    Value borrowed_args[] = {caller_array, create_int(0), create_int(2)};
    Value copied_input = invoke("array_slice", borrowed_args, 3, NULL);
    assert(copied_input.type == VAL_ARRAY && copied_input.as.array_val != caller_array.as.array_val);
    assert(((long long *)copied_input.as.array_val->data)[1] == 29);
    if (kind == 0 || kind == 1) {
        Value args[] = {create_int(2), kind ? caller_record : text_value("kept")};
        Value array = invoke("array_new", args, 2, NULL); assert(array.type == VAL_ARRAY);
        assert(array.as.array_val != caller_array.as.array_val);
        Value replacement = create_void(); replacement.type = kind ? VAL_STRUCT : VAL_STRING;
        if (kind) replacement.as.struct_val = ((StructValue **)array.as.array_val->data)[0];
        else replacement.as.string_val = ((char **)array.as.array_val->data)[0];
        Value update[] = {array, create_int(0), replacement};
        invoke("array_set", update, 3, NULL);
        if (kind) assert(!strcmp(((StructValue **)array.as.array_val->data)[0]->field_values[0].as.string_val, "kept"));
        else assert(!strcmp(((char **)array.as.array_val->data)[0], "kept"));
        Value slice_args[] = {array, create_int(0), create_int(1)};
        Value slice = invoke("array_slice", slice_args, 3, NULL); assert(slice.type == VAL_ARRAY);
        assert(slice.as.array_val != array.as.array_val);
        if (kind) assert(((StructValue **)slice.as.array_val->data)[0] != ((StructValue **)array.as.array_val->data)[0]);
        else assert(((char **)slice.as.array_val->data)[0] != ((char **)array.as.array_val->data)[0]);
        /* A further allocation can fail while populated string/record owners remain live. */
        Value last[] = {create_int(0), create_int(0)};
        Value empty = invoke("array_new", last, 2, NULL); assert(empty.type == VAL_ARRAY && empty.as.array_val->length == 0);
    } else if (kind == 5 || kind == 6) {
        Value built = call_function("build", NULL, 0, collection_env);
        assert(built.type == VAL_ARRAY && built.as.array_val->length == 2);
        if (kind == 5) assert(!strcmp(((char **)built.as.array_val->data)[1], "later"));
        else assert(!strcmp(((StructValue **)built.as.array_val->data)[1]->field_values[0].as.string_val, "later"));
    } else if (kind == 2 || kind == 4) {
        Value map = invoke("map_new", NULL, 0, "HashMap_string_string"); assert(map.type == VAL_INT && map.as.int_val);
        NLHashMapCore *storage = (NLHashMapCore *)map.as.int_val;
        if (kind == 4) {
            storage->size = (int64_t)INT_MAX + 1;
            invoke("map_keys", &map, 1, NULL);
            assert(!"oversized map projection must refuse before reading entries");
        }
        Value entry[] = {map, text_value("key"), text_value("value")};
        invoke("map_put", entry, 3, NULL);
        Value keys = invoke("map_keys", &map, 1, NULL);
        Value values = invoke("map_values", &map, 1, NULL);
        assert(keys.type == VAL_ARRAY && values.type == VAL_ARRAY);
        assert(!strcmp(((char **)keys.as.array_val->data)[0], "key"));
        Value update[] = {keys, create_int(0), text_value("changed")};
        invoke("array_set", update, 3, NULL);
        Value lookup[] = {map, text_value("key")};
        Value present = invoke("map_has", lookup, 2, NULL); assert(present.type == VAL_BOOL && present.as.bool_val);
        invoke("map_free", &map, 1, NULL);
        assert(!strcmp(((char **)values.as.array_val->data)[0], "value"));
        assert(!strcmp(((char **)keys.as.array_val->data)[0], "changed"));
    } else {
        assert(kind == 3);
        Value huge[] = {create_int((long long)INT_MAX + 1), create_int(0)};
        invoke("array_new", huge, 2, NULL);
        assert(!"oversized array must refuse before narrowing");
    }
    published = 1;
    size_t measured = allocation_count;
    cleanup_case(); return measured;
}

static void child_case(unsigned kind, size_t position, int mode, bool refusal) {
    int errors[2]; assert(pipe(errors) == 0);
    pid_t child = spawn_fault_case(errors, 1, (int)kind, position, mode); close(errors[1]);
    const char *message = (kind == 3 || kind == 4) ? "I cannot represent this evaluator array extent.\n" :
        refusal ? "I cannot allocate evaluator collection ownership.\n" : "";
    read_child_diagnostic(errors[0], child, message);
    int status; assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == (refusal ? 1 : 0));
}

static void public_string_alias(void) {
    RunCtx ctx;
    assert(run_ctx_init(&ctx,
        "fn replace(a:array<string>, value:string)->void { (array_set a 0 value) }\n"
        "fn main()->int { return 0 }\n"));
    Value array = create_array(VAL_STRING, 1, 1);
    ((char **)array.as.array_val->data)[0] = strdup("public alias");
    Value args[] = {array, text_value(((char **)array.as.array_val->data)[0])};
    (void)call_function("replace", args, 2, ctx.env);
    assert(!strcmp(((char **)array.as.array_val->data)[0], "public alias"));
    run_ctx_free(&ctx);
    assert(!strcmp(((char **)array.as.array_val->data)[0], "public alias"));
    free(((char **)array.as.array_val->data)[0]);
    free(array.as.array_val->data); free(array.as.array_val);
}

extern int lifetime_enqueue_named(Environment *, const char *);
extern Value lifetime_task_result(Environment *, int, bool);
static void completed_collection_tasks(void) {
    const char *source =
        "fn numbers()->array<int> { return [41, 42] }\n"
        "fn mapping()->HashMap<string,int> { let result:HashMap<string,int> = (map_new) (map_put result \"key\" 73) return result }\n"
        "fn main()->int { return 0 }\n";
    nano_scheduler_init();
    for (int map = 0; map < 2; ++map) {
        RunCtx ctx; assert(run_ctx_init(&ctx, source));
        assert(env_acquire_evaluation_lease(ctx.env)); /* My explicit caller lease. */
        int task = lifetime_enqueue_named(ctx.env, map ? "mapping" : "numbers");
        assert(task >= 0);
        Value result = lifetime_task_result(ctx.env, task, true);
        assert(nano_coro_is_done(task));
        /* DONE dropped the argument bundle; the result owner must still remain. */
        env_release_evaluation_lease(ctx.env);
        assert(!env_can_destroy(ctx.env));
        assert(ctx.env->collection_allocations != NULL);
        if (map) {
            assert(result.type == VAL_INT && result.as.int_val);
            NLHashMapCore *storage = (NLHashMapCore *)result.as.int_val;
            Value key = text_value("key"); bool found = false;
            int64_t slot = eval_hm_find_slot(storage, &key, &found);
            assert(found && slot >= 0 && storage->entries[slot].value.i == 73);
        } else {
            assert(result.type == VAL_ARRAY && result.as.array_val->length == 2);
            assert(((long long *)result.as.array_val->data)[1] == 42);
        }
        assert(nano_coro_release(task));
        assert(env_can_destroy(ctx.env));
        /* Only now may actual Environment collection destruction happen. */
        run_ctx_free(&ctx);
    }
}

int main(int argc, char **argv) {
    fixture_executable = argv[0];
    assert(atexit(cleanup_case) == 0);
    if (argc > 1) {
        unsigned kind, position, mode;
        if (argc != 5 || strcmp(argv[1], "_lookup_fault") ||
            !fixture_number(argv[2], 6, &kind) || !fixture_number(argv[3], 100000, &position) ||
            !fixture_number(argv[4], 1, &mode)) return 2;
        (void)collection_case(kind, position, (int)mode); return 0;
    }
    const unsigned kinds[] = {0, 1, 2, 5, 6};
    for (size_t k = 0; k < sizeof kinds / sizeof *kinds; ++k) {
        unsigned kind = kinds[k];
        size_t count = collection_case(kind, 0, 0); assert(count > 0);
        for (size_t position = 1; position <= count; ++position) {
            child_case(kind, position, 0, true);
            child_case(kind, position, 1, true);
            child_case(kind, 0, 0, false);
        }
        printf("I checked collection kind %u across %zu allocation positions.\n", kind, count);
    }
    child_case(3, 0, 0, true); child_case(4, 0, 0, true);
    public_string_alias();
    completed_collection_tasks();
    /* I retain the actual source callback/partial-literal assertions unchanged. */
    test_eval_handler_return_partial_literal_cleanup();
    test_eval_handler_return_higher_order();
    test_eval_empty_array_aliases();
    test_eval_nested_arrays();
    puts("I passed evaluator collection ownership controls.");
    return 0;
}
