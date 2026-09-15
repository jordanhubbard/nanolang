#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int reject_allocation;
static int reject_string_copy;
static int test_aligned_allocate(void **memory, size_t alignment, size_t bytes) {
    if (reject_allocation) return ENOMEM;
    return posix_memalign(memory, alignment, bytes);
}
static char *test_string_copy(const char *value) {
    if (reject_string_copy) return NULL;
    return strdup(value);
}
#define posix_memalign test_aligned_allocate
#define strdup test_string_copy
#include "../src/runtime/dyn_array.c"
#undef posix_memalign
#undef strdup
#undef NDEBUG
#include <assert.h>

static void failure_push(void) {
    DynArray *a = dyn_array_new(ELEM_INT);
    for (int i = 0; i < 8; ++i) dyn_array_push_int(a, i);
    reject_allocation = 1;
    dyn_array_push_int(a, 9);
}

static void failure_reserve(void) {
    DynArray *a = dyn_array_new(ELEM_INT);
    reject_allocation = 1;
    dyn_array_reserve(a, 100);
}

static void failure_struct(void) {
    DynArray *a = dyn_array_new(ELEM_STRUCT);
    int value = 42;
    reject_allocation = 1;
    dyn_array_push_struct(a, &value, sizeof value);
}

static void failure_string_copy(void) {
    DynArray *a = dyn_array_new(ELEM_STRING);
    reject_string_copy = 1;
    dyn_array_push_string_copy(a, "retained");
}

static void overflow_growth(void) {
    DynArray *a = dyn_array_new(ELEM_INT);
    a->capacity = a->length = INT64_MAX;
    dyn_array_push_int(a, 1);
}

static void overflow_reserve(void) {
    DynArray *a = dyn_array_new(ELEM_INT);
    dyn_array_reserve(a, INT64_MAX);
}

static void overflow_struct(void) {
    DynArray *a = dyn_array_new(ELEM_STRUCT);
    int value = 42;
    dyn_array_push_struct(a, &value, SIZE_MAX);
}

static void empty_struct(void) {
    DynArray *a = dyn_array_new(ELEM_STRUCT);
    int value = 42;
    dyn_array_push_struct(a, &value, 0);
}

static void excessive_struct_width(void) {
    DynArray *a = dyn_array_new(ELEM_STRUCT);
    char value[256] = {0};
    dyn_array_push_struct(a, value, sizeof value);
}

static void expect_abort(void (*operation)(void)) {
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        struct rlimit core = {0, 0};
        setrlimit(RLIMIT_CORE, &core);
        operation();
        _exit(99);
    }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

int main(void) {
    gc_init();
    size_t baseline = gc_get_stats().num_objects;
    assert(darray_aligned_alloc(SIZE_MAX) == NULL);
    assert(dyn_array_new_with_capacity(ELEM_INT, INT64_MAX) == NULL);
    assert(gc_get_stats().num_objects == baseline);
    reject_allocation = 1;
    assert(dyn_array_new(ELEM_STRING) == NULL);
    assert(gc_get_stats().num_objects == baseline);
    reject_allocation = 0;

    DynArray *a = dyn_array_new(ELEM_INT);
    for (int i = 0; i < 100; ++i) dyn_array_push_int(a, i);
    reject_allocation = 1;
    assert(dyn_array_clone(a) == NULL);
    assert(a->length == 100 && dyn_array_get_int(a, 99) == 99);
    reject_allocation = 0;
    DynArray *copy = dyn_array_clone(a);
    assert(copy && copy->length == 100 && dyn_array_get_int(copy, 99) == 99);
    gc_release(copy);
    gc_release(a);

    struct Record { int64_t key, value; } record = {7, 42};
    a = dyn_array_new(ELEM_STRUCT);
    for (int i = 0; i < 8; ++i) dyn_array_push_struct(a, &record, sizeof record);
    dyn_array_push_struct(a, dyn_array_get_struct(a, 0), sizeof record);
    assert(((struct Record *)dyn_array_get_struct(a, 8))->value == 42);
    gc_release(a);
    a = dyn_array_new(ELEM_STRUCT);
    dyn_array_reserve(a, 40);
    assert(a->data == NULL && a->capacity == 40);
    for (int i = 0; i < 50; ++i) dyn_array_push_struct(a, &record, sizeof record);
    copy = dyn_array_clone(a);
    assert(copy && copy->length == 50 && copy->elem_size == sizeof record);
    assert(((struct Record *)dyn_array_get_struct(copy, 49))->value == 42);
    ((struct Record *)dyn_array_get_struct(copy, 49))->value = 9;
    assert(((struct Record *)dyn_array_get_struct(a, 49))->value == 42);
    gc_release(copy);
    reject_allocation = 1;
    assert(dyn_array_clone(a) == NULL);
    reject_allocation = 0;
    gc_release(a);
    a = dyn_array_new(ELEM_STRUCT);
    copy = dyn_array_clone(a);
    assert(copy && copy->length == 0 && copy->data == NULL);
    dyn_array_push_struct(copy, &record, sizeof record);
    gc_release(copy);
    gc_release(a);
    assert(gc_get_stats().num_objects == baseline);

    unsigned char widest[255];
    memset(widest, 0xa5, sizeof widest);
    a = dyn_array_new(ELEM_STRUCT);
    dyn_array_push_struct(a, widest, sizeof widest);
    copy = dyn_array_clone(a);
    assert(copy && copy->elem_size == 255 &&
           memcmp(dyn_array_get_struct(copy, 0), widest, sizeof widest) == 0);
    gc_release(copy);
    gc_release(a);

    expect_abort(failure_push);
    expect_abort(failure_reserve);
    expect_abort(failure_struct);
    expect_abort(failure_string_copy);
    expect_abort(overflow_growth);
    expect_abort(overflow_reserve);
    expect_abort(overflow_struct);
    expect_abort(empty_struct);
    expect_abort(excessive_struct_width);
    gc_shutdown();
    puts("I passed native array allocation boundary tests.");
    return 0;
}
