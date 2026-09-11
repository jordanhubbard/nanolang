/* I intercept the existing fail-fast boundary to inspect pre-exit ownership. */
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static jmp_buf failure;
static bool reject_copy;
static bool expect_exit;
static char *copies[32];
static unsigned live_copies;

static char *test_strdup(const char *value) {
    if (reject_copy) return NULL;
    char *copy = strdup(value);
    assert(copy);
    for (unsigned i = 0; i < 32; i++) {
        if (!copies[i]) {
            copies[i] = copy;
            live_copies++;
            return copy;
        }
    }
    abort();
}

static void test_free(void *pointer) {
    for (unsigned i = 0; i < 32; i++) {
        if (pointer && pointer == copies[i]) {
            copies[i] = NULL;
            live_copies--;
            break;
        }
    }
    free(pointer);
}

static void test_exit(int status) {
    assert(expect_exit && status == 1);
    longjmp(failure, status);
}

#define strdup test_strdup
#define free test_free
#define exit test_exit
#include "../src/runtime/list_string.c"
#undef strdup
#undef free
#undef exit

int main(void) {
    for (int operation = 0; operation < 3; operation++) {
        List_string *xs = list_string_new();
        list_string_push(xs, "alpha");
        list_string_push(xs, "beta");
        list_string_push(xs, "gamma");
        char *original[] = {xs->data[0], xs->data[1], xs->data[2]};
        reject_copy = expect_exit = true;
        if (setjmp(failure) == 0) {
            if (operation == 0) list_string_push(xs, original[0]);
            if (operation == 1) list_string_insert(xs, 1, original[0]);
            if (operation == 2) list_string_set(xs, 0, original[0] + 1);
            abort();
        }
        reject_copy = expect_exit = false;
        assert(xs->length == 3 && live_copies == 3);
        for (int i = 0; i < 3; i++) assert(xs->data[i] == original[i]);
        assert(strcmp(xs->data[0], "alpha") == 0);
        assert(strcmp(xs->data[1], "beta") == 0);
        assert(strcmp(xs->data[2], "gamma") == 0);
        list_string_set(xs, 0, xs->data[0] + 1);
        assert(strcmp(xs->data[0], "lpha") == 0 && live_copies == 3);
        list_string_set(xs, 0, xs->data[0]);
        assert(strcmp(xs->data[0], "lpha") == 0 && live_copies == 3);
        list_string_insert(xs, 1, xs->data[0]);
        list_string_push(xs, xs->data[1]);
        assert(xs->length == 5 && live_copies == 5);
        test_free(list_string_pop(xs));
        test_free(list_string_remove(xs, 1));
        list_string_free(xs);
        assert(live_copies == 0);
    }
    return 0;
}
