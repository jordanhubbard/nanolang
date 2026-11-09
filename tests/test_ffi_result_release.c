#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "../src/runtime/ffi_loader.h"
#include "../src/runtime/dyn_array.h"
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static void *array_function;
static bool (*array_cleanup)(void *);
static void *resolve_concurrently(void *unused) {
    (void)unused;
    for (int i = 0; i < 500; i++) {
        bool (*release)(void *) = NULL; char error[256];
        assert(ffi_loader_array_release("collections", "nl_hm_keys", array_function, &release, error, sizeof error));
        assert(release == array_cleanup);
    }
    return NULL;
}
int main(int argc, char **argv) {
    assert(argc == 3 && ffi_loader_init(false));
    assert(ffi_loader_open("collections", argv[1]) && ffi_loader_open("origin", argv[2]));
    void (*initialize)(void) = (void (*)(void))ffi_loader_resolve_module("gc_init", "collections");
    void (*shutdown)(void) = (void (*)(void))ffi_loader_resolve_module("gc_shutdown", "collections");
    assert(initialize && shutdown); initialize();
    const char *exports[] = {"nl_hm_keys", "nl_hm_values", "nl_set_values"};
    for (int i = 0; i < 3; i++) {
        void *function = ffi_loader_resolve_module(exports[i], "collections");
        bool (*release)(void *) = NULL; char error[256]; assert(function);
        assert(ffi_loader_array_release("collections", exports[i], function, &release, error, sizeof error));
        assert(release);
        DynArray *result = ((DynArray *(*)(void *))function)(NULL);
        assert(result && result->length == 0 && release(result) && !release(result));
        if (!i) { array_function = function; array_cleanup = release; }
    }
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) assert(!pthread_create(&threads[i], NULL, resolve_concurrently, NULL));
    for (int i = 0; i < 4; i++) assert(!pthread_join(threads[i], NULL));
    bool (*array_release)(void *) = array_cleanup;
    char error[256];
    void *function = ffi_loader_resolve_module("foreign_array", "origin"); assert(function);
    assert(!ffi_loader_array_release("origin", "foreign_array", function, &array_release, error, sizeof error));
    assert(!array_release && strstr(error, "own image"));
    function = ffi_loader_resolve_module("borrowed_array", "origin"); assert(function);
    assert(ffi_loader_array_release("origin", "borrowed_array", function, &array_release, error, sizeof error));
    assert(!array_release);
    assert(!ffi_loader_array_release("missing", "borrowed_array", function, &array_release, error, sizeof error));
    assert(!ffi_loader_array_release(NULL, "borrowed_array", function, &array_release, error, sizeof error));
    assert(!ffi_loader_array_release("origin", NULL, function, &array_release, error, sizeof error));
    assert(!ffi_loader_array_release("origin", "borrowed_array", NULL, &array_release, error, sizeof error));
    assert(!ffi_loader_array_release("origin", "borrowed_array", function, NULL, error, sizeof error));

    void (*string_release)(const char *) = NULL;
    function = ffi_loader_resolve_module("owned_string", "origin"); assert(function);
    assert(ffi_loader_string_release("origin", "owned_string", function, &string_release, error, sizeof error));
    assert(string_release);
    const char *text = ((const char *(*)(void))function)(); assert(text && !strcmp(text, "copied")); string_release(text);
    function = ffi_loader_resolve_module("borrowed_string", "origin"); assert(function);
    assert(ffi_loader_string_release("origin", "borrowed_string", function, &string_release, error, sizeof error));
    assert(!string_release);
    function = ffi_loader_resolve_module("foreign_string", "origin"); assert(function);
    assert(!ffi_loader_string_release("origin", "foreign_string", function, &string_release, error, sizeof error));
    assert(!string_release && strstr(error, "own image"));
    shutdown(); ffi_loader_shutdown();
    return 0;
}
