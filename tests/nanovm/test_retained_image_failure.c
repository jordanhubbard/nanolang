#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

static bool fail_allocation, fail_open;
static int fail_string_copy;
static char *image_strdup(const char *value) {
    if (fail_string_copy > 0 && --fail_string_copy == 0) return NULL;
    return strdup(value);
}
static void *image_malloc(size_t size) {
    return fail_allocation ? NULL : malloc(size);
}
static void *image_dlopen(const char *path, int flags) {
    return fail_open ? NULL : dlopen(path, flags);
}
#define malloc image_malloc
#define dlopen image_dlopen
#define strdup image_strdup
#include "../../src/runtime/ffi_loader.c"
#undef strdup
#undef dlopen
#undef malloc

int main(void) {
    const char *path = "obj/ffi_callback_fixture.so";
    for (int copy = 1; copy <= 2; copy++) {
        fail_string_copy = copy;
        assert(!ffi_loader_open("fixture", path));
        assert(module_count == 0);
        assert(!ffi_loader_find("fixture"));
    }
    assert(ffi_loader_open("fixture", path));
    fail_allocation = true;
    assert(!ffi_loader_resolve_retained("retained_wait", "fixture"));
    assert(!retained_images);
    assert(ffi_loader_resolve_module("retained_wait", "fixture"));
    fail_allocation = false;
    fail_open = true;
    assert(!ffi_loader_resolve_retained("retained_wait", "fixture"));
    assert(!retained_images);
    fail_open = false;
    void *symbol = ffi_loader_resolve_retained("retained_wait", "fixture");
    assert(symbol && retained_images && !retained_images->next);
    RetainedImage *image = retained_images;
    fail_allocation = true;
    assert(ffi_loader_resolve_retained("retained_wait", "fixture") == symbol);
    assert(!ffi_loader_resolve_retained("missing_symbol", "fixture"));
    assert(!ffi_loader_resolve_retained("retained_wait", "missing_module"));
    ffi_loader_shutdown();
    assert(ffi_loader_open("new_alias", path));
    assert(ffi_loader_resolve_retained("retained_wait", "new_alias") == symbol);
    assert(retained_images == image && !image->next);
    ffi_loader_shutdown();
    int64_t (*wait_fn)(void *) = (int64_t (*)(void *))symbol;
    assert(wait_fn(NULL) == -1);
    puts("I passed retained-image allocation/open failure, recovery, and shutdown reuse checks.");
    return 0;
}
