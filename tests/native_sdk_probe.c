#define _GNU_SOURCE 1
#include "runtime/module_build_dir.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>

static void report(const char *name, const char *value) {
    printf("%s ", name);
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) printf("%02x", *p);
    putchar('\n');
}
static char callback_directory[4096];
static pid_t callback_owner;
static unsigned callback_count;
static void callback(void) {
    assert(getpid() == callback_owner);
    struct stat st;
    assert(!stat(callback_directory, &st) && S_ISDIR(st.st_mode));
    ++callback_count;
    puts("CALLBACK before private cleanup");
}
static void other_callback(void) { abort(); }
static void after_cleanup(void) {
    if (getpid() != callback_owner) return;
    assert(callback_count == 1);
    assert(access(callback_directory, F_OK) == -1 && errno == ENOENT);
    puts("PASS callback and private cleanup order");
}
int main(int argc, char **argv) {
    assert(argc == 4);
    char path[4096], before[sizeof(path)];
    memset(path, 0xa5, sizeof(path)); memcpy(before, path, sizeof(path));
    bool installed = false;
    NanoSdkStatus expected = (NanoSdkStatus)strtol(argv[2], NULL, 10);
    size_t capacity = !strcmp(argv[1], "short") ? 1 : sizeof(path);
    NanoSdkStatus status = nano_native_sdk_root(path, capacity, &installed);
    printf("STATUS %d\n", (int)status);
    assert(status == expected);
    if (status != NANO_SDK_OK) {
        assert(!memcmp(path, before, sizeof(path)) && !installed);
        puts("PASS unchanged root outputs");
        return 0;
    }
    assert(installed == (strcmp(argv[3], "installed") == 0));
    report("ROOT", path);
    if (!strcmp(argv[1], "callback")) {
        callback_owner = getpid();
        assert(atexit(after_cleanup) == 0);
        assert(!nano_native_register_loader_shutdown(NULL));
        assert(nano_native_register_loader_shutdown(callback));
        assert(nano_native_register_loader_shutdown(callback));
        assert(!nano_native_register_loader_shutdown(other_callback));
        assert(nano_native_sdk_prepare() == NANO_SDK_OK);
        char include[4096], expected_include[4096];
        assert(nano_native_prepared_include(include, sizeof(include)));
        int n = snprintf(expected_include, sizeof(expected_include), "%s/src", path);
        assert(n > 0 && (size_t)n < sizeof(expected_include));
        assert(!strcmp(include, installed ? expected_include : ""));
        char sentinel = 'q';
        assert(!nano_native_prepared_include(&sentinel, 0) && sentinel == 'q');
        assert(nano_native_module_objects_dir(callback_directory, sizeof(callback_directory)) == NANO_SDK_OK);
        pid_t child = fork(); assert(child >= 0);
        if (!child) {
            assert(!nano_native_register_loader_shutdown(callback));
            exit(0);
        }
        int child_status;
        assert(waitpid(child, &child_status, 0) == child);
        assert(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);
        assert(access(callback_directory, F_OK) == 0 && callback_count == 0);
    }
    if (!strcmp(argv[1], "objects") || !strcmp(argv[1], "lists")) {
        assert(nano_native_sdk_prepare() == NANO_SDK_OK);
        char first[4096], second[4096];
        assert(nano_native_module_objects_dir(first, sizeof(first)) == NANO_SDK_OK);
        assert(nano_native_module_objects_dir(second, sizeof(second)) == NANO_SDK_OK);
        assert(strcmp(first, second));
        struct stat st;
        assert(!stat(first, &st) && S_ISDIR(st.st_mode));
        assert(!stat(second, &st) && S_ISDIR(st.st_mode));
        report("FIRST", first); report("SECOND", second);
        const char *cache = getenv("NANO_BUILD_CACHE");
        report("CACHE", cache ? cache : "");
        if (!strcmp(argv[1], "lists")) {
            assert(!nano_native_generate_list(path, first, "invalid;name", "SdkPair"));
            assert(!nano_native_generate_list(path, first, "SdkPair", "invalid type"));
            assert(nano_native_generate_list(path, first, "SdkPair", "SdkPair"));
            nano_native_retain_private_work();
        }
    }
    puts("PASS native SDK root and private lifetime");
    return 0;
}
