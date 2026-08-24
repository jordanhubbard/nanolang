#define _POSIX_C_SOURCE 200809L
#include "../src/runtime/module_build_dir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", (msg)); \
        g_fail++; \
        return; \
    } \
} while (0)

static void test_default_in_tree_build_dir(void) {
    unsetenv("NANO_BUILD_CACHE");
    char dest[256];
    ASSERT(nano_module_build_dir("std/datetime", dest, sizeof(dest)), "default path");
    ASSERT(strcmp(dest, "std/datetime/.build") == 0, "in-tree .build next to module");
    g_pass++;
    printf("  default in-tree .build                               PASS\n");
}

static void test_cache_env_sanitizes_slashes(void) {
    setenv("NANO_BUILD_CACHE", "/tmp/nano_module_cache", 1);
    char dest[256];
    ASSERT(nano_module_build_dir("std/datetime", dest, sizeof(dest)), "cache path");
    ASSERT(strcmp(dest, "/tmp/nano_module_cache/std_datetime") == 0,
           "NANO_BUILD_CACHE uses sanitized module dir");
    unsetenv("NANO_BUILD_CACHE");
    g_pass++;
    printf("  NANO_BUILD_CACHE sanitizes slashes                   PASS\n");
}

static void test_strips_dot_slash_prefix(void) {
    setenv("NANO_BUILD_CACHE", "/tmp/nano_module_cache", 1);
    char dest[256];
    ASSERT(nano_module_build_dir("./modules/sdl", dest, sizeof(dest)), "dot-slash");
    ASSERT(strcmp(dest, "/tmp/nano_module_cache/modules_sdl") == 0,
           "leading ./ stripped before sanitize");
    unsetenv("NANO_BUILD_CACHE");
    g_pass++;
    printf("  leading ./ stripped                                  PASS\n");
}

int main(void) {
    printf("test_module_build_dir\n");
    test_default_in_tree_build_dir();
    test_cache_env_sanitizes_slashes();
    test_strips_dot_slash_prefix();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
