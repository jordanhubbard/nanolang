/* test_wasm_profiler.c — stub unit test for WASM profiler
 *
 * Full tests will live in feat/wasm-runtime-profiler once merged.
 * This stub ensures the build completes without the wasm profiler branch.
 */
#include <stdio.h>

/* g_argc/g_argv are referenced by runtime/cli.c; define them here */
int g_argc = 0;
char **g_argv = NULL;

int main(void) {
    printf("wasm_profiler tests: stub (feat/wasm-runtime-profiler pending)\n");
    return 0;
}
