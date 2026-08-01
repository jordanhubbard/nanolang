/*
 * test_wasm_profiler.c — --profile-runtime backend support policy
 *
 * PR #51 shipped runtime flamegraph profiling for the native backend only: the
 * counters and the .nano.prof writer are emitted into the C that the transpiler
 * generates. Despite the PR title, no WASM profiler was ever part of it, and
 * the WASM backend returns from compile_file() long before that code generation
 * runs. These tests pin that scoping so --profile-runtime cannot silently
 * regress back to accepting a backend it produces no profile for.
 *
 * Tests:
 *   1.  backend_name: --target/--llvm resolve to the right backend label
 *   2.  native_is_supported: the one backend that can emit .nano.prof
 *   3.  wasm_is_unsupported: the backend this file is named for
 *   4.  alternate_backends_unsupported: ptx, opencl, c, riscv, llvm
 *   5.  flamegraph_emitter_present: native path still emits the writer
 *   6.  flamegraph_explicit_path: --profile-runtime-output is honoured
 */

#include "../src/stdlib_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* g_argc/g_argv are referenced by runtime/cli.c; define them here */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define TEST(name) static void test_##name(void)
#define RUN(name) do { test_##name(); printf("  %-50s PASS\n", #name "..."); g_pass++; } while(0)
#define ASSERT(cond) do { if (!(cond)) { \
    printf("  FAIL: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); g_fail++; return; } } while(0)

/* ── Helper: render the flamegraph emitter to a string ──────────────────── */

static char *emit_flamegraph(const char *path) {
    StringBuilder *sb = sb_create();
    if (!sb) return NULL;
    generate_flamegraph_profiling_system(sb, path);
    char *out = strdup(sb->buffer ? sb->buffer : "");
    free(sb->buffer);
    free(sb);
    return out;
}

/* ── Backend resolution ─────────────────────────────────────────────────── */

TEST(backend_name) {
    /* No --target and no --llvm is the native path. */
    ASSERT(strcmp(profile_runtime_backend_name(NULL, false), "native") == 0);
    ASSERT(strcmp(profile_runtime_backend_name("", false), "native") == 0);
    /* An explicit --target always wins. */
    ASSERT(strcmp(profile_runtime_backend_name("wasm", false), "wasm") == 0);
    ASSERT(strcmp(profile_runtime_backend_name("riscv", false), "riscv") == 0);
    /* --llvm short-circuits the transpiler, so it is its own backend. */
    ASSERT(strcmp(profile_runtime_backend_name(NULL, true), "llvm") == 0);
    /* --target with --llvm: the target branch runs first in compile_file(). */
    ASSERT(strcmp(profile_runtime_backend_name("wasm", true), "wasm") == 0);
}

/* ── Support policy ─────────────────────────────────────────────────────── */

TEST(native_is_supported) {
    ASSERT(profile_runtime_backend_supported("native"));
    /* A NULL/empty --target is the native default, not an unknown backend. */
    ASSERT(profile_runtime_backend_supported(NULL));
    ASSERT(profile_runtime_backend_supported(""));
}

TEST(wasm_is_unsupported) {
    ASSERT(!profile_runtime_backend_supported("wasm"));
    ASSERT(!profile_runtime_backend_supported(
               profile_runtime_backend_name("wasm", false)));
}

TEST(alternate_backends_unsupported) {
    static const char *backends[] = { "ptx", "opencl", "c", "riscv", "llvm" };
    for (size_t i = 0; i < sizeof(backends) / sizeof(backends[0]); i++) {
        ASSERT(!profile_runtime_backend_supported(backends[i]));
    }
}

/* ── The native emitter that makes "supported" true ─────────────────────── */

TEST(flamegraph_emitter_present) {
    char *code = emit_flamegraph(NULL);
    ASSERT(code != NULL);
    /* The atexit hook transpiler.c registers must exist. */
    ASSERT(strstr(code, "_nl_prof_flamegraph_report") != NULL);
    /* Collapsed-stack format is "<name> <count>", one line per function. */
    ASSERT(strstr(code, "fprintf(f, \"%s %lld") != NULL);
    /* With no explicit path the runtime derives <argv[0]>.nano.prof. */
    ASSERT(strstr(code, "_nl_flamegraph_path = NULL") != NULL);
    ASSERT(strstr(code, ".nano.prof") != NULL);
    free(code);
}

TEST(flamegraph_explicit_path) {
    char *code = emit_flamegraph("/tmp/explicit_profile.nano.prof");
    ASSERT(code != NULL);
    ASSERT(strstr(code, "\"/tmp/explicit_profile.nano.prof\"") != NULL);
    ASSERT(strstr(code, "_nl_flamegraph_path = NULL") == NULL);
    free(code);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[wasm_profiler] --profile-runtime backend support policy tests...\n\n");
    RUN(backend_name);
    RUN(native_is_supported);
    RUN(wasm_is_unsupported);
    RUN(alternate_backends_unsupported);
    RUN(flamegraph_emitter_present);
    RUN(flamegraph_explicit_path);
    printf("\n");
    if (g_fail == 0) { printf("All %d tests passed.\n", g_pass); return 0; }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
