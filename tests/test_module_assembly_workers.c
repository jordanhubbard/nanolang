/* I observe my real worker processes and inject only spawn/wait failures. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include <spawn.h>
#include <sys/wait.h>
#include <errno.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static int spawn_calls, active, maximum, fail_spawn = -1;
static bool interrupt_once, interrupted;
static pid_t observed_children[64];
static size_t observed_count;
static int observed_spawn(pid_t *, const char *, const posix_spawn_file_actions_t *,
                          const posix_spawnattr_t *, char *const [], char *const []);
static pid_t observed_wait(pid_t, int *, int);
#define posix_spawn observed_spawn
#define waitpid observed_wait
#include "../src/module_builder.c"
#undef posix_spawn
#undef waitpid

static int observed_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
                           const posix_spawnattr_t *attributes, char *const args[], char *const env[]) {
    if (spawn_calls++ == fail_spawn) return EAGAIN;
    int status = posix_spawn(pid, path, actions, attributes, args, env);
    if (!status) {
        assert(observed_count < sizeof observed_children / sizeof *observed_children);
        observed_children[observed_count++] = *pid;
        if (++active > maximum) maximum = active;
    }
    return status;
}

static pid_t observed_wait(pid_t pid, int *status, int options) {
    assert(pid > 0 && options == 0);
    if (interrupt_once && !interrupted) { interrupted = true; errno = EINTR; return -1; }
    pid_t result = waitpid(pid, status, options);
    if (result == pid) --active;
    return result;
}

int main(int argc, char **argv) {
    assert(argc == 10);
    unsigned limit = (unsigned)atoi(argv[3]);
    size_t ordinary = (size_t)atoi(argv[4]), shared = (size_t)atoi(argv[5]);
    fail_spawn = atoi(argv[6]); interrupt_once = atoi(argv[7]) != 0;
    bool non_c = atoi(argv[8]) != 0;
    assert(ordinary <= 32 && shared <= 32);
    char names[2][32][64];
    char *sources[2][32];
    for (size_t group = 0; group < 2; ++group) {
        size_t count = group ? shared : ordinary;
        for (size_t i = 0; i < count; ++i) {
            snprintf(names[group][i], sizeof names[group][i], "unit_%zu_%zu.%s",
                     group, i, non_c ? (i % 2 ? "S" : "s") : "c");
            sources[group][i] = names[group][i];
        }
    }
    ModuleBuildMetadata meta = {0};
    meta.c_sources = sources[0]; meta.c_sources_count = ordinary;
    meta.shared_c_sources = sources[1]; meta.shared_c_sources_count = shared;
    pid_t unrelated;
    char *args[] = {"sh", "-c", "exit 23", NULL};
    extern char **environ;
    assert(posix_spawn(&unrelated, "/bin/sh", NULL, NULL, args, environ) == 0);
    bool ok = module_prepare_assemblies(&meta, argv[2], argv[1], argv[9], limit);
    int status;
    assert(waitpid(unrelated, &status, 0) == unrelated && WIFEXITED(status) && WEXITSTATUS(status) == 23);
    assert(active == 0);
    for (size_t i = 0; i < observed_count; ++i) {
        assert(waitpid(observed_children[i], &status, WNOHANG) == -1 && errno == ECHILD);
    }
    printf("{\"ok\":%d,\"maximum\":%d,\"spawn_calls\":%d,\"interrupted\":%d,\"default_limit\":%u}\n",
           ok, maximum, spawn_calls, interrupted, module_assembly_workers());
    return ok ? 0 : 1;
}
