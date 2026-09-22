/* I inspect the real private lock only to synchronize reader/writer controls. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "../../src/runtime/ffi_loader.c"
#include <assert.h>
#include <sys/wait.h>

int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

typedef struct {
    bool writer;
    int ready[2];
    int release[2];
} LockWorker;
static void *hold_lock(void *arg) {
    LockWorker *worker = arg;
    assert(ffi_registry_lock(worker->writer));
    char byte = 'r';
    assert(write(worker->ready[1], &byte, 1) == 1);
    assert(read(worker->release[0], &byte, 1) == 1);
    ffi_registry_unlock();
    return NULL;
}
static void busy_refusal(bool writer) {
    LockWorker worker = {.writer = writer};
    assert(!pipe(worker.ready) && !pipe(worker.release));
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, hold_lock, &worker));
    char byte;
    assert(read(worker.ready[0], &byte, 1) == 1);
    FfiLoaderFork token = {0};
    assert(!ffi_loader_fork_prepare(&token));
    assert(write(worker.release[1], &byte, 1) == 1);
    assert(!pthread_join(thread, NULL));
    for (int i = 0; i < 2; ++i) {
        close(worker.ready[i]); close(worker.release[i]);
    }
    assert(ffi_loader_fork_prepare(&token));
    assert(!ffi_loader_fork_child(&token));
    assert(ffi_loader_fork_parent(&token));
    assert(!ffi_loader_fork_parent(&token));
}
static void waiting_writer_refusal(void) {
    LockWorker worker = {.writer = true};
    assert(!pipe(worker.ready) && !pipe(worker.release));
    assert(ffi_registry_lock(false));
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, hold_lock, &worker));
    /* My outer alarm bounds this wait; count 2 includes the blocked writer. */
    while (__atomic_load_n(&ffi_admission, __ATOMIC_ACQUIRE) != 2u) sched_yield();
    FfiLoaderFork token;
    assert(!ffi_loader_fork_prepare(&token));
    ffi_registry_unlock();
    char byte;
    assert(read(worker.ready[0], &byte, 1) == 1);
    assert(write(worker.release[1], &byte, 1) == 1);
    assert(!pthread_join(thread, NULL));
    for (int i = 0; i < 2; ++i) {
        close(worker.ready[i]); close(worker.release[i]);
    }
}
static void *wait_admission(void *arg) {
    int *ready = arg;
    char byte = 'w';
    assert(write(ready[1], &byte, 1) == 1);
    assert(ffi_loader_is_initialized());
    return NULL;
}
static void concurrent_reopen(void) {
    FfiLoaderFork token;
    assert(ffi_loader_fork_prepare(&token));
    int ready[2]; assert(!pipe(ready));
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, wait_admission, ready));
    char byte;
    assert(read(ready[0], &byte, 1) == 1);
    assert(ffi_loader_fork_parent(&token));
    assert(!pthread_join(thread, NULL));
    close(ready[0]); close(ready[1]);
}
static void child_status(pid_t child) {
    assert(child > 0);
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
static void exact_symbol(const char *name) {
    const char *(*root)(void) = (const char *(*)(void))
        ffi_loader_resolve_module("nlc_runtime_root", name);
    assert(root && root());
}
static void unprepared_child(void) {
    pid_t child = fork();
    if (!child) {
        FfiLoaderFork invalid = {0};
        assert(!ffi_loader_fork_child(&invalid));
        assert(!ffi_loader_init(false));
        assert(!ffi_loader_is_initialized());
        _exit(0);
    }
    child_status(child);
}
static void prepared_child(const char *inherited, const char *fresh, bool descendant) {
    FfiLoaderFork token;
    assert(ffi_loader_fork_prepare(&token));
    pid_t child = fork();
    if (!child) {
        assert(ffi_loader_fork_child(&token));
        assert(!ffi_loader_fork_child(&token));
        assert(!nano_native_register_loader_shutdown(ffi_loader_shutdown));
        assert(ffi_loader_init(false));
        exact_symbol(inherited);
        assert(ffi_loader_open(fresh, fresh));
        exact_symbol(fresh);
        unprepared_child();
        if (descendant) prepared_child(inherited, fresh, false);
        ffi_loader_shutdown();
        assert(!ffi_loader_is_initialized());
        assert(ffi_loader_init(false));
        assert(ffi_loader_open(fresh, fresh));
        exact_symbol(fresh);
        ffi_loader_shutdown();
        _exit(0);
    }
    assert(ffi_loader_fork_parent(&token));
    child_status(child);
    exact_symbol(inherited);
}
static void conflict_callback(void) { abort(); }
static void fresh_registration(const char *library, bool conflict) {
    if (conflict) assert(nano_native_register_loader_shutdown(conflict_callback));
    FfiLoaderFork token;
    assert(ffi_loader_fork_prepare(&token));
    pid_t child = fork();
    if (!child) {
        assert(ffi_loader_fork_child(&token));
        assert(ffi_loader_init(false) == !conflict);
        if (!conflict) {
            assert(ffi_loader_open(library, library));
            exact_symbol(library);
            ffi_loader_shutdown();
        }
        _exit(0);
    }
    assert(ffi_loader_fork_parent(&token));
    child_status(child);
}
int main(int argc, char **argv) {
    assert(argc == 3 || argc == 4);
    alarm(15);
    if (argc == 4) {
        assert(!strcmp(argv[3], "fresh") || !strcmp(argv[3], "conflict"));
        fresh_registration(argv[1], !strcmp(argv[3], "conflict"));
        puts("I preserved fresh-child SDK registration authority.");
        return 0;
    }
    assert(ffi_loader_init(false));
    assert(ffi_loader_open(argv[1], argv[1]));
    char parent_work[4096];
    assert(nano_native_module_objects_dir(parent_work, sizeof parent_work) == NANO_SDK_OK);
    assert(access(parent_work, F_OK) == 0);
    busy_refusal(false);
    busy_refusal(true);
    waiting_writer_refusal();
    concurrent_reopen();
    unprepared_child();
    prepared_child(argv[1], argv[2], true);
    assert(access(parent_work, F_OK) == 0);
    assert(!ffi_loader_find(argv[2]));
    assert(nano_native_register_loader_shutdown(ffi_loader_shutdown));
    ffi_loader_shutdown();
    alarm(0);
    puts("I checked prepared loader admission and process ownership.");
    return 0;
}
