/* I exercise the actual launcher with bounded startup fault injection. */
#include <assert.h>
#include <spawn.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../../src/nanovm/vm_ffi.h"
#include "../../src/nanovm/cop_protocol.h"
static int spawn_mode;
static const char *self_path;
static int fixture_spawn(pid_t *, const char *, const posix_spawn_file_actions_t *,
                         const posix_spawnattr_t *, char *const [], char *const []);
#define posix_spawn fixture_spawn
#include "../../src/nanovm/vm_ffi.c"
#undef posix_spawn
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static int fixture_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
                         const posix_spawnattr_t *attributes, char *const argv[], char *const envp[]) {
    if (spawn_mode == 1) return EAGAIN;
    if (spawn_mode == 2) {
        char *wrong[] = {(char *)self_path, "--wrong-ready", NULL};
        return posix_spawn(pid, self_path, actions, attributes, wrong, envp);
    }
    return posix_spawn(pid, path, actions, attributes, argv, envp);
}
static void init_vm(VmState *vm) {
    memset(vm, 0, sizeof *vm);
    vm->cop_pid = vm->cop_in_fd = vm->cop_out_fd = -1;
    vm->cop_sig_send_fd = vm->cop_sig_recv_fd = -1;
    vm->cop_timeout_ms = 5000;
}
static NanoValue call(VmState *vm, const NvmModule *module, VmHeap *heap,
                      uint32_t index, NanoValue argument) {
    char error[256] = {0}; NanoValue result = val_void();
    bool ok = vm ? vm_ffi_call_cop(vm, module, index, &argument, 1, &result, heap, error, sizeof error)
                 : vm_ffi_call(module, index, &argument, 1, &result, heap, error, sizeof error);
    if (!ok) fprintf(stderr, "%s\n", error);
    assert(ok && result.tag == TAG_INT);
    return result;
}
static void stopped(VmState *vm) {
    vm_ffi_cop_stop(vm);
    assert(vm->cop_pid == -1 && !vm->cop_mailbox && !vm->cop_opaque.generation);
}
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static int waiting;
static bool release_threads;
static void *concurrent(void *opaque) {
    const NvmModule *module = opaque;
    pthread_mutex_lock(&lock);
    ++waiting; pthread_cond_broadcast(&condition);
    while (!release_threads) pthread_cond_wait(&condition, &lock);
    pthread_mutex_unlock(&lock);
    VmState vm; init_vm(&vm);
    VmHeap heap; vm_heap_init(&heap);
    assert(call(&vm, module, &heap, 0, val_int(1)).as.i64 == 1);
    assert(call(&vm, module, &heap, 0, val_int(1)).as.i64 == 2);
    stopped(&vm); vm_heap_destroy(&heap);
    return NULL;
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--wrong-ready")) {
        assert(write(5, "WRONGv1!", 8) == 8);
        return 0;
    }
    assert(argc == 2); self_path = argv[0];
    const char *sdk = getenv("NANOLANG_SDK_ROOT"); assert(sdk);
    char *saved_sdk = strdup(sdk); assert(saved_sdk);
    NvmModule *module = nvm_module_new(); assert(module);
    uint32_t owner = nvm_add_string(module, argv[1], (uint32_t)strlen(argv[1]));
    const char *names[] = {"exec_add", "exec_text", "exec_fd_closed"};
    for (int i = 0; i < 3; ++i) {
        uint32_t name = nvm_add_string(module, names[i], (uint32_t)strlen(names[i]));
        uint8_t parameter = i == 1 ? TAG_STRING : TAG_INT;
        nvm_add_import(module, owner, name, 1, TAG_INT, &parameter);
        module->imports[i].kind = NVM_IMPORT_ARTIFACT;
    }
    VmState vm; init_vm(&vm);
    VmHeap heap; vm_heap_init(&heap);
    /* The parent mutation must not seed the fresh worker's image. */
    assert(call(NULL, module, &heap, 0, val_int(100)).as.i64 == 100);
    assert(setenv("NANOLANG_SDK_ROOT", "/nonexistent/nanolang-exec-sdk", 1) == 0);
    assert(!vm_ffi_cop_start(&vm, module));
    assert(setenv("NANOLANG_SDK_ROOT", saved_sdk, 1) == 0); free(saved_sdk);
    for (spawn_mode = 1; spawn_mode <= 2; ++spawn_mode) {
        for (int repeat = 0; repeat < 3; ++repeat) {
            assert(!vm_ffi_cop_start(&vm, module));
            assert(vm.cop_pid == -1 && !vm.cop_mailbox && !vm.cop_opaque.generation);
        }
    }
    spawn_mode = 0;
    int fd = open("/dev/null", O_RDONLY); assert(fd >= 0);
    int sentinel = fcntl(fd, F_DUPFD, 256); assert(sentinel >= 256); close(fd);
    assert(call(&vm, module, &heap, 2, val_int(sentinel)).as.i64 == 1);
    assert(fcntl(sentinel, F_GETFD) >= 0); close(sentinel);
    assert(call(&vm, module, &heap, 0, val_int(1)).as.i64 == 1);
    pid_t original = vm.cop_pid;
    NanoValue arguments[] = {val_int(2), val_int(3)}, results[2];
    CopBatchCall batch[] = {{0, arguments, 1}, {0, arguments + 1, 1}};
    char error[256] = {0};
    assert(vm_ffi_call_cop_batch(&vm, module, batch, 2, results, &heap, error, sizeof error));
    assert(results[0].as.i64 == 3 && results[1].as.i64 == 6);
    char large[12290]; memset(large, 'x', sizeof large - 1); large[sizeof large - 1] = 0;
    NanoValue text = val_string(vm_string_new(&heap, large, sizeof large - 1));
    assert(call(&vm, module, &heap, 1, text).as.i64 == 12295);
    vm_release(&heap, text);
    assert(vm.cop_pid == original);
    stopped(&vm);
    assert(call(&vm, module, &heap, 0, val_int(1)).as.i64 == 1);
    assert(vm.cop_pid != original); stopped(&vm);
    assert(call(NULL, module, &heap, 0, val_int(0)).as.i64 == 100);
    /* An unavailable parent loader must not be inherited or entered by launch. */
    FfiLoaderFork token;
    assert(ffi_loader_fork_prepare(&token));
    pthread_t threads[4];
    for (int i = 0; i < 4; ++i) assert(!pthread_create(threads + i, NULL, concurrent, module));
    pthread_mutex_lock(&lock);
    while (waiting != 4) pthread_cond_wait(&condition, &lock);
    release_threads = true; pthread_cond_broadcast(&condition);
    pthread_mutex_unlock(&lock);
    for (int i = 0; i < 4; ++i) assert(!pthread_join(threads[i], NULL));
    ffi_loader_fork_parent(&token);
    assert(call(NULL, module, &heap, 0, val_int(0)).as.i64 == 100);
    vm_heap_destroy(&heap); nvm_module_free(module); vm_ffi_shutdown();
    puts("I checked exec startup, independent images, transports, descriptors and concurrent workers.");
    return 0;
}
