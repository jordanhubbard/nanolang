#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
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
#include "../../src/runtime/callback_runtime.h"
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
    /* I observe normal exit independently of the production 50 ms stop grace.
     * A zero wait status includes the exec image's sanitizer exit handlers. */
    pid_t owned = vm->cop_pid;
    assert(owned > 0);
    if (vm->cop_sig_send_fd >= 0) { close(vm->cop_sig_send_fd); vm->cop_sig_send_fd = -1; }
    if (vm->cop_in_fd >= 0) { close(vm->cop_in_fd); vm->cop_in_fd = -1; }
    int status = -1;
    int64_t start = cop_now_ms(); assert(start >= 0);
    pid_t observed = 0;
    do {
        observed = waitpid(owned, &status, WNOHANG);
        if (observed < 0 && errno == EINTR) continue;
        if (observed != 0) break;
        usleep(10000);
    } while (cop_now_ms() >= 0 && cop_now_ms() - start < 5000);
    if (observed != owned) {
        kill(owned, SIGKILL);
        while (waitpid(owned, NULL, 0) < 0 && errno == EINTR) {}
    }
    assert(observed == owned && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    printf("I observed exec worker %ld exit 0 before parent cleanup.\n", (long)owned);
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
typedef struct { const NvmModule *module; pthread_t owner; int executed, dropped; } CallbackPayload;
typedef struct { NanoCallbackV1 *callback; int64_t (*invoke)(NanoCallbackV1 *); int64_t result; } CallbackThread;
static void *callback_native(void *opaque) {
    CallbackThread *thread = opaque;
    thread->result = thread->invoke(thread->callback);
    return NULL;
}
static NanoCallbackStatus callback_execute(void *opaque, const NanoCallbackValue *arguments,
                                          uint32_t count, NanoCallbackValue *result) {
    CallbackPayload *payload = opaque;
    (void)arguments;
    assert(!count && pthread_equal(payload->owner, pthread_self()));
    const NvmModule *module = payload->module;
    /* An unavailable parent loader must not be inherited or entered by launch. */
    FfiLoaderFork token;
    assert(ffi_loader_fork_prepare(&token));
    pthread_t threads[4];
    for (int i = 0; i < 4; ++i) assert(!pthread_create(threads + i, NULL, concurrent, (void *)module));
    pthread_mutex_lock(&lock);
    while (waiting != 4) pthread_cond_wait(&condition, &lock);
    release_threads = true; pthread_cond_broadcast(&condition);
    pthread_mutex_unlock(&lock);
    for (int i = 0; i < 4; ++i) assert(!pthread_join(threads[i], NULL));
    ffi_loader_fork_parent(&token);
    ++payload->executed;
    *result = (NanoCallbackValue){.tag = NANO_CALLBACK_INT, .as.integer = 42};
    return NANO_CALLBACK_OK;
}
static void callback_drop(void *opaque) {
    CallbackPayload *payload = opaque;
    assert(pthread_equal(payload->owner, pthread_self())); ++payload->dropped;
}
static void snapshot_metadata(void) {
    /* I use the same public V2 codec as launch, without executing this import. */
    NvmModule *module = nvm_module_new(); assert(module);
    uint32_t library = nvm_add_string(module, "/abs/declared-provider.so", (uint32_t)strlen("/abs/declared-provider.so"));
    uint32_t symbol = nvm_add_string(module, "submit", 6);
    uint32_t adapter = nvm_add_string(module, "retained_submit", 15);
    uint8_t parameters[] = {TAG_OPAQUE, TAG_FUNCTION};
    assert(nvm_add_import(module, library, symbol, 2, TAG_VOID, parameters) == 0);
    module->imports[0].kind = NVM_IMPORT_ARTIFACT;
    NvmCallbackContract callback = {.import_idx = 0, .adapter_name_idx = adapter,
        .parameter_idx = 1, .abi_version = NVM_CALLBACK_ABI_RETAINED_V1,
        .execution = NVM_FOREIGN_WORKER_THREAD, .param_count = 2,
        .return_tag = TAG_BOOL, .param_tags = {TAG_INT, TAG_FLOAT}};
    assert(nvm_add_callback_contract(module, &callback));
    uint32_t key = nvm_add_string(module, "user.audit", 10);
    const char bytes[] = {'a', 0, 'b'};
    uint32_t value = nvm_add_string(module, bytes, sizeof bytes);
    assert(nvm_add_metadata(module, key, value));
    assert(nvm_add_module_ref(module, library) != UINT32_MAX);
    NanoisaErr error; uint32_t size = 0;
    uint8_t *snapshot = nanoisa_save_bytes(module, &size, &error); assert(snapshot && size);
    NvmModule *copy = nanoisa_load_bytes(snapshot, size, &error); assert(copy);
    assert(copy->import_count == 1 && copy->imports[0].kind == NVM_IMPORT_ARTIFACT);
    assert(copy->imports[0].module_name_idx == library && copy->imports[0].function_name_idx == symbol);
    assert(copy->imports[0].param_count == 2 && copy->imports[0].return_type == TAG_VOID);
    assert(!memcmp(copy->import_param_types[0], parameters, sizeof parameters));
    assert(copy->callback_contract_count == 1);
    NvmCallbackContract *actual = copy->callback_contracts;
    assert(actual->import_idx == callback.import_idx && actual->parameter_idx == callback.parameter_idx);
    assert(actual->adapter_name_idx == adapter && actual->abi_version == callback.abi_version);
    assert(actual->execution == callback.execution && actual->param_count == callback.param_count);
    assert(actual->return_tag == callback.return_tag && !memcmp(actual->param_tags, callback.param_tags, 2));
    assert(copy->metadata_count == 1 && copy->metadata[0].key_idx == key && copy->metadata[0].value_idx == value);
    assert(copy->string_lengths[value] == sizeof bytes && !memcmp(copy->strings[value], bytes, sizeof bytes));
    assert(copy->module_ref_count == 1 && copy->module_refs[0].module_name_idx == library);
    assert(!copy->call_descriptors && !copy->call_descriptor_count);
    nvm_module_free(copy); free(snapshot); nvm_module_free(module);
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--wrong-ready")) {
        assert(write(5, "WRONGv1!", 8) == 8);
        return 0;
    }
    assert(argc == 2); self_path = argv[0];
    snapshot_metadata();
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
    int sentinel = fcntl(fd, F_DUPFD_CLOEXEC, 16); assert(sentinel >= 16); close(fd);
    int descriptor_flags = fcntl(sentinel, F_GETFD); assert(descriptor_flags >= 0);
    /* I deliberately expose this owned descriptor to prove the spawn actions
     * close it, rather than relying on ordinary kernel CLOEXEC behavior. */
    descriptor_flags &= ~FD_CLOEXEC;
    assert(fcntl(sentinel, F_SETFD, descriptor_flags) == 0);
    struct stat sentinel_before, sentinel_after;
    assert(fstat(sentinel, &sentinel_before) == 0);
    assert(call(&vm, module, &heap, 2, val_int(sentinel)).as.i64 == 1);
    assert(fcntl(sentinel, F_GETFD) == descriptor_flags);
    assert(fstat(sentinel, &sentinel_after) == 0);
    assert(sentinel_after.st_dev == sentinel_before.st_dev &&
           sentinel_after.st_ino == sentinel_before.st_ino &&
           sentinel_after.st_mode == sentinel_before.st_mode);
    close(sentinel);
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
    /* A real retained native callback stays queued on a foreign thread while
     * its owner launches workers under unavailable parent loader admission. */
    NanoCallbackRuntime *runtime = nano_callback_runtime_create(); assert(runtime);
    CallbackPayload payload = {.module = module, .owner = pthread_self()};
    NanoCallbackSignature signature = {.result_tag = NANO_CALLBACK_INT};
    NanoCallbackV1 *callback = nano_callback_create(runtime, &signature, callback_execute, callback_drop, &payload);
    assert(callback);
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL); assert(library);
    CallbackThread native = {.callback = callback};
    *(void **)(&native.invoke) = dlsym(library, "exec_retained_callback"); assert(native.invoke);
    pthread_t callback_thread;
    assert(!pthread_create(&callback_thread, NULL, callback_native, &native));
    while (!payload.executed) assert(nano_callback_pump(runtime, true) >= 0);
    assert(!pthread_join(callback_thread, NULL));
    assert(native.result == 42 && payload.executed == 1);
    callback->release(callback); nano_callback_collect(runtime);
    assert(payload.dropped == 1);
    assert(nano_callback_runtime_destroy(runtime) == NANO_CALLBACK_OK);
    dlclose(library);
    assert(call(NULL, module, &heap, 0, val_int(0)).as.i64 == 100);
    vm_heap_destroy(&heap); nvm_module_free(module); vm_ffi_shutdown();
    puts("I checked exec startup, independent images, transports, descriptors and concurrent workers.");
    return 0;
}
