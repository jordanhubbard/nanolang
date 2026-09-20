#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* Required by runtime/cli.c. */
int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#include "../../src/nanovm/vm_ffi.h"
#include "../../src/nanovm/heap.h"
#include "../../src/nanovm/value.h"
#include "../../src/nanoisa/nvm_format.h"

static void init_isolated_vm(VmState *vm) {
    memset(vm, 0, sizeof(*vm));
    vm->cop_pid = -1;
    vm->cop_in_fd = -1;
    vm->cop_out_fd = -1;
    vm->cop_sig_send_fd = -1;
    vm->cop_sig_recv_fd = -1;
    vm->cop_timeout_ms = 5000;
    vm->isolate_ffi = true;
}

static bool child_exited_without_reaping(pid_t pid) {
    siginfo_t info;
    memset(&info, 0, sizeof(info));
    while (waitid(P_PID, (id_t)pid, &info, WEXITED | WNOWAIT) != 0) {
        if (errno != EINTR) return false;
    }
    return info.si_pid == pid &&
           (info.si_code == CLD_KILLED || info.si_code == CLD_DUMPED ||
            info.si_code == CLD_EXITED);
}

int main(void) {
    int status = 1;
    int sentinel_pipe[2] = {-1, -1};
    pid_t sentinel = -1;
    pid_t first_worker = -1;
    pid_t replacement_worker = -1;
    NvmModule *module = NULL;
    VmState vm;
    VmHeap heap;
    bool heap_ready = false;
    char error[256] = {0};
    NanoValue argument = val_int(-17), result = val_void();

#define REQUIRE(condition, message) do {                                      \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL: %s\n", (message));                            \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)

    vm_ffi_init();
    init_isolated_vm(&vm);
    vm_heap_init(&heap);
    heap_ready = true;

    module = nvm_module_new();
    REQUIRE(module != NULL, "I could not allocate the lifecycle module");
    uint32_t library = nvm_add_string(module, "", 0);
    uint32_t name = nvm_add_string(module, "abs", 3);
    uint8_t parameter = TAG_INT;
    uint32_t imported = nvm_add_import(module, library, name, 1, TAG_INT,
                                       &parameter);

    REQUIRE(vm.cop_pid <= 0, "I launched a co-process before an extern call");
    REQUIRE(pipe(sentinel_pipe) == 0, "I could not create the sentinel pipe");
    sentinel = fork();
    REQUIRE(sentinel >= 0, "I could not start the unrelated child");
    if (sentinel == 0) {
        char byte;
        close(sentinel_pipe[1]);
        while (read(sentinel_pipe[0], &byte, 1) < 0 && errno == EINTR) {}
        close(sentinel_pipe[0]);
        _exit(0);
    }
    close(sentinel_pipe[0]);
    sentinel_pipe[0] = -1;

    REQUIRE(vm_ffi_call_cop(&vm, module, imported, &argument, 1, &result,
                            &heap, error, sizeof(error)), error);
    REQUIRE(result.tag == TAG_INT && result.as.i64 == 17,
            "my first isolated result is wrong");
    first_worker = vm.cop_pid;
    REQUIRE(first_worker > 0, "I did not observe the owned worker identity");

    argument = val_int(-23);
    REQUIRE(vm_ffi_call_cop(&vm, module, imported, &argument, 1, &result,
                            &heap, error, sizeof(error)), error);
    REQUIRE(result.tag == TAG_INT && result.as.i64 == 23,
            "my repeated isolated result is wrong");
    REQUIRE(vm.cop_pid == first_worker,
            "I replaced a live worker between ordinary calls");

    REQUIRE(kill(first_worker, SIGKILL) == 0,
            "I could not inject the owned-worker crash");
    REQUIRE(child_exited_without_reaping(first_worker),
            "I did not observe the injected worker exit");

    argument = val_int(-29);
    REQUIRE(vm_ffi_call_cop(&vm, module, imported, &argument, 1, &result,
                            &heap, error, sizeof(error)), error);
    replacement_worker = vm.cop_pid;
    REQUIRE(result.tag == TAG_INT && result.as.i64 == 29,
            "my recovered isolated result is wrong");
    REQUIRE(replacement_worker > 0 && replacement_worker != first_worker,
            "I did not observe a distinct replacement worker");
    REQUIRE(kill(sentinel, 0) == 0,
            "I disturbed an unrelated child during worker recovery");

    vm_ffi_cop_stop(&vm);
    REQUIRE(vm.cop_pid == -1, "I retained an owned worker after stop");
    errno = 0;
    REQUIRE(waitpid(replacement_worker, NULL, WNOHANG) == -1 && errno == ECHILD,
            "I did not reap the stopped worker");
    REQUIRE(kill(sentinel, 0) == 0,
            "I disturbed an unrelated child during owned cleanup");

    printf("owned worker %ld crashed; replacement %ld was stopped and reaped\n",
           (long)first_worker, (long)replacement_worker);
    status = 0;

cleanup:
    if (vm.cop_pid > 0) vm_ffi_cop_stop(&vm);
    if (sentinel_pipe[0] >= 0) close(sentinel_pipe[0]);
    if (sentinel_pipe[1] >= 0) close(sentinel_pipe[1]);
    if (sentinel > 0) {
        while (waitpid(sentinel, NULL, 0) < 0 && errno == EINTR) {}
    }
    if (heap_ready) vm_heap_destroy(&heap);
    nvm_module_free(module);
    vm_ffi_shutdown();
    return status;

#undef REQUIRE
}
