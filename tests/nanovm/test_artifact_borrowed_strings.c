/* I run actual descriptor calls and replace only the result-copy allocator. */
#include <assert.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/nanovm/vm_ffi.h"
#include "../../src/nanovm/cop_protocol.h"
static bool refuse_copy;
static bool refuse_fork;
static pid_t fixture_fork(void);
static VmString *fixture_string_new(VmHeap *heap, const char *text, uint32_t length);
#define vm_string_new fixture_string_new
#define fork fixture_fork
#include "../../src/nanovm/vm_ffi.c"
#undef fork
#undef vm_string_new
static VmString *fixture_string_new(VmHeap *heap, const char *text, uint32_t length) {
    return refuse_copy ? NULL : vm_string_new(heap, text, length);
}
static pid_t fixture_fork(void) {
    if (refuse_fork) { errno = EAGAIN; return -1; }
    return fork();
}
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static NanoValue text(VmHeap *heap, const char *value) {
    VmString *copy = vm_string_new(heap, value, (uint32_t)strlen(value));
    assert(copy);
    return val_string(copy);
}
static void equals(NanoValue value, const char *expected) {
    assert(value.tag == TAG_STRING && value.as.string);
    assert(!strcmp(vmstring_cstr(value.as.string), expected));
}
int main(int argc, char **argv) {
    assert(argc == 2);
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(library);
    int64_t (*calls)(void) = (int64_t (*)(void))dlsym(library, "artifact_calls");
    int64_t (*releases)(void) = (int64_t (*)(void))dlsym(library, "artifact_releases");
    assert(calls && releases);
    NvmModule *module = nvm_module_new();
    assert(module);
    uint32_t owner = nvm_add_string(module, argv[1], (uint32_t)strlen(argv[1]));
    const char *names[] = {"nlc_runtime_root", "nlc_module_artifact", "nl_fs_join_path", "nl_nanoisa_last_error"};
    uint8_t types[] = {TAG_STRING, TAG_STRING};
    for (int i = 0; i < 4; ++i) {
        uint32_t symbol = nvm_add_string(module, names[i], (uint32_t)strlen(names[i]));
        int arity = i == 3 ? 0 : i;
        nvm_add_import(module, owner, symbol, (uint16_t)arity, TAG_STRING, types);
        module->imports[i].kind = NVM_IMPORT_ARTIFACT;
    }
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue args[] = {text(&heap, "left"), text(&heap, "right")};
    NanoValue sentinel = val_int(77), out, before;
    char error[256];
    for (int i = 0; i < 4; ++i) {
        int arity = i == 3 ? 0 : i;
        memcpy(&out, &sentinel, sizeof out);
        memcpy(&before, &out, sizeof before);
        int64_t prior_calls = calls(), prior_releases = releases();
        refuse_copy = true;
        assert(!vm_ffi_call(module, (uint32_t)i, args, arity, &out, &heap, error, sizeof error));
        refuse_copy = false;
        assert(strstr(error, "could not retain the provider string result"));
        assert(!memcmp(&out, &before, sizeof out));
        assert(calls() == prior_calls + 1);
        assert(releases() == prior_releases + (i == 3));
        assert(vm_ffi_call(module, (uint32_t)i, args, arity, &out, &heap, error, sizeof error));
        assert(out.tag == TAG_STRING);
        vm_release(&heap, out);
    }
    for (int i = 1; i <= 2; ++i) {
        NanoValue invalid[] = {val_int(9), args[1]};
        memcpy(&out, &sentinel, sizeof out);
        memcpy(&before, &out, sizeof before);
        int64_t prior_calls = calls();
        assert(!vm_ffi_call(module, (uint32_t)i, invalid, i, &out, &heap, error, sizeof error));
        assert(!memcmp(&out, &before, sizeof out) && calls() == prior_calls);
        assert(!vm_ffi_call(module, (uint32_t)i, args, i - 1, &out, &heap, error, sizeof error));
        assert(!memcmp(&out, &before, sizeof out) && calls() == prior_calls);
        NanoValue null_args[] = {text(&heap, "null"), args[1]};
        assert(!vm_ffi_call(module, (uint32_t)i, null_args, i, &out, &heap, error, sizeof error));
        assert(!memcmp(&out, &before, sizeof out));
        vm_release(&heap, null_args[0]);
        assert(vm_ffi_call(module, (uint32_t)i, args, i, &out, &heap, error, sizeof error));
        equals(out, i == 1 ? "one-left" : "two-left:right");
        vm_release(&heap, out);
    }
    VmState *isolated = calloc(1, sizeof *isolated);
    assert(isolated);
    isolated->cop_pid = isolated->cop_in_fd = isolated->cop_out_fd = -1;
    isolated->cop_sig_send_fd = isolated->cop_sig_recv_fd = -1;
    isolated->cop_timeout_ms = 5000;
    isolated->isolate_ffi = true;
    refuse_fork = true;
    assert(!vm_ffi_cop_start(isolated, module));
    refuse_fork = false;
    assert(ffi_loader_is_initialized());
    assert(isolated->cop_pid == -1);
    CopBatchCall batch[] = {{0, NULL, 0}, {1, args, 1}, {2, args, 2}};
    NanoValue results[3];
    assert(vm_ffi_call_cop_batch(isolated, module, batch, 3, results, &heap, error, sizeof error));
    assert(results[0].tag == TAG_STRING);
    equals(results[1], "one-left");
    equals(results[2], "two-left:right");
    char *large = malloc(8193);
    assert(large);
    memset(large, 'x', 8192); large[8192] = 0;
    NanoValue large_arg = text(&heap, large);
    free(large);
    assert(vm_ffi_call_cop(isolated, module, 1, &large_arg, 1, &out, &heap, error, sizeof error));
    assert(out.tag == TAG_STRING && out.as.string->length == 8196);
    assert(!strncmp(vmstring_cstr(out.as.string), "one-", 4));
    for (uint32_t i = 4; i < out.as.string->length; ++i) assert(out.as.string->data[i] == 'x');
    vm_release(&heap, out);
    vm_release(&heap, large_arg);
    equals(results[1], "one-left");
    equals(results[2], "two-left:right");
    for (int i = 0; i < 3; ++i) vm_release(&heap, results[i]);
    vm_ffi_cop_stop(isolated);
    assert(isolated->cop_pid == -1);
    free(isolated);
    for (int i = 0; i < 2; ++i) vm_release(&heap, args[i]);
    vm_heap_destroy(&heap);
    nvm_module_free(module);
    vm_ffi_shutdown();
    dlclose(library);
    puts("I checked borrowed-string ABI, refusal, recovery and isolated snapshots.");
    return 0;
}
