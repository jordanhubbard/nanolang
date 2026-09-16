#include "nanovm/vm.h"
#include <assert.h>
#include <stdlib.h>

static bool fail_root, fail_publication;
static void *callback_malloc(size_t size) { return fail_root ? NULL : malloc(size); }
static NanoCallbackV1 *callback_create(NanoCallbackRuntime *runtime,
    const NanoCallbackSignature *signature, NanoCallbackExecute execute,
    NanoCallbackDrop drop, void *payload) {
    return fail_publication ? NULL : nano_callback_create(runtime, signature, execute, drop, payload);
}
#define malloc callback_malloc
#define nano_callback_create callback_create
#include "../../src/nanovm/vm_callback.c"
#undef malloc
#undef nano_callback_create

int g_argc;
char **g_argv;

int main(void) {
    NvmModule *module = nvm_module_new();
    assert(module);
    uint8_t code[] = {OP_RET};
    NvmFunctionEntry fn = {.upvalue_count = 1, .result_tag = TAG_VOID, .code_length = 1};
    fn.name_idx = nvm_add_string(module, "callback", 8);
    nvm_append_code(module, code, sizeof(code));
    assert(nvm_add_function(module, &fn) == 0);
    VmState vm;
    vm_init(&vm, module);
    size_t baseline = vm.heap.stats.num_objects;
    VmClosure *closure = vm_closure_new(&vm.heap, 0, 1);
    VmString *capture = vm_string_new(&vm.heap, "capture", 7);
    assert(closure && capture);
    closure->captures[0] = val_string(capture);
    NvmCallbackContract contract = {.abi_version = NVM_CALLBACK_ABI_RETAINED_V1,
        .parameter_idx = 0, .return_tag = TAG_VOID};
    fail_root = true;
    assert(!vm_callback_create(&vm, val_closure(closure), &contract));
    fail_root = false;
    assert(closure->header.ref_count == 1 && capture->header.ref_count == 1);
    fail_publication = true;
    assert(!vm_callback_create(&vm, val_closure(closure), &contract));
    fail_publication = false;
    assert(closure->header.ref_count == 1 && capture->header.ref_count == 1);
    NanoCallbackV1 *handle = vm_callback_create(&vm, val_closure(closure), &contract);
    assert(handle && closure->header.ref_count == 2);
    vm_release(&vm.heap, val_closure(closure));
    handle->release(handle);
    assert(vm_callback_pump(&vm, false) == 0);
    vm_gc_collect_cycles(&vm.heap);
    assert(vm.heap.stats.num_objects == baseline);
    assert(vm_callback_shutdown(&vm) == NANO_CALLBACK_OK);
    assert(vm_callback_shutdown(&vm) == NANO_CALLBACK_OK);
    assert(!vm_callback_create(&vm, val_function(0), &contract));
    vm_destroy(&vm);
    nvm_module_free(module);
    puts("I passed callback root allocation, failed publication, and owner cleanup checks.");
    return 0;
}
