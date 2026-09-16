/* I exercise only module loading and VM ownership, so LeakSanitizer can
 * remain enabled without measuring compiler frontend allocations. */
#include "nanovm/vm.h"
#include "nanoisa/verifier.h"
#include "../../modules/nanoisa/nanoisa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int g_argc = 0;
char **g_argv = NULL;
int main(int argc, char **argv) {
    assert(argc == 2);
    NanoisaErr error;
    NvmModule *module = nanoisa_load_file(argv[1], &error);
    assert(module && nvm_verify(module).ok);
    VmState vm;
    vm_init(&vm, module);
    assert(vm.last_error == VM_OK && vm.verified);
    size_t baseline = vm.heap.stats.num_objects;
    for (int i = 0; i < 100; i++) {
        NanoValue result = val_void();
        assert(vm_invoke_callable(&vm, val_function(module->header.entry_point), NULL, 0, &result) == VM_OK);
        assert(result.tag == TAG_INT && result.as.i64 == 280);
        vm_release(&vm.heap, result);
        assert(vm.frame_count == 0 && vm.handler_count == 0 && vm.stack_size == 0);
        vm_gc_collect_cycles(&vm.heap);
        assert(vm.heap.stats.num_objects == baseline);
    }
    vm_destroy(&vm);
    assert(vm.heap.stats.num_objects == 0);
    nvm_module_free(module);
    puts("I completed 100 invocations, each with 20 recursive resumption and lexical-exit pairs; frames, handlers, stack and heap returned to baseline.");
    return 0;
}
