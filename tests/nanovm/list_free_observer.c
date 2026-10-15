/* I observe live heap ownership at guest print boundaries, before return. */
#include "nanovm/vm.h"
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
    assert(module && module->header.entry_point < module->function_count);
    VmState vm;
    vm_init(&vm, module);
    uint32_t entry = module->header.entry_point;
    const NvmFunctionEntry *function = &module->functions[entry];
    assert(function->arity == 0 && function->local_count < vm.stack_capacity);
    for (uint16_t i = 0; i < function->local_count; ++i)
        vm.stack[vm.stack_size++] = val_void();
    vm.frame_count = 1;
    vm.frames[0].fn_idx = entry;
    vm.frames[0].module = module;
    vm.frames[0].local_count = function->local_count;
    vm.frames[0].owned_callable = val_void();
    vm.current_fn = entry;
    vm.ip = function->code_offset;
    for (;;) {
        VmTrap trap = vm_core_execute(&vm);
        if (trap.type == TRAP_NONE) break;
        assert(trap.type == TRAP_PRINT);
        assert(vm.frame_count > 0 && trap.data.print.value.tag == TAG_INT);
        /* I drain deferred cycle suspects at every boundary. The collector
         * must preserve live aliases and reclaim the last released owner
         * while its function frame is still alive. */
        vm_gc_collect_cycles(&vm.heap);
        printf("%lld %zu\n", (long long)trap.data.print.value.as.i64,
               vm.heap.stats.num_objects);
        vm_release(&vm.heap, trap.data.print.value);
    }
    vm_destroy(&vm);
    nvm_module_free(module);
    return 0;
}
