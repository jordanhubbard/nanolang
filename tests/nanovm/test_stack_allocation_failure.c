/* I compile the real VM with isolated stack reallocation failure injection. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
static bool reject_realloc;
static unsigned realloc_calls;
static void *test_realloc(void *pointer, size_t size) {
    realloc_calls++;
    return reject_realloc ? NULL : realloc(pointer, size);
}
#define realloc test_realloc
#include "../../src/nanovm/vm.c"
#undef realloc
int g_argc;
char **g_argv;

int main(void) {
    VmState empty = {0};
    assert(stack_reserve(&empty, 1) == VM_OK);
    assert(empty.stack && empty.stack_capacity >= 1);
    unsigned calls = realloc_calls;
    NanoValue *original = empty.stack;
    uint32_t capacity = empty.stack_capacity;
    assert(stack_reserve(&empty, (uint64_t)UINT32_MAX + 1) == VM_ERR_MEMORY);
    assert(realloc_calls == calls && empty.stack == original);
    reject_realloc = true;
    assert(stack_reserve(&empty, (uint64_t)capacity + 1) == VM_ERR_MEMORY);
    assert(empty.stack == original && empty.stack_capacity == capacity);
    reject_realloc = false;
    free(empty.stack);

    NvmModule *module = nvm_module_new();
    uint8_t code[] = {OP_RET};
    NvmFunctionEntry fn = {.local_count = 4, .result_tag = TAG_VOID};
    fn.name_idx = nvm_add_string(module, "main", 4);
    fn.code_offset = nvm_append_code(module, code, sizeof(code));
    fn.code_length = sizeof(code);
    nvm_add_function(module, &fn);
    for (unsigned invoke = 0; invoke < 2; invoke++) {
        VmState vm;
        vm_init(&vm, module);
        vm.stack_capacity = 2; /* Force growth without exhausting host memory. */
        vm.stack[vm.stack_size++] = val_int(42);
        original = vm.stack;
        reject_realloc = true;
        VmResult result = invoke ? vm_invoke(&vm, 0, NULL, 0, NULL)
                                 : vm_call_function(&vm, 0, NULL, 0);
        reject_realloc = false;
        assert(result == VM_ERR_MEMORY);
        assert(vm.stack == original && vm.stack_capacity == 2);
        assert(vm.stack_size == 1 && vm.stack[0].as.i64 == 42);
        assert(vm.frame_count == 0);
        assert(vm_invoke(&vm, 0, NULL, 0, NULL) == VM_OK);
        assert(vm.stack_size == 1 && vm.stack[0].as.i64 == 42);
        vm_destroy(&vm);
    }
    nvm_module_free(module);
    puts("I passed stack reserve overflow, failure, entry atomicity and recovery checks.");
    return 0;
}
