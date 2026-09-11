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

static void test_foreign_result_reservation(void) {
    for (unsigned argc = 0; argc <= 1; argc++) {
        for (unsigned has_result = 0; has_result <= 1; has_result++) {
            uint8_t code[16];
            DecodedInstruction call = {.opcode = OP_CALL_EXTERN, .operand_count = 1};
            size_t size = isa_encode(&call, code, sizeof(code));
            code[size++] = OP_HALT;
            NvmModule *module = nvm_module_new();
            NvmFunctionEntry fn = {.result_count = 1};
            fn.name_idx = nvm_add_string(module, "main", 4);
            fn.code_offset = nvm_append_code(module, code, (uint32_t)size);
            fn.code_length = (uint32_t)size;
            nvm_add_function(module, &fn);
            uint32_t library = nvm_add_string(module, "test", 4);
            uint32_t name = nvm_add_string(module, "foreign", 7);
            uint8_t arg_type = TAG_INT;
            nvm_add_import(module, library, name, (uint16_t)argc,
                           has_result ? TAG_INT : TAG_VOID, &arg_type);
            VmState vm;
            vm_init(&vm, module);
            vm.frame_count = 1;
            vm.frames[0].module = module;
            vm.frames[0].stack_base = 1;
            vm.stack_capacity = 1 + argc;
            vm.stack[vm.stack_size++] = val_int(100);
            if (argc) vm.stack[vm.stack_size++] = val_int(7);
            unsigned calls = realloc_calls;
            reject_realloc = true;
            /* I stop at the real trap boundary; no host function is invoked. */
            VmTrap trap = vm_core_execute(&vm);
            reject_realloc = false;
            if (!argc && has_result) {
                assert(trap.type == TRAP_ERROR && trap.data.error.code == VM_ERR_MEMORY);
                assert(realloc_calls == calls + 1);
            } else {
                assert(trap.type == TRAP_EXTERN_CALL);
                assert(realloc_calls == calls);
                assert(trap.data.extern_call.argc == (int)argc);
                if (argc) assert(trap.data.extern_call.args[0].as.i64 == 7);
            }
            assert(vm.stack_size == 1 && vm.stack[0].as.i64 == 100);
            vm_destroy(&vm);
            nvm_module_free(module);
        }
    }
}

static void test_instruction_growth(void) {
    const uint8_t ops[] = {OP_PUSH_I64, OP_DUP, OP_PICK,
                          OP_TUPLE_NEW, OP_CLOSURE_NEW, OP_ADD};
    for (size_t i = 0; i < sizeof(ops); i++) {
        unsigned inputs = ops[i] == OP_ADD ? 2
            : (ops[i] == OP_DUP || ops[i] == OP_PICK ? 1 : 0);
        uint8_t code[64];
        size_t size = 0;
        DecodedInstruction push = {.opcode = OP_PUSH_I64, .operand_count = 1};
        push.operands[0].i64 = 7;
        for (unsigned j = 0; j < inputs; j++)
            size += isa_encode(&push, code + size, sizeof(code) - size);
        DecodedInstruction operation = {.opcode = ops[i]};
        operation.operand_count = isa_get_info(ops[i])->operand_count;
        size += isa_encode(&operation, code + size, sizeof(code) - size);
        code[size++] = OP_HALT;
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.result_count = 1};
        fn.name_idx = nvm_add_string(module, "main", 4);
        fn.code_offset = nvm_append_code(module, code, (uint32_t)size);
        fn.code_length = (uint32_t)size;
        nvm_add_function(module, &fn);
        module->header.flags |= NVM_FLAG_HAS_MAIN;
        VmState vm;
        vm_init(&vm, module);
        vm.stack_capacity = 1 + inputs;
        vm.stack[vm.stack_size++] = val_int(100);
        NanoValue *original = vm.stack;
        uint64_t allocated = vm.heap.stats.allocated;
        unsigned calls = realloc_calls;
        reject_realloc = true;
        VmResult result = vm_execute(&vm);
        reject_realloc = false;
        assert(vm.stack == original && vm.stack[0].as.i64 == 100);
        assert(vm.heap.stats.allocated == allocated);
        if (ops[i] == OP_ADD) {
            assert(result == VM_OK && realloc_calls == calls);
            assert(vm.stack_size == 2 && vm.stack[1].as.i64 == 14);
        } else {
            assert(result == VM_ERR_MEMORY);
            assert(vm.stack_size == 1 + inputs);
            for (unsigned j = 0; j < inputs; j++) assert(vm.stack[j + 1].as.i64 == 7);
        }
        vm_destroy(&vm);
        nvm_module_free(module);
    }
}

static void test_internal_calls(void) {
    const uint8_t ops[] = {OP_CALL, OP_TAIL_CALL, OP_CALL_INDIRECT, OP_CALL_MODULE};
    for (size_t i = 0; i < sizeof(ops); i++) {
        uint8_t code[64];
        DecodedInstruction push = {.opcode = OP_PUSH_I64, .operand_count = 1};
        push.operands[0].i64 = 7;
        size_t size = isa_encode(&push, code, sizeof(code));
        bool indirect = ops[i] == OP_CALL_INDIRECT;
        if (indirect) {
            DecodedInstruction ref = {.opcode = OP_FUNCREF, .operand_count = 1};
            ref.operands[0].u32 = 1;
            size += isa_encode(&ref, code + size, sizeof(code) - size);
        }
        DecodedInstruction call = {.opcode = ops[i]};
        call.operand_count = isa_get_info(ops[i])->operand_count;
        if (ops[i] == OP_CALL_MODULE) call.operands[2].u16 = 1;
        else call.operands[0].u32 = 1;
        size += isa_encode(&call, code + size, sizeof(code) - size);
        code[size++] = OP_RET;
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry main_fn = {.local_count = 2, .result_tag = TAG_VOID};
        main_fn.name_idx = nvm_add_string(module, "main", 4);
        main_fn.code_offset = nvm_append_code(module, code, (uint32_t)size);
        main_fn.code_length = (uint32_t)size;
        nvm_add_function(module, &main_fn);
        module->header.flags |= NVM_FLAG_HAS_MAIN;
        NvmFunctionEntry callee = {.arity = 1, .local_count = 16, .result_tag = TAG_VOID};
        uint8_t body[] = {OP_RET};
        callee.name_idx = nvm_add_string(module, "callee", 6);
        callee.code_offset = nvm_append_code(module, body, sizeof(body));
        callee.code_length = sizeof(body);
        nvm_add_function(module, &callee);
        NvmModule *linked = nvm_module_new();
        callee.name_idx = nvm_add_string(linked, "callee", 6);
        callee.code_offset = nvm_append_code(linked, body, sizeof(body));
        nvm_add_function(linked, &callee);
        VmState vm;
        vm_init(&vm, module);
        vm_link_module(&vm, linked);
        vm.stack_capacity = 8;
        vm.stack[vm.stack_size++] = val_int(100);
        NanoValue *original = vm.stack;
        reject_realloc = true;
        assert(vm_execute(&vm) == VM_ERR_MEMORY);
        reject_realloc = false;
        assert(vm.stack == original && vm.stack_capacity == 8);
        assert(vm.frame_count == 1 && vm.frames[0].fn_idx == 0);
        assert(vm.frames[0].local_count == 2);
        assert(vm.stack_size == 4 + indirect);
        assert(vm.stack[0].as.i64 == 100);
        assert(vm.stack[1].tag == TAG_VOID && vm.stack[2].tag == TAG_VOID);
        assert(vm.stack[3].as.i64 == 7);
        if (indirect) assert(vm.stack[4].tag == TAG_FUNCTION);
        vm_destroy(&vm);
        nvm_module_free(module);
        nvm_module_free(linked);
    }
}

int main(void) {
    test_foreign_result_reservation();
    test_instruction_growth();
    test_internal_calls();
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
