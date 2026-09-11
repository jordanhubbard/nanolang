/* I inject failure only into the heap implementation compiled in this test.
 * Production allocation APIs and unrelated allocations remain unchanged. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

static bool reject_calloc;
static unsigned frees;
static void *heap_test_calloc(size_t count, size_t size) {
    return reject_calloc ? NULL : calloc(count, size);
}
static void heap_test_free(void *pointer) {
    if (pointer) frees++;
    free(pointer);
}
#define calloc heap_test_calloc
#define free heap_test_free
#include "../../src/nanovm/heap.c"
#undef calloc
#undef free
#include "nanovm/vm.h"

int g_argc;
char **g_argv;

static void test_constructor_traps(void) {
    const uint8_t opcodes[] = {OP_ARR_LITERAL, OP_STRUCT_LITERAL,
        OP_UNION_CONSTRUCT, OP_TUPLE_NEW, OP_CLOSURE_NEW, OP_AGG_PACK};
    for (size_t i = 0; i < sizeof(opcodes); i++) {
        uint8_t code[64];
        DecodedInstruction push = {.opcode = OP_PUSH_I64, .operand_count = 1};
        push.operands[0].i64 = 42;
        size_t length = isa_encode(&push, code, sizeof(code));
        DecodedInstruction construct = {.opcode = opcodes[i]};
        construct.operand_count = isa_get_info(opcodes[i])->operand_count;
        switch (opcodes[i]) {
        case OP_ARR_LITERAL:
            construct.operands[0].u8 = TAG_INT;
            construct.operands[1].u16 = 1;
            break;
        case OP_STRUCT_LITERAL:
        case OP_CLOSURE_NEW:
            construct.operands[1].u16 = 1;
            break;
        case OP_UNION_CONSTRUCT:
            construct.operands[2].u16 = 1;
            break;
        case OP_TUPLE_NEW:
            construct.operands[0].u16 = 1;
            break;
        case OP_AGG_PACK:
            construct.operands[0].u8 = AGG_TUPLE;
            construct.operands[3].u16 = 1;
            break;
        }
        length += isa_encode(&construct, code + length, sizeof(code) - length);
        code[length++] = OP_HALT;
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.result_count = 1, .local_count = 2};
        fn.name_idx = nvm_add_string(module, "main", 4);
        fn.code_offset = nvm_append_code(module, code, (uint32_t)length);
        fn.code_length = (uint32_t)length;
        module->header.entry_point = nvm_add_function(module, &fn);
        module->header.flags |= NVM_FLAG_HAS_MAIN;
        VmState vm;
        vm_init(&vm, module);
        vm.stack[vm.stack_size++] = val_int(100);
        uint64_t allocated = vm.heap.stats.allocated;
        uint64_t objects = vm.heap.stats.num_objects;
        reject_calloc = true;
        assert(vm_execute(&vm) == VM_ERR_MEMORY);
        reject_calloc = false;
        assert(vm.stack_size == 4);
        assert(vm.stack[0].as.i64 == 100);
        assert(vm.stack[1].tag == TAG_VOID && vm.stack[2].tag == TAG_VOID);
        assert(vm.stack[3].tag == TAG_INT && vm.stack[3].as.i64 == 42);
        assert(vm.heap.stats.allocated == allocated);
        assert(vm.heap.stats.num_objects == objects);
        vm_destroy(&vm);
        nvm_module_free(module);
    }
}

int main(void) {
    VmHeap heap;
    vm_heap_init(&heap);
    uint64_t allocated = heap.stats.allocated;
    uint64_t objects = heap.stats.num_objects;
    uint64_t calls = heap.stats.allocation_calls;
    unsigned before = frees;
    reject_calloc = true;
    assert(vm_struct_new(&heap, 0, 2) == NULL);
    assert(frees == before + 1);
    assert(vm_union_new(&heap, 0, 0, 2) == NULL);
    assert(frees == before + 2);
    assert(heap.stats.allocated == allocated);
    assert(heap.stats.num_objects == objects);
    assert(heap.stats.allocation_calls == calls);
    /* A null zero-sized field allocation is legal, not an allocation error. */
    VmStruct *record = vm_struct_new(&heap, 0, 0);
    VmUnion *variant = vm_union_new(&heap, 0, 0, 0);
    assert(record && variant);
    reject_calloc = false;
    vm_release(&heap, val_struct(record));
    vm_release(&heap, val_union(variant));
    record = vm_struct_new(&heap, 0, 2);
    variant = vm_union_new(&heap, 0, 0, 2);
    assert(record && record->fields && variant && variant->fields);
    vm_release(&heap, val_struct(record));
    vm_release(&heap, val_union(variant));
    vm_heap_destroy(&heap);
    test_constructor_traps();
    puts("I passed struct/union field-allocation failure and recovery checks.");
    return 0;
}
