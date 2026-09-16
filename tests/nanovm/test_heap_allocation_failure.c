/* I inject failure only into the heap implementation compiled in this test.
 * Production allocation APIs and unrelated allocations remain unchanged. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

static bool reject_calloc;
static bool reject_realloc;
static unsigned fail_malloc_at;
static unsigned malloc_calls;
static void *heap_test_malloc(size_t size) {
    malloc_calls++;
    if (fail_malloc_at && malloc_calls == fail_malloc_at) return NULL;
    return malloc(size);
}
static unsigned frees;
static void *heap_test_calloc(size_t count, size_t size) {
    return reject_calloc ? NULL : calloc(count, size);
}
static void heap_test_free(void *pointer) {
    if (pointer) frees++;
    free(pointer);
}
static void *heap_test_realloc(void *pointer, size_t size) {
    return reject_realloc ? NULL : realloc(pointer, size);
}
#define calloc heap_test_calloc
#define free heap_test_free
#define realloc heap_test_realloc
#define malloc heap_test_malloc
#include "../../src/nanovm/heap.c"
#undef calloc
#undef free
#undef realloc
#undef malloc
#include "nanovm/vm.h"

int g_argc;
char **g_argv;

static void test_string_allocation_boundaries(void) {
    VmHeap heap;
    reject_calloc = true;
    vm_heap_init(&heap);
    assert(!heap.intern_buckets && heap.intern_bucket_count == 0);
    assert(!vm_string_new(&heap, "first", 5));
    assert(heap.stats.num_objects == 0 && heap.stats.allocated == 0);
    reject_calloc = false;
    VmString *first = vm_string_new(&heap, "first", 5);
    assert(first && first->header.ref_count == 1 && heap.intern_count == 1);
    vm_release(&heap, val_string(first));
    assert(heap.stats.num_objects == 0);
    /* Only headers exist: the guard must run before any payload read. */
    VmString huge = {.length = UINT32_MAX};
    VmString one = {.length = 1};
    malloc_calls = 0;
    assert(!vm_string_concat(&heap, &huge, &one));
    assert(!vm_string_concat(&heap, &one, &huge));
    assert(!vm_string_concat(&heap, NULL, &one));
    assert(malloc_calls == 0);
    assert(!vm_string_new(&heap, NULL, 1));
    VmString *empty = vm_string_new(&heap, NULL, 0);
    assert(empty && empty->length == 0);
    malloc_calls = 0;
    fail_malloc_at = 1;
    VmString *joined = vm_string_concat(&heap, empty, empty);
    fail_malloc_at = 0;
    assert(joined == empty && empty->header.ref_count == 2 && malloc_calls == 0);
    vm_release(&heap, val_string(joined));
    vm_release(&heap, val_string(empty));
    assert(heap.stats.num_objects == 0);
    vm_heap_destroy(&heap);
}

static void test_string_instruction_allocation_failure(void) {
    const uint8_t ops[] = {OP_ADD, OP_STR_CONCAT, OP_ARRAY_ADD};
    for (unsigned i = 0; i < sizeof(ops); i++) {
        uint8_t code[] = {OP_LOAD_LOCAL, 0, 0, OP_LOAD_LOCAL, 1, 0, ops[i], OP_RET};
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.arity = 2, .local_count = 2, .result_count = 1,
                               .result_tag = i == 2 ? TAG_ARRAY : TAG_STRING};
        fn.name_idx = nvm_add_string(module, "concatenate", 11);
        fn.code_offset = nvm_append_code(module, code, sizeof(code));
        fn.code_length = sizeof(code);
        nvm_add_function(module, &fn);
        VmState vm;
        vm_init(&vm, module);
        uint64_t baseline = vm.heap.stats.num_objects;
        NanoValue a = val_string(vm_string_new(&vm.heap, "aa", 2));
        NanoValue b = val_string(vm_string_new(&vm.heap, "cc", 2));
        if (i == 2) {
            VmArray *array = vm_array_new(&vm.heap, TAG_STRING, 2);
            assert(vm_array_push(&vm.heap, array, a));
            vm_release(&vm.heap, a);
            a = val_string(vm_string_new(&vm.heap, "bb", 2));
            assert(vm_array_push(&vm.heap, array, a));
            vm_release(&vm.heap, a);
            a = val_array(array);
        }
        NanoValue args[] = {a, b};
        uint64_t inputs = vm.heap.stats.num_objects;
        for (unsigned failure = 1; failure <= (i == 2 ? 4u : 2u); failure++) {
            NanoValue output = val_void();
            malloc_calls = 0;
            fail_malloc_at = failure;
            assert(vm_invoke(&vm, 0, args, 2, &output) == VM_ERR_MEMORY);
            fail_malloc_at = 0;
            assert(output.tag == TAG_VOID);
            assert(((VmHeapHeader *)a.as.obj)->ref_count == 1);
            assert(b.as.string->header.ref_count == 1);
            vm_gc_collect_cycles(&vm.heap);
            assert(vm.heap.stats.num_objects == inputs);
            assert(vm_invoke(&vm, 0, args, 2, &output) == VM_OK);
            NanoValue string = i == 2 ? vm_array_get(output.as.array, 0) : output;
            assert(strcmp(vmstring_cstr(string.as.string), "aacc") == 0);
            assert(string.as.string->header.ref_count == 1);
            vm_release(&vm.heap, output);
            vm_gc_collect_cycles(&vm.heap);
            assert(vm.heap.stats.num_objects == inputs);
        }
        vm_release(&vm.heap, a);
        vm_release(&vm.heap, b);
        vm_gc_collect_cycles(&vm.heap);
        assert(vm.heap.stats.num_objects == baseline);
        vm_destroy(&vm);
        nvm_module_free(module);
    }
}

static void test_arithmetic_allocation_failure(void) {
    const uint8_t ops[] = {OP_ADD, OP_SUB, OP_MUL, OP_DIV,
                          OP_ARRAY_ADD, OP_ARRAY_SUB, OP_ARRAY_MUL, OP_ARRAY_DIV};
    for (unsigned i = 0; i < sizeof(ops); i++) {
        uint8_t code[] = {OP_LOAD_LOCAL, 0, 0, OP_LOAD_LOCAL, 1, 0, ops[i], OP_RET};
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.arity = 2, .local_count = 2,
                               .result_count = 1, .result_tag = TAG_ARRAY};
        fn.name_idx = nvm_add_string(module, "arithmetic", 10);
        fn.code_offset = nvm_append_code(module, code, sizeof(code));
        fn.code_length = sizeof(code);
        nvm_add_function(module, &fn);
        VmState vm;
        vm_init(&vm, module);
        uint64_t baseline = vm.heap.stats.num_objects;
        VmArray *array = vm_array_new(&vm.heap, TAG_FLOAT, 1);
        assert(vm_array_push(&vm.heap, array, val_float(4.5)));
        for (unsigned shape = 0; shape < 3; shape++) {
            NanoValue args[] = {shape == 2 ? val_float(2.5) : val_array(array),
                                shape == 1 ? val_float(2.5) : val_array(array)};
            NanoValue output = val_void();
            reject_calloc = true;
            assert(vm_invoke(&vm, 0, args, 2, &output) == VM_ERR_MEMORY);
            reject_calloc = false;
            assert(array->header.ref_count == 1 && array->length == 1);
            assert(vm_array_get(array, 0).as.f64 == 4.5);
            assert(vm_invoke(&vm, 0, args, 2, &output) == VM_OK);
            assert(output.tag == TAG_ARRAY && output.as.array->elem_type == TAG_FLOAT);
            vm_release(&vm.heap, output);
        }
        vm_release(&vm.heap, val_array(array));
        vm_gc_collect_cycles(&vm.heap);
        assert(vm.heap.stats.num_objects == baseline);
        vm_destroy(&vm);
        nvm_module_free(module);
    }
}

static void test_append_failure(void) {
    uint8_t code[] = {OP_LOAD_LOCAL, 0, 0, OP_LOAD_LOCAL, 1, 0, OP_ARR_PUSH, OP_RET};
    NvmModule *module = nvm_module_new();
    NvmFunctionEntry fn = {.arity = 2, .local_count = 2,
                           .result_count = 1, .result_tag = TAG_ARRAY};
    fn.name_idx = nvm_add_string(module, "append", 6);
    fn.code_offset = nvm_append_code(module, code, sizeof(code));
    fn.code_length = sizeof(code);
    nvm_add_function(module, &fn);
    VmState vm;
    vm_init(&vm, module);
    uint64_t baseline_objects = vm.heap.stats.num_objects;
    VmArray *array = vm_array_new(&vm.heap, TAG_STRING, 8);
    VmString *existing = vm_string_new(&vm.heap, "old", 3);
    VmString *candidate = vm_string_new(&vm.heap, "new", 3);
    for (unsigned i = 0; i < 8; i++) assert(vm_array_push(&vm.heap, array, val_string(existing)));
    vm_release(&vm.heap, val_string(existing));
    NanoValue args[] = {val_array(array), val_string(candidate)};
    NanoValue output = val_void();
    reject_realloc = true;
    assert(!vm_array_push(&vm.heap, array, args[1]));
    assert(vm_invoke(&vm, 0, args, 2, &output) == VM_ERR_MEMORY);
    reject_realloc = false;
    assert(array->length == 8 && array->capacity == 8);
    assert(array->header.ref_count == 1 && candidate->header.ref_count == 1);
    for (unsigned i = 0; i < 8; i++) assert(vm_array_get(array, i).as.string == existing);
    assert(existing->header.ref_count == 8);
    assert(vm_invoke(&vm, 0, args, 2, &output) == VM_OK);
    assert(array->length == 9 && vm_array_get(array, 8).as.string == candidate);
    assert(array->header.ref_count == 2 && candidate->header.ref_count == 2);
    vm_release(&vm.heap, output);
    vm_release(&vm.heap, args[0]);
    vm_release(&vm.heap, args[1]);
    vm_gc_collect_cycles(&vm.heap);
    assert(vm.heap.stats.num_objects == baseline_objects);
    vm_destroy(&vm);
    nvm_module_free(module);
}

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
    test_append_failure();
    test_arithmetic_allocation_failure();
    test_string_allocation_boundaries();
    test_string_instruction_allocation_failure();
    puts("I passed struct/union field-allocation failure and recovery checks.");
    return 0;
}
