/* I test fresh ordinary slices and corrected allocation recovery only. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static bool reject_string_allocation;
static void *substring_malloc(size_t size) {
    return reject_string_allocation ? NULL : malloc(size);
}
#define malloc substring_malloc
#include "../../src/nanovm/heap.c"
#undef malloc
#include "nanovm/vm.h"
int g_argc;
char **g_argv;

static void heap_slices(void) {
    VmHeap heap;
    vm_heap_init(&heap);
    VmString *source = vm_string_new(&heap, "ab\0cd", 5);
    assert(source);
    const struct { uint32_t start, length, expected; const char *bytes; } cases[] = {
        {1, 3, 3, "b\0c"}, {0, 5, 5, "ab\0cd"}, {3, 20, 2, "cd"},
        {0, 0, 0, ""}, {5, 2, 0, ""}, {6, 2, 0, ""}
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        VmString *slice = vm_string_substr(&heap, source, cases[i].start, cases[i].length);
        assert(slice && slice->length == cases[i].expected);
        assert(!memcmp(slice->data, cases[i].bytes, cases[i].expected));
        vm_release(&heap, val_string(slice));
        assert(source->header.ref_count == 1 && heap.stats.num_objects == 1);
    }
    vm_release(&heap, val_string(source));
    assert(!heap.stats.num_objects);
    vm_heap_destroy(&heap);
}
static void opcode_recovery(void) {
    const uint8_t code[] = {OP_LOAD_LOCAL,0,0, OP_LOAD_LOCAL,1,0,
                           OP_LOAD_LOCAL,2,0, OP_STR_SUBSTR, OP_RET};
    NvmModule *module = nvm_module_new();
    NvmFunctionEntry fn = {.arity=3, .local_count=3, .result_count=1, .result_tag=TAG_STRING};
    fn.name_idx = nvm_add_string(module, "slice", 5);
    fn.code_offset = nvm_append_code(module, code, sizeof code);
    fn.code_length = sizeof code;
    nvm_add_function(module, &fn);
    VmState vm;
    vm_init(&vm, module);
    uint64_t baseline = vm.heap.stats.num_objects;
    VmString *source = vm_string_new(&vm.heap, "ab\0cd", 5);
    assert(source);
    NanoValue args[] = {val_string(source), val_int(1), val_int(3)};
    NanoValue output = val_void();
    uint64_t allocated = vm.heap.stats.allocated;
    reject_string_allocation = true;
    assert(vm_invoke(&vm, 0, args, 3, &output) == VM_ERR_MEMORY);
    reject_string_allocation = false;
    assert(output.tag == TAG_VOID && vm.stack_size == 0 && vm.frame_count == 0);
    assert(strstr(vm.error_msg, "allocate the substring"));
    assert(source->header.ref_count == 1 && !memcmp(source->data, "ab\0cd", 5));
    assert(vm.heap.stats.num_objects == baseline + 1 && vm.heap.stats.allocated == allocated);
    for (unsigned i = 0; i < 8; i++) {
        assert(vm_invoke(&vm, 0, args, 3, &output) == VM_OK);
        assert(output.tag == TAG_STRING && output.as.string->length == 3);
        assert(!memcmp(output.as.string->data, "b\0c", 3));
        vm_release(&vm.heap, output);
        assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    }
    /* I preserve the existing non-integer index fallback without fabricating
     * pointers or lengths: these are ordinary owned strings from this heap. */
    args[1] = val_string(source);
    args[2] = val_string(source);
    assert(vm_invoke(&vm, 0, args, 3, &output) == VM_OK);
    assert(output.tag == TAG_STRING && output.as.string->length == 0);
    vm_release(&vm.heap, output);
    assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    args[0] = val_int(7);
    output = val_void();
    assert(vm_invoke(&vm, 0, args, 3, &output) == VM_ERR_TYPE_ERROR);
    assert(output.tag == TAG_VOID && vm.stack_size == 0 && vm.frame_count == 0);
    assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    vm_release(&vm.heap, val_string(source));
    assert(vm.heap.stats.num_objects == baseline);
    vm_destroy(&vm);
    nvm_module_free(module);
}
static void trim_recovery(void) {
    const uint8_t code[] = {OP_LOAD_LOCAL, 0, 0, OP_STR_TRIM, OP_RET};
    NvmModule *module = nvm_module_new();
    NvmFunctionEntry fn = {.arity=1, .local_count=1, .result_count=1, .result_tag=TAG_STRING};
    fn.name_idx = nvm_add_string(module, "trim", 4);
    fn.code_offset = nvm_append_code(module, code, sizeof code);
    fn.code_length = sizeof code;
    nvm_add_function(module, &fn);
    VmState vm;
    vm_init(&vm, module);
    uint64_t baseline = vm.heap.stats.num_objects;
    VmString *source = vm_string_new(&vm.heap, "  a b  ", 7);
    assert(source);
    NanoValue input = val_string(source), output = val_void();
    reject_string_allocation = true;
    assert(vm_invoke(&vm, 0, &input, 1, &output) == VM_ERR_MEMORY);
    reject_string_allocation = false;
    assert(output.tag == TAG_VOID && vm.stack_size == 0 && vm.frame_count == 0);
    assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    for (unsigned i = 0; i < 8; i++) {
        assert(vm_invoke(&vm, 0, &input, 1, &output) == VM_OK);
        assert(output.tag == TAG_STRING && output.as.string->length == 3);
        assert(!memcmp(output.as.string->data, "a b", 3));
        vm_release(&vm.heap, output);
        assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    }
    vm_release(&vm.heap, input);
    assert(vm.heap.stats.num_objects == baseline);
    vm_destroy(&vm);
    nvm_module_free(module);
}
static void character_operand_lifetime(void) {
    const uint8_t code[] = {OP_LOAD_LOCAL, 0, 0, OP_LOAD_LOCAL, 1, 0, OP_STR_CHAR_AT, OP_RET};
    NvmModule *module = nvm_module_new();
    NvmFunctionEntry fn = {.arity=2, .local_count=2, .result_count=1, .result_tag=TAG_INT};
    fn.name_idx = nvm_add_string(module, "byte_at", 7);
    fn.code_offset = nvm_append_code(module, code, sizeof code);
    fn.code_length = sizeof code;
    nvm_add_function(module, &fn);
    VmState vm;
    vm_init(&vm, module);
    uint64_t baseline = vm.heap.stats.num_objects;
    VmString *source = vm_string_new(&vm.heap, "a\0\xff", 3);
    assert(source);
    const int64_t indices[] = {-1, 0, 1, 2, 3, INT64_MAX};
    const int64_t expected[] = {-1, 97, 0, 255, -1, -1};
    NanoValue args[] = {val_string(source), val_int(0)}, output = val_void();
    for (unsigned i = 0; i < sizeof indices / sizeof indices[0]; i++) {
        args[1] = val_int(indices[i]);
        assert(vm_invoke(&vm, 0, args, 2, &output) == VM_OK);
        assert(output.tag == TAG_INT && output.as.i64 == expected[i]);
        assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    }
    /* My existing fallback uses zero for a normal owned non-integer value.
     * Each argument borrows the caller's live object; neither consumes it. */
    args[1] = val_string(source);
    for (unsigned i = 0; i < 8; i++) {
        assert(vm_invoke(&vm, 0, args, 2, &output) == VM_OK);
        assert(output.tag == TAG_INT && output.as.i64 == 97);
        assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    }
    args[0] = val_int(7);
    output = val_void();
    assert(vm_invoke(&vm, 0, args, 2, &output) == VM_ERR_TYPE_ERROR);
    assert(output.tag == TAG_VOID && !vm.stack_size && !vm.frame_count);
    assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
    vm_release(&vm.heap, val_string(source));
    assert(vm.heap.stats.num_objects == baseline);
    vm_destroy(&vm);
    nvm_module_free(module);
}
int main(void) {
    heap_slices();
    opcode_recovery();
    trim_recovery();
    character_operand_lifetime();
    puts("I passed ordinary string bytes, operand ownership, allocation status and recovery.");
    return 0;
}
