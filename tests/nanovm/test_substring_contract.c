/* I test fresh ordinary slices and corrected allocation recovery only. */
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
static bool reject_string_allocation;
static void *substring_malloc(size_t size) {
    return reject_string_allocation ? NULL : malloc(size);
}
static unsigned format_status_override;
static int format_snprintf(char *buffer, size_t capacity, const char *format, ...) {
    if (format_status_override == 1) return -1;
    if (format_status_override == 2) return (int)capacity;
    va_list args;
    va_start(args, format);
    int result = vsnprintf(buffer, capacity, format, args);
    va_end(args);
    return result;
}
#define snprintf format_snprintf
#define malloc substring_malloc
#include "../../src/nanovm/heap.c"
#undef malloc
#undef snprintf
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
static void case_conversion_recovery(void) {
    const unsigned char pattern[] = {'A', 'z', 0, 255, ' ', 'a', 'Z'};
    char bytes[300];
    for (unsigned i = 0; i < sizeof bytes; i++) bytes[i] = (char)pattern[i % sizeof pattern];
    const unsigned lengths[] = {0, 7, 255, 256, 300};
    for (unsigned mode = 0; mode < 2; mode++) {
        const uint8_t code[] = {OP_LOAD_LOCAL, 0, 0,
                               mode ? OP_STR_TO_UPPER : OP_STR_TO_LOWER, OP_RET};
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.arity=1, .local_count=1, .result_count=1, .result_tag=TAG_STRING};
        fn.name_idx = nvm_add_string(module, "case", 4);
        fn.code_offset = nvm_append_code(module, code, sizeof code);
        fn.code_length = sizeof code;
        nvm_add_function(module, &fn);
        VmState vm;
        vm_init(&vm, module);
        uint64_t baseline = vm.heap.stats.num_objects;
        for (unsigned n = 0; n < sizeof lengths / sizeof lengths[0]; n++) {
            VmString *source = vm_string_new(&vm.heap, bytes, lengths[n]);
            assert(source);
            NanoValue input = val_string(source), output = val_void();
            reject_string_allocation = true;
            VmResult attempt = vm_invoke(&vm, 0, &input, 1, &output);
            reject_string_allocation = false;
            if (!lengths[n]) {
                /* Empty output already exists: interning needs no allocation. */
                assert(attempt == VM_OK && output.tag == TAG_STRING && output.as.string == source);
                vm_release(&vm.heap, output);
            } else {
                assert(attempt == VM_ERR_MEMORY && output.tag == TAG_VOID);
                assert(strstr(vm.error_msg, "allocate the case-converted string"));
            }
            assert(!vm.stack_size && !vm.frame_count);
            assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
            for (unsigned repeat = 0; repeat < 4; repeat++) {
                assert(vm_invoke(&vm, 0, &input, 1, &output) == VM_OK);
                assert(output.tag == TAG_STRING);
                if (lengths[n]) assert(output.as.string != source);
                assert(output.as.string->length == lengths[n]);
                for (unsigned k = 0; k < lengths[n]; k++) {
                    unsigned char byte = (unsigned char)bytes[k];
                    unsigned char expected = mode ? (byte >= 'a' && byte <= 'z' ? byte - 32 : byte)
                                                  : (byte >= 'A' && byte <= 'Z' ? byte + 32 : byte);
                    assert((unsigned char)output.as.string->data[k] == expected);
                }
                vm_release(&vm.heap, output);
                assert(!memcmp(source->data, bytes, lengths[n]));
                assert(source->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
            }
            vm_release(&vm.heap, input);
            assert(vm.heap.stats.num_objects == baseline);
        }
        vm_destroy(&vm);
        nvm_module_free(module);
    }
}
static void primitive_format_lifetime(void) {
    for (unsigned mode = 0; mode < 2; mode++) {
        const uint8_t code[] = {OP_LOAD_LOCAL, 0, 0,
                               mode ? OP_STR_FROM_FLOAT : OP_STR_FROM_INT, OP_RET};
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.arity=1, .local_count=1, .result_count=1, .result_tag=TAG_STRING};
        fn.name_idx = nvm_add_string(module, "format", 6);
        fn.code_offset = nvm_append_code(module, code, sizeof code);
        fn.code_length = sizeof code;
        nvm_add_function(module, &fn);
        VmState vm;
        vm_init(&vm, module);
        uint64_t baseline = vm.heap.stats.num_objects;
        VmString *owned = vm_string_new(&vm.heap, "owned-input", 11);
        assert(owned);
        NanoValue values[] = {val_int(INT64_MIN), val_int(INT64_MAX), val_int(-42),
                              val_float(-0.0), val_float(1.25), val_bool(true),
                              val_u8(255), val_void(), val_string(owned)};
        const char *integers[] = {"-9223372036854775808", "9223372036854775807", "-42",
                                  "0", "0", "0", "0", "0", "0"};
        const char *floats[] = {"0", "0", "0", "-0", "1.25", "0", "0", "0", "0"};
        for (unsigned i = 0; i < sizeof values / sizeof values[0]; i++) {
            NanoValue output = val_void();
            reject_string_allocation = true;
            assert(vm_invoke(&vm, 0, &values[i], 1, &output) == VM_ERR_MEMORY);
            reject_string_allocation = false;
            assert(output.tag == TAG_VOID && !vm.stack_size && !vm.frame_count);
            assert(owned->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
            for (unsigned repeat = 0; repeat < 3; repeat++) {
                assert(vm_invoke(&vm, 0, &values[i], 1, &output) == VM_OK);
                const char *expected = mode ? floats[i] : integers[i];
                assert(output.tag == TAG_STRING && output.as.string->length == strlen(expected));
                assert(!memcmp(output.as.string->data, expected, strlen(expected)));
                vm_release(&vm.heap, output);
                assert(owned->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
            }
        }
        for (format_status_override = 1; format_status_override <= 2; format_status_override++) {
            assert(!vm_string_from_int(&vm.heap, 42));
            assert(!vm_string_from_float(&vm.heap, 1.25));
            NanoValue input = val_string(owned), output = val_void();
            assert(vm_invoke(&vm, 0, &input, 1, &output) == VM_ERR_MEMORY);
            assert(output.tag == TAG_VOID && !vm.stack_size && !vm.frame_count);
            assert(owned->header.ref_count == 1 && vm.heap.stats.num_objects == baseline + 1);
        }
        format_status_override = 0;
        vm_release(&vm.heap, val_string(owned));
        assert(vm.heap.stats.num_objects == baseline);
        vm_destroy(&vm);
        nvm_module_free(module);
    }
}
int main(void) {
    heap_slices();
    opcode_recovery();
    trim_recovery();
    character_operand_lifetime();
    case_conversion_recovery();
    primitive_format_lifetime();
    puts("I passed ordinary string bytes, operand ownership, allocation status and recovery.");
    return 0;
}
