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
static void replacement_bounds_and_recovery(void) {
    uint32_t length = 99;
    assert(vm_string_replacement_length(10, 2, 0, 5, &length) && length == 0);
    assert(vm_string_replacement_length(10, 2, 3, 5, &length) && length == 15);
    assert(vm_string_replacement_length(UINT32_MAX, 1, 1, UINT32_MAX, &length) && length == UINT32_MAX);
    length = 99;
    assert(!vm_string_replacement_length(UINT32_MAX, 1, 2, UINT32_MAX, &length) && length == 99);
    assert(!vm_string_replacement_length(10, 2, 1, 6, &length) && length == 99);
    assert(!vm_string_replacement_length(10, 2, 1, UINT64_MAX, &length) && length == 99);
    assert(!vm_string_replacement_length(10, 0, 1, 1, &length) && length == 99);
    assert(!vm_string_replacement_length(10, 1, 1, 1, NULL));
    assert(vm_string_replacement_length(10, 0, 1, 0, &length) && length == 10);
    char large[600], grown[1200];
    memset(large, 'a', sizeof large);
    for (unsigned i = 0; i < sizeof grown; i++) grown[i] = i % 2 ? 'C' : 'B';
    const struct {const char *s, *needle, *replacement, *expected;uint32_t sl, nl, rl, el;} cases[] = {
        {"abcabc", "bc", "", "aa", 6, 2, 0, 2},
        {"abab", "ab", "XYZ", "XYZXYZ", 4, 2, 3, 6},
        {"a\0ba\0b", "\0b", "Z\0", "aZ\0aZ\0", 6, 2, 2, 6},
        {"aaaaa", "aa", "b", "bba", 5, 2, 1, 3},
        {"abc", "x", "y", "abc", 3, 1, 1, 3},
        {"abc", "", "XYZ", "abc", 3, 0, 3, 3},
        {"", "a", "b", "", 0, 1, 1, 0},
        {"aaa", "a", "", "", 3, 1, 0, 0},
        {large, "a", "BC", grown, sizeof large, 1, 2, sizeof grown}
    };
    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const uint8_t code[] = {OP_LOAD_LOCAL,0,0, OP_LOAD_LOCAL,1,0,
                               OP_LOAD_LOCAL,2,0, OP_STR_REPLACE, OP_RET};
        NvmModule *module = nvm_module_new();
        NvmFunctionEntry fn = {.arity=3, .local_count=3, .result_count=1, .result_tag=TAG_STRING};
        fn.name_idx = nvm_add_string(module, "replace", 7);
        fn.code_offset = nvm_append_code(module, code, sizeof code);
        fn.code_length = sizeof code;
        nvm_add_function(module, &fn);
        VmState vm;
        vm_init(&vm, module);
        uint64_t baseline = vm.heap.stats.num_objects;
        VmString *inputs[] = {vm_string_new(&vm.heap, cases[c].s, cases[c].sl),
                              vm_string_new(&vm.heap, cases[c].needle, cases[c].nl),
                              vm_string_new(&vm.heap, cases[c].replacement, cases[c].rl)};
        unsigned owners[3] = {0};
        bool interned = false;
        for (unsigned j = 0; j < 3; j++) {
            assert(inputs[j]);
            for (unsigned k = 0; k < 3; k++) owners[j] += inputs[j] == inputs[k];
            if (inputs[j]->length == cases[c].el && !memcmp(inputs[j]->data, cases[c].expected, cases[c].el)) interned = true;
        }
        uint64_t populated = vm.heap.stats.num_objects;
        NanoValue args[] = {val_string(inputs[0]), val_string(inputs[1]), val_string(inputs[2])};
        NanoValue output = val_void();
        reject_string_allocation = true;
        VmResult attempted = vm_invoke(&vm, 0, args, 3, &output);
        reject_string_allocation = false;
        if (interned) {
            assert(attempted == VM_OK && output.tag == TAG_STRING);
            vm_release(&vm.heap, output);
        } else {
            assert(attempted == VM_ERR_MEMORY && output.tag == TAG_VOID);
        }
        assert(!vm.stack_size && !vm.frame_count && vm.heap.stats.num_objects == populated);
        for (unsigned repeat = 0; repeat < 3; repeat++) {
            assert(vm_invoke(&vm, 0, args, 3, &output) == VM_OK);
            assert(output.tag == TAG_STRING && output.as.string->length == cases[c].el);
            assert(!memcmp(output.as.string->data, cases[c].expected, cases[c].el));
            vm_release(&vm.heap, output);
            for (unsigned j = 0; j < 3; j++) assert(inputs[j]->header.ref_count == owners[j]);
            assert(vm.heap.stats.num_objects == populated);
        }
        args[0] = val_int(7);
        output = val_void();
        assert(vm_invoke(&vm, 0, args, 3, &output) == VM_ERR_TYPE_ERROR);
        assert(output.tag == TAG_VOID && !vm.stack_size && !vm.frame_count);
        for (unsigned j = 0; j < 3; j++) assert(inputs[j]->header.ref_count == owners[j]);
        for (unsigned j = 0; j < 3; j++) vm_release(&vm.heap, val_string(inputs[j]));
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
    replacement_bounds_and_recovery();
    puts("I passed ordinary string bytes, operand ownership, allocation status and recovery.");
    return 0;
}
