/* I inject only the converted-string heap allocation after VM/input setup. */
#include "nanovm/vm.h"
#include "nanoisa/assembler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;
static unsigned checks, attempts, fail_at;
#define CHECK(condition) do { checks++; if (!(condition)) { \
    fprintf(stderr, "I failed %s at line %d\n", #condition, __LINE__); exit(1); \
} } while (0)

void *cast_heap_malloc(size_t size) {
    if (++attempts == fail_at) return NULL;
    return malloc(size);
}

static void exercise(NvmModule *module, NanoValue input, const char *expected) {
    VmState vm;
    fail_at = 0;
    vm_init(&vm, module);
    size_t initial_objects = vm.heap.stats.num_objects;
    if (input.tag == TAG_ARRAY) {
        input = val_array(vm_array_new(&vm.heap, TAG_INT, 0));
        CHECK(input.as.array != NULL);
    } else if (input.tag == TAG_STRING) {
        input = val_string(vm_string_new(&vm.heap, expected, (uint32_t)strlen(expected)));
        CHECK(input.as.string != NULL);
    }
    size_t baseline = vm.heap.stats.num_objects;
    attempts = 0;
    fail_at = 1;
    NanoValue result = val_void();
    VmResult status = vm_invoke(&vm, 0, &input, 1, &result);
    fail_at = 0;
    CHECK(vm.stack_size == 0 && vm.frame_count == 0);
    if (input.tag == TAG_STRING) {
        CHECK(status == VM_OK && attempts == 0);
        CHECK(result.tag == TAG_STRING && result.as.string == input.as.string);
        vm_release(&vm.heap, result);
    } else {
        CHECK(status == VM_ERR_MEMORY);
        CHECK(attempts == 1);
        CHECK(result.tag == TAG_VOID);
    }
    vm_gc_collect_cycles(&vm.heap);
    CHECK(vm.heap.stats.num_objects == baseline);
    if (input.tag == TAG_ARRAY) CHECK(input.as.array->length == 0);

    /* I can invoke again after the failed activation without losing input. */
    result = val_void();
    CHECK(vm_invoke(&vm, 0, &input, 1, &result) == VM_OK);
    CHECK(result.tag == TAG_STRING && result.as.string != NULL);
    CHECK(strcmp(vmstring_cstr(result.as.string), expected) == 0);
    CHECK(vm.stack_size == 0 && vm.frame_count == 0);
    vm_release(&vm.heap, result);
    vm_gc_collect_cycles(&vm.heap);
    CHECK(vm.heap.stats.num_objects == baseline);
    vm_release(&vm.heap, input);
    vm_gc_collect_cycles(&vm.heap);
    CHECK(vm.heap.stats.num_objects == initial_objects);
    vm_destroy(&vm);
}

int main(void) {
    AsmResult assembled;
    NvmModule *module = asm_assemble(
        ".entry convert\n.function convert 1 1 0 string 1\n"
        "LOAD_LOCAL 0\nCAST_STRING\nRET\n.end\n", &assembled);
    if (!module) fprintf(stderr, "%s\n", assembled.message);
    CHECK(module != NULL);
    exercise(module, val_int(-9223372036854775807LL - 1), "-9223372036854775808");
    exercise(module, val_float(3.25), "3.25");
    exercise(module, val_bool(true), "true");
    exercise(module, val_bool(false), "false");
    exercise(module, val_u8(255), "255");
    exercise(module, val_void(), "");
    exercise(module, val_array(NULL), "");
    exercise(module, val_string(NULL), "retained");
    nvm_module_free(module);
    printf("I passed %u CAST_STRING allocation and cleanup checks.\n", checks);
    return 0;
}
