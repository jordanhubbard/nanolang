/* I exercise owned binary payloads through the actual VM builtin boundary. */
#include "nanovm/vm.h"
#include "nanovm/vm_ffi.h"
#include "nanovm/cop_protocol.h"
#include "nanoisa/assembler.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;

static NvmModule *module(void) {
    AsmResult error;
    NvmModule *m = asm_assemble(
        ".import \"\" \"vm_bstring_new\" bstring string\n"
        ".import \"\" \"vm_bstring_new_binary\" bstring array\n"
        ".import \"\" \"vm_bstring_length\" int bstring\n"
        ".import \"\" \"vm_bstring_concat\" bstring bstring bstring\n"
        ".import \"\" \"vm_bstring_substring\" bstring bstring int int\n"
        ".import \"\" \"vm_bstring_equals\" bool bstring bstring\n"
        ".import \"\" \"vm_bstring_byte_at\" int bstring int\n"
        ".import \"\" \"vm_bstring_validate_utf8\" bool bstring\n"
        ".import \"\" \"vm_bstring_utf8_length\" int bstring\n"
        ".import \"\" \"vm_bstring_utf8_char_at\" int bstring int\n"
        ".import \"\" \"vm_bstring_to_cstr\" string bstring\n"
        ".import \"\" \"vm_bstring_free\" void bstring\n"
        ".entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n", &error);
    if (!m) fprintf(stderr, "%s\n", error.message);
    assert(m);
    return m;
}

static NanoValue invoke(NvmModule *m, VmHeap *heap, int index, NanoValue *args, int count) {
    NanoValue result = val_void();
    char error[256] = "";
    bool ok = vm_ffi_call(m, (uint32_t)index, args, count, &result, heap, error, sizeof error);
    if (!ok) fprintf(stderr, "%s\n", error);
    assert(ok);
    return result;
}

static NanoValue text(NvmModule *m, VmHeap *heap, const char *bytes) {
    NanoValue value = val_string(vm_string_new(heap, bytes, (uint32_t)strlen(bytes)));
    assert(value.as.string);
    NanoValue result = invoke(m, heap, 0, &value, 1);
    vm_release(heap, value);
    return result;
}

static void payloads(NvmModule *m, VmHeap *heap) {
    VmArray *array = vm_array_new(heap, TAG_U8, 3);
    assert(array && vm_array_push(heap, array, val_u8(65)) &&
           vm_array_push(heap, array, val_u8(0)) && vm_array_push(heap, array, val_u8(66)));
    NanoValue bytes = val_array(array);
    NanoValue binary = invoke(m, heap, 1, &bytes, 1);
    vm_release(heap, bytes);
    assert(binary.tag == TAG_BSTRING);
    VmBString *owner = binary.as.obj;
    assert(owner->value->length == 3 && !memcmp(owner->value->data, "A\0B", 3));
    assert(invoke(m, heap, 2, &binary, 1).as.i64 == 3);
    NanoValue args[] = {binary, val_int(1), val_int(2)};
    assert(invoke(m, heap, 6, args, 2).as.i64 == 0);
    args[1] = val_int(-1);
    assert(invoke(m, heap, 6, args, 2).as.i64 == 0);
    args[1] = val_int(INT64_MAX);
    assert(invoke(m, heap, 6, args, 2).as.i64 == 0);
    args[1] = val_int(1); args[2] = val_int(INT64_MAX);
    NanoValue tail = invoke(m, heap, 4, args, 3);
    assert(((VmBString *)tail.as.obj)->value->length == 2);
    assert(!memcmp(((VmBString *)tail.as.obj)->value->data, "\0B", 2));
    NanoValue snapshot = invoke(m, heap, 10, &binary, 1);
    assert(snapshot.tag == TAG_STRING && snapshot.as.string->length == 1);
    assert(snapshot.as.string->data[0] == 'A');
    vm_retain(heap, binary);
    assert(owner->header.ref_count == 2);
    invoke(m, heap, 11, &binary, 1);
    assert(!owner->value && snapshot.as.string->data[0] == 'A');
    char error[256]; NanoValue refused;
    assert(!vm_ffi_call(m, 2, &binary, 1, &refused, heap, error, sizeof error));
    assert(strstr(error, "released"));
    vm_release(heap, binary); vm_release(heap, binary);
    vm_release(heap, tail); vm_release(heap, snapshot);
}

static void unicode_and_cycles(NvmModule *m, VmHeap *heap) {
    NanoValue value = text(m, heap, "Aé茶");
    assert(invoke(m, heap, 8, &value, 1).as.i64 == -1);
    assert(invoke(m, heap, 7, &value, 1).as.boolean);
    assert(invoke(m, heap, 8, &value, 1).as.i64 == 3);
    NanoValue args[] = {value, val_int(1)};
    assert(invoke(m, heap, 9, args, 2).as.i64 == 233);
    args[1] = val_int(2);
    assert(invoke(m, heap, 9, args, 2).as.i64 == 33590);
    NanoValue suffix = text(m, heap, "!");
    NanoValue pair[] = {value, suffix};
    NanoValue joined = invoke(m, heap, 3, pair, 2);
    invoke(m, heap, 11, &value, 1); invoke(m, heap, 11, &suffix, 1);
    vm_release(heap, value); vm_release(heap, suffix);
    assert(((VmBString *)joined.as.obj)->value->length == 7);
    VmStruct *cycle = vm_struct_new(heap, 0, 2);
    assert(cycle);
    cycle->fields[0] = val_struct(cycle); vm_retain(heap, cycle->fields[0]);
    cycle->fields[1] = joined; vm_retain(heap, joined);
    vm_release(heap, joined); vm_release(heap, val_struct(cycle));
    assert(vm_gc_collect_cycles(heap) >= 1);
}

static void refusals(NvmModule *m, VmHeap *heap) {
    NanoValue value = val_string(vm_string_new(heap, "input", 5));
    char error[256]; NanoValue out;
    NvmImportEntry original = m->imports[0];
    m->imports[0].param_count = 2;
    assert(!vm_ffi_call(m, 0, &value, 1, &out, heap, error, sizeof error));
    m->imports[0] = original;
    m->import_param_types[0][0] = TAG_INT;
    assert(!vm_ffi_call(m, 0, &value, 1, &out, heap, error, sizeof error));
    m->import_param_types[0][0] = TAG_STRING;
    m->imports[0].kind = NVM_IMPORT_ARTIFACT;
    assert(!vm_ffi_call(m, 0, &value, 1, &out, heap, error, sizeof error));
    m->imports[0] = original;
    m->imports[0].module_name_idx = nvm_add_string(m, "\0foreign", 8);
    assert(!vm_ffi_call(m, 0, &value, 1, &out, heap, error, sizeof error));
    m->imports[0] = original;
    vm_release(heap, value);
}

static void isolated_owner_dispatch(NvmModule *m) {
    VmState vm;
    vm_init(&vm, m);
    vm.isolate_ffi = true;
    NanoValue arg = val_string(vm_string_new(&vm.heap, "owned", 5));
    NanoValue result;
    char error[256];
    assert(vm_ffi_call_vm(&vm, m, 0, &arg, 1, &result, error, sizeof error));
    assert(result.tag == TAG_BSTRING && vm.cop_pid < 0);
    assert(!vm_ffi_call_cop(&vm, m, 0, &arg, 1, &result, &vm.heap, error, sizeof error));
    assert(strstr(error, "owner-heap") && vm.cop_pid < 0);
    CopBatchCall call = {.import_idx = 0, .arg_count = 1, .args = &arg};
    NanoValue batch_result;
    assert(!vm_ffi_call_cop_batch(&vm, m, &call, 1, &batch_result, &vm.heap, error, sizeof error));
    assert(strstr(error, "owner-heap") && vm.cop_pid < 0);
    vm_release(&vm.heap, arg); vm_release(&vm.heap, result);
    vm_destroy(&vm);
}

int main(void) {
    NvmModule *m = module();
    VmHeap heap; vm_heap_init(&heap);
    payloads(m, &heap); unicode_and_cycles(m, &heap); refusals(m, &heap);
    assert(heap.stats.num_objects == 0);
    assert(heap.stats.allocated == heap.stats.freed);
    vm_heap_destroy(&heap);
    isolated_owner_dispatch(m);
    nvm_module_free(m);
    vm_ffi_shutdown();
    puts("I preserve binary bytes, validation state, aliases, cycles and exact VM ownership dispatch.");
    return 0;
}
