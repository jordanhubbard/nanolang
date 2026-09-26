#include "nanovm/vm_ffi.h"
#include "nanovm/value.h"
#include "nanoisa/nvm_format.h"
#include <assert.h>
#include <string.h>
int g_argc;
char **g_argv;

static uint32_t add(NvmModule *module, uint32_t owner, const char *name,
                    uint16_t count, uint8_t result, const uint8_t *parameters) {
    uint32_t text = nvm_add_string(module, name, (uint32_t)strlen(name));
    assert(text != UINT32_MAX);
    uint32_t index = nvm_add_import(module, owner, text, count, result, parameters);
    assert(index != UINT32_MAX);
    module->imports[index].kind = NVM_IMPORT_ARTIFACT;
    return index;
}
static NanoValue invoke(NvmModule *module, VmHeap *heap, uint32_t index,
                        NanoValue *arguments, int count) {
    NanoValue result = val_int(-999); char error[256] = {0};
    assert(vm_ffi_call(module, index, arguments, count, &result, heap, error, sizeof error));
    return result;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    NvmModule *module = nvm_module_new(); assert(module);
    uint32_t owner = nvm_add_string(module, argv[1], (uint32_t)strlen(argv[1]));
    assert(owner != UINT32_MAX);
    uint8_t integer = TAG_INT, enumeration = TAG_ENUM, boolean = TAG_BOOL;
    uint8_t byte = TAG_U8, string = TAG_STRING;
    uint8_t mixed[] = {TAG_INT, TAG_FLOAT, TAG_STRING};
    uint32_t count = add(module, owner, "scalar_calls", 0, TAG_INT, NULL);
    uint32_t i = add(module, owner, "scalar_integer", 1, TAG_INT, &integer);
    uint32_t e = add(module, owner, "scalar_enum", 1, TAG_ENUM, &enumeration);
    uint32_t b = add(module, owner, "scalar_boolean", 1, TAG_BOOL, &boolean);
    uint32_t u = add(module, owner, "scalar_byte", 1, TAG_U8, &byte);
    uint32_t s = add(module, owner, "scalar_text", 1, TAG_INT, &string);
    uint32_t m = add(module, owner, "scalar_mixed", 3, TAG_FLOAT, mixed);
    uint32_t v = add(module, owner, "scalar_void", 0, TAG_VOID, NULL);
    VmHeap heap; vm_heap_init(&heap);
    NanoValue arg = val_int(INT64_C(-4294967296));
    assert(invoke(module, &heap, i, &arg, 1).as.i64 == INT64_C(-4294967303));
    arg = val_enum(41); NanoValue result = invoke(module, &heap, e, &arg, 1);
    assert(result.tag == TAG_INT && result.as.i64 == 42);
    arg = val_int(42); assert(invoke(module, &heap, e, &arg, 1).as.i64 == 43);
    arg = val_bool(true); result = invoke(module, &heap, b, &arg, 1);
    assert(result.tag == TAG_BOOL && !result.as.boolean);
    arg = val_u8(254); result = invoke(module, &heap, u, &arg, 1);
    assert(result.tag == TAG_U8 && result.as.i64 == 255);
    VmString *text = vm_string_new(&heap, "four", 4); assert(text);
    NanoValue text_value = val_string(text);
    assert(invoke(module, &heap, s, &text_value, 1).as.i64 == 4);
    NanoValue args[] = {val_int(-8), val_float(1.5), text_value};
    result = invoke(module, &heap, m, args, 3);
    assert(result.tag == TAG_FLOAT && result.as.f64 == -2.5);
    args[1] = val_int(2); assert(invoke(module, &heap, m, args, 3).as.f64 == -2.0);
    assert(invoke(module, &heap, v, NULL, 0).tag == TAG_VOID);
    int64_t before = invoke(module, &heap, count, NULL, 0).as.i64;
    uint8_t unsupported[] = {TAG_BSTRING, TAG_STRUCT, TAG_FUNCTION};
    for (size_t n = 0; n < sizeof unsupported; ++n) {
        uint32_t bad = add(module, owner, "scalar_integer", 1, TAG_INT, &unsupported[n]);
        char error[256] = {0}; result = val_int(12345); arg = val_int(7);
        assert(!vm_ffi_call(module, bad, &arg, 1, &result, &heap, error, sizeof error));
        assert(error[0] && result.tag == TAG_INT && result.as.i64 == 12345);
        bad = add(module, owner, "scalar_integer", 1, unsupported[n], &integer);
        error[0] = 0;
        assert(!vm_ffi_call(module, bad, &arg, 1, &result, &heap, error, sizeof error));
        assert(error[0] && result.as.i64 == 12345);
    }
    uint32_t valid[] = {i, b, u, s};
    for (size_t n = 0; n < sizeof valid / sizeof valid[0]; ++n) {
        char error[256] = {0}; result = val_int(12345); arg = val_float(3.5);
        assert(!vm_ffi_call(module, valid[n], &arg, 1, &result, &heap, error, sizeof error));
        assert(error[0] && result.tag == TAG_INT && result.as.i64 == 12345);
    }
    char error[256] = {0}; result = val_int(12345);
    assert(!vm_ffi_call(module, i, NULL, 0, &result, &heap, error, sizeof error));
    assert(error[0] && result.as.i64 == 12345);
    assert(invoke(module, &heap, count, NULL, 0).as.i64 == before);
    vm_release(&heap, text_value); vm_heap_destroy(&heap);
    nvm_module_free(module); vm_ffi_shutdown();
    return 0;
}
