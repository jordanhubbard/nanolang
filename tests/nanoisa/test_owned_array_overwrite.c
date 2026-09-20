/* I supplement distinct-origin replacement without changing the original corpus. */
#define NANO_OWNER_ARRAY_RUNTIME_MAIN retained_overwrite_runtime_main
#include "test_private_owned_array_runtime.c"
static VmState *overwrite_vm;
static uint32_t replacement_pc;
static unsigned replacement_failures;
static void replacement_fault(void) {
    if (!overwrite_vm || overwrite_vm->current_fn || overwrite_vm->ip != replacement_pc) return;
    CHECK(overwrite_vm->frame_count == 1 && overwrite_vm->stack_size >= 6);
    NanoValue a = overwrite_vm->stack[0], alias = overwrite_vm->stack[1];
    NanoValue owner = overwrite_vm->stack[2];
    CHECK(a.tag == TAG_ARRAY && alias.tag == TAG_ARRAY && a.as.array == alias.as.array);
    CHECK(owner.tag == TAG_STRUCT && owner.as.sval->field_count == 3);
    NanoValue field = owner.as.sval->fields[1];
    CHECK(field.tag == TAG_ARRAY && field.as.array == a.as.array);
    NanoValue item = vm_array_get(a.as.array, 0);
    CHECK(item.tag == TAG_FLOAT && item.as.f64 == 1.5);
    CHECK(a.as.array->header.ref_count >= 3);
    replacement_failures++;
    fprintf(stderr, "overwrite replacement allocation refused before local publication pc=%u\n", replacement_pc);
}
void *overwrite_malloc(size_t bytes) {
    if (bytes && ++heap_attempts == heap_fail) {heap_hits++; replacement_fault(); return NULL;}
    return malloc(bytes);
}
void *overwrite_calloc(size_t count, size_t bytes) {
    if (count && bytes && ++heap_attempts == heap_fail) {heap_hits++; replacement_fault(); return NULL;}
    return calloc(count, bytes);
}
void *overwrite_realloc(void *pointer, size_t bytes) {
    if (bytes && ++heap_attempts == heap_fail) {heap_hits++; replacement_fault(); return NULL;}
    return realloc(pointer, bytes);
}
static NvmModule *overwrite_module(bool assertion) {
    char body[8192];
    int length = snprintf(body, sizeof body,
        "PUSH_F64 1.5\nARR_LITERAL 3 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 1\n"
        "PUSH_I64 7\nOWN_PACK 0\nLOAD_LOCAL 0\nPUSH_STR value\nOWN_PACK 1\nOWN_STORE_LOCAL 2\n"
        "PUSH_I64 11\nPRINTLN\nPUSH_F64 9.5\nARR_LITERAL 3 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 9.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 1.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 2\nAGG_GET 1\nPUSH_I64 0\nPUSH_F64 2.5\nARR_SET\nPOP\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 9.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nPUSH_F64 7.5\nARR_SET\nPOP\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\n"
        "PUSH_I64 12\nPRINTLN\n%s"
        "OWN_UNPACK_LOCAL 2\nPOP\nSTORE_LOCAL 4\nOWN_STORE_LOCAL 3\n"
        "OWN_UNPACK_LOCAL 3\nPUSH_I64 7\nI64_EQ\nASSERT\n"
        "ARR_NEW 3\nSTORE_LOCAL 4\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nPUSH_F64 2.5\nF64_EQ\nASSERT\n"
        "PUSH_I64 0\nSTORE_LOCAL 5\nagain:\nLOAD_LOCAL 5\nPUSH_I64 12\nI64_LT_S\nJMP_FALSE done\n"
        "LOAD_LOCAL 1\nPUSH_F64 3.5\nARR_PUSH\nPOP\nLOAD_LOCAL 5\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 5\nJMP again\n"
        "done:\nLOAD_LOCAL 1\nARR_LEN\nPUSH_I64 13\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_F64 7.5\nF64_EQ\nASSERT\n"
        "LOAD_LOCAL 4\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "PUSH_I64 13\nPRINTLN\nPUSH_I64 0\nRET\n", assertion ? "PUSH_BOOL 0\nASSERT\n" : "");
    CHECK(length > 0 && (size_t)length < sizeof body);
    Function function = {body,0,6,T(TAG_INT),{T(TAG_ARRAY),T(TAG_ARRAY),OWNER(1),OWNER(0),T(TAG_ARRAY),T(TAG_INT)}};
    NvmModule *module = build(&function, 1);
    NvmOwnedArrayPlan *plan = NULL;
    NvmOwnerAuthorityResult result = nvm_owned_array_admit(module, &plan);
    if (result.status != NVM_OWNER_AUTH_PREPARED) fprintf(stderr, "overwrite admission: %s\n", result.message);
    CHECK(result.status == NVM_OWNER_AUTH_PREPARED && nvm_verify(module).ok);
    nvm_owned_array_plan_free(plan);
    unsigned literals = 0;
    replacement_pc = UINT32_MAX;
    for (uint32_t pc = 0; pc < module->functions[0].code_length;) {
        DecodedInstruction in;
        uint32_t width = isa_decode(module->code + pc, module->functions[0].code_length - pc, &in);
        CHECK(width);
        if (in.opcode == OP_ARR_LITERAL && ++literals == 2) replacement_pc = pc + width;
        pc += width;
    }
    CHECK(literals == 2 && replacement_pc != UINT32_MAX);
    return module;
}
static bool overwrite_run(VmState *vm, bool assertion, unsigned fault, size_t roots, size_t bytes) {
    FILE *output = tmpfile(); CHECK(output); vm->output = output;
    heap_attempts = heap_hits = 0; heap_fail = fault; overwrite_vm = vm;
    NanoValue value = val_int(-91);
    VmResult result = runtime_entry(vm, &value);
    overwrite_vm = NULL; heap_fail = 0;
    bool hit = heap_hits != 0;
    CHECK(hit ? (heap_hits == 1 && result == VM_ERR_MEMORY) : result == (assertion ? VM_ERR_ASSERT_FAILED : VM_OK));
    CHECK(result == VM_OK ? (value.tag == TAG_INT && value.as.i64 == 0) : runtime_failure_value(value));
    CHECK(!fflush(output)); long count = ftell(output); rewind(output);
    char text[32] = {0}; CHECK(count >= 0 && count < (long)sizeof text);
    CHECK(fread(text, 1, (size_t)count, output) == (size_t)count);
    CHECK(!fclose(output)); vm->output = NULL;
    CHECK(hit ? (!strcmp(text, "") || !strcmp(text, "11\n") || !strcmp(text, "11\n12\n")) :
          !strcmp(text, assertion ? "11\n12\n" : "11\n12\n13\n"));
    fprintf(stderr, "overwrite api=%u assertion=%u fault=%u hits=%u status=%d allocations=%u prefix=%ld\n",
            public_api, assertion, fault, heap_hits, result, heap_attempts, count);
    clean(vm, roots, bytes);
    return hit;
}
int main(int argc, char **argv) {
    CHECK(argc == 2);
    for (unsigned assertion = 0; assertion < 2; assertion++) {
        NvmModule *module = overwrite_module(assertion != 0);
        char error[256]; char *source = nvm2c_emit(module, error, sizeof error); CHECK(source);
        char path[1024]; int length = snprintf(path, sizeof path, "%s/overwrite%u.c", argv[1], assertion);
        CHECK(length > 0 && (size_t)length < sizeof path);
        FILE *file = fopen(path, "wb"); CHECK(file);
        CHECK(fwrite(source, 1, strlen(source), file) == strlen(source)); CHECK(!fclose(file)); free(source);
        for (public_api = 0; public_api < 4; public_api++) for (unsigned fused = 0; fused < 2; fused++) {
            VmState vm; vm_init(&vm, module); CHECK(vm.last_error == VM_OK);
            VmDispatchProfile profile = {.fuse_load_local_field = fused}; vm_set_dispatch_profile(&vm, profile); CHECK(vm.dispatch_module_valid);
            size_t roots = vm.heap.stats.num_objects, bytes = vm.heap.stats.allocated - vm.heap.stats.freed;
            replacement_failures = 0;
            CHECK(!overwrite_run(&vm, assertion != 0, 0, roots, bytes));
            CHECK(!overwrite_run(&vm, assertion != 0, 0, roots, bytes));
            bool done = false;
            for (unsigned fault = 1; fault < 128; fault++) {
                if (!overwrite_run(&vm, assertion != 0, fault, roots, bytes)) {done = true; break;}
                CHECK(!overwrite_run(&vm, assertion != 0, 0, roots, bytes));
            }
            CHECK(done && replacement_failures > 0);
            vm_destroy(&vm); CHECK(!vm.heap.stats.num_objects);
        }
        nvm_module_free(module);
    }
    printf("%u distinct-array overwrite checks passed\n", checks);
    return 0;
}
