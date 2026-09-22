/* I test exact token membership and real artifact calls on all COP transports. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "../../src/nanovm/vm_ffi.h"
#include "../../src/nanovm/cop_protocol.h"
static size_t allocation_position, fail_at;
static void *fault_malloc(size_t size) {
    ++allocation_position;
    return fail_at && allocation_position == fail_at ? NULL : malloc(size);
}
#define malloc fault_malloc
#include "../../src/nanovm/cop_opaque.c"
#undef malloc
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

static NanoValue wire(uint64_t slot) {
    NanoValue value = val_opaque(NULL);
    value.as.i64 = (int64_t)slot;
    return value;
}
static void metadata_controls(void) {
    _Static_assert(sizeof(NanoValue) == 16, "I preserve the value carrier size");
    CopOpaqueOwner owner = {0};
    CopOpaqueWorker worker = {0};
    int first, second;
    assert(cop_opaque_owner_start(&owner));
    assert(cop_opaque_owner_reserve(&owner, 2));
    assert(cop_opaque_worker_reserve(&worker, 2, 32));
    uint64_t one, two, repeated;
    assert(cop_opaque_worker_capture(&worker, &first, &one) && one == 1);
    /* A failed/unpublished first reply leaves metadata but no parent authority. */
    NanoValue ignored, published;
    assert(cop_opaque_owner_preview(&owner, wire(one), &ignored));
    assert(!cop_opaque_owner_argument(&owner, ignored));
    assert(cop_opaque_worker_capture(&worker, &second, &two) && two == 2);
    assert(cop_opaque_owner_preview(&owner, wire(two), &published));
    assert(cop_opaque_owner_publish(&owner, published));
    assert(cop_opaque_owner_argument(&owner, published));
    assert(!cop_opaque_owner_argument(&owner, ignored));
    assert(cop_opaque_worker_capture(&worker, &second, &repeated) && repeated == two);
    NanoValue local;
    assert(cop_opaque_worker_argument(&worker, wire(two), &local));
    assert(local.opaque_owner == 0 && local.as.obj == &second);
    assert(!cop_opaque_worker_argument(&worker, published, &local));
    assert(!cop_opaque_worker_argument(&worker, wire(3), &local));
    assert(!cop_opaque_owner_argument(&owner, val_opaque(&first)));
    assert(cop_opaque_owner_argument(&owner, val_opaque(NULL)));
    NanoValue wide = wire(UINT64_C(1) << 32), narrow = wire(0);
    assert(!val_equal(wide, narrow));
    assert(val_compare(wide, narrow) > 0);
    NanoValue foreign = published; ++foreign.opaque_owner;
    assert(!val_equal(published, foreign));
    assert(!cop_opaque_owner_publish(&owner, foreign));
    VmHeap heap;
    vm_heap_init(&heap);
    CopOpaqueOwner reply_owner = {0};
    assert(cop_opaque_owner_start(&reply_owner));
    assert(cop_opaque_owner_reserve(&reply_owner, 2));
    uint8_t reply[32];
    NanoValue result = val_int(919), saved, first_wire = wire(one);
    memcpy(&saved, &result, sizeof saved);
    uint32_t length = cop_encode_call_values(&first_wire, 1, reply, sizeof reply);
    assert(length);
    assert(!cop_apply_call_reply_owned(reply, length - 1, NULL, 0, &result,
                                      &heap, &reply_owner, TAG_OPAQUE));
    assert(!memcmp(&saved, &result, sizeof result) && reply_owner.count == 0);
    NanoValue second_wire = wire(two);
    length = cop_encode_call_values(&second_wire, 1, reply, sizeof reply);
    assert(cop_apply_call_reply_owned(reply, length, NULL, 0, &result,
                                     &heap, &reply_owner, TAG_OPAQUE));
    assert(result.as.i64 == 2 && cop_opaque_owner_argument(&reply_owner, result));
    ignored.opaque_owner = reply_owner.generation;
    assert(!cop_opaque_owner_argument(&reply_owner, ignored));
    cop_opaque_owner_clear(&reply_owner);
    vm_heap_destroy(&heap);
    NanoValue old = published;
    cop_opaque_owner_clear(&owner);
    assert(cop_opaque_owner_start(&owner));
    assert(!cop_opaque_owner_argument(&owner, old));
    cop_opaque_owner_clear(&owner);
    cop_opaque_worker_clear(&worker);
    for (size_t position = 1; position <= 2; ++position) {
        allocation_position = 0; fail_at = position;
        assert(!cop_opaque_worker_reserve(&worker, 1, 32));
        assert(worker.count == 0);
        fail_at = 0;
        assert(cop_opaque_worker_reserve(&worker, 1, 32));
        assert(cop_opaque_worker_capture(&worker, &first, &one) && one == 1);
        cop_opaque_worker_clear(&worker);
    }
    assert(cop_opaque_owner_start(&owner));
    allocation_position = 0; fail_at = 1;
    assert(!cop_opaque_owner_reserve(&owner, 1) && owner.count == 0);
    fail_at = 0;
    assert(cop_opaque_owner_reserve(&owner, 1));
    allocation_position = 0; fail_at = 1;
    assert(!cop_opaque_owner_reply(&owner, 32) && owner.reply == NULL);
    fail_at = 0;
    assert(cop_opaque_owner_reply(&owner, 32));
    assert(!cop_opaque_owner_reserve(&owner, SIZE_MAX));
    assert(!cop_opaque_worker_reserve(&worker, SIZE_MAX, 32));
    cop_opaque_owner_clear(&owner);
    uint32_t saved_generation = last_generation;
    last_generation = UINT32_MAX;
    assert(!cop_opaque_owner_start(&owner) && !owner.generation);
    last_generation = saved_generation;
}
static VmState *isolated(void) {
    VmState *vm = calloc(1, sizeof *vm);
    assert(vm);
    vm->cop_pid = vm->cop_in_fd = vm->cop_out_fd = -1;
    vm->cop_sig_send_fd = vm->cop_sig_recv_fd = -1;
    vm->cop_timeout_ms = 5000;
    vm->isolate_ffi = true;
    return vm;
}
static NanoValue call(VmState *vm, NvmModule *module, VmHeap *heap,
                      uint32_t index, NanoValue *args, int count) {
    NanoValue out = val_int(919);
    char error[256] = {0};
    bool ok = vm ? vm_ffi_call_cop(vm, module, index, args, count, &out, heap, error, sizeof error)
                 : vm_ffi_call(module, index, args, count, &out, heap, error, sizeof error);
    if (!ok) fprintf(stderr, "%s\n", error);
    assert(ok);
    return out;
}
static void refused(VmState *vm, NvmModule *module, VmHeap *heap, NanoValue value) {
    NanoValue out = val_int(919), saved;
    memcpy(&saved, &out, sizeof saved);
    char error[256] = {0};
    int64_t before = call(vm, module, heap, 4, NULL, 0).as.i64;
    assert(!vm_ffi_call_cop(vm, module, 2, &value, 1, &out, heap, error, sizeof error));
    assert(!memcmp(&out, &saved, sizeof out));
    assert(call(vm, module, heap, 4, NULL, 0).as.i64 == before);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    metadata_controls();
    NvmModule *module = nvm_module_new();
    assert(module);
    uint32_t owner = nvm_add_string(module, argv[1], (uint32_t)strlen(argv[1]));
    const char *names[] = {"opaque_make", "opaque_same", "opaque_read", "opaque_large", "opaque_calls", "opaque_invalid_array"};
    uint8_t arguments[] = {TAG_INT, TAG_OPAQUE, TAG_OPAQUE, TAG_STRING, TAG_VOID, TAG_ARRAY};
    uint8_t returns[] = {TAG_OPAQUE, TAG_OPAQUE, TAG_INT, TAG_OPAQUE, TAG_INT, TAG_OPAQUE};
    for (int i = 0; i < 6; ++i) {
        uint32_t symbol = nvm_add_string(module, names[i], (uint32_t)strlen(names[i]));
        nvm_add_import(module, owner, symbol, i == 4 ? 0 : 1, returns[i], &arguments[i]);
        module->imports[i].kind = NVM_IMPORT_ARTIFACT;
    }
    VmHeap heap;
    vm_heap_init(&heap);
    NanoValue zero = val_int(0);
    NanoValue local = call(NULL, module, &heap, 0, &zero, 1);
    assert(local.tag == TAG_OPAQUE && !local.opaque_owner && local.as.obj);
    assert(call(NULL, module, &heap, 2, &local, 1).as.i64 == 41);
    VmState *failed = isolated();
    NanoValue array = val_array(vm_array_new(&heap, TAG_INT, 1));
    assert(array.as.array && vm_array_push(&heap, array.as.array, val_int(7)));
    NanoValue failed_out = val_int(919), failed_before;
    memcpy(&failed_before, &failed_out, sizeof failed_before);
    char failed_error[256] = {0};
    assert(!vm_ffi_call_cop(failed, module, 5, &array, 1, &failed_out,
                            &heap, failed_error, sizeof failed_error));
    assert(strstr(failed_error, "array"));
    assert(!memcmp(&failed_out, &failed_before, sizeof failed_out));
    assert(vm_array_get(array.as.array, 0).as.i64 == 7);
    assert(failed->cop_opaque.count == 0);
    NanoValue later = call(failed, module, &heap, 0, &zero, 1);
    assert(later.as.i64 == 2 && failed->cop_opaque.count == 1);
    NanoValue hole = later; hole.as.i64 = 1;
    refused(failed, module, &heap, hole);
    NanoValue index_one = val_int(1);
    NanoValue recovered = call(failed, module, &heap, 0, &index_one, 1);
    assert(recovered.as.i64 == 1 && val_equal(hole, recovered));
    assert(call(failed, module, &heap, 2, &recovered, 1).as.i64 == 42);
    vm_release(&heap, array);
    vm_ffi_cop_stop(failed); free(failed);
    VmState *a = isolated(), *b = isolated();
    NanoValue first = call(a, module, &heap, 0, &zero, 1);
    assert(first.tag == TAG_OPAQUE && first.opaque_owner);
    assert(call(a, module, &heap, 2, &first, 1).as.i64 == 41);
    assert(val_equal(first, call(a, module, &heap, 1, &first, 1)));
    NanoValue second = call(b, module, &heap, 0, &zero, 1);
    assert(first.as.i64 == second.as.i64 && !val_equal(first, second));
    refused(b, module, &heap, first);
    refused(a, module, &heap, local);
    refused(a, module, &heap, val_int(1));
    NanoValue never = first; never.as.i64 += 100;
    refused(a, module, &heap, never);
    NanoValue output = val_int(919);
    char error[256] = {0};
    assert(!vm_ffi_call(module, 2, &first, 1, &output, &heap, error, sizeof error));
    assert(output.tag == TAG_INT && output.as.i64 == 919);
    assert(!vm_ffi_call_vm(a, module, 2, &first, 1, &output, error, sizeof error));
    assert(output.tag == TAG_INT && output.as.i64 == 919);
    NanoValue negative = val_int(-1);
    NanoValue null = call(a, module, &heap, 0, &negative, 1);
    assert(null.tag == TAG_OPAQUE && !null.opaque_owner && !null.as.i64);
    assert(call(a, module, &heap, 2, &null, 1).as.i64 == -1);
    assert(call(a, module, &heap, 2, &zero, 1).as.i64 == -1);
    CopBatchCall batch[] = {{1, &first, 1}, {1, &first, 1}, {0, &zero, 1}};
    NanoValue results[3];
    assert(vm_ffi_call_cop_batch(a, module, batch, 3, results, &heap, error, sizeof error));
    for (int i = 0; i < 3; ++i) assert(val_equal(results[i], first));
    const uint32_t large_length = COP_MAILBOX_SLOT_SIZE + 8193;
    char *bytes = malloc((size_t)large_length + 1);
    assert(bytes);
    memset(bytes, 'x', large_length); bytes[large_length] = 0;
    NanoValue large = val_string(vm_string_new(&heap, bytes, large_length));
    free(bytes);
    assert(large.as.string);
    NanoValue pipe_token = call(a, module, &heap, 3, &large, 1);
    assert(pipe_token.opaque_owner == first.opaque_owner);
    assert(call(a, module, &heap, 2, &pipe_token, 1).as.i64 == 43);
    vm_release(&heap, large);
    vm_ffi_cop_stop(a);
    NanoValue restarted = call(a, module, &heap, 0, &zero, 1);
    assert(restarted.opaque_owner != first.opaque_owner);
    refused(a, module, &heap, first);
    assert(call(a, module, &heap, 2, &restarted, 1).as.i64 == 41);
    vm_ffi_cop_stop(a); vm_ffi_cop_stop(b);
    free(a); free(b);
    vm_ffi_shutdown();
    vm_heap_destroy(&heap);
    nvm_module_free(module);
    puts("I checked opaque membership, allocation refusal and all three isolated transports.");
    return 0;
}
