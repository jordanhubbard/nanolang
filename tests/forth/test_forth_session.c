/*
 * test_forth_session.c — Forth session runtime: one NvmModule, one VmState,
 * Forth stacks distinct from the NanoVM operand stack, virtual address space,
 * and validated file handles.
 */

#include "forth/forth_session.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanoisa/verifier.h"
#include "nanovm/value.h"
#include "nanovm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <unistd.h>

int g_argc = 0;
char **g_argv = NULL;

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

static uint32_t emit(uint8_t *buf, NanoOpcode op, ...) {
    DecodedInstruction instr;
    const InstructionInfo *info;
    va_list args;
    int i;

    memset(&instr, 0, sizeof(instr));
    instr.opcode = op;
    info = isa_get_info(op);
    if (!info) return 0;
    va_start(args, op);
    for (i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_U8:  instr.operands[i].u8  = (uint8_t)va_arg(args, int);      break;
            case OPERAND_U16: instr.operands[i].u16 = (uint16_t)va_arg(args, int);     break;
            case OPERAND_U32: instr.operands[i].u32 = va_arg(args, uint32_t);          break;
            case OPERAND_I32: instr.operands[i].i32 = va_arg(args, int32_t);           break;
            case OPERAND_I64: instr.operands[i].i64 = va_arg(args, int64_t);           break;
            case OPERAND_F64: instr.operands[i].f64 = va_arg(args, double);            break;
            default: break;
        }
    }
    va_end(args);
    return isa_encode(&instr, buf, 64);
}

static uint32_t publish_i64_const(ForthSession *session, const char *name, int64_t value) {
    uint8_t code[32];
    uint32_t off = 0;
    uint32_t name_idx;
    uint32_t code_off;
    NvmFunctionEntry fn;
    NvmModule *mod;
    NvmVerifyResult verified;

    memset(&fn, 0, sizeof(fn));
    off += emit(code + off, OP_PUSH_I64, value);
    off += emit(code + off, OP_RET);
    mod = forth_session_module(session);
    name_idx = nvm_add_string(mod, name, (uint32_t)strlen(name));
    code_off = nvm_append_code(mod, code, off);
    fn.name_idx = name_idx;
    fn.arity = 0;
    fn.code_offset = code_off;
    fn.code_length = off;
    fn.local_count = 0;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    nvm_add_function(mod, &fn);
    verified = nvm_verify_function(mod, mod->function_count - 1);
    if (!verified.ok) return UINT32_MAX;
    if (!forth_session_rebuild(session)) return UINT32_MAX;
    return mod->function_count - 1;
}

static void test_session_owns_module_and_vm(void) {
    const char *test_name = "session: one mutable module and persistent VM";
    ForthSession *session = forth_session_create();
    NvmModule *mod;
    VmState *vm;
    NvmModule *mod_again;
    VmState *vm_again;

    ASSERT(session != NULL, "create failed");
    mod = forth_session_module(session);
    vm = forth_session_vm(session);
    ASSERT(mod != NULL, "session has no module");
    ASSERT(vm != NULL, "session has no VM");
    ASSERT(vm->module == mod, "VM must execute the session module");
    ASSERT(vm->decoded_module_valid, "session VM must decode the module");

    mod_again = forth_session_module(session);
    vm_again = forth_session_vm(session);
    ASSERT(mod_again == mod, "module pointer must be stable");
    ASSERT(vm_again == vm, "VmState pointer must be stable");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_null_session_is_rejected(void) {
    const char *test_name = "session: NULL session is rejected";
    int64_t cell = 0;
    double fp = 0.0;
    uint32_t value = 0;
    ForthCtrlKind kind = FORTH_CTRL_ORIG;
    uint64_t addr = 0;
    uint8_t byte = 0;
    uint32_t fileid = 0;

    ASSERT(forth_session_module(NULL) == NULL, "NULL module");
    ASSERT(forth_session_vm(NULL) == NULL, "NULL VM");
    ASSERT(!forth_session_rebuild(NULL), "NULL rebuild");
    ASSERT(forth_session_invoke(NULL, 0, NULL, 0, NULL) != VM_OK, "NULL invoke");
    ASSERT(!forth_data_push(NULL, 1), "NULL data push");
    ASSERT(!forth_data_pop(NULL, &cell), "NULL data pop");
    ASSERT(forth_data_depth(NULL) == 0, "NULL data depth");
    ASSERT(!forth_return_push(NULL, 1), "NULL return push");
    ASSERT(!forth_return_pop(NULL, &cell), "NULL return pop");
    ASSERT(!forth_float_push(NULL, 1.0), "NULL float push");
    ASSERT(!forth_float_pop(NULL, &fp), "NULL float pop");
    ASSERT(!forth_control_push(NULL, FORTH_CTRL_ORIG, 1), "NULL control push");
    ASSERT(!forth_control_pop(NULL, &kind, &value), "NULL control pop");
    ASSERT(!forth_allocate(NULL, 8, &addr), "NULL allocate");
    ASSERT(!forth_free(NULL, 8), "NULL free");
    ASSERT(!forth_store_cell(NULL, 8, 1), "NULL store");
    ASSERT(!forth_fetch_cell(NULL, 8, &cell), "NULL fetch");
    ASSERT(!forth_store_byte(NULL, 8, 1), "NULL store byte");
    ASSERT(!forth_fetch_byte(NULL, 8, &byte), "NULL fetch byte");
    ASSERT(!forth_file_open(NULL, "/tmp/x", "w", &fileid), "NULL file open");
    ASSERT(!forth_file_close(NULL, 1), "NULL file close");
    ASSERT(!forth_file_is_open(NULL, 1), "NULL file is_open");
    PASS(test_name);
}

static void test_forth_stacks_are_not_operand_stack(void) {
    const char *test_name = "stacks: Forth stacks survive vm_invoke";
    ForthSession *session = forth_session_create();
    VmState *vm;
    uint32_t fn;
    NanoValue result;
    int64_t cell = 0;
    double fp = 0.0;
    ForthCtrlKind kind = FORTH_CTRL_DEST;
    uint32_t ctrl = 0;
    uint32_t i;

    ASSERT(session != NULL, "create failed");
    vm = forth_session_vm(session);
    ASSERT(forth_data_push(session, 7), "data push");
    ASSERT(forth_return_push(session, 11), "return push");
    ASSERT(forth_float_push(session, 0.5), "float push");
    ASSERT(forth_control_push(session, FORTH_CTRL_ORIG, 99), "control push");
    ASSERT(forth_data_depth(session) == 1, "data depth");
    ASSERT(forth_return_depth(session) == 1, "return depth");
    ASSERT(forth_float_depth(session) == 1, "float depth");
    ASSERT(forth_control_depth(session) == 1, "control depth");

    fn = publish_i64_const(session, "forty_two", (int64_t)42);
    ASSERT(fn != UINT32_MAX, "publish failed");
    ASSERT(forth_session_module(session) == vm->module, "module still bound");

    result = val_void();
    ASSERT(forth_session_invoke(session, fn, NULL, 0, &result) == VM_OK,
           "first invoke");
    ASSERT(result.tag == TAG_INT && result.as.i64 == 42, "first result");
    ASSERT(vm->stack_size == 0, "operand stack must be empty after invoke");
    ASSERT(vm->frame_count == 0, "call frames must be empty after invoke");
    ASSERT(forth_data_depth(session) == 1, "data stack must survive invoke");
    ASSERT(forth_return_depth(session) == 1, "return stack must survive invoke");
    ASSERT(forth_float_depth(session) == 1, "float stack must survive invoke");
    ASSERT(forth_control_depth(session) == 1, "control stack must survive invoke");

    result = val_void();
    ASSERT(forth_session_invoke(session, fn, NULL, 0, &result) == VM_OK,
           "second invoke");
    ASSERT(result.as.i64 == 42, "second result");
    ASSERT(forth_data_pop(session, &cell) && cell == 7, "data cell preserved");
    ASSERT(forth_return_pop(session, &cell) && cell == 11, "return cell preserved");
    ASSERT(forth_float_pop(session, &fp) && fp == 0.5, "float preserved");
    ASSERT(forth_control_pop(session, &kind, &ctrl) && kind == FORTH_CTRL_ORIG
           && ctrl == 99, "control item preserved");
    ASSERT(forth_data_depth(session) == 0, "data empty after pop");
    ASSERT(!forth_data_pop(session, &cell), "data underflow");
    ASSERT(!forth_return_pop(session, &cell), "return underflow");
    ASSERT(!forth_float_pop(session, &fp), "float underflow");
    ASSERT(!forth_control_pop(session, &kind, &ctrl), "control underflow");

    for (i = 0; i < FORTH_STACK_CELLS; i++) {
        ASSERT(forth_data_push(session, (int64_t)i), "fill data stack");
    }
    ASSERT(!forth_data_push(session, 1), "data overflow");
    ASSERT(forth_data_depth(session) == FORTH_STACK_CELLS, "data at limit");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_stale_decode_requires_rebuild(void) {
    const char *test_name = "session: mutation requires rebuild before invoke";
    ForthSession *session = forth_session_create();
    uint8_t code[32];
    uint32_t off = 0;
    NvmFunctionEntry fn;
    NvmModule *mod;
    NanoValue result;
    uint32_t idx;

    ASSERT(session != NULL, "create failed");
    memset(&fn, 0, sizeof(fn));
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_RET);
    mod = forth_session_module(session);
    fn.name_idx = nvm_add_string(mod, "one", 3);
    fn.code_offset = nvm_append_code(mod, code, off);
    fn.code_length = off;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    idx = nvm_add_function(mod, &fn);
    result = val_void();
    ASSERT(forth_session_invoke(session, idx, NULL, 0, &result) == VM_ERR_DECODE,
           "stale decode must not execute new code");
    ASSERT(forth_session_rebuild(session), "rebuild after append");
    result = val_void();
    ASSERT(forth_session_invoke(session, idx, NULL, 0, &result) == VM_OK,
           "rebuilt function executes");
    ASSERT(result.as.i64 == 1, "rebuilt operand");
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_address_space_and_allocation(void) {
    const char *test_name = "memory: virtual addresses, not host pointers";
    ForthSession *session = forth_session_create();
    uint64_t a = 0, b = 0;
    int64_t cell = 0;
    uint8_t byte = 0;
    uint64_t hostish;
    int stack_cell = 0;

    ASSERT(session != NULL, "create failed");
    ASSERT(!forth_allocate(session, 0, &a), "ALLOCATE 0 is rejected");
    ASSERT(forth_allocate(session, 16, &a), "allocate 16");
    ASSERT(a != 0, "address 0 is not a valid allocation");
    ASSERT((a % FORTH_CELL_BYTES) == 0, "cell alignment");
    ASSERT(forth_store_cell(session, a, (int64_t)1234567890123LL), "store cell");
    ASSERT(forth_fetch_cell(session, a, &cell) && cell == 1234567890123LL,
           "fetch cell");
    ASSERT(forth_store_byte(session, a + 8, (uint8_t)0xAB), "store byte");
    ASSERT(forth_fetch_byte(session, a + 8, &byte) && byte == 0xAB, "fetch byte");
    ASSERT(!forth_store_cell(session, a + 1, 1), "unaligned cell store");
    ASSERT(!forth_fetch_cell(session, a + 1, &cell), "unaligned cell fetch");
    ASSERT(!forth_store_cell(session, 0, 1), "address 0 store");
    ASSERT(!forth_fetch_cell(session, 0, &cell), "address 0 fetch");

    ASSERT(forth_allocate(session, 8, &b), "second allocation");
    ASSERT(b != a, "allocations must not alias");
    ASSERT(b >= a + 16 || a >= b + 8, "regions must not overlap");
    ASSERT(forth_store_cell(session, b, 99), "store second");
    ASSERT(forth_fetch_cell(session, a, &cell) && cell == 1234567890123LL,
           "first region intact");

    ASSERT(forth_free(session, a), "free first");
    ASSERT(!forth_fetch_cell(session, a, &cell), "fetch after free");
    ASSERT(!forth_store_cell(session, a, 1), "store after free");
    ASSERT(!forth_free(session, a), "double free");
    ASSERT(forth_fetch_cell(session, b, &cell) && cell == 99,
           "second region intact after free");

    hostish = (uint64_t)(uintptr_t)&stack_cell;
    ASSERT(!forth_store_cell(session, hostish, 1), "host pointer is not a Forth address");
    ASSERT(!forth_free(session, hostish), "cannot free a host pointer");
    ASSERT(!forth_free(session, 8), "cannot free an address that was never allocated");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_file_handles(void) {
    const char *test_name = "files: handles are table ids, not raw FILE*";
    ForthSession *session = forth_session_create();
    char path[] = "/tmp/forth_session_XXXXXX";
    int fd;
    uint32_t id = 0;
    uint32_t stale;

    ASSERT(session != NULL, "create failed");
    fd = mkstemp(path);
    ASSERT(fd >= 0, "mkstemp");
    close(fd);

    ASSERT(!forth_file_open(session, path, "r;rm", &id), "reject unsafe mode");
    ASSERT(!forth_file_open(session, NULL, "w", &id), "reject NULL path");
    ASSERT(forth_file_open(session, path, "w+", &id), "open temp file");
    ASSERT(id != 0, "fileid 0 is reserved");
    ASSERT(forth_file_is_open(session, id), "open handle is valid");
    ASSERT(!forth_file_is_open(session, 0), "fileid 0 is invalid");
    ASSERT(!forth_file_is_open(session, id + 1), "unknown fileid");

    stale = id;
    ASSERT(forth_file_close(session, id), "close");
    ASSERT(!forth_file_is_open(session, stale), "closed handle is invalid");
    ASSERT(!forth_file_close(session, stale), "stale close");

    ASSERT(forth_file_open(session, path, "r", &id), "reopen");
    ASSERT(id != stale, "reused slot must not revive the stale id");
    ASSERT(forth_file_is_open(session, id), "new handle");
    ASSERT(!forth_file_is_open(session, stale), "stale id stays invalid");
    ASSERT(forth_file_close(session, id), "close new");

    unlink(path);
    forth_session_destroy(session);
    PASS(test_name);
}

int main(void) {
    printf("Forth session runtime tests\n");
    test_session_owns_module_and_vm();
    test_null_session_is_rejected();
    test_forth_stacks_are_not_operand_stack();
    test_stale_decode_requires_rebuild();
    test_address_space_and_allocation();
    test_file_handles();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
