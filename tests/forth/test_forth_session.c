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
    ASSERT(forth_forth_wordlist(NULL) == 0, "NULL forth wordlist");
    ASSERT(!forth_find(NULL, "X", 1, NULL, NULL, NULL), "NULL find");
    ASSERT(!forth_source_load_terminal(NULL, (const uint8_t *)"x", 1),
           "NULL terminal");
    ASSERT(forth_source_depth(NULL) == 0, "NULL source depth");
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
    ASSERT(!forth_free(session, 1), "cannot free an address that was never allocated");
    ASSERT(!forth_free(session, forth_to_in_addr(session)),
           "cannot free the >IN cell");

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

static bool copy_forth_text(ForthSession *session, uint64_t addr, uint64_t n,
                            char *buf, size_t cap) {
    uint64_t i;
    uint8_t byte;
    if (n >= cap) return false;
    for (i = 0; i < n; i++) {
        if (!forth_fetch_byte(session, addr + i, &byte)) return false;
        buf[i] = (char)byte;
    }
    buf[n] = '\0';
    return true;
}

static void test_dictionary_headers_and_early_binding(void) {
    const char *test_name = "dict: headers, immediacy, early binding, word lists";
    ForthSession *session = forth_session_create();
    ForthNt nt1 = 0, nt2 = 0, found = 0, hidden = 0;
    ForthXt xt = 0, xt_found = 0;
    bool immediate = false;
    ForthWid forth, extra, current;
    ForthWid order[FORTH_ORDER_MAX];
    uint32_t norder = 0;
    uint64_t name_addr = 0;
    uint32_t name_len = 0;
    char namebuf[16];

    ASSERT(session != NULL, "create failed");
    forth = forth_forth_wordlist(session);
    ASSERT(forth != 0, "FORTH-WORDLIST");
    ASSERT(forth_get_current(session) == forth, "CURRENT starts as FORTH");
    ASSERT(forth_get_order(session, order, FORTH_ORDER_MAX, &norder), "GET-ORDER");
    ASSERT(norder == 1 && order[0] == forth, "search order is FORTH");

    ASSERT(forth_define(session, "DUP", 3, (ForthXt)1, false, false, &nt1),
           "define DUP xt=1");
    ASSERT(nt1 != 0, "nt is not zero");
    ASSERT(forth_latest(session) == nt1, "LATEST");
    ASSERT(forth_nt_wid(session, nt1) == forth, "header lives in CURRENT");
    ASSERT(forth_nt_xt(session, nt1, &xt) && xt == 1, "nt maps to xt 1");
    ASSERT(!forth_nt_immediate(session, nt1), "not immediate");
    ASSERT(!forth_nt_hidden(session, nt1), "not hidden");
    ASSERT(forth_nt_name(session, nt1, &name_addr, &name_len) && name_len == 3,
           "name length");
    ASSERT(copy_forth_text(session, name_addr, name_len, namebuf, sizeof(namebuf)),
           "copy name");
    ASSERT(strcmp(namebuf, "DUP") == 0, "name bytes");

    ASSERT(forth_find(session, "dup", 3, &found, &xt_found, &immediate),
           "FIND is case-insensitive");
    ASSERT(found == nt1 && xt_found == 1 && !immediate, "FIND DUP");

    ASSERT(forth_define(session, "DUP", 3, (ForthXt)2, true, false, &nt2),
           "redefine DUP xt=2 immediate");
    ASSERT(nt2 != nt1, "redefinition is a new header");
    ASSERT(forth_find(session, "DUP", 3, &found, &xt_found, &immediate),
           "FIND after redefine");
    ASSERT(found == nt2 && xt_found == 2 && immediate, "FIND sees the new xt");
    ASSERT(forth_nt_xt(session, nt1, &xt) && xt == 1,
           "old nt still binds the old xt");

    ASSERT(forth_define(session, "SMUDGE", 6, (ForthXt)3, false, true, &hidden),
           "hidden header");
    ASSERT(forth_nt_hidden(session, hidden), "hidden");
    ASSERT(!forth_find(session, "SMUDGE", 6, &found, &xt_found, &immediate),
           "FIND skips hidden");
    ASSERT(forth_reveal(session, hidden), "reveal");
    ASSERT(forth_find(session, "SMUDGE", 6, &found, &xt_found, &immediate),
           "FIND after reveal");
    ASSERT(found == hidden && xt_found == 3, "revealed xt");

    ASSERT(forth_wordlist_create(session, &extra) && extra != forth,
           "WORDLIST");
    ASSERT(forth_set_current(session, extra), "SET-CURRENT extra");
    ASSERT(forth_define(session, "ONLYEXTRA", 9, (ForthXt)4, false, false, &nt1),
           "define in extra");
    ASSERT(!forth_find(session, "ONLYEXTRA", 9, &found, &xt_found, &immediate),
           "not in FORTH search order");
    order[0] = extra;
    order[1] = forth;
    ASSERT(forth_set_order(session, order, 2), "SET-ORDER extra then FORTH");
    ASSERT(forth_find(session, "ONLYEXTRA", 9, &found, &xt_found, &immediate),
           "FIND with extra first");
    ASSERT(xt_found == 4, "extra word");
    ASSERT(forth_find(session, "DUP", 3, &found, &xt_found, &immediate),
           "FORTH still searched");
    ASSERT(xt_found == 2, "DUP from FORTH");

    current = forth_get_current(session);
    ASSERT(current == extra, "CURRENT stays extra");
    ASSERT(!forth_set_current(session, 0), "invalid wid");
    ASSERT(!forth_set_order(session, order, FORTH_ORDER_MAX + 1), "order overflow");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_nested_input_sources(void) {
    const char *test_name = "source: terminal, evaluate, file, block, restore";
    ForthSession *session = forth_session_create();
    uint64_t caddr = 0, u = 0, eval_addr = 0, saved_to_in = 0;
    int64_t to_in = 0, blk = 0;
    char text[64];
    char path[] = "/tmp/forth_source_XXXXXX";
    int fd;
    FILE *fp;
    uint32_t fileid = 0;
    uint32_t i;

    ASSERT(session != NULL, "create failed");
    ASSERT(forth_source_depth(session) == 1, "base source");
    ASSERT(forth_source_id(session) == 0, "terminal SOURCE-ID");
    ASSERT(forth_to_in_addr(session) != 0, ">IN address");
    ASSERT(forth_blk_addr(session) != 0, "BLK address");
    ASSERT(!forth_source_pop(session), "cannot pop the base source");

    ASSERT(forth_source_load_terminal(session, (const uint8_t *)"1 2 +", 5),
           "load TIB");
    ASSERT(forth_source(session, &caddr, &u) && u == 5, "SOURCE length");
    ASSERT(copy_forth_text(session, caddr, u, text, sizeof(text)), "copy TIB");
    ASSERT(strcmp(text, "1 2 +") == 0, "TIB contents");
    ASSERT(forth_store_cell(session, forth_to_in_addr(session), 2), "set >IN");
    ASSERT(forth_fetch_cell(session, forth_to_in_addr(session), &to_in)
           && to_in == 2, ">IN stored in Forth memory");

    ASSERT(forth_allocate(session, 8, &eval_addr), "evaluate string");
    ASSERT(forth_store_byte(session, eval_addr, (uint8_t)'A'), "A");
    ASSERT(forth_store_byte(session, eval_addr + 1, (uint8_t)'B'), "B");
    ASSERT(forth_store_byte(session, eval_addr + 2, (uint8_t)'C'), "C");
    ASSERT(forth_source_push_evaluate(session, eval_addr, 3), "EVALUATE");
    ASSERT(forth_source_depth(session) == 2, "nested");
    ASSERT(forth_source_id(session) == -1, "EVALUATE SOURCE-ID");
    ASSERT(forth_source(session, &caddr, &u) && u == 3, "evaluate SOURCE");
    ASSERT(copy_forth_text(session, caddr, u, text, sizeof(text))
           && strcmp(text, "ABC") == 0, "evaluate text");
    ASSERT(forth_fetch_cell(session, forth_to_in_addr(session), &to_in)
           && to_in == 0, "EVALUATE resets >IN");
    ASSERT(forth_fetch_cell(session, forth_blk_addr(session), &blk) && blk == 0,
           "BLK 0 while evaluating");
    ASSERT(!forth_refill(session), "REFILL is false on a string");
    ASSERT(forth_source_pop(session), "pop evaluate");
    ASSERT(forth_source_depth(session) == 1, "restored terminal");
    ASSERT(forth_source_id(session) == 0, "restored SOURCE-ID");
    ASSERT(forth_source(session, &caddr, &u) && u == 5, "restored TIB");
    ASSERT(forth_fetch_cell(session, forth_to_in_addr(session), &to_in)
           && to_in == 2, "restored >IN");

    fd = mkstemp(path);
    ASSERT(fd >= 0, "mkstemp");
    fp = fdopen(fd, "w+");
    ASSERT(fp != NULL, "fdopen");
    ASSERT(fwrite("hello\r\nworld\n", 1, 13, fp) == 13, "write lines");
    ASSERT(fflush(fp) == 0, "flush");
    fclose(fp);
    ASSERT(forth_file_open(session, path, "r", &fileid), "open include");
    ASSERT(forth_source_push_file(session, fileid), "INCLUDE-FILE");
    ASSERT(forth_source_id(session) == (int64_t)fileid, "file SOURCE-ID");
    ASSERT(forth_refill(session), "first line");
    ASSERT(forth_source(session, &caddr, &u) && u == 5, "hello length");
    ASSERT(copy_forth_text(session, caddr, u, text, sizeof(text))
           && strcmp(text, "hello") == 0, "CRLF stripped");
    ASSERT(forth_refill(session), "second line");
    ASSERT(forth_source(session, &caddr, &u) && u == 5, "world length");
    ASSERT(copy_forth_text(session, caddr, u, text, sizeof(text))
           && strcmp(text, "world") == 0, "LF line");
    ASSERT(!forth_refill(session), "EOF");
    ASSERT(forth_source_pop(session), "pop file");
    ASSERT(forth_source_id(session) == 0, "back to terminal");
    ASSERT(forth_file_close(session, fileid), "close include");
    unlink(path);

    ASSERT(forth_source_push_block(session, 20), "LOAD 20");
    ASSERT(forth_source_id(session) == 0, "block SOURCE-ID is 0");
    ASSERT(forth_fetch_cell(session, forth_blk_addr(session), &blk) && blk == 20,
           "BLK");
    ASSERT(forth_source(session, &caddr, &u) && u == FORTH_BLOCK_SIZE,
           "block is 1024");
    ASSERT(forth_store_byte(session, caddr, (uint8_t)'Z'), "write block");
    ASSERT(!forth_refill(session), "block REFILL is false here");
    ASSERT(forth_source_pop(session), "pop block");
    ASSERT(forth_fetch_cell(session, forth_blk_addr(session), &blk) && blk == 0,
           "BLK restored");
    ASSERT(forth_source_push_block(session, 20), "reload 20");
    ASSERT(forth_source(session, &caddr, &u), "block SOURCE");
    {
        uint8_t byte = 0;
        ASSERT(forth_fetch_byte(session, caddr, &byte) && byte == (uint8_t)'Z',
               "block image persists");
    }
    ASSERT(forth_source_pop(session), "pop block again");
    ASSERT(!forth_source_push_block(session, FORTH_BLOCK_COUNT), "block range");

    ASSERT(forth_allocate(session, 1, &eval_addr), "tiny evaluate");
    ASSERT(forth_store_byte(session, eval_addr, (uint8_t)'x'), "x");
    for (i = 1; i < FORTH_SOURCE_NEST; i++) {
        ASSERT(forth_source_push_evaluate(session, eval_addr, 1), "nest");
    }
    ASSERT(!forth_source_push_evaluate(session, eval_addr, 1), "nest overflow");
    while (forth_source_depth(session) > 1) {
        ASSERT(forth_source_pop(session), "unwind");
    }

    saved_to_in = forth_to_in_addr(session);
    ASSERT(saved_to_in % FORTH_CELL_BYTES == 0, ">IN is cell-aligned");
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
    test_dictionary_headers_and_early_binding();
    test_nested_input_sources();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
