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
#include "nanovm/vm_ffi.h"

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
    ASSERT(!forth_colon_begin(NULL, "X", 1), "NULL colon begin");
    ASSERT(!forth_colon_literal(NULL, 1), "NULL colon literal");
    ASSERT(!forth_colon_call(NULL, 0), "NULL colon call");
    ASSERT(!forth_colon_recurse(NULL), "NULL colon recurse");
    ASSERT(!forth_colon_if(NULL), "NULL colon if");
    ASSERT(!forth_colon_then(NULL), "NULL colon then");
    ASSERT(!forth_catch(NULL, 0, NULL), "NULL catch");
    ASSERT(!forth_colon_finish(NULL, NULL), "NULL colon finish");
    ASSERT(!forth_colon_abort(NULL), "NULL colon abort");
    ASSERT(!forth_colon_is_open(NULL), "NULL colon is_open");
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

static bool function_calls_index(const NvmModule *mod, uint32_t fn_idx, uint32_t target) {
    const NvmFunctionEntry *fn;
    uint32_t off = 0;

    if (!mod || fn_idx >= mod->function_count) return false;
    fn = &mod->functions[fn_idx];
    while (off < fn->code_length) {
        DecodedInstruction instr;
        uint32_t n = isa_decode(mod->code + fn->code_offset + off,
                                fn->code_length - off, &instr);
        if (n == 0) return false;
        if (instr.opcode == OP_CALL && instr.operands[0].u32 == target) return true;
        off += n;
    }
    return false;
}

static void test_colon_compile_verify_and_early_binding(void) {
    const char *test_name = "colon: private compile, verify, OP_CALL, RECURSE";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0, nt_first = 0, nt_second = 0, nt_old = 0;
    ForthXt xt = 0, xt_first = 0, xt_second = 0, xt_new = 0, xt_bad = 0;
    ForthNt found = 0;
    bool immediate = false;
    int64_t cell = 0;
    uint32_t fn_count_before = 0;
    NvmModule *mod;
    NvmVerifyResult verified;
    uint32_t recurse_xt = 0;

    ASSERT(session != NULL, "create failed");
    ASSERT(!forth_colon_is_open(session), "no colon at start");
    ASSERT(!forth_colon_literal(session, 1), "literal requires an open colon");
    ASSERT(!forth_colon_call(session, 0), "call requires an open colon");
    ASSERT(!forth_colon_recurse(session), "recurse requires an open colon");
    ASSERT(!forth_colon_finish(session, &nt), "finish requires an open colon");
    ASSERT(!forth_colon_abort(session), "abort requires an open colon");

    ASSERT(forth_colon_begin(session, "ANSWER", 6), "begin ANSWER");
    ASSERT(forth_colon_is_open(session), "colon is open");
    ASSERT(!forth_find(session, "ANSWER", 6, &found, &xt, &immediate),
           "hidden until publish");
    ASSERT(!forth_colon_begin(session, "OTHER", 5), "one colon at a time");
    ASSERT(forth_colon_literal(session, 42), "literal 42");
    ASSERT(forth_colon_finish(session, &nt), "publish ANSWER");
    ASSERT(!forth_colon_is_open(session), "closed after publish");
    ASSERT(forth_find(session, "ANSWER", 6, &found, &xt, &immediate), "FIND ANSWER");
    ASSERT(found == nt, "published nt");
    ASSERT(!immediate, "not immediate");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "run ANSWER");
    ASSERT(forth_data_pop(session, &cell) && cell == 42, "ANSWER left 42");
    ASSERT(forth_data_depth(session) == 0, "stack empty");

    ASSERT(forth_colon_begin(session, "FIRST", 5), "begin FIRST");
    ASSERT(forth_colon_literal(session, 1), "FIRST is 1");
    ASSERT(forth_colon_finish(session, &nt_first), "publish FIRST");
    ASSERT(forth_nt_xt(session, nt_first, &xt_first), "FIRST xt");

    ASSERT(forth_colon_begin(session, "SECOND", 6), "begin SECOND");
    ASSERT(forth_colon_call(session, xt_first), "compile CALL FIRST");
    ASSERT(forth_colon_finish(session, &nt_second), "publish SECOND");
    ASSERT(forth_nt_xt(session, nt_second, &xt_second), "SECOND xt");
    ASSERT(function_calls_index(forth_session_module(session), xt_second, xt_first),
           "SECOND contains OP_CALL to FIRST");

    nt_old = nt_first;
    ASSERT(forth_colon_begin(session, "FIRST", 5), "redefine FIRST");
    ASSERT(forth_colon_literal(session, 2), "new FIRST is 2");
    ASSERT(forth_colon_finish(session, &nt_first), "publish new FIRST");
    ASSERT(nt_first != nt_old, "new name token");
    ASSERT(forth_nt_xt(session, nt_old, &xt_first), "old FIRST xt");
    ASSERT(forth_find(session, "FIRST", 5, &found, &xt_new, &immediate), "FIND new FIRST");
    ASSERT(xt_new != xt_first, "new execution token");
    ASSERT(forth_session_invoke(session, xt_second, NULL, 0, NULL) == VM_OK,
           "run SECOND");
    ASSERT(forth_data_pop(session, &cell) && cell == 1,
           "SECOND still calls the old FIRST");
    ASSERT(forth_session_invoke(session, xt_new, NULL, 0, NULL) == VM_OK,
           "run new FIRST");
    ASSERT(forth_data_pop(session, &cell) && cell == 2, "new FIRST is 2");

    ASSERT(forth_colon_begin(session, "SELF", 4), "begin SELF");
    recurse_xt = forth_colon_xt(session);
    ASSERT(recurse_xt != 0 || forth_colon_is_open(session), "reserved xt");
    ASSERT(forth_colon_recurse(session), "RECURSE");
    ASSERT(forth_colon_finish(session, &nt), "publish SELF");
    ASSERT(forth_nt_xt(session, nt, &xt), "SELF xt");
    ASSERT(xt == recurse_xt, "RECURSE bound to the reserved definition");
    ASSERT(function_calls_index(forth_session_module(session), xt, xt),
           "SELF contains OP_CALL to itself");

    ASSERT(forth_colon_begin(session, "DROPIT", 6), "begin abort");
    ASSERT(forth_colon_literal(session, 9), "unused literal");
    ASSERT(forth_colon_abort(session), "abort DROPIT");
    ASSERT(!forth_colon_is_open(session), "closed after abort");
    ASSERT(!forth_find(session, "DROPIT", 6, &found, &xt, &immediate),
           "aborted name is unpublished");

    xt_bad = publish_i64_const(session, "not_a_colon", (int64_t)99);
    ASSERT(xt_bad != UINT32_MAX, "result-producing helper");
    mod = forth_session_module(session);
    fn_count_before = mod->function_count;
    ASSERT(forth_colon_begin(session, "BAD", 3), "begin BAD");
    ASSERT(forth_colon_call(session, xt_bad), "call a 1-result function");
    ASSERT(!forth_colon_finish(session, &nt), "verify must reject leftover results");
    ASSERT(!forth_colon_is_open(session), "failed finish aborts");
    ASSERT(!forth_find(session, "BAD", 3, &found, &xt, &immediate),
           "failed verify does not publish");
    ASSERT(mod->function_count == fn_count_before,
           "failed verify rolls back the reserved function");
    verified = nvm_verify_function(mod, xt_second);
    ASSERT(verified.ok, "earlier SECOND still verifies");

    ASSERT(!forth_colon_begin(session, "", 0), "empty name");
    {
        char long_name[256];
        memset(long_name, 'A', sizeof(long_name));
        ASSERT(!forth_colon_begin(session, long_name, 256), "name too long");
    }

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_colon_structured_control_flow(void) {
    const char *test_name = "colon: IF ELSE THEN, BEGIN UNTIL, WHILE REPEAT";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0;
    ForthXt xt = 0;
    int64_t cell = 0;

    ASSERT(session != NULL, "create failed");
    ASSERT(!forth_colon_if(session), "IF requires an open colon");
    ASSERT(!forth_colon_then(session), "THEN requires an open colon");
    ASSERT(!forth_colon_cs_begin(session), "BEGIN requires an open colon");

    ASSERT(forth_colon_begin(session, "CHOOSE", 6), "begin CHOOSE");
    ASSERT(forth_colon_if(session), "IF");
    ASSERT(forth_colon_literal(session, 1), "true body");
    ASSERT(forth_colon_else(session), "ELSE");
    ASSERT(forth_colon_literal(session, 2), "false body");
    ASSERT(forth_colon_then(session), "THEN");
    ASSERT(forth_colon_finish(session, &nt), "publish CHOOSE");
    ASSERT(forth_nt_xt(session, nt, &xt), "CHOOSE xt");

    ASSERT(forth_data_push(session, -1), "true flag");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "CHOOSE true");
    ASSERT(forth_data_pop(session, &cell) && cell == 1, "true path is 1");

    ASSERT(forth_data_push(session, 0), "false flag");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "CHOOSE false");
    ASSERT(forth_data_pop(session, &cell) && cell == 2, "false path is 2");

    ASSERT(forth_colon_begin(session, "WHEN", 4), "begin WHEN");
    ASSERT(forth_colon_if(session), "IF WHEN");
    ASSERT(forth_colon_literal(session, 7), "only if true");
    ASSERT(forth_colon_then(session), "THEN WHEN");
    ASSERT(forth_colon_finish(session, &nt), "publish WHEN");
    ASSERT(forth_nt_xt(session, nt, &xt), "WHEN xt");
    ASSERT(forth_data_push(session, 0), "false WHEN");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "WHEN false");
    ASSERT(forth_data_depth(session) == 0, "false IF leaves nothing");
    ASSERT(forth_data_push(session, 1), "true WHEN");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "WHEN true");
    ASSERT(forth_data_pop(session, &cell) && cell == 7, "true IF");

    ASSERT(forth_colon_begin(session, "ONCE", 4), "begin ONCE");
    ASSERT(forth_colon_cs_begin(session), "BEGIN");
    ASSERT(forth_colon_literal(session, 1), "until true");
    ASSERT(forth_colon_until(session), "UNTIL");
    ASSERT(forth_colon_finish(session, &nt), "publish ONCE");
    ASSERT(forth_nt_xt(session, nt, &xt), "ONCE xt");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "ONCE");
    ASSERT(forth_data_depth(session) == 0, "UNTIL consumed the flag");

    ASSERT(forth_colon_begin(session, "SKIP", 4), "begin SKIP");
    ASSERT(forth_colon_cs_begin(session), "BEGIN SKIP");
    ASSERT(forth_colon_literal(session, 0), "WHILE false");
    ASSERT(forth_colon_while(session), "WHILE");
    ASSERT(forth_colon_literal(session, 99), "should not run");
    ASSERT(forth_colon_repeat(session), "REPEAT");
    ASSERT(forth_colon_finish(session, &nt), "publish SKIP");
    ASSERT(forth_nt_xt(session, nt, &xt), "SKIP xt");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "SKIP");
    ASSERT(forth_data_depth(session) == 0, "zero-trip WHILE");

    ASSERT(forth_colon_begin(session, "BADTHEN", 7), "begin BADTHEN");
    ASSERT(!forth_colon_then(session), "THEN without IF");
    ASSERT(forth_colon_abort(session), "abort BADTHEN");

    ASSERT(forth_colon_begin(session, "OPENIF", 6), "begin OPENIF");
    ASSERT(forth_colon_if(session), "unmatched IF");
    ASSERT(!forth_colon_finish(session, &nt), "finish rejects unmatched IF");
    ASSERT(!forth_colon_is_open(session), "failed finish aborted");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_catch_and_throw(void) {
    const char *test_name = "catch: restore stacks on THROW, THROW 0 is a no-op";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0;
    ForthXt xt = 0;
    int64_t code = -1;
    int64_t cell = 0;

    ASSERT(session != NULL, "create failed");
    ASSERT(!forth_colon_throw(session), "THROW requires an open colon");
    ASSERT(!forth_catch(session, 0, &code), "CATCH needs a function");

    ASSERT(forth_colon_begin(session, "OKWORD", 6), "begin OKWORD");
    ASSERT(forth_colon_literal(session, 7), "7");
    ASSERT(forth_colon_finish(session, &nt), "publish OKWORD");
    ASSERT(forth_nt_xt(session, nt, &xt), "OKWORD xt");
    ASSERT(forth_catch(session, xt, &code), "CATCH OKWORD");
    ASSERT(code == 0, "no throw");
    ASSERT(forth_data_pop(session, &cell) && cell == 7, "stack effect kept");

    ASSERT(forth_colon_begin(session, "BOOM", 4), "begin BOOM");
    ASSERT(forth_colon_literal(session, 1), "1");
    ASSERT(forth_colon_literal(session, 2), "2");
    ASSERT(forth_colon_literal(session, 99), "throw code");
    ASSERT(forth_colon_throw(session), "THROW");
    ASSERT(forth_colon_finish(session, &nt), "publish BOOM");
    ASSERT(forth_nt_xt(session, nt, &xt), "BOOM xt");
    ASSERT(forth_data_push(session, 42), "marker");
    ASSERT(forth_return_push(session, 11), "return marker");
    ASSERT(forth_catch(session, xt, &code), "CATCH BOOM");
    ASSERT(code == 99, "throw code");
    ASSERT(forth_data_pop(session, &cell) && cell == 42, "data stack restored");
    ASSERT(forth_data_depth(session) == 0, "no leftover thrown cells");
    ASSERT(forth_return_pop(session, &cell) && cell == 11, "return stack restored");

    ASSERT(forth_colon_begin(session, "ZERO", 4), "begin ZERO");
    ASSERT(forth_colon_literal(session, 0), "THROW 0");
    ASSERT(forth_colon_throw(session), "throw zero");
    ASSERT(forth_colon_literal(session, 8), "continues");
    ASSERT(forth_colon_finish(session, &nt), "publish ZERO");
    ASSERT(forth_nt_xt(session, nt, &xt), "ZERO xt");
    ASSERT(forth_catch(session, xt, &code), "CATCH ZERO");
    ASSERT(code == 0, "THROW 0 does not throw");
    ASSERT(forth_data_pop(session, &cell) && cell == 8, "execution continued");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_typed_import_abs(void) {
    const char *test_name = "import: abs lowers to CALL_EXTERN and runs";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0;
    ForthXt xt = 0;
    ForthNt found_nt = 0;
    ForthXt found_xt = 0;
    bool immediate = true;
    uint8_t params[1];
    int64_t cell = 0;
    NvmModule *mod;
    uint32_t i;
    bool saw_extern = false;

    params[0] = TAG_INT;
    ASSERT(session != NULL, "create failed");
    mod = forth_session_module(session);
    ASSERT(mod != NULL, "module");
    {
        uint32_t imports0 = mod->import_count;
        ASSERT(forth_import_declare(session, "", "abs", params, 1, TAG_INT, &nt),
               "declare abs");
        mod = forth_session_module(session);
        ASSERT(mod != NULL && mod->import_count == imports0 + 1, "one new import entry");
        ASSERT(forth_nt_xt(session, nt, &xt), "abs xt");
        ASSERT(forth_find(session, "abs", 3, &found_nt, &found_xt, &immediate),
               "abs is in the dictionary");
        ASSERT(found_xt == xt && !immediate, "early-bound non-immediate xt");
        for (i = 0; i < mod->functions[xt].code_length; ) {
            DecodedInstruction instr;
            uint32_t n = isa_decode(mod->code + mod->functions[xt].code_offset + i,
                                    mod->functions[xt].code_length - i, &instr);
            ASSERT(n > 0, "decode wrapper");
            if (instr.opcode == OP_CALL_EXTERN && instr.operands[0].u32 == imports0)
                saw_extern = true;
            i += n;
        }
        ASSERT(saw_extern, "wrapper emits OP_CALL_EXTERN for abs");
    }
    ASSERT(forth_data_push(session, -42), "push -42");
    ASSERT(forth_session_invoke(session, xt, NULL, 0, NULL) == VM_OK, "invoke abs");
    ASSERT(forth_data_pop(session, &cell) && cell == 42, "abs(-42) is 42");
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_import_rejects_unusable_abi(void) {
    const char *test_name = "import: reject mixed float and oversized FFI";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0;
    uint8_t mixed[2];
    uint8_t too_many[NANO_MAX_FFI_ARGS + 1];
    uint8_t bad[1];
    uint16_t i;
    uint32_t imports0;

    mixed[0] = TAG_INT;
    mixed[1] = TAG_FLOAT;
    for (i = 0; i < NANO_MAX_FFI_ARGS + 1; i++)
        too_many[i] = TAG_INT;
    bad[0] = TAG_STRING;
    ASSERT(session != NULL, "create failed");
    imports0 = forth_session_module(session)->import_count;
    ASSERT(!forth_import_declare(session, "", "f", mixed, 2, TAG_INT, &nt),
           "mixed float/int");
    ASSERT(!forth_import_declare(session, "", "g", too_many,
                                 (uint16_t)(NANO_MAX_FFI_ARGS + 1), TAG_INT, &nt),
           "too many args");
    ASSERT(!forth_import_declare(session, "", "h", bad, 1, TAG_INT, &nt),
           "string param is not a Forth cell");
    ASSERT(forth_session_module(session)->import_count == imports0, "no import leaked");
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_import_stops_cop(void) {
    const char *test_name = "import: stop FFI co-process after table mutation";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0;
    uint8_t params[1];
    VmState *vm;
    bool started;

    params[0] = TAG_INT;
    ASSERT(session != NULL, "create failed");
    vm = forth_session_vm(session);
    ASSERT(vm != NULL, "vm");
    vm->isolate_ffi = true;
    started = vm_ffi_cop_start(vm, forth_session_module(session));
    ASSERT(started && vm->cop_pid > 0, "co-process started");
    ASSERT(forth_import_declare(session, "", "abs", params, 1, TAG_INT, &nt),
           "declare after cop is live");
    ASSERT(vm->cop_pid <= 0, "import stops the stale co-process");
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_see_disassembles_colon_and_import(void) {
    const char *test_name = "see: NanoISA of colon words and imported words";
    ForthSession *session = forth_session_create();
    ForthNt nt = 0;
    ForthXt xt = 0;
    uint8_t params[1];
    char *text = NULL;

    params[0] = TAG_INT;
    ASSERT(session != NULL, "create failed");
    ASSERT(forth_colon_begin(session, "LIT7", 4), "begin");
    ASSERT(forth_colon_literal(session, 7), "7");
    ASSERT(forth_colon_finish(session, &nt), "finish");
    ASSERT(forth_nt_xt(session, nt, &xt), "xt");
    text = forth_see(session, xt);
    ASSERT(text != NULL, "see colon");
    ASSERT(strstr(text, "7") != NULL, "literal in disassembly");
    free(text);

    ASSERT(forth_import_declare(session, "", "abs", params, 1, TAG_INT, &nt),
           "import abs");
    ASSERT(forth_nt_xt(session, nt, &xt), "abs xt");
    text = forth_see(session, xt);
    ASSERT(text != NULL, "see import");
    ASSERT(strstr(text, "extern-call") != NULL, "imported word shows CALL_EXTERN");
    ASSERT(strstr(text, "abs") != NULL, "imported word names the symbol");
    free(text);
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_interpret_numbers(void) {
    const char *test_name = "interpret: numbers go to the data stack";
    ForthSession *session = forth_session_create();
    int64_t cell = 0;
    const uint8_t line[] = "7 8";

    ASSERT(session != NULL, "create failed");
    ASSERT(forth_interpret(session, line, 3), "interpret 7 8");
    ASSERT(forth_data_pop(session, &cell) && cell == 8, "TOS 8");
    ASSERT(forth_data_pop(session, &cell) && cell == 7, "NOS 7");
    ASSERT(forth_data_depth(session) == 0, "empty after");
    ASSERT(!forth_interpret(session, (const uint8_t *)"no-such-word", 12),
           "unknown word fails");
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_interpret_plus_dup_star(void) {
    const char *test_name = "interpret: + DUP * and colon definitions";
    ForthSession *session = forth_session_create();
    int64_t cell = 0;
    const uint8_t add[] = "7 8 +";
    const uint8_t sqr[] = ": SQR DUP * ; 6 SQR";

    ASSERT(session != NULL, "create failed");
    ASSERT(forth_interpret(session, add, 5), "7 8 +");
    ASSERT(forth_data_pop(session, &cell) && cell == 15, "7 8 + is 15");
    ASSERT(forth_interpret(session, sqr, 19), ": SQR DUP * ; 6 SQR");
    ASSERT(forth_data_pop(session, &cell) && cell == 36, "6 squared is 36");
    forth_session_destroy(session);
    PASS(test_name);
}

static bool interpret_cstr(ForthSession *session, const char *text) {
    return forth_interpret(session, (const uint8_t *)text, (uint32_t)strlen(text));
}

static bool expect_cells(ForthSession *session, const int64_t *want, uint32_t n) {
    uint32_t i;
    int64_t cell = 0;
    if (forth_data_depth(session) != n) return false;
    for (i = n; i > 0; i--) {
        if (!forth_data_pop(session, &cell) || cell != want[i - 1]) return false;
    }
    return true;
}

static void test_core_arithmetic_stack_compare(void) {
    const char *test_name = "core: arithmetic, stack, compare, bitwise";
    ForthSession *session = forth_session_create();
    int64_t want[8];

    ASSERT(session != NULL, "create failed");
    ASSERT(interpret_cstr(session, "5 3 -"), "5 3 -");
    want[0] = 2;
    ASSERT(expect_cells(session, want, 1), "5 3 - is 2");

    ASSERT(interpret_cstr(session, "-7 2 /MOD"), "-7 2 /MOD");
    want[0] = 1;
    want[1] = -4;
    ASSERT(expect_cells(session, want, 2), "floored /MOD");

    ASSERT(interpret_cstr(session, "7 3 MOD"), "7 3 MOD");
    want[0] = 1;
    ASSERT(expect_cells(session, want, 1), "7 3 MOD is 1");

    ASSERT(interpret_cstr(session, "-7 3 MOD"), "-7 3 MOD");
    want[0] = 2;
    ASSERT(expect_cells(session, want, 1), "floored MOD");

    ASSERT(interpret_cstr(session, "1 2 3 ROT"), "rot");
    want[0] = 2;
    want[1] = 3;
    want[2] = 1;
    ASSERT(expect_cells(session, want, 3), "1 2 3 ROT");

    ASSERT(interpret_cstr(session, "0 0 = 1 0 = 0 0= -1 0<"), "flags");
    want[0] = -1;
    want[1] = 0;
    want[2] = -1;
    want[3] = -1;
    ASSERT(expect_cells(session, want, 4), "true is -1");

    ASSERT(interpret_cstr(session, "1 3 AND 1 2 OR 0 INVERT"), "bitwise");
    want[0] = 1;
    want[1] = 3;
    want[2] = -1;
    ASSERT(expect_cells(session, want, 3), "AND OR INVERT");

    ASSERT(interpret_cstr(session, "1 8 LSHIFT 256 8 RSHIFT"), "shifts");
    want[0] = 256;
    want[1] = 1;
    ASSERT(expect_cells(session, want, 2), "LSHIFT RSHIFT");

    ASSERT(interpret_cstr(session, "-5 ABS 2 1 MAX 2 1 MIN"), "abs min max");
    want[0] = 5;
    want[1] = 2;
    want[2] = 1;
    ASSERT(expect_cells(session, want, 3), "ABS MAX MIN");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_core_memory_control_base(void) {
    const char *test_name = "core: VARIABLE CONSTANT IF BEGIN BASE HERE";
    ForthSession *session = forth_session_create();
    int64_t want[8];
    int64_t here1 = 0;
    int64_t here2 = 0;

    ASSERT(session != NULL, "create failed");
    ASSERT(interpret_cstr(session, "VARIABLE MV1 42 MV1 ! MV1 @"), "variable");
    want[0] = 42;
    ASSERT(expect_cells(session, want, 1), "VARIABLE @ !");

    ASSERT(interpret_cstr(session, "5 CONSTANT MK5 MK5"), "constant");
    want[0] = 5;
    ASSERT(expect_cells(session, want, 1), "CONSTANT");

    ASSERT(interpret_cstr(session, ": CHOOSE IF 1 ELSE 2 THEN ; -1 CHOOSE 0 CHOOSE"),
           "if else then");
    want[0] = 1;
    want[1] = 2;
    ASSERT(expect_cells(session, want, 2), "IF ELSE THEN");

    ASSERT(interpret_cstr(session, ": BU 0 BEGIN 1+ DUP 5 = UNTIL ; BU"),
           "begin until");
    want[0] = 5;
    ASSERT(expect_cells(session, want, 1), "BEGIN UNTIL");

    ASSERT(interpret_cstr(session, ": DL 0 5 0 DO 1+ LOOP ; DL"), "do loop");
    want[0] = 5;
    ASSERT(expect_cells(session, want, 1), "DO LOOP");

    ASSERT(interpret_cstr(session, ": DI 5 0 DO I LOOP ; DI"), "do i");
    want[0] = 0;
    want[1] = 1;
    want[2] = 2;
    want[3] = 3;
    want[4] = 4;
    ASSERT(expect_cells(session, want, 5), "I inside DO");

    ASSERT(interpret_cstr(session, ": DPL 10 0 DO I 2 +LOOP ; DPL"), "+loop");
    want[0] = 0;
    want[1] = 2;
    want[2] = 4;
    want[3] = 6;
    want[4] = 8;
    ASSERT(expect_cells(session, want, 5), "+LOOP");

    ASSERT(interpret_cstr(session, ": QD 5 5 ?DO I LOOP ; QD"), "?do empty");
    ASSERT(forth_data_depth(session) == 0, "?DO equal");

    ASSERT(interpret_cstr(session, ": LV 10 0 DO I DUP 3 = IF DROP LEAVE THEN LOOP ; LV"),
           "leave");
    want[0] = 0;
    want[1] = 1;
    want[2] = 2;
    ASSERT(expect_cells(session, want, 3), "LEAVE");

    ASSERT(interpret_cstr(session, "1 >R R@ R>"), "rstack");
    want[0] = 1;
    want[1] = 1;
    ASSERT(expect_cells(session, want, 2), ">R R@ R>");

    ASSERT(interpret_cstr(session, "HEX FF DECIMAL"), "hex");
    want[0] = 255;
    ASSERT(expect_cells(session, want, 1), "HEX FF is 255");

    ASSERT(interpret_cstr(session, "HERE"), "here");
    ASSERT(forth_data_pop(session, &here1), "pop HERE");
    ASSERT(interpret_cstr(session, "8 ALLOT HERE"), "allot");
    ASSERT(forth_data_pop(session, &here2), "pop HERE after ALLOT");
    ASSERT(here2 - here1 == 8, "ALLOT 8");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_core_io_create_evaluate(void) {
    const char *test_name = "core: EMIT S\" CREATE EVALUATE ENVIRONMENT?";
    ForthSession *session = forth_session_create();
    int64_t want[8];

    ASSERT(session != NULL, "create failed");
    forth_output_clear(session);
    ASSERT(interpret_cstr(session, "65 EMIT 66 EMIT CR"), "emit");
    ASSERT(strcmp(forth_output(session), "AB\n") == 0, "EMIT CR text");

    forth_output_clear(session);
    ASSERT(interpret_cstr(session, "S\" hi\" TYPE"), "squote type");
    ASSERT(strcmp(forth_output(session), "hi") == 0, "S\" TYPE");

    ASSERT(interpret_cstr(session, "CREATE CR0 5 , CR0 @"), "create comma");
    want[0] = 5;
    ASSERT(expect_cells(session, want, 1), "CREATE , @");

    ASSERT(interpret_cstr(session, ": KONST CREATE , DOES> @ ; 7 KONST SEVEN SEVEN"),
           "does");
    want[0] = 7;
    ASSERT(expect_cells(session, want, 1), "DOES> CONSTANT");

    ASSERT(interpret_cstr(session, "S\" 2 3 +\" EVALUATE"), "evaluate");
    want[0] = 5;
    ASSERT(expect_cells(session, want, 1), "EVALUATE");

    ASSERT(interpret_cstr(session, "S\" FLOORED\" ENVIRONMENT?"), "env");
    want[0] = -1;
    want[1] = -1;
    ASSERT(expect_cells(session, want, 2), "FLOORED is true");

    ASSERT(interpret_cstr(session, "1 2 3 1 PICK"), "pick");
    want[0] = 1;
    want[1] = 2;
    want[2] = 3;
    want[3] = 2;
    ASSERT(expect_cells(session, want, 4), "1 PICK");

    forth_output_clear(session);
    ASSERT(interpret_cstr(session, "42 ."), "dot");
    ASSERT(strcmp(forth_output(session), "42 ") == 0, ". prints 42");

    ASSERT(interpret_cstr(session, "7 S>D 2 SM/REM"), "sm/rem");
    want[0] = 1;
    want[1] = 3;
    ASSERT(expect_cells(session, want, 2), "SM/REM");

    ASSERT(interpret_cstr(session, "-7 S>D 2 FM/MOD"), "fm/mod");
    want[0] = 1;
    want[1] = -4;
    ASSERT(expect_cells(session, want, 2), "FM/MOD floored");

    forth_session_destroy(session);
    PASS(test_name);
}

static void test_kernel_defects(void) {
    const char *test_name = "kernel: VARIABLE CATCH BYE EXECUTE sessions S\"";
    ForthSession *session;
    ForthSession *other;
    int64_t want[8];
    int64_t cell = 0;

    session = forth_session_create();
    ASSERT(session != NULL, "create failed");

    ASSERT(interpret_cstr(session, "ALIGN VARIABLE VX HERE VX -"),
           "variable layout");
    want[0] = 8;
    ASSERT(expect_cells(session, want, 1), "VARIABLE cell is after the name");

    ASSERT(interpret_cstr(session, "VARIABLE MV1 9 MV1 ! MV1 @"), "variable store");
    want[0] = 9;
    ASSERT(expect_cells(session, want, 1), "VARIABLE @");

    forth_output_clear(session);
    ASSERT(interpret_cstr(session, "S\"  hi\" TYPE"), "squote extra blanks");
    ASSERT(strcmp(forth_output(session), "hi") == 0, "S\" skips blanks after the word");

    forth_output_clear(session);
    ASSERT(interpret_cstr(session, "S\" aa\" S\" bb\" TYPE"), "squote clobber");
    ASSERT(strcmp(forth_output(session), "bb") == 0,
           "interpret S\" uses one transient buffer");
    ASSERT(forth_data_pop(session, &cell), "drop leftover S\" length");
    ASSERT(forth_data_pop(session, &cell), "drop leftover S\" address");

    ASSERT(interpret_cstr(session, ": BOOM 99 THROW ; ' BOOM CATCH"), "catch throw");
    want[0] = 99;
    ASSERT(expect_cells(session, want, 1), "CATCH of THROW is 99");

    ASSERT(interpret_cstr(session, ": OKW 7 ; ' OKW CATCH"), "catch ok");
    want[0] = 7;
    want[1] = 0;
    ASSERT(expect_cells(session, want, 2), "CATCH of a word is 0");

    ASSERT(interpret_cstr(session, ": INNER 42 THROW ; : WRAP ' INNER CATCH ; WRAP"),
           "catch in colon");
    want[0] = 42;
    ASSERT(expect_cells(session, want, 1), "CATCH compiled in a colon");

    ASSERT(interpret_cstr(session, ": NESTED INNER ; ' NESTED CATCH"),
           "throw from callee");
    want[0] = 42;
    ASSERT(expect_cells(session, want, 1), "THROW HALTs the outer NanoISA function");

    ASSERT(!interpret_cstr(session, "99 THROW"), "uncaught THROW fails interpret");
    ASSERT(!interpret_cstr(session, "' IF EXECUTE"), "EXECUTE of IF is rejected");

    other = forth_session_create();
    ASSERT(other != NULL, "second session");
    ASSERT(interpret_cstr(other, "5 CONSTANT K1"), "other CONSTANT");
    ASSERT(!interpret_cstr(other, "VX @"), "other session has no VX");
    ASSERT(interpret_cstr(session, "MV1 @"), "first session still has MV1");
    ASSERT(forth_data_pop(session, &cell) && cell == 9, "sessions do not share dict");
    ASSERT(interpret_cstr(other, "K1"), "other K1");
    ASSERT(forth_data_pop(other, &cell) && cell == 5, "other value");
    forth_session_destroy(other);

    ASSERT(interpret_cstr(session, ": FOOX BYE ; FOOX"), "compiled BYE");
    ASSERT(forth_exit_requested(session), "BYE requests process exit");
    ASSERT(!forth_exit_requested(NULL), "NULL session is not exiting");
    forth_session_destroy(session);
    PASS(test_name);
}

static void test_core_remaining_words(void) {
    const char *test_name = "core: >BODY >NUMBER POSTPONE ABORT\" KEY ACCEPT QUIT";
    ForthSession *session = forth_session_create();
    int64_t want[8];
    int64_t cell = 0;
    uint8_t ch = 0;

    ASSERT(session != NULL, "create failed");

    ASSERT(interpret_cstr(session, "CREATE GB ' GB >BODY GB ="), ">BODY");
    want[0] = -1;
    ASSERT(expect_cells(session, want, 1), "CREATE xt >BODY is the body");

    ASSERT(interpret_cstr(session, "0 0 S\" 123\" >NUMBER"), ">NUMBER 123");
    ASSERT(forth_data_pop(session, &cell) && cell == 0, ">NUMBER leftover count");
    ASSERT(forth_data_pop(session, &cell), ">NUMBER leftover addr");
    want[0] = 123;
    want[1] = 0;
    ASSERT(expect_cells(session, want, 2), ">NUMBER accumulates 123");

    ASSERT(interpret_cstr(session, "0 0 S\" 12X\" >NUMBER"), ">NUMBER stops");
    ASSERT(forth_data_pop(session, &cell) && cell == 1, "one leftover char");
    ASSERT(forth_data_pop(session, &cell), "caddr of X");
    ASSERT(forth_fetch_byte(session, (uint64_t)cell, &ch) && ch == (uint8_t)'X',
           "leftover is X");
    want[0] = 12;
    want[1] = 0;
    ASSERT(expect_cells(session, want, 2), ">NUMBER 12 from 12X");

    ASSERT(interpret_cstr(session,
                          ": LITPOST POSTPONE DUP ; IMMEDIATE : USES 7 LITPOST ; USES"),
           "postpone DUP");
    want[0] = 7;
    want[1] = 7;
    ASSERT(expect_cells(session, want, 2), "POSTPONE DUP compiled DUP");

    ASSERT(interpret_cstr(session,
                          ": IFPOST POSTPONE IF ; IMMEDIATE "
                          ": T IFPOST 1 THEN ; 0 T"),
           "postpone IF");
    ASSERT(forth_data_depth(session) == 0, "POSTPONE IF skips when false");
    ASSERT(interpret_cstr(session, "1 T"), "postpone IF true");
    want[0] = 1;
    ASSERT(expect_cells(session, want, 1), "POSTPONE IF takes the true path");

    ASSERT(interpret_cstr(session, "0 ABORT\" no\""), "ABORT\" false");
    ASSERT(forth_data_depth(session) == 0, "false ABORT\" is a no-op");
    ASSERT(!interpret_cstr(session, "1 ABORT\" yes\""), "ABORT\" true fails");

    ASSERT(interpret_cstr(session, "KEY A"), "KEY");
    want[0] = 65;
    ASSERT(expect_cells(session, want, 1), "KEY A is 65");

    ASSERT(interpret_cstr(session, "PAD 5 ACCEPT hello"), "ACCEPT");
    want[0] = 5;
    ASSERT(expect_cells(session, want, 1), "ACCEPT copies 5");
    forth_output_clear(session);
    ASSERT(interpret_cstr(session, "PAD 5 TYPE"), "type accepted");
    ASSERT(strcmp(forth_output(session), "hello") == 0, "ACCEPT text");

    ASSERT(interpret_cstr(session, "1 >R QUIT"), "QUIT");
    ASSERT(interpret_cstr(session, "STATE @"), "state after QUIT");
    want[0] = 0;
    ASSERT(expect_cells(session, want, 1), "QUIT enters interpretation state");
    ASSERT(forth_return_depth(session) == 0, "QUIT empties the return stack");

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
    test_colon_compile_verify_and_early_binding();
    test_colon_structured_control_flow();
    test_catch_and_throw();
    test_typed_import_abs();
    test_import_rejects_unusable_abi();
    test_import_stops_cop();
    test_see_disassembles_colon_and_import();
    test_interpret_numbers();
    test_interpret_plus_dup_star();
    test_core_arithmetic_stack_compare();
    test_core_memory_control_base();
    test_core_io_create_evaluate();
    test_kernel_defects();
    test_core_remaining_words();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
