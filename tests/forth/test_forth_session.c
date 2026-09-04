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
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
