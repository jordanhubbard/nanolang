/*
 * test_verifier.c — unit tests for nanoisa/verifier.c
 *
 * Tests nvm_verify() with valid and invalid NvmModule instances,
 * covering structural validation and per-function bytecode checks.
 */

#include "nanoisa/verifier.h"
#include "nanoisa/nvm_format.h"
#include "nanoisa/isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS(name) do { g_pass++; printf("  %-60s PASS\n", (name)); } while(0)
#define FAIL(name, msg) do { g_fail++; printf("  %-60s FAIL: %s\n", (name), (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(test_name, (msg)); return; } } while(0)

/* ── ISA emit helper (encodes one instruction into buf) ──────────────────── */

static uint32_t emit(uint8_t *buf, NanoOpcode op, ...) {
    DecodedInstruction instr = {0};
    instr.opcode = op;
    const InstructionInfo *info = isa_get_info(op);
    if (!info) return 0;
    va_list args;
    va_start(args, op);
    for (int i = 0; i < info->operand_count; i++) {
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

/* ── Helper: build a minimal valid module with one function ──────────────── */

static NvmModule *make_simple_module(const uint8_t *code, uint32_t code_size,
                                     uint16_t local_count, uint16_t upvalue_count) {
    NvmModule *mod = nvm_module_new();
    uint32_t name_idx = nvm_add_string(mod, "main", 4);
    uint32_t code_off = nvm_append_code(mod, code, code_size);
    NvmFunctionEntry fn = {0};
    fn.name_idx     = name_idx;
    fn.arity        = 0;
    fn.code_offset  = code_off;
    fn.code_length  = code_size;
    fn.local_count  = local_count;
    fn.upvalue_count = upvalue_count;
    uint32_t fn_idx = nvm_add_function(mod, &fn);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;
    return mod;
}

/* ── Tests ───────────────────────────────────────────────────────────────── */

static void test_null_module(void) {
    const char *test_name = "nvm_verify: NULL module returns error";
    NvmVerifyResult r = nvm_verify(NULL);
    ASSERT(!r.ok, "NULL module should fail verification");
    ASSERT(strlen(r.error_msg) > 0, "error_msg should be set");
    PASS(test_name);
}

static void test_empty_module_no_main(void) {
    const char *test_name = "nvm_verify: empty module (no functions) ok";
    NvmModule *mod = nvm_module_new();
    /* No functions, no main flag */
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "empty module should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_valid_simple_function(void) {
    const char *test_name = "nvm_verify: valid function with NOP+RET";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_NOP);
    off += emit(code + off, OP_PUSH_VOID);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "simple NOP+RET function should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_bad_entry_point(void) {
    const char *test_name = "nvm_verify: entry_point >= function_count fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    /* Set entry_point beyond function table */
    mod->header.entry_point = 99;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "bad entry_point should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_function_code_offset_overflow(void) {
    const char *test_name = "nvm_verify: function code_offset > code_size fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    /* Corrupt the function's code_offset so it's beyond the code section */
    mod->functions[0].code_offset = 9999;
    mod->header.entry_point = 0; /* still points to fn[0] */
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "code_offset > code_size should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_function_name_idx_overflow(void) {
    const char *test_name = "nvm_verify: function name_idx >= string_count fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    /* Corrupt the name index */
    mod->functions[0].name_idx = 9999;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "name_idx out-of-bounds should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_invalid_instruction_byte(void) {
    const char *test_name = "nvm_verify: unknown opcode byte fails";
    /* 0xFF is not a valid opcode */
    uint8_t code[] = { 0xFF };
    NvmModule *mod = make_simple_module(code, sizeof(code), 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "unknown opcode should fail verification");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_valid_jump_forward(void) {
    const char *test_name = "nvm_verify: valid forward JMP passes";
    /*
     * NOP (1 byte)
     * JMP +1 (5 bytes) -> jumps to RET
     * RET (1 byte)
     * Total: 7 bytes
     * JMP at pos=1, JMP is 5 bytes, so after JMP, pos=6
     * offset=+1 means target=1+1=2, which is 2. Wait, let me recalculate.
     *
     * The verifier checks: target = pos + offset, where pos is after opcode
     * Actually looking at the verifier: int32_t offset = instr.operands[0].i32;
     * int64_t target = (int64_t)pos + offset;
     * pos here is the instruction start position.
     *
     * NOP at pos=0 (1 byte), JMP at pos=1 (5 bytes), RET at pos=6
     * JMP offset=+5: target = 1 + 5 = 6, which is RET's position. Valid (target <= code_end=7).
     */
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_NOP);
    off += emit(code + off, OP_JMP, (int32_t)5); /* jump past the JMP itself to RET */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "valid forward jump should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_jump_out_of_bounds(void) {
    const char *test_name = "nvm_verify: JMP target outside function fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_NOP);
    off += emit(code + off, OP_JMP, (int32_t)9999); /* way out of bounds */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "out-of-bounds jump should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_jump_negative_out_of_bounds(void) {
    const char *test_name = "nvm_verify: JMP target negative offset fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_JMP, (int32_t)-100); /* negative out of bounds */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "negative out-of-bounds jump should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_jmp_true_valid(void) {
    const char *test_name = "nvm_verify: valid JMP_TRUE passes";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_NOP);
    off += emit(code + off, OP_JMP_TRUE, (int32_t)5);  /* jump to RET */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "valid JMP_TRUE should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_jmp_false_out_of_bounds(void) {
    const char *test_name = "nvm_verify: JMP_FALSE target out of bounds fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_JMP_FALSE, (int32_t)9999);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "out-of-bounds JMP_FALSE should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_call_valid(void) {
    const char *test_name = "nvm_verify: OP_CALL to valid function index passes";
    uint8_t code[16];
    uint32_t off = 0;
    /* OP_CALL 0 calls function[0] (itself) - recursive but valid index */
    off += emit(code + off, OP_CALL, (uint32_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_CALL to valid function should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_call_invalid_fn_idx(void) {
    const char *test_name = "nvm_verify: OP_CALL to invalid function index fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL, (uint32_t)999); /* only fn[0] exists */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_CALL with bad fn_idx should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_push_str_valid(void) {
    const char *test_name = "nvm_verify: OP_PUSH_STR to valid string index passes";
    NvmModule *mod = nvm_module_new();
    uint32_t str_idx = nvm_add_string(mod, "hello", 5);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_STR, str_idx);
    off += emit(code + off, OP_RET);
    uint32_t code_off = nvm_append_code(mod, code, off);

    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    uint32_t fn_idx = nvm_add_function(mod, &fn);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_PUSH_STR with valid str_idx should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_push_str_invalid(void) {
    const char *test_name = "nvm_verify: OP_PUSH_STR with invalid string index fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_STR, (uint32_t)9999);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_PUSH_STR with bad str_idx should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_load_local_valid(void) {
    const char *test_name = "nvm_verify: OP_LOAD_LOCAL valid slot passes";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_LOAD_LOCAL, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 1 /* local_count */, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_LOAD_LOCAL slot 0 with local_count=1 should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_load_local_invalid(void) {
    const char *test_name = "nvm_verify: OP_LOAD_LOCAL slot >= local_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_LOAD_LOCAL, (uint16_t)5);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 2 /* local_count=2, slot=5 */, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_LOAD_LOCAL slot >= local_count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_store_local_invalid(void) {
    const char *test_name = "nvm_verify: OP_STORE_LOCAL slot >= local_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_STORE_LOCAL, (uint16_t)10);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 3, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_STORE_LOCAL slot >= local_count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_load_upvalue_valid(void) {
    const char *test_name = "nvm_verify: OP_LOAD_UPVALUE valid slot passes";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_LOAD_UPVALUE, (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 1 /* upvalue_count */);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_LOAD_UPVALUE slot 0 with upvalue_count=1 should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_load_upvalue_invalid(void) {
    const char *test_name = "nvm_verify: OP_LOAD_UPVALUE slot >= upvalue_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_LOAD_UPVALUE, (uint16_t)5, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 2 /* upvalue_count=2, slot=5 */);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_LOAD_UPVALUE slot >= upvalue_count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_closure_new_invalid(void) {
    const char *test_name = "nvm_verify: OP_CLOSURE_NEW with invalid fn_idx fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CLOSURE_NEW, (uint32_t)999, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_CLOSURE_NEW with bad fn_idx should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_closure_new_valid(void) {
    const char *test_name = "nvm_verify: OP_CLOSURE_NEW with valid fn_idx passes";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CLOSURE_NEW, (uint32_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_CLOSURE_NEW with valid fn_idx should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_match_tag_out_of_bounds(void) {
    const char *test_name = "nvm_verify: OP_MATCH_TAG target out of bounds fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_MATCH_TAG, (uint16_t)0, (int32_t)9999);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_MATCH_TAG with out-of-bounds target should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_null_code_nonzero_size(void) {
    const char *test_name = "nvm_verify: null code pointer with code_size > 0 fails";
    NvmModule *mod = nvm_module_new();
    mod->code = NULL;
    mod->code_size = 100; /* Non-zero size with NULL pointer */
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "null code with non-zero size should fail");
    /* Reset to avoid free issues */
    mod->code_size = 0;
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_call_extern_invalid(void) {
    const char *test_name = "nvm_verify: OP_CALL_EXTERN with invalid import index fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_EXTERN, (uint32_t)999);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    /* No imports registered, so import_count=0 */
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_CALL_EXTERN with no imports should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_arithmetic_instructions(void) {
    const char *test_name = "nvm_verify: arithmetic opcodes pass verification";
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_PUSH_I64, (int64_t)3);
    off += emit(code + off, OP_SUB);
    off += emit(code + off, OP_PUSH_I64, (int64_t)4);
    off += emit(code + off, OP_MUL);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_DIV);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "arithmetic opcodes should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[verifier] NanoVM bytecode verifier tests...\n\n");

    test_null_module();
    test_empty_module_no_main();
    test_valid_simple_function();
    test_bad_entry_point();
    test_function_code_offset_overflow();
    test_function_name_idx_overflow();
    test_invalid_instruction_byte();
    test_valid_jump_forward();
    test_jump_out_of_bounds();
    test_jump_negative_out_of_bounds();
    test_jmp_true_valid();
    test_jmp_false_out_of_bounds();
    test_op_call_valid();
    test_op_call_invalid_fn_idx();
    test_op_push_str_valid();
    test_op_push_str_invalid();
    test_op_load_local_valid();
    test_op_load_local_invalid();
    test_op_store_local_invalid();
    test_op_load_upvalue_valid();
    test_op_load_upvalue_invalid();
    test_op_closure_new_invalid();
    test_op_closure_new_valid();
    test_match_tag_out_of_bounds();
    test_null_code_nonzero_size();
    test_call_extern_invalid();
    test_arithmetic_instructions();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
