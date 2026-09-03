/*
 * test_verifier.c — unit tests for nanoisa/verifier.c
 *
 * Tests nvm_verify() with valid and invalid NvmModule instances,
 * covering structural validation and per-function bytecode checks.
 */

#include "nanoisa/verifier.h"
#include "nanoisa/nvm_format.h"
#include "nanoisa/isa.h"
#include "nanovm/vm.h"
#include "nanovm/vm_decode.h"
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

/* A function that leaves a value on the stack returns it, so it has to say
 * so: OP_RET traps at run time when the count on the stack differs from the
 * declared result_count, and the verifier now proves that statically. These
 * fixtures compute one value and return, so they declare one. */
static void declares_one_result(NvmModule *mod) {
    mod->functions[0].result_tag = TAG_INT;
    mod->functions[0].result_count = 1;
}

/* ── Types through basic blocks ──────────────────────────────────────────
 *
 * Every rule the type pass enforces restates a check the VM already performs
 * at run time: I64_ADD traps with "requires two integers", F64_ADD with
 * "requires two floats". Proving them statically turns a trap into a
 * rejection. A slot whose type is not known never fails, so imprecision costs
 * only missed diagnostics; a known contradiction always fails.
 */

static void test_integer_op_on_a_string_fails(void) {
    const char *test_name = "types: an integer op on a string operand fails";
    uint8_t code[64];
    uint32_t n = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t s_idx = nvm_add_string(mod, "hello", 5);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);
    n += emit(code + n, OP_PUSH_STR, s_idx);
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_I64_ADD);
    n += emit(code + n, OP_RET);
    uint32_t off = nvm_append_code(mod, code, n);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx; fn.code_offset = off; fn.code_length = n;
    fn.result_tag = TAG_INT; fn.result_count = 1;
    uint32_t idx = nvm_add_function(mod, &fn);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = idx;

    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "adding an integer to a string must fail");
    ASSERT(strstr(r.error_msg, "expects int") != NULL, r.error_msg);
    ASSERT(strstr(r.error_msg, "string") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_integer_op_on_floats_fails(void) {
    const char *test_name = "types: an integer op on float operands fails";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_F64, 1.5);
    n += emit(code + n, OP_PUSH_F64, 2.5);
    n += emit(code + n, OP_I64_ADD);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "integer add on floats must fail");
    ASSERT(strstr(r.error_msg, "float") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_float_op_on_floats_passes(void) {
    const char *test_name = "types: the matching float op verifies";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_F64, 1.5);
    n += emit(code + n, OP_PUSH_F64, 2.5);
    n += emit(code + n, OP_F64_ADD);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    mod->functions[0].result_tag = TAG_FLOAT;
    mod->functions[0].result_count = 1;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

/* A value loaded from a local carries no type in a v1 module, so nothing is
 * known about it and nothing may be rejected. This is the property that keeps
 * the analysis from rejecting working programs. */
static void test_unknown_types_never_fail(void) {
    const char *test_name = "types: an unknown operand type is not an error";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_LOAD_LOCAL, (uint16_t)0);
    n += emit(code + n, OP_LOAD_LOCAL, (uint16_t)0);
    n += emit(code + n, OP_I64_ADD);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 1, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

/* Two paths reaching one instruction with different known types widen to
 * unknown rather than failing. Generated code does this whenever one arm of a
 * conditional pushes a value and another pushes void, so rejecting it would
 * reject working programs -- and the widened slot is then simply not known,
 * which is the truth. */
static void test_conflicting_merge_widens_rather_than_failing(void) {
    const char *test_name = "types: a merge of different types widens to unknown";
    uint8_t code[128];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_BOOL, 1);
    uint32_t jf_at = n;
    n += emit(code + n, OP_JMP_FALSE, (int32_t)0);
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    uint32_t jmp_at = n;
    n += emit(code + n, OP_JMP, (int32_t)0);
    uint32_t else_at = n;
    n += emit(code + n, OP_PUSH_F64, 1.0);
    uint32_t join_at = n;
    n += emit(code + n, OP_RET);

    int32_t rel = (int32_t)else_at - (int32_t)jf_at;
    memcpy(code + jf_at + 1, &rel, 4);
    rel = (int32_t)join_at - (int32_t)jmp_at;
    memcpy(code + jmp_at + 1, &rel, 4);

    NvmModule *mod = make_simple_module(code, n, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

/* ── Ownership effects ───────────────────────────────────────────────────
 *
 * GC_RETAIN and GC_RELEASE adjust an object's reference count without touching
 * the operand stack, so stack height says nothing about whether they pair up.
 * An unbalanced pair is a leak or a premature free, and both stay invisible
 * until long after the instruction that caused them. Nothing emits these
 * today, but the assembler accepts them -- which is exactly the path with no
 * other check.
 */

static void test_unbalanced_retain_at_return_fails(void) {
    const char *test_name = "ownership: returning while still holding a retain fails";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_GC_RETAIN);    /* retains the top, no release */
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "an unreleased retain must be caught");
    ASSERT(strstr(r.error_msg, "retained") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_release_without_retain_fails(void) {
    const char *test_name = "ownership: releasing a reference never retained fails";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_GC_RELEASE);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "an unmatched release must be caught");
    ASSERT(strstr(r.error_msg, "does not hold") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_balanced_retain_release_passes(void) {
    const char *test_name = "ownership: a matched retain/release pair verifies";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_GC_RETAIN);
    n += emit(code + n, OP_GC_RELEASE);   /* release also consumes the value */
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

/* One branch retains and the other does not, so the balance at the join
 * depends on which way control went. Stack height agrees on both paths, so
 * only the ownership check can see this. */
static void test_branch_dependent_ownership_fails(void) {
    const char *test_name = "ownership: a balance that depends on the branch taken fails";
    uint8_t code[128];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_PUSH_BOOL, 1);
    n += emit(code + n, OP_JMP_FALSE, (int32_t)0);
    uint32_t jf_at = n - 5;                       /* patch below */
    n += emit(code + n, OP_GC_RETAIN);
    uint32_t join = n;
    n += emit(code + n, OP_GC_RELEASE);
    n += emit(code + n, OP_RET);
    /* Point the false branch at the release, skipping the retain. */
    int32_t rel = (int32_t)join - (int32_t)(jf_at);
    code[jf_at + 1] = (uint8_t)rel;
    code[jf_at + 2] = (uint8_t)(rel >> 8);
    code[jf_at + 3] = (uint8_t)(rel >> 16);
    code[jf_at + 4] = (uint8_t)(rel >> 24);

    NvmModule *mod = make_simple_module(code, n, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "a path-dependent ownership balance must be caught");
    nvm_module_free(mod);
    PASS(test_name);
}

/* ── Return shape ────────────────────────────────────────────────────────
 *
 * OP_RET traps when the number of values on the stack differs from the
 * function's declared result_count, and reaching the end of a function's code
 * is an implicit return the VM checks the same way. Both were run-time-only
 * failures; the verifier now proves them statically.
 */

static void test_returning_more_than_declared_fails(void) {
    const char *test_name = "return shape: returning more values than declared fails";
    uint8_t code[64];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_PUSH_I64, (int64_t)2);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "two values with one declared result must fail");
    ASSERT(strstr(r.error_msg, "declares") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_returning_fewer_than_declared_fails(void) {
    const char *test_name = "return shape: returning nothing when a result is declared fails";
    uint8_t code[32];
    uint32_t n = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "an empty stack with one declared result must fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_implicit_return_shape_is_checked(void) {
    const char *test_name = "return shape: the implicit return at end of code is checked too";
    /* No RET at all. The VM treats reaching the end as a return and checks the
     * result count there, so leaving a value while declaring none is just as
     * wrong as doing it explicitly -- and used to verify clean. */
    uint8_t code[32];
    uint32_t n = emit(code, OP_PUSH_I64, (int64_t)7);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "falling off the end with a value and no declared result must fail");
    ASSERT(strstr(r.error_msg, "reaches its end") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_implicit_return_with_matching_shape_passes(void) {
    const char *test_name = "return shape: a clean implicit return is still legal";
    uint8_t code[32];
    uint32_t n = emit(code, OP_NOP);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

/* ── Maximum operand depth ───────────────────────────────────────────────
 *
 * verify_stack_heights already walks every reachable instruction and knows the
 * stack height at each one; it just discarded the maximum. Returning it is what
 * lets a v2 producer declare max_stack and a loader confirm it, so the value a
 * module carries is one the verifier has agreed to rather than one it trusts.
 */

static void test_max_stack_of_an_empty_function(void) {
    const char *test_name = "max_stack: a function with no instructions is 0";
    NvmModule *mod = make_simple_module(NULL, 0, 0, 0);
    uint16_t depth = 0xFFFF;
    NvmVerifyResult r = nvm_verify_function_max_stack(mod, 0, &depth);
    ASSERT(r.ok, r.error_msg);
    ASSERT(depth == 0, "an empty function needs no operand slots");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_max_stack_counts_the_deepest_point(void) {
    const char *test_name = "max_stack: reports the deepest reachable height";
    /* push, push, push, add (3 -> 2), ret. The deepest point is 3, which is
     * neither the first nor the last height -- a maximum that only tracked the
     * end would report 2 and a module declaring 2 would then overflow. */
    uint8_t code[128];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_PUSH_I64, (int64_t)2);
    n += emit(code + n, OP_PUSH_I64, (int64_t)3);
    n += emit(code + n, OP_ADD);
    /* Down to nothing before returning: the function declares no results, and
     * the verifier now requires a return to leave exactly what is declared. */
    n += emit(code + n, OP_POP);
    n += emit(code + n, OP_POP);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    uint16_t depth = 0;
    NvmVerifyResult r = nvm_verify_function_max_stack(mod, 0, &depth);
    ASSERT(r.ok, r.error_msg);
    ASSERT(depth == 3, "three values are live at once before the add");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_max_stack_is_refused_for_an_unverifiable_function(void) {
    const char *test_name = "max_stack: an underflowing function reports no depth";
    /* A bare ADD underflows. There is no honest maximum for code the verifier
     * rejects, so the failure propagates rather than yielding a number. */
    uint8_t code[64];
    uint32_t n = emit(code, OP_ADD);
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    uint16_t depth = 0;
    NvmVerifyResult r = nvm_verify_function_max_stack(mod, 0, &depth);
    ASSERT(!r.ok, "an underflowing function must not report a depth");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_max_stack_out_of_range_function(void) {
    const char *test_name = "max_stack: a function index past the table fails";
    NvmModule *mod = make_simple_module(NULL, 0, 0, 0);
    uint16_t depth = 0;
    NvmVerifyResult r = nvm_verify_function_max_stack(mod, 7, &depth);
    ASSERT(!r.ok, "index 7 does not exist");
    nvm_module_free(mod);
    PASS(test_name);
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
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_NOP);
    off += emit(code + off, OP_PUSH_VOID);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    declares_one_result(mod);
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

static void test_function_code_length_overflow(void) {
    const char *test_name = "nvm_verify: wrapped function code range fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    /* The old offset+length check wrapped back below code_size. */
    mod->functions[0].code_offset = UINT32_MAX - 1;
    mod->functions[0].code_length = 2;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "wrapped code range should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static NvmModule *make_two_function_module(void) {
    uint8_t code[4];
    uint32_t code_size = 0;
    code_size += emit(code + code_size, OP_RET);
    code_size += emit(code + code_size, OP_RET);

    NvmModule *mod = make_simple_module(code, code_size, 0, 0);
    mod->functions[0].code_length = code_size / 2;
    NvmFunctionEntry fn = mod->functions[0];
    fn.code_offset = code_size / 2;
    nvm_add_function(mod, &fn);
    return mod;
}

static void test_adjacent_function_code_ranges(void) {
    const char *test_name = "nvm_verify: adjacent function code ranges pass";
    NvmModule *mod = make_two_function_module();
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "adjacent function ranges should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_overlapping_function_code_ranges(void) {
    const char *test_name = "nvm_verify: overlapping function code ranges fail";
    NvmModule *mod = make_two_function_module();
    mod->functions[1].code_offset = 0;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "identical function ranges should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_contained_function_code_range(void) {
    const char *test_name = "nvm_verify: contained function code range fails";
    NvmModule *mod = make_two_function_module();
    mod->functions[0].code_length = mod->code_size;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "contained function range should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_empty_function_code_ranges(void) {
    const char *test_name = "nvm_verify: empty function ranges own no bytes";
    NvmModule *mod = make_two_function_module();
    mod->functions[0].code_offset = 0;
    mod->functions[0].code_length = 0;
    mod->functions[1].code_offset = 0;
    mod->functions[1].code_length = mod->code_size;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "empty function at non-empty range start should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_function_code_length_beyond_end(void) {
    const char *test_name = "nvm_verify: code_length beyond code_size fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    /* Valid offset that does not wrap: the length simply runs past the end of
     * the code section. Distinct from the wrapped case above, which the
     * subtraction-form check catches for a different reason. */
    mod->functions[0].code_offset = 0;
    mod->functions[0].code_length = mod->code_size + 100u;
    mod->header.entry_point = 0;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "over-long code range must fail verification");
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

static void test_invalid_function_result_signature(void) {
    const char *test_name = "nvm_verify: invalid function result signature fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->functions[0].result_tag = TAG_INT;
    mod->functions[0].result_count = 0;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "non-void tag with zero results should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_multiple_function_results(void) {
    const char *test_name = "nvm_verify: multiple homogeneous results pass";
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->functions[0].result_tag = TAG_INT;
    mod->functions[0].result_count = 2;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "two integer results should verify");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_invalid_function_result_tag(void) {
    const char *test_name = "nvm_verify: invalid function result tag fails";
    uint8_t code[4];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->functions[0].result_tag = TAG_COUNT;
    mod->functions[0].result_count = 1;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "out-of-range result tag should fail");
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
    off += emit(code + off, OP_PUSH_BOOL, 1);
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

static void test_jump_into_operand_fails(void) {
    const char *test_name = "nvm_verify: JMP target inside operand fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_JMP, (int32_t)2);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "jump into its encoded operand should fail");
    ASSERT(strstr(r.error_msg, "instruction boundary") != NULL,
           "failure should identify the instruction-boundary violation");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_match_tag_into_operand_fails(void) {
    const char *test_name = "nvm_verify: MATCH_TAG target inside operand fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_MATCH_TAG, (uint16_t)0, (int32_t)2);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "MATCH_TAG into its encoded operand should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_branch_target_past_function_end_fails(void) {
    const char *test_name = "nvm_verify: branch past function end fails";
    uint8_t code[16];
    uint32_t off = emit(code, OP_JMP, (int32_t)6);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "branch past the function end should fail");
    ASSERT(strstr(r.error_msg, "instruction boundary") != NULL,
           "failure should identify the invalid control-flow target");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_jump_to_function_end_passes(void) {
    const char *test_name = "nvm_verify: JMP target at function end passes";
    uint8_t code[16];
    uint32_t off = emit(code, OP_JMP, (int32_t)5);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "existing jump-to-end behavior should remain valid");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_predecoded_module_boundaries(void) {
    const char *test_name = "vm_decode: module records instructions and boundaries";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    VmDecodedModule decoded;
    char error[VM_DECODE_ERROR_SIZE];
    ASSERT(vm_decode_module(mod, &decoded, error), error);
    ASSERT(decoded.function_count == 1, "one function should be decoded");
    ASSERT(decoded.functions[0].instruction_count == 2,
           "two instructions should be decoded");
    ASSERT(decoded.functions[0].instructions[0].byte_offset == 0,
           "first instruction should begin at zero");
    ASSERT(decoded.functions[0].instructions[1].byte_offset == 9,
           "RET should follow the encoded i64 instruction");
    ASSERT(vm_decoded_function_has_boundary(&decoded.functions[0], 0),
           "zero should be an instruction boundary");
    ASSERT(!vm_decoded_function_has_boundary(&decoded.functions[0], 1),
           "operand bytes should not be instruction boundaries");
    ASSERT(vm_decoded_function_has_boundary(&decoded.functions[0], off),
           "function end should be a boundary sentinel");
    vm_decoded_module_free(&decoded);
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

static void test_tail_call_signatures(void) {
    const char *test_name = "nvm_verify: OP_TAIL_CALL validates target and result signature";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_TAIL_CALL, (uint32_t)0);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->functions[0].result_tag = TAG_INT;
    mod->functions[0].result_count = 1;
    ASSERT(nvm_verify(mod).ok, "compatible recursive tail call should pass");
    mod->code[1] = 1;
    ASSERT(!nvm_verify(mod).ok, "tail call with bad function index should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_call_arity_underflow(void) {
    const char *test_name = "nvm_verify: OP_CALL requires callee arity";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL, (uint32_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 1, 0);
    mod->functions[0].arity = 1;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "call without its declared argument should fail");
    ASSERT(strstr(r.error_msg, "stack underflow") != NULL,
           "failure should identify call arity underflow");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_call_result_shape(void) {
    const char *test_name = "nvm_verify: OP_CALL contributes declared result count";
    uint8_t code[32];
    uint32_t off = 0;
    /* The function declares one result, so RET consumes one. PUSH then CALL
     * then a single POP leaves exactly that one value -- but only if CALL
     * pushed its declared result. Had it contributed nothing, the POP would
     * have taken the pushed value and RET would underflow, which is what makes
     * this fixture a test of the result count rather than of nothing. */
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_CALL, (uint32_t)0);
    off += emit(code + off, OP_POP);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->functions[0].result_tag = TAG_INT;
    mod->functions[0].result_count = 1;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
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
    fn.result_tag = TAG_STRING;   /* the pushed string is the result */
    fn.result_count = 1;
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
    declares_one_result(mod);
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
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_LOAD_UPVALUE slot 0 with upvalue_count=1 should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_load_upvalue_invalid(void) {
    const char *test_name = "nvm_verify: OP_LOAD_UPVALUE index >= upvalue_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    /* depth=0, idx=5 — idx out of range for upvalue_count=2 */
    off += emit(code + off, OP_LOAD_UPVALUE, (uint16_t)0, (uint16_t)5);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 2 /* upvalue_count=2, idx=5 */);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_LOAD_UPVALUE index >= upvalue_count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_upvalue_depth_invalid(void) {
    const char *test_name = "nvm_verify: flattened upvalue depth must be zero";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_LOAD_UPVALUE, (uint16_t)1, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 1);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "nonzero flattened upvalue depth should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_global_index_invalid(void) {
    const char *test_name = "nvm_verify: global index beyond VM table fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_LOAD_GLOBAL, (uint32_t)VM_MAX_GLOBALS);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "global index at the fixed limit should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_struct_new_valid(void) {
    const char *test_name = "nvm_verify: OP_STRUCT_NEW with valid def_idx passes";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_STRUCT_NEW, (uint32_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->struct_count = 2;  /* def_idx=0 is valid */
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_STRUCT_NEW def_idx=0 with struct_count=2 should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_struct_new_invalid(void) {
    const char *test_name = "nvm_verify: OP_STRUCT_NEW with def_idx >= struct_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_STRUCT_NEW, (uint32_t)5);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->struct_count = 3;  /* def_idx=5 is out of range */
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_STRUCT_NEW def_idx >= struct_count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_struct_new_zero_count_skips(void) {
    const char *test_name = "nvm_verify: OP_STRUCT_NEW with struct_count=0 skips validation";
    uint8_t code[16];
    uint32_t off = 0;
    /* Any def_idx should be tolerated when struct_count is 0 (module has no
     * struct definitions registered — e.g., hand-assembled bytecode). */
    off += emit(code + off, OP_STRUCT_NEW, (uint32_t)999);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->struct_count = 0;
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_STRUCT_NEW with struct_count=0 should skip validation");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_enum_val_invalid(void) {
    const char *test_name = "nvm_verify: OP_ENUM_VAL with def_idx >= enum_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_ENUM_VAL, (uint32_t)4, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->enum_count = 2;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_ENUM_VAL def_idx >= enum_count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_op_union_construct_invalid(void) {
    const char *test_name = "nvm_verify: OP_UNION_CONSTRUCT with def_idx >= union_count fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_UNION_CONSTRUCT, (uint32_t)7, (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    mod->union_count = 4;
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "OP_UNION_CONSTRUCT def_idx >= union_count should fail");
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
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "OP_CLOSURE_NEW with valid fn_idx should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_closure_capture_count_mismatch(void) {
    const char *test_name = "nvm_verify: closure capture count matches callee upvalues";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CLOSURE_NEW, (uint32_t)0, (uint16_t)1);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "mismatched closure capture count should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_aggregate_count_underflow(void) {
    const char *test_name = "nvm_verify: aggregate count requires enough fields";
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_AGG_PACK, (int)AGG_TUPLE, (uint32_t)0,
                (uint16_t)0, (uint16_t)2);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "aggregate with too few fields should fail");
    ASSERT(strstr(r.error_msg, "stack underflow") != NULL,
           "failure should identify aggregate count underflow");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_invalid_type_check_tag(void) {
    const char *test_name = "nvm_verify: OP_TYPE_CHECK rejects invalid tag";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_VOID);
    off += emit(code + off, OP_TYPE_CHECK, (int)TAG_COUNT);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "out-of-range type-check tag should fail");
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

static void test_call_extern_valid_signature(void) {
    const char *test_name = "nvm_verify: OP_CALL_EXTERN with valid import signature passes";
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_CALL_EXTERN, (uint32_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    uint32_t mname = nvm_add_string(mod, "mathlib", 7);
    uint32_t fname = nvm_add_string(mod, "add", 3);
    uint8_t params[2] = { TAG_INT, TAG_INT };
    nvm_add_import(mod, mname, fname, 2, TAG_INT, params);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "extern call to a well-formed import signature should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_import_signature_valid(void) {
    const char *test_name = "nvm_verify: import with valid signature tags passes";
    uint8_t code[8];
    uint32_t off = emit(code, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    uint32_t mname = nvm_add_string(mod, "mathlib", 7);
    uint32_t fname = nvm_add_string(mod, "add", 3);
    uint8_t params[2] = { TAG_INT, TAG_FLOAT };
    nvm_add_import(mod, mname, fname, 2, TAG_INT, params);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_import_bad_return_type(void) {
    const char *test_name = "nvm_verify: import with invalid return type tag fails";
    NvmModule *mod = make_simple_module((const uint8_t[]){0x00}, 0, 0, 0);
    uint32_t mname = nvm_add_string(mod, "lib", 3);
    uint32_t fname = nvm_add_string(mod, "f", 1);
    nvm_add_import(mod, mname, fname, 0, (uint8_t)TAG_COUNT, NULL);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "import with out-of-range return type should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_import_bad_param_type(void) {
    const char *test_name = "nvm_verify: import with invalid param type tag fails";
    NvmModule *mod = make_simple_module((const uint8_t[]){0x00}, 0, 0, 0);
    uint32_t mname = nvm_add_string(mod, "lib", 3);
    uint32_t fname = nvm_add_string(mod, "f", 1);
    uint8_t params[1] = { (uint8_t)TAG_COUNT };
    nvm_add_import(mod, mname, fname, 1, TAG_VOID, params);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "import with out-of-range param type should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_import_at_arg_limit_verifies(void) {
    const char *test_name = "nvm_verify: import with NANO_MAX_FFI_ARGS params verifies";
    NvmModule *mod = make_simple_module((const uint8_t[]){0x00}, 0, 0, 0);
    uint32_t mname = nvm_add_string(mod, "lib", 3);
    uint32_t fname = nvm_add_string(mod, "f", 1);
    uint8_t params[NANO_MAX_FFI_ARGS];
    for (int i = 0; i < NANO_MAX_FFI_ARGS; i++) params[i] = TAG_INT;
    nvm_add_import(mod, mname, fname, (uint16_t)NANO_MAX_FFI_ARGS, TAG_INT, params);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "import at the foreign-call argument limit should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_import_over_arg_limit_fails(void) {
    const char *test_name = "nvm_verify: import exceeding NANO_MAX_FFI_ARGS fails";
    NvmModule *mod = make_simple_module((const uint8_t[]){0x00}, 0, 0, 0);
    uint32_t mname = nvm_add_string(mod, "lib", 3);
    uint32_t fname = nvm_add_string(mod, "f", 1);
    uint8_t params[NANO_MAX_FFI_ARGS + 1];
    for (int i = 0; i < NANO_MAX_FFI_ARGS + 1; i++) params[i] = TAG_INT;
    nvm_add_import(mod, mname, fname, (uint16_t)(NANO_MAX_FFI_ARGS + 1), TAG_INT, params);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "import above the foreign-call argument limit should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_call_module_recognized(void) {
    const char *test_name = "nvm_verify: OP_CALL_MODULE is a recognized linked call";
    uint8_t code[16];
    uint32_t off = 0;
    /* No arguments, one result: the encoded shape is what makes this
     * verifiable without a linked-module table. */
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)0, (uint32_t)0,
                (uint16_t)0, (uint16_t)1);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_call_module_linked_valid(void) {
    const char *test_name = "nvm_verify_linked: OP_CALL_MODULE resolves against linked table";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)0, (uint32_t)0,
                (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *caller = make_simple_module(code, off, 0, 0);

    uint8_t callee_code[8];
    uint32_t coff = 0;
    coff += emit(callee_code + coff, OP_RET);
    NvmModule *callee = make_simple_module(callee_code, coff, 0, 0);

    const NvmModule *table[1] = { callee };
    NvmVerifyResult r = nvm_verify_linked(caller, table, 1);
    ASSERT(r.ok, "in-range linked call should verify");
    nvm_module_free(caller);
    nvm_module_free(callee);
    PASS(test_name);
}

/* The call site declares the shape its own stack discipline was proven
 * against. If linking binds it to a callee of a different shape, the proof no
 * longer describes what will run -- so linking is where the two are compared.
 * Before the shape was encoded there was nothing to compare. */
static void test_call_module_shape_mismatch_fails(void) {
    const char *test_name = "nvm_verify_linked: declared call shape must match the callee";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)0, (uint32_t)0,
                (uint16_t)2, (uint16_t)0);   /* claims two arguments */
    off += emit(code + off, OP_RET);
    NvmModule *caller = make_simple_module(code, off, 0, 0);

    uint8_t callee_code[8];
    uint32_t coff = emit(callee_code, OP_RET);
    NvmModule *callee = make_simple_module(callee_code, coff, 0, 0);  /* takes none */

    const NvmModule *table[1] = { callee };
    NvmVerifyResult r = nvm_verify_linked(caller, table, 1);
    ASSERT(!r.ok, "a call declaring two arguments to a nullary callee must fail");
    ASSERT(strstr(r.error_msg, "arity") != NULL,
           "the message should name the mismatch");
    nvm_module_free(caller);
    nvm_module_free(callee);
    PASS(test_name);
}

/* Fail closed: an instruction with no known stack effect used to be skipped,
 * which also skipped enqueueing its successors, so the walk stopped and
 * everything after it went unverified while nvm_verify still returned ok.
 * With every effect now declared, the way to observe the rule is a module
 * whose stack discipline is only checkable because the walk continues past a
 * portable-ISA instruction. */
static void test_verification_continues_past_portable_isa_instructions(void) {
    const char *test_name = "nvm_verify: the walk does not stop at a portable-ISA opcode";
    uint8_t code[128];
    uint32_t n = 0;
    n += emit(code + n, OP_PUSH_I64, (int64_t)1);
    n += emit(code + n, OP_PUSH_I64, (int64_t)2);
    n += emit(code + n, OP_I64_ADD);      /* 2 -> 1 */
    n += emit(code + n, OP_POP);
    n += emit(code + n, OP_POP);          /* underflows: nothing left */
    n += emit(code + n, OP_RET);
    NvmModule *mod = make_simple_module(code, n, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "the second POP underflows and must be caught");
    ASSERT(strstr(r.error_msg, "underflow") != NULL, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_call_module_linked_bad_module_idx(void) {
    const char *test_name = "nvm_verify_linked: OP_CALL_MODULE module index out of range fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)5, (uint32_t)0,
                (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *caller = make_simple_module(code, off, 0, 0);

    uint8_t callee_code[8];
    uint32_t coff = 0;
    coff += emit(callee_code + coff, OP_RET);
    NvmModule *callee = make_simple_module(callee_code, coff, 0, 0);

    const NvmModule *table[1] = { callee };
    NvmVerifyResult r = nvm_verify_linked(caller, table, 1);
    ASSERT(!r.ok, "module index >= linked_count should fail");
    ASSERT(strstr(r.error_msg, "module_idx") != NULL,
           "failure should identify the module index");
    nvm_module_free(caller);
    nvm_module_free(callee);
    PASS(test_name);
}

static void test_call_module_linked_unresolved(void) {
    const char *test_name = "nvm_verify_linked: OP_CALL_MODULE NULL linked module fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)0, (uint32_t)0,
                (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *caller = make_simple_module(code, off, 0, 0);

    const NvmModule *table[1] = { NULL };
    NvmVerifyResult r = nvm_verify_linked(caller, table, 1);
    ASSERT(!r.ok, "unresolved (NULL) linked module should fail");
    ASSERT(strstr(r.error_msg, "unresolved") != NULL,
           "failure should identify the unresolved link");
    nvm_module_free(caller);
    PASS(test_name);
}

static void test_call_module_linked_bad_fn_idx(void) {
    const char *test_name = "nvm_verify_linked: OP_CALL_MODULE callee fn index out of range fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)0, (uint32_t)9,
                (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *caller = make_simple_module(code, off, 0, 0);

    uint8_t callee_code[8];
    uint32_t coff = 0;
    coff += emit(callee_code + coff, OP_RET);
    NvmModule *callee = make_simple_module(callee_code, coff, 0, 0);

    const NvmModule *table[1] = { callee };
    NvmVerifyResult r = nvm_verify_linked(caller, table, 1);
    ASSERT(!r.ok, "callee function index >= function_count should fail");
    ASSERT(strstr(r.error_msg, "linked function_count") != NULL,
           "failure should identify the linked function bound");
    nvm_module_free(caller);
    nvm_module_free(callee);
    PASS(test_name);
}

static void test_call_module_no_table_still_ok(void) {
    const char *test_name = "nvm_verify_linked: OP_CALL_MODULE with empty table verifies structurally";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_CALL_MODULE, (uint32_t)3, (uint32_t)7,
                (uint16_t)0, (uint16_t)0);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify_linked(mod, NULL, 0);
    /* Without a table the module and function indices cannot be resolved, but
     * the encoded call shape still lets the stack walk continue -- which is why
     * the shape is encoded rather than looked up. */
    ASSERT(r.ok, r.error_msg);
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_arr_new_bad_type_tag(void) {
    const char *test_name = "nvm_verify: OP_ARR_NEW with invalid element type tag fails";
    uint8_t code[8];
    uint32_t off = 0;
    off += emit(code + off, OP_ARR_NEW, (int)TAG_COUNT + 3);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "type tag >= TAG_COUNT should fail");
    ASSERT(strstr(r.error_msg, "TAG_COUNT") != NULL,
           "failure should identify the invalid type tag");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_hm_new_bad_value_tag(void) {
    const char *test_name = "nvm_verify: OP_HM_NEW with invalid value type tag fails";
    uint8_t code[8];
    uint32_t off = 0;
    off += emit(code + off, OP_HM_NEW, (int)TAG_INT, (int)TAG_COUNT + 1);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "value tag >= TAG_COUNT should fail");
    ASSERT(strstr(r.error_msg, "value tag") != NULL,
           "failure should identify the invalid value tag");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_type_check_bad_tag(void) {
    const char *test_name = "nvm_verify: OP_TYPE_CHECK with invalid expected tag fails";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_TYPE_CHECK, (int)TAG_COUNT + 2);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "expected tag >= TAG_COUNT should fail");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_valid_type_tags_pass(void) {
    const char *test_name = "nvm_verify: valid type tags on ARR_NEW/HM_NEW/TYPE_CHECK pass";
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_ARR_NEW, (int)TAG_INT);
    off += emit(code + off, OP_POP);
    off += emit(code + off, OP_HM_NEW, (int)TAG_STRING, (int)TAG_INT);
    off += emit(code + off, OP_POP);
    off += emit(code + off, OP_PUSH_I64, (int64_t)7);
    off += emit(code + off, OP_TYPE_CHECK, (int)TAG_INT);
    off += emit(code + off, OP_POP);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "valid type tags should verify");
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
    declares_one_result(mod);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "arithmetic opcodes should verify OK");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_stack_underflow(void) {
    const char *test_name = "nvm_verify: fixed stack effect detects underflow";
    uint8_t code[8];
    uint32_t off = 0;
    off += emit(code + off, OP_POP);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "POP on an empty stack should fail");
    ASSERT(strstr(r.error_msg, "stack underflow") != NULL,
           "failure should identify stack underflow");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_incompatible_branch_stack_heights(void) {
    const char *test_name = "nvm_verify: incompatible branch stack heights fail";
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_JMP_TRUE, (int32_t)14);
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(!r.ok, "branch paths with different heights should fail");
    ASSERT(strstr(r.error_msg, "incompatible stack heights") != NULL,
           "failure should identify incompatible stack heights");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_compatible_branch_stack_heights(void) {
    const char *test_name = "nvm_verify: compatible branch stack heights pass";
    uint8_t code[32];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_JMP_TRUE, (int32_t)6);
    off += emit(code + off, OP_NOP);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    NvmVerifyResult r = nvm_verify(mod);
    ASSERT(r.ok, "branch paths with equal heights should pass");
    nvm_module_free(mod);
    PASS(test_name);
}

static void test_verify_one_function(void) {
    const char *test_name = "nvm_verify_function: validates incremental function";
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_simple_module(code, off, 0, 0);
    declares_one_result(mod);
    NvmVerifyResult valid = nvm_verify_function(mod, 0);
    ASSERT(valid.ok, "valid incremental function should pass");
    NvmVerifyResult missing = nvm_verify_function(mod, 1);
    ASSERT(!missing.ok, "missing incremental function should fail");
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
    test_function_code_length_overflow();
    test_adjacent_function_code_ranges();
    test_overlapping_function_code_ranges();
    test_contained_function_code_range();
    test_empty_function_code_ranges();
    test_function_code_length_beyond_end();
    test_function_name_idx_overflow();
    test_invalid_function_result_signature();
    test_multiple_function_results();
    test_invalid_function_result_tag();
    test_invalid_instruction_byte();
    test_valid_jump_forward();
    test_jump_out_of_bounds();
    test_jump_negative_out_of_bounds();
    test_jmp_true_valid();
    test_jmp_false_out_of_bounds();
    test_jump_into_operand_fails();
    test_match_tag_into_operand_fails();
    test_branch_target_past_function_end_fails();
    test_jump_to_function_end_passes();
    test_predecoded_module_boundaries();
    test_op_call_valid();
    test_op_call_invalid_fn_idx();
    test_tail_call_signatures();
    test_call_arity_underflow();
    test_call_result_shape();
    test_op_push_str_valid();
    test_op_push_str_invalid();
    test_op_load_local_valid();
    test_op_load_local_invalid();
    test_op_store_local_invalid();
    test_op_load_upvalue_valid();
    test_op_load_upvalue_invalid();
    test_upvalue_depth_invalid();
    test_global_index_invalid();
    test_op_struct_new_valid();
    test_op_struct_new_invalid();
    test_op_struct_new_zero_count_skips();
    test_op_enum_val_invalid();
    test_op_union_construct_invalid();
    test_op_closure_new_invalid();
    test_op_closure_new_valid();
    test_closure_capture_count_mismatch();
    test_aggregate_count_underflow();
    test_invalid_type_check_tag();
    test_match_tag_out_of_bounds();
    test_null_code_nonzero_size();
    test_call_extern_invalid();
    test_call_extern_valid_signature();
    test_import_signature_valid();
    test_import_bad_return_type();
    test_import_bad_param_type();
    test_import_at_arg_limit_verifies();
    test_import_over_arg_limit_fails();
    test_call_module_recognized();
    test_arithmetic_instructions();
    test_stack_underflow();
    test_incompatible_branch_stack_heights();
    test_compatible_branch_stack_heights();
    test_verify_one_function();
    test_call_module_linked_valid();
    test_call_module_shape_mismatch_fails();
    test_verification_continues_past_portable_isa_instructions();
    test_call_module_linked_bad_module_idx();
    test_call_module_linked_unresolved();
    test_call_module_linked_bad_fn_idx();
    test_call_module_no_table_still_ok();
    test_arr_new_bad_type_tag();
    test_hm_new_bad_value_tag();
    test_type_check_bad_tag();
    test_valid_type_tags_pass();
    test_integer_op_on_a_string_fails();
    test_integer_op_on_floats_fails();
    test_float_op_on_floats_passes();
    test_unknown_types_never_fail();
    test_conflicting_merge_widens_rather_than_failing();
    test_unbalanced_retain_at_return_fails();
    test_release_without_retain_fails();
    test_balanced_retain_release_passes();
    test_branch_dependent_ownership_fails();
    test_returning_more_than_declared_fails();
    test_returning_fewer_than_declared_fails();
    test_implicit_return_shape_is_checked();
    test_implicit_return_with_matching_shape_passes();
    test_max_stack_of_an_empty_function();
    test_max_stack_counts_the_deepest_point();
    test_max_stack_is_refused_for_an_unverifiable_function();
    test_max_stack_out_of_range_function();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
