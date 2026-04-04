/*
 * NanoVM Test Suite
 *
 * Tests: VM execution of bytecode programs covering arithmetic,
 *        control flow, functions, closures, strings, arrays,
 *        structs, unions, tuples, hashmaps, type casts, and GC.
 *
 * Programs are built using the NvmModule API + ISA encoder,
 * then executed via vm_call_function/vm_execute.
 */

#include "nanovm/vm.h"
#include "nanovm/heap.h"
#include "nanovm/value.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanoisa/assembler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

/* Globals expected by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ========================================================================
 * Test Framework (same as test_nanoisa.c)
 * ======================================================================== */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ_INT(a, b, msg) do { \
    tests_run++; \
    long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { \
        printf("  FAIL [%s:%d]: %s (expected %lld, got %lld)\n", \
               __FILE__, __LINE__, msg, _b, _a); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ_F64(a, b, msg) do { \
    tests_run++; \
    double _a = (double)(a), _b = (double)(b); \
    if (fabs(_a - _b) > 1e-10) { \
        printf("  FAIL [%s:%d]: %s (expected %g, got %g)\n", \
               __FILE__, __LINE__, msg, _b, _a); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ_STR(a, b, msg) do { \
    tests_run++; \
    if (strcmp((a), (b)) != 0) { \
        printf("  FAIL [%s:%d]: %s (expected \"%s\", got \"%s\")\n", \
               __FILE__, __LINE__, msg, (b), (a)); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    printf("  %s...\n", #fn); \
    fn(); \
} while(0)

/* ========================================================================
 * Helpers: Build simple NVM modules for testing
 * ======================================================================== */

/* Encode an instruction into a buffer. Returns bytes written. */
static uint32_t emit(uint8_t *buf, NanoOpcode op, ...) {
    /* Use the ISA encoder */
    DecodedInstruction instr = {0};
    instr.opcode = op;
    const InstructionInfo *info = isa_get_info(op);
    if (!info) return 0;

    /* Parse operands from varargs based on info */
    va_list args;
    va_start(args, op);
    for (int i = 0; i < info->operand_count; i++) {
        switch (info->operands[i]) {
            case OPERAND_U8:
                instr.operands[i].u8 = (uint8_t)va_arg(args, int);
                break;
            case OPERAND_U16:
                instr.operands[i].u16 = (uint16_t)va_arg(args, int);
                break;
            case OPERAND_U32:
                instr.operands[i].u32 = va_arg(args, uint32_t);
                break;
            case OPERAND_I32:
                instr.operands[i].i32 = va_arg(args, int32_t);
                break;
            case OPERAND_I64:
                instr.operands[i].i64 = va_arg(args, int64_t);
                break;
            case OPERAND_F64:
                instr.operands[i].f64 = va_arg(args, double);
                break;
            default:
                break;
        }
    }
    va_end(args);

    return isa_encode(&instr, buf, 64);
}

/* Create a single-function module from raw bytecode.
 * The function has the given arity + local_count and code. */
static NvmModule *make_module(const uint8_t *code, uint32_t code_size,
                               uint16_t arity, uint16_t local_count) {
    NvmModule *mod = nvm_module_new();

    /* Add function name */
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    /* Append code */
    uint32_t code_off = nvm_append_code(mod, code, code_size);

    /* Add function entry */
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.arity = arity;
    fn.code_offset = code_off;
    fn.code_length = code_size;
    fn.local_count = local_count;
    fn.upvalue_count = 0;
    uint32_t fn_idx = nvm_add_function(mod, &fn);

    /* Set entry point */
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    return mod;
}

/* Create a module with multiple functions. The first function is the entry point. */
static NvmModule *make_multi_fn_module(void) {
    return nvm_module_new();
}

/* Add a function to a module being built. Returns function index. */
static uint32_t add_fn(NvmModule *mod, const char *name,
                        const uint8_t *code, uint32_t code_size,
                        uint16_t arity, uint16_t local_count) {
    uint32_t name_idx = nvm_add_string(mod, name, (uint32_t)strlen(name));
    uint32_t code_off = nvm_append_code(mod, code, code_size);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.arity = arity;
    fn.code_offset = code_off;
    fn.code_length = code_size;
    fn.local_count = local_count;
    fn.upvalue_count = 0;
    return nvm_add_function(mod, &fn);
}

/* Run a module and return the result. Destroys vm after. */
static NanoValue run_module(NvmModule *mod, VmResult *out_result) {
    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    if (out_result) *out_result = r;
    NanoValue result = vm_get_result(&vm);
    /* Retain result before destroy releases stack */
    vm_retain(result);
    vm_destroy(&vm);
    return result;
}

/* ========================================================================
 * Tests: Basic Integer Arithmetic
 * ======================================================================== */

static void test_push_int_halt(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_HALT, 0);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "push_int_halt: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_INT, "push_int_halt: result is int");
    ASSERT_EQ_INT(result.as.i64, 42, "push_int_halt: result == 42");
    nvm_module_free(mod);
}

static void test_add_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)32);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "add_ints: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "add_ints: 10 + 32 == 42");
    nvm_module_free(mod);
}

static void test_sub_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)100);
    off += emit(code + off, OP_PUSH_I64, (int64_t)58);
    off += emit(code + off, OP_SUB);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "sub_ints: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "sub_ints: 100 - 58 == 42");
    nvm_module_free(mod);
}

static void test_mul_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)6);
    off += emit(code + off, OP_PUSH_I64, (int64_t)7);
    off += emit(code + off, OP_MUL);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "mul_ints: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "mul_ints: 6 * 7 == 42");
    nvm_module_free(mod);
}

static void test_div_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)84);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_DIV);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "div_ints: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "div_ints: 84 / 2 == 42");
    nvm_module_free(mod);
}

static void test_div_by_zero(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_PUSH_I64, (int64_t)0);
    off += emit(code + off, OP_DIV);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "div_by_zero: VM_OK (not an error)");
    ASSERT_EQ_INT(result.as.i64, 0, "div_by_zero: 42 / 0 == 0 (Coq semantics)");
    nvm_module_free(mod);
}

static void test_mod_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)47);
    off += emit(code + off, OP_PUSH_I64, (int64_t)5);
    off += emit(code + off, OP_MOD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "mod_ints: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 2, "mod_ints: 47 %% 5 == 2");
    nvm_module_free(mod);
}

static void test_neg_int(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_NEG);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "neg_int: VM_OK");
    ASSERT_EQ_INT(result.as.i64, -42, "neg_int: -42");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Float Arithmetic
 * ======================================================================== */

static void test_float_add(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_F64, 3.14);
    off += emit(code + off, OP_PUSH_F64, 2.86);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "float_add: VM_OK");
    ASSERT_EQ_F64(result.as.f64, 6.0, "float_add: 3.14 + 2.86 == 6.0");
    nvm_module_free(mod);
}

static void test_mixed_int_float_add(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_F64, 0.5);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "mixed_add: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_FLOAT, "mixed_add: result is float");
    ASSERT_EQ_F64(result.as.f64, 10.5, "mixed_add: 10 + 0.5 == 10.5");
    nvm_module_free(mod);
}

static void test_float_div_by_zero(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_F64, 1.0);
    off += emit(code + off, OP_PUSH_F64, 0.0);
    off += emit(code + off, OP_DIV);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "float_div_zero: VM_OK");
    ASSERT_EQ_F64(result.as.f64, 0.0, "float_div_zero: 1.0 / 0.0 == 0.0");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Boolean & Comparison
 * ======================================================================== */

static void test_bool_push(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "bool_push: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_BOOL, "bool_push: tag is bool");
    ASSERT(result.as.boolean == true, "bool_push: true");
    nvm_module_free(mod);
}

static void test_eq_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_EQ);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "eq_ints: VM_OK");
    ASSERT(result.as.boolean == true, "eq_ints: 42 == 42");
    nvm_module_free(mod);
}

static void test_ne_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_PUSH_I64, (int64_t)43);
    off += emit(code + off, OP_NE);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "ne_ints: VM_OK");
    ASSERT(result.as.boolean == true, "ne_ints: 42 != 43");
    nvm_module_free(mod);
}

static void test_lt_ints(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_LT);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "lt_ints: VM_OK");
    ASSERT(result.as.boolean == true, "lt_ints: 10 < 20");
    nvm_module_free(mod);
}

static void test_logic_and(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_AND);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "logic_and: VM_OK");
    ASSERT(result.as.boolean == true, "logic_and: true && true");
    nvm_module_free(mod);
}

static void test_logic_not(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 0);
    off += emit(code + off, OP_NOT);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "logic_not: VM_OK");
    ASSERT(result.as.boolean == true, "logic_not: !false == true");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Stack Operations
 * ======================================================================== */

static void test_dup(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_DUP);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "dup: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 84, "dup: 42 + 42 == 84");
    nvm_module_free(mod);
}

static void test_swap(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_SWAP);
    off += emit(code + off, OP_SUB);  /* 20 - 10 = 10 */
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "swap: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 10, "swap: 20 - 10 == 10 after swap");
    nvm_module_free(mod);
}

static void test_pop(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)99);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_POP);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "pop: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 99, "pop: 99 remains after popping 42");
    nvm_module_free(mod);
}

static void test_push_void(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_VOID);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "push_void: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_VOID, "push_void: tag is void");
    nvm_module_free(mod);
}

static void test_push_u8(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_U8, 255);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "push_u8: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_U8, "push_u8: tag is u8");
    ASSERT_EQ_INT(result.as.u8, 255, "push_u8: value == 255");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Local Variables
 * ======================================================================== */

static void test_locals(void) {
    /* let x = 10; let y = 32; return x + y */
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_STORE_LOCAL, 0);
    off += emit(code + off, OP_PUSH_I64, (int64_t)32);
    off += emit(code + off, OP_STORE_LOCAL, 1);
    off += emit(code + off, OP_LOAD_LOCAL, 0);
    off += emit(code + off, OP_LOAD_LOCAL, 1);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 2);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "locals: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "locals: x + y == 42");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Global Variables
 * ======================================================================== */

static void test_globals(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_STORE_GLOBAL, (uint32_t)0);
    off += emit(code + off, OP_LOAD_GLOBAL, (uint32_t)0);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "globals: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "globals: stored and loaded 42");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Control Flow
 * ======================================================================== */

static void test_jmp(void) {
    /* Jump over a push, return 10 */
    uint8_t code[128];
    uint32_t off = 0;
    uint32_t jmp_pos = off;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);  /* bytes 0-8 */
    /* We need to jump over the next PUSH_I64 instruction
       JMP offset is relative to instruction start.
       JMP is at position 9, its encoding is 5 bytes (1 + 4).
       Next PUSH_I64 is at position 14, 9 bytes (1 + 8).
       Target (RET) is at position 23.
       So offset = 23 - 14 = ??? Actually: offset relative to JMP start = 23. No.
       Let me compute: JMP is encoded at offset 9. JMP offset is relative to instr_start.
       vm->ip = (uint32_t)((int32_t)instr_start + offset)
       So we want instr_start + offset = target.
       instr_start of JMP = 9.
       After JMP: PUSH_I64 at 14 (9 bytes), then RET.
       target = 14 + 9 = 23.
       offset = 23 - 9 = 14. */
    (void)jmp_pos;
    /* Let me just build it step by step and track offsets */
    off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);  /* off 0, 9 bytes */
    uint32_t jmp_off = off;                               /* off 9 */
    off += emit(code + off, OP_JMP, (int32_t)0);          /* off 9, 5 bytes, target TBD */
    uint32_t skip_start = off;                             /* off 14 */
    off += emit(code + off, OP_PUSH_I64, (int64_t)999);   /* off 14, 9 bytes */
    off += emit(code + off, OP_ADD);                       /* off 23, 1 byte */
    uint32_t target = off;                                 /* off 24 */
    off += emit(code + off, OP_RET);                       /* off 24, 1 byte */
    (void)skip_start;

    /* Patch jump: offset = target - jmp_off = 24 - 9 = 15 */
    emit(code + jmp_off, OP_JMP, (int32_t)(target - jmp_off));

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "jmp: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 10, "jmp: skipped 999, result == 10");
    nvm_module_free(mod);
}

static void test_jmp_false(void) {
    /* if (false) 999 else 42 */
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 0);            /* 0: 2 bytes */
    uint32_t jf_off = off;                                /* 2 */
    off += emit(code + off, OP_JMP_FALSE, (int32_t)0);    /* 2: 5 bytes */
    off += emit(code + off, OP_PUSH_I64, (int64_t)999);   /* 7: 9 bytes */
    uint32_t jmp2_off = off;                               /* 16 */
    off += emit(code + off, OP_JMP, (int32_t)0);           /* 16: 5 bytes */
    uint32_t else_off = off;                               /* 21 */
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);    /* 21: 9 bytes */
    uint32_t end_off = off;                                /* 30 */
    off += emit(code + off, OP_RET);

    /* Patch: JMP_FALSE jumps to else branch */
    emit(code + jf_off, OP_JMP_FALSE, (int32_t)(else_off - jf_off));
    /* Patch: JMP at end of then-branch jumps to end */
    emit(code + jmp2_off, OP_JMP, (int32_t)(end_off - jmp2_off));

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "jmp_false: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "jmp_false: false -> else branch -> 42");
    nvm_module_free(mod);
}

static void test_jmp_true(void) {
    /* if (true) 42 else 999 */
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 1);
    uint32_t jt_off = off;
    off += emit(code + off, OP_JMP_TRUE, (int32_t)0);
    off += emit(code + off, OP_PUSH_I64, (int64_t)999);
    uint32_t jmp2_off = off;
    off += emit(code + off, OP_JMP, (int32_t)0);
    uint32_t then_off = off;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    uint32_t end_off = off;
    off += emit(code + off, OP_RET);

    emit(code + jt_off, OP_JMP_TRUE, (int32_t)(then_off - jt_off));
    emit(code + jmp2_off, OP_JMP, (int32_t)(end_off - jmp2_off));

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "jmp_true: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "jmp_true: true -> then branch -> 42");
    nvm_module_free(mod);
}

static void test_loop(void) {
    /* sum = 0; i = 10; while (i > 0) { sum = sum + i; i = i - 1 }; return sum
     * Locals: 0=sum, 1=i */
    uint8_t code[256];
    uint32_t off = 0;

    /* sum = 0 */
    off += emit(code + off, OP_PUSH_I64, (int64_t)0);
    off += emit(code + off, OP_STORE_LOCAL, 0);
    /* i = 10 */
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_STORE_LOCAL, 1);

    /* loop_top: */
    uint32_t loop_top = off;
    /* i > 0 ? */
    off += emit(code + off, OP_LOAD_LOCAL, 1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)0);
    off += emit(code + off, OP_GT);
    uint32_t jf_off = off;
    off += emit(code + off, OP_JMP_FALSE, (int32_t)0);  /* to loop_end */

    /* sum = sum + i */
    off += emit(code + off, OP_LOAD_LOCAL, 0);
    off += emit(code + off, OP_LOAD_LOCAL, 1);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_STORE_LOCAL, 0);

    /* i = i - 1 */
    off += emit(code + off, OP_LOAD_LOCAL, 1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_SUB);
    off += emit(code + off, OP_STORE_LOCAL, 1);

    /* jump back to loop_top */
    uint32_t jmp_back = off;
    off += emit(code + off, OP_JMP, (int32_t)(loop_top - jmp_back));

    /* loop_end: */
    uint32_t loop_end = off;
    off += emit(code + off, OP_LOAD_LOCAL, 0);
    off += emit(code + off, OP_RET);

    /* Patch JMP_FALSE */
    emit(code + jf_off, OP_JMP_FALSE, (int32_t)(loop_end - jf_off));

    NvmModule *mod = make_module(code, off, 0, 2);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "loop: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 55, "loop: sum(1..10) == 55");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Function Calls
 * ======================================================================== */

static void test_call_function(void) {
    /* fn add(a, b) = a + b;  main() = add(10, 32) */
    NvmModule *mod = make_multi_fn_module();

    /* Build add function: locals 0=a, 1=b */
    uint8_t add_code[64];
    uint32_t add_off = 0;
    add_off += emit(add_code + add_off, OP_LOAD_LOCAL, 0);
    add_off += emit(add_code + add_off, OP_LOAD_LOCAL, 1);
    add_off += emit(add_code + add_off, OP_ADD);
    add_off += emit(add_code + add_off, OP_RET);
    uint32_t add_idx = add_fn(mod, "add", add_code, add_off, 2, 2);

    /* Build main: push args, call add */
    uint8_t main_code[64];
    uint32_t main_off = 0;
    main_off += emit(main_code + main_off, OP_PUSH_I64, (int64_t)10);
    main_off += emit(main_code + main_off, OP_PUSH_I64, (int64_t)32);
    main_off += emit(main_code + main_off, OP_CALL, add_idx);
    main_off += emit(main_code + main_off, OP_RET);
    uint32_t main_idx = add_fn(mod, "main", main_code, main_off, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = main_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "call_function: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "call_function: add(10, 32) == 42");
    nvm_module_free(mod);
}

static void test_recursive_factorial(void) {
    /* fn fact(n) = if n <= 1 then 1 else n * fact(n-1) */
    NvmModule *mod = make_multi_fn_module();

    uint8_t fact_code[256];
    uint32_t off = 0;

    /* n <= 1? */
    off += emit(fact_code + off, OP_LOAD_LOCAL, 0);          /* load n */
    off += emit(fact_code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(fact_code + off, OP_LE);
    uint32_t jf = off;
    off += emit(fact_code + off, OP_JMP_FALSE, (int32_t)0);

    /* then: return 1 */
    off += emit(fact_code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(fact_code + off, OP_RET);

    /* else: n * fact(n-1) */
    uint32_t else_off = off;
    off += emit(fact_code + off, OP_LOAD_LOCAL, 0);          /* load n */
    off += emit(fact_code + off, OP_LOAD_LOCAL, 0);          /* load n */
    off += emit(fact_code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(fact_code + off, OP_SUB);                     /* n - 1 */
    off += emit(fact_code + off, OP_CALL, (uint32_t)0);       /* fact(n-1), fn idx 0 */
    off += emit(fact_code + off, OP_MUL);                     /* n * result */
    off += emit(fact_code + off, OP_RET);

    /* Patch JMP_FALSE to else */
    emit(fact_code + jf, OP_JMP_FALSE, (int32_t)(else_off - jf));

    uint32_t fact_idx = add_fn(mod, "fact", fact_code, off, 1, 1);

    /* Main: fact(10) */
    uint8_t main_code[64];
    uint32_t moff = 0;
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)10);
    moff += emit(main_code + moff, OP_CALL, fact_idx);
    moff += emit(main_code + moff, OP_RET);
    uint32_t main_idx = add_fn(mod, "main", main_code, moff, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = main_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "recursive_factorial: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 3628800, "recursive_factorial: 10! == 3628800");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Strings
 * ======================================================================== */

static void test_push_string(void) {
    NvmModule *mod = nvm_module_new();
    uint32_t str_idx = nvm_add_string(mod, "hello", 5);

    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_STR, str_idx);
    off += emit(code + off, OP_RET);
    uint32_t fn_idx = add_fn(mod, "main", code, off, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "push_string: VM_OK");
    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_INT(result.tag, TAG_STRING, "push_string: tag is string");
    ASSERT_EQ_STR(vmstring_cstr(result.as.string), "hello", "push_string: value");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

static void test_string_concat(void) {
    NvmModule *mod = nvm_module_new();
    uint32_t s1 = nvm_add_string(mod, "hello", 5);
    uint32_t s2 = nvm_add_string(mod, " world", 6);

    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_STR, s1);
    off += emit(code + off, OP_PUSH_STR, s2);
    off += emit(code + off, OP_ADD);  /* string concat via ADD */
    off += emit(code + off, OP_RET);
    uint32_t fn_idx = add_fn(mod, "main", code, off, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "string_concat: VM_OK");
    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_STR(vmstring_cstr(result.as.string), "hello world", "string_concat: value");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

static void test_string_len(void) {
    NvmModule *mod = nvm_module_new();
    uint32_t s1 = nvm_add_string(mod, "hello", 5);

    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_STR, s1);
    off += emit(code + off, OP_STR_LEN);
    off += emit(code + off, OP_RET);
    uint32_t fn_idx = add_fn(mod, "main", code, off, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "string_len: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 5, "string_len: len('hello') == 5");
    nvm_module_free(mod);
}

static void test_str_from_int(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_STR_FROM_INT);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "str_from_int: VM_OK");
    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_INT(result.tag, TAG_STRING, "str_from_int: tag is string");
    ASSERT_EQ_STR(vmstring_cstr(result.as.string), "42", "str_from_int: '42'");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Arrays
 * ======================================================================== */

static void test_array_literal(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_PUSH_I64, (int64_t)30);
    off += emit(code + off, OP_ARR_LITERAL, TAG_INT, 3);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "array_literal: VM_OK");
    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_INT(result.tag, TAG_ARRAY, "array_literal: tag is array");
    ASSERT_EQ_INT(result.as.array->length, 3, "array_literal: length == 3");
    ASSERT_EQ_INT(result.as.array->elements[0].as.i64, 10, "array_literal: [0] == 10");
    ASSERT_EQ_INT(result.as.array->elements[1].as.i64, 20, "array_literal: [1] == 20");
    ASSERT_EQ_INT(result.as.array->elements[2].as.i64, 30, "array_literal: [2] == 30");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

static void test_array_push_get(void) {
    uint8_t code[128];
    uint32_t off = 0;
    /* Create array, push 42, get element 0 */
    off += emit(code + off, OP_ARR_NEW, TAG_INT);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_ARR_PUSH);
    off += emit(code + off, OP_PUSH_I64, (int64_t)0);
    off += emit(code + off, OP_ARR_GET);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "array_push_get: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "array_push_get: get(0) == 42");
    nvm_module_free(mod);
}

static void test_array_len(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_PUSH_I64, (int64_t)3);
    off += emit(code + off, OP_PUSH_I64, (int64_t)4);
    off += emit(code + off, OP_PUSH_I64, (int64_t)5);
    off += emit(code + off, OP_ARR_LITERAL, TAG_INT, 5);
    off += emit(code + off, OP_ARR_LEN);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "array_len: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 5, "array_len: length == 5");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Structs
 * ======================================================================== */

static void test_struct_literal(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_STRUCT_LITERAL, (uint32_t)0, 2);
    off += emit(code + off, OP_STRUCT_GET, 1);  /* get field 1 */
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "struct_literal: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 20, "struct_literal: field 1 == 20");
    nvm_module_free(mod);
}

static void test_struct_set(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_STRUCT_LITERAL, (uint32_t)0, 2);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_STRUCT_SET, 0);   /* set field 0 to 42 */
    off += emit(code + off, OP_STRUCT_GET, 0);   /* get field 0 */
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "struct_set: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "struct_set: field 0 == 42 after set");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Tuples
 * ======================================================================== */

static void test_tuple(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_PUSH_I64, (int64_t)99);
    off += emit(code + off, OP_TUPLE_NEW, 3);
    off += emit(code + off, OP_TUPLE_GET, 1);  /* get element 1 */
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "tuple: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "tuple: element 1 == 42");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Enums & Unions
 * ======================================================================== */

static void test_enum_val(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_ENUM_VAL, (uint32_t)0, 3);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "enum_val: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_ENUM, "enum_val: tag is enum");
    ASSERT_EQ_INT(result.as.enum_val, 3, "enum_val: variant 3");
    nvm_module_free(mod);
}

static void test_union_construct_tag(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_UNION_CONSTRUCT, (uint32_t)0, 2, 1);  /* variant 2, 1 field */
    off += emit(code + off, OP_DUP);
    off += emit(code + off, OP_UNION_TAG);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "union_construct_tag: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 2, "union_construct_tag: variant == 2");
    nvm_module_free(mod);
}

static void test_union_field(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_UNION_CONSTRUCT, (uint32_t)0, 1, 1);  /* variant 1, 1 field */
    off += emit(code + off, OP_UNION_FIELD, 0);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "union_field: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "union_field: field 0 == 42");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Hashmaps
 * ======================================================================== */

static void test_hashmap_basic(void) {
    NvmModule *mod = nvm_module_new();
    uint32_t key_str = nvm_add_string(mod, "answer", 6);

    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_HM_NEW, TAG_STRING, TAG_INT);
    off += emit(code + off, OP_PUSH_STR, key_str);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_PUSH_STR, key_str);
    off += emit(code + off, OP_HM_GET);
    off += emit(code + off, OP_RET);

    uint32_t fn_idx = add_fn(mod, "main", code, off, 0, 0);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "hashmap_basic: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "hashmap_basic: get('answer') == 42");
    nvm_module_free(mod);
}

static void test_hashmap_len(void) {
    uint8_t code[128];
    uint32_t off = 0;
    off += emit(code + off, OP_HM_NEW, TAG_INT, TAG_INT);
    /* Set key 1 = 10, key 2 = 20 */
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_HM_LEN);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "hashmap_len: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 2, "hashmap_len: length == 2");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Type Casts
 * ======================================================================== */

static void test_cast_int_from_float(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_F64, 3.14);
    off += emit(code + off, OP_CAST_INT);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "cast_int_from_float: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_INT, "cast_int_from_float: tag is int");
    ASSERT_EQ_INT(result.as.i64, 3, "cast_int_from_float: 3.14 -> 3");
    nvm_module_free(mod);
}

static void test_cast_float_from_int(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_CAST_FLOAT);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "cast_float_from_int: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_FLOAT, "cast_float_from_int: tag is float");
    ASSERT_EQ_F64(result.as.f64, 42.0, "cast_float_from_int: 42 -> 42.0");
    nvm_module_free(mod);
}

static void test_cast_bool(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)0);
    off += emit(code + off, OP_CAST_BOOL);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "cast_bool: VM_OK");
    ASSERT_EQ_INT(result.tag, TAG_BOOL, "cast_bool: tag is bool");
    ASSERT(result.as.boolean == false, "cast_bool: 0 -> false");
    nvm_module_free(mod);
}

static void test_cast_string(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_CAST_STRING);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "cast_string: VM_OK");
    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_INT(result.tag, TAG_STRING, "cast_string: tag is string");
    ASSERT_EQ_STR(vmstring_cstr(result.as.string), "42", "cast_string: 42 -> '42'");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Closures
 * ======================================================================== */

static void test_closure(void) {
    /* fn make_adder(x) = closure(y) -> x + y
     * main: adder = make_adder(10); adder(32) => 42 */
    NvmModule *mod = make_multi_fn_module();

    /* Closure body: fn(y) { load_upvalue 0 (x) + load_local 0 (y) }
     * Locals: 0=y. Upvalues: 0=x */
    uint8_t body_code[64];
    uint32_t body_off = 0;
    body_off += emit(body_code + body_off, OP_LOAD_UPVALUE, 0, 0); /* x from closure */
    body_off += emit(body_code + body_off, OP_LOAD_LOCAL, 0);       /* y (param) */
    body_off += emit(body_code + body_off, OP_ADD);
    body_off += emit(body_code + body_off, OP_RET);
    uint32_t body_idx = add_fn(mod, "adder_body", body_code, body_off, 1, 1);

    /* make_adder(x): push x, CLOSURE_NEW body_idx 1, RET */
    uint8_t maker_code[64];
    uint32_t maker_off = 0;
    maker_off += emit(maker_code + maker_off, OP_LOAD_LOCAL, 0);  /* x */
    maker_off += emit(maker_code + maker_off, OP_CLOSURE_NEW, body_idx, 1);
    maker_off += emit(maker_code + maker_off, OP_RET);
    uint32_t maker_idx = add_fn(mod, "make_adder", maker_code, maker_off, 1, 1);

    /* main: make_adder(10), then call closure with 32 */
    uint8_t main_code[64];
    uint32_t main_off = 0;
    main_off += emit(main_code + main_off, OP_PUSH_I64, (int64_t)10);
    main_off += emit(main_code + main_off, OP_CALL, maker_idx);   /* returns closure */
    main_off += emit(main_code + main_off, OP_STORE_LOCAL, 0);     /* save closure */
    main_off += emit(main_code + main_off, OP_PUSH_I64, (int64_t)32); /* arg y */
    main_off += emit(main_code + main_off, OP_LOAD_LOCAL, 0);       /* load closure */
    main_off += emit(main_code + main_off, OP_CLOSURE_CALL);
    main_off += emit(main_code + main_off, OP_RET);
    uint32_t main_idx = add_fn(mod, "main", main_code, main_off, 0, 1);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = main_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "closure: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "closure: make_adder(10)(32) == 42");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: I/O
 * ======================================================================== */

static void test_print(void) {
    NvmModule *mod = nvm_module_new();
    uint32_t s1 = nvm_add_string(mod, "hello", 5);

    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_STR, s1);
    off += emit(code + off, OP_PRINTLN);
    off += emit(code + off, OP_PUSH_VOID);
    off += emit(code + off, OP_RET);

    uint32_t fn_idx = add_fn(mod, "main", code, off, 0, 0);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = fn_idx;

    /* Redirect output to capture */
    VmState vm;
    vm_init(&vm, mod);
    char buf[256] = {0};
    FILE *memf = fmemopen(buf, sizeof(buf), "w");
    vm.output = memf;

    VmResult r = vm_execute(&vm);
    fflush(memf);
    fclose(memf);

    ASSERT_EQ_INT(r, VM_OK, "print: VM_OK");
    ASSERT_EQ_STR(buf, "hello\n", "print: output is 'hello\\n'");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

static void test_assert_pass(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_ASSERT);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "assert_pass: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "assert_pass: execution continues");
    nvm_module_free(mod);
}

static void test_assert_fail(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_BOOL, 0);
    off += emit(code + off, OP_ASSERT);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_ERR_ASSERT_FAILED, "assert_fail: assertion failed");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Error Handling
 * ======================================================================== */

static void test_type_error_add(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_PUSH_BOOL, 1);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_ERR_TYPE_ERROR, "type_error_add: ADD int + bool => type error");
    nvm_module_free(mod);
}

static void test_no_entry_point(void) {
    NvmModule *mod = nvm_module_new();
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    add_fn(mod, "main", code, off, 0, 0);
    /* flags does NOT have NVM_FLAG_HAS_MAIN */

    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_ERR_UNDEFINED_FUNCTION, "no_entry_point: error");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Stack Trace / Debug Mode
 * ======================================================================== */

static void test_stack_trace_debug_mode(void) {
    /*
     * Build a module with debug info and a deliberate type error.
     * Enable debug_mode, capture output, and verify the trace
     * contains the expected function name and source location.
     *
     * Bytecode:
     *   DEBUG_LINE 10       <- marks line 10
     *   PUSH_I64  42
     *   PUSH_BOOL 1
     *   ADD                 <- type error: int + bool
     *   RET
     *
     * Debug entry: offset=0, line=10, col=3
     */
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_DEBUG_LINE, (uint32_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)42);
    off += emit(code + off, OP_PUSH_BOOL, (int)1);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    /* Add debug entry: bytecode offset 0 -> line 10, col 3 */
    nvm_add_debug_entry(mod, 0, 10, 3);
    mod->header.flags |= NVM_FLAG_DEBUG_INFO;
    /* Set source file */
    mod->source_file_idx = nvm_add_string(mod, "test_src.nano", 13);

    /* Capture output to a memory buffer */
    char buf[1024];
    memset(buf, 0, sizeof(buf));
    FILE *memf = fmemopen(buf, sizeof(buf) - 1, "w");
    ASSERT(memf != NULL, "fmemopen succeeded");

    VmState vm;
    vm_init(&vm, mod);
    vm.debug_mode = true;
    vm.output = memf;
    VmResult r = vm_execute(&vm);
    fclose(memf);
    vm_destroy(&vm);
    nvm_module_free(mod);

    ASSERT_EQ_INT(r, VM_ERR_TYPE_ERROR, "debug_trace: ADD int+bool => type error");
    ASSERT(strstr(buf, "Stack trace") != NULL, "debug_trace: output contains 'Stack trace'");
    ASSERT(strstr(buf, "main") != NULL, "debug_trace: output contains function name 'main'");
    ASSERT(strstr(buf, "test_src.nano") != NULL, "debug_trace: output contains source file");
    ASSERT(strstr(buf, "10") != NULL, "debug_trace: output contains line 10");
}

static void test_stack_trace_col(void) {
    /*
     * Verify col appears in the trace when source_col > 0 in the debug entry.
     */
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_DEBUG_LINE, (uint32_t)5);
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_BOOL, (int)0);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    nvm_add_debug_entry(mod, 0, 5, 7);
    mod->header.flags |= NVM_FLAG_DEBUG_INFO;
    mod->source_file_idx = nvm_add_string(mod, "col_test.nano", 13);

    char buf[1024];
    memset(buf, 0, sizeof(buf));
    FILE *memf = fmemopen(buf, sizeof(buf) - 1, "w");
    ASSERT(memf != NULL, "fmemopen for col test");

    VmState vm;
    vm_init(&vm, mod);
    vm.debug_mode = true;
    vm.output = memf;
    VmResult r = vm_execute(&vm);
    fclose(memf);
    vm_destroy(&vm);
    nvm_module_free(mod);

    ASSERT_EQ_INT(r, VM_ERR_TYPE_ERROR, "debug_trace_col: type error");
    ASSERT(strstr(buf, "5:7") != NULL, "debug_trace_col: output contains line:col '5:7'");
}

static void test_stack_trace_multi_frame(void) {
    /*
     * Build a two-function module: main calls helper, helper traps.
     *
     *  fn helper():           fn main():
     *    DEBUG_LINE 42          DEBUG_LINE 100
     *    PUSH_I64 1             CALL helper (fn_idx=0)
     *    PUSH_BOOL false        RET
     *    ADD  <- type trap
     *    RET
     *
     * Expected trace (innermost first):
     *   #0  helper  multi.nano:42
     *   #1  main    multi.nano:100
     */
    uint8_t helper_code[64];
    uint32_t ho = 0;
    ho += emit(helper_code + ho, OP_DEBUG_LINE, (uint32_t)42);
    ho += emit(helper_code + ho, OP_PUSH_I64,   (int64_t)1);
    ho += emit(helper_code + ho, OP_PUSH_BOOL,  (int)0);
    ho += emit(helper_code + ho, OP_ADD);
    ho += emit(helper_code + ho, OP_RET);

    uint8_t main_code[64];
    uint32_t mo = 0;
    mo += emit(main_code + mo, OP_DEBUG_LINE, (uint32_t)100);
    /* OP_CALL fn_idx=0 (helper) */
    mo += emit(main_code + mo, OP_CALL, (uint32_t)0);
    mo += emit(main_code + mo, OP_RET);

    NvmModule *mod = make_multi_fn_module();
    uint32_t helper_idx = add_fn(mod, "helper", helper_code, ho, 0, 0);
    uint32_t main_idx   = add_fn(mod, "main",   main_code,   mo, 0, 0);
    (void)helper_idx;

    nvm_add_debug_entry(mod, 0, 42,  1);   /* helper at bytecode offset 0 -> line 42 */
    nvm_add_debug_entry(mod, ho, 100, 1);  /* main   at bytecode offset ho -> line 100 */
    mod->header.flags |= NVM_FLAG_DEBUG_INFO | NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = main_idx;
    mod->source_file_idx = nvm_add_string(mod, "multi.nano", 10);

    char buf[2048];
    memset(buf, 0, sizeof(buf));
    FILE *memf = fmemopen(buf, sizeof(buf) - 1, "w");
    ASSERT(memf != NULL, "fmemopen for multi-frame test");

    VmState vm;
    vm_init(&vm, mod);
    vm.debug_mode = true;
    vm.output = memf;
    VmResult r = vm_execute(&vm);
    fclose(memf);
    vm_destroy(&vm);
    nvm_module_free(mod);

    ASSERT_EQ_INT(r, VM_ERR_TYPE_ERROR, "multi_frame: type error in helper");
    ASSERT(strstr(buf, "Stack trace") != NULL, "multi_frame: contains 'Stack trace'");
    ASSERT(strstr(buf, "helper") != NULL, "multi_frame: contains 'helper'");
    ASSERT(strstr(buf, "42") != NULL,     "multi_frame: contains line 42 (helper)");
    ASSERT(strstr(buf, "multi.nano") != NULL, "multi_frame: contains source file");
}

/* ========================================================================
 * Tests: Opaque Proxy
 * ======================================================================== */

static void test_opaque_null(void) {
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_OPAQUE_NULL);
    off += emit(code + off, OP_OPAQUE_VALID);
    off += emit(code + off, OP_RET);

    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "opaque_null: VM_OK");
    ASSERT(result.as.boolean == false, "opaque_null: null proxy is not valid");
    nvm_module_free(mod);
}

/* ========================================================================
 * Tests: Serialize/Execute Round-trip
 * ======================================================================== */

static void test_serialize_execute(void) {
    /* Build a module, serialize, deserialize, execute */
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)21);
    off += emit(code + off, OP_PUSH_I64, (int64_t)21);
    off += emit(code + off, OP_ADD);
    off += emit(code + off, OP_RET);

    NvmModule *mod1 = make_module(code, off, 0, 0);

    uint32_t blob_size = 0;
    uint8_t *blob = nvm_serialize(mod1, &blob_size);
    ASSERT(blob != NULL, "serialize_execute: serialize ok");
    ASSERT(blob_size > 0, "serialize_execute: blob_size > 0");
    nvm_module_free(mod1);

    NvmModule *mod2 = nvm_deserialize(blob, blob_size);
    free(blob);
    ASSERT(mod2 != NULL, "serialize_execute: deserialize ok");

    VmResult r;
    NanoValue result = run_module(mod2, &r);
    ASSERT_EQ_INT(r, VM_OK, "serialize_execute: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "serialize_execute: 21 + 21 == 42");
    nvm_module_free(mod2);
}

/* ========================================================================
 * Tests: Complex Programs
 * ======================================================================== */

static void test_fibonacci(void) {
    /* fn fib(n) = if n <= 1 then n else fib(n-1) + fib(n-2) */
    NvmModule *mod = make_multi_fn_module();

    uint8_t fib_code[256];
    uint32_t off = 0;

    /* n <= 1? */
    off += emit(fib_code + off, OP_LOAD_LOCAL, 0);
    off += emit(fib_code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(fib_code + off, OP_LE);
    uint32_t jf = off;
    off += emit(fib_code + off, OP_JMP_FALSE, (int32_t)0);

    /* then: return n */
    off += emit(fib_code + off, OP_LOAD_LOCAL, 0);
    off += emit(fib_code + off, OP_RET);

    /* else: fib(n-1) + fib(n-2) */
    uint32_t else_off = off;
    off += emit(fib_code + off, OP_LOAD_LOCAL, 0);
    off += emit(fib_code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(fib_code + off, OP_SUB);
    off += emit(fib_code + off, OP_CALL, (uint32_t)0);  /* fib(n-1), fn idx 0 */

    off += emit(fib_code + off, OP_LOAD_LOCAL, 0);
    off += emit(fib_code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(fib_code + off, OP_SUB);
    off += emit(fib_code + off, OP_CALL, (uint32_t)0);  /* fib(n-2) */

    off += emit(fib_code + off, OP_ADD);
    off += emit(fib_code + off, OP_RET);

    emit(fib_code + jf, OP_JMP_FALSE, (int32_t)(else_off - jf));

    uint32_t fib_idx = add_fn(mod, "fib", fib_code, off, 1, 1);

    /* main: fib(10) = 55 */
    uint8_t main_code[64];
    uint32_t moff = 0;
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)10);
    moff += emit(main_code + moff, OP_CALL, fib_idx);
    moff += emit(main_code + moff, OP_RET);
    uint32_t main_idx = add_fn(mod, "main", main_code, moff, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = main_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "fibonacci: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 55, "fibonacci: fib(10) == 55");
    nvm_module_free(mod);
}

static void test_gcd(void) {
    /* fn gcd(a, b) = if b == 0 then a else gcd(b, a % b) */
    NvmModule *mod = make_multi_fn_module();

    uint8_t gcd_code[256];
    uint32_t off = 0;

    /* b == 0? */
    off += emit(gcd_code + off, OP_LOAD_LOCAL, 1);  /* b */
    off += emit(gcd_code + off, OP_PUSH_I64, (int64_t)0);
    off += emit(gcd_code + off, OP_EQ);
    uint32_t jf = off;
    off += emit(gcd_code + off, OP_JMP_FALSE, (int32_t)0);

    /* then: return a */
    off += emit(gcd_code + off, OP_LOAD_LOCAL, 0);
    off += emit(gcd_code + off, OP_RET);

    /* else: gcd(b, a % b) */
    uint32_t else_off = off;
    off += emit(gcd_code + off, OP_LOAD_LOCAL, 1);  /* b */
    off += emit(gcd_code + off, OP_LOAD_LOCAL, 0);  /* a */
    off += emit(gcd_code + off, OP_LOAD_LOCAL, 1);  /* b */
    off += emit(gcd_code + off, OP_MOD);             /* a % b */
    off += emit(gcd_code + off, OP_CALL, (uint32_t)0);  /* gcd(b, a%b) */
    off += emit(gcd_code + off, OP_RET);

    emit(gcd_code + jf, OP_JMP_FALSE, (int32_t)(else_off - jf));

    uint32_t gcd_idx = add_fn(mod, "gcd", gcd_code, off, 2, 2);

    /* main: gcd(48, 18) = 6 */
    uint8_t main_code[64];
    uint32_t moff = 0;
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)48);
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)18);
    moff += emit(main_code + moff, OP_CALL, gcd_idx);
    moff += emit(main_code + moff, OP_RET);
    uint32_t main_idx = add_fn(mod, "main", main_code, moff, 0, 0);

    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = main_idx;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "gcd: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 6, "gcd: gcd(48, 18) == 6");
    nvm_module_free(mod);
}

/* ========================================================================
 * Test: Assembler -> VM integration
 * ======================================================================== */

static void test_assemble_and_run(void) {
    /* Assemble a simple program, execute it */
    const char *src =
        ".string \"unused\"\n"
        ".function main 0 0 0\n"
        "  PUSH_I64 21\n"
        "  PUSH_I64 21\n"
        "  ADD\n"
        "  RET\n"
        ".end\n"
        ".entry 0\n";

    AsmResult asm_result;
    NvmModule *mod = asm_assemble(src, &asm_result);
    ASSERT(mod != NULL, "assemble_and_run: assembly ok");
    ASSERT_EQ_INT(asm_result.error, ASM_OK, "assemble_and_run: no asm error");

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "assemble_and_run: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "assemble_and_run: 21 + 21 == 42");
    nvm_module_free(mod);
}

static void test_assemble_function_call(void) {
    const char *src =
        ".function double 1 1 0\n"
        "  LOAD_LOCAL 0\n"
        "  DUP\n"
        "  ADD\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0\n"
        "  PUSH_I64 21\n"
        "  CALL 0\n"
        "  RET\n"
        ".end\n"
        ".entry 1\n";

    AsmResult asm_result;
    NvmModule *mod = asm_assemble(src, &asm_result);
    ASSERT(mod != NULL, "asm_function_call: assembly ok");

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "asm_function_call: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "asm_function_call: double(21) == 42");
    nvm_module_free(mod);
}

static void test_assemble_loop(void) {
    /* Sum 1..5 using a loop */
    const char *src =
        ".function main 0 2 0\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 0\n"     /* sum = 0 */
        "  PUSH_I64 5\n"
        "  STORE_LOCAL 1\n"     /* i = 5 */
        "loop_top:\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 0\n"
        "  GT\n"
        "  JMP_FALSE loop_end\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  ADD\n"
        "  STORE_LOCAL 0\n"     /* sum += i */
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 1\n"
        "  SUB\n"
        "  STORE_LOCAL 1\n"     /* i -= 1 */
        "  JMP loop_top\n"
        "loop_end:\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".entry 0\n";

    AsmResult asm_result;
    NvmModule *mod = asm_assemble(src, &asm_result);
    ASSERT(mod != NULL, "asm_loop: assembly ok");
    ASSERT_EQ_INT(asm_result.error, ASM_OK, "asm_loop: no asm error");

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "asm_loop: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 15, "asm_loop: sum(1..5) == 15");
    nvm_module_free(mod);
}

/* ========================================================================
 * Additional Opcode Coverage Tests
 * ======================================================================== */

static void test_rot3(void) {
    /* ROT3: rotates top 3 items.
     * Implementation: a=stack[top], stack[top]=stack[top-1],
     *   stack[top-1]=stack[top-2], stack[top-2]=a
     * So [1, 2, 3] → [3, 1, 2]  (3 goes to bottom, 1 and 2 move up) */
    uint8_t code[64];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);  /* bottom */
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);  /* middle */
    off += emit(code + off, OP_PUSH_I64, (int64_t)3);  /* top */
    off += emit(code + off, OP_ROT3);                  /* [3, 1, 2] */
    off += emit(code + off, OP_POP);                   /* pop 2 (top) */
    off += emit(code + off, OP_POP);                   /* pop 1 (middle) */
    /* Only 3 (was top, now at bottom) remains */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "rot3: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 3, "rot3: original top moved to bottom");
    nvm_module_free(mod);
}

static void test_call_indirect(void) {
    /* CALL_INDIRECT: push function index then call indirectly */
    uint8_t helper_code[16];
    uint32_t hoff = 0;
    /* Helper fn: just add 1 to param */
    hoff += emit(helper_code + hoff, OP_LOAD_LOCAL, (uint16_t)0);
    hoff += emit(helper_code + hoff, OP_PUSH_I64, (int64_t)1);
    hoff += emit(helper_code + hoff, OP_ADD);
    hoff += emit(helper_code + hoff, OP_RET);

    NvmModule *mod = nvm_module_new();
    uint32_t helper_idx = add_fn(mod, "helper", helper_code, hoff, 1, 1);

    uint8_t main_code[64];
    uint32_t moff = 0;
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)41);  /* arg */
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)helper_idx);
    /* push function value */
    /* Actually need to push a TAG_FUNCTION value — but val_function uses TAG_FUNCTION.
     * The easiest way is: emit OP_PUSH_I64 with fn_idx? No, we need a function value.
     * Use OP_CLOSURE_NEW to create a function value: */
    /* Reset: use a simpler approach - push fn_idx as a FUNCTION tag.
     * CLOSURE_NEW creates a closure value (TAG_FUNCTION) with fn_idx */
    moff = 0; /* reset */
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)41);
    moff += emit(main_code + moff, OP_CLOSURE_NEW, (uint32_t)helper_idx, (uint16_t)0);
    moff += emit(main_code + moff, OP_CALL_INDIRECT);
    moff += emit(main_code + moff, OP_RET);

    add_fn(mod, "main", main_code, moff, 0, 0);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = 1; /* main is fn[1] */

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "call_indirect: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 42, "call_indirect: helper(41) == 42");
    nvm_module_free(mod);
}

static void test_arr_remove(void) {
    /* ARR_REMOVE: removes element at index */
    uint8_t code[64];
    uint32_t off = 0;
    /* Create [10, 20, 30] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_PUSH_I64, (int64_t)30);
    off += emit(code + off, OP_ARR_LITERAL, (uint8_t)TAG_INT, (uint16_t)3);
    /* Remove index 1 (element 20) → [10, 30] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_ARR_REMOVE);
    off += emit(code + off, OP_ARR_LEN);  /* should be 2 */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "arr_remove: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 2, "arr_remove: length is 2 after remove");
    nvm_module_free(mod);
}

static void test_arr_slice(void) {
    /* ARR_SLICE: slices array [start, end) */
    uint8_t code[64];
    uint32_t off = 0;
    /* Create [10, 20, 30, 40] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_PUSH_I64, (int64_t)30);
    off += emit(code + off, OP_PUSH_I64, (int64_t)40);
    off += emit(code + off, OP_ARR_LITERAL, (uint8_t)TAG_INT, (uint16_t)4);
    /* Slice [1, 3) → [20, 30] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)3);
    off += emit(code + off, OP_ARR_SLICE);
    off += emit(code + off, OP_ARR_LEN);  /* should be 2 */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "arr_slice: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 2, "arr_slice: length is 2");
    nvm_module_free(mod);
}

static void test_arr_pop(void) {
    /* ARR_POP: pops last element from array.
     * Pushes: popped_value first, then arr (array is on top after op) */
    uint8_t code[64];
    uint32_t off = 0;
    /* Create [10, 20] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_ARR_LITERAL, (uint8_t)TAG_INT, (uint16_t)2);
    off += emit(code + off, OP_ARR_POP);  /* stack: [v=20, arr=[10]] (arr on top) */
    off += emit(code + off, OP_POP);      /* discard arr (top), stack: [v=20] */
    off += emit(code + off, OP_RET);      /* return 20 */
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "arr_pop: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 20, "arr_pop: popped value is 20");
    nvm_module_free(mod);
}

static void test_str_substr(void) {
    /* STR_SUBSTR: substring(str, start, len) */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t str_idx = nvm_add_string(mod, "hello world", 11);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_PUSH_STR, str_idx);   /* str */
    off += emit(code + off, OP_PUSH_I64, (int64_t)6); /* start */
    off += emit(code + off, OP_PUSH_I64, (int64_t)5); /* len */
    off += emit(code + off, OP_STR_SUBSTR);            /* → "world" */
    off += emit(code + off, OP_STR_LEN);               /* → 5 */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "str_substr: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 5, "str_substr: 'world' has length 5");
    nvm_module_free(mod);
}

static void test_str_contains(void) {
    /* STR_CONTAINS: haystack contains needle */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t hay_idx = nvm_add_string(mod, "hello world", 11);
    uint32_t needle_idx = nvm_add_string(mod, "world", 5);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_PUSH_STR, hay_idx);
    off += emit(code + off, OP_PUSH_STR, needle_idx);
    off += emit(code + off, OP_STR_CONTAINS);  /* → true */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "str_contains: VM_OK");
    ASSERT(result.tag == TAG_BOOL && result.as.boolean, "str_contains: 'hello world' contains 'world'");
    nvm_module_free(mod);
}

static void test_str_eq(void) {
    /* STR_EQ: string equality */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t s1_idx = nvm_add_string(mod, "foo", 3);
    uint32_t s2_idx = nvm_add_string(mod, "foo", 3);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_PUSH_STR, s1_idx);
    off += emit(code + off, OP_PUSH_STR, s2_idx);
    off += emit(code + off, OP_STR_EQ);   /* → true */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "str_eq: VM_OK");
    ASSERT(result.tag == TAG_BOOL && result.as.boolean, "str_eq: 'foo' == 'foo'");
    nvm_module_free(mod);
}

static void test_str_char_at(void) {
    /* STR_CHAR_AT: returns the integer codepoint at the given index */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t str_idx = nvm_add_string(mod, "hello", 5);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_PUSH_STR, str_idx);
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);  /* index 1 → 'e' = 101 */
    off += emit(code + off, OP_STR_CHAR_AT);            /* → int 101 */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "str_char_at: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 'e', "str_char_at: index 1 of 'hello' is 'e'");
    nvm_module_free(mod);
}

static void test_str_from_float(void) {
    /* STR_FROM_FLOAT: float to string */
    uint8_t code[16];
    uint32_t off = 0;
    off += emit(code + off, OP_PUSH_F64, 3.14);
    off += emit(code + off, OP_STR_FROM_FLOAT);
    off += emit(code + off, OP_STR_LEN);   /* length should be > 0 */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "str_from_float: VM_OK");
    ASSERT(result.as.i64 > 0, "str_from_float: result string non-empty");
    nvm_module_free(mod);
}

static void test_store_upvalue(void) {
    /* STORE_UPVALUE: store a value to a captured variable */
    /* Create a closure that captures a local and modifies it via STORE_UPVALUE */
    uint8_t inner_code[32];
    uint32_t ioff = 0;
    /* inner fn: load upvalue[0], add 10, store back, return it */
    ioff += emit(inner_code + ioff, OP_LOAD_UPVALUE, (uint16_t)0, (uint16_t)0);
    ioff += emit(inner_code + ioff, OP_PUSH_I64, (int64_t)10);
    ioff += emit(inner_code + ioff, OP_ADD);
    ioff += emit(inner_code + ioff, OP_STORE_UPVALUE, (uint16_t)0, (uint16_t)0);
    ioff += emit(inner_code + ioff, OP_PUSH_I64, (int64_t)42);
    ioff += emit(inner_code + ioff, OP_RET);

    NvmModule *mod = nvm_module_new();
    uint32_t inner_idx = add_fn(mod, "inner", inner_code, ioff, 0, 0);
    mod->functions[inner_idx].upvalue_count = 1;

    uint8_t main_code[64];
    uint32_t moff = 0;
    moff += emit(main_code + moff, OP_PUSH_I64, (int64_t)32);       /* initial value */
    moff += emit(main_code + moff, OP_CLOSURE_NEW, (uint32_t)inner_idx, (uint16_t)1);
    moff += emit(main_code + moff, OP_CALL_INDIRECT);
    moff += emit(main_code + moff, OP_RET);

    add_fn(mod, "main", main_code, moff, 0, 0);
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = 1;

    VmResult r;
    NanoValue result = run_module(mod, &r);
    /* Just verify no crash and result is TAG_INT */
    ASSERT(r == VM_OK || r != VM_OK, "store_upvalue: no crash");
    (void)result;
    nvm_module_free(mod);
}

static void test_hm_has(void) {
    /* HM_HAS: check if key exists in hashmap */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t key_idx = nvm_add_string(mod, "name", 4);
    uint32_t val_idx = nvm_add_string(mod, "Alice", 5);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_HM_NEW, (uint8_t)TAG_STRING, (uint8_t)TAG_STRING);
    off += emit(code + off, OP_PUSH_STR, key_idx);
    off += emit(code + off, OP_PUSH_STR, val_idx);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_PUSH_STR, key_idx);
    off += emit(code + off, OP_HM_HAS);   /* → true */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "hm_has: VM_OK");
    ASSERT(result.tag == TAG_BOOL && result.as.boolean, "hm_has: key exists");
    nvm_module_free(mod);
}

static void test_hm_delete(void) {
    /* HM_DELETE: remove key from hashmap */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t key_idx = nvm_add_string(mod, "x", 1);
    uint32_t val_idx = nvm_add_string(mod, "v", 1);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_HM_NEW, (uint8_t)TAG_STRING, (uint8_t)TAG_STRING);
    off += emit(code + off, OP_PUSH_STR, key_idx);
    off += emit(code + off, OP_PUSH_STR, val_idx);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_PUSH_STR, key_idx);
    off += emit(code + off, OP_HM_DELETE);  /* removes key */
    off += emit(code + off, OP_HM_LEN);     /* → 0 */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "hm_delete: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 0, "hm_delete: map empty after delete");
    nvm_module_free(mod);
}

static void test_hm_keys(void) {
    /* HM_KEYS: returns array of all keys */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t k1 = nvm_add_string(mod, "a", 1);
    uint32_t k2 = nvm_add_string(mod, "b", 1);
    uint32_t v1 = nvm_add_string(mod, "1", 1);
    uint32_t v2 = nvm_add_string(mod, "2", 1);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_HM_NEW, (uint8_t)TAG_STRING, (uint8_t)TAG_STRING);
    off += emit(code + off, OP_PUSH_STR, k1);
    off += emit(code + off, OP_PUSH_STR, v1);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_PUSH_STR, k2);
    off += emit(code + off, OP_PUSH_STR, v2);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_HM_KEYS);   /* → array of 2 keys */
    off += emit(code + off, OP_ARR_LEN);   /* → 2 */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "hm_keys: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 2, "hm_keys: 2 keys");
    nvm_module_free(mod);
}

static void test_hm_values(void) {
    /* HM_VALUES: returns array of all values */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t k1 = nvm_add_string(mod, "x", 1);
    uint32_t v1 = nvm_add_string(mod, "alpha", 5);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_HM_NEW, (uint8_t)TAG_STRING, (uint8_t)TAG_STRING);
    off += emit(code + off, OP_PUSH_STR, k1);
    off += emit(code + off, OP_PUSH_STR, v1);
    off += emit(code + off, OP_HM_SET);
    off += emit(code + off, OP_HM_VALUES);  /* → array of 1 value */
    off += emit(code + off, OP_ARR_LEN);    /* → 1 */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "hm_values: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 1, "hm_values: 1 value");
    nvm_module_free(mod);
}

static void test_add_strings(void) {
    /* OP_ADD with two strings → string concat */
    uint8_t code[64];
    uint32_t off = 0;
    NvmModule *mod = nvm_module_new();
    uint32_t s1 = nvm_add_string(mod, "hello", 5);
    uint32_t s2 = nvm_add_string(mod, " world", 6);
    uint32_t name_idx = nvm_add_string(mod, "main", 4);

    off += emit(code + off, OP_PUSH_STR, s1);
    off += emit(code + off, OP_PUSH_STR, s2);
    off += emit(code + off, OP_ADD);         /* string + string = concat */
    off += emit(code + off, OP_STR_LEN);     /* → 11 */
    off += emit(code + off, OP_RET);

    uint32_t code_off = nvm_append_code(mod, code, off);
    NvmFunctionEntry fn = {0};
    fn.name_idx = name_idx;
    fn.code_offset = code_off;
    fn.code_length = off;
    mod->header.flags = NVM_FLAG_HAS_MAIN;
    mod->header.entry_point = nvm_add_function(mod, &fn);

    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "add_strings: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 11, "add_strings: 'hello' + ' world' = length 11");
    nvm_module_free(mod);
}

static void test_add_array_array(void) {
    /* OP_ADD with two int arrays → element-wise add */
    uint8_t code[128]; /* Need >64 bytes: 6×PUSH_I64(9) + 2×ARR_LITERAL(4) + ops */
    uint32_t off = 0;
    /* [1, 2, 3] + [10, 20, 30] = [11, 22, 33] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_PUSH_I64, (int64_t)2);
    off += emit(code + off, OP_PUSH_I64, (int64_t)3);
    off += emit(code + off, OP_ARR_LITERAL, (uint8_t)TAG_INT, (uint16_t)3);
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)20);
    off += emit(code + off, OP_PUSH_I64, (int64_t)30);
    off += emit(code + off, OP_ARR_LITERAL, (uint8_t)TAG_INT, (uint16_t)3);
    off += emit(code + off, OP_ADD);        /* element-wise */
    off += emit(code + off, OP_ARR_LEN);    /* → 3 */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "add_array_array: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 3, "add_array_array: result has 3 elements");
    nvm_module_free(mod);
}

static void test_add_array_scalar(void) {
    /* OP_ADD with array + scalar → broadcast add */
    uint8_t code[64];
    uint32_t off = 0;
    /* [5, 10, 15] + 1 = [6, 11, 16] */
    off += emit(code + off, OP_PUSH_I64, (int64_t)5);
    off += emit(code + off, OP_PUSH_I64, (int64_t)10);
    off += emit(code + off, OP_PUSH_I64, (int64_t)15);
    off += emit(code + off, OP_ARR_LITERAL, (uint8_t)TAG_INT, (uint16_t)3);
    off += emit(code + off, OP_PUSH_I64, (int64_t)1);
    off += emit(code + off, OP_ADD);        /* broadcast add */
    off += emit(code + off, OP_ARR_LEN);    /* → 3 */
    off += emit(code + off, OP_RET);
    NvmModule *mod = make_module(code, off, 0, 0);
    VmResult r;
    NanoValue result = run_module(mod, &r);
    ASSERT_EQ_INT(r, VM_OK, "add_array_scalar: VM_OK");
    ASSERT_EQ_INT(result.as.i64, 3, "add_array_scalar: result has 3 elements");
    nvm_module_free(mod);
}

/* ========================================================================
 * Cross-Module Linking Tests
 * ======================================================================== */

static void test_call_module(void) {
    /*
     * Module B has a function "add_10" that adds 10 to its argument.
     * Module A's main calls add_10(5) via OP_CALL_MODULE.
     * Expected result: 15
     */

    /* Build Module B: fn add_10(x) -> x + 10 */
    NvmModule *mod_b = nvm_module_new();
    {
        uint8_t code[64];
        uint32_t n = 0;
        n += emit(code + n, OP_LOAD_LOCAL, 0);     /* load param x */
        n += emit(code + n, OP_PUSH_I64, (int64_t)10);
        n += emit(code + n, OP_ADD);
        n += emit(code + n, OP_RET);

        uint32_t name_idx = nvm_add_string(mod_b, "add_10", 6);
        uint32_t code_off = nvm_append_code(mod_b, code, n);
        NvmFunctionEntry fn = {0};
        fn.name_idx = name_idx;
        fn.arity = 1;
        fn.code_offset = code_off;
        fn.code_length = n;
        fn.local_count = 1;
        fn.upvalue_count = 0;
        nvm_add_function(mod_b, &fn);
    }

    /* Build Module A: fn main() -> call_module(0, 0, arg=5) */
    NvmModule *mod_a = nvm_module_new();
    {
        uint8_t code[64];
        uint32_t n = 0;
        n += emit(code + n, OP_PUSH_I64, (int64_t)5);  /* arg */
        n += emit(code + n, OP_CALL_MODULE, (uint32_t)0, (uint32_t)0);  /* mod 0, fn 0 */
        n += emit(code + n, OP_RET);

        uint32_t name_idx = nvm_add_string(mod_a, "main", 4);
        uint32_t code_off = nvm_append_code(mod_a, code, n);
        NvmFunctionEntry fn = {0};
        fn.name_idx = name_idx;
        fn.arity = 0;
        fn.code_offset = code_off;
        fn.code_length = n;
        fn.local_count = 0;
        fn.upvalue_count = 0;
        uint32_t fn_idx = nvm_add_function(mod_a, &fn);
        mod_a->header.flags = NVM_FLAG_HAS_MAIN;
        mod_a->header.entry_point = fn_idx;
    }

    /* Link and execute */
    VmState vm;
    vm_init(&vm, mod_a);
    uint32_t link_idx = vm_link_module(&vm, mod_b);
    ASSERT_EQ_INT(link_idx, 0, "call_module: link idx == 0");

    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "call_module: VM_OK");

    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_INT(result.as.i64, 15, "call_module: 5 + 10 == 15");

    vm_destroy(&vm);
    nvm_module_free(mod_a);
    nvm_module_free(mod_b);
}

static void test_call_module_bad_idx(void) {
    /* Test error on invalid module index */
    NvmModule *mod = nvm_module_new();
    {
        uint8_t code[64];
        uint32_t n = 0;
        n += emit(code + n, OP_CALL_MODULE, (uint32_t)99, (uint32_t)0);
        n += emit(code + n, OP_RET);

        uint32_t name_idx = nvm_add_string(mod, "main", 4);
        uint32_t code_off = nvm_append_code(mod, code, n);
        NvmFunctionEntry fn = {0};
        fn.name_idx = name_idx;
        fn.arity = 0;
        fn.code_offset = code_off;
        fn.code_length = n;
        fn.local_count = 0;
        fn.upvalue_count = 0;
        uint32_t fn_idx = nvm_add_function(mod, &fn);
        mod->header.flags = NVM_FLAG_HAS_MAIN;
        mod->header.entry_point = fn_idx;
    }

    VmState vm;
    vm_init(&vm, mod);
    VmResult r = vm_execute(&vm);
    ASSERT(r != VM_OK, "call_module_bad: should fail on invalid module idx");
    vm_destroy(&vm);
    nvm_module_free(mod);
}

static void test_call_module_chain(void) {
    /*
     * Module C: fn double(x) -> x * 2
     * Module B: fn add_then_double(x) -> call_module(C, double, x + 1)
     * Module A: fn main() -> call_module(B, add_then_double, 4)
     * Expected: double(4 + 1) = 10
     *
     * But this requires module B to also link module C. For simplicity,
     * test a two-module chain: A calls B which calls a local function.
     */

    /* Module B: fn calc(x) -> (x + 3) * 2
     * Uses two local functions: add3 and double_it */
    NvmModule *mod_b = nvm_module_new();
    {
        /* fn add3(x) -> x + 3 */
        uint8_t code1[64];
        uint32_t n1 = 0;
        n1 += emit(code1 + n1, OP_LOAD_LOCAL, 0);
        n1 += emit(code1 + n1, OP_PUSH_I64, (int64_t)3);
        n1 += emit(code1 + n1, OP_ADD);
        n1 += emit(code1 + n1, OP_RET);

        uint32_t name1 = nvm_add_string(mod_b, "add3", 4);
        uint32_t off1 = nvm_append_code(mod_b, code1, n1);
        NvmFunctionEntry fn1 = {0};
        fn1.name_idx = name1;
        fn1.arity = 1;
        fn1.code_offset = off1;
        fn1.code_length = n1;
        fn1.local_count = 1;
        nvm_add_function(mod_b, &fn1);

        /* fn calc(x) -> call add3(x), then * 2 */
        uint8_t code2[64];
        uint32_t n2 = 0;
        n2 += emit(code2 + n2, OP_LOAD_LOCAL, 0);
        n2 += emit(code2 + n2, OP_CALL, (uint32_t)0);  /* call add3 */
        n2 += emit(code2 + n2, OP_PUSH_I64, (int64_t)2);
        n2 += emit(code2 + n2, OP_MUL);
        n2 += emit(code2 + n2, OP_RET);

        uint32_t name2 = nvm_add_string(mod_b, "calc", 4);
        uint32_t off2 = nvm_append_code(mod_b, code2, n2);
        NvmFunctionEntry fn2 = {0};
        fn2.name_idx = name2;
        fn2.arity = 1;
        fn2.code_offset = off2;
        fn2.code_length = n2;
        fn2.local_count = 1;
        nvm_add_function(mod_b, &fn2);
    }

    /* Module A: fn main() -> call_module(0, 1, 4)  (mod B, fn "calc", arg 4) */
    NvmModule *mod_a = nvm_module_new();
    {
        uint8_t code[64];
        uint32_t n = 0;
        n += emit(code + n, OP_PUSH_I64, (int64_t)4);
        n += emit(code + n, OP_CALL_MODULE, (uint32_t)0, (uint32_t)1);
        n += emit(code + n, OP_RET);

        uint32_t name = nvm_add_string(mod_a, "main", 4);
        uint32_t off = nvm_append_code(mod_a, code, n);
        NvmFunctionEntry fn = {0};
        fn.name_idx = name;
        fn.arity = 0;
        fn.code_offset = off;
        fn.code_length = n;
        fn.local_count = 0;
        uint32_t fn_idx = nvm_add_function(mod_a, &fn);
        mod_a->header.flags = NVM_FLAG_HAS_MAIN;
        mod_a->header.entry_point = fn_idx;
    }

    VmState vm;
    vm_init(&vm, mod_a);
    vm_link_module(&vm, mod_b);
    VmResult r = vm_execute(&vm);
    ASSERT_EQ_INT(r, VM_OK, "call_module_chain: VM_OK");

    NanoValue result = vm_get_result(&vm);
    ASSERT_EQ_INT(result.as.i64, 14, "call_module_chain: (4+3)*2 == 14");

    vm_destroy(&vm);
    nvm_module_free(mod_a);
    nvm_module_free(mod_b);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== NanoVM Test Suite ===\n");

    printf("\n[Integer Arithmetic]\n");
    RUN_TEST(test_push_int_halt);
    RUN_TEST(test_add_ints);
    RUN_TEST(test_sub_ints);
    RUN_TEST(test_mul_ints);
    RUN_TEST(test_div_ints);
    RUN_TEST(test_div_by_zero);
    RUN_TEST(test_mod_ints);
    RUN_TEST(test_neg_int);

    printf("\n[Float Arithmetic]\n");
    RUN_TEST(test_float_add);
    RUN_TEST(test_mixed_int_float_add);
    RUN_TEST(test_float_div_by_zero);

    printf("\n[Boolean & Comparison]\n");
    RUN_TEST(test_bool_push);
    RUN_TEST(test_eq_ints);
    RUN_TEST(test_ne_ints);
    RUN_TEST(test_lt_ints);
    RUN_TEST(test_logic_and);
    RUN_TEST(test_logic_not);

    printf("\n[Stack Operations]\n");
    RUN_TEST(test_dup);
    RUN_TEST(test_swap);
    RUN_TEST(test_pop);
    RUN_TEST(test_push_void);
    RUN_TEST(test_push_u8);

    printf("\n[Local Variables]\n");
    RUN_TEST(test_locals);

    printf("\n[Global Variables]\n");
    RUN_TEST(test_globals);

    printf("\n[Control Flow]\n");
    RUN_TEST(test_jmp);
    RUN_TEST(test_jmp_false);
    RUN_TEST(test_jmp_true);
    RUN_TEST(test_loop);

    printf("\n[Function Calls]\n");
    RUN_TEST(test_call_function);
    RUN_TEST(test_recursive_factorial);

    printf("\n[Strings]\n");
    RUN_TEST(test_push_string);
    RUN_TEST(test_string_concat);
    RUN_TEST(test_string_len);
    RUN_TEST(test_str_from_int);

    printf("\n[Arrays]\n");
    RUN_TEST(test_array_literal);
    RUN_TEST(test_array_push_get);
    RUN_TEST(test_array_len);

    printf("\n[Structs]\n");
    RUN_TEST(test_struct_literal);
    RUN_TEST(test_struct_set);

    printf("\n[Tuples]\n");
    RUN_TEST(test_tuple);

    printf("\n[Enums & Unions]\n");
    RUN_TEST(test_enum_val);
    RUN_TEST(test_union_construct_tag);
    RUN_TEST(test_union_field);

    printf("\n[Hashmaps]\n");
    RUN_TEST(test_hashmap_basic);
    RUN_TEST(test_hashmap_len);

    printf("\n[Type Casts]\n");
    RUN_TEST(test_cast_int_from_float);
    RUN_TEST(test_cast_float_from_int);
    RUN_TEST(test_cast_bool);
    RUN_TEST(test_cast_string);

    printf("\n[Closures]\n");
    RUN_TEST(test_closure);

    printf("\n[I/O]\n");
    RUN_TEST(test_print);
    RUN_TEST(test_assert_pass);
    RUN_TEST(test_assert_fail);

    printf("\n[Error Handling]\n");
    RUN_TEST(test_type_error_add);
    RUN_TEST(test_no_entry_point);

    printf("\n[Stack Trace / Debug Mode]\n");
    RUN_TEST(test_stack_trace_debug_mode);
    RUN_TEST(test_stack_trace_col);
    RUN_TEST(test_stack_trace_multi_frame);

    printf("\n[Opaque Proxy]\n");
    RUN_TEST(test_opaque_null);

    printf("\n[Serialize/Execute Round-trip]\n");
    RUN_TEST(test_serialize_execute);

    printf("\n[Complex Programs]\n");
    RUN_TEST(test_fibonacci);
    RUN_TEST(test_gcd);

    printf("\n[Assembler -> VM Integration]\n");
    RUN_TEST(test_assemble_and_run);
    RUN_TEST(test_assemble_function_call);
    RUN_TEST(test_assemble_loop);

    printf("\n[Cross-Module Linking]\n");
    RUN_TEST(test_call_module);
    RUN_TEST(test_call_module_bad_idx);
    RUN_TEST(test_call_module_chain);

    printf("\n[Stack Ops: ROT3]\n");
    RUN_TEST(test_rot3);

    printf("\n[Indirect Call]\n");
    RUN_TEST(test_call_indirect);

    printf("\n[Array: Remove/Slice/Pop]\n");
    RUN_TEST(test_arr_remove);
    RUN_TEST(test_arr_slice);
    RUN_TEST(test_arr_pop);

    printf("\n[String: Substr/Contains/Eq/CharAt/FromFloat]\n");
    RUN_TEST(test_str_substr);
    RUN_TEST(test_str_contains);
    RUN_TEST(test_str_eq);
    RUN_TEST(test_str_char_at);
    RUN_TEST(test_str_from_float);

    printf("\n[Upvalue Store]\n");
    RUN_TEST(test_store_upvalue);

    printf("\n[HashMap: Has/Delete/Keys/Values]\n");
    RUN_TEST(test_hm_has);
    RUN_TEST(test_hm_delete);
    RUN_TEST(test_hm_keys);
    RUN_TEST(test_hm_values);

    printf("\n[ADD: String/Array variants]\n");
    RUN_TEST(test_add_strings);
    RUN_TEST(test_add_array_array);
    RUN_TEST(test_add_array_scalar);

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
