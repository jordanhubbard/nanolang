/*
 * test_codegen.c - Test the nano_virt codegen by compiling .nano source
 * to bytecode and executing via the VM.
 *
 * Each test: parse source → typecheck → codegen → VM execute → assert result
 */

#include "nanolang.h"
#include "nanovirt/codegen.h"
#include "nanoisa/isa.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/vm.h"
#include "nanovm/value.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Stubs for globals expected by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test framework ─────────────────────────────────────────────── */

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_INT(actual, expected) do { \
    int64_t _a = (actual), _e = (expected); \
    if (_a != _e) { \
        fprintf(stderr, "  FAIL: expected %lld, got %lld (line %d)\n", \
                (long long)_e, (long long)_a, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_FLOAT(actual, expected) do { \
    double _a = (actual), _e = (expected); \
    if (fabs(_a - _e) > 0.0001) { \
        fprintf(stderr, "  FAIL: expected %f, got %f (line %d)\n", _e, _a, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_BOOL(actual, expected) do { \
    bool _a = (actual), _e = (expected); \
    if (_a != _e) { \
        fprintf(stderr, "  FAIL: expected %s, got %s (line %d)\n", \
                _e ? "true" : "false", _a ? "true" : "false", __LINE__); \
        tests_failed++; \
        return; \
    } \
} while (0)

#define TEST_PASS() do { tests_passed++; } while (0)

/* ── Helper: compile and run a .nano source, return result from main() ── */

typedef struct {
    bool ok;
    NanoValue result;
    VmResult vm_result;
    NvmModule *module; /* caller must free if ok */
    char error[256];
} TestResult;

static TestResult compile_and_run(const char *source) {
    TestResult tr = {0};

    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        snprintf(tr.error, sizeof(tr.error), "lexer failed");
        return tr;
    }

    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        snprintf(tr.error, sizeof(tr.error), "parser failed");
        free_tokens(tokens, token_count);
        return tr;
    }

    Environment *env = create_environment();
    env->suppress_shadow_warnings = true;
    if (!type_check(program, env)) {
        snprintf(tr.error, sizeof(tr.error), "typecheck failed");
        free_ast(program);
        free_environment(env);
        free_tokens(tokens, token_count);
        return tr;
    }

    CodegenResult cg = codegen_compile(program, env, NULL, NULL);
    free_ast(program);
    free_environment(env);
    free_tokens(tokens, token_count);

    if (!cg.ok) {
        snprintf(tr.error, sizeof(tr.error), "codegen: %.240s", cg.error_msg);
        return tr;
    }

    VmState vm;
    vm_init(&vm, cg.module);

    /* Call __init__ to initialize globals before main */
    for (uint32_t i = 0; i < cg.module->function_count; i++) {
        const char *fn_name = nvm_get_string(cg.module,
                                              cg.module->functions[i].name_idx);
        if (fn_name && strcmp(fn_name, "__init__") == 0) {
            vm_call_function(&vm, i, NULL, 0);
            break;
        }
    }

    tr.vm_result = vm_execute(&vm);
    tr.result = vm_get_result(&vm);
    if (tr.result.tag == TAG_STRING || tr.result.tag == TAG_ARRAY) {
        vm_retain(&vm.heap, tr.result);
    }
    vm_destroy(&vm);

    tr.ok = true;
    tr.module = cg.module;
    return tr;
}

static void test_debug_metadata_is_not_executable(void) {
    const char *source =
        "fn main() -> int {\n"
        "    let x: int = 40\n"
        "    return (+ x 2)\n"
        "}\n"
        "shadow main { assert true }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, "debug metadata compile succeeds");
    ASSERT(tr.module->debug_count > 0, "source map side table is populated");
    for (uint32_t i = 0; i < tr.module->code_size; i++) {
        ASSERT(tr.module->code[i] != OP_DEBUG_LINE,
               "NanoVirt emits no executable DEBUG_LINE instructions");
    }
    nvm_module_free(tr.module);
}

static void test_scalar_codegen_uses_typed_opcodes(void) {
    const char *source =
        "fn main() -> int {\n"
        "    let i: int = (+ 20 22)\n"
        "    let f: float = (+ 1.0 2.0)\n"
        "    let b: bool = (and true false)\n"
        "    if (and (== i 42) (> f 2.0)) { return i }\n"
        "    return 0\n"
        "}\n"
        "shadow main { assert true }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, "typed scalar codegen succeeds");
    bool saw_i64 = false;
    bool saw_f64 = false;
    bool saw_bool = false;
    for (uint32_t i = 0; i < tr.module->code_size;) {
        DecodedInstruction instruction;
        uint32_t width = isa_decode(tr.module->code + i,
                                    tr.module->code_size - i, &instruction);
        ASSERT(width > 0, "typed scalar bytecode decodes");
        ASSERT(instruction.opcode != OP_ADD && instruction.opcode != OP_SUB
               && instruction.opcode != OP_MUL && instruction.opcode != OP_DIV
               && instruction.opcode != OP_MOD && instruction.opcode != OP_NEG
               && instruction.opcode != OP_AND && instruction.opcode != OP_OR
               && instruction.opcode != OP_NOT,
               "scalar codegen emits no legacy polymorphic operations");
        if (instruction.opcode == OP_I64_ADD) saw_i64 = true;
        if (instruction.opcode == OP_F64_ADD) saw_f64 = true;
        if (instruction.opcode == OP_BOOL_AND) saw_bool = true;
        i += width;
    }
    ASSERT(saw_i64 && saw_f64 && saw_bool,
           "typed integer, float, and boolean operations are present");
    nvm_module_free(tr.module);
}

static void test_function_result_signatures(void) {
    const char *source =
        "fn side_effect() -> void { return }\n"
        "fn answer() -> int { return 42 }\n"
        "fn main() -> int { (side_effect) return (answer) }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, "result signature codegen succeeds");
    ASSERT(tr.vm_result == VM_OK, "void call leaves no phantom result");
    ASSERT_INT(tr.result.as.i64, 42);
    for (uint32_t i = 0; i < tr.module->function_count; i++) {
        const char *name = nvm_get_string(tr.module,
                                          tr.module->functions[i].name_idx);
        if (name && strcmp(name, "side_effect") == 0) {
            ASSERT(tr.module->functions[i].result_tag == TAG_VOID,
                   "void function has void result tag");
            ASSERT(tr.module->functions[i].result_count == 0,
                   "void function has zero results");
        }
        if (name && strcmp(name, "answer") == 0) {
            ASSERT(tr.module->functions[i].result_tag == TAG_INT,
                   "int function has int result tag");
            ASSERT(tr.module->functions[i].result_count == 1,
                   "int function has one result");
        }
    }
    nvm_module_free(tr.module);
}

static void test_empty_array_return_tags(void) {
    const char *types[] = {"int", "float", "bool", "string", "Point"};
    const char *values[] = {"42", "1.5", "true", "\"answer\"", "Point { x: 42 }"};
    const uint8_t tags[] = {TAG_INT, TAG_FLOAT, TAG_BOOL, TAG_STRING, TAG_STRUCT};
    for (size_t type = 0; type < sizeof tags / sizeof tags[0]; ++type) {
        char source[2048];
        snprintf(source, sizeof source,
            "struct Point { x: int }\n"
            "fn make(empty: bool) -> array<%s> {\n"
            "  fn nested() -> array<int> { return [] }\n"
            "  let other: array<int> = (nested)\n"
            "  if empty { return [] }\n"
            "  return [%s]\n}\n"
            "shadow make { assert (== (array_length (make true)) 0) }\n"
            "fn main() -> int {\n"
            "  assert (== (array_length (make true)) 0)\n"
            "  assert (== (array_length (make false)) 1)\n"
            "  return 0\n}\n"
            "shadow main { assert (== (main) 0) }\n", types[type], values[type]);
        TestResult tr = compile_and_run(source);
        ASSERT(tr.ok, "I compile declared empty array return types");
        ASSERT(tr.vm_result == VM_OK, "I execute empty and nonempty return paths");
        bool found = false;
        for (uint32_t i = 0; i < tr.module->function_count; ++i) {
            NvmFunctionEntry *fn = &tr.module->functions[i];
            const char *name = nvm_get_string(tr.module, fn->name_idx);
            if (!name || strcmp(name, "make") != 0) continue;
            for (uint32_t pc = fn->code_offset; pc < fn->code_offset + fn->code_length;) {
                DecodedInstruction ins;
                uint32_t width = isa_decode(tr.module->code + pc, fn->code_offset + fn->code_length - pc, &ins);
                ASSERT(width > 0, "I decode empty array return instructions");
                if (ins.opcode == OP_ARR_LITERAL && ins.operands[1].u16 == 0) {
                    ASSERT(ins.operands[0].u8 == tags[type], "I restore the enclosing return element type after nested compilation");
                    found = true;
                }
                pc += width;
            }
        }
        ASSERT(found, "I exercise an emitted empty return literal");
        nvm_module_free(tr.module);
        TEST_PASS();
    }
}

static void test_empty_struct_list_result_keeps_element_tag(void) {
    const char *source =
        "struct Point { x: int }\n"
        "fn make_points() -> List<Point> { return (list_Point_new) }\n"
        "fn main() -> int {\n"
        "    let points: List<Point> = (make_points)\n"
        "    return (list_Point_length points)\n"
        "}\n"
        "shadow make_points { assert (== (list_Point_length (make_points)) 0) }\n"
        "shadow main { assert (== (main) 0) }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, "empty struct list result compiles");
    ASSERT(tr.vm_result == VM_OK, "empty struct list result executes");
    ASSERT_INT(tr.result.as.i64, 0);

    bool saw_make_points = false;
    bool saw_struct_array = false;
    for (uint32_t i = 0; i < tr.module->function_count; i++) {
        NvmFunctionEntry *fn = &tr.module->functions[i];
        const char *name = nvm_get_string(tr.module, fn->name_idx);
        if (!name || strcmp(name, "make_points") != 0) continue;
        saw_make_points = true;
        ASSERT(fn->result_count == 1 && fn->result_tag == TAG_ARRAY,
               "List<Point> function has one array result");
        for (uint32_t pc = fn->code_offset; pc < fn->code_offset + fn->code_length;) {
            DecodedInstruction instruction;
            uint32_t width = isa_decode(tr.module->code + pc,
                                        fn->code_offset + fn->code_length - pc,
                                        &instruction);
            ASSERT(width > 0, "empty struct list bytecode decodes");
            if (instruction.opcode == OP_ARR_NEW &&
                instruction.operands[0].u8 == TAG_STRUCT) {
                saw_struct_array = true;
            }
            pc += width;
        }
    }
    ASSERT(saw_make_points, "make_points function is present");
    ASSERT(saw_struct_array, "empty List<Point> uses ARR_NEW TAG_STRUCT");
    nvm_module_free(tr.module);
}

/* Helper: compile and call a specific function by name */
static TestResult compile_and_call(const char *source, const char *fn_name,
                                    NanoValue *args, uint16_t argc) {
    TestResult tr = {0};

    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        snprintf(tr.error, sizeof(tr.error), "lexer failed");
        return tr;
    }

    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        snprintf(tr.error, sizeof(tr.error), "parser failed");
        free_tokens(tokens, token_count);
        return tr;
    }

    Environment *env = create_environment();
    env->suppress_shadow_warnings = true;
    if (!type_check(program, env)) {
        snprintf(tr.error, sizeof(tr.error), "typecheck failed");
        free_ast(program);
        free_environment(env);
        free_tokens(tokens, token_count);
        return tr;
    }

    CodegenResult cg = codegen_compile(program, env, NULL, NULL);
    free_ast(program);
    free_environment(env);
    free_tokens(tokens, token_count);

    if (!cg.ok) {
        snprintf(tr.error, sizeof(tr.error), "codegen: %.240s", cg.error_msg);
        return tr;
    }

    /* Find function by name */
    int32_t fn_idx = -1;
    for (uint32_t i = 0; i < cg.module->function_count; i++) {
        const char *name = nvm_get_string(cg.module, cg.module->functions[i].name_idx);
        if (name && strcmp(name, fn_name) == 0) {
            fn_idx = (int32_t)i;
            break;
        }
    }

    if (fn_idx < 0) {
        snprintf(tr.error, sizeof(tr.error), "function '%s' not found", fn_name);
        nvm_module_free(cg.module);
        return tr;
    }

    VmState vm;
    vm_init(&vm, cg.module);
    tr.vm_result = vm_call_function(&vm, (uint32_t)fn_idx, args, argc);
    tr.result = vm_get_result(&vm);
    if (tr.result.tag == TAG_STRING || tr.result.tag == TAG_ARRAY) {
        vm_retain(&vm.heap, tr.result);
    }
    vm_destroy(&vm);

    tr.ok = true;
    tr.module = cg.module;
    return tr;
}

static void free_test_result(TestResult *tr) {
    if (tr->module) {
        nvm_module_free(tr->module);
        tr->module = NULL;
    }
}

/* ── Tests: Integer Arithmetic ──────────────────────────────────── */

static void test_return_int(void) {
    fprintf(stderr, "  test_return_int...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return 42 }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_addition(void) {
    fprintf(stderr, "  test_addition...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (+ 10 32) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_subtraction(void) {
    fprintf(stderr, "  test_subtraction...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (- 50 8) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_multiplication(void) {
    fprintf(stderr, "  test_multiplication...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (* 6 7) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_division(void) {
    fprintf(stderr, "  test_division...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (/ 84 2) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_modulo(void) {
    fprintf(stderr, "  test_modulo...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (% 47 5) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 2);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_nested_arithmetic(void) {
    fprintf(stderr, "  test_nested_arithmetic...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (+ (* 2 3) (- 10 4)) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 12);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_negation(void) {
    fprintf(stderr, "  test_negation...");
    TestResult tr = compile_and_run(
        "fn main() -> int { return (- 0 42) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, -42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Let / Set Variables ─────────────────────────────────── */

static void test_let_simple(void) {
    fprintf(stderr, "  test_let_simple...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let x: int = 42\n"
        "  return x\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_set_mutable(void) {
    fprintf(stderr, "  test_set_mutable...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut x: int = 10\n"
        "  set x 42\n"
        "  return x\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_multiple_variables(void) {
    fprintf(stderr, "  test_multiple_variables...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let a: int = 10\n"
        "  let b: int = 32\n"
        "  return (+ a b)\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_set_with_expression(void) {
    fprintf(stderr, "  test_set_with_expression...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut x: int = 10\n"
        "  set x (+ x 32)\n"
        "  return x\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Comparison ──────────────────────────────────────────── */

static void test_comparisons(void) {
    fprintf(stderr, "  test_comparisons...");
    const char *src =
        "fn test_eq() -> int { if (== 5 5) { return 1 } else { return 0 } }\n"
        "fn test_ne() -> int { if (!= 5 3) { return 1 } else { return 0 } }\n"
        "fn test_lt() -> int { if (< 3 5) { return 1 } else { return 0 } }\n"
        "fn test_gt() -> int { if (> 5 3) { return 1 } else { return 0 } }\n"
        "fn test_le() -> int { if (<= 5 5) { return 1 } else { return 0 } }\n"
        "fn test_ge() -> int { if (>= 5 5) { return 1 } else { return 0 } }\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "test_eq", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    tr = compile_and_call(src, "test_ne", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    tr = compile_and_call(src, "test_lt", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    tr = compile_and_call(src, "test_gt", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    tr = compile_and_call(src, "test_le", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    tr = compile_and_call(src, "test_ge", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Boolean and Logic ───────────────────────────────────── */

static void test_bool_literal(void) {
    fprintf(stderr, "  test_bool_literal...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if true { return 1 } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 1);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_logic_and(void) {
    fprintf(stderr, "  test_logic_and...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if (and true true) { return 1 } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 1);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_logic_or(void) {
    fprintf(stderr, "  test_logic_or...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if (or false true) { return 1 } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 1);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_logic_not(void) {
    fprintf(stderr, "  test_logic_not...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if (not false) { return 1 } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 1);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Control Flow ────────────────────────────────────────── */

static void test_if_then_else(void) {
    fprintf(stderr, "  test_if_then_else...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if (> 5 3) { return 1 } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 1);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_if_else_branch(void) {
    fprintf(stderr, "  test_if_else_branch...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if (< 5 3) { return 1 } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 0);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_nested_if(void) {
    fprintf(stderr, "  test_nested_if...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  if (> 10 5) {\n"
        "    if (< 3 7) { return 42 } else { return 0 }\n"
        "  } else { return 0 }\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_while_loop(void) {
    fprintf(stderr, "  test_while_loop...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut sum: int = 0\n"
        "  let mut i: int = 1\n"
        "  while (<= i 10) {\n"
        "    set sum (+ sum i)\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return sum\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 55);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_while_break(void) {
    fprintf(stderr, "  test_while_break...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut i: int = 0\n"
        "  while true {\n"
        "    if (== i 5) { break }\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return i\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 5);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_while_continue(void) {
    fprintf(stderr, "  test_while_continue...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut sum: int = 0\n"
        "  let mut i: int = 0\n"
        "  while (< i 10) {\n"
        "    set i (+ i 1)\n"
        "    if (== (% i 2) 0) { continue }\n"
        "    set sum (+ sum i)\n"
        "  }\n"
        "  return sum\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    /* sum of odd numbers 1-9: 1+3+5+7+9 = 25 */
    ASSERT_INT(tr.result.as.i64, 25);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Functions ───────────────────────────────────────────── */

static void test_array_search_types(void) {
    fprintf(stderr, "  test_array_search_types...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  assert (array_contains [1.5, 2.5] 2.5)\n"
        "  assert (== (array_index_of [1.5, 2.5, 1.5] 1.5) 0)\n"
        "  assert (not (array_contains [true, true] false))\n"
        "  assert (== (array_index_of [false, true] true) 1)\n"
        "  let text: string = (str_concat \"ca\" \"fé\")\n"
        "  assert (array_contains [\"café\", \"tea\"] text)\n"
        "  assert (== (array_index_of [\"café\", \"tea\"] \"tea\") 1)\n"
        "  return 0\n"
        "}\n"
        "shadow main { assert (== (main) 0) }\n");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM search error");
    ASSERT_INT(tr.result.as.i64, 0);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_function_call(void) {
    fprintf(stderr, "  test_function_call...");
    TestResult tr = compile_and_run(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn main() -> int { return (add 10 32) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_function_multiple_calls(void) {
    fprintf(stderr, "  test_function_multiple_calls...");
    TestResult tr = compile_and_run(
        "fn double(x: int) -> int { return (* x 2) }\n"
        "fn main() -> int { return (+ (double 10) (double 11)) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_recursion_factorial(void) {
    fprintf(stderr, "  test_recursion_factorial...");
    TestResult tr = compile_and_call(
        "fn fact(n: int) -> int {\n"
        "  if (<= n 1) { return 1 }\n"
        "  return (* n (fact (- n 1)))\n"
        "}\n"
        "fn main() -> int { return 0 }",
        "fact", (NanoValue[]){ val_int(10) }, 1);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 3628800);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_recursion_fibonacci(void) {
    fprintf(stderr, "  test_recursion_fibonacci...");
    TestResult tr = compile_and_call(
        "fn fib(n: int) -> int {\n"
        "  if (<= n 1) { return n }\n"
        "  return (+ (fib (- n 1)) (fib (- n 2)))\n"
        "}\n"
        "fn main() -> int { return 0 }",
        "fib", (NanoValue[]){ val_int(10) }, 1);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 55);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_function_nested_calls(void) {
    fprintf(stderr, "  test_function_nested_calls...");
    TestResult tr = compile_and_run(
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn mul(a: int, b: int) -> int { return (* a b) }\n"
        "fn main() -> int { return (add (mul 2 3) (mul 4 9)) }");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Strings ─────────────────────────────────────────────── */

static void test_string_literal(void) {
    fprintf(stderr, "  test_string_literal...");
    /* Just compile and run - no crash means success */
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let s: string = \"hello\"\n"
        "  assert (== s \"hello\")\n"
        "  return 0\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 0);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Print ───────────────────────────────────────────────── */

static void test_print_int(void) {
    fprintf(stderr, "  test_print_int...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  print 42\n"
        "  return 0\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Assert ──────────────────────────────────────────────── */

static void test_assert_true(void) {
    fprintf(stderr, "  test_assert_true...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  assert true\n"
        "  return 0\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_assert_false(void) {
    fprintf(stderr, "  test_assert_false...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  assert false\n"
        "  return 0\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_ERR_ASSERT_FAILED, "expected assert failure");
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Float ───────────────────────────────────────────────── */

static void test_float_arithmetic(void) {
    fprintf(stderr, "  test_float_arithmetic...");
    TestResult tr = compile_and_call(
        "fn compute() -> float { return (+ 3.14 2.86) }\n"
        "fn main() -> int { return 0 }",
        "compute", NULL, 0);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_FLOAT(tr.result.as.f64, 6.0);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Selfhost-style programs ─────────────────────────────── */

static void test_selfhost_arithmetic(void) {
    fprintf(stderr, "  test_selfhost_arithmetic...");
    const char *src =
        "fn test_addition() -> int {\n"
        "  let a: int = (+ 5 3)\n"
        "  let b: int = (+ 10 20)\n"
        "  return (+ a b)\n"
        "}\n"
        "fn test_complex() -> int {\n"
        "  return (* (+ 2 3) (- 10 3))\n"
        "}\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "test_addition", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 38); free_test_result(&tr);
    tr = compile_and_call(src, "test_complex", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 35); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_selfhost_let_set(void) {
    fprintf(stderr, "  test_selfhost_let_set...");
    const char *src =
        "fn test_immutable() -> int { let x: int = 42\n return x }\n"
        "fn test_mutable() -> int { let mut x: int = 10\n set x 20\n return x }\n"
        "fn test_multi_set() -> int { let mut x: int = 1\n set x 2\n set x 3\n return x }\n"
        "fn test_set_expr() -> int { let mut x: int = 10\n set x (+ x 5)\n return x }\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "test_immutable", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 42); free_test_result(&tr);
    tr = compile_and_call(src, "test_mutable", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 20); free_test_result(&tr);
    tr = compile_and_call(src, "test_multi_set", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 3); free_test_result(&tr);
    tr = compile_and_call(src, "test_set_expr", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 15); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_selfhost_if_else(void) {
    fprintf(stderr, "  test_selfhost_if_else...");
    const char *src =
        "fn test_simple_if() -> int {\n"
        "  if (> 5 3) { return 1 } else { return 0 }\n"
        "}\n"
        "fn test_nested() -> int {\n"
        "  if (> 10 5) {\n"
        "    if (< 3 7) { return 42 } else { return 0 }\n"
        "  } else { return 0 }\n"
        "}\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "test_simple_if", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 1); free_test_result(&tr);
    tr = compile_and_call(src, "test_nested", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 42); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_selfhost_while(void) {
    fprintf(stderr, "  test_selfhost_while...");
    const char *src =
        "fn test_sum_loop() -> int {\n"
        "  let mut sum: int = 0\n"
        "  let mut i: int = 1\n"
        "  while (<= i 100) {\n"
        "    set sum (+ sum i)\n"
        "    set i (+ i 1)\n"
        "  }\n"
        "  return sum\n"
        "}\n"
        "fn test_countdown() -> int {\n"
        "  let mut n: int = 10\n"
        "  while (> n 0) {\n"
        "    set n (- n 1)\n"
        "  }\n"
        "  return n\n"
        "}\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "test_sum_loop", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 5050); free_test_result(&tr);
    tr = compile_and_call(src, "test_countdown", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 0); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_selfhost_functions(void) {
    fprintf(stderr, "  test_selfhost_functions...");
    const char *src =
        "fn add(a: int, b: int) -> int { return (+ a b) }\n"
        "fn square(x: int) -> int { return (* x x) }\n"
        "fn test_basic_call() -> int { return (add 10 32) }\n"
        "fn test_nested_call() -> int { return (add (square 3) (square 4)) }\n"
        "fn test_three_args(a: int, b: int, c: int) -> int {\n"
        "  return (+ (+ a b) c)\n"
        "}\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "test_basic_call", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 42); free_test_result(&tr);
    tr = compile_and_call(src, "test_nested_call", NULL, 0);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 25); free_test_result(&tr);
    tr = compile_and_call(src, "test_three_args",
                          (NanoValue[]){ val_int(10), val_int(20), val_int(12) }, 3);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 42); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_selfhost_recursion(void) {
    fprintf(stderr, "  test_selfhost_recursion...");
    const char *src =
        "fn factorial(n: int) -> int {\n"
        "  if (<= n 1) { return 1 }\n"
        "  return (* n (factorial (- n 1)))\n"
        "}\n"
        "fn fibonacci(n: int) -> int {\n"
        "  if (<= n 1) { return n }\n"
        "  return (+ (fibonacci (- n 1)) (fibonacci (- n 2)))\n"
        "}\n"
        "fn gcd(a: int, b: int) -> int {\n"
        "  if (== b 0) { return a }\n"
        "  return (gcd b (% a b))\n"
        "}\n"
        "fn main() -> int { return 0 }\n";

    TestResult tr;
    tr = compile_and_call(src, "factorial", (NanoValue[]){ val_int(10) }, 1);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 3628800); free_test_result(&tr);
    tr = compile_and_call(src, "fibonacci", (NanoValue[]){ val_int(10) }, 1);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 55); free_test_result(&tr);
    tr = compile_and_call(src, "gcd", (NanoValue[]){ val_int(48), val_int(18) }, 2);
    ASSERT(tr.ok, tr.error); ASSERT_INT(tr.result.as.i64, 6); free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Serialize + Execute round-trip ──────────────────────── */

static void test_serialize_and_run(void) {
    fprintf(stderr, "  test_serialize_and_run...");
    /* Compile, serialize to bytes, deserialize, run */
    const char *source = "fn main() -> int { return (+ 10 32) }";

    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    ASSERT(tokens != NULL, "lexer failed");
    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program != NULL, "parser failed");
    Environment *env = create_environment();
    env->suppress_shadow_warnings = true;
    ASSERT(type_check(program, env), "typecheck failed");
    CodegenResult cg = codegen_compile(program, env, NULL, NULL);
    ASSERT(cg.ok, cg.error_msg);

    uint32_t size = 0;
    uint8_t *blob = nvm_serialize(cg.module, &size);
    ASSERT(blob != NULL, "serialize failed");
    nvm_module_free(cg.module);

    NvmModule *mod2 = nvm_deserialize(blob, size);
    ASSERT(mod2 != NULL, "deserialize failed");
    free(blob);

    VmState vm;
    vm_init(&vm, mod2);
    VmResult r = vm_execute(&vm);
    ASSERT(r == VM_OK, "VM error");
    ASSERT_INT(vm_get_result(&vm).as.i64, 42);
    vm_destroy(&vm);
    nvm_module_free(mod2);

    free_ast(program);
    free_environment(env);
    free_tokens(tokens, token_count);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Array Literals ──────────────────────────────────────── */

static void test_array_literal(void) {
    fprintf(stderr, "  test_array_literal...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let arr: array<int> = [10, 20, 30]\n"
        "  return (at arr 1)\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 20);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_array_length(void) {
    fprintf(stderr, "  test_array_length...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let arr: array<int> = [1, 2, 3, 4, 5]\n"
        "  return (array_length arr)\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 5);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_array_push(void) {
    fprintf(stderr, "  test_array_push...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut arr: array<int> = [1, 2]\n"
        "  set arr (array_push arr 3)\n"
        "  return (+ (array_length arr) (at arr 2))\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 6);  /* 3 + 3 */
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_array_set(void) {
    fprintf(stderr, "  test_array_set...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut arr: array<int> = [10, 20, 30]\n"
        "  (array_set arr 1 99)\n"
        "  return (at arr 1)\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 99);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Range ──────────────────────────────────────────────── */

static void test_range_two_args(void) {
    fprintf(stderr, "  test_range_two_args...");
    /* range(start, end) used in for loop (typechecker expects 2-arg range) */
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut sum: int = 0\n"
        "  for i in (range 3 7) {\n"
        "    set sum (+ sum i)\n"
        "  }\n"
        "  return sum\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 18);  /* 3+4+5+6 */
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_for_in_range(void) {
    fprintf(stderr, "  test_for_in_range...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let mut sum: int = 0\n"
        "  for i in (range 1 6) {\n"
        "    set sum (+ sum i)\n"
        "  }\n"
        "  return sum\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 15);  /* 1+2+3+4+5 */
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Tests: Type Cast Builtins ─────────────────────────────────── */

static void test_int_to_string(void) {
    fprintf(stderr, "  test_int_to_string...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let s: string = (int_to_string 42)\n"
        "  return (str_length s)\n"
        "}");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "VM error");
    ASSERT_INT(tr.result.as.i64, 2);  /* "42" is 2 chars */
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Main ───────────────────────────────────────────────────────── */

/* ── Phase 4: Complex Types Tests ──────────────────────────────── */

static void test_struct_literal(void) {
    fprintf(stderr, "  test_struct_literal...");
    TestResult tr = compile_and_run(
        "struct Point { x: int, y: int }\n"
        "fn main() -> int {\n"
        "  let p: Point = Point { x: 10, y: 20 }\n"
        "  return p.x\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 10);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_struct_field_access(void) {
    fprintf(stderr, "  test_struct_field_access...");
    TestResult tr = compile_and_run(
        "struct Point { x: int, y: int }\n"
        "fn main() -> int {\n"
        "  let p: Point = Point { x: 3, y: 7 }\n"
        "  return (+ p.x p.y)\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 10);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_tuple_literal(void) {
    fprintf(stderr, "  test_tuple_literal...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let t: (int, int) = (42, 99)\n"
        "  return t.0\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 42);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_tuple_index(void) {
    fprintf(stderr, "  test_tuple_index...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let t: (int, int, int) = (10, 20, 30)\n"
        "  return (+ t.0 (+ t.1 t.2))\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 60);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_enum_variant(void) {
    fprintf(stderr, "  test_enum_variant...");
    TestResult tr = compile_and_run(
        "enum Color { Red, Green, Blue }\n"
        "fn main() -> int {\n"
        "  let c: Color = Color.Green\n"
        "  assert (== c Color.Green)\n"
        "  return 42\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 42);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_union_construct_match(void) {
    fprintf(stderr, "  test_union_construct_match...");
    TestResult tr = compile_and_run(
        "union MyResult {\n"
        "  Ok { value: int },\n"
        "  Err { error: string }\n"
        "}\n"
        "fn main() -> int {\n"
        "  let r: MyResult = MyResult.Ok { value: 42 }\n"
        "  match r {\n"
        "    Ok(v) => { return v.value }\n"
        "    Err(_e) => { return 0 }\n"
        "  }\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 42);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_union_match_err_path(void) {
    fprintf(stderr, "  test_union_match_err_path...");
    TestResult tr = compile_and_run(
        "union MyResult {\n"
        "  Ok { value: int },\n"
        "  Err { error: string }\n"
        "}\n"
        "fn main() -> int {\n"
        "  let r: MyResult = MyResult.Err { error: \"bad\" }\n"
        "  match r {\n"
        "    Ok(v) => { return v.value }\n"
        "    Err(_e) => { return -1 }\n"
        "  }\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, -1);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_global_constant(void) {
    fprintf(stderr, "  test_global_constant...");
    TestResult tr = compile_and_run(
        "let MAX: int = 100\n"
        "fn main() -> int {\n"
        "  return MAX\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 100);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_more_than_128_globals(void) {
    fprintf(stderr, "  test_more_than_128_globals...");
    char source[8192];
    size_t used = 0;

    for (int i = 0; i < 129; i++) {
        int written = snprintf(source + used, sizeof(source) - used,
                               "let GLOBAL_%d: int = %d\n", i, i);
        ASSERT(written > 0 && (size_t)written < sizeof(source) - used,
               "test source buffer overflow");
        used += (size_t)written;
    }
    int written = snprintf(source + used, sizeof(source) - used,
                           "fn main() -> int { return GLOBAL_128 }\n");
    ASSERT(written > 0 && (size_t)written < sizeof(source) - used,
           "test source buffer overflow");

    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 128);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_function_as_value(void) {
    fprintf(stderr, "  test_function_as_value...");
    TestResult tr = compile_and_run(
        "fn double(x: int) -> int { return (* x 2) }\n"
        "fn apply(f: fn(int) -> int, x: int) -> int {\n"
        "  return (f x)\n"
        "}\n"
        "fn main() -> int {\n"
        "  return (apply double 21)\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 42);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_filter_builtin(void) {
    fprintf(stderr, "  test_filter_builtin...");
    TestResult tr = compile_and_run(
        "fn is_even(x: int) -> bool { return (== (% x 2) 0) }\n"
        "fn main() -> int {\n"
        "  let nums: array<int> = [1, 2, 3, 4, 5, 6]\n"
        "  let evens: array<int> = (filter nums is_even)\n"
        "  return (array_length evens)\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 3);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_map_builtin(void) {
    fprintf(stderr, "  test_map_builtin...");
    TestResult tr = compile_and_run(
        "fn double(x: int) -> int { return (* x 2) }\n"
        "fn main() -> int {\n"
        "  let nums: array<int> = [1, 2, 3]\n"
        "  let doubled: array<int> = (map nums double)\n"
        "  return (at doubled 1)\n"
        "}\n"
    );
    ASSERT(tr.ok, "compile/run failed");
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 4);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

/* ── Closure capture tests ──────────────────────────────────────── */

static void test_transitive_anonymous_captures(void) {
    fprintf(stderr, "  test_transitive_anonymous_captures...");
    TestResult tr = compile_and_run(
        "fn make(n: int) -> fn() -> fn() -> int {\n"
        "  return fn() -> fn() -> int { return fn() -> int { return (+ n 2) } }\n"
        "}\n"
        "fn main() -> int {\n"
        "  let middle: fn() -> fn() -> int = (make 40)\n"
        "  let inner: fn() -> int = (middle)\n"
        "  return (inner)\n"
        "}\n");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_transitive_named_captures(void) {
    fprintf(stderr, "  test_transitive_named_captures...");
    TestResult tr = compile_and_run(
        "fn make(n: int, extra: int) -> fn() -> fn() -> int {\n"
        "  fn middle() -> fn() -> int {\n"
        "    let base: int = n\n"
        "    fn inner() -> int { return (+ (+ base n) extra) }\n"
        "    return inner\n"
        "  }\n"
        "  return middle\n"
        "}\n"
        "fn main() -> int {\n"
        "  let middle: fn() -> fn() -> int = (make 20 2)\n"
        "  let inner: fn() -> int = (middle)\n"
        "  return (inner)\n"
        "}\n");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 42);
    free_test_result(&tr);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_closure_single_capture(void) {
    fprintf(stderr, "  test_closure_single_capture...");
    TestResult tr = compile_and_run(
        "fn make_adder(n: int) -> fn(int) -> int {\n"
        "  fn adder(x: int) -> int { return (+ x n) }\n"
        "  return adder\n"
        "}\n"
        "fn main() -> int {\n"
        "  let add5: fn(int) -> int = (make_adder 5)\n"
        "  return (add5 10)\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 15);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_closure_multiple_captures(void) {
    fprintf(stderr, "  test_closure_multiple_captures...");
    TestResult tr = compile_and_run(
        "fn make_linear(a: int, b: int) -> fn(int) -> int {\n"
        "  fn linear(x: int) -> int {\n"
        "    return (+ (* a x) b)\n"
        "  }\n"
        "  return linear\n"
        "}\n"
        "fn main() -> int {\n"
        "  let f: fn(int) -> int = (make_linear 3 5)\n"
        "  return (f 10)\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 35);  /* 3*10 + 5 */
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_closure_two_closures(void) {
    fprintf(stderr, "  test_closure_two_closures...");
    TestResult tr = compile_and_run(
        "fn make_adder(n: int) -> fn(int) -> int {\n"
        "  fn adder(x: int) -> int { return (+ x n) }\n"
        "  return adder\n"
        "}\n"
        "fn main() -> int {\n"
        "  let add3: fn(int) -> int = (make_adder 3)\n"
        "  let add7: fn(int) -> int = (make_adder 7)\n"
        "  return (+ (add3 10) (add7 10))\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 30);  /* 13 + 17 */
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_closure_capture_local_var(void) {
    fprintf(stderr, "  test_closure_capture_local_var...");
    TestResult tr = compile_and_run(
        "fn make_multiplier(factor: int) -> fn(int) -> int {\n"
        "  let f: int = factor\n"
        "  fn mul(x: int) -> int { return (* x f) }\n"
        "  return mul\n"
        "}\n"
        "fn main() -> int {\n"
        "  let triple: fn(int) -> int = (make_multiplier 3)\n"
        "  return (triple 7)\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 21);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_declared_function_parameter_tags(void) {
    TestResult tr = compile_and_run(
        "fn choose(value: float, enabled: bool) -> float {\n"
        "  if enabled { return value } return 0.0\n"
        "}\n"
        "shadow choose { assert (== (choose 2.5 true) 2.5) }\n"
        "fn outer(base: int) -> int {\n"
        "  fn inner(value: int) -> int { return (+ base value) }\n"
        "  fn scale(amount: float) -> float { return (* amount 2.0) }\n"
        "  assert (== (scale 1.5) 3.0)\n"
        "  fn check_base() -> void { assert (== base 3) }\n"
        "  (check_base)\n"
        "  return (inner 4)\n"
        "}\n"
        "shadow outer { assert (== (outer 3) 7) }\n"
        "fn main() -> int { return (outer 3) }\n"
        "shadow main { assert (== (main) 7) }\n");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I execute the typed signature fixture");
    ASSERT_INT(tr.result.as.i64, 7);
    uint32_t choose = nvm_find_function(tr.module, "choose");
    uint32_t inner = nvm_find_function(tr.module, "inner");
    ASSERT(choose != UINT32_MAX && inner != UINT32_MAX, "I retain named and nested functions");
    ASSERT(tr.module->function_param_types[choose] != NULL, "I retain declared parameter tags");
    ASSERT(tr.module->function_param_types[choose][0] == TAG_FLOAT &&
           tr.module->function_param_types[choose][1] == TAG_BOOL,
           "I preserve mixed parameter order");
    ASSERT(tr.module->function_param_types[inner] != NULL &&
           tr.module->function_param_types[inner][0] == TAG_INT,
           "I preserve captured function parameter tags");
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_callback_contract_binding(void) {
    const char *source =
        "extern fn submit(context: int, first: fn(int, float) -> bool, second: fn(u8) -> void) -> void\n"
        "extern fn wait(context: int) -> void\n"
        "fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n";
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    ASSERT(tokens, "I tokenize callback declarations");
    ASTNode *program = parse_program(tokens, token_count);
    ASSERT(program && program->as.program.count == 4, "I parse callback declarations");
    Environment *env = create_environment();
    ASSERT(type_check(program, env), "I typecheck callback declarations");
    NvmModule *m = nvm_module_new();
    uint32_t library = nvm_add_string(m, "fixture", 7);
    uint32_t name = nvm_add_string(m, "submit", 6);
    uint8_t tags[] = {TAG_INT, TAG_FUNCTION, TAG_FUNCTION};
    ASSERT(nvm_add_import(m, library, name, 3, TAG_VOID, tags) == 0, "I add the submit import");
    ASTNode *declaration = program->as.program.items[0];
    FunctionSignature *signature = declaration->as.function.params[1].fn_sig;
    ASSERT(signature, "I retain the declared callback shape");
    declaration->as.function.params[1].fn_sig = NULL;
    ASSERT(!codegen_bind_callback_contract(m, 0, declaration, env, "retained_submit", true),
           "I reject a missing callback signature");
    declaration->as.function.params[1].fn_sig = signature;
    signature->param_count = 17;
    ASSERT(!codegen_bind_callback_contract(m, 0, declaration, env, "retained_submit", true),
           "I reject a callback above the native argument limit");
    signature->param_count = 2;
    signature->param_types[0] = TYPE_STRING;
    ASSERT(!codegen_bind_callback_contract(m, 0, declaration, env, "retained_submit", true),
           "I reject non-scalar callback arguments");
    ASSERT(m->callback_contract_count == 0, "I reject shapes before adding any contract");
    signature->param_types[0] = TYPE_INT;
    signature->return_type = TYPE_ARRAY;
    ASSERT(!codegen_bind_callback_contract(m, 0, declaration, env, "retained_submit", true),
           "I reject non-scalar callback results");
    signature->return_type = TYPE_BOOL;
    ASSERT(codegen_bind_callback_contract(m, 0, declaration, env, "retained_submit", true),
           "I bind both declared callbacks");
    ASSERT(m->callback_contract_count == 2, "I preserve both callback parameters");
    ASSERT(m->callback_contracts[0].param_count == 2 &&
           m->callback_contracts[0].param_tags[0] == TAG_INT &&
           m->callback_contracts[0].param_tags[1] == TAG_FLOAT &&
           m->callback_contracts[0].return_tag == TAG_BOOL, "I preserve the first callback shape");
    ASSERT(m->callback_contracts[1].param_count == 1 &&
           m->callback_contracts[1].param_tags[0] == TAG_U8 &&
           m->callback_contracts[1].return_tag == TAG_VOID, "I preserve the second callback shape");
    name = nvm_add_string(m, "wait", 4);
    ASSERT(nvm_add_import(m, library, name, 1, TAG_VOID, tags) == 1, "I add a wait import");
    ASSERT(codegen_bind_callback_contract(m, 1, program->as.program.items[1], env, "retained_wait", false),
           "I bind a policy-only import");
    ASSERT(m->callback_contracts[2].parameter_idx == NVM_CALLBACK_NO_PARAMETER &&
           m->callback_contracts[2].execution == NVM_FOREIGN_OWNER_THREAD,
           "I preserve an explicit owner-thread policy");
    ASSERT(nvm_callback_contracts_valid(m), "I validate the bound table");
    nvm_module_free(m);
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    TEST_PASS();
}

static void test_compiler_local_limit(void) {
    for (int count = 1024; count <= 1025; ++count) {
        char *source = malloc(65536);
        ASSERT(source != NULL, "local-limit source allocation");
        size_t used = (size_t)snprintf(source, 65536, "fn main() -> int {\n");
        for (int i = 0; i < count; ++i) {
            used += (size_t)snprintf(source + used, 65536 - used,
                                     "let local_%d: int = %d\n", i, i);
        }
        snprintf(source + used, 65536 - used, "return local_%d\n}\n", count - 1);
        TestResult result = compile_and_run(source);
        free(source);
        if (count == 1024) {
            ASSERT(result.ok, "I compile 1024 compiler-sized locals");
            ASSERT(result.vm_result == VM_OK, "I execute the expanded local frame");
            ASSERT_INT(result.result.as.i64, 1023);
            nvm_module_free(result.module);
        } else {
            ASSERT(!result.ok && strstr(result.error, "too many local variables"),
                   "I reject a local beyond the bounded compiler table");
        }
    }
    TEST_PASS();
}

static void test_block_local_shadowing(void) {
    fprintf(stderr, "  test_block_local_shadowing...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let value: int = 5\n"
        "  if true { let value: int = 9 assert (== value 9) }\n"
        "  return value\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 5);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_unsafe_block_local_shadowing(void) {
    fprintf(stderr, "  test_unsafe_block_local_shadowing...");
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let fd: int = 5\n"
        "  unsafe { let fd: int = 9 assert (== fd 9) }\n"
        "  return fd\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 5);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_nested_closure_keeps_block_capture(void) {
    fprintf(stderr, "  test_nested_closure_keeps_block_capture...");
    TestResult tr = compile_and_run(
        "fn make_callback() -> fn(int) -> int {\n"
        "  let value: int = 5\n"
        "  fn callback(x: int) -> int { return (+ value x) }\n"
        "  if true { let value: int = 8 assert (== value 8) }\n"
        "  return callback\n"
        "}\n"
        "fn main() -> int {\n"
        "  let f: fn(int) -> int = (make_callback)\n"
        "  return (f 2)\n"
        "}\n"
    );
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "vm error");
    ASSERT_INT(tr.result.as.i64, 7);
    nvm_module_free(tr.module);
    TEST_PASS();
    fprintf(stderr, " ok\n");
}

static void test_anonymous_callback_captures_lexical_array(void) {
    TestResult tr = compile_and_run(
        "fn main() -> int {\n"
        "  let observed: array<int> = [5]\n"
        "  unsafe {\n"
        "    let callback: fn() -> int = fn() -> int {\n"
        "      (array_set observed 0 (+ (at observed 0) 2))\n"
        "      return (at observed 0)\n"
        "    }\n"
        "    assert (== (callback) 7)\n"
        "  }\n"
        "  return (at observed 0)\n"
        "}\n");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I execute the lifted callback");
    ASSERT_INT(tr.result.as.i64, 7);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_handler_observes_perform(void) {
    const char *source =
        "\n"
        "effect Recorder { emit : int -> void }\n"
        "let mut recorded: int = 0\n"
        "\n"
        "fn send(value: int) -> void {\n"
        "    perform Recorder.emit(value)\n"
        "}\n"
        "\n"
        "fn exercise() -> int {\n"
        "    set recorded 0\n"
        "    let ignored = handle { (send 7) } with {\n"
        "        emit value -> { set recorded value }\n"
        "    }\n"
        "    return recorded\n"
        "}\n"
        "\n"
        "shadow send { assert (== (exercise) 7) }\n"
        "shadow exercise { assert (== (exercise) 7) }\n"
        "fn main() -> int {\n"
        "    assert (== (exercise) 7)\n"
        "    (println \"I dispatched the effect.\")\n"
        "    return 0\n"
        "}\n"
        "shadow main { assert (== (main) 0) }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 0);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_lexical_return_and_final_expression(void) {
    const char *source =
        "\n"
        "effect Ask { ask : int -> int }\n"
        "let mut trace: int = 0\n"
        "fn send() -> int {\n"
        "    let x = perform Ask.ask(7)\n"
        "    set trace (+ trace 1)\n"
        "    return (+ x 10)\n"
        "}\n"
        "fn leave() -> int {\n"
        "    let x = handle { (send) } with { ask n -> { return n } }\n"
        "    set trace 100\n"
        "    return x\n"
        "}\n"
        "fn resume_value() -> int {\n"
        "    let x = handle { (send) } with { ask n -> { (+ n 1) } }\n"
        "    return (+ x 100)\n"
        "}\n"
        "fn exercise() -> int {\n"
        "    set trace 0\n"
        "    let left = (leave)\n"
        "    assert (== trace 0)\n"
        "    let resumed = (resume_value)\n"
        "    assert (== trace 1)\n"
        "    return (+ left resumed)\n"
        "}\n"
        "shadow send { assert (== (exercise) 125) }\n"
        "shadow leave { assert (== (exercise) 125) }\n"
        "shadow resume_value { assert (== (exercise) 125) }\n"
        "shadow exercise { assert (== (exercise) 125) }\n"
        "fn main() -> int {\n"
        "    assert (== (exercise) 125)\n"
        "    (println \"I dispatched the effect.\")\n"
        "    return 0\n"
        "}\n"
        "shadow main { assert (== (main) 0) }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 0);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_ordered_multiple_and_zero_arguments(void) {
    const char *source =
        "\n"
        "effect Recorder { pair : int int -> void, tick : void -> void }\n"
        "let mut trace: int = 0\n"
        "let mut recorded: int = 0\n"
        "fn argument(value: int) -> int { set trace (+ (* trace 10) value) return value }\n"
        "fn exercise() -> int {\n"
        "    set trace 0 set recorded 0\n"
        "    let first: int = 9\n"
        "    let ignored = handle { perform Recorder.pair((argument 1) (+ first (argument 2))) } with {\n"
        "        pair first second -> { set recorded (+ (* first 100) second) }\n"
        "    }\n"
        "    let ticked = handle { perform Recorder.tick() } with {\n"
        "        tick -> { set recorded (+ recorded 1000) }\n"
        "    }\n"
        "    return (+ (* trace 10000) recorded)\n"
        "}\n"
        "shadow argument { assert (== (exercise) 121111) }\n"
        "shadow exercise { assert (== (exercise) 121111) }\n"
        "fn main() -> int {\n"
        "    assert (== (exercise) 121111)\n"
        "    (println \"I dispatched the effect.\")\n"
        "    return 0\n"
        "}\n"
        "shadow main { assert (== (main) 0) }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 0);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_effect_lexical_locals(void) {
    const char *source =
        "\n"
        "effect Change { change : int -> int }\n"
        "fn send(n: int) -> int { return perform Change.change(n) }\n"
        "fn main() -> int {\n"
        " let mut state: int = 3\n"
        " let result = handle { (+ 20 (send 4)) } with { change n -> { set state (+ state n) state } }\n"
        " return (+ (* state 100) result)\n"
        "}\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 727);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_effect_nested_handlers(void) {
    const char *source =
        "\n"
        "effect Ask { ask : int -> int }\n"
        "fn send(n: int) -> int { return perform Ask.ask(n) }\n"
        "fn inner() -> int {\n"
        " let result = handle { (send 2) } with { ask n -> { (+ n 10) } }\n"
        " return (+ (* result 1000) (send 3))\n"
        "}\n"
        "fn main() -> int {\n"
        " return handle { (inner) } with { ask n -> { (+ n 100) } }\n"
        "}\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 12103);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_effect_recursive_parameters(void) {
    const char *source =
        "\n"
        "effect Ask { ask : int -> int }\n"
        "fn send(n: int) -> int { return perform Ask.ask(n) }\n"
        "fn main() -> int {\n"
        " let result = handle { (send 3) } with { ask n -> {\n"
        "   let mut child: int = 0\n"
        "   if (> n 0) { set child (send (- n 1)) }\n"
        "   (+ n child)\n"
        " } }\n"
        " return result\n"
        "}\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 6);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_effect_float_signature(void) {
    const char *source =
        "\n"
        "effect Scale { scale : float -> float }\n"
        "fn send(n: float) -> float { return perform Scale.scale(n) }\n"
        "fn main() -> int {\n"
        " let result = handle { (send 1.5) } with { scale n -> { (* n 2.0) } }\n"
        " assert (== result 3.0)\n"
        " return 17\n"
        "}\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 17);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_effect_local_string_ownership(void) {
    const char *source =
        "\n"
        "effect Text { replace : string -> string }\n"
        "fn send(n: string) -> string { return perform Text.replace(n) }\n"
        "fn main() -> int {\n"
        " let mut text: string = \"old\"\n"
        " let result = handle { (send \"new\") } with { replace n -> { set text (+ n \"!\") text } }\n"
        " assert (== text \"new!\")\n"
        " assert (== result \"new!\")\n"
        " return 19\n"
        "}\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 19);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_effect_recursive_lexical_return(void) {
    const char *source =
        "\n"
        "effect Ask { ask : int -> int }\n"
        "fn send(n: int) -> int { return (+ 100 (perform Ask.ask(n))) }\n"
        "fn leave() -> int {\n"
        " return (+ 1000 (handle { (send 3) } with { ask n -> {\n"
        "   if (> n 0) { let ignored = (send (- n 1)) }\n"
        "   return (+ n 7)\n"
        " } }))\n"
        "}\n"
        "fn main() -> int { return (leave) }\n";
    TestResult tr = compile_and_run(source);
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_OK, "I require successful effect execution");
    ASSERT_INT(tr.result.as.i64, 7);
    nvm_module_free(tr.module);
    TEST_PASS();
}

static void test_unhandled_effect_traps(void) {
    TestResult tr = compile_and_run(
        "effect Ask { ask : int -> int }\n"
        "fn main() -> int { return perform Ask.ask(1) }\n");
    ASSERT(tr.ok, tr.error);
    ASSERT(tr.vm_result == VM_ERR_TYPE_ERROR, "I trap an unhandled effect");
    nvm_module_free(tr.module);
    TEST_PASS();
}

int main(void) {
    test_unhandled_effect_traps();
    test_handler_observes_perform();
    test_lexical_return_and_final_expression();
    test_ordered_multiple_and_zero_arguments();
    test_effect_lexical_locals();
    test_effect_nested_handlers();
    test_effect_recursive_parameters();
    test_effect_float_signature();
    test_effect_local_string_ownership();
    test_effect_recursive_lexical_return();
    test_anonymous_callback_captures_lexical_array();
    test_empty_array_return_tags();
    test_array_search_types();
    test_compiler_local_limit();
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    fprintf(stderr, "\n=== NanoVirt Codegen Tests ===\n\n");

    fprintf(stderr, "Debug Metadata:\n");
    test_callback_contract_binding();
    test_debug_metadata_is_not_executable();
    test_scalar_codegen_uses_typed_opcodes();
    test_function_result_signatures();
    test_declared_function_parameter_tags();
    test_empty_struct_list_result_keeps_element_tag();

    fprintf(stderr, "\nInteger Arithmetic:\n");
    test_return_int();
    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();
    test_modulo();
    test_nested_arithmetic();
    test_negation();

    fprintf(stderr, "\nVariables (let/set):\n");
    test_let_simple();
    test_set_mutable();
    test_multiple_variables();
    test_set_with_expression();

    fprintf(stderr, "\nComparisons:\n");
    test_comparisons();

    fprintf(stderr, "\nBoolean/Logic:\n");
    test_bool_literal();
    test_logic_and();
    test_logic_or();
    test_logic_not();

    fprintf(stderr, "\nControl Flow:\n");
    test_if_then_else();
    test_if_else_branch();
    test_nested_if();
    test_while_loop();
    test_while_break();
    test_while_continue();

    fprintf(stderr, "\nFunctions:\n");
    test_function_call();
    test_function_multiple_calls();
    test_recursion_factorial();
    test_recursion_fibonacci();
    test_function_nested_calls();

    fprintf(stderr, "\nStrings:\n");
    test_string_literal();

    fprintf(stderr, "\nPrint/Assert:\n");
    test_print_int();
    test_assert_true();
    test_assert_false();

    fprintf(stderr, "\nFloat:\n");
    test_float_arithmetic();

    fprintf(stderr, "\nSelfhost-Style Programs:\n");
    test_selfhost_arithmetic();
    test_selfhost_let_set();
    test_selfhost_if_else();
    test_selfhost_while();
    test_selfhost_functions();
    test_selfhost_recursion();

    fprintf(stderr, "\nArray Literals:\n");
    test_array_literal();
    test_array_length();
    test_array_push();
    test_array_set();

    fprintf(stderr, "\nRange:\n");
    test_range_two_args();
    test_for_in_range();

    fprintf(stderr, "\nType Cast Builtins:\n");
    test_int_to_string();

    fprintf(stderr, "\nComplex Types - Structs:\n");
    test_struct_literal();
    test_struct_field_access();

    fprintf(stderr, "\nComplex Types - Tuples:\n");
    test_tuple_literal();
    test_tuple_index();

    fprintf(stderr, "\nComplex Types - Enums:\n");
    test_enum_variant();

    fprintf(stderr, "\nComplex Types - Unions/Match:\n");
    test_union_construct_match();
    test_union_match_err_path();

    fprintf(stderr, "\nComplex Types - Globals:\n");
    test_global_constant();
    test_more_than_128_globals();

    fprintf(stderr, "\nHigher-Order Functions:\n");
    test_function_as_value();
    test_filter_builtin();
    test_map_builtin();

    fprintf(stderr, "\nClosure Captures:\n");
    test_closure_single_capture();
    test_transitive_anonymous_captures();
    test_transitive_named_captures();
    test_closure_multiple_captures();
    test_closure_two_closures();
    test_closure_capture_local_var();

    fprintf(stderr, "\nLexical Blocks:\n");
    test_block_local_shadowing();
    test_unsafe_block_local_shadowing();
    test_nested_closure_keeps_block_capture();

    fprintf(stderr, "\nRound-Trip:\n");
    test_serialize_and_run();

    fprintf(stderr, "\n=== Results: %d passed, %d failed, %d total ===\n",
            tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
