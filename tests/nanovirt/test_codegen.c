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
    if (!type_check(program, env)) {
        snprintf(tr.error, sizeof(tr.error), "typecheck failed");
        free_ast(program);
        free_environment(env);
        free_tokens(tokens, token_count);
        return tr;
    }

    CodegenResult cg = codegen_compile(program, env);
    free_ast(program);
    free_environment(env);
    free_tokens(tokens, token_count);

    if (!cg.ok) {
        snprintf(tr.error, sizeof(tr.error), "codegen: %s", cg.error_msg);
        return tr;
    }

    VmState vm;
    vm_init(&vm, cg.module);
    tr.vm_result = vm_execute(&vm);
    tr.result = vm_get_result(&vm);
    if (tr.result.tag == TAG_STRING || tr.result.tag == TAG_ARRAY) {
        vm_retain(tr.result);
    }
    vm_destroy(&vm);

    tr.ok = true;
    tr.module = cg.module;
    return tr;
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
    if (!type_check(program, env)) {
        snprintf(tr.error, sizeof(tr.error), "typecheck failed");
        free_ast(program);
        free_environment(env);
        free_tokens(tokens, token_count);
        return tr;
    }

    CodegenResult cg = codegen_compile(program, env);
    free_ast(program);
    free_environment(env);
    free_tokens(tokens, token_count);

    if (!cg.ok) {
        snprintf(tr.error, sizeof(tr.error), "codegen: %s", cg.error_msg);
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
        vm_retain(tr.result);
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
    ASSERT(type_check(program, env), "typecheck failed");
    CodegenResult cg = codegen_compile(program, env);
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

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    fprintf(stderr, "\n=== NanoVirt Codegen Tests ===\n\n");

    fprintf(stderr, "Integer Arithmetic:\n");
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

    fprintf(stderr, "\nRound-Trip:\n");
    test_serialize_and_run();

    fprintf(stderr, "\n=== Results: %d passed, %d failed, %d total ===\n",
            tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
