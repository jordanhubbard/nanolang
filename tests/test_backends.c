/*
 * test_backends.c — unit tests for alternative code generation backends
 *
 * Exercises c_backend.c, llvm_backend.c, riscv_backend.c, ptx_backend.c,
 * and wasm_backend.c by parsing simple nano programs and verifying that
 * each backend produces output without errors.
 */

#include "../src/nanolang.h"
#include "../src/c_backend.h"
#include "../src/llvm_backend.h"
#include "../src/riscv_backend.h"
#include "../src/ptx_backend.h"
#include "../src/wasm_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Required by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

/* ── Test runner ─────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define PASS() do { g_pass++; printf("  %-55s PASS\n", test_name); } while(0)
#define FAIL(msg) do { g_fail++; printf("  %-55s FAIL: %s\n", test_name, (msg)); } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static ASTNode *parse_nano(const char *src) {
    int token_count = 0;
    Token *tokens = tokenize(src, &token_count);
    if (!tokens) return NULL;
    ASTNode *prog = parse_program(tokens, token_count);
    free_tokens(tokens, token_count);
    return prog;
}

static FILE *s_orig_stderr = NULL;
static void suppress_stderr(void) {
    fflush(stderr);
    s_orig_stderr = stderr;
    stderr = fopen("/dev/null", "w");
}
static void restore_stderr(void) {
    if (stderr && stderr != s_orig_stderr) fclose(stderr);
    stderr = s_orig_stderr;
    s_orig_stderr = NULL;
}

/* Common nano programs used across backend tests */
static const char *SRC_SIMPLE =
    "fn add(a: int, b: int) -> int {\n"
    "  return (+ a b)\n"
    "}\n";

static const char *SRC_HELLO =
    "fn main() -> int {\n"
    "  (println \"hello\")\n"
    "  return 0\n"
    "}\n";

static const char *SRC_ARITH =
    "fn square(n: int) -> int {\n"
    "  return (* n n)\n"
    "}\n"
    "fn cube(n: int) -> int {\n"
    "  return (* n (* n n))\n"
    "}\n";

static const char *SRC_IF_ELSE =
    "fn abs_val(x: int) -> int {\n"
    "  if (< x 0) {\n"
    "    return (- 0 x)\n"
    "  } else {\n"
    "    return x\n"
    "  }\n"
    "}\n";

static const char *SRC_LOOP =
    "fn sum_to(n: int) -> int {\n"
    "  let acc: int = 0\n"
    "  let i: int = 0\n"
    "  while (< i n) {\n"
    "    set acc (+ acc i)\n"
    "    set i (+ i 1)\n"
    "  }\n"
    "  return acc\n"
    "}\n";

static const char *SRC_STRUCT =
    "struct Point {\n"
    "  x: int,\n"
    "  y: int\n"
    "}\n"
    "fn make_point(x: int, y: int) -> Point {\n"
    "  return Point { x: x, y: y }\n"
    "}\n";

static const char *SRC_FLOAT =
    "fn circle_area(r: float) -> float {\n"
    "  let pi: float = 3.14159\n"
    "  return (* pi (* r r))\n"
    "}\n";

/* ── c_backend tests ────────────────────────────────────────────────────── */

static void test_c_backend_simple(void) {
    const char *test_name = "c_backend: simple function";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_hello(void) {
    const char *test_name = "c_backend: main with println";
    ASTNode *prog = parse_nano(SRC_HELLO);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_arithmetic(void) {
    const char *test_name = "c_backend: multiple arithmetic functions";
    ASTNode *prog = parse_nano(SRC_ARITH);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_if_else(void) {
    const char *test_name = "c_backend: if/else branch";
    ASTNode *prog = parse_nano(SRC_IF_ELSE);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_while_loop(void) {
    const char *test_name = "c_backend: while loop with set";
    ASTNode *prog = parse_nano(SRC_LOOP);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_struct(void) {
    const char *test_name = "c_backend: struct definition and literal";
    ASTNode *prog = parse_nano(SRC_STRUCT);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_float(void) {
    const char *test_name = "c_backend: float arithmetic";
    ASTNode *prog = parse_nano(SRC_FLOAT);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_no_stdlib(void) {
    const char *test_name = "c_backend: no_stdlib option";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = { .no_stdlib = true, .no_main = true };
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_to_file(void) {
    const char *test_name = "c_backend: emit to file path";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    const char *out_path = "/tmp/test_backends_c.c";
    int rc = c_backend_emit(prog, out_path, "<test>", &opts);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit returned non-zero");
    PASS();
}

static void test_c_backend_enum(void) {
    const char *test_name = "c_backend: enum definition";
    const char *src =
        "enum Color { Red, Green, Blue }\n"
        "fn get_red() -> Color { return Color.Red }\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

static void test_c_backend_let_binding(void) {
    const char *test_name = "c_backend: let bindings and string";
    const char *src =
        "fn greet(name: string) -> string {\n"
        "  let greeting: string = \"Hello\"\n"
        "  return greeting\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");
    CBOptions opts = {0};
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    int rc = c_backend_emit_fp(prog, out, "<test>", &opts);
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "c_backend_emit_fp returned non-zero");
    PASS();
}

/* ── LLVM backend tests ──────────────────────────────────────────────────── */

static void test_llvm_simple(void) {
    const char *test_name = "llvm_backend: simple function";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = llvm_backend_emit(prog, "/tmp/test_backends_llvm_simple.ll",
                                "<test>", false, false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "llvm_backend_emit returned non-zero");
    PASS();
}

static void test_llvm_arithmetic(void) {
    const char *test_name = "llvm_backend: arithmetic functions";
    ASTNode *prog = parse_nano(SRC_ARITH);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = llvm_backend_emit(prog, "/tmp/test_backends_llvm_arith.ll",
                                "<test>", false, false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "llvm_backend_emit returned non-zero");
    PASS();
}

static void test_llvm_if_else(void) {
    const char *test_name = "llvm_backend: if/else branch";
    ASTNode *prog = parse_nano(SRC_IF_ELSE);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = llvm_backend_emit(prog, "/tmp/test_backends_llvm_if.ll",
                                "<test>", false, false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "llvm_backend_emit returned non-zero");
    PASS();
}

static void test_llvm_loop(void) {
    const char *test_name = "llvm_backend: while loop";
    ASTNode *prog = parse_nano(SRC_LOOP);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = llvm_backend_emit(prog, "/tmp/test_backends_llvm_loop.ll",
                                "<test>", false, false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "llvm_backend_emit returned non-zero");
    PASS();
}

static void test_llvm_debug_mode(void) {
    const char *test_name = "llvm_backend: debug mode";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = llvm_backend_emit(prog, "/tmp/test_backends_llvm_debug.ll",
                                "<test>", true, true);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "llvm_backend_emit returned non-zero");
    PASS();
}

/* ── RISC-V backend tests ────────────────────────────────────────────────── */

static void test_riscv_simple(void) {
    const char *test_name = "riscv_backend: simple function";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = riscv_backend_emit_fp(prog, out, "<test>", false, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "riscv_backend_emit_fp returned non-zero");
    PASS();
}

static void test_riscv_arithmetic(void) {
    const char *test_name = "riscv_backend: arithmetic functions";
    ASTNode *prog = parse_nano(SRC_ARITH);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = riscv_backend_emit_fp(prog, out, "<test>", false, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "riscv_backend_emit_fp returned non-zero");
    PASS();
}

static void test_riscv_if_else(void) {
    const char *test_name = "riscv_backend: if/else branch";
    ASTNode *prog = parse_nano(SRC_IF_ELSE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = riscv_backend_emit_fp(prog, out, "<test>", false, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "riscv_backend_emit_fp returned non-zero");
    PASS();
}

static void test_riscv_loop(void) {
    const char *test_name = "riscv_backend: while loop";
    ASTNode *prog = parse_nano(SRC_LOOP);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = riscv_backend_emit_fp(prog, out, "<test>", false, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "riscv_backend_emit_fp returned non-zero");
    PASS();
}

static void test_riscv_to_file(void) {
    const char *test_name = "riscv_backend: emit to file path";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = riscv_backend_emit(prog, "/tmp/test_backends_riscv.s",
                                 "<test>", false, false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "riscv_backend_emit returned non-zero");
    PASS();
}

/* ── PTX backend tests ───────────────────────────────────────────────────── */

static void test_ptx_simple(void) {
    const char *test_name = "ptx_backend: simple function";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = ptx_backend_emit_fp(prog, out, "<test>", false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "ptx_backend_emit_fp returned non-zero");
    PASS();
}

static void test_ptx_arithmetic(void) {
    const char *test_name = "ptx_backend: arithmetic functions";
    ASTNode *prog = parse_nano(SRC_ARITH);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = ptx_backend_emit_fp(prog, out, "<test>", false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "ptx_backend_emit_fp returned non-zero");
    PASS();
}

static void test_ptx_if_else(void) {
    const char *test_name = "ptx_backend: if/else branch";
    ASTNode *prog = parse_nano(SRC_IF_ELSE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = ptx_backend_emit_fp(prog, out, "<test>", false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "ptx_backend_emit_fp returned non-zero");
    PASS();
}

static void test_ptx_loop(void) {
    const char *test_name = "ptx_backend: while loop";
    ASTNode *prog = parse_nano(SRC_LOOP);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = ptx_backend_emit_fp(prog, out, "<test>", false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "ptx_backend_emit_fp returned non-zero");
    PASS();
}

static void test_ptx_to_file(void) {
    const char *test_name = "ptx_backend: emit to file path";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = ptx_backend_emit(prog, "/tmp/test_backends_ptx.ptx",
                               "<test>", false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "ptx_backend_emit returned non-zero");
    PASS();
}

static void test_ptx_verbose(void) {
    const char *test_name = "ptx_backend: verbose mode";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = ptx_backend_emit_fp(prog, out, "<test>", true);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "ptx_backend_emit_fp verbose returned non-zero");
    PASS();
}

/* ── WASM backend tests ──────────────────────────────────────────────────── */

static void test_wasm_simple(void) {
    const char *test_name = "wasm_backend: simple function";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = wasm_backend_emit_fp(prog, out, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit_fp returned non-zero");
    PASS();
}

static void test_wasm_arithmetic(void) {
    const char *test_name = "wasm_backend: arithmetic functions";
    ASTNode *prog = parse_nano(SRC_ARITH);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = wasm_backend_emit_fp(prog, out, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit_fp returned non-zero");
    PASS();
}

static void test_wasm_if_else(void) {
    const char *test_name = "wasm_backend: if/else branch";
    ASTNode *prog = parse_nano(SRC_IF_ELSE);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = wasm_backend_emit_fp(prog, out, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit_fp returned non-zero");
    PASS();
}

static void test_wasm_bool(void) {
    const char *test_name = "wasm_backend: bool return";
    const char *src =
        "fn is_positive(n: int) -> bool {\n"
        "  if (> n 0) { return true } else { return false }\n"
        "}\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = wasm_backend_emit_fp(prog, out, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit_fp returned non-zero");
    PASS();
}

static void test_wasm_to_file(void) {
    const char *test_name = "wasm_backend: emit to file path";
    ASTNode *prog = parse_nano(SRC_SIMPLE);
    ASSERT(prog != NULL, "parse failed");
    suppress_stderr();
    int rc = wasm_backend_emit(prog, "/tmp/test_backends_wasm.wasm",
                                "<test>", NULL, false);
    restore_stderr();
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit returned non-zero");
    PASS();
}

static void test_wasm_multi_func(void) {
    const char *test_name = "wasm_backend: multiple functions";
    const char *src =
        "fn double(n: int) -> int { return (* 2 n) }\n"
        "fn triple(n: int) -> int { return (* 3 n) }\n"
        "fn apply_twice(n: int) -> int { return (double (double n)) }\n";
    ASTNode *prog = parse_nano(src);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = wasm_backend_emit_fp(prog, out, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit_fp returned non-zero");
    PASS();
}

static void test_wasm_float(void) {
    const char *test_name = "wasm_backend: float arithmetic";
    ASTNode *prog = parse_nano(SRC_FLOAT);
    ASSERT(prog != NULL, "parse failed");
    FILE *out = fopen("/dev/null", "w");
    ASSERT(out != NULL, "fopen /dev/null failed");
    suppress_stderr();
    int rc = wasm_backend_emit_fp(prog, out, false);
    restore_stderr();
    fclose(out);
    free_ast(prog);
    ASSERT(rc == 0, "wasm_backend_emit_fp returned non-zero");
    PASS();
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[backends] Alternative code generation backend tests...\n\n");

    printf("C Backend:\n");
    test_c_backend_simple();
    test_c_backend_hello();
    test_c_backend_arithmetic();
    test_c_backend_if_else();
    test_c_backend_while_loop();
    test_c_backend_struct();
    test_c_backend_float();
    test_c_backend_no_stdlib();
    test_c_backend_to_file();
    test_c_backend_enum();
    test_c_backend_let_binding();

    printf("\nLLVM Backend:\n");
    test_llvm_simple();
    test_llvm_arithmetic();
    test_llvm_if_else();
    test_llvm_loop();
    test_llvm_debug_mode();

    printf("\nRISC-V Backend:\n");
    test_riscv_simple();
    test_riscv_arithmetic();
    test_riscv_if_else();
    test_riscv_loop();
    test_riscv_to_file();

    printf("\nPTX Backend:\n");
    test_ptx_simple();
    test_ptx_arithmetic();
    test_ptx_if_else();
    test_ptx_loop();
    test_ptx_to_file();
    test_ptx_verbose();

    printf("\nWASM Backend:\n");
    test_wasm_simple();
    test_wasm_arithmetic();
    test_wasm_if_else();
    test_wasm_bool();
    test_wasm_to_file();
    test_wasm_multi_func();
    test_wasm_float();

    printf("\n");
    if (g_fail == 0) {
        printf("All %d tests passed.\n", g_pass);
        return 0;
    }
    printf("%d/%d tests FAILED.\n", g_fail, g_pass + g_fail);
    return 1;
}
