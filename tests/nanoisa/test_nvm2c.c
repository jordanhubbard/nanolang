/*
 * nvm2c must emit structured C11 from NanoISA, not a bytecode blob plus nano_vm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include "assembler.h"
#include "nvm2c.h"
#include "isa.h"
#include "nvm_format.h"
#include "nanoisa.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

static NvmModule *assemble_ok(const char *src, const char *label) {
    AsmResult result;
    memset(&result, 0, sizeof result);
    NvmModule *m = asm_assemble(src, &result);
    if (!m) {
        printf("  FAIL: %s assemble: %s (line %u)\n", label, result.message, result.line);
        g_fail++;
    }
    return m;
}

static int compile_and_run(const char *c_src, int *status_out) {
    char dir[] = "/tmp/nvm2cXXXXXX";
    if (!mkdtemp(dir)) return -1;
    char src_path[128];
    char bin_path[128];
    snprintf(src_path, sizeof src_path, "%s/out.c", dir);
    snprintf(bin_path, sizeof bin_path, "%s/out", dir);

    FILE *f = fopen(src_path, "w");
    if (!f) {
        rmdir(dir);
        return -1;
    }
    fputs(c_src, f);
    fclose(f);

    const char *cc = getenv("CC");
    if (!cc || !cc[0]) cc = "cc";
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "perl -e 'alarm 30; exec @ARGV' %s -std=c11 -Wall -Wextra -Werror -o %s %s",
             cc, bin_path, src_path);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "---- generated C (cc failed) ----\n%s\n----\n", c_src);
        unlink(src_path);
        rmdir(dir);
        return -2;
    }

    snprintf(cmd, sizeof cmd, "perl -e 'alarm 30; exec @ARGV' %s", bin_path);
    rc = system(cmd);
    int status = -1;
    if (WIFEXITED(rc)) status = WEXITSTATUS(rc);
    *status_out = status;

    unlink(src_path);
    unlink(bin_path);
    rmdir(dir);
    return 0;
}

static char *emit_or_fail(NvmModule *m, const char *label);

static void test_add_is_structured_c_and_runs(void) {
    const char *src =
        ".entry 1\n"
        ".function add 2 2 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 40\n"
        "  PUSH_I64 2\n"
        "  CALL add\n"
        "  RET\n"
        ".end\n";

    NvmModule *m = assemble_ok(src, "add fixture");
    CHECK(m != NULL, "add fixture assembles");
    if (!m) return;

    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c emits C for the add fixture");
    if (!c) {
        printf("    nvm2c error: %s\n", err);
        nvm_module_free(m);
        return;
    }

    CHECK(strstr(c, " + ") != NULL, "emitted C contains integer addition");
    CHECK(strstr(c, "nano_vm") == NULL, "emitted C does not name nano_vm");
    CHECK(strstr(c, "nvm_blob") == NULL, "emitted C is not a bytecode blob wrapper");
    CHECK(strstr(c, "unsigned char") == NULL, "emitted C has no bytecode byte array");
    CHECK(strstr(c, "nl_add") != NULL, "callee is a C function");
    CHECK(strstr(c, "nl_main") != NULL, "entry is a C function");

    int status = -1;
    int run = compile_and_run(c, &status);
    CHECK(run == 0, "generated C compiles and runs under cc -std=c11");
    CHECK(status == 42, "native binary exits 42 (40+2) without a VM process");

    free(c);
    nvm_module_free(m);
}

static void test_store_load_local(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 7\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "locals fixture");
    CHECK(m != NULL, "locals fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c emits C for locals");
    if (!c) {
        printf("    nvm2c error: %s\n", err);
        nvm_module_free(m);
        return;
    }
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "locals C compiles");
    CHECK(status == 8, "STORE/LOAD local then add yields 8");
    free(c);
    nvm_module_free(m);
}

static void test_call_extern_is_refused(void) {
    NvmModule *m = nvm_module_new();
    CHECK(m != NULL, "empty module allocates");
    if (!m) return;
    uint32_t s_main = nvm_add_string(m, "main", 4);
    uint32_t s_libc = nvm_add_string(m, "libc", 4);
    uint32_t s_puts = nvm_add_string(m, "puts", 4);
    uint8_t halt = OP_HALT;
    nvm_append_code(m, &halt, 1);
    NvmFunctionEntry f;
    memset(&f, 0, sizeof f);
    f.name_idx = s_main;
    f.code_length = 1;
    f.result_tag = TAG_INT;
    f.result_count = 1;
    nvm_add_function(m, &f);
    uint8_t ptypes[1] = { TAG_STRING };
    nvm_add_import(m, s_libc, s_puts, 1, TAG_INT, ptypes);
    m->header.entry_point = 0;
    m->header.flags = NVM_FLAG_HAS_MAIN | NVM_FLAG_NEEDS_EXTERN;

    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "a module with imports is refused");
    CHECK(strstr(err, "CALL_EXTERN") != NULL || strstr(err, "import") != NULL,
          "refusal names the FFI/import path");
    free(c);
    nvm_module_free(m);
}

static void test_str_substr_is_refused(void) {
    const char *src =
        ".string s \"hi\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR s\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 1\n"
        "  STR_SUBSTR\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "substr fixture");
    CHECK(m != NULL, "substr fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "STR_SUBSTR stays outside the closed subset");
    CHECK(strstr(err, "STR_SUBSTR") != NULL, "error names STR_SUBSTR");
    free(c);
    nvm_module_free(m);
}

static void test_push_str_len_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "PUSH_STR STR_LEN fixture");
    CHECK(m != NULL, "PUSH_STR STR_LEN fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for PUSH_STR/STR_LEN");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "PUSH_STR C does not name nano_vm");
    CHECK(strstr(c, "nvm_blob") == NULL, "PUSH_STR C is not a bytecode blob wrapper");
    CHECK(strstr(c, "\"hi\"") != NULL, "PUSH_STR becomes a C string literal");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "PUSH_STR/STR_LEN C compiles and runs");
    CHECK(status == 2, "len(\"hi\") exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_str_concat_len_runs_without_nano_vm(void) {
    const char *src =
        ".string a \"a\"\n"
        ".string b \"b\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR a\n"
        "  PUSH_STR b\n"
        "  STR_CONCAT\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "STR_CONCAT fixture");
    CHECK(m != NULL, "STR_CONCAT fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for STR_CONCAT");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "STR_CONCAT C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "STR_CONCAT C compiles and runs");
    CHECK(status == 2, "len(\"a\"+\"b\") exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_greeting_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function greeting 0 0 0 string 1\n"
        "  PUSH_STR hi\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL greeting\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "greeting fixture");
    CHECK(m != NULL, "greeting fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for greeting");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "greeting C does not name nano_vm");
    CHECK(strstr(c, "nl_greeting") != NULL, "greeting is a C function");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "greeting C compiles and runs");
    CHECK(status == 2, "len(greeting()) exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_glue_runs_without_nano_vm(void) {
    const char *src =
        ".string a \"a\"\n"
        ".string b \"b\"\n"
        ".entry 1\n"
        ".function glue 2 2 0 string 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  STR_CONCAT\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR a\n"
        "  PUSH_STR b\n"
        "  CALL glue\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "glue fixture");
    CHECK(m != NULL, "glue fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for glue");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "glue C does not name nano_vm");
    CHECK(strstr(c, "nl_glue") != NULL, "glue is a C function");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "glue C compiles and runs");
    CHECK(status == 2, "len(glue(\"a\",\"b\")) exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_arr_set_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 9\n"
        "  ARR_SET\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "ARR_SET fixture");
    CHECK(m != NULL, "ARR_SET fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "ARR_SET stays outside the closed subset");
    CHECK(strstr(err, "ARR_SET") != NULL, "error names ARR_SET");
    free(c);
    nvm_module_free(m);
}

static void test_len3_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function len3 0 1 0 int 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_I64 2\n"
        "  PUSH_I64 3\n"
        "  ARR_LITERAL 1 3\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL len3\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "len3 fixture");
    CHECK(m != NULL, "len3 fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for len3");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "len3 C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "len3 C compiles and runs");
    CHECK(status == 3, "len3() exits 3 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_first_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function first 0 1 0 int 1\n"
        "  PUSH_I64 7\n"
        "  PUSH_I64 8\n"
        "  PUSH_I64 9\n"
        "  ARR_LITERAL 1 3\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL first\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "first fixture");
    CHECK(m != NULL, "first fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for first");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "first C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "first C compiles and runs");
    CHECK(status == 7, "first() exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_null_module(void) {
    char err[64];
    char *c = nvm2c_emit(NULL, err, sizeof err);
    CHECK(c == NULL, "null module is refused");
    CHECK(err[0] != '\0', "null module sets an error");
}

static char *emit_or_fail(NvmModule *m, const char *label) {
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, label);
    if (!c) {
        printf("    nvm2c error: %s\n", err);
    }
    return c;
}

static void check_aot_c(const char *c) {
    CHECK(strstr(c, "nano_vm") == NULL, "emitted C does not name nano_vm");
    CHECK(strstr(c, "nvm_blob") == NULL, "emitted C is not a bytecode blob wrapper");
    CHECK(strstr(c, "goto ") != NULL, "control uses goto as the translator fallback");
}

/* choose: if (> c 0) return 1 else return 0. Then-arm RET then else is extra
 * bytecode after RET; nvm2c must keep translating the other arm. */
static void test_choose_then_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function choose 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE else\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "else:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 3\n"
        "  CALL choose\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "choose then fixture");
    CHECK(m != NULL, "choose then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for choose (then)");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    check_aot_c(c);
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "choose then C compiles and runs");
    CHECK(status == 1, "choose(3) exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_choose_else_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function choose 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE else\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "else:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 0\n"
        "  CALL choose\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "choose else fixture");
    CHECK(m != NULL, "choose else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for choose (else)");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    check_aot_c(c);
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "choose else C compiles and runs");
    CHECK(status == 0, "choose(0) exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* loop_sum: let mut i,s; while (< i n) { set s (+ s i); set i (+ i 1) }; return s.
 * Backward JMP must be valid C (temps declared once, not mid-function). */
static void test_loop_sum_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function loop_sum 1 3 0 int 1\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 1\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 2\n"
        "loop_top:\n"
        "  LOAD_LOCAL 1\n"
        "  LOAD_LOCAL 0\n"
        "  I64_LT_S\n"
        "  JMP_FALSE loop_end\n"
        "  LOAD_LOCAL 2\n"
        "  LOAD_LOCAL 1\n"
        "  I64_ADD\n"
        "  STORE_LOCAL 2\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 1\n"
        "  I64_ADD\n"
        "  STORE_LOCAL 1\n"
        "  JMP loop_top\n"
        "loop_end:\n"
        "  LOAD_LOCAL 2\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 4\n"
        "  CALL loop_sum\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "loop_sum fixture");
    CHECK(m != NULL, "loop_sum fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for loop_sum");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    check_aot_c(c);
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "loop_sum C compiles and runs");
    CHECK(status == 6, "loop_sum(4) exits 6 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_tail_call_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function add 2 2 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 40\n"
        "  PUSH_I64 2\n"
        "  TAIL_CALL add\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "tail-call fixture");
    CHECK(m != NULL, "tail-call fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for TAIL_CALL");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "TAIL_CALL C does not name nano_vm");
    CHECK(strstr(c, "nl_add") != NULL, "TAIL_CALL becomes a C call");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "TAIL_CALL C compiles and runs");
    CHECK(status == 42, "tail-call add(40, 2) exits 42 without a VM process");
    free(c);
    nvm_module_free(m);
}

static char *quote_path(const char *path) {
    size_t len = strlen(path);
    char *quoted = malloc(len + 3);
    if (!quoted) return NULL;
    quoted[0] = '\'';
    memcpy(quoted + 1, path, len);
    quoted[len + 1] = '\'';
    quoted[len + 2] = '\0';
    return quoted;
}

static int capture_cmd(const char *cmd, char *output, size_t output_size, int *status) {
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return -1;
    size_t used = 0;
    while (used + 1 < output_size) {
        size_t n = fread(output + used, 1, output_size - used - 1, pipe);
        if (n == 0) break;
        used += n;
    }
    output[used] = '\0';
    int wait_status = pclose(pipe);
    if (WIFEXITED(wait_status)) {
        *status = WEXITSTATUS(wait_status);
    } else {
        *status = 127;
    }
    return 0;
}

static void test_cli_translates_add_and_does_not_name_nano_vm(const char *cli) {
    const char *src =
        ".entry 1\n"
        ".function add 2 2 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 40\n"
        "  PUSH_I64 2\n"
        "  CALL add\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "cli add fixture");
    CHECK(m != NULL, "cli add fixture assembles");
    if (!m) return;

    NanoisaErr err;
    const char *nvm_path = "/tmp/nanolang_nvm2c_cli_add.nvm";
    const char *c_path = "/tmp/nanolang_nvm2c_cli_add.c";
    CHECK(nanoisa_save_file(m, nvm_path, &err) == NANOISA_OK, "cli fixture saves");
    nvm_module_free(m);

    char *qcli = quote_path(cli);
    char cmd[512];
    snprintf(cmd, sizeof cmd, "%s '%s' -o '%s'", qcli, nvm_path, c_path);
    free(qcli);
    char out[256];
    int status = -1;
    CHECK(capture_cmd(cmd, out, sizeof out, &status) == 0, "nvm2c CLI runs");
    CHECK(status == 0, "nvm2c CLI exits 0 on the closed subset");

    FILE *f = fopen(c_path, "r");
    CHECK(f != NULL, "nvm2c wrote C");
    if (!f) return;
    char *c = malloc(65536);
    CHECK(c != NULL, "C buffer allocates");
    if (!c) {
        fclose(f);
        return;
    }
    size_t n = fread(c, 1, 65535, f);
    c[n] = '\0';
    fclose(f);

    CHECK(strstr(c, "nano_vm") == NULL, "CLI C does not name nano_vm");
    CHECK(strstr(c, "nvm_blob") == NULL, "CLI C is not a bytecode wrapper");
    CHECK(strstr(c, " + ") != NULL, "CLI C contains integer addition");

    int run_status = -1;
    CHECK(compile_and_run(c, &run_status) == 0, "CLI C compiles");
    CHECK(run_status == 42, "CLI native process exits 42 without linking nano_vm");
    free(c);
}

static void test_cli_refuses_call_extern(const char *cli) {
    NvmModule *m = nvm_module_new();
    CHECK(m != NULL, "extern CLI module allocates");
    if (!m) return;
    uint32_t s_main = nvm_add_string(m, "main", 4);
    uint32_t s_libc = nvm_add_string(m, "libc", 4);
    uint32_t s_puts = nvm_add_string(m, "puts", 4);
    uint8_t halt = OP_HALT;
    nvm_append_code(m, &halt, 1);
    NvmFunctionEntry f;
    memset(&f, 0, sizeof f);
    f.name_idx = s_main;
    f.code_length = 1;
    f.result_tag = TAG_INT;
    f.result_count = 1;
    nvm_add_function(m, &f);
    uint8_t ptypes[1] = { TAG_STRING };
    nvm_add_import(m, s_libc, s_puts, 1, TAG_INT, ptypes);
    m->header.entry_point = 0;
    m->header.flags = NVM_FLAG_HAS_MAIN | NVM_FLAG_NEEDS_EXTERN;

    NanoisaErr err;
    const char *nvm_path = "/tmp/nanolang_nvm2c_cli_extern.nvm";
    CHECK(nanoisa_save_file(m, nvm_path, &err) == NANOISA_OK, "extern fixture saves");
    nvm_module_free(m);

    char *qcli = quote_path(cli);
    char cmd[512];
    snprintf(cmd, sizeof cmd, "%s '%s' 2>&1", qcli, nvm_path);
    free(qcli);
    char out[1024];
    int status = -1;
    CHECK(capture_cmd(cmd, out, sizeof out, &status) == 0, "nvm2c CLI runs on extern module");
    CHECK(status != 0, "nvm2c CLI refuses CALL_EXTERN / imports");
    CHECK(strstr(out, "CALL_EXTERN") != NULL || strstr(out, "import") != NULL,
          "CLI refusal names the FFI path");
}

int main(int argc, char **argv) {
    printf("\n[nvm2c] structured C11 from NanoISA...\n\n");
    test_add_is_structured_c_and_runs();
    test_store_load_local();
    test_call_extern_is_refused();
    test_str_substr_is_refused();
    test_push_str_len_runs_without_nano_vm();
    test_str_concat_len_runs_without_nano_vm();
    test_greeting_runs_without_nano_vm();
    test_glue_runs_without_nano_vm();
    test_arr_set_is_refused();
    test_len3_runs_without_nano_vm();
    test_first_runs_without_nano_vm();
    test_null_module();
    test_choose_then_runs_without_nano_vm();
    test_choose_else_runs_without_nano_vm();
    test_loop_sum_runs_without_nano_vm();
    test_tail_call_runs_without_nano_vm();
    if (argc >= 2 && argv[1] && argv[1][0]) {
        test_cli_translates_add_and_does_not_name_nano_vm(argv[1]);
        test_cli_refuses_call_extern(argv[1]);
    } else {
        g_fail++;
        printf("  FAIL: nvm2c CLI path is required (bin/nvm2c)\n");
    }
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
