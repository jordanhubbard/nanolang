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

static int compile_and_run_capture(const char *c_src, int *status_out,
                                   char *captured, size_t cap) {
    char dir[] = "/tmp/nvm2cXXXXXX";
    if (!mkdtemp(dir)) return -1;
    char src_path[128];
    char bin_path[128];
    char out_path[128];
    snprintf(src_path, sizeof src_path, "%s/out.c", dir);
    snprintf(bin_path, sizeof bin_path, "%s/out", dir);
    snprintf(out_path, sizeof out_path, "%s/stdout.txt", dir);

    FILE *f = fopen(src_path, "w");
    if (!f) {
        rmdir(dir);
        return -1;
    }
    fputs(c_src, f);
    fclose(f);

    const char *cc = getenv("CC");
    if (!cc || !cc[0]) cc = "cc";
    char cmd[768];
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

    snprintf(cmd, sizeof cmd, "perl -e 'alarm 30; exec @ARGV' %s > %s", bin_path, out_path);
    rc = system(cmd);
    int status = -1;
    if (WIFEXITED(rc)) status = WEXITSTATUS(rc);
    *status_out = status;

    if (captured && cap > 0) {
        captured[0] = '\0';
        FILE *o = fopen(out_path, "rb");
        if (o) {
            size_t n = fread(captured, 1, cap - 1, o);
            captured[n] = '\0';
            fclose(o);
        }
    }

    unlink(src_path);
    unlink(bin_path);
    unlink(out_path);
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

static void test_str_trim_is_refused(void) {
    const char *src =
        ".string s \"hi\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR s\n"
        "  STR_TRIM\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "trim fixture");
    CHECK(m != NULL, "trim fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "STR_TRIM stays outside the closed subset");
    CHECK(strstr(err, "STR_TRIM") != NULL, "error names STR_TRIM");
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

static void test_agg_set_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_I64 2\n"
        "  AGG_PACK 0 0 0 2\n"
        "  PUSH_I64 9\n"
        "  AGG_SET 0\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "AGG_SET fixture");
    CHECK(m != NULL, "AGG_SET fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "AGG_SET stays outside the closed subset");
    CHECK(strstr(err, "AGG_SET") != NULL, "error names AGG_SET");
    free(c);
    nvm_module_free(m);
}

static void test_getx_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function getx 0 1 0 int 1\n"
        "  PUSH_I64 3\n"
        "  PUSH_I64 4\n"
        "  AGG_PACK 0 0 0 2\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL getx\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "getx fixture");
    CHECK(m != NULL, "getx fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for getx");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "getx C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "getx C compiles and runs");
    CHECK(status == 3, "getx() exits 3 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_is_pos_then_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function is_pos 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 4\n"
        "  CALL is_pos\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "is_pos then fixture");
    CHECK(m != NULL, "is_pos then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for is_pos then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "is_pos then C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "is_pos then C compiles and runs");
    CHECK(status == 1, "is_pos(4) then-arm exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_is_pos_else_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function is_pos 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 0\n"
        "  CALL is_pos\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "is_pos else fixture");
    CHECK(m != NULL, "is_pos else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for is_pos else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "is_pos else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "is_pos else C compiles and runs");
    CHECK(status == 0, "is_pos(0) else-arm exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_yes_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function yes 0 0 0 bool 1\n"
        "  PUSH_BOOL 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL yes\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "yes fixture");
    CHECK(m != NULL, "yes fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for yes");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "yes C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "yes C compiles and runs");
    CHECK(status == 1, "yes() then-arm exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_no_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function no 0 0 0 bool 1\n"
        "  PUSH_BOOL 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL no\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "no fixture");
    CHECK(m != NULL, "no fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for no");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "no C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "no C compiles and runs");
    CHECK(status == 0, "no() else-arm exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_invert_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function invert 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  BOOL_NOT\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_BOOL 1\n"
        "  CALL invert\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "invert fixture");
    CHECK(m != NULL, "invert fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for invert");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "invert C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "invert C compiles and runs");
    CHECK(status == 0, "invert(true) else-arm exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_both_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function both 2 2 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  BOOL_AND\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_BOOL 1\n"
        "  PUSH_BOOL 1\n"
        "  CALL both\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "both fixture");
    CHECK(m != NULL, "both fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for both");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "both C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "both C compiles and runs");
    CHECK(status == 1, "both(true, true) then-arm exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_either_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function either 2 2 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  BOOL_OR\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_BOOL 0\n"
        "  PUSH_BOOL 0\n"
        "  CALL either\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "either fixture");
    CHECK(m != NULL, "either fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for either");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "either C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "either C compiles and runs");
    CHECK(status == 0, "either(false, false) else-arm exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_pick_then_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function pick 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  JMP L1\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "L1:\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 4\n"
        "  CALL pick\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "pick then fixture");
    CHECK(m != NULL, "pick then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for pick then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "pick then C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "pick then C compiles and runs");
    CHECK(status == 1, "pick(4) exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_pick_else_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function pick 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  JMP L1\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "L1:\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 0\n"
        "  CALL pick\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "pick else fixture");
    CHECK(m != NULL, "pick else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for pick else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "pick else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "pick else C compiles and runs");
    CHECK(status == 0, "pick(0) exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_say_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function say 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  PRINT\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL say\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "say fixture");
    CHECK(m != NULL, "say fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for say");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "say C does not name nano_vm");
    CHECK(strstr(c, "printf") != NULL, "say C prints with printf");
    int status = -1;
    char out[32];
    CHECK(compile_and_run_capture(c, &status, out, sizeof out) == 0,
          "say C compiles and runs");
    CHECK(status == 0, "say exits 0 without a VM process");
    CHECK(strcmp(out, "7") == 0, "say writes 7 without a newline");
    free(c);
    nvm_module_free(m);
}

static void test_shout_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function shout 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  PRINTLN\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL shout\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "shout fixture");
    CHECK(m != NULL, "shout fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for shout");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "shout C does not name nano_vm");
    int status = -1;
    char out[32];
    CHECK(compile_and_run_capture(c, &status, out, sizeof out) == 0,
          "shout C compiles and runs");
    CHECK(status == 0, "shout exits 0 without a VM process");
    CHECK(strcmp(out, "7\n") == 0, "shout writes 7 with a newline");
    free(c);
    nvm_module_free(m);
}

static void test_mutter_runs_without_nano_vm(void) {
    const char *src =
        ".string empty \"\"\n"
        ".entry 1\n"
        ".function mutter 0 0 0 int 1\n"
        "  PUSH_STR empty\n"
        "  PRINT\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL mutter\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "mutter fixture");
    CHECK(m != NULL, "mutter fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for mutter");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "mutter C does not name nano_vm");
    int status = -1;
    char out[32];
    CHECK(compile_and_run_capture(c, &status, out, sizeof out) == 0,
          "mutter C compiles and runs");
    CHECK(status == 0, "mutter exits 0 without a VM process");
    CHECK(strcmp(out, "") == 0, "mutter writes an empty string");
    free(c);
    nvm_module_free(m);
}

static void test_print_array_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  PRINT\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "PRINT array fixture");
    CHECK(m != NULL, "PRINT array fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "PRINT of arrays stays outside the closed subset");
    CHECK(strstr(err, "PRINT") != NULL, "error names PRINT");
    free(c);
    nvm_module_free(m);
}

static void test_prove_then_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function prove 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  ASSERT\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_BOOL 1\n"
        "  CALL prove\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "prove then fixture");
    CHECK(m != NULL, "prove then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for prove then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "prove then C does not name nano_vm");
    CHECK(strstr(c, "abort") != NULL, "prove then C aborts on a false assert");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "prove then C compiles and runs");
    CHECK(status == 0, "prove(true) exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_prove_else_aborts_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function prove 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  ASSERT\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_BOOL 0\n"
        "  CALL prove\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "prove else fixture");
    CHECK(m != NULL, "prove else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for prove else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "prove else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "prove else C compiles and runs");
    CHECK(status != 0, "prove(false) aborts without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_grow_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function grow 0 1 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 2\n"
        "  ARR_PUSH\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL grow\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "grow fixture");
    CHECK(m != NULL, "grow fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for grow");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "grow C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "grow C compiles and runs");
    CHECK(status == 2, "grow() exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_arr_push_string_is_refused(void) {
    const char *src =
        ".string empty \"\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_STR empty\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "ARR_PUSH string fixture");
    CHECK(m != NULL, "ARR_PUSH string fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "ARR_PUSH of a string stays outside the closed subset");
    CHECK(strstr(err, "ARR_PUSH") != NULL, "error names ARR_PUSH");
    free(c);
    nvm_module_free(m);
}

static void test_has_hi_then_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function has_hi 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  STR_CONTAINS\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  CALL has_hi\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "has_hi then fixture");
    CHECK(m != NULL, "has_hi then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for has_hi then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "has_hi then C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "has_hi then C compiles and runs");
    CHECK(status == 1, "has_hi(\"hi\") exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_has_hi_else_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function has_hi 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  STR_CONTAINS\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR no\n"
        "  CALL has_hi\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "has_hi else fixture");
    CHECK(m != NULL, "has_hi else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for has_hi else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "has_hi else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "has_hi else C compiles and runs");
    CHECK(status == 0, "has_hi(\"no\") exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_has_pre_then_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function has_pre 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  STR_STARTS_WITH\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  CALL has_pre\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "has_pre then fixture");
    CHECK(m != NULL, "has_pre then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for has_pre then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "has_pre then C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "has_pre then C compiles and runs");
    CHECK(status == 1, "has_pre(\"hi\") exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_has_pre_else_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function has_pre 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  STR_STARTS_WITH\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR no\n"
        "  CALL has_pre\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "has_pre else fixture");
    CHECK(m != NULL, "has_pre else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for has_pre else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "has_pre else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "has_pre else C compiles and runs");
    CHECK(status == 0, "has_pre(\"no\") exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_has_suf_then_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function has_suf 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  STR_ENDS_WITH\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  CALL has_suf\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "has_suf then fixture");
    CHECK(m != NULL, "has_suf then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for has_suf then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "has_suf then C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "has_suf then C compiles and runs");
    CHECK(status == 1, "has_suf(\"hi\") exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_has_suf_else_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function has_suf 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  STR_ENDS_WITH\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR no\n"
        "  CALL has_suf\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "has_suf else fixture");
    CHECK(m != NULL, "has_suf else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for has_suf else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "has_suf else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "has_suf else C compiles and runs");
    CHECK(status == 0, "has_suf(\"no\") exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_digits_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function digits 1 1 0 string 1\n"
        "  LOAD_LOCAL 0\n"
        "  CAST_STRING\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  CALL digits\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "digits fixture");
    CHECK(m != NULL, "digits fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for digits");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "digits C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "digits C compiles and runs");
    CHECK(status == 1, "digits(7) has length 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_cast_string_array_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  CAST_STRING\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "CAST_STRING array fixture");
    CHECK(m != NULL, "CAST_STRING array fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "CAST_STRING of an array stays outside the closed subset");
    CHECK(strstr(err, "CAST_STRING") != NULL, "error names CAST_STRING");
    free(c);
    nvm_module_free(m);
}

static void test_names_runs_without_nano_vm(void) {
    const char *src =
        ".string a \"a\"\n"
        ".string b \"b\"\n"
        ".entry 1\n"
        ".function names 0 1 0 int 1\n"
        "  PUSH_STR a\n"
        "  ARR_LITERAL 5 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR b\n"
        "  ARR_PUSH\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL names\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "names fixture");
    CHECK(m != NULL, "names fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for names");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "names C does not name nano_vm");
    CHECK(strstr(c, "nsarr_") != NULL, "names C uses string-array helpers");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "names C compiles and runs");
    CHECK(status == 2, "names() exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_head_s_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function head_s 0 1 0 string 1\n"
        "  PUSH_STR hi\n"
        "  ARR_LITERAL 5 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL head_s\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "head_s fixture");
    CHECK(m != NULL, "head_s fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for head_s");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "head_s C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "head_s C compiles and runs");
    CHECK(status == 2, "head_s() length is 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_same_then_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function same 2 2 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  EQ\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_STR hi\n"
        "  CALL same\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "same then fixture");
    CHECK(m != NULL, "same then fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for same then");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "same then C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "same then C compiles and runs");
    CHECK(status == 1, "same(\"hi\", \"hi\") exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_same_else_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function same 2 2 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  EQ\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_STR no\n"
        "  CALL same\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "same else fixture");
    CHECK(m != NULL, "same else fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for same else");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "same else C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "same else C compiles and runs");
    CHECK(status == 0, "same(\"hi\", \"no\") exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_diff_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function diff 2 2 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  NE\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_STR no\n"
        "  CALL diff\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "diff fixture");
    CHECK(m != NULL, "diff fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for diff");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "diff C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "diff C compiles and runs");
    CHECK(status == 1, "diff(\"hi\", \"no\") exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_eq_array_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  EQ\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "EQ array fixture");
    CHECK(m != NULL, "EQ array fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "EQ of arrays stays outside the closed subset");
    CHECK(strstr(err, "EQ") != NULL, "error names EQ");
    free(c);
    nvm_module_free(m);
}

static void test_via_at_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function via_at 0 1 0 int 1\n"
        "  PUSH_I64 7\n"
        "  PUSH_I64 8\n"
        "  ARR_LITERAL 1 2\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_at\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_at fixture");
    CHECK(m != NULL, "via_at fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_at");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_at C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_at C compiles and runs");
    CHECK(status == 7, "via_at() exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_slen_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function slen 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  CALL slen\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "slen fixture");
    CHECK(m != NULL, "slen fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for slen");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "slen C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "slen C compiles and runs");
    CHECK(status == 2, "slen(\"hi\") exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_slice_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string h \"h\"\n"
        ".entry 1\n"
        ".function slice 1 1 0 string 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 1\n"
        "  STR_SUBSTR\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  CALL slice\n"
        "  PUSH_STR h\n"
        "  EQ\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "slice fixture");
    CHECK(m != NULL, "slice fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for slice");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "slice C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "slice C compiles and runs");
    CHECK(status == 1, "slice(\"hi\") equals \"h\" without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_str_substr_array_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 1\n"
        "  STR_SUBSTR\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "STR_SUBSTR array fixture");
    CHECK(m != NULL, "STR_SUBSTR array fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "STR_SUBSTR of an array stays outside the closed subset");
    CHECK(strstr(err, "STR_SUBSTR") != NULL, "error names STR_SUBSTR");
    free(c);
    nvm_module_free(m);
}

static void test_blank_l_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function blank_l 0 1 0 int 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL blank_l\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "blank_l fixture");
    CHECK(m != NULL, "blank_l fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for blank_l");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "blank_l C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "blank_l C compiles and runs");
    CHECK(status == 0, "blank_l() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_grow_l_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function grow_l 0 1 0 int 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 7\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL grow_l\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "grow_l fixture");
    CHECK(m != NULL, "grow_l fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for grow_l");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "grow_l C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "grow_l C compiles and runs");
    CHECK(status == 7, "grow_l() exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_put_l_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function put_l 0 1 0 int 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 7\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 9\n"
        "  ARR_SET\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL put_l\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "put_l fixture");
    CHECK(m != NULL, "put_l fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for put_l");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "put_l C does not name nano_vm");
    CHECK(strstr(c, "narr_set") != NULL, "put_l C mutates the int array in place");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "put_l C compiles and runs");
    CHECK(status == 9, "put_l() exits 9 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_ch_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function ch 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_I64 0\n"
        "  STR_CHAR_AT\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL ch\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "ch fixture");
    CHECK(m != NULL, "ch fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for ch");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "ch C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "ch C compiles and runs");
    CHECK(status == 104, "ch() exits 104 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_ch_oob_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function miss 0 0 0 int 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_I64 9\n"
        "  STR_CHAR_AT\n"
        "  PUSH_I64 0\n"
        "  I64_LT_S\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL miss\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "STR_CHAR_AT oob fixture");
    CHECK(m != NULL, "STR_CHAR_AT oob fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for STR_CHAR_AT oob");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "STR_CHAR_AT oob C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "STR_CHAR_AT oob C compiles and runs");
    CHECK(status == 1, "STR_CHAR_AT out of range is -1");
    free(c);
    nvm_module_free(m);
}

static void test_blank_s_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function blank_s 0 1 0 int 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL blank_s\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "blank_s fixture");
    CHECK(m != NULL, "blank_s fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for blank_s");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "blank_s C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "blank_s C compiles and runs");
    CHECK(status == 0, "blank_s() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_grow_s_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function grow_s 0 1 0 string 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL grow_s\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "grow_s fixture");
    CHECK(m != NULL, "grow_s fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for grow_s");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "grow_s C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "grow_s C compiles and runs");
    CHECK(status == 2, "grow_s() length is 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_put_s_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function put_s 0 1 0 string 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR hi\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_STR no\n"
        "  ARR_SET\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL put_s\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "put_s fixture");
    CHECK(m != NULL, "put_s fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for put_s");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "put_s C does not name nano_vm");
    CHECK(strstr(c, "nsarr_set") != NULL, "put_s C mutates the string array in place");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "put_s C compiles and runs");
    CHECK(status == 2, "put_s() length is 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_get_s_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function get_s 0 1 0 string 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL get_s\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "get_s fixture");
    CHECK(m != NULL, "get_s fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for get_s");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "get_s C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "get_s C compiles and runs");
    CHECK(status == 2, "len(get_s()) exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_blank_t_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function blank_t 0 1 0 int 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL blank_t\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "blank_t fixture");
    CHECK(m != NULL, "blank_t fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for blank_t");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "blank_t C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "blank_t C compiles and runs");
    CHECK(status == 0, "blank_t() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_grow_t_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function grow_t 0 2 0 string 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL grow_t\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "grow_t fixture");
    CHECK(m != NULL, "grow_t fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for grow_t");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "grow_t C does not name nano_vm");
    CHECK(strstr(c, "nrarr_push") != NULL, "grow_t C pushes nrec_t onto nrarr_t");
    CHECK(strstr(c, "nrarr_get") != NULL, "grow_t C loads nrec_t from nrarr_t");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "grow_t C compiles and runs");
    CHECK(status == 2, "grow_t() length is 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_put_t_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".string no \"no\"\n"
        ".entry 1\n"
        ".function put_t 0 2 0 string 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 2\n"
        "  PUSH_STR no\n"
        "  AGG_PACK 0 0 0 2\n"
        "  ARR_SET\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL put_t\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "put_t fixture");
    CHECK(m != NULL, "put_t fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for put_t");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "put_t C does not name nano_vm");
    CHECK(strstr(c, "nrarr_set") != NULL, "put_t C mutates nrec_t slots in place");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "put_t C compiles and runs");
    CHECK(status == 2, "put_t() length is 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_get_v_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function get_v 0 1 0 string 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_I64 2\n"
        "  PUSH_I64 3\n"
        "  AGG_PACK 0 0 0 4\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL get_v\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "get_v fixture");
    CHECK(m != NULL, "get_v fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for get_v");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "get_v C does not name nano_vm");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "get_v C compiles and runs");
    CHECK(status == 2, "len(get_v()) exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_grow_lex_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function grow_lex 0 2 0 string 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_I64 2\n"
        "  PUSH_I64 3\n"
        "  AGG_PACK 0 0 0 4\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL grow_lex\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "grow_lex fixture");
    CHECK(m != NULL, "grow_lex fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for grow_lex");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "grow_lex C does not name nano_vm");
    CHECK(strstr(c, "nrarr_push") != NULL, "grow_lex C pushes nrec_t onto nrarr_t");
    CHECK(strstr(c, "nrarr_get") != NULL, "grow_lex C loads nrec_t from nrarr_t");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "grow_lex C compiles and runs");
    CHECK(status == 2, "grow_lex() length is 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_nested_record_pack_runs_without_nano_vm(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 2 0 int 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_I64 2\n"
        "  AGG_PACK 0 0 0 2\n"
        "  PUSH_I64 3\n"
        "  AGG_PACK 0 0 0 2\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 0\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "nested record fixture");
    CHECK(m != NULL, "nested record fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c emits C for nested records");
    if (!c) {
        fprintf(stderr, "nvm2c nested record: %s\n", err);
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "nested record C does not name nano_vm");
    CHECK(strstr(c, "nrec_store") != NULL, "nested record C stores inner nrec_t");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "nested record C compiles and runs");
    CHECK(status == 2, "nested AGG_GET of inner field 1 exits 2 without a VM process");
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

/* upto: for x in [1, 2, 3] { set s (+ s x) }. Generic LT, not I64_LT_S. */
static void test_upto_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function upto 0 5 0 int 1\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  PUSH_I64 2\n"
        "  PUSH_I64 3\n"
        "  ARR_LITERAL 1 3\n"
        "  STORE_LOCAL 1\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 2\n"
        "  LOAD_LOCAL 1\n"
        "  ARR_LEN\n"
        "  STORE_LOCAL 3\n"
        "loop_top:\n"
        "  LOAD_LOCAL 2\n"
        "  LOAD_LOCAL 3\n"
        "  LT\n"
        "  JMP_FALSE loop_end\n"
        "  LOAD_LOCAL 1\n"
        "  LOAD_LOCAL 2\n"
        "  ARR_GET\n"
        "  STORE_LOCAL 4\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 4\n"
        "  I64_ADD\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 2\n"
        "  PUSH_I64 1\n"
        "  I64_ADD\n"
        "  STORE_LOCAL 2\n"
        "  JMP loop_top\n"
        "loop_end:\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL upto\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "upto fixture");
    CHECK(m != NULL, "upto fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for upto");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    check_aot_c(c);
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "upto C compiles and runs");
    CHECK(status == 6, "upto() exits 6 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_quiet: CALL of an empty void function, then return 0. No POP. */
static void test_via_quiet_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function quiet 0 0 0 void 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL quiet\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_quiet fixture");
    CHECK(m != NULL, "via_quiet fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_quiet");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_quiet C does not name nano_vm");
    CHECK(strstr(c, "void") != NULL, "via_quiet C has a void callee");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_quiet C compiles and runs");
    CHECK(status == 0, "via_quiet() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_o: CALL of a pin record (TAG_STRUCT / nrec_t), then AGG_GET y. */
static void test_via_o_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function origin 0 0 0 struct 1\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 1\n"
        "  AGG_PACK 0 0 0 2\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0 int 1\n"
        "  CALL origin\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_o fixture");
    CHECK(m != NULL, "via_o fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_o");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_o C does not name nano_vm");
    CHECK(strstr(c, "nrec_t") != NULL, "via_o C has a record result");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_o C compiles and runs");
    CHECK(status == 1, "via_o() exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_tok: CALL of a mixed int/string record, then AGG_GET of the string. */
static void test_via_tok_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function make_tok 0 0 0 struct 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0 int 1\n"
        "  CALL make_tok\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 1\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_tok fixture");
    CHECK(m != NULL, "via_tok fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_tok");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_tok C does not name nano_vm");
    CHECK(strstr(c, "nrec_t") != NULL, "via_tok C has a record result");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_tok C compiles and runs");
    CHECK(status == 2, "via_tok() exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_ones: CALL of array<int>, then ARR_LEN. */
static void test_via_ones_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function ones 0 0 0 array 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 2\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0 int 1\n"
        "  CALL ones\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_ones fixture");
    CHECK(m != NULL, "via_ones fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_ones");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_ones C does not name nano_vm");
    CHECK(strstr(c, "narr_t") != NULL, "via_ones C has an array result");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_ones C compiles and runs");
    CHECK(status == 2, "via_ones() exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_new_l: CALL of List<int> (ARR_NEW), then ARR_LEN. */
static void test_via_new_l_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function new_l 0 0 0 array 1\n"
        "  ARR_NEW 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0 int 1\n"
        "  CALL new_l\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_new_l fixture");
    CHECK(m != NULL, "via_new_l fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_new_l");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_new_l C does not name nano_vm");
    CHECK(strstr(c, "narr_t") != NULL, "via_new_l C has an array result");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_new_l C compiles and runs");
    CHECK(status == 0, "via_new_l() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_one_t: CALL of List<Tok> (ARR_PUSH of nrec_t before RET), then ARR_GET. */
static void test_via_one_t_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function one_t 0 1 0 array 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 2 0 int 1\n"
        "  CALL one_t\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_one_t fixture");
    CHECK(m != NULL, "via_one_t fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_one_t");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_one_t C does not name nano_vm");
    CHECK(strstr(c, "nrarr_t") != NULL, "via_one_t C has a record-array result");
    CHECK(strstr(c, "nrarr_get") != NULL, "via_one_t C loads nrec_t from nrarr_t");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_one_t C compiles and runs");
    CHECK(status == 2, "via_one_t() exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_one_lex: CALL of List<LexerToken> (four-field AGG_PACK before RET). */
static void test_via_one_lex_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function one_lex 0 1 0 array 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_I64 2\n"
        "  PUSH_I64 3\n"
        "  AGG_PACK 0 0 0 4\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 2 0 int 1\n"
        "  CALL one_lex\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  STR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_one_lex fixture");
    CHECK(m != NULL, "via_one_lex fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_one_lex");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_one_lex C does not name nano_vm");
    CHECK(strstr(c, "nrarr_t") != NULL, "via_one_lex C has a record-array result");
    CHECK(strstr(c, "nrarr_get") != NULL, "via_one_lex C loads nrec_t from nrarr_t");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_one_lex C compiles and runs");
    CHECK(status == 2, "via_one_lex() exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_az: CALL of in_az (I64_GE_S / I64_LE_S / BOOL_AND). */
static void test_via_az_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function in_az 1 1 0 bool 1\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 65\n"
        "  I64_GE_S\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 90\n"
        "  I64_LE_S\n"
        "  BOOL_AND\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 65\n"
        "  CALL in_az\n"
        "  JMP_FALSE else\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "else:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_az fixture");
    CHECK(m != NULL, "via_az fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_az");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_az C does not name nano_vm");
    CHECK(strstr(c, ">=") != NULL, "via_az C uses >=");
    CHECK(strstr(c, "<=") != NULL, "via_az C uses <=");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_az C compiles and runs");
    CHECK(status == 1, "via_az() exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* tag / via_tag: ENUM_VAL discriminant as i64. */
static void test_via_tag_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function tag 0 0 0 int 1\n"
        "  ENUM_VAL 0 19\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL tag\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_tag fixture");
    CHECK(m != NULL, "via_tag fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_tag");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_tag C does not name nano_vm");
    CHECK(strstr(c, "19LL") != NULL, "via_tag C uses the discriminant 19");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_tag C compiles and runs");
    CHECK(status == 19, "via_tag() exits 19 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* tok_mod / via_tok_mod: imported ENUM_VAL discriminant as i64. */
static void test_via_tok_mod_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function tok_mod 0 0 0 int 1\n"
        "  ENUM_VAL 1 19\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL tok_mod\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_tok_mod fixture");
    CHECK(m != NULL, "via_tok_mod fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_tok_mod");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_tok_mod C does not name nano_vm");
    CHECK(strstr(c, "19LL") != NULL, "via_tok_mod C uses the discriminant 19");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_tok_mod C compiles and runs");
    CHECK(status == 19, "via_tok_mod() exits 19 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_imp_add: CALL/TAIL_CALL of an imported i64 function. */
static void test_via_imp_add_runs_without_nano_vm(void) {
    const char *src =
        ".entry 1\n"
        ".function imp_add 2 2 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 40\n"
        "  PUSH_I64 2\n"
        "  TAIL_CALL imp_add\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_imp_add fixture");
    CHECK(m != NULL, "via_imp_add fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_imp_add");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_imp_add C does not name nano_vm");
    CHECK(strstr(c, "nl_imp_add") != NULL, "via_imp_add C calls the imported function");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_imp_add C compiles and runs");
    CHECK(status == 42, "via_imp_add() exits 42 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_raw: unsafe { return 7 } is the same bytecode as return 7. */
static void test_via_raw_runs_without_nano_vm(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_raw fixture");
    CHECK(m != NULL, "via_raw fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_raw");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_raw C does not name nano_vm");
    CHECK(strstr(c, "7LL") != NULL, "via_raw C uses 7");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_raw C compiles and runs");
    CHECK(status == 7, "via_raw() exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_parse_n_runs_without_nano_vm(void) {
    const char *src =
        ".string seven \"7\"\n"
        ".entry 1\n"
        ".function parse_n 0 0 0 int 1\n"
        "  PUSH_STR seven\n"
        "  CAST_INT\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL parse_n\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_parse_n fixture");
    CHECK(m != NULL, "via_parse_n fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_parse_n");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_parse_n C does not name nano_vm");
    CHECK(strstr(c, "nstr_to_i64") != NULL, "via_parse_n C parses the string");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_parse_n C compiles and runs");
    CHECK(status == 7, "via_parse_n() exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_nest: nested AGG_PACK (Loc string+int inside Nest), then GET inner.line. */
static void test_via_nest_runs_without_nano_vm(void) {
    const char *src =
        ".string f \"f\"\n"
        ".entry 2\n"
        ".function nest_d 0 0 0 struct 1\n"
        "  PUSH_STR f\n"
        "  PUSH_I64 7\n"
        "  AGG_PACK 0 0 0 2\n"
        "  PUSH_I64 1\n"
        "  AGG_PACK 0 0 0 2\n"
        "  RET\n"
        ".end\n"
        ".function via_nest 0 2 0 int 1\n"
        "  CALL nest_d\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 0\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_nest\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_nest fixture");
    CHECK(m != NULL, "via_nest fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_nest");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_nest C does not name nano_vm");
    CHECK(strstr(c, "nrec_store") != NULL, "via_nest C stores the inner Loc");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_nest C compiles and runs");
    CHECK(status == 7, "via_nest() exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_g_len: LOAD_GLOBAL of an empty string array stored by void __init__. */
static void test_via_g_len_runs_without_nano_vm(void) {
    const char *src =
        ".entry 2\n"
        ".function __init__ 0 0 0 void 0\n"
        "  ARR_LITERAL 1 0\n"
        "  STORE_GLOBAL 0\n"
        "  RET\n"
        ".end\n"
        ".function via_g_len 0 0 0 int 1\n"
        "  LOAD_GLOBAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_g_len\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_g_len fixture");
    CHECK(m != NULL, "via_g_len fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_g_len");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_g_len C does not name nano_vm");
    CHECK(strstr(c, "nl___init__") != NULL, "via_g_len C calls void __init__");
    CHECK(strstr(c, "LOAD_GLOBAL") == NULL, "via_g_len C does not name LOAD_GLOBAL");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_g_len C compiles and runs");
    CHECK(status == 0, "via_g_len() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_g_set: STORE_GLOBAL of ["a"] then ARR_LEN. */
static void test_via_g_set_runs_without_nano_vm(void) {
    const char *src =
        ".string a \"a\"\n"
        ".entry 2\n"
        ".function __init__ 0 0 0 void 0\n"
        "  ARR_LITERAL 5 0\n"
        "  STORE_GLOBAL 0\n"
        "  RET\n"
        ".end\n"
        ".function via_g_set 0 0 0 int 1\n"
        "  PUSH_STR a\n"
        "  ARR_LITERAL 5 1\n"
        "  STORE_GLOBAL 0\n"
        "  LOAD_GLOBAL 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_g_set\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_g_set fixture");
    CHECK(m != NULL, "via_g_set fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_g_set");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_g_set C does not name nano_vm");
    CHECK(strstr(c, "STORE_GLOBAL") == NULL, "via_g_set C does not name STORE_GLOBAL");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_g_set C compiles and runs");
    CHECK(status == 1, "via_g_set() exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_blank_hm: HM_NEW then map_has "k" is false. */
static void test_via_blank_hm_runs_without_nano_vm(void) {
    const char *src =
        ".string k \"k\"\n"
        ".entry 2\n"
        ".function blank_hm 0 0 0 hashmap 1\n"
        "  HM_NEW 5 1\n"
        "  RET\n"
        ".end\n"
        ".function via_blank_hm 0 1 0 int 1\n"
        "  CALL blank_hm\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR k\n"
        "  HM_HAS\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_blank_hm\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_blank_hm fixture");
    CHECK(m != NULL, "via_blank_hm fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_blank_hm");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_blank_hm C does not name nano_vm");
    CHECK(strstr(c, "nhm_new") != NULL, "via_blank_hm C allocates a hashmap");
    CHECK(strstr(c, "nhm_has") != NULL, "via_blank_hm C tests membership");
    CHECK(strstr(c, "HM_NEW") == NULL, "via_blank_hm C does not name HM_NEW");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_blank_hm C compiles and runs");
    CHECK(status == 0, "via_blank_hm() exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_put_hm: HM_SET then POP, then map_has "k" is true. */
static void test_via_put_hm_runs_without_nano_vm(void) {
    const char *src =
        ".string k \"k\"\n"
        ".string v \"v\"\n"
        ".entry 2\n"
        ".function put_hm 0 1 0 hashmap 1\n"
        "  HM_NEW 5 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR k\n"
        "  PUSH_STR v\n"
        "  HM_SET\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".function via_put_hm 0 1 0 int 1\n"
        "  CALL put_hm\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR k\n"
        "  HM_HAS\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_put_hm\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_put_hm fixture");
    CHECK(m != NULL, "via_put_hm fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_put_hm");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_put_hm C does not name nano_vm");
    CHECK(strstr(c, "nhm_set") != NULL, "via_put_hm C mutates the map in place");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_put_hm C compiles and runs");
    CHECK(status == 1, "via_put_hm() exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

/* via_hm: BagHM packs an HM field then GET of the int field. */
static void test_via_hm_runs_without_nano_vm(void) {
    const char *src =
        ".entry 2\n"
        ".function origin_hm 0 0 0 struct 1\n"
        "  HM_NEW 5 1\n"
        "  PUSH_I64 1\n"
        "  AGG_PACK 0 0 0 2\n"
        "  RET\n"
        ".end\n"
        ".function via_hm 0 1 0 int 1\n"
        "  CALL origin_hm\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_hm\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_hm fixture");
    CHECK(m != NULL, "via_hm fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_hm");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_hm C does not name nano_vm");
    CHECK(strstr(c, "nhm_new") != NULL, "via_hm C packs a hashmap field");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_hm C compiles and runs");
    CHECK(status == 1, "via_hm() exits 1 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_hm_get_is_refused(void) {
    const char *src =
        ".string k \"k\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  HM_NEW 5 1\n"
        "  PUSH_STR k\n"
        "  HM_GET\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "HM_GET fixture");
    CHECK(m != NULL, "HM_GET fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "HM_GET stays outside the Cut A nvm2c subset");
    CHECK(strstr(err, "HM_GET") != NULL, "error names HM_GET");
    free(c);
    nvm_module_free(m);
}

static void test_via_cwd_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"vm_getcwd\" string\n"
        ".entry 1\n"
        ".function via_cwd 0 0 0 string 1\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_cwd\n"
        "  STR_LEN\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_cwd fixture");
    CHECK(m != NULL, "via_cwd fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_cwd");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_cwd C does not name nano_vm");
    CHECK(strstr(c, "nano_cop") == NULL, "via_cwd C is not a co-process client");
    CHECK(strstr(c, "nhost_getcwd") != NULL, "via_cwd C calls the host getcwd ABI");
    CHECK(strstr(c, "CALL_EXTERN") == NULL, "via_cwd C does not name CALL_EXTERN");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_cwd C compiles and runs");
    CHECK(status == 1, "via_cwd() is a non-empty cwd without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_env_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"vm_getenv\" string string\n"
        ".string path \"PATH\"\n"
        ".entry 1\n"
        ".function via_env 0 0 0 string 1\n"
        "  PUSH_STR path\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_env\n"
        "  STR_LEN\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_env fixture");
    CHECK(m != NULL, "via_env fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_env");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_env C does not name nano_vm");
    CHECK(strstr(c, "nhost_getenv") != NULL, "via_env C calls the host getenv ABI");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_env C compiles and runs");
    CHECK(status == 1, "via_env(PATH) is non-empty without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_sys_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"vm_system\" int string\n"
        ".string cmd \"true\"\n"
        ".entry 1\n"
        ".function via_sys 0 0 0 int 1\n"
        "  PUSH_STR cmd\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_sys\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_sys fixture");
    CHECK(m != NULL, "via_sys fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_sys");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_sys C does not name nano_vm");
    CHECK(strstr(c, "nhost_system") != NULL, "via_sys C calls the host system ABI");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_sys C compiles and runs");
    CHECK(status == 0, "via_sys(true) exits 0 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_chstr_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"vm_string_from_char\" string int\n"
        ".string A \"A\"\n"
        ".entry 1\n"
        ".function via_chstr 0 0 0 string 1\n"
        "  PUSH_I64 65\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_chstr\n"
        "  PUSH_STR A\n"
        "  EQ\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_chstr fixture");
    CHECK(m != NULL, "via_chstr fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_chstr");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_chstr C does not name nano_vm");
    CHECK(strstr(c, "nhost_from_char") != NULL, "via_chstr C calls the host char ABI");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_chstr C compiles and runs");
    CHECK(status == 1, "via_chstr(65) equals A without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_argc_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"get_argc\" int\n"
        ".entry 1\n"
        ".function via_argc 0 0 0 int 1\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_argc\n"
        "  PUSH_I64 0\n"
        "  I64_LT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_argc fixture");
    CHECK(m != NULL, "via_argc fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_argc");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_argc C does not name nano_vm");
    CHECK(strstr(c, "nhost_argc") != NULL, "via_argc C reads host argc");
    CHECK(strstr(c, "int main(int argc, char **argv)") != NULL,
          "via_argc C main takes argc/argv");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_argc C compiles and runs");
    CHECK(status == 0, "via_argc() is non-negative without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_tmp_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"vm_tmp_dir\" string\n"
        ".entry 1\n"
        ".function via_tmp 0 0 0 string 1\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_tmp\n"
        "  STR_LEN\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_tmp fixture");
    CHECK(m != NULL, "via_tmp fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_tmp");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_tmp C does not name nano_vm");
    CHECK(strstr(c, "nhost_tmp_dir") != NULL, "via_tmp C calls the host tmp_dir ABI");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_tmp C compiles and runs");
    CHECK(status == 1, "via_tmp() is a non-empty path without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_via_argv_runs_without_nano_vm(void) {
    const char *src =
        ".import \"\" \"get_argv\" string int\n"
        ".entry 1\n"
        ".function via_argv 0 0 0 string 1\n"
        "  PUSH_I64 0\n"
        "  CALL_EXTERN 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL via_argv\n"
        "  STR_LEN\n"
        "  PUSH_I64 0\n"
        "  I64_GT_S\n"
        "  JMP_FALSE L0\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        "L0:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "via_argv fixture");
    CHECK(m != NULL, "via_argv fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for via_argv");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "nano_vm") == NULL, "via_argv C does not name nano_vm");
    CHECK(strstr(c, "nhost_argv") != NULL, "via_argv C reads host argv");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "via_argv C compiles and runs");
    CHECK(status == 1, "via_argv(0) is non-empty without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_cast_int_array_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  CAST_INT\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "CAST_INT array fixture");
    CHECK(m != NULL, "CAST_INT array fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "CAST_INT of an array stays outside the closed subset");
    CHECK(strstr(err, "CAST_INT") != NULL, "error names CAST_INT");
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
    test_str_trim_is_refused();
    test_push_str_len_runs_without_nano_vm();
    test_str_concat_len_runs_without_nano_vm();
    test_greeting_runs_without_nano_vm();
    test_glue_runs_without_nano_vm();
    test_len3_runs_without_nano_vm();
    test_first_runs_without_nano_vm();
    test_agg_set_is_refused();
    test_getx_runs_without_nano_vm();
    test_is_pos_then_runs_without_nano_vm();
    test_is_pos_else_runs_without_nano_vm();
    test_yes_runs_without_nano_vm();
    test_no_runs_without_nano_vm();
    test_invert_runs_without_nano_vm();
    test_both_runs_without_nano_vm();
    test_either_runs_without_nano_vm();
    test_pick_then_runs_without_nano_vm();
    test_pick_else_runs_without_nano_vm();
    test_say_runs_without_nano_vm();
    test_shout_runs_without_nano_vm();
    test_mutter_runs_without_nano_vm();
    test_print_array_is_refused();
    test_prove_then_runs_without_nano_vm();
    test_prove_else_aborts_without_nano_vm();
    test_grow_runs_without_nano_vm();
    test_arr_push_string_is_refused();
    test_has_hi_then_runs_without_nano_vm();
    test_has_hi_else_runs_without_nano_vm();
    test_has_pre_then_runs_without_nano_vm();
    test_has_pre_else_runs_without_nano_vm();
    test_has_suf_then_runs_without_nano_vm();
    test_has_suf_else_runs_without_nano_vm();
    test_digits_runs_without_nano_vm();
    test_cast_string_array_is_refused();
    test_names_runs_without_nano_vm();
    test_head_s_runs_without_nano_vm();
    test_same_then_runs_without_nano_vm();
    test_same_else_runs_without_nano_vm();
    test_diff_runs_without_nano_vm();
    test_eq_array_is_refused();
    test_via_at_runs_without_nano_vm();
    test_slen_runs_without_nano_vm();
    test_slice_runs_without_nano_vm();
    test_str_substr_array_is_refused();
    test_blank_l_runs_without_nano_vm();
    test_grow_l_runs_without_nano_vm();
    test_put_l_runs_without_nano_vm();
    test_ch_runs_without_nano_vm();
    test_ch_oob_runs_without_nano_vm();
    test_blank_s_runs_without_nano_vm();
    test_grow_s_runs_without_nano_vm();
    test_put_s_runs_without_nano_vm();
    test_get_s_runs_without_nano_vm();
    test_blank_t_runs_without_nano_vm();
    test_grow_t_runs_without_nano_vm();
    test_put_t_runs_without_nano_vm();
    test_get_v_runs_without_nano_vm();
    test_grow_lex_runs_without_nano_vm();
    test_nested_record_pack_runs_without_nano_vm();
    test_null_module();
    test_choose_then_runs_without_nano_vm();
    test_choose_else_runs_without_nano_vm();
    test_loop_sum_runs_without_nano_vm();
    test_upto_runs_without_nano_vm();
    test_via_quiet_runs_without_nano_vm();
    test_via_o_runs_without_nano_vm();
    test_via_tok_runs_without_nano_vm();
    test_via_ones_runs_without_nano_vm();
    test_via_new_l_runs_without_nano_vm();
    test_via_one_t_runs_without_nano_vm();
    test_via_one_lex_runs_without_nano_vm();
    test_via_az_runs_without_nano_vm();
    test_via_tag_runs_without_nano_vm();
    test_via_tok_mod_runs_without_nano_vm();
    test_via_imp_add_runs_without_nano_vm();
    test_via_raw_runs_without_nano_vm();
    test_via_parse_n_runs_without_nano_vm();
    test_via_nest_runs_without_nano_vm();
    test_via_g_len_runs_without_nano_vm();
    test_via_g_set_runs_without_nano_vm();
    test_via_blank_hm_runs_without_nano_vm();
    test_via_put_hm_runs_without_nano_vm();
    test_via_hm_runs_without_nano_vm();
    test_hm_get_is_refused();
    test_via_cwd_runs_without_nano_vm();
    test_via_env_runs_without_nano_vm();
    test_via_sys_runs_without_nano_vm();
    test_via_chstr_runs_without_nano_vm();
    test_via_argc_runs_without_nano_vm();
    test_via_tmp_runs_without_nano_vm();
    test_via_argv_runs_without_nano_vm();
    test_cast_int_array_is_refused();
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
