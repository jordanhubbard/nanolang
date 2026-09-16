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

static int compile_and_run_with_args(const char *c_src, int *status_out,
                                     const char *args) {
    char dir[] = "/tmp/nvm2cXXXXXX";
    if (!mkdtemp(dir)) return -1;
    char src_path[128], bin_path[128], cmd[512];
    snprintf(src_path, sizeof src_path, "%s/out.c", dir);
    snprintf(bin_path, sizeof bin_path, "%s/out", dir);
    FILE *f = fopen(src_path, "w");
    if (!f) { rmdir(dir); return -1; }
    fputs(c_src, f);
    fclose(f);
    const char *cc = getenv("CC");
    if (!cc || !cc[0]) cc = "cc";
    snprintf(cmd, sizeof cmd,
             "perl -e 'alarm 30; exec @ARGV' %s -std=c11 -Wall -Wextra -Werror -o %s %s",
             cc, bin_path, src_path);
    int rc = system(cmd);
    if (rc != 0) { unlink(src_path); rmdir(dir); return -2; }
    snprintf(cmd, sizeof cmd, "perl -e 'alarm 30; exec @ARGV' %s %s", bin_path, args);
    rc = system(cmd);
    *status_out = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
    unlink(src_path);
    unlink(bin_path);
    rmdir(dir);
    return 0;
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

static void test_record_result_crosses_direct_call(void) {
    const char *src =
        ".string seven \"seven\"\n"
        ".entry 1\n"
        ".function pair 0 0 0 struct 1\n"
        "  PUSH_I64 7\n"
        "  PUSH_STR seven\n"
        "  AGG_PACK 0 0 0 2\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL pair\n"
        "  AGG_GET 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "record result fixture");
    CHECK(m != NULL, "record result fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c emits a record-valued direct call");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0, "record-valued generated C compiles and runs");
        CHECK(status == 7, "record result preserves its integer field");
        free(c);
    } else {
        printf("    nvm2c error: %s\n", err);
    }
    nvm_module_free(m);
}

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
    CHECK(c != NULL, "ARR_SET emits structured C");
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

static void test_array_get_preserves_void_and_narrows_index(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 23\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 4294967296\n"
        "  ARR_GET\n"
        "  CAST_INT\n"
        "  PUSH_I64 7\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 -1\n"
        "  ARR_GET\n"
        "  TYPE_CHECK 0\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "ARR_GET void and narrowing fixture");
    CHECK(m != NULL, "ARR_GET void and narrowing fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits tagged array reads");
    if (c) {
        int status = -1;
        CHECK(strstr(c, "uint32_t narrowed = (uint32_t)idx") != NULL,
              "array indexes narrow like NanoVM");
        CHECK(strstr(c, "return (nmap_value){0}") != NULL,
              "out-of-bounds array reads preserve void");
        CHECK(compile_and_run(c, &status) == 0, "tagged array read C compiles and runs");
        CHECK(status == 24, "narrowed in-bounds and void type-check results are preserved");
        free(c);
    }
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
        "  ARR_LEN\n"
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

static void test_grow_t_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function grow_t 0 2 0 string 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 7\n"
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
    CHECK(strstr(c, "nrec_t data[NVM2C_RECORD_ARRAY_CAP]") != NULL,
          "grow_t C stores bounded record elements by value");
    CHECK(strstr(c, "nrarr_push") != NULL, "grow_t C uses record-array helpers");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "grow_t C compiles and runs");
    CHECK(status == 2, "len(grow_t()) exits 2 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_one_t_result_runs_without_nano_vm(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function one_t 0 1 0 array 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 7\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0 int 1\n"
        "  CALL one_t\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  AGG_GET 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "one_t result fixture");
    CHECK(m != NULL, "one_t result fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for one_t result");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "static nrarr_t nl_one_t") != NULL,
          "one_t result is emitted as a record array");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "one_t result C compiles and runs");
    CHECK(status == 7, "via one_t result exits 7 without a VM process");
    free(c);
    nvm_module_free(m);
}

static void test_string_array_push_tags_parameter_and_preserves_alias(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 1\n"
        ".function append 2 2 0 array 1\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  ARR_PUSH\n"
        "  LOAD_LOCAL 1\n"
        "  ARR_PUSH\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  ARR_NEW 5\n"
        "  PUSH_STR hi\n"
        "  CALL append\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "string array parameter fixture");
    CHECK(m != NULL, "string array parameter fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c tags a projected string parameter");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "static nsarr_t nl_append(nsarr_t a0") != NULL,
          "string-array parameter retains its native kind");
    CHECK(strstr(c, "const char * a1") != NULL,
          "array write tags the scalar parameter as a string");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "aliased string-array C compiles and runs");
    CHECK(status == 2, "both writes through the array alias are preserved");
    free(c);
    nvm_module_free(m);
}

static void test_string_array_push_rejects_integer_payload(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  ARR_NEW 5\n"
        "  PUSH_I64 7\n"
        "  ARR_PUSH\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "invalid string array payload fixture");
    CHECK(m != NULL, "invalid string array payload fixture assembles");
    if (!m) return;
    char err[256] = {0};
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "nvm2c rejects an integer payload for a native string array");
    CHECK(strstr(err, "requires a string value") != NULL,
          "string-array payload refusal names the required runtime kind");
    free(c);
    nvm_module_free(m);
}

static void test_nested_record_pack_is_refused(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 1\n"
        "  PUSH_I64 2\n"
        "  AGG_PACK 0 0 0 2\n"
        "  PUSH_I64 3\n"
        "  AGG_PACK 0 0 0 2\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "nested record fixture");
    CHECK(m != NULL, "nested record fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "nested records stay outside the closed subset");
    CHECK(strstr(err, "conflicting") != NULL || strstr(err, "record") != NULL,
          "error explains the unsupported nested record shape");
    free(c);
    nvm_module_free(m);
}

static void test_unsupported_classifier_instructions(void) {
    const uint8_t opcodes[] = {OP_HM_KEYS, OP_HM_VALUES, OP_PUSH_F64,
        OP_PUSH_VOID, OP_LOAD_GLOBAL, OP_STORE_GLOBAL, OP_CAST_FLOAT,
        OP_STR_TRIM, OP_CALL_INDIRECT, OP_ROT3};
    for (size_t i = 0; i < sizeof opcodes / sizeof opcodes[0]; ++i) {
        NvmModule *m = assemble_ok(".entry main\n.function main 0 0 0 int 1\n"
            "NOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\n"
            "PUSH_I64 0\nRET\n.end\n", "unsupported classifier instruction");
        if (!m) continue;
        DecodedInstruction instruction = {0};
        instruction.opcode = opcodes[i];
        uint32_t written = isa_encode(&instruction, m->code + m->functions[0].code_offset, 16);
        CHECK(written != 0, "I encode the unsupported instruction using ISA metadata");
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        const InstructionInfo *info = isa_get_info(opcodes[i]);
        CHECK(c == NULL && strstr(error, "cannot classify unsupported opcode") &&
              strstr(error, info->name) && strstr(error, "function 0 at offset 0"),
              "I reject unsupported stack effects at their own instruction");
        free(c);
        nvm_module_free(m);
    }
}

static void test_emitted_map_get(void) {
    for (int strings = 0; strings < 2; ++strings) {
        char source[4096];
        snprintf(source, sizeof source,
            ".string key \"key\"\n.string old \"42\"\n.string newer \"changed\"\n.string empty \"\"\n"
            ".entry main\n.function main 0 3 0 int 1\nHM_NEW 5 %d\nSTORE_LOCAL 0\n"
            "LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nDUP\nPOP\nSTORE_LOCAL 1\n"
            "LOAD_LOCAL 1\nTYPE_CHECK 0\nASSERT\n"
            "LOAD_LOCAL 1\nCAST_STRING\nPUSH_STR empty\nEQ\nASSERT\n"
            "LOAD_LOCAL 1\nJMP_FALSE missing_ok\nPUSH_BOOL 0\nASSERT\nmissing_ok:\n"
            "LOAD_LOCAL 1\nLOAD_LOCAL 1\nEQ\nASSERT\n"
            "LOAD_LOCAL 1\n%s\nEQ\nBOOL_NOT\nASSERT\n"
            "LOAD_LOCAL 1\nCALL cast_value\nPUSH_I64 0\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_STR key\n%s\nHM_SET\nPOP\n"
            "LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nSTORE_LOCAL 2\n"
            "LOAD_LOCAL 2\nTYPE_CHECK 0\nBOOL_NOT\nASSERT\n"
            "LOAD_LOCAL 2\nCAST_STRING\nPUSH_STR old\nEQ\nASSERT\n"
            "LOAD_LOCAL 2\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_STR key\n%s\nHM_SET\nPUSH_STR key\nHM_DELETE\nPOP\n"
            "LOAD_LOCAL 2\n%s\nASSERT\n"
            "PUSH_BOOL %d\nJMP_FALSE alternate\nLOAD_LOCAL 2\nJMP joined\n"
            "alternate:\nLOAD_LOCAL 2\nDUP\nSWAP\nPOP\njoined:\nCALL cast_value\n"
            "PUSH_I64 42\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 2\nCALL consume\nPUSH_I64 %d\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n"
            ".function cast_value 1 1 0 int 1\nLOAD_LOCAL 0\nCAST_INT\nRET\n.end\n"
            ".function consume 1 1 0 int 1\nLOAD_LOCAL 0\n%s\nRET\n.end\n",
            strings ? 5 : 1, strings ? "PUSH_STR empty" :
                "LOAD_LOCAL 0\nPUSH_STR key\nPUSH_I64 0\nHM_SET\nPUSH_STR key\nHM_GET",
            strings ? "PUSH_STR old" : "PUSH_I64 42",
            strings ? "PUSH_STR newer" : "PUSH_I64 99",
            strings ? "PUSH_STR old\nEQ" : "CAST_INT\nPUSH_I64 42\nI64_EQ", strings,
            strings ? 2 : 43, strings ? "STR_LEN" : "PUSH_I64 1\nI64_ADD");
        NvmModule *m = assemble_ok(source, "tagged map lookup flow");
        if (!m) continue;
        char *c = emit_or_fail(m, "I retain tagged lookups through locals, branches and calls");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I distinguish missing values and preserve fetched values after replacement and deletion");
            free(c);
        }
        nvm_module_free(m);
    }
    for (int strings = 0; strings < 2; ++strings) {
        char source[512];
        snprintf(source, sizeof source, ".string key \"missing\"\n.entry main\n"
            ".function main 0 0 0 int 1\nHM_NEW 5 %d\nPUSH_STR key\nHM_GET\n%s\nRET\n.end\n",
            strings ? 5 : 1, strings ? "STR_LEN" : "PUSH_I64 1\nI64_ADD");
        NvmModule *m = assemble_ok(source, "missing lookup consumption");
        if (!m) continue;
        char *c = emit_or_fail(m, "I check lookup tags at scalar consumption");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && status != 0,
                  "I reject a missing lookup when an integer or string is required");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *unresolved[] = {
        "PUSH_I64 0\nEQ\n", "PUSH_BOOL 0\nEQ\n", "RET\n",
        "BOOL_NOT\n", "PUSH_BOOL 1\nBOOL_AND\n", "PUSH_BOOL 0\nBOOL_OR\n"
    };
    for (size_t i = 0; i < sizeof unresolved / sizeof unresolved[0]; ++i) {
        char source[512], error[256] = {0};
        snprintf(source, sizeof source, ".string key \"key\"\n.entry main\n"
            ".function main 0 0 0 int 1\nHM_NEW 5 1\nPUSH_STR key\nHM_GET\n%s"
            "POP\nPUSH_I64 0\nRET\n.end\n", unresolved[i]);
        NvmModule *m = assemble_ok(source, "unresolved lookup representation");
        if (!m) continue;
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && error[0], "I reject missing tag information and unsupported tagged return flow");
        free(c); nvm_module_free(m);
    }
}

static void test_emitted_map_flow(void) {
    test_emitted_map_get();
    for (int strings = 0; strings < 2; ++strings) {
        char source[4096];
        snprintf(source, sizeof source, ".string key \"name\"\n.string value \"text\"\n.entry main\n"
            ".function main 0 2 0 int 1\nCALL make\nDUP\nSTORE_LOCAL 0\nPUSH_I64 3\nCALL relay\n"
            "PUSH_BOOL %d\nJMP_FALSE alternate\nPUSH_STR key\n%s\nHM_SET\nJMP joined\n"
            "alternate:\nPUSH_STR key\n%s\nHM_SET\njoined:\nSTORE_LOCAL 1\n"
            "LOAD_LOCAL 0\nPUSH_STR key\nHM_HAS\nASSERT\nLOAD_LOCAL 1\nPUSH_STR key\n%s\nCALL put\nPOP\n"
            "LOAD_LOCAL 0\nHM_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 1\nPUSH_STR key\nHM_DELETE\nPOP\nLOAD_LOCAL 0\nPUSH_STR key\nHM_HAS\nBOOL_NOT\nASSERT\n"
            "LOAD_LOCAL 0\nHM_LEN\nRET\n.end\n"
            ".function make 0 0 0 hashmap 1\nHM_NEW 5 %d\nRET\n.end\n"
            ".function relay 2 2 0 hashmap 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_FALSE recurse\nLOAD_LOCAL 0\nRET\n"
            "recurse:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nTAIL_CALL relay\n.end\n"
            ".function put 3 3 0 hashmap 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nLOAD_LOCAL 2\nHM_SET\nRET\n.end\n",
            strings, strings ? "PUSH_STR value" : "PUSH_I64 42",
            strings ? "PUSH_STR value" : "PUSH_I64 17", strings ? "PUSH_STR value" : "PUSH_I64 99",
            strings ? 5 : 1);
        NvmModule *m = assemble_ok(source, "emitted map flow");
        if (!m) continue;
        char *c = emit_or_fail(m, "I emit map types through forward calls and branches");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I preserve map aliases through mutation, replacement, deletion and tail recursion");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *invalid[] = {"HM_NEW 1 1\nPOP\n", "HM_NEW 5 3\nPOP\n",
        "HM_NEW 5 1\nPUSH_STR key\nPUSH_STR key\nHM_SET\nPOP\n",
        "HM_NEW 5 1\nPUSH_I64 1\nHM_HAS\nPOP\n"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        char source[512], error[256] = {0};
        snprintf(source, sizeof source, ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n%sPUSH_I64 0\nRET\n.end\n", invalid[i]);
        NvmModule *m = assemble_ok(source, "invalid map types");
        if (!m) continue;
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "map"), "I reject unsupported or conflicting map types");
        free(c); nvm_module_free(m);
    }

    const char *stress =
        ".string key \"key\"\n.entry main\n"
        ".function main 0 2 0 int 1\nPUSH_I64 0\nSTORE_LOCAL 0\n"
        "loop:\nLOAD_LOCAL 0\nPUSH_I64 20000\nI64_LT_S\nJMP_FALSE done\n"
        "HM_NEW 5 1\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nLOAD_LOCAL 0\nHM_SET\nPOP\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nHM_GET\nPOP\n"
        "HM_NEW 5 1\nPOP\n"
        "LOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nSTORE_LOCAL 0\nJMP loop\n"
        "done:\nPUSH_I64 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(stress, "bounded map ownership stress");
    if (m) {
        char *c = emit_or_fail(m, "I emit early map reclamation at loop back-edges");
        if (c) {
            const char *needle = "    return result;\n}";
            char *at = strstr(c, needle);
            CHECK(at != NULL, "I locate the generated entry epilogue for the stress probe");
            if (at) {
                const char *probe =
                    "    if (nmap_owned_peak > 8 || nmap_owned_live != 0) abort();\n"
                    "    return result;\n}";
                size_t prefix = (size_t)(at - c), total = prefix + strlen(probe) + 1;
                char *instrumented = malloc(total);
                if (instrumented) {
                    memcpy(instrumented, c, prefix);
                    memcpy(instrumented + prefix, probe, strlen(probe) + 1);
                    int status = -1;
                    CHECK(compile_and_run(instrumented, &status) == 0 && status == 0,
                          "I keep maps and fetched values bounded through repeated construction");
                    free(instrumented);
                } else {
                    CHECK(0, "I allocate the bounded-memory stress source");
                }
            }
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_native_map_runtime(void) {
    test_emitted_map_flow();
    const char *source =
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdio.h>\n#include <assert.h>\n"
        "typedef struct { uint8_t kind; int64_t integer; char *text; } nmap_value;\n"
#include "../../src/nanoisa/nvm2c_map_runtime.inc"
        "int main(int argc, char **argv) {\n"
        "    if (argc > 1) {\n"
        "        if (argv[1][0] == 't') { nmap_t m = nmap_new(1); nmap_set(m, \"key\", (nmap_value){5, 0, \"wrong\"}); }\n"
        "        else if (argv[1][0] == 'k') { (void)nmap_new(3); }\n"
        "        else { nmap_s m = {0}; m.capacity = SIZE_MAX; nmap_grow(&m); }\n"
        "        return 0;\n"
        "    }\n"
        "    nmap_t numbers = nmap_new(1), alias = numbers, strings = nmap_new(5);\n"
        "    char key[64], value[64];\n"
        "    assert(nmap_get(numbers, \"missing\").kind == 0 && !nmap_has(numbers, \"missing\"));\n"
        "    assert(nmap_delete(numbers, \"missing\") == numbers && nmap_len(numbers) == 0);\n"
        "    for (int i = 0; i < 4096; ++i) { snprintf(key, sizeof key, \"key-%d\", i);\n"
        "        assert(nmap_set(numbers, key, (nmap_value){1, -3 * i, NULL}) == alias); }\n"
        "    assert(nmap_len(alias) == 4096 && numbers->capacity > 16);\n"
        "    for (int i = 0; i < 4096; ++i) { snprintf(key, sizeof key, \"key-%d\", i);\n"
        "        nmap_value v = nmap_get(alias, key); assert(v.kind == 1 && v.integer == -3 * i); nmap_release_value(v);\n"
        "        if (i % 2) nmap_delete(alias, key); }\n"
        "    assert(nmap_len(numbers) == 2048);\n"
        "    for (int i = 1; i < 4096; i += 2) { snprintf(key, sizeof key, \"key-%d\", i);\n"
        "        assert(!nmap_has(numbers, key)); nmap_set(numbers, key, (nmap_value){1, INT64_MIN, NULL}); }\n"
        "    nmap_set(numbers, \"\", (nmap_value){1, INT64_MAX, NULL});\n"
        "    assert(nmap_get(numbers, \"\").integer == INT64_MAX && nmap_len(numbers) == 4097);\n"
        "    strcpy(key, \"owned-key\"); strcpy(value, \"before\");\n"
        "    nmap_set(strings, key, (nmap_value){5, 0, value}); key[0] = 'X'; value[0] = 'X';\n"
        "    nmap_value saved = nmap_get(strings, \"owned-key\"); assert(strcmp(saved.text, \"before\") == 0);\n"
        "    nmap_set(strings, \"owned-key\", (nmap_value){5, 0, \"after\"});\n"
        "    nmap_value replaced = nmap_get(strings, \"owned-key\");\n"
        "    assert(strcmp(replaced.text, \"after\") == 0 && strcmp(saved.text, \"before\") == 0 && nmap_len(strings) == 1);\n"
        "    nmap_release_value(replaced); nmap_delete(strings, \"owned-key\");\n"
        "    assert(!nmap_has(strings, \"owned-key\") && !nmap_len(strings));\n"
        "    nmap_set(strings, \"\", (nmap_value){5, 0, \"\"}); nmap_value empty = nmap_get(strings, \"\");\n"
        "    assert(empty.kind == 5 && empty.text[0] == 0 && nmap_get(strings, \"missing\").kind == 0);\n"
        "    nmap_release_value(empty); nmap_destroy(strings);\n"
        "    assert(strcmp(saved.text, \"before\") == 0); nmap_release_value(saved);\n"
        "    nmap_t collisions = nmap_new(1); char keys[32][64]; int found = 0;\n"
        "    for (int i = 0; found < 32; ++i) { snprintf(key, sizeof key, \"collision-%d\", i);\n"
        "        if ((nmap_hash(key) & 255) == 0) { strcpy(keys[found], key);\n"
        "            nmap_set(collisions, key, (nmap_value){1, found, NULL}); ++found; } }\n"
        "    for (int i = 0; i < 32; ++i) assert(nmap_get(collisions, keys[i]).integer == i);\n"
        "    for (int i = 0; i < 32; ++i) { nmap_delete(collisions, keys[i]); assert(!nmap_has(collisions, keys[i])); }\n"
        "    assert(!nmap_len(collisions)); nmap_destroy(collisions); nmap_destroy(numbers); nmap_destroy(NULL); return 0;\n}\n";
    int status = -1;
    CHECK(compile_and_run(source, &status) == 0 && status == 0,
          "I grow and destroy native maps without losing values, collisions or retained lookups");
    const char *invalid[] = {"type", "kind", "overflow"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        status = 0;
        CHECK(compile_and_run_with_args(source, &status, invalid[i]) == 0 && status != 0,
              "I fail closed on invalid map values, types and unrepresentable growth");
    }
}

static void test_null_module(void) {
    test_native_map_runtime();
    test_unsupported_classifier_instructions();
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

static void test_wide_direct_calls(void) {
    size_t cap = 65536;
    char *src = malloc(cap);
    CHECK(src != NULL, "wide direct-call fixture allocates");
    if (!src) return;
    size_t pos = (size_t)snprintf(src, cap,
        ".entry 3\n.function wide_void 256 256 0 void 0\n"
        "  RET\n.end\n"
        ".function wide 256 256 0 int 1\n"
        "  LOAD_LOCAL 255\n  RET\n.end\n"
        ".function ordinary 0 0 0 void 0\n");
    for (int i = 0; i < 256; i++) {
        pos += (size_t)snprintf(src + pos, cap - pos, "  PUSH_I64 %d\n", i);
    }
    pos += (size_t)snprintf(src + pos, cap - pos,
        "  CALL wide_void\n  RET\n.end\n"
        ".function main 0 0 0 int 1\n  CALL ordinary\n");
    for (int i = 0; i < 256; i++) {
        pos += (size_t)snprintf(src + pos, cap - pos, "  PUSH_I64 %d\n", i);
    }
    (void)snprintf(src + pos, cap - pos, "  TAIL_CALL wide\n.end\n");

    NvmModule *m = assemble_ok(src, "wide direct-call fixture");
    free(src);
    CHECK(m != NULL, "wide ordinary/tail direct-call fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits full-arity ordinary and tail calls");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0,
              "full-arity ordinary and tail call C compiles and runs");
        CHECK(status == 255, "full-arity tail call preserves its last argument");
        free(c);
    }
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

static NvmModule *make_local_count_fixture(uint16_t arity, uint16_t local_count) {
    NvmModule *m = nvm_module_new();
    if (!m) return NULL;
    uint32_t name = nvm_add_string(m, "main", 4);
    uint8_t code[] = {
        OP_PUSH_I64, 7, 0, 0, 0, 0, 0, 0, 0,
        OP_STORE_LOCAL, 0xff, 0x03,
        OP_LOAD_LOCAL, 0xff, 0x03,
        OP_RET
    };
    if (local_count < 1024) {
        code[10] = 0;
        code[11] = 0;
        code[13] = 0;
        code[14] = 0;
    }
    nvm_append_code(m, code, sizeof code);
    NvmFunctionEntry fn;
    memset(&fn, 0, sizeof fn);
    fn.name_idx = name;
    fn.arity = arity;
    fn.local_count = local_count;
    fn.code_length = sizeof code;
    fn.result_tag = TAG_INT;
    fn.result_count = 1;
    nvm_add_function(m, &fn);
    m->header.entry_point = 0;
    m->header.flags = NVM_FLAG_HAS_MAIN;
    return m;
}

static void test_1024_locals_compile_and_run(void) {
    NvmModule *m = make_local_count_fixture(0, 1024);
    CHECK(m != NULL, "1024-local fixture allocates");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c accepts NanoVirt's 1024-local limit");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0, "1024-local generated C compiles and runs");
        CHECK(status == 7, "highest valid local preserves its value");
    } else {
        printf("    nvm2c error: %s\n", err);
    }
    free(c);
    nvm_module_free(m);
}

static void test_arity_exceeding_locals_is_refused(void) {
    NvmModule *m = make_local_count_fixture(2, 1);
    CHECK(m != NULL, "malformed arity fixture allocates");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "arity greater than local_count is refused");
    CHECK(strstr(err, "arity exceeds local_count") != NULL,
          "malformed arity error names the violated relation");
    free(c);
    nvm_module_free(m);
}

int main(int argc, char **argv) {
    printf("\n[nvm2c] structured C11 from NanoISA...\n\n");
    test_record_result_crosses_direct_call();
    test_add_is_structured_c_and_runs();
    test_store_load_local();
    test_call_extern_is_refused();
    test_str_trim_is_refused();
    test_push_str_len_runs_without_nano_vm();
    test_str_concat_len_runs_without_nano_vm();
    test_greeting_runs_without_nano_vm();
    test_glue_runs_without_nano_vm();
    test_arr_set_is_refused();
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
    test_digits_runs_without_nano_vm();
    test_cast_string_array_is_refused();
    test_names_runs_without_nano_vm();
    test_head_s_runs_without_nano_vm();
    test_same_then_runs_without_nano_vm();
    test_same_else_runs_without_nano_vm();
    test_diff_runs_without_nano_vm();
    test_eq_array_is_refused();
    test_via_at_runs_without_nano_vm();
    test_array_get_preserves_void_and_narrows_index();
    test_slen_runs_without_nano_vm();
    test_slice_runs_without_nano_vm();
    test_str_substr_array_is_refused();
    test_blank_l_runs_without_nano_vm();
    test_grow_l_runs_without_nano_vm();
    test_ch_runs_without_nano_vm();
    test_ch_oob_runs_without_nano_vm();
    test_blank_s_runs_without_nano_vm();
    test_grow_s_runs_without_nano_vm();
    test_get_s_runs_without_nano_vm();
    test_grow_t_runs_without_nano_vm();
    test_one_t_result_runs_without_nano_vm();
    test_string_array_push_tags_parameter_and_preserves_alias();
    test_string_array_push_rejects_integer_payload();
    test_nested_record_pack_is_refused();
    test_null_module();
    test_choose_then_runs_without_nano_vm();
    test_choose_else_runs_without_nano_vm();
    test_loop_sum_runs_without_nano_vm();
    test_tail_call_runs_without_nano_vm();
    test_1024_locals_compile_and_run();
    test_arity_exceeding_locals_is_refused();
    test_wide_direct_calls();
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
