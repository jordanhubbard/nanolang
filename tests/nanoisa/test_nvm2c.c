/*
 * nvm2c must emit structured C11 from NanoISA, not a bytecode blob plus nano_vm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>

#include "assembler.h"
#include "nvm2c.h"
#include "isa.h"
#include "nvm_format.h"
#include "nanoisa.h"

static int g_pass = 0, g_fail = 0;
#ifdef __linux__
#define TEST_DL_LIB " -ldl"
#else
#define TEST_DL_LIB ""
#endif

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

static int compile_and_run_with_args(const char *c_src, int *status_out, const char *args) {
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
             "perl -e 'alarm 30; exec @ARGV' %s -std=c11 -O0 -fno-optimize-sibling-calls -Wall -Wextra -Werror -o %s %s" TEST_DL_LIB,
             cc, bin_path, src_path);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "---- generated C (cc failed) ----\n%s\n----\n", c_src);
        unlink(src_path);
        rmdir(dir);
        return -2;
    }

    snprintf(cmd, sizeof cmd, "perl -e 'alarm 30; exec @ARGV' %s %s", bin_path, args);
    rc = system(cmd);
    int status = -1;
    if (WIFEXITED(rc)) status = WEXITSTATUS(rc);
    *status_out = status;

    unlink(src_path);
    unlink(bin_path);
    rmdir(dir);
    return 0;
}

static int compile_and_run(const char *c_src, int *status_out) {
    return compile_and_run_with_args(c_src, status_out, "");
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

static void test_artifact_array_import_is_not_a_builtin(void) {
    NvmModule *module = assemble_ok(
        ".import \"\" \"fs_walkdir\" array string\n"
        ".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",
        "filesystem artifact array ABI");
    if (!module) return;
    module->imports[0].kind = NVM_IMPORT_ARTIFACT;
    const char *artifact = "/retained/generation/libstd.dylib";
    module->imports[0].module_name_idx = nvm_add_string(module, artifact, (uint32_t)strlen(artifact));
    char error[256];
    char *source = nvm2c_emit(module, error, sizeof error);
    CHECK(source != NULL, "I emit an exact filesystem artifact adapter");
    CHECK(source && strstr(source, artifact) && strstr(source, "fs_walkdir_release"),
          "I retain the artifact path and explicit ownership release");
    if (source) {
        int status = -1;
        CHECK(compile_and_run(source, &status) == 0 && status == 0,
              "I compile an unused artifact adapter without loading it");
    }
    free(source);
    NvmImportEntry original = module->imports[0];
    for (int bad = 0; bad < 5; ++bad) {
        if (bad == 0) module->imports[0].kind = NVM_IMPORT_FFI;
        if (bad == 1) module->imports[0].return_type = TAG_STRING;
        if (bad == 2) module->imports[0].param_count = 0;
        if (bad == 3) module->imports[0].module_name_idx = nvm_add_string(module, "relative.so", 11);
        if (bad == 4) module->imports[0].module_name_idx = nvm_add_string(module, "/path\0suffix", 12);
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I refuse a noncanonical filesystem artifact contract");
        free(source);
        module->imports[0] = original;
    }
    nvm_module_free(module);
}

static void test_owned_artifact_execution(void) {
    char directory[] = "/tmp/nvm2c-walk-XXXXXX";
    CHECK(mkdtemp(directory) != NULL, "I create a private artifact fixture");
    char path[256], library[256], command[1024];
    snprintf(path, sizeof path, "%s/library.c", directory);
    snprintf(library, sizeof library, "%s/library.so", directory);
    FILE *file = fopen(path, "w");
    CHECK(file != NULL, "I write an owned artifact fixture");
    if (!file) { rmdir(directory); return; }
    fputs("#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n"
          "#include <stdlib.h>\n"
          "typedef struct { int64_t length, capacity; int type; uint8_t width; void *data; } A;\n"
          "#ifndef ABI_VERSION\n#define ABI_VERSION 1\n#endif\n"
          "const uint32_t fs_walkdir__nano_array_abi = ABI_VERSION;\n"
          "static char value[] = \"retained\"; static char *items[] = {value};\n"
          "static A array = {1, 1, 3, sizeof(char *), items};\n"
          "A *fs_walkdir(const char *root) {\n"
          " if (!strcmp(root, \"bad-layout\")) array.width = 1;\n"
          " return &array; }\n"
          "#ifndef OMIT_RELEASE\n"
          "bool fs_walkdir_release(A *a) { if (a != &array) abort();\n"
          " memset(value, 'x', sizeof(value)-1); return true; }\n#endif\n", file);
    CHECK(fclose(file) == 0, "I finish the artifact fixture");
    snprintf(command, sizeof command, "cc -shared -fPIC -o %s %s", library, path);
    CHECK(system(command) == 0, "I build the artifact fixture");
    for (int variant = 0; variant < 5; ++variant) {
        if (variant >= 3) {
            snprintf(command, sizeof command, "cc -shared -fPIC %s -o %s %s",
                     variant == 3 ? "-DABI_VERSION=99" : "-DOMIT_RELEASE", library, path);
            CHECK(system(command) == 0, "I build an incompatible artifact fixture");
        }
        NvmModule *module = assemble_ok(
            ".string root \"valid\"\n.string expected \"retained\"\n"
            ".import \"\" \"fs_walkdir\" array string\n"
            ".entry 0\n.function main 0 0 0 int 1\n"
            "PUSH_STR root\nCALL_EXTERN 0\nPUSH_I64 0\nARR_GET\n"
            "PUSH_STR expected\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
            "owned artifact execution");
        if (!module) continue;
        module->imports[0].kind = NVM_IMPORT_ARTIFACT;
        const char *binding = variant == 2 ? "/nonexistent/nanolang-owned-artifact.so" : library;
        module->imports[0].module_name_idx = nvm_add_string(module, binding, (uint32_t)strlen(binding));
        if (variant == 1) {
            free(module->strings[0]);
            module->strings[0] = strdup("bad-layout");
            module->string_lengths[0] = 10;
        }
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I translate an artifact call and string-array access");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0, "I compile the artifact caller");
            CHECK(variant == 0 ? status == 0 : status != 0,
                  "I preserve copies after release and refuse invalid artifacts");
        } else fprintf(stderr, "%s\n", error);
        free(source);
        nvm_module_free(module);
    }
    unlink(path); unlink(library); rmdir(directory);
}

static void test_real_walk_artifact(void) {
    char directory[] = "/tmp/nvm2c-real-walk-XXXXXX";
    if (!mkdtemp(directory)) { CHECK(0, "I create a real walk fixture"); return; }
    char library[256], command[1024], assembly[2048];
    snprintf(library, sizeof library, "%s/library.so", directory);
    snprintf(command, sizeof command,
             "cc -shared -fPIC -D_GNU_SOURCE -Isrc -o %s modules/std/fs.c "
             "src/runtime/dyn_array.c src/runtime/gc.c src/runtime/gc_struct.c", library);
    int built = system(command);
    CHECK(built == 0, "I build the real filesystem artifact");
    if (built == 0) {
        snprintf(assembly, sizeof assembly,
                 ".string root \"%s\"\n.string expected \"%s\"\n"
                 ".import \"\" \"fs_walkdir\" array string\n"
                 ".entry 0\n.function main 0 0 0 int 1\n"
                 "PUSH_STR root\nCALL_EXTERN 0\nPUSH_I64 0\nARR_GET\n"
                 "PUSH_STR expected\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                 directory, library);
        NvmModule *module = assemble_ok(assembly, "real filesystem artifact");
        if (module) {
            module->imports[0].kind = NVM_IMPORT_ARTIFACT;
            module->imports[0].module_name_idx = nvm_add_string(module, library, (uint32_t)strlen(library));
            char error[256];
            char *source = nvm2c_emit(module, error, sizeof error);
            CHECK(source != NULL, "I emit a real filesystem artifact call");
            if (source) {
                int status = -1;
                CHECK(compile_and_run(source, &status) == 0 && status == 0,
                      "I copy and release a real filesystem result");
            }
            free(source);
            nvm_module_free(module);
        }
        char data_path[256], copy_path[256], empty_path[256], copied_dir[256], absent_path[256];
        snprintf(data_path, sizeof data_path, "%s/data", directory);
        snprintf(copy_path, sizeof copy_path, "%s/copy", directory);
        snprintf(empty_path, sizeof empty_path, "%s/empty", directory);
        snprintf(copied_dir, sizeof copied_dir, "%s/copied", directory);
        snprintf(absent_path, sizeof absent_path, "%s/absent", directory);
        struct ScalarCase { const char *name, *a, *z, *expected, *type; int argc; } cases[] = {
            {"path_normalize", "/foo/./bar/../baz", "", "/foo/baz", "string", 1},
            {"path_canonical", "", "", "", "string", 1},
            {"path_join", "left", "right", "left/right", "string", 2},
            {"path_basename", "left/right", "", "right", "string", 1},
            {"path_dirname", "left/right", "", "left", "string", 1},
            {"path_relpath", "/a/b", "/a", "b", "string", 2},
            {"file_exists", library, "", "1", "bool", 1},
            {"file_delete", absent_path, "", "-1", "int", 1},
            {"file_compare_identity", library, library, "1", "int", 2},
            {"file_compare_destinations", library, library, "1", "int", 2},
            {"file_write", data_path, "first", "0", "int", 2},
            {"file_append", data_path, "second", "0", "int", 2},
            {"file_read", data_path, "", "firstsecond", "string", 1},
            {"file_copy", data_path, copy_path, "0", "int", 2},
            {"file_read", copy_path, "", "firstsecond", "string", 1},
            {"file_delete", data_path, "", "0", "int", 1},
            {"file_delete", copy_path, "", "0", "int", 1},
            {"fs_mkdir_p", empty_path, "", "0", "int", 1},
            {"dir_copy", empty_path, copied_dir, "0", "int", 2},
            {"file_delete", empty_path, "", "0", "int", 1},
            {"file_delete", copied_dir, "", "0", "int", 1},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
            const struct ScalarCase *item = &cases[i];
            snprintf(assembly, sizeof assembly,
                     ".string a \"%s\"\n.string z \"%s\"\n.string expected \"%s\"\n"
                     ".import \"\" \"%s\" %s string %s\n"
                     ".entry 0\n.function main 0 0 0 int 1\n"
                     "PUSH_STR a\n%sCALL_EXTERN 0\n%s%s\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                     item->a, item->z, item->expected, item->name, item->type,
                     item->argc == 2 ? "string" : "", item->argc == 2 ? "PUSH_STR z\n" : "",
                     strcmp(item->type, "string") == 0 ? "PUSH_STR " : "PUSH_I64 ",
                     strcmp(item->type, "string") == 0 ? "expected" : item->expected);
            NvmModule *module = assemble_ok(assembly, item->name);
            if (!module) continue;
            module->imports[0].kind = NVM_IMPORT_ARTIFACT;
            module->imports[0].module_name_idx = nvm_add_string(module, library, (uint32_t)strlen(library));
            char error[256];
            char *source = nvm2c_emit(module, error, sizeof error);
            CHECK(source != NULL, "I emit a scalar filesystem artifact call");
            if (source) {
                int status = -1;
                CHECK(compile_and_run(source, &status) == 0 && status == 0,
                      "I preserve scalar filesystem argument order and return ABI");
            } else fprintf(stderr, "%s: %s\n", item->name, error);
            free(source);
            module->import_param_types[0][item->argc - 1] = TAG_INT;
            source = nvm2c_emit(module, error, sizeof error);
            CHECK(source == NULL, "I refuse a mistyped scalar filesystem argument");
            free(source);
            nvm_module_free(module);
        }
        unlink(data_path); unlink(copy_path); rmdir(empty_path); rmdir(copied_dir);
    }
    unlink(library); rmdir(directory);
}

static void test_builtin_text_reader(void) {
    char directory[] = "/tmp/nvm2c-text-XXXXXX";
    if (!mkdtemp(directory)) { CHECK(0, "I create a text fixture"); return; }
    char path[256], assembly[1024], large[9001];
    snprintf(path, sizeof path, "%s/text", directory);
    memset(large, 'x', sizeof large - 1); large[sizeof large - 1] = 0;
    const char *aliases[] = {"file_read", "vm_file_read", "nl_os_file_read"};
    for (int variant = 0; variant < 9; ++variant) {
        const char *payload = variant == 0 ? "hello\n" : variant == 1 ? large :
                              variant == 2 ? "a\0b" : variant >= 6 ? "streamed" : "";
        const char *expected = variant <= 1 || variant == 6 ? payload : "";
        pid_t writer = -1;
        if (variant < 4 || variant >= 7) {
            FILE *file = fopen(path, "wb");
            CHECK(file != NULL, "I create the text input");
            if (!file) continue;
            size_t size = variant == 2 ? 3 : strlen(payload);
            size_t written = fwrite(payload, 1, size, file);
            int closed = fclose(file);
            CHECK(written == size && closed == 0,
                  "I finish the text input");
        } else unlink(path);
        if (variant == 6) {
            CHECK(mkfifo(path, 0600) == 0, "I create a non-seekable input");
            writer = fork();
            CHECK(writer >= 0, "I start my bounded FIFO writer");
            if (writer == 0) {
                alarm(30);
                FILE *file = fopen(path, "wb");
                if (!file) _exit(1);
                size_t count = fwrite(payload, 1, strlen(payload), file);
                int closed = fclose(file);
                _exit(count == strlen(payload) && closed == 0 ? 0 : 1);
            }
        }
        snprintf(assembly, sizeof assembly,
                 ".string path \"%s\"\n.string expected \"__expected_text_value__\"\n"
                 ".import \"\" \"%s\" string string\n"
                 ".entry 0\n.function main 0 0 0 int 1\n"
                 "PUSH_STR path\nCALL_EXTERN 0\nPUSH_STR expected\nEQ\nASSERT\n"
                 "PUSH_I64 0\nRET\n.end\n", variant == 5 ? directory : path, aliases[variant % 3]);
        NvmModule *module = assemble_ok(assembly, "builtin streaming text reader");
        if (!module) continue;
        free(module->strings[1]);
        module->strings[1] = strdup(expected);
        module->string_lengths[1] = (uint32_t)strlen(expected);
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit the builtin text reader");
        if (source) {
            if (variant >= 7) {
                const char *prefix = variant == 7 ?
                    "#include <stdio.h>\nstatic int failed_close(FILE *f) { fclose(f); return EOF; }\n#define fclose failed_close\n" :
                    "#include <stdio.h>\n#define ferror(f) 1\n";
                char *injected = malloc(strlen(prefix) + strlen(source) + 1);
                if (!injected) abort();
                strcpy(injected, prefix); strcat(injected, source);
                free(source); source = injected;
            }
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I preserve text and reject invalid or unavailable input");
        }
        if (writer > 0) {
            int status = 0;
            CHECK(waitpid(writer, &status, 0) == writer && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                  "I reap my FIFO writer");
        }
        if (variant == 6) unlink(path);
        free(source);
        module->imports[0].return_type = TAG_BOOL;
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I reject a mismatched builtin text result");
        free(source); nvm_module_free(module);
    }
    unlink(path); rmdir(directory);
}

static void test_builtin_text_writer(void) {
    char directory[] = "/tmp/nvm2c-write-XXXXXX";
    if (!mkdtemp(directory)) { CHECK(0, "I create a writer fixture"); return; }
    char path[256], payload_path[256], assembly[1024];
    snprintf(path, sizeof path, "%s/output", directory);
    snprintf(payload_path, sizeof payload_path, "%s/payload", directory);
    const char *aliases[] = {"file_write", "vm_file_write", "nl_os_file_write"};
    for (int variant = 0; variant < 5; ++variant) {
        const char *payload = variant == 1 ? "" : payload_path;
        snprintf(assembly, sizeof assembly,
                 ".string path \"%s\"\n.string text \"%s\"\n"
                 ".import \"\" \"%s\" int string string\n"
                 ".entry 0\n.function main 0 0 0 int 1\n"
                 "PUSH_STR path\nPUSH_STR text\nCALL_EXTERN 0\nPUSH_I64 %d\nEQ\nASSERT\n"
                 "PUSH_I64 0\nRET\n.end\n", variant == 2 ? directory : path,
                 payload, aliases[variant % 3], variant < 2 ? 0 : -1);
        NvmModule *module = assemble_ok(assembly, "builtin text writer");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit the two-argument builtin writer");
        if (source) {
            if (variant >= 3) {
                const char *prefix = variant == 3 ?
                    "#include <stdio.h>\nstatic int failed_close(FILE *f) { fclose(f); return EOF; }\n#define fclose failed_close\n" :
                    "#include <stdio.h>\nstatic int closes;\n"
                    "static size_t short_write(const void *p, size_t s, size_t n, FILE *f) { (void)p; (void)s; (void)n; (void)f; return 0; }\n"
                    "static int counted_close(FILE *f) { ++closes; return fclose(f); }\n"
                    "#define fwrite short_write\n#define fclose counted_close\n#define main generated_main\n";
                const char *suffix = variant == 4 ?
                    "\n#undef main\nint main(int argc, char **argv) { int result = generated_main(argc, argv); return closes == 1 ? result : 77; }\n" : "";
                char *injected = malloc(strlen(prefix) + strlen(source) + strlen(suffix) + 1);
                if (!injected) abort();
                strcpy(injected, prefix); strcat(injected, source); strcat(injected, suffix);
                free(source); source = injected;
            }
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I report open/write/close failures and close after short writes");
            if (variant < 2) {
                FILE *file = fopen(path, "rb");
                CHECK(file != NULL, "I wrote the path argument, not the contents argument");
                if (file) {
                    char text[256] = {0};
                    size_t size = fread(text, 1, sizeof text, file);
                    CHECK(size == strlen(payload) && memcmp(text, payload, size) == 0,
                          "I preserve exact written contents");
                    fclose(file);
                }
            }
        }
        free(source);
        module->import_param_types[0][1] = TAG_INT;
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I validate the second builtin parameter tag");
        free(source); nvm_module_free(module);
    }
    unlink(path); unlink(payload_path); rmdir(directory);
}

static void test_builtin_filesystem_predicates(void) {
    char directory[] = "/tmp/nvm2c-exists-XXXXXX";
    if (!mkdtemp(directory)) { CHECK(0, "I create a predicate fixture"); return; }
    char file[256], missing[256], link[256], broken[256], dirlink[256];
    snprintf(file, sizeof file, "%s/file", directory);
    snprintf(missing, sizeof missing, "%s/missing", directory);
    snprintf(link, sizeof link, "%s/link", directory);
    snprintf(broken, sizeof broken, "%s/broken", directory);
    snprintf(dirlink, sizeof dirlink, "%s/dirlink", directory);
    FILE *stream = fopen(file, "wb");
    CHECK(stream != NULL, "I create a regular file");
    if (stream) fclose(stream);
    CHECK(symlink(file, link) == 0 && symlink(missing, broken) == 0 &&
          symlink(directory, dirlink) == 0, "I create followed and broken links");
    const char *names[] = {"file_exists", "vm_file_exists", "nl_os_file_exists",
                           "dir_exists", "vm_dir_exists", "nl_os_dir_exists"};
    const char *paths[] = {file, directory, missing, link, broken, dirlink, ""};
    for (size_t n = 0; n < sizeof names / sizeof names[0]; ++n) {
        char assembly[8192];
        size_t used = (size_t)snprintf(assembly, sizeof assembly,
            ".import \"\" \"%s\" bool string\n.entry 0\n", names[n]);
        for (size_t p = 0; p < sizeof paths / sizeof paths[0]; ++p)
            used += (size_t)snprintf(assembly + used, sizeof assembly - used,
                                    ".string p%zu \"%s\"\n", p, paths[p]);
        used += (size_t)snprintf(assembly + used, sizeof assembly - used,
                                ".function main 0 0 0 int 1\n");
        for (size_t p = 0; p < sizeof paths / sizeof paths[0]; ++p) {
            int expected = n < 3 ? p == 0 || p == 1 || p == 3 || p == 5 : p == 1 || p == 5;
            used += (size_t)snprintf(assembly + used, sizeof assembly - used,
                "PUSH_STR p%zu\nCALL_EXTERN 0\nPUSH_BOOL %d\nEQ\nASSERT\n", p, expected);
        }
        snprintf(assembly + used, sizeof assembly - used, "PUSH_I64 0\nRET\n.end\n");
        NvmModule *module = assemble_ok(assembly, names[n]);
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit builtin filesystem predicates");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I distinguish directory checks and follow existing symlink targets");
        }
        free(source);
        module->imports[0].return_type = TAG_INT;
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I require the declared boolean predicate result");
        free(source); nvm_module_free(module);
    }
    unlink(file); unlink(link); unlink(broken); unlink(dirlink); rmdir(directory);
}

static void test_builtin_removal_and_rename(void) {
    char directory[] = "/tmp/nvm2c-remove-XXXXXX";
    if (!mkdtemp(directory)) { CHECK(0, "I create a removal fixture"); return; }
    char from[256], to[256], assembly[2048];
    snprintf(from, sizeof from, "%s/from", directory);
    snprintf(to, sizeof to, "%s/to", directory);
    const char *removers[] = {"file_delete", "file_remove", "nl_os_file_delete", "nl_os_file_remove"};
    for (size_t i = 0; i < sizeof removers / sizeof removers[0]; ++i) {
        FILE *file = fopen(from, "wb");
        CHECK(file != NULL, "I create an owned rename source");
        if (file) fclose(file);
        snprintf(assembly, sizeof assembly,
            ".string from \"%s\"\n.string to \"%s\"\n"
            ".import \"\" \"%s\" int string string\n"
            ".import \"\" \"%s\" int string\n.entry 0\n"
            ".function main 0 0 0 int 1\n"
            "PUSH_STR from\nPUSH_STR to\nCALL_EXTERN 0\nPUSH_I64 0\nEQ\nASSERT\n"
            "PUSH_STR from\nCALL_EXTERN 1\nPUSH_I64 -1\nEQ\nASSERT\n"
            "PUSH_STR to\nCALL_EXTERN 1\nPUSH_I64 0\nEQ\nASSERT\n"
            "PUSH_STR from\nPUSH_STR to\nCALL_EXTERN 0\nPUSH_I64 -1\nEQ\nASSERT\n"
            "PUSH_I64 0\nRET\n.end\n", from, to,
            i % 2 ? "nl_os_file_rename" : "file_rename", removers[i]);
        NvmModule *module = assemble_ok(assembly, "private removal and rename");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit builtin rename and removal");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I preserve rename order and report missing removal/rename sources");
        }
        free(source); nvm_module_free(module);
        CHECK(access(from, F_OK) != 0 && access(to, F_OK) != 0, "I removed only my owned fixture paths");
    }
    unlink(from); unlink(to); rmdir(directory);
}

static void test_builtin_identity(void) {
    char directory[] = "/tmp/nvm2c-identity-XXXXXX";
    if (!mkdtemp(directory)) { CHECK(0, "I create an identity fixture"); return; }
    char original[256], other[256], hard[256], soft[256], missing[256], absent[256], broken[256];
    snprintf(original, sizeof original, "%s/original", directory);
    snprintf(other, sizeof other, "%s/other", directory);
    snprintf(hard, sizeof hard, "%s/hard", directory);
    snprintf(soft, sizeof soft, "%s/soft", directory);
    snprintf(missing, sizeof missing, "%s/missing", directory);
    snprintf(absent, sizeof absent, "%s/absent", directory);
    snprintf(broken, sizeof broken, "%s/broken", directory);
    FILE *file = fopen(original, "wb");
    CHECK(file != NULL, "I create the identity source"); if (file) fclose(file);
    file = fopen(other, "wb");
    CHECK(file != NULL, "I create a distinct file"); if (file) fclose(file);
    CHECK(link(original, hard) == 0 && symlink(original, soft) == 0 && symlink(missing, broken) == 0,
          "I create hard, symbolic and broken links");
    struct IdentityCase { const char *a, *z; int identity, destination; } cases[] = {
        {original, original, 1, 1}, {original, hard, 1, 1}, {original, soft, 1, 1},
        {original, other, 0, 0}, {original, missing, 0, 0}, {missing, original, -1, 0},
        {missing, missing, -1, 1}, {missing, absent, -1, 0},
        {original, broken, 0, -1}, {"", original, -1, -1},
    };
    for (int destination = 0; destination < 3; ++destination) {
        char assembly[16384];
        size_t used = (size_t)snprintf(assembly, sizeof assembly,
            ".import \"\" \"%s\" int string string\n.entry 0\n",
            destination ? "file_compare_destinations" : "file_compare_identity");
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
            used += (size_t)snprintf(assembly + used, sizeof assembly - used,
                ".string a%zu \"%s\"\n.string z%zu \"%s\"\n", i, cases[i].a, i, cases[i].z);
        used += (size_t)snprintf(assembly + used, sizeof assembly - used,
                                ".function main 0 0 0 int 1\n");
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
            used += (size_t)snprintf(assembly + used, sizeof assembly - used,
                "PUSH_STR a%zu\nPUSH_STR z%zu\nCALL_EXTERN 0\nPUSH_I64 %d\nEQ\nASSERT\n",
                i, i, destination == 2 && (i == 6 || i == 7) ? -1 :
                destination ? cases[i].destination : cases[i].identity);
        snprintf(assembly + used, sizeof assembly - used, "PUSH_I64 0\nRET\n.end\n");
        NvmModule *module = assemble_ok(assembly, "builtin identity checks");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit builtin identity checks");
        if (source) {
            if (destination == 2) {
                const char *prefix = "#define _POSIX_C_SOURCE 200809L\n#include <unistd.h>\n"
                    "static int failed_cleanup(const char *path) { rmdir(path); return -1; }\n"
                    "#define rmdir failed_cleanup\n";
                char *injected = malloc(strlen(prefix) + strlen(source) + 1);
                if (!injected) abort();
                strcpy(injected, prefix); strcat(injected, source);
                free(source); source = injected;
            }
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I preserve identity, missing-source and absent-destination semantics");
        } else fprintf(stderr, "%s\n", error);
        free(source);
        module->import_param_types[0][1] = TAG_INT;
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I reject an incorrect identity candidate type");
        free(source); nvm_module_free(module);
        CHECK(access(missing, F_OK) != 0 && access(absent, F_OK) != 0,
              "I remove absent-destination probes");
    }
    unlink(original); unlink(other); unlink(hard); unlink(soft); unlink(broken);
    rmdir(missing); rmdir(absent); rmdir(directory);
}

static void test_builtin_normalize(void) {
    char parents[2101], components[1401], cancelled[3501], long_name[5001];
    for (int i = 0; i < 700; ++i) {
        memcpy(parents + i * 3, "../", 3);
        memcpy(components + i * 2, "x/", 2);
    }
    parents[2099] = 0; components[1399] = 0;
    strcpy(cancelled, components);
    for (int i = 0; i < 700; ++i) memcpy(cancelled + 1399 + i * 3, "/..", 3);
    cancelled[3499] = 0;
    memset(long_name, 'x', 5000); long_name[5000] = 0;
    struct NormalizeCase { const char *input, *expected; } cases[] = {
        {"", "."}, {".", "."}, {"/", "/"}, {"///", "/"},
        {"a//b/./c/..", "a/b"}, {"../../a/../b", "../../b"},
        {"/../../a/..", "/"}, {"a/../../b", "../b"},
        {".../..", "."}, {"a/../..", ".."},
        {parents, parents}, {components, components}, {cancelled, "."},
        {long_name, long_name},
    };
    for (int alias = 0; alias < 2; ++alias) {
        char *assembly = malloc(65536);
        if (!assembly) abort();
        size_t used = (size_t)snprintf(assembly, 65536,
            ".import \"\" \"%s\" string string\n.entry 0\n",
            alias ? "nl_os_path_normalize" : "path_normalize");
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
            used += (size_t)snprintf(assembly + used, 65536 - used,
                ".string p%zu \"input_%zu\"\n.string e%zu \"expected_%zu\"\n", i, i, i, i);
        used += (size_t)snprintf(assembly + used, 65536 - used, ".function main 0 0 0 int 1\n");
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
            used += (size_t)snprintf(assembly + used, 65536 - used,
                "PUSH_STR p%zu\nCALL_EXTERN 0\nPUSH_STR e%zu\nEQ\nASSERT\n", i, i);
        snprintf(assembly + used, 65536 - used, "PUSH_I64 0\nRET\n.end\n");
        NvmModule *module = assemble_ok(assembly, "dynamic lexical normalization");
        free(assembly);
        if (!module) continue;
        for (uint32_t s = 0; s < module->string_count; ++s) {
            size_t index = 0;
            const char *value = NULL;
            if (sscanf(module->strings[s], "input_%zu", &index) == 1 && index < sizeof cases / sizeof cases[0])
                value = cases[index].input;
            else if (sscanf(module->strings[s], "expected_%zu", &index) == 1 && index < sizeof cases / sizeof cases[0])
                value = cases[index].expected;
            if (value) {
                free(module->strings[s]); module->strings[s] = strdup(value);
                module->string_lengths[s] = (uint32_t)strlen(value);
            }
        }
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit builtin lexical normalization");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I preserve roots, relative parents and paths beyond old fixed limits");
        } else fprintf(stderr, "%s\n", error);
        free(source);
        module->import_param_types[0][0] = TAG_INT;
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I reject a non-string normalization input");
        free(source); nvm_module_free(module);
    }
}

static void test_builtin_capture(void) {
    NvmModule *module = assemble_ok(
        ".string first \"printf retained && exit 9\"\n.string second \"printf second\"\n"
        ".string expected \"retained\"\n.string next \"second\"\n"
        ".string empty \":\"\n.string large \"printf '%0300000d' 0\"\n"
        ".import \"\" \"nl_exec_capture\" string string\n"
        ".entry 0\n.function main 0 1 0 int 1\n"
        "PUSH_STR first\nCALL_EXTERN 0\nSTORE_LOCAL 0\n"
        "PUSH_STR second\nCALL_EXTERN 0\nPUSH_STR next\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_STR expected\nEQ\nASSERT\n"
        "PUSH_STR empty\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nEQ\nASSERT\n"
        "PUSH_STR large\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 65535\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_STR expected\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
        "bounded owned capture");
    if (!module) return;
    char error[256];
    char *source = nvm2c_emit(module, error, sizeof error);
    CHECK(source != NULL, "I emit the exact capture adapter");
    if (source) {
        int status = -1;
        CHECK(compile_and_run(source, &status) == 0 && status == 0,
              "I retain independent captures and drain output beyond the retained bound");
    } else fprintf(stderr, "%s\n", error);
    free(source);
    module->imports[0].return_type = TAG_INT;
    source = nvm2c_emit(module, error, sizeof error);
    CHECK(source == NULL, "I reject an incompatible capture return signature");
    free(source); nvm_module_free(module);
}

static void test_builtin_from_char(void) {
    NvmModule *extrema = assemble_ok(
        ".entry 0\n.function main 0 0 0 int 1\n"
        "PUSH_I64 -9223372036854775808\nPUSH_I64 9223372036854775807\nI64_ADD\n"
        "PUSH_I64 -1\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n", "signed integer extrema");
    if (extrema) {
        char error[256];
        char *source = nvm2c_emit(extrema, error, sizeof error);
        CHECK(source != NULL, "I emit exact signed extrema");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I preserve signed extrema without out-of-range C literals");
        }
        free(source); nvm_module_free(extrema);
    }
    int64_t codes[] = {0, 65, 127, 128, 255, 256, 257, -1, INT64_MIN, INT64_MAX};
    for (int alias = 0; alias < 2; ++alias) {
        for (size_t i = 0; i < sizeof codes / sizeof codes[0]; ++i) {
            char assembly[1024];
            snprintf(assembly, sizeof assembly,
                ".string expected \"byte-placeholder\"\n"
                ".import \"\" \"%s\" string int\n.entry 0\n.function main 0 1 0 int 1\n"
                "PUSH_I64 %lld\nCALL_EXTERN 0\nSTORE_LOCAL 0\n"
                "PUSH_I64 66\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 1\nEQ\nASSERT\n"
                "LOAD_LOCAL 0\nPUSH_STR expected\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                alias ? "string_from_char" : "vm_string_from_char", (long long)codes[i]);
            NvmModule *module = assemble_ok(assembly, "byte character conversion");
            if (!module) continue;
            char expected[2] = {(char)codes[i], 0};
            free(module->strings[0]); module->strings[0] = strdup(expected);
            module->string_lengths[0] = (uint32_t)strlen(expected);
            char error[256];
            char *source = nvm2c_emit(module, error, sizeof error);
            CHECK(source != NULL, "I emit byte-oriented character conversion");
            if (source) {
                int status = -1;
                CHECK(compile_and_run(source, &status) == 0 && status == 0,
                      "I preserve C-byte boundaries and independent conversion results");
            } else fprintf(stderr, "%s\n", error);
            free(source);
            module->import_param_types[0][0] = TAG_STRING;
            source = nvm2c_emit(module, error, sizeof error);
            CHECK(source == NULL, "I reject a non-integer character input");
            free(source); nvm_module_free(module);
        }
    }
}

static void test_builtin_host_imports(void) {
    struct HostCase { const char *name, *body; uint8_t argc, param, result; } cases[] = {
        {"nl_exec_shell", "PUSH_STR shell_ok\nCALL_EXTERN 0\nPUSH_I64 0\nEQ\nASSERT\n"
                          "PUSH_STR shell_failure\nCALL_EXTERN 0\nPUSH_I64 1792\nEQ\nASSERT\n",
                          1, TAG_STRING, TAG_INT},
        {"get_argc", "CALL_EXTERN 0\nPUSH_I64 1\nEQ\nASSERT\n", 0, TAG_VOID, TAG_INT},
        {"get_argv", "PUSH_I64 -1\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nEQ\nASSERT\n"
                     "PUSH_I64 9223372036854775807\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nEQ\nASSERT\n"
                     "PUSH_I64 0\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nNE\nASSERT\n", 1, TAG_INT, TAG_STRING},
        {"vm_getenv", "PUSH_STR key\nCALL_EXTERN 0\nPUSH_STR expected\nEQ\nASSERT\n"
                      "PUSH_STR absent\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nEQ\nASSERT\n", 1, TAG_STRING, TAG_STRING},
        {"nl_os_getenv", "PUSH_STR key\nCALL_EXTERN 0\nPUSH_STR expected\nEQ\nASSERT\n", 1, TAG_STRING, TAG_STRING},
        {"vm_tmp_dir", "CALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nNE\nASSERT\n", 0, TAG_VOID, TAG_STRING},
        {"vm_getcwd", "CALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nNE\nASSERT\n", 0, TAG_VOID, TAG_STRING},
    };
    const char *key = "NANOLANG_NVM2C_HOST_TEST";
    const char *old = getenv(key);
    char *saved = old ? strdup(old) : NULL;
    CHECK(setenv(key, "retained", 1) == 0, "host fixture environment is set");
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        char source[2048];
        snprintf(source, sizeof source,
                 ".string key \"NANOLANG_NVM2C_HOST_TEST\"\n"
                 ".string shell_ok \":\"\n.string shell_failure \"exit 7\"\n"
                 ".string expected \"retained\"\n.string absent \"\"\n"
                 ".import \"\" \"%s\" %s %s\n"
                 ".entry 0\n.function main 0 0 0 int 1\n%sPUSH_I64 0\nRET\n.end\n",
                 cases[i].name, cases[i].result == TAG_STRING ? "string" : "int",
                 !cases[i].argc ? "" : cases[i].param == TAG_STRING ? "string" : "int", cases[i].body);
        NvmModule *m = assemble_ok(source, cases[i].name);
        if (!m) continue;
        char err[256];
        char *c = nvm2c_emit(m, err, sizeof err);
        CHECK(c != NULL, "exact builtin host import translates");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "native host adapter preserves its tested contract");
            free(c);
        } else printf("    host error: %s\n", err);
        for (int bad = 0; bad < 7; ++bad) {
            NvmImportEntry original = m->imports[0];
            if (bad == 0) m->imports[0].kind = NVM_IMPORT_COPROCESS;
            if (bad == 1) m->imports[0].kind = NVM_IMPORT_ARTIFACT;
            if (bad == 2) m->imports[0].return_type = TAG_BOOL;
            if (bad == 3) m->imports[0].module_name_idx = nvm_add_string(m, "libc", 4);
            if (bad == 4) m->imports[0].param_count = 2;
            if (bad == 5) m->imports[0].module_name_idx = nvm_add_string(m, "\0foreign", 8);
            if (bad == 6) m->imports[0].function_name_idx = nvm_add_string(m, cases[i].name,
                                                                            (uint32_t)strlen(cases[i].name) + 1);
            c = nvm2c_emit(m, err, sizeof err);
            CHECK(c == NULL, "noncanonical host signature or namespace is rejected");
            if (bad == 1)
                CHECK(strstr(err, "artifact-backed") != NULL,
                      "I distinguish artifact binding from a builtin signature mismatch");
            free(c);
            m->imports[0] = original;
        }
        if (strcmp(cases[i].name, "get_argv") == 0) {
            uint32_t original = m->imports[0].function_name_idx;
            m->imports[0].function_name_idx = nvm_add_string(m, "vm_getenv", 9);
            m->import_param_types[0][0] = TAG_STRING;
            c = nvm2c_emit(m, err, sizeof err);
            CHECK(c == NULL && strstr(err, "argument kind"),
                  "host call rejects operand storage incompatible with its signature");
            free(c);
            m->imports[0].function_name_idx = original;
        }
        if (cases[i].argc) {
            m->import_param_types[0][0] = TAG_BOOL;
            c = nvm2c_emit(m, err, sizeof err);
            CHECK(c == NULL, "host parameter requires its exact tag");
            free(c);
        }
        nvm_module_free(m);
    }
    if (saved) { setenv(key, saved, 1); free(saved); }
    else unsetenv(key);

    NvmModule *args = assemble_ok(
        ".import \"\" \"get_argc\" int\n.import \"\" \"get_argv\" string int\n"
        ".string first \"two words\"\n.string second \"--help\"\n"
        ".entry 0\n.function main 0 0 0 int 1\n"
        "CALL_EXTERN 0\nPUSH_I64 3\nEQ\nASSERT\n"
        "PUSH_I64 1\nCALL_EXTERN 1\nPUSH_STR first\nEQ\nASSERT\n"
        "PUSH_I64 2\nCALL_EXTERN 1\nPUSH_STR second\nEQ\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n", "native argument transport");
    if (args) {
        char err[256];
        char *c = nvm2c_emit(args, err, sizeof err);
        CHECK(c != NULL, "combined argument host imports translate");
        if (c) {
            int status = -1;
            CHECK(compile_and_run_with_args(c, &status, "'two words' --help") == 0 && status == 0,
                  "native main preserves argument boundaries and option spelling");
            free(c);
        }
        nvm_module_free(args);
    }
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

static void test_arr_set_runs_natively(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 1\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 9\n"
        "  ARR_SET\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "ARR_SET fixture");
    CHECK(m != NULL, "ARR_SET fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "ARR_SET emits native C");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0, "ARR_SET C compiles and runs");
        CHECK(status == 9, "ARR_SET returns the updated array");
    }
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
    CHECK(strstr(err, "int or string") != NULL, "error names int or string fields");
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

static void test_classifier_branch_stack(void) {
    for (int condition = 0; condition <= 1; condition++) {
        char source[1024];
        snprintf(source, sizeof source,
            ".entry 0\n.function main 0 0 0 int 1\n"
            "PUSH_I64 40\nPUSH_BOOL %d\nJMP_FALSE alternate\n"
            "PUSH_I64 2\nI64_ADD\nRET\n"
            "alternate:\nPUSH_I64 3\nI64_ADD\nRET\n.end\n", condition);
        NvmModule *m = assemble_ok(source, "branch with live operand");
        if (!m) continue;
        char *c = emit_or_fail(m, "branch with live operand");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile a branch carrying a live operand");
            CHECK(status == (condition ? 42 : 43), "I preserve the operand on either branch");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_classifier_unreachable_and_invalid_joins(void) {
    const char *sources[] = {
        ".entry 0\n.function main 0 0 0 int 1\n"
        "PUSH_I64 42\nRET\nPOP\nPOP\n.end\n",
        ".entry 0\n.function main 0 0 0 int 1\n"
        "PUSH_I64 42\nPUSH_BOOL 1\nJMP_FALSE join\nPOP\n"
        "PUSH_I64 42\njoin:\nPOP\nPUSH_I64 0\nRET\n.end\n",
        ".string wrong \"wrong\"\n.entry 0\n.function main 0 0 0 int 1\n"
        "PUSH_I64 42\nPUSH_BOOL 1\nJMP_FALSE join\nPOP\n"
        "PUSH_I64 42\njoin:\nPOP\nPUSH_I64 0\nRET\n.end\n"
    };
    for (int i = 0; i < 3; i++) {
        NvmModule *m = assemble_ok(sources[i], "classifier control flow");
        if (!m) continue;
        if (i != 0) {
            /* I corrupt a verified module to exercise the direct C API. */
            int constants = 0;
            for (uint32_t pc = 0; pc < m->code_size;) {
                DecodedInstruction ins;
                uint32_t n = isa_decode(m->code + pc, m->code_size - pc, &ins);
                if (!n) break;
                if (ins.opcode == OP_PUSH_I64 && ++constants == 2) {
                    memset(m->code + pc, OP_NOP, n);
                    if (i == 2) {
                        m->code[pc] = OP_PUSH_STR;
                        memset(m->code + pc + 1, 0, 4);
                    }
                    break;
                }
                pc += n;
            }
        }
        char err[256];
        char *c = nvm2c_emit(m, err, sizeof err);
        if (i == 0) {
            CHECK(c != NULL, "I ignore unreachable stack operations after return");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 42,
                      "I execute the reachable return");
            }
        } else {
            CHECK(c == NULL && strstr(err, "join") != NULL, "I reject incompatible classifier joins");
        }
        free(c);
        nvm_module_free(m);
    }
}

static void test_classifier_local_bounds(void) {
    NvmModule *m = assemble_ok(".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",
                              "classifier bounds");
    if (!m) return;
    char err[256];
    m->functions[0].local_count = UINT16_MAX;
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL && strstr(err, "counts") != NULL, "I reject oversized locals before writing classifier state");
    free(c);
    m->functions[0].local_count = 0;
    m->functions[0].arity = 1;
    c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL && strstr(err, "counts") != NULL, "I reject arity exceeding local storage");
    free(c);
    nvm_module_free(m);
}

static void test_loop_carried_stack(void) {
    for (int variant = 0; variant < 4; variant++) {
        int swap = variant & 1;
        int conditional = variant & 2;
        char source[1024];
        snprintf(source, sizeof source,
            ".entry 0\n.function main 0 1 0 int 1\n"
            "PUSH_I64 3\nSTORE_LOCAL 0\n%s"
            "again:\nLOAD_LOCAL 0\nPUSH_I64 0\nI64_GT_S\nJMP_FALSE end\n"
            "%sLOAD_LOCAL 0\nPUSH_I64 1\nI64_SUB\nSTORE_LOCAL 0\n%s"
            "end:\n%sRET\n.end\n",
            swap ? "PUSH_I64 7\nPUSH_I64 2\n" : "PUSH_I64 40\n",
            swap ? "SWAP\n" : "PUSH_I64 1\nI64_ADD\n",
            conditional ? "LOAD_LOCAL 0\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n" : "JMP again\n",
            swap ? "I64_SUB\n" : "");
        NvmModule *m = assemble_ok(source, "loop-carried stack");
        if (!m) continue;
        char *c = emit_or_fail(m, "loop-carried stack");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile loop-carried stack transfers");
            CHECK(status == (swap ? 251 : 43), "I preserve simultaneous backedge values");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_variant_tags_and_payloads(void) {
    const char *bodies[] = {
        "AGG_PACK 1 0 0 0\nAGG_TAG\nRET\n",
        "AGG_PACK 1 0 65535 0\nAGG_TAG\nPUSH_I64 65535\nI64_EQ\nRET\n",
        "PUSH_I64 42\nAGG_PACK 1 0 7 1\nDUP\nAGG_TAG\nPUSH_I64 7\nI64_EQ\nASSERT\nAGG_GET 0\nRET\n",
        "PUSH_STR payload\nAGG_PACK 1 0 9 1\nDUP\nAGG_TAG\nPUSH_I64 9\nI64_EQ\nASSERT\nAGG_GET 0\nSTR_LEN\nRET\n",
        "PUSH_I64 42\nRET\nJMP dead\ndead:\nPOP\nJMP end\nend:\n"
    };
    const int expected[] = {0, 1, 42, 5, 42};
    for (int i = 0; i < 5; i++) {
        char source[1024];
        snprintf(source, sizeof source,
                 ".string payload \"hello\"\n.entry 0\n.function main 0 0 0 int 1\n%s.end\n", bodies[i]);
        NvmModule *m = assemble_ok(source, "variant values and dead labels");
        if (!m) continue;
        char *c = emit_or_fail(m, "variant values and dead labels");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile variant payloads and dead-label paths");
            CHECK(status == expected[i], "I preserve the variant tag, payload and reachable return");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_aggregate_runtime_kind_checks(void) {
    const char *sources[] = {
        ".entry 0\n.function main 0 0 0 int 1\nAGG_PACK 1 0 0 0\nAGG_TAG\nRET\n.end\n",
        ".string payload \"hello\"\n.entry 1\n"
        ".function read 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_STR payload\nAGG_PACK 1 0 0 1\nCALL read\nRET\n.end\n"
    };
    for (int i = 0; i < 2; i++) {
        NvmModule *m = assemble_ok(sources[i], "aggregate runtime kind checks");
        if (!m) continue;
        if (i == 0) {
            /* I exercise the direct API with a non-variant AGG_TAG input. */
            m->code[1] = AGG_RECORD;
        }
        if (i == 1) {
            char error[256];
            char *rejected = nvm2c_emit(m, error, sizeof error);
            CHECK(rejected == NULL && strstr(error, "RET") != NULL,
                  "I reject a known string field returned as an integer before execution");
            free(rejected);
            nvm_module_free(m);
            continue;
        }
        char *c = emit_or_fail(m, "aggregate runtime kind checks");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0, "I compile aggregate runtime guards");
            CHECK(status == -1, "I abort on non-variant tags or mismatched field storage");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_aggregate_call_facts(void) {
    for (int variant = 0; variant < 4; variant++) {
        const char *tag = (variant & 1) ? "union" : "struct";
        int kind = (variant & 1) ? AGG_VARIANT : AGG_RECORD;
        char make[512], relay[512], main_source[1024], source[4096];
        snprintf(make, sizeof make,
                 ".function make 2 2 0 %s 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n"
                 "AGG_PACK %d 0 65535 2\nRET\n.end\n", tag, kind);
        snprintf(relay, sizeof relay,
                 ".function relay 2 2 0 %s 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nTAIL_CALL make\n.end\n"
                 ".function identity 1 1 0 %s 1\nLOAD_LOCAL 0\nRET\n.end\n", tag, tag);
        snprintf(main_source, sizeof main_source,
                 ".function main 0 1 0 int 1\nPUSH_I64 42\nPUSH_STR hello\n"
                 "CALL relay\nCALL identity\nSTORE_LOCAL 0\n%s"
                 "LOAD_LOCAL 0\nAGG_GET 1\nSTR_LEN\nPUSH_I64 5\nI64_EQ\nASSERT\n"
                 "LOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n",
                 kind == AGG_VARIANT ? "LOAD_LOCAL 0\nAGG_TAG\nPUSH_I64 65535\nI64_EQ\nASSERT\n" : "");
        snprintf(source, sizeof source, ".string hello \"hello\"\n.entry main\n%s%s%s",
                 variant & 2 ? main_source : make, relay, variant & 2 ? make : main_source);
        NvmModule *m = assemble_ok(source, "aggregate direct-call facts");
        if (!m) continue;
        char *c = emit_or_fail(m, "aggregate direct-call facts");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile aggregate return and tail-call chains");
            CHECK(status == 42, "I preserve string and integer fields independently of function order");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_unrepresentable_call_facts(void) {
    const char *sources[] = {
        ".string hello \"hello\"\n.entry main\n"
        ".function read 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_I64 42\nAGG_PACK 0 0 0 1\nCALL read\nPOP\n"
        "PUSH_STR hello\nAGG_PACK 0 0 0 1\nCALL read\nRET\n.end\n",
        ".entry main\n.function make 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",
        ".entry main\n.function take 1 1 0 int 1\nPUSH_I64 0\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_I64 1\nCALL take\nPOP\n"
        "PUSH_I64 2\nAGG_PACK 0 0 0 1\nCALL take\nRET\n.end\n"
    };
    for (int i = 0; i < 3; i++) {
        NvmModule *m = assemble_ok(sources[i], "unrepresentable function facts");
        if (!m) continue;
        char error[256];
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, i == 1 ? "AGG_PACK" : "conflicting") != NULL,
              "I reject conflicting or unresolved field types instead of guessing");
        free(c);
        nvm_module_free(m);
    }
}

static void test_recursive_and_branch_record_facts(void) {
    const char *source =
        ".entry main\n"
        ".function select 2 2 0 struct 1\nLOAD_LOCAL 1\nJMP_FALSE alternate\n"
        "PUSH_I64 42\nAGG_PACK 0 0 0 1\nJMP joined\n"
        "alternate:\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\njoined:\nRET\n.end\n"
        ".function recurse 2 2 0 struct 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n"
        "LOAD_LOCAL 0\nRET\nagain:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\n"
        "CALL recurse\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_I64 7\nPUSH_BOOL 0\nCALL select\nPUSH_I64 3\n"
        "CALL recurse\nAGG_GET 0\nPUSH_I64 7\nI64_EQ\nASSERT\n"
        "PUSH_I64 7\nPUSH_BOOL 1\nCALL select\nPUSH_I64 3\nCALL recurse\nAGG_GET 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "recursive and joined record facts");
    if (!m) return;
    char *c = emit_or_fail(m, "recursive and joined record facts");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0, "I compile recursive aggregate calls and partial-fact joins");
        CHECK(status == 42, "I preserve aggregate values across both selected branches and recursive returns");
        free(c);
    }
    nvm_module_free(m);
}

static void test_self_tail_restart_preserves_values(void) {
    const char *sources[] = {
        ".entry 1\n.function swap_count 3 3 0 int 1\n"
        "LOAD_LOCAL 2\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\nLOAD_LOCAL 0\nRET\n"
        "again:\nLOAD_LOCAL 1\nLOAD_LOCAL 0\nLOAD_LOCAL 2\nPUSH_I64 1\nI64_SUB\n"
        "TAIL_CALL swap_count\n.end\n.function main 0 0 0 int 1\n"
        "PUSH_I64 7\nPUSH_I64 9\nPUSH_I64 100001\nCALL swap_count\nRET\n.end\n",
        ".string text \"retained\"\n.entry 1\n.function rec_count 2 2 0 struct 1\n"
        "LOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\nLOAD_LOCAL 0\nRET\n"
        "again:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nTAIL_CALL rec_count\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_I64 7\nPUSH_STR text\nAGG_PACK 0 0 0 2\n"
        "PUSH_I64 100000\nCALL rec_count\nAGG_GET 1\nSTR_LEN\nRET\n.end\n",
        ".string left \"left\"\n.string right \"right\"\n.entry 1\n"
        ".function string_swap 3 3 0 string 1\nLOAD_LOCAL 2\nPUSH_I64 0\nI64_EQ\n"
        "JMP_FALSE again\nLOAD_LOCAL 0\nRET\nagain:\nLOAD_LOCAL 1\nLOAD_LOCAL 0\n"
        "LOAD_LOCAL 2\nPUSH_I64 1\nI64_SUB\nTAIL_CALL string_swap\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_STR left\nPUSH_STR right\nPUSH_I64 100001\n"
        "CALL string_swap\nSTR_LEN\nRET\n.end\n",
        ".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 7\nRET\nTAIL_CALL main\n.end\n"
    };
    const int expected[] = {9, 8, 5, 7};
    for (size_t i = 0; i < sizeof sources / sizeof sources[0]; ++i) {
        NvmModule *m = assemble_ok(sources[i], "self-tail restart");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        if (!c) fprintf(stderr, "self-tail restart: %s\n", error);
        CHECK(c != NULL, "self-tail restart emits native C");
        if (c) {
            CHECK(strstr(c, "goto L_tco") != NULL, "self-tail lowering uses a local restart");
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "deep self-tail C compiles and executes at default O0");
            CHECK(status == expected[i], "self-tail restart preserves simultaneous arguments and result");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_self_tail_rejects_malformed_calls(void) {
    const char *sources[] = {
        ".entry 1\n.function repeat 1 1 0 int 1\nPUSH_I64 2\nLOAD_LOCAL 0\n"
        "TAIL_CALL repeat\n.end\n.function main 0 0 0 int 1\nPUSH_I64 1\nCALL repeat\nRET\n.end\n",
        ".string bad \"bad\"\n.entry 1\n.function repeat 1 1 0 int 1\nPUSH_STR bad\n"
        "TAIL_CALL repeat\n.end\n.function main 0 0 0 int 1\nPUSH_I64 1\nCALL repeat\nRET\n.end\n"
    };
    for (size_t i = 0; i < sizeof sources / sizeof sources[0]; ++i) {
        NvmModule *m = assemble_ok(sources[i], "invalid self-tail call");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL, "I reject leftover stack values and inconsistent self-tail argument kinds");
        free(c);
        nvm_module_free(m);
    }
}

static void test_array_set_aliases_bounds_and_types(void) {
    const char *initial[] = {
        "PUSH_I64 1\nARR_LITERAL 1 1\n",
        "PUSH_STR old\nARR_LITERAL 5 1\n",
        "ARR_NEW 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 1\nPUSH_STR old\nAGG_PACK 0 0 0 2\nARR_PUSH\n"
    };
    const char *replacement[] = {
        "PUSH_I64 7\n", "PUSH_STR updated\n",
        "PUSH_I64 7\nPUSH_STR updated\nAGG_PACK 0 0 0 2\n"
    };
    const char *readback[] = { "", "STR_LEN\n", "AGG_GET 0\n" };
    const char *indices[] = { "0", "-1", "1", "9223372036854775807" };
    for (int kind = 0; kind < 3; ++kind) {
        for (int index = 0; index < 4; ++index) {
            char source[2048];
            snprintf(source, sizeof source,
                ".string old \"a\"\n.string updated \"changed\"\n.entry 0\n"
                ".function main 0 2 0 int 1\n%sSTORE_LOCAL 0\n"
                "LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nPUSH_I64 %s\n%s"
                "ARR_SET\nPOP\nLOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\n%sRET\n.end\n",
                initial[kind], indices[index], replacement[kind], readback[kind]);
            NvmModule *m = assemble_ok(source, "array mutation boundary");
            if (!m) continue;
            char error[256] = {0};
            char *c = nvm2c_emit(m, error, sizeof error);
            if (!c) fprintf(stderr, "array mutation: %s\n", error);
            CHECK(c != NULL, "array mutation emits C");
            if (c) {
                int status = 0;
                CHECK(compile_and_run(c, &status) == 0, "array mutation C compiles and executes");
                CHECK(index == 0 ? status == 7 : (status == -1 || status == 134),
                      "aliases see mutation; invalid indices abort");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    const char *invalid[] = {
        "PUSH_I64 1\nARR_LITERAL 1 1\nPUSH_I64 0\nPUSH_STR text\n",
        "PUSH_STR text\nARR_LITERAL 5 1\nPUSH_I64 0\nPUSH_I64 1\n",
        "PUSH_I64 1\nARR_LITERAL 1 1\nPUSH_STR text\nPUSH_I64 1\n",
        ("ARR_NEW 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_PUSH\n"
         "STORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_STR text\nAGG_PACK 0 0 0 1\n"),
        ("ARR_NEW 1\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_PUSH\n"
         "STORE_LOCAL 0\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_I64 2\nPUSH_I64 3\nAGG_PACK 0 0 0 2\n")
    };
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        char source[2048];
        snprintf(source, sizeof source,
            ".string text \"bad\"\n.entry 0\n.function main 0 1 0 int 1\n"
            "%sARR_SET\nPOP\nPUSH_I64 0\nRET\n.end\n", invalid[i]);
        NvmModule *m = assemble_ok(source, "invalid array mutation");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        if (i == 4) {
            CHECK(c != NULL, "record-width mutation has runtime representation guards");
            if (c) {
                int status = 0;
                CHECK(compile_and_run(c, &status) == 0, "record-width guard compiles and executes");
                CHECK(status == -1 || status == 134, "I reject incompatible record widths at runtime");
            }
        } else {
            CHECK(c == NULL && strstr(error, "ARR_SET"), "I reject incompatible array mutation");
        }
        free(c);
        nvm_module_free(m);
    }
}

static void test_string_edges_run_as_native_c(void) {
    const struct { const char *text, *part; int starts, ends; } cases[] = {
        {"", "", 1, 1}, {"abc", "", 1, 1}, {"", "a", 0, 0},
        {"abc", "abc", 1, 1}, {"abc", "abcd", 0, 0},
        {"abc", "ab", 1, 0}, {"abc", "bc", 0, 1},
        {"abc", "b", 0, 0}, {"éclair", "é", 1, 0},
        {"é7", "7", 0, 1}, {"\0017", "\001", 1, 0}
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        for (int suffix = 0; suffix < 2; ++suffix) {
            char source[512];
            snprintf(source, sizeof source,
                ".string text \"%s\"\n.string part \"%s\"\n.entry 1\n"
                ".function predicate 0 0 0 bool 1\n"
                "PUSH_STR text\nPUSH_STR part\n%s\nRET\n.end\n"
                ".function main 0 0 0 int 1\nCALL predicate\nJMP_FALSE no\n"
                "PUSH_I64 1\nRET\nno:\nPUSH_I64 0\nRET\n.end\n",
                cases[i].text, cases[i].part,
                suffix ? "STR_ENDS_WITH" : "STR_STARTS_WITH");
            NvmModule *m = assemble_ok(source, "native string edge");
            CHECK(m != NULL, "string edge fixture assembles");
            if (!m) continue;
            char error[256] = {0};
            char *c = nvm2c_emit(m, error, sizeof error);
            if (!c) fprintf(stderr, "string edge emission: %s\n", error);
            CHECK(c != NULL, "string edge emits C");
            if (c) {
                CHECK(strstr(c, "nano_vm") == NULL, "string edge has no VM dependency");
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0, "string edge C compiles and runs");
                CHECK(status == (suffix ? cases[i].ends : cases[i].starts),
                      "native string edge has expected byte semantics");
                free(c);
            }
            nvm_module_free(m);
        }
    }
}

int main(int argc, char **argv) {
    test_self_tail_restart_preserves_values();
    test_self_tail_rejects_malformed_calls();
    test_array_set_aliases_bounds_and_types();
    test_string_edges_run_as_native_c();
    printf("\n[nvm2c] structured C11 from NanoISA...\n\n");
    test_record_result_crosses_direct_call();
    test_add_is_structured_c_and_runs();
    test_store_load_local();
    test_builtin_host_imports();
    test_builtin_text_reader();
    test_builtin_text_writer();
    test_builtin_filesystem_predicates();
    test_builtin_removal_and_rename();
    test_builtin_identity();
    test_builtin_normalize();
    test_builtin_capture();
    test_builtin_from_char();
    test_artifact_array_import_is_not_a_builtin();
    test_owned_artifact_execution();
    test_real_walk_artifact();
    test_call_extern_is_refused();
    test_str_trim_is_refused();
    test_push_str_len_runs_without_nano_vm();
    test_str_concat_len_runs_without_nano_vm();
    test_greeting_runs_without_nano_vm();
    test_glue_runs_without_nano_vm();
    test_arr_set_runs_natively();
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
    test_nested_record_pack_is_refused();
    test_null_module();
    test_choose_then_runs_without_nano_vm();
    test_choose_else_runs_without_nano_vm();
    test_loop_sum_runs_without_nano_vm();
    test_tail_call_runs_without_nano_vm();
    test_classifier_branch_stack();
    test_classifier_unreachable_and_invalid_joins();
    test_classifier_local_bounds();
    test_loop_carried_stack();
    test_variant_tags_and_payloads();
    test_aggregate_runtime_kind_checks();
    test_aggregate_call_facts();
    test_unrepresentable_call_facts();
    test_recursive_and_branch_record_facts();
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
