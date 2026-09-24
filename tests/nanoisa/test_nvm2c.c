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
#include "utf8.h"

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

    snprintf(cmd, sizeof cmd, "exec perl -e 'alarm 30; exec @ARGV' %s %s", bin_path, args);
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

    snprintf(cmd, sizeof cmd, "exec perl -e 'alarm 30; exec @ARGV' %s > %s", bin_path, out_path);
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
                     strcmp(item->type, "string") == 0 ? "PUSH_STR " :
                     strcmp(item->type, "bool") == 0 ? "PUSH_BOOL " : "PUSH_I64 ",
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

static void test_builtin_temp_directory(void) {
    char root[] = "/tmp/nvm2c-temp-root-XXXXXX";
    if (!mkdtemp(root)) { CHECK(0, "I create a private temporary root"); return; }
    const char *previous = getenv("TMPDIR");
    char *saved = previous ? strdup(previous) : NULL;
    for (int failure = 0; failure < 2; ++failure) {
        char temporary_root[256], assembly[4096];
        snprintf(temporary_root, sizeof temporary_root, "%s", root);
        CHECK(setenv("TMPDIR", temporary_root, 1) == 0, "I scope temporary creation to my private root");
        const char *body = failure ?
            "PUSH_STR prefix\nCALL_EXTERN 0\nSTR_LEN\nPUSH_I64 0\nEQ\nASSERT\n" :
            "PUSH_STR prefix\nCALL_EXTERN 0\nSTORE_LOCAL 0\n"
            "PUSH_STR prefix\nCALL_EXTERN 0\nSTORE_LOCAL 1\n"
            "LOAD_LOCAL 0\nLOAD_LOCAL 1\nNE\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_STR root\nSTR_STARTS_WITH\nASSERT\n"
            "LOAD_LOCAL 1\nPUSH_STR root\nSTR_STARTS_WITH\nASSERT\n"
            "LOAD_LOCAL 0\nCALL_EXTERN 1\nASSERT\nLOAD_LOCAL 1\nCALL_EXTERN 1\nASSERT\n"
            "LOAD_LOCAL 0\nCALL_EXTERN 2\nPUSH_I64 0\nEQ\nASSERT\n"
            "LOAD_LOCAL 1\nCALL_EXTERN 2\nPUSH_I64 0\nEQ\nASSERT\n";
        snprintf(assembly, sizeof assembly,
            ".string prefix \"%s\"\n.string root \"%s/owned_\"\n"
            ".import \"\" \"vm_mktemp_dir\" string string\n"
            ".import \"\" \"dir_exists\" bool string\n"
            ".import \"\" \"file_remove\" int string\n"
            ".entry 0\n.function main 0 2 0 int 1\n%sPUSH_I64 0\nRET\n.end\n",
            failure ? "missing/owned_" : "owned_", root, body);
        NvmModule *module = assemble_ok(assembly, "exclusive temporary directories");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit the temporary-directory adapter");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I create distinct owned directories or return empty on failure");
        } else fprintf(stderr, "%s\n", error);
        free(source);
        module->import_param_types[0][0] = TAG_INT;
        source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I reject a non-string temporary prefix");
        free(source); nvm_module_free(module);
    }
    if (saved) { setenv("TMPDIR", saved, 1); free(saved); }
    else unsetenv("TMPDIR");
    CHECK(rmdir(root) == 0, "I leave no temporary directories in my private root");
}

static void test_tagged_record_array(void) {
    const char *programs[] = {
        ".entry 0\n.function main 0 0 0 int 1\nARR_NEW 8\nARR_LEN\nRET\n.end\n",
        ".string text \"retained\"\n.entry 0\n.function main 0 1 0 int 1\n"
        "ARR_NEW 8\nSTORE_LOCAL 0\nLOAD_LOCAL 0\n"
        "PUSH_I64 7\nPUSH_STR text\nAGG_PACK 0 0 0 2\nARR_PUSH\nPOP\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 1\nPUSH_STR text\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
        ".string text \"direct\"\n.entry 0\n.function main 0 0 0 int 1\n"
        "ARR_NEW 8\nPUSH_I64 9\nPUSH_STR text\nAGG_PACK 0 0 0 2\nARR_PUSH\n"
        "PUSH_I64 0\nARR_GET\nAGG_GET 1\nPUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
    };
    for (size_t i = 0; i < sizeof programs / sizeof programs[0]; ++i) {
        NvmModule *module = assemble_ok(programs[i], "explicit struct-array construction");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I emit explicitly tagged record arrays");
        if (source) {
            if (i == 0) CHECK(strstr(source, "(void)nrarr_new; (void)nrarr_reserve;") != NULL,
                              "I keep optional record-array helpers referenced for strict C compilers");
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0 && status == 0,
                  "I preserve empty and mixed-field record arrays through stack/local flow");
        } else fprintf(stderr, "%s\n", error);
        free(source); nvm_module_free(module);
    }
    const char *rejected[] = {
        ".entry 0\n.function main 0 0 0 int 1\nARR_NEW 8\nPUSH_I64 1\nARR_PUSH\nARR_LEN\nRET\n.end\n",
        ".string text \"text\"\n.entry 0\n.function main 0 0 0 int 1\n"
        "ARR_NEW 8\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_PUSH\n"
        "PUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_PUSH\nARR_LEN\nRET\n.end\n",
    };
    for (size_t i = 0; i < sizeof rejected / sizeof rejected[0]; ++i) {
        NvmModule *module = assemble_ok(rejected[i], "unsupported record-array representation");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source == NULL, "I refuse scalar and mixed-field record construction");
        free(source); nvm_module_free(module);
    }
}

static void test_character_host_imports(void) {
    const char *names[] = {"vm_is_digit", "vm_is_alpha", "vm_is_alnum", "vm_is_space",
                           "vm_is_upper", "vm_is_lower", "vm_is_whitespace", "vm_digit_value"};
    const int64_t codes[] = {-257, -1, 0, 9, 10, 11, 12, 13, 32, 47, 48, 57, 58,
                             64, 65, 90, 91, 96, 97, 122, 123, 127, 128, 255, 256,
                             288, INT64_C(4294967361), INT64_MIN, INT64_MAX};
    for (size_t kind = 0; kind < sizeof names / sizeof names[0]; ++kind) {
        char source[8192];
        size_t used = (size_t)snprintf(source, sizeof source,
            ".import \"\" \"%s\" %s int\n.entry main\n.function main 0 0 0 int 1\n",
            names[kind], kind == 7 ? "int" : "bool");
        for (size_t i = 0; i < sizeof codes / sizeof codes[0]; ++i) {
            int64_t code = codes[i], expected;
            int c = (int)code;
            switch (kind) {
                case 0: expected = nl_ascii_isdigit(c); break;
                case 1: expected = nl_ascii_isalpha(c); break;
                case 2: expected = nl_ascii_isalnum(c); break;
                case 3: expected = nl_ascii_isspace(c); break;
                case 4: expected = nl_ascii_isupper(c); break;
                case 5: expected = nl_ascii_islower(c); break;
                case 6: expected = code == ' ' || code == '\t' || code == '\n' || code == '\r'; break;
                default: expected = code >= '0' && code <= '9' ? code - '0' : -1; break;
            }
            int written = snprintf(source + used, sizeof source - used,
                "PUSH_I64 %lld\nCALL_EXTERN 0\n%s %lld\nEQ\nASSERT\n",
                (long long)code, kind == 7 ? "PUSH_I64" : "PUSH_BOOL", (long long)expected);
            CHECK(written >= 0 && (size_t)written < sizeof source - used,
                  "I retain the complete character host fixture");
            if (written < 0 || (size_t)written >= sizeof source - used) break;
            used += (size_t)written;
        }
        snprintf(source + used, sizeof source - used, "PUSH_I64 0\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, names[kind]);
        if (!m) continue;
        char error[256] = {0};
        char *c = emit_or_fail(m, "I bind exact character classification host signatures");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I preserve ASCII classes, whitespace differences, and full-width VM input behavior");
            free(c);
        }
        for (int bad = 0; bad < 8; ++bad) {
            NvmImportEntry saved = m->imports[0];
            uint8_t parameter = m->import_param_types[0][0];
            switch (bad) {
                case 0: m->imports[0].kind = NVM_IMPORT_COPROCESS; break;
                case 1: m->imports[0].kind = NVM_IMPORT_ARTIFACT; break;
                case 2: m->imports[0].return_type = kind == 7 ? TAG_BOOL : TAG_INT; break;
                case 3: m->imports[0].module_name_idx = nvm_add_string(m, "libc", 4); break;
                case 4: m->imports[0].param_count = 0; break;
                case 5: m->import_param_types[0][0] = TAG_BOOL; break;
                case 6: m->imports[0].module_name_idx = nvm_add_string(m, "\0foreign", 8); break;
                default: m->imports[0].function_name_idx = nvm_add_string(m, names[kind],
                                                                         (uint32_t)strlen(names[kind]) + 1); break;
            }
            c = nvm2c_emit(m, error, sizeof error);
            CHECK(c == NULL, "I refuse character hosts with noncanonical namespace, kind, arity or tags");
            free(c);
            m->imports[0] = saved;
            m->import_param_types[0][0] = parameter;
        }
        nvm_module_free(m);
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

static void test_globals_cross_functions_and_preserve_identity(void) {
    const char *src =
        ".entry 2\n"
        ".function init 0 0 0 void 0\n"
        "  ARR_NEW 1\n"
        "  STORE_GLOBAL 3\n"
        "  RET\n"
        ".end\n"
        ".function append 0 0 0 void 0\n"
        "  LOAD_GLOBAL 3\n"
        "  PUSH_I64 41\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL init\n"
        "  CALL append\n"
        "  LOAD_GLOBAL 3\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  PUSH_I64 1\n"
        "  I64_ADD\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "globals fixture");
    CHECK(m != NULL, "globals fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c emits typed globals across functions");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0, "global generated C compiles and runs");
        CHECK(status == 42, "global array mutation preserves shared identity");
        free(c);
    } else {
        printf("    nvm2c error: %s\n", err);
    }
    nvm_module_free(m);
}

static void test_projected_global_stores(void) {
    const struct { const char *value, *check; } cases[] = {
        {"PUSH_I64 42", "PUSH_I64 42\nEQ\nASSERT"},
        {"PUSH_F64 1.5", "PUSH_F64 1.5\nF64_EQ\nASSERT"},
        {"PUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1",
         "PUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_I64 1\nEQ\nASSERT"},
        {"PUSH_BOOL 1", "DUP\nTYPE_CHECK 4\nASSERT\nASSERT"},
        {"PUSH_STR text", "PUSH_STR text\nEQ\nASSERT"},
        {"PUSH_I64 42\nARR_LITERAL 1 1", "PUSH_I64 0\nARR_GET\nPUSH_I64 42\nEQ\nASSERT"},
        {"PUSH_BOOL 1\nARR_LITERAL 4 1", "PUSH_I64 0\nARR_GET\nDUP\nTYPE_CHECK 4\nASSERT\nASSERT"},
        {"PUSH_STR text\nARR_LITERAL 5 1", "PUSH_I64 0\nARR_GET\nPUSH_STR text\nEQ\nASSERT"},
    };
    const char *store = ".function store 1 1 0 void 0\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nSTORE_GLOBAL 0\nRET\n.end\n";
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        for (int before = 0; before < 2; ++before) {
            char main[2048], source[8192];
            snprintf(main, sizeof main,
                ".function main 0 0 0 int 1\n%s\nAGG_PACK 0 0 0 1\nAGG_PACK 0 0 0 1\n"
                "CALL store\nLOAD_GLOBAL 0\n%s\nPUSH_I64 0\nRET\n.end\n",
                cases[i].value, cases[i].check);
            snprintf(source, sizeof source, ".string text \"retained\"\n.entry main\n%s%s",
                     before ? store : main, before ? main : store);
            NvmModule *m = assemble_ok(source, "projected global storage");
            if (!m) continue;
            char *c = emit_or_fail(m, "I resolve nested global-store projections after collecting all shape facts");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I preserve scalar and primitive-array values across nested-record global stores");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    const char *unsupported[] = {
        "PUSH_I64 1\nAGG_PACK 0 0 0 1"
    };
    for (size_t i = 0; i < sizeof unsupported / sizeof unsupported[0]; ++i) {
        char source[2048];
        snprintf(source, sizeof source, ".entry main\n.function main 0 0 0 int 1\n%s\n"
            "AGG_PACK 0 0 0 1\nAGG_PACK 0 0 0 1\nCALL store\nPUSH_I64 0\nRET\n.end\n%s",
            unsupported[i], store);
        NvmModule *m = assemble_ok(source, "unsupported projected global storage");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL, "I reject unsupported global storage after resolving nested fields");
        free(c); nvm_module_free(m);
    }
}

static void test_tagged_scalar_local_assignments(void) {
    const struct { const char *value, *consume; } cases[] = {
        {"PUSH_I64 42", "PUSH_I64 1\nI64_ADD\nPUSH_I64 43\nEQ\nASSERT"},
        {"PUSH_BOOL 1", "BOOL_NOT\nBOOL_NOT\nASSERT"},
        {"PUSH_STR text", "STR_LEN\nPUSH_I64 4\nEQ\nASSERT"},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        for (int parameter = 0; parameter < 2; ++parameter) {
            for (int boxed_first = 0; boxed_first < 2; ++boxed_first) {
                char source[4096];
                snprintf(source, sizeof source,
                    ".string text \"text\"\n.entry main\n.function main 0 0 0 int 1\n"
                    "%s\nSTORE_GLOBAL 0\n%s\nCALL consume\nPUSH_I64 0\nRET\n.end\n"
                    ".function consume %d 1 0 void 0\n%s\nSTORE_LOCAL 0\n%s\nSTORE_LOCAL 0\n"
                    "LOAD_LOCAL 0\n%s\nRET\n.end\n",
                    cases[i].value, parameter ? cases[i].value : "", parameter,
                    boxed_first ? "LOAD_GLOBAL 0" : cases[i].value,
                    boxed_first ? cases[i].value : "LOAD_GLOBAL 0", cases[i].consume);
                NvmModule *m = assemble_ok(source, "tagged scalar local assignment order");
                if (!m) continue;
                char *c = emit_or_fail(m, "I retain tagged local storage across scalar writes and parameter calls");
                if (c) {
                    int status = -1;
                    CHECK(compile_and_run(c, &status) == 0 && status == 0,
                          "I consume the actual scalar payload after either assignment order");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
    for (int strings = 0; strings < 2; ++strings) {
        char source[1024];
        snprintf(source, sizeof source,
            ".string key \"key\"\n.entry main\n.function main 0 1 0 int 1\n"
            "%s\nSTORE_LOCAL 0\nHM_NEW 5 %d\nPUSH_STR key\nHM_GET\nSTORE_LOCAL 0\n"
            "PUSH_I64 0\nRET\n.end\n", strings ? "PUSH_STR key" : "PUSH_I64 1", strings ? 1 : 5);
        NvmModule *m = assemble_ok(source, "incompatible tagged local payload");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I retain exact local payload conflicts after storage widening");
        free(c); nvm_module_free(m);
    }
    const char *invalid[] = {"PUSH_BOOL 1", "PUSH_STR text", "LOAD_GLOBAL 1", "ARR_NEW 1"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        for (int take = 0; take < 2; ++take) {
            char source[2048];
            snprintf(source, sizeof source,
                ".string text \"text\"\n.entry main\n.function main 0 1 0 int 1\n"
                "%s\nSTORE_GLOBAL 0\nPUSH_I64 42\nSTORE_LOCAL 0\nPUSH_BOOL %d\nJMP_FALSE consume\n"
                "LOAD_GLOBAL 0\nSTORE_LOCAL 0\nconsume:\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\n"
                "PUSH_I64 43\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n", invalid[i], take);
            NvmModule *m = assemble_ok(source, "checked tagged scalar local path");
            if (!m) continue;
            char *c = emit_or_fail(m, "I keep dynamic scalar validation at its executed consumption");
            if (c) {
                int status = 0;
                CHECK(compile_and_run(c, &status) == 0 && (take ? status != 0 : status == 0),
                      "I reject wrong tags on the taken path and preserve the untouched scalar otherwise");
                free(c);
            }
            nvm_module_free(m);
        }
    }
}

static void test_uninitialized_global_result_traps(void) {
    NvmModule *m = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\n"
        "LOAD_GLOBAL 0\nRET\n.end\n", "uninitialized global result");
    if (!m) return;
    char *c = emit_or_fail(m, "I preserve uninitialized globals as tagged void values");
    if (c) {
        int status = 0;
        CHECK(compile_and_run(c, &status) == 0 && status == -1,
              "I reject consuming an uninitialized global as an integer result");
        free(c);
    }
    nvm_module_free(m);
}

static void test_float_comparison_transport(void) {
    const char *source =
        ".entry main\n.function less 2 2 0 bool 1\n"
        "LOAD_LOCAL 0\nLOAD_LOCAL 1\nLT\nRET\n.end\n"
        ".function relay 2 2 0 bool 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nTAIL_CALL less\n.end\n"
        ".function main 0 1 0 int 1\nPUSH_F64 1.5\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nDUP\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 2\nCALL relay\nTYPE_CHECK 4\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 2\nCALL less\nASSERT\n"
        "PUSH_I64 2\nLOAD_LOCAL 0\nGT\nASSERT\n"
        "PUSH_F64 2.0\nPUSH_I64 2\nEQ\nASSERT\n"
        "PUSH_F64 -0.0\nPUSH_I64 0\nEQ\nASSERT\n"
        "PUSH_F64 inf\nPUSH_F64 1.5\nGT\nASSERT\n"
        "PUSH_F64 nan\nPUSH_F64 1.5\nLE\nASSERT\n"
        "PUSH_F64 nan\nPUSH_F64 1.5\nGE\nASSERT\n"
        "PUSH_F64 nan\nPUSH_F64 nan\nEQ\nBOOL_NOT\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "float comparison transport");
    if (!m) return;
    char *c = emit_or_fail(m, "I preserve float constants, locals and direct/tail call arguments");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0 && status == 0,
              "I retain numeric comparison and VM NaN ordering through float storage");
        free(c);
    }
    nvm_module_free(m);
}

static void test_generic_comparisons_are_typed(void) {
    const char *src =
        ".string apple \"apple\"\n"
        ".string berry \"berry\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR apple\n"
        "  PUSH_STR berry\n"
        "  LT\n"
        "  CAST_INT\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "generic string comparison fixture");
    CHECK(m != NULL, "generic string comparison fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "nvm2c lowers generic string LT");
    if (c) {
        int status = -1;
        CHECK(strstr(c, "strcmp") != NULL, "generic string LT uses lexical comparison");
        CHECK(compile_and_run(c, &status) == 0, "generic string LT C compiles and runs");
        CHECK(status == 1, "apple compares less than berry");
        free(c);
    }
    nvm_module_free(m);

    src =
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_F64 1.5\n"
        "  PUSH_I64 2\n"
        "  LT\n"
        "  CAST_INT\n"
        "  RET\n"
        ".end\n";
    m = assemble_ok(src, "generic numeric comparison fixture");
    CHECK(m != NULL, "generic numeric comparison fixture assembles");
    if (m) {
        c = nvm2c_emit(m, err, sizeof err);
        CHECK(c != NULL, "nvm2c lowers generic float/int LT without integer coercion");
        if (c) {
            int status = -1;
            CHECK(strstr(c, "double") != NULL, "generic float/int LT retains a double operand");
            CHECK(compile_and_run(c, &status) == 0, "generic float/int LT C compiles and runs");
            CHECK(status == 1, "1.5 compares less than 2");
            free(c);
        }
        nvm_module_free(m);
    }

    src =
        ".string one \"one\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR one\n"
        "  PUSH_I64 1\n"
        "  LT\n"
        "  CAST_INT\n"
        "  RET\n"
        ".end\n";
    m = assemble_ok(src, "generic comparison mismatch fixture");
    CHECK(m != NULL, "generic comparison mismatch fixture assembles");
    if (m) {
        c = nvm2c_emit(m, err, sizeof err);
        CHECK(c != NULL, "I preserve VM generic ordering across distinct scalar tags");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "string tags order after integer tags in the VM contract");
        }
        free(c);
        nvm_module_free(m);
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

static void test_str_to_upper_is_refused(void) {
    const char *src =
        ".string s \"hi\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR s\n"
        "  STR_TO_UPPER\n"
        "  POP\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "uppercase fixture");
    CHECK(m != NULL, "uppercase fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL, "STR_TO_UPPER stays outside the closed subset");
    CHECK(strstr(err, "STR_TO_UPPER") != NULL, "error names STR_TO_UPPER");
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

/* I preserve optional scalar call facts while requiring a real integer at use. */
static void test_boxed_array_arguments(void) {
    const struct { int tag; const char *initial, *replacement; } kinds[] = {
        {1, "PUSH_I64 3", "PUSH_I64 7"},
        {4, "PUSH_BOOL 0", "PUSH_BOOL 1"},
        {5, "PUSH_STR before", "PUSH_STR after"}
    };
    for (size_t k = 0; k < sizeof kinds / sizeof kinds[0]; ++k) {
        for (int before = 0; before < 2; ++before) {
            for (int boxed_first = 0; boxed_first < 2; ++boxed_first) {
                char workers[2048], body[3072], source[8192];
                snprintf(workers, sizeof workers,
                    ".function update 1 1 0 int 1\nLOAD_LOCAL 0\nTYPE_CHECK 0\nJMP_FALSE present\n"
                    "PUSH_I64 0\nRET\npresent:\nLOAD_LOCAL 0\nPUSH_I64 0\n%s\nARR_SET\nPOP\n"
                    "LOAD_LOCAL 0\nARR_LEN\nRET\n.end\n"
                    ".function relay 1 1 0 int 1\nLOAD_LOCAL 0\nTAIL_CALL update\n.end\n",
                    kinds[k].replacement);
                snprintf(body, sizeof body,
                    ".function main 0 1 0 int 1\n%s\nARR_LITERAL %d 1\nSTORE_LOCAL 0\n"
                    "LOAD_LOCAL 0\nSTORE_GLOBAL 0\n%s\nCALL update\nPUSH_I64 1\nEQ\nASSERT\n"
                    "%s\nCALL relay\nPUSH_I64 1\nEQ\nASSERT\n"
                    "LOAD_GLOBAL 1\nCALL relay\nPUSH_I64 0\nEQ\nASSERT\n"
                    "PUSH_I64 99\nSTORE_GLOBAL 0\n"
                    "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n%s\nEQ\nASSERT\n"
                    "PUSH_I64 0\nRET\n.end\n",
                    kinds[k].initial, kinds[k].tag,
                    boxed_first ? "LOAD_GLOBAL 0" : "LOAD_LOCAL 0",
                    boxed_first ? "LOAD_LOCAL 0" : "LOAD_GLOBAL 0", kinds[k].replacement);
                snprintf(source, sizeof source,
                    ".string before \"before\"\n.string after \"after\"\n.entry main\n%s%s",
                    before ? workers : body, before ? body : workers);
                NvmModule *m = assemble_ok(source, "boxed array parameter ordering");
                if (!m) continue;
                char *c = emit_or_fail(m, "I infer boxed array parameters independently of caller and function order");
                if (c) {
                    int status = -1;
                    CHECK(compile_and_run(c, &status) == 0 && status == 0,
                          "I preserve array tags, absent values, aliases and tail calls after global replacement");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
    /* I must not let a boxed source erase incompatible concrete payload facts. */
    for (int tag = 1; tag <= 4; tag += 3) {
        char source[1024];
        snprintf(source, sizeof source,
            ".entry main\n.function main 0 0 0 int 1\nLOAD_GLOBAL 0\nCALL size\nPOP\n"
            "ARR_NEW 5\nCALL size\nPOP\n%s\nARR_LITERAL %d 1\nCALL size\nRET\n.end\n"
            ".function size 1 1 0 int 1\nLOAD_LOCAL 0\nARR_LEN\nRET\n.end\n",
            tag == 1 ? "PUSH_I64 1" : "PUSH_BOOL 1", tag);
        NvmModule *m = assemble_ok(source, "incompatible boxed array payload");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I reject incompatible array elements after parameter boxing");
        if (c || !strstr(error, "shape")) fprintf(stderr, "array payload tag %d: %s\n", tag, error);
        free(c); nvm_module_free(m);
    }
    const char *invalid[] = {"PUSH_I64 9", "PUSH_BOOL 1", "PUSH_STR text", "PUSH_I64 1\nARR_LITERAL 1 1"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        char source[1024];
        snprintf(source, sizeof source,
            ".string text \"text\"\n.entry main\n.function main 0 0 0 int 1\n"
            "PUSH_STR text\nARR_LITERAL 5 1\nCALL update\n%s\nSTORE_GLOBAL 0\n"
            "LOAD_GLOBAL 0\nCALL update\nPUSH_I64 0\nRET\n.end\n"
            ".function update 1 1 0 void 0\nLOAD_LOCAL 0\nPUSH_I64 0\nPUSH_STR text\nARR_SET\nPOP\nRET\n.end\n",
            invalid[i]);
        NvmModule *m = assemble_ok(source, "invalid dynamic array argument");
        if (!m) continue;
        char *c = emit_or_fail(m, "I retain dynamic array argument checks at consumption");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && status != 0,
                  "I reject nonarray tags and mismatched dynamic array element updates");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_boxed_array_indices(void) {
    const struct {
        const char *array, *value, *check, *index;
        int succeeds;
    } cases[] = {
        {"PUSH_I64 1\nARR_LITERAL 1 1\n", "PUSH_I64 2\n", "PUSH_I64 2\nEQ\nASSERT\n", "PUSH_I64 0\n", 1},
        {"PUSH_BOOL 0\nARR_LITERAL 4 1\n", "PUSH_BOOL 1\n", "ASSERT\n", "PUSH_I64 0\n", 1},
        {"PUSH_STR before\nARR_LITERAL 5 1\n", "PUSH_STR after\n", "PUSH_STR after\nEQ\nASSERT\n", "PUSH_I64 0\n", 1},
        {"PUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n", "PUSH_I64 2\nAGG_PACK 0 0 0 1\n", "AGG_GET 0\nPUSH_I64 2\nEQ\nASSERT\n", "PUSH_I64 0\n", 1},
        {"PUSH_I64 1\nARR_LITERAL 1 1\n", "PUSH_I64 2\n", "POP\n", "PUSH_BOOL 0\n", 0},
        {"PUSH_I64 1\nARR_LITERAL 1 1\n", "PUSH_I64 2\n", "POP\n", "PUSH_STR before\n", 0},
        {"PUSH_I64 1\nARR_LITERAL 1 1\n", "PUSH_I64 2\n", "POP\n", "LOAD_LOCAL 0\n", 0},
        {"PUSH_I64 1\nARR_LITERAL 1 1\n", "PUSH_I64 2\n", "POP\n", "PUSH_I64 -1\n", 0},
        {"PUSH_I64 1\nARR_LITERAL 1 1\n", "PUSH_I64 2\n", "POP\n", "PUSH_I64 1\n", 0},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        char assembly[4096];
        snprintf(assembly, sizeof assembly,
                 ".string before \"before\"\n.string after \"after\"\n.entry main\n"
                 ".function update 2 4 0 void 0\nLOAD_LOCAL 1\nJMP_FALSE done\n"
                 "%sSTORE_LOCAL 2\nLOAD_LOCAL 2\nSTORE_LOCAL 3\n"
                 "LOAD_LOCAL 2\nLOAD_LOCAL 0\n%sARR_SET\nPOP\n"
                 "LOAD_LOCAL 3\nPUSH_I64 0\nARR_GET\n%sdone:\nRET\n.end\n"
                 ".function main 0 1 0 int 1\nLOAD_LOCAL 0\nPUSH_BOOL 0\nCALL update\n"
                 "%sPUSH_BOOL 1\nCALL update\nPUSH_I64 0\nRET\n.end\n",
                 cases[i].array, cases[i].value, cases[i].check, cases[i].index);
        NvmModule *module = assemble_ok(assembly, "boxed array index");
        if (!module) continue;
        char error[256] = {0};
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "I lower boxed indices through checked integer extraction");
        if (!source) fprintf(stderr, "boxed index case %zu: %s\n", i, error);
        if (source) {
            int status = 0;
            CHECK(compile_and_run(source, &status) == 0, "I compile a boxed array-index caller");
            CHECK(cases[i].succeeds ? status == 0 : status != 0,
                  "I preserve aliased writes and reject noninteger or out-of-range indices");
            free(source);
        }
        nvm_module_free(module);
    }
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
        "  CAST_INT\n"
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
        "  CAST_INT\n"
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

static void test_cast_int_values(void) {
    const char *source =
        ".string number \"  -42tail\"\n.string empty \"\"\n.string invalid \"no number\"\n"
        ".string maximum \"9223372036854775807\"\n.string minimum \"-9223372036854775808\"\n"
        ".string overflow \"99999999999999999999999999999\"\n"
        ".string underflow \"-99999999999999999999999999999\"\n.entry main\n"
        ".function identity 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".function main 0 0 0 int 1\n"
        "PUSH_STR number\nCAST_INT\nCALL identity\nPUSH_I64 -42\nI64_EQ\nASSERT\n"
        "PUSH_I64 42\nCAST_INT\nCALL identity\nPUSH_I64 42\nI64_EQ\nASSERT\n"
        "PUSH_BOOL 1\nCAST_INT\nPUSH_I64 1\nI64_EQ\nASSERT\n"
        "PUSH_STR empty\nCAST_INT\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "PUSH_STR invalid\nCAST_INT\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "PUSH_STR maximum\nCAST_INT\nPUSH_I64 9223372036854775807\nI64_EQ\nASSERT\n"
        "PUSH_STR minimum\nCAST_INT\nPUSH_I64 -9223372036854775808\nI64_EQ\nASSERT\n"
        "PUSH_STR overflow\nCAST_INT\nPUSH_I64 9223372036854775807\nI64_EQ\nASSERT\n"
        "PUSH_STR underflow\nCAST_INT\nPUSH_I64 -9223372036854775808\nI64_EQ\nASSERT\n"
        "ARR_NEW 1\nCAST_INT\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "ARR_NEW 5\nCAST_INT\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "ARR_NEW 8\nCAST_INT\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "PUSH_I64 7\nAGG_PACK 0 0 0 1\nCAST_INT\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "CAST_INT values and mixed callers");
    if (!m) return;
    char *c = emit_or_fail(m, "I infer CAST_INT results as integers before calls");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0 && status == 0,
              "I preserve decimal conversion, scalar identity and aggregate-zero CAST_INT behavior");
        free(c);
    }
    nvm_module_free(m);
}

static void test_cast_string_array_is_refused(void) {
    test_cast_int_values();
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
        "  CAST_INT\n"
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
        "  CAST_INT\n"
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
        "  CAST_INT\n"
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
        "  CAST_INT\n"
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
        "  CAST_INT\n"
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
        ".function miss 0 0 0 bool 1\n"
        "  PUSH_STR hi\n"
        "  PUSH_I64 9\n"
        "  STR_CHAR_AT\n"
        "  PUSH_I64 0\n"
        "  I64_LT_S\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  CALL miss\n"
        "  CAST_INT\n"
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
    CHECK(strstr(c, "nrec_t *data; size_t len; struct nrarr_owner *owner;") != NULL,
          "grow_t C stores dynamically allocated record elements by value");
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

static void test_nested_record_values(void) {
    const char *returned = ".string first \"hello\"\n.string second \"replacement\"\n.entry main\n"
        ".function make 2 2 0 struct 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nAGG_PACK 0 0 0 2\n"
        "AGG_PACK 0 0 0 1\nAGG_PACK 0 0 0 1\nRET\n.end\n"
        ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".function main 0 3 0 int 1\nPUSH_STR first\nPUSH_I64 42\nCALL make\nCALL relay\nSTORE_LOCAL 0\n"
        "PUSH_STR second\nPUSH_I64 17\nCALL make\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nAGG_GET 0\nSTORE_LOCAL 2\nLOAD_LOCAL 2\nSTR_LEN\n"
        "PUSH_I64 5\nI64_EQ\nASSERT\nLOAD_LOCAL 1\nAGG_GET 0\nAGG_GET 0\nAGG_GET 1\n"
        "PUSH_I64 17\nI64_EQ\nASSERT\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nAGG_GET 1\nRET\n.end\n";
    NvmModule *returned_module = assemble_ok(returned, "returned nested snapshots");
    if (returned_module) {
        char *output = emit_or_fail(returned_module, "I translate returned nested snapshots");
        if (output) {
            int status = -1;
            CHECK(compile_and_run(output, &status) == 0 && status == 42,
                  "I preserve earlier nested values after a second constructor call");
            free(output);
        }
        nvm_module_free(returned_module);
    }
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
        "  AGG_GET 1\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "nested record fixture");
    CHECK(m != NULL, "nested record fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "I emit nested record values");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0 && status == 2, "I extract nested record fields");
    }
    free(c);
    nvm_module_free(m);
}

static void test_unsupported_classifier_instructions(void) {
    const uint8_t opcodes[] = {OP_HM_KEYS, OP_HM_VALUES,
        OP_STR_TO_UPPER, OP_ROLL};
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

    NvmModule *malformed = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\n"
        "NOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\nNOP\n"
        "PUSH_I64 0\nRET\n.end\n", "malformed indirect call");
    if (malformed) {
        DecodedInstruction indirect = {0};
        indirect.opcode = OP_CALL_INDIRECT;
        indirect.operands[0].u16 = 1;
        indirect.operands[1].u16 = 1;
        CHECK(isa_encode(&indirect, malformed->code + malformed->functions[0].code_offset, 16) != 0,
              "I encode a malformed indirect call using ISA metadata");
        char error[256] = {0};
        char *c = nvm2c_emit(malformed, error, sizeof error);
        CHECK(c == NULL && strstr(error, "CALL_INDIRECT has an unsupported stack shape"),
              "I reject an indirect call without its exact arguments and callable");
        free(c);
        nvm_module_free(malformed);
    }

    const char *exact =
        ".entry main\n"
        ".function identity 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".parameters identity int\n"
        ".function main 0 0 0 int 1\nPUSH_I64 42\nFUNCREF identity\n"
        "CALL_INDIRECT 1 1\nRET\n.end\n";
    NvmModule *exact_module = assemble_ok(exact, "exact scalar indirect call");
    if (exact_module) {
        char *c = emit_or_fail(exact_module, "I emit an exact scalar indirect call");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 42,
                  "I execute an exact scalar indirect call");
            free(c);
        }
        nvm_module_free(exact_module);
    }
}

static void test_exact_aggregate_callback_provenance(void) {
    const char *main_fn =
        ".function main 0 1 0 int 1\nCALL choose\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nCALL forward\nAGG_GET 0\nPUSH_I64 35\nI64_ADD\nRET\n.end\n";
    const char *workers =
        ".function choose 0 0 0 function 1\nPUSH_I64 11\nPRINTLN\nFUNCREF make\nRET\n.end\n"
        ".function forward 1 1 0 union 1\nLOAD_LOCAL 0\nTAIL_CALL apply\n.end\n"
        ".parameters forward function\n"
        ".function apply 1 1 0 union 1\nLOAD_LOCAL 0\nCALL_INDIRECT 0 1\nRET\n.end\n"
        ".parameters apply function\n"
        ".function make 0 0 0 union 1\nPUSH_I64 22\nPRINTLN\n"
        "PUSH_I64 7\nAGG_PACK 1 0 0 1\nRET\n.end\n"
        ".function decoy 0 0 0 float 1\nPUSH_F64 9.5\nRET\n.end\n";
    for (int before = 0; before < 2; ++before) {
        char source[2048];
        snprintf(source, sizeof source, ".types 0 0 1\n.entry main\n%s%s",
                 before ? workers : main_fn, before ? main_fn : workers);
        NvmModule *m = assemble_ok(source, "returned and forwarded exact aggregate callback");
        if (!m) continue;
        char *c = emit_or_fail(m, "I follow the actual aggregate target instead of an unrelated scalar decoy");
        if (c) {
            char output[128];
            int status = -1;
            CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 &&
                  status == 42 && strcmp(output, "11\n22\n") == 0,
                  "I evaluate the selector and selected aggregate callback once in order");
            free(c);
        }
        nvm_module_free(m);
    }

    NvmModule *wrong = assemble_ok(
        ".entry main\n.string text \"not an array\"\n"
        ".function main 0 0 0 int 1\nPUSH_STR text\nFUNCREF identity\n"
        "CALL_INDIRECT 1 1\nPOP\nPUSH_I64 0\nRET\n.end\n"
        ".function identity 1 1 0 array 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".parameters identity array\n", "exact target with an incompatible argument tag");
    if (wrong) {
        char error[512];
        char *c = nvm2c_emit(wrong, error, sizeof error);
        CHECK(c == NULL && strstr(error, "matching aggregate callback argument tags"),
              "I do not use target identity to erase a callback argument mismatch");
        free(c);
        nvm_module_free(wrong);
    }
}

static void test_indirect_target_inference_order(void) {
    const char *functions[] = {
        ".function main 0 0 0 int 1\nPUSH_I64 42\nAGG_PACK 0 0 0 1\n"
        "FUNCREF read\nCALL apply\nRET\n.end\n",
        ".function apply 2 2 0 int 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n"
        "CALL_INDIRECT 1 1\nRET\n.end\n.parameters apply struct function\n",
        ".function read 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n"
        ".parameters read struct\n"
    };
    const unsigned orders[][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
    for (size_t order = 0; order < sizeof orders / sizeof orders[0]; ++order) {
        char source[2048];
        snprintf(source, sizeof source, ".types 1 0 0\n.entry main\n%s%s%s",
                 functions[orders[order][0]], functions[orders[order][1]],
                 functions[orders[order][2]]);
        NvmModule *m = assemble_ok(source, "indirect record argument in every function order");
        if (!m) continue;
        char *c = emit_or_fail(m, "I converge indirect target facts independently of function order");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 42,
                  "I execute the same record callback in every function order");
            free(c);
        }
        nvm_module_free(m);
    }

    NvmModule *missing = assemble_ok(
        ".entry main\n.string text \"wrong argument\"\n"
        ".function main 0 0 0 int 1\nPUSH_STR text\nFUNCREF identity\n"
        "CALL_INDIRECT 1 1\nRET\n.end\n"
        ".function identity 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".parameters identity int\n", "unresolved indirect target after convergence");
    if (missing) {
        char error[512];
        char *c = nvm2c_emit(missing, error, sizeof error);
        CHECK(c == NULL && strstr(error, "CALL_INDIRECT has no exact scalar target"),
              "I still refuse a missing target after callback facts converge");
        free(c);
        nvm_module_free(missing);
    }
}

static void test_array_result_kinds(void) {
    const int tags[] = {1, 5, 8};
    const char *elements[] = {"PUSH_I64 42", "PUSH_STR text", "PUSH_I64 42\nAGG_PACK 0 0 0 1"};
    const char *checks[] = {"PUSH_I64 42\nI64_EQ", "PUSH_STR text\nEQ", "AGG_GET 0\nPUSH_I64 42\nI64_EQ"};
    for (int kind = 0; kind < 3; ++kind) {
        for (int tail = 0; tail < 2; ++tail) {
            for (int before = 0; before < 2; ++before) {
                char body[768], workers[1536], source[8192];
                snprintf(body, sizeof body,
                    ".function main 0 1 0 int 1\nPUSH_I64 3\nCALL recur\nCALL identity\nSTORE_LOCAL 0\n"
                    "LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n"
                    "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n%s\nASSERT\n"
                    "CALL empty\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n", checks[kind]);
                snprintf(workers, sizeof workers,
                    ".function recur 1 1 0 array 1\nLOAD_LOCAL 0\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n"
                    "%s make\n%sagain:\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_SUB\n%s recur\n%s.end\n"
                    ".function make 0 0 0 array 1\n%s\nARR_LITERAL %d 1\nRET\n.end\n"
                    ".function empty 0 0 0 array 1\nARR_NEW %d\nRET\n.end\n"
                    ".function identity 1 1 0 array 1\nLOAD_LOCAL 0\nRET\n.end\n",
                    tail ? "TAIL_CALL" : "CALL", tail ? "" : "RET\n",
                    tail ? "TAIL_CALL" : "CALL", tail ? "" : "RET\n", elements[kind], tags[kind], tags[kind]);
                snprintf(source, sizeof source, ".string text \"forty-two\"\n.entry main\n%s%s",
                         before ? workers : body, before ? body : workers);
                NvmModule *m = assemble_ok(source, "array result element kinds");
                if (!m) continue;
                char *c = emit_or_fail(m, "I infer array result kinds through forward and recursive calls");
                if (c) {
                    int status = -1;
                    CHECK(compile_and_run(c, &status) == 0 && status == 0,
                          "I return and index scalar and record arrays through ordinary and tail recursion");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
    NvmModule *m = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nCALL mixed\nPOP\nPUSH_I64 0\nRET\n.end\n"
        ".function mixed 1 1 0 array 1\nLOAD_LOCAL 0\nJMP_FALSE other\nARR_NEW 1\nRET\n"
        "other:\nARR_NEW 5\nRET\n.end\n", "incompatible array return paths");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "array result"), "I reject conflicting native array result representations explicitly");
        free(c); nvm_module_free(m);
    }
}

static void test_optional_record_arguments(void) {
    test_array_result_kinds();
    for (int tail = 0; tail < 2; ++tail) {
        for (int wrong = 0; wrong < 2; ++wrong) {
            char source[3072];
            snprintf(source, sizeof source,
                ".string text \"kept\"\n.entry main\n"
                ".function main 0 0 0 int 1\nPUSH_BOOL 1\nCALL choose\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n"
                "PUSH_BOOL 0\nCALL choose\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nTYPE_CHECK 0\nASSERT\nPUSH_I64 0\nRET\n.end\n"
                ".function choose 1 1 0 array 1\nLOAD_LOCAL 0\nJMP_FALSE missing\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nRET\n"
                "missing:\n%s\n.end\n"
                ".function absent 0 0 0 array 1\nHM_NEW 5 %d\nPUSH_STR text\nHM_GET\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nRET\n.end\n",
                tail ? "TAIL_CALL absent" : "CALL absent\nRET", wrong ? 1 : 5);
            NvmModule *m = assemble_ok(source, "optional record array returns");
            if (!m) continue;
            char err[512];
            char *c = nvm2c_emit(m, err, sizeof err);
            CHECK((c != NULL) == !wrong, "I reconcile array return fields only for compatible payloads");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I preserve present and missing array fields across ordinary and tail returns");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    for (int reverse = 0; reverse < 2; ++reverse) {
        for (int before = 0; before < 2; ++before) {
            char body[2048], source[8192];
            const char *worker = ".function inspect 1 1 0 int 1\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\n"
                "DUP\nTYPE_CHECK 0\nJMP_FALSE present\nPOP\nPUSH_I64 0\nRET\npresent:\nTYPE_CHECK 5\nASSERT\nPUSH_I64 5\nRET\n.end\n";
            const char *plain = "LOAD_LOCAL 0\nCALL inspect\nPUSH_I64 5\nEQ\nASSERT\n";
            const char *missing = "LOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nCALL inspect\nPUSH_I64 0\nEQ\nASSERT\n";
            snprintf(body, sizeof body,
                ".function main 0 1 0 int 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n%s%s"
                "PUSH_STR text\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nCALL inspect\nPUSH_I64 5\nEQ\nASSERT\n"
                "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nSTR_LEN\nPUSH_I64 4\nEQ\nASSERT\n"
                "PUSH_I64 0\nRET\n.end\n", reverse ? missing : plain, reverse ? plain : missing);
            snprintf(source, sizeof source, ".string text \"kept\"\n.entry main\n%s%s",
                     before ? worker : body, before ? body : worker);
            NvmModule *m = assemble_ok(source, "optional record array arguments");
            if (!m) continue;
            char *c = emit_or_fail(m, "I preserve tagged record fields in array arguments");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I observe present and absent array fields without rewriting source storage");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    for (int tail = 0; tail < 2; ++tail) {
        for (int reverse = 0; reverse < 2; ++reverse) {
            for (int before = 0; before < 2; ++before) {
                const char *plain = "LOAD_LOCAL 0\nCALL relay\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n";
                const char *missing = "LOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nCALL relay\nAGG_GET 0\nTYPE_CHECK 0\nASSERT\n";
                char body[2048], workers[512], source[8192];
                snprintf(body, sizeof body,
                    ".function main 0 1 0 int 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n%s%s"
                    "PUSH_STR text\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nAGG_PACK 0 0 0 1\nCALL relay\n"
                    "AGG_GET 0\nPUSH_STR text\nEQ\nASSERT\n"
                    "LOAD_LOCAL 0\nAGG_GET 0\nSTR_LEN\nPUSH_I64 4\nI64_EQ\nASSERT\n"
                    "PUSH_I64 0\nRET\n.end\n", reverse ? missing : plain, reverse ? plain : missing);
                snprintf(workers, sizeof workers,
                    ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\n%s\n.end\n"
                    ".function identity 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n",
                    tail ? "TAIL_CALL identity" : "CALL identity\nRET");
                snprintf(source, sizeof source, ".string text \"kept\"\n.entry main\n%s%s",
                         before ? workers : body, before ? body : workers);
                NvmModule *m = assemble_ok(source, "mixed record argument fields");
                if (!m) continue;
                char *c = emit_or_fail(m, "I reconcile ordinary and tagged record argument fields independent of ordering");
                if (c) {
                    int status = -1;
                    CHECK(compile_and_run(c, &status) == 0 && status == 0,
                          "I preserve present and missing fields through calls and leave the caller record unchanged");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
    NvmModule *m = assemble_ok(
        ".string text \"key\"\n.entry main\n.function main 0 0 0 int 1\n"
        "PUSH_STR text\nAGG_PACK 0 0 0 1\nCALL inspect\nPOP\n"
        "HM_NEW 5 1\nPUSH_STR text\nHM_GET\nAGG_PACK 0 0 0 1\nCALL inspect\nPOP\n"
        "PUSH_I64 0\nRET\n.end\n"
        ".function inspect 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nCAST_INT\nRET\n.end\n",
        "incompatible record argument payload");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I still reject incompatible optional payload constraints at record calls");
        free(c); nvm_module_free(m);
    }
}

static void test_unresolved_local_storage(void) {
    /* I exercise a nested field whose flat kind is still unknown when stored.
     * Its second call observes the first call's tagged local requirement;
     * returning the container must not impose that requirement on its field. */
    const char *functions[] = {
        ".function main 0 1 0 int 1\n"
        "PUSH_STR text\n"
        "AGG_PACK 0 0 0 1\n"
        "ARR_LITERAL 8 1\n"
        "AGG_PACK 0 0 0 1\n"
        "STORE_LOCAL 0\n"
        "LOAD_LOCAL 0\n"
        "CALL project\n"
        "STORE_LOCAL 0\n"
        "LOAD_LOCAL 0\n"
        "CALL rewrite\n"
        "POP\n"
        "PUSH_STR text\n"
        "ARR_LITERAL 5 1\n"
        "PUSH_I64 0\n"
        "ARR_GET\n"
        "CALL length\n"
        "POP\n"
        "PUSH_I64 0\n"
        "RET\n"
        ".end\n",
        ".function project 1 3 0 struct 1\n"
        "LOAD_LOCAL 0\n"
        "AGG_GET 0\n"
        "PUSH_I64 0\n"
        "ARR_GET\n"
        "AGG_GET 0\n"
        "STORE_LOCAL 1\n"
        "LOAD_LOCAL 1\n"
        "CALL length\n"
        "POP\n"
        "LOAD_LOCAL 1\n"
        "CALL length\n"
        "PUSH_I64 4\n"
        "EQ\n"
        "ASSERT\n"
        "LOAD_LOCAL 0\n"
        "RET\n"
        ".end\n",
        ".function rewrite 1 1 0 struct 1\n"
        "LOAD_LOCAL 0\n"
        "AGG_GET 0\n"
        "PUSH_I64 0\n"
        "PUSH_STR text\n"
        "CALL identity\n"
        "AGG_PACK 0 0 0 1\n"
        "ARR_SET\n"
        "AGG_PACK 0 0 0 1\n"
        "RET\n"
        ".end\n",
        ".function length 1 1 0 int 1\n"
        "LOAD_LOCAL 0\n"
        "STR_LEN\n"
        "RET\n"
        ".end\n",
        ".function identity 1 1 0 string 1\n"
        "LOAD_LOCAL 0\n"
        "RET\n"
        ".end\n"
    };
    const unsigned orders[][5] = {
        {0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}, {1, 0, 3, 4, 2},
        {2, 4, 0, 3, 1}, {3, 2, 1, 0, 4}, {4, 1, 3, 2, 0}
    };
    for (size_t order = 0; order < sizeof orders / sizeof orders[0]; ++order) {
        char source[4096] = ".string text \"kept\"\n.entry main\n";
        for (unsigned i = 0; i < 5; ++i)
            strcat(source, functions[orders[order][i]]);
        NvmModule *m = assemble_ok(source, "unresolved local and exact producer storage");
        if (!m) continue;
        char *c = emit_or_fail(m, "I keep a tagged local separate from its unresolved producer");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I preserve exact record fields after repeated scalar calls and container returns");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_projected_string_call_storage(void) {
    test_unresolved_local_storage();
    for (int boxed = 0; boxed < 2; ++boxed) {
        for (int reverse = 0; reverse < 2; ++reverse) {
            for (int tail = 0; tail < 2; ++tail) {
                for (int before = 0; before < 2; ++before) {
                    char body[3072], workers[768], source[8192], nested[1024];
                    const char *value = boxed ? "LOAD_GLOBAL 0\n" : "LOAD_LOCAL 1\nPUSH_STR key\nHM_GET\n";
                    const char *plain = "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nCALL length\nPUSH_I64 4\nEQ\nASSERT\n";
                    snprintf(nested, sizeof nested,
                        "%sAGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\nCALL project\nPUSH_I64 4\nEQ\nASSERT\n", value);
                    snprintf(body, sizeof body,
                        ".function main 0 2 0 int 1\nPUSH_STR text\nARR_LITERAL 5 1\nSTORE_LOCAL 0\n"
                        "PUSH_STR text\nSTORE_GLOBAL 0\nHM_NEW 5 5\nSTORE_LOCAL 1\n"
                        "LOAD_LOCAL 1\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPOP\n%s%s"
                        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                        reverse ? nested : plain, reverse ? plain : nested);
                    snprintf(workers, sizeof workers,
                        ".function project 1 2 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\n"
                        "LOAD_LOCAL 1\nAGG_GET 0\n%s length\n%s.end\n"
                        ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nSTR_LEN\nRET\n.end\n",
                        tail ? "TAIL_CALL" : "CALL", tail ? "" : "RET\n");
                    snprintf(source, sizeof source, ".string text \"kept\"\n.string key \"key\"\n.entry main\n%s%s",
                             before ? workers : body, before ? body : workers);
                    NvmModule *m = assemble_ok(source, "projected string parameter storage");
                    if (!m) continue;
                    char *c = emit_or_fail(m, "I separate parameter storage from exact string producers");
                    if (c) {
                        int status = -1;
                        CHECK(compile_and_run(c, &status) == 0 && status == 0,
                              "I accept projected tagged strings independent of caller, tail-call and function order");
                        free(c);
                    }
                    nvm_module_free(m);
                }
            }
        }
    }
    const char *invalid[] = {"PUSH_I64 7", "PUSH_BOOL 0", "LOAD_GLOBAL 1", "ARR_NEW 1",
        "HM_NEW 5 5\nPUSH_STR key\nHM_GET"};
    for (size_t v = 0; v < sizeof invalid / sizeof invalid[0]; ++v) {
        for (int taken = 0; taken < 2; ++taken) {
            char source[3072];
            snprintf(source, sizeof source,
                ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n"
                "%s\nSTORE_GLOBAL 0\nPUSH_BOOL %d\nJMP_FALSE done\nLOAD_GLOBAL 0\n"
                "AGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\nCALL project\nPOP\n"
                "done:\nPUSH_I64 0\nRET\n.end\n"
                ".function project 1 2 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\n"
                "LOAD_LOCAL 1\nAGG_GET 0\nCALL length\nRET\n.end\n"
                ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nSTR_LEN\nRET\n.end\n", invalid[v], taken);
            NvmModule *m = assemble_ok(source, "checked projected string consumption");
            if (!m) continue;
            char *c = emit_or_fail(m, "I retain runtime checks for a projected string parameter");
            if (c) {
                int status = 0;
                CHECK(compile_and_run(c, &status) == 0 && (taken ? status != 0 : status == 0),
                      "I reject wrong or absent tags only when the string consumer executes");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    NvmModule *m = assemble_ok(
        ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n"
        "HM_NEW 5 1\nPUSH_STR key\nHM_GET\nAGG_PACK 0 0 0 1\nAGG_PACK 1 0 0 1\nCALL project\nRET\n.end\n"
        ".function project 1 2 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nAGG_GET 0\nCALL length\nRET\n.end\n"
        ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nSTR_LEN\nRET\n.end\n",
        "exact projected payload conflict");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I keep exact optional payload conflicts rejected");
        free(c); nvm_module_free(m);
    }
}

static void test_optional_record_results(void) {
    test_optional_record_arguments();
    const char *main_body =
        ".function main 0 3 0 int 1\nHM_NEW 5 5\nSTORE_LOCAL 0\nPUSH_STR text\nSTORE_LOCAL 2\n"
        "PUSH_BOOL 0\nLOAD_LOCAL 0\nLOAD_LOCAL 2\nCALL choose\nAGG_GET 0\nTYPE_CHECK 0\nASSERT\n"
        "PUSH_BOOL 1\nLOAD_LOCAL 0\nLOAD_LOCAL 2\nCALL choose\nAGG_GET 0\nDUP\nTYPE_CHECK 5\nASSERT\n"
        "PUSH_STR text\nEQ\nASSERT\nLOAD_LOCAL 2\nSTR_LEN\nPUSH_I64 2\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPOP\n"
        "PUSH_BOOL 0\nLOAD_LOCAL 0\nLOAD_LOCAL 2\nCALL choose\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 0\nPUSH_STR key\nHM_DELETE\nPOP\nLOAD_LOCAL 1\nAGG_GET 0\n"
        "PUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n";
    for (int tail = 0; tail < 2; ++tail) {
        for (int before = 0; before < 2; ++before) {
            char workers[1024], source[4096];
            snprintf(workers, sizeof workers,
                ".function choose 3 3 0 struct 1\nLOAD_LOCAL 0\nJMP_FALSE lookup\nLOAD_LOCAL 2\n%s"
                "lookup:\nLOAD_LOCAL 1\nPUSH_STR key\nHM_GET\nAGG_PACK 0 0 0 1\nRET\n.end\n%s",
                tail ? "TAIL_CALL plain\n" : "AGG_PACK 0 0 0 1\nRET\n",
                tail ? ".function plain 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n" : "");
            snprintf(source, sizeof source, ".string key \"key\"\n.string text \"42\"\n.entry main\n%s%s",
                     before ? workers : main_body, before ? main_body : workers);
            NvmModule *m = assemble_ok(source, "optional record result conversion");
            if (!m) continue;
            char *c = emit_or_fail(m, "I join ordinary and optional returned fields without changing source strings");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I preserve void and present tags across ordinary and tail record returns");
                CHECK(strstr(c, "const char * a2") != NULL,
                      "I keep the original string parameter in ordinary string storage");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    NvmModule *m = assemble_ok(
        ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n"
        "PUSH_BOOL 0\nCALL choose\nPOP\nPUSH_I64 0\nRET\n.end\n"
        ".function choose 1 1 0 struct 1\nLOAD_LOCAL 0\nJMP_FALSE lookup\n"
        "PUSH_STR key\nAGG_PACK 0 0 0 1\nRET\nlookup:\nHM_NEW 5 1\nPUSH_STR key\nHM_GET\n"
        "AGG_PACK 0 0 0 1\nRET\n.end\n", "conflicting optional result payload");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I reject incompatible payloads in optional record results");
        free(c); nvm_module_free(m);
    }
}

static void test_tagged_record_fields(void) {
    test_projected_string_call_storage();
    test_optional_record_results();
    for (int strings = 0; strings < 2; ++strings) {
        char source[4096];
        snprintf(source, sizeof source,
            ".string key \"key\"\n.string text \"42\"\n.entry main\n"
            ".function main 0 3 0 int 1\nHM_NEW 5 %d\nSTORE_LOCAL 0\n"
            "ARR_NEW 8\nLOAD_LOCAL 0\nCALL relay\nARR_PUSH\nSTORE_LOCAL 1\n"
            "LOAD_LOCAL 0\nPUSH_STR key\n%s\nHM_SET\nPOP\n"
            "LOAD_LOCAL 1\nLOAD_LOCAL 0\nCALL relay\nARR_PUSH\nPOP\n"
            "LOAD_LOCAL 0\nPUSH_STR key\nHM_DELETE\nPOP\n"
            "LOAD_LOCAL 1\nCALL pass_array\nSTORE_LOCAL 1\n"
            "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nAGG_GET 1\nTYPE_CHECK 0\nASSERT\n"
            "LOAD_LOCAL 1\nPUSH_I64 1\nARR_GET\nSTORE_LOCAL 2\n"
            "LOAD_LOCAL 2\nAGG_GET 0\nAGG_GET 0\nPUSH_I64 7\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 0\nAGG_GET 1\nDUP\nTYPE_CHECK %d\nASSERT\n"
            "CAST_INT\nPUSH_I64 42\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n"
            ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nTAIL_CALL pack\n.end\n"
            ".function pack 1 1 0 struct 1\nPUSH_I64 7\nLOAD_LOCAL 0\nPUSH_STR key\nHM_GET\n"
            "AGG_PACK 0 0 0 2\nAGG_PACK 0 0 0 1\nRET\n.end\n"
            ".function pass_array 1 1 0 array 1\nLOAD_LOCAL 0\nRET\n.end\n",
            strings ? 5 : 1, strings ? "PUSH_STR text" : "PUSH_I64 42", strings ? 5 : 1);
        NvmModule *m = assemble_ok(source, "nested tagged record fields");
        if (!m) continue;
        char *c = emit_or_fail(m, "I preserve tagged fields through nested records and returned record arrays");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I retain missing tags and fetched payloads in nested records after map deletion");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *mixed = ".string key \"key\"\n.entry main\n"
        ".function main 0 0 0 int 1\nPUSH_STR key\nAGG_PACK 0 0 0 1\nCALL consume\nPOP\n"
        "HM_NEW 5 5\nPUSH_STR key\nHM_GET\nAGG_PACK 0 0 0 1\nCALL consume\nRET\n.end\n"
        ".function consume 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nCAST_STRING\nSTR_LEN\nRET\n.end\n";
    NvmModule *m = assemble_ok(mixed, "mixed ordinary and optional record fields");
    if (m) {
        char *c = emit_or_fail(m, "I preserve tags in mixed record parameter fields");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I consume missing and present fields without changing their tags");
        }
        free(c); nvm_module_free(m);
    }
}

static void test_mixed_lookup_arguments(void) {
    test_tagged_record_fields();
    const char *workers =
        ".function consume 1 1 0 int 1\nLOAD_LOCAL 0\nTYPE_CHECK 0\nJMP_FALSE present\n"
        "PUSH_I64 0\nRET\npresent:\nLOAD_LOCAL 0\nCAST_STRING\nSTR_LEN\nRET\n.end\n"
        ".function forward 1 1 0 int 1\nLOAD_LOCAL 0\nTAIL_CALL consume\n.end\n"
        ".function recurse 2 2 0 int 1\nLOAD_LOCAL 1\nPUSH_I64 0\nI64_EQ\nJMP_FALSE again\n"
        "LOAD_LOCAL 0\nTAIL_CALL forward\nagain:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_SUB\nTAIL_CALL recurse\n.end\n";
    const char *raw = "PUSH_STR text\nCALL consume\nPUSH_I64 2\nI64_EQ\nASSERT\n";
    const char *missing = "LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nCALL consume\nPUSH_I64 0\nI64_EQ\nASSERT\n";
    for (int before = 0; before < 2; ++before) {
        for (int raw_first = 0; raw_first < 2; ++raw_first) {
            char main_body[2048], source[8192];
            snprintf(main_body, sizeof main_body,
                ".function main 0 1 0 int 1\nHM_NEW 5 5\nSTORE_LOCAL 0\n%s%s"
                "LOAD_LOCAL 0\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPUSH_STR key\nHM_GET\n"
                "PUSH_I64 3\nCALL recurse\nPUSH_I64 2\nI64_EQ\nASSERT\n"
                "PUSH_STR text\nPUSH_I64 2\nCALL recurse\nPUSH_I64 2\nI64_EQ\nASSERT\n"
                "LOAD_LOCAL 0\nPUSH_STR key\nHM_DELETE\nPUSH_STR key\nHM_GET\n"
                "PUSH_I64 1\nCALL recurse\nPUSH_I64 0\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                raw_first ? raw : missing, raw_first ? missing : raw);
            snprintf(source, sizeof source, ".string key \"key\"\n.string text \"42\"\n.entry main\n%s%s",
                     before ? workers : main_body, before ? main_body : workers);
            NvmModule *m = assemble_ok(source, "mixed lookup argument order");
            if (!m) continue;
            char *c = emit_or_fail(m, "I infer mixed string arguments independently of function and caller order");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I preserve missing and present strings across mixed normal, forward and tail calls");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    NvmModule *m = assemble_ok(
        ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n"
        "HM_NEW 5 1\nPUSH_STR key\nHM_GET\nCALL consume\nPOP\n"
        "PUSH_STR key\nCALL consume\nRET\n.end\n"
        ".function consume 1 1 0 int 1\nLOAD_LOCAL 0\nCAST_INT\nRET\n.end\n",
        "incompatible optional argument payload");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I keep optional payload compatibility after parameter widening");
        free(c); nvm_module_free(m);
    }
}

static void test_tagged_scalar_returns(void) {
    const char *types[] = {"int", "string", "bool"};
    for (int type = 0; type < 3; ++type) {
        for (int value = 0; value < 3; ++value) {
            for (int present_path = 0; present_path < 2; ++present_path) {
                char source[2048];
                snprintf(source, sizeof source,
                    ".string key \"key\"\n.string text \"42\"\n.entry main\n"
                    ".function main 0 0 0 int 1\nCALL relay\n%s\n"
                    "PUSH_I64 0\nRET\n.end\n"
                    ".function relay 0 0 0 %s 1\nTAIL_CALL result\n.end\n"
                    ".function result 0 1 0 %s 1\nPUSH_BOOL %d\nJMP_FALSE lookup\n"
                    "%s\nRET\nlookup:\nHM_NEW 5 %d\n%s\n"
                    "PUSH_STR key\nHM_GET\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nRET\n.end\n",
                    type == 0 ? "PUSH_I64 42\nI64_EQ\nASSERT" :
                    type == 1 ? "PUSH_STR text\nEQ\nASSERT" : "ASSERT",
                    types[type], types[type], present_path,
                    type == 0 ? "PUSH_I64 42" : type == 1 ? "PUSH_STR text" : "PUSH_BOOL 1",
                    value == 2 ? 5 : 1,
                    value == 0 ? "" : value == 1 ? "PUSH_STR key\nPUSH_I64 42\nHM_SET" :
                    "PUSH_STR key\nPUSH_STR text\nHM_SET");
                NvmModule *m = assemble_ok(source, "tagged scalar return boundary");
                if (!m) continue;
                char *c = emit_or_fail(m, "I emit checked tagged scalar returns without changing source storage");
                if (c) {
                    int status = 0;
                    int succeeds = present_path || (type == 0 && value == 1) || (type == 1 && value == 2);
                    CHECK(compile_and_run(c, &status) == 0 &&
                          (succeeds ? status == 0 : status == -1),
                          "I return matching tags and trap missing or wrong tags through ordinary and tail callers");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
}

static void test_tagged_array_read_bounds(void) {
    const struct { int tag; const char *value, *check; } arrays[] = {
        {1, "PUSH_I64 7", "PUSH_I64 7\nEQ\nASSERT"},
        {4, "PUSH_BOOL 1", "ASSERT"},
        {5, "PUSH_STR text", "PUSH_STR text\nEQ\nASSERT"}
    };
    const int64_t indices[] = {0, -1, 1, 99, INT64_C(4294967296), INT64_MAX, INT64_MIN};
    for (size_t a = 0; a < sizeof arrays / sizeof arrays[0]; ++a) {
        for (unsigned test = 0; test < 12; ++test) {
            char source[2048], index[128];
            if (test < 7) snprintf(index, sizeof index, "PUSH_I64 %lld", (long long)indices[test]);
            else if (test == 7) strcpy(index, "PUSH_I64 0");
            else if (test == 8) strcpy(index, "PUSH_BOOL 0");
            else if (test == 9) strcpy(index, "PUSH_STR text");
            else if (test == 10) strcpy(index, "LOAD_GLOBAL 2");
            else strcpy(index, "ARR_NEW 1");
            snprintf(source, sizeof source,
                ".string text \"kept\"\n.entry main\n.function main 0 0 0 int 1\n"
                "%s\nARR_LITERAL %d %d\nSTORE_GLOBAL 0\n%s\nSTORE_GLOBAL 1\n"
                "LOAD_GLOBAL 0\nLOAD_GLOBAL 1\nARR_GET\n%s\nPUSH_I64 0\nRET\n.end\n",
                test == 7 ? "" : arrays[a].value, arrays[a].tag, test == 7 ? 0 : 1, index,
                test == 0 ? arrays[a].check : "TYPE_CHECK 0\nASSERT");
            NvmModule *m = assemble_ok(source, "tagged array read bounds");
            if (!m) continue;
            char *c = emit_or_fail(m, "I translate full-width tagged array reads");
            if (c) {
                int status = 0;
                CHECK(compile_and_run(c, &status) == 0 && (test < 8 ? status == 0 : status != 0),
                      "I preserve valid/missing reads and reject noninteger index tags");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    for (unsigned test = 0; test < 2; ++test) {
        char source[1024];
        snprintf(source, sizeof source,
            ".string text \"receiver\"\n.entry main\n.function main 0 0 0 int 1\n"
            "PUSH_STR text\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 %d\nARR_GET\nPOP\nPUSH_I64 0\nRET\n.end\n", test ? -1 : 0);
        NvmModule *m = assemble_ok(source, "tagged invalid array receiver");
        if (!m) continue;
        char *c = emit_or_fail(m, "I retain dynamic array receiver checks before missing-read handling");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && status != 0,
                  "I reject nonarray receivers even when the index is negative");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_tagged_array_update_bounds(void) {
    const struct { int tag; const char *before, *after, *check; } arrays[] = {
        {1, "PUSH_I64 1", "PUSH_I64 2", "PUSH_I64 2\nEQ\nASSERT"},
        {4, "PUSH_BOOL 0", "PUSH_BOOL 1", "ASSERT"},
        {5, "PUSH_STR before", "PUSH_STR after", "PUSH_STR after\nEQ\nASSERT"}
    };
    const int64_t indices[] = {0, -1, 1, 99, INT64_C(4294967296), INT64_MAX, INT64_MIN};
    for (size_t a = 0; a < sizeof arrays / sizeof arrays[0]; ++a) {
        for (size_t i = 0; i < sizeof indices / sizeof indices[0]; ++i) {
            char source[2048];
            snprintf(source, sizeof source,
                ".string before \"before\"\n.string after \"after\"\n.entry main\n"
                ".function main 0 1 0 int 1\n%s\nARR_LITERAL %d 1\nSTORE_LOCAL 0\n"
                "LOAD_LOCAL 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 %lld\n%s\nARR_SET\nPOP\n"
                "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n%s\nPUSH_I64 0\nRET\n.end\n",
                arrays[a].before, arrays[a].tag, (long long)indices[i], arrays[a].after, arrays[a].check);
            NvmModule *m = assemble_ok(source, "tagged array update bounds");
            if (!m) continue;
            char *c = emit_or_fail(m, "I retain a full-width index for tagged array updates");
            if (c) {
                int status = 0;
                CHECK(compile_and_run(c, &status) == 0 && (i ? status != 0 : status == 0),
                      "I preserve aliases for valid writes and reject every invalid full-width index");
                free(c);
            }
            nvm_module_free(m);
        }
    }
}

static void test_array_globals(void) {
    for (int strings = 0; strings < 2; ++strings) {
        char source[3072];
        snprintf(source, sizeof source,
            ".string first \"first\"\n.string next \"next\"\n.entry main\n"
            ".function main 0 1 0 int 1\nLOAD_GLOBAL 0\nTYPE_CHECK 7\nASSERT\n"
            "LOAD_GLOBAL 0\nASSERT\nLOAD_GLOBAL 0\nSTORE_LOCAL 0\nLOAD_GLOBAL 0\nSTORE_GLOBAL 1\n"
            "LOAD_GLOBAL 0\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n"
            "LOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nTYPE_CHECK 0\nASSERT\n"
            "CALL append\nLOAD_LOCAL 0\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n"
            "LOAD_GLOBAL 1\nLOAD_LOCAL 0\nEQ\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n%s\nEQ\nASSERT\n"
            "LOAD_GLOBAL 0\nPUSH_I64 0\n%s\nSTORE_GLOBAL 2\nLOAD_GLOBAL 2\nARR_SET\nPOP\n"
            "PUSH_I64 0\nSTORE_GLOBAL 0\nLOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\n%s\nEQ\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_I64 -1\nARR_GET\nTYPE_CHECK 0\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_I64 4294967296\nARR_GET\nTYPE_CHECK 0\nASSERT\n"
            "LOAD_LOCAL 0\nPRINTLN\n"
            "PUSH_I64 0\nRET\n.end\n"
            ".function __init__ 0 0 0 void 0\nARR_LITERAL %d 0\nSTORE_GLOBAL 0\nRET\n.end\n"
            ".function append 0 0 0 void 0\nLOAD_GLOBAL 0\n%s\nARR_PUSH\nSTORE_GLOBAL 0\nRET\n.end\n",
            strings ? "PUSH_STR first" : "PUSH_I64 42",
            strings ? "PUSH_STR next" : "PUSH_I64 7",
            strings ? "PUSH_STR next" : "PUSH_I64 7",
            strings ? 5 : 1, strings ? "PUSH_STR first" : "PUSH_I64 42");
        NvmModule *m = assemble_ok(source, "tagged global arrays");
        if (!m) continue;
        char *c = emit_or_fail(m, "I preserve tagged array handles across global mutation");
        if (c) {
            char output[64] = {0};
            int status = -1;
            CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 && status == 0 &&
                  strcmp(output, strings ? "[next]\n" : "[7]\n") == 0,
                  "I preserve aliases, element tags, full-width missing reads and printed array values");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *bad[] = {
        "LOAD_GLOBAL 0\nARR_LEN\nPOP",
        "ARR_NEW 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 1\nI64_ADD\nPOP",
        "ARR_NEW 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_BOOL 1\nARR_PUSH\nPOP"
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
        char source[512];
        snprintf(source, sizeof source, ".entry main\n.function main 0 0 0 int 1\n%s\nPUSH_I64 0\nRET\n.end\n", bad[i]);
        NvmModule *m = assemble_ok(source, "invalid tagged array consumption");
        if (!m) continue;
        char *c = emit_or_fail(m, "I defer dynamic array checks until consumption");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && status == -1,
                  "I trap non-array use, missing integer consumption and wrong element tags");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_scalar_globals(void) {
    test_array_globals();
    test_tagged_array_read_bounds();
    test_tagged_array_update_bounds();
    const char *source =
        ".string text \"saved\"\n.string yes \"true\"\n.entry main\n"
        ".function main 0 1 0 int 1\n"
        "LOAD_GLOBAL 4095\nTYPE_CHECK 0\nASSERT\n"
        "LOAD_GLOBAL 0\nTYPE_CHECK 4\nASSERT\nCALL get_bool\nASSERT\n"
        "LOAD_GLOBAL 0\nCAST_INT\nPUSH_I64 1\nI64_EQ\nASSERT\n"
        "LOAD_GLOBAL 0\nCAST_STRING\nPUSH_STR yes\nEQ\nASSERT\n"
        "LOAD_GLOBAL 0\nBOOL_NOT\nBOOL_NOT\nASSERT\n"
        "LOAD_GLOBAL 0\nPUSH_BOOL 1\nBOOL_AND\nASSERT\n"
        "LOAD_GLOBAL 0\nPUSH_BOOL 0\nBOOL_OR\nASSERT\n"
        "LOAD_GLOBAL 0\nPUSH_I64 1\nEQ\nBOOL_NOT\nASSERT\n"
        "PUSH_BOOL 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nJMP_FALSE false_ok\nPUSH_BOOL 0\nASSERT\nfalse_ok:\n"
        "PUSH_STR text\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nSTORE_LOCAL 0\n"
        "CALL mutate\nLOAD_GLOBAL 0\nPUSH_I64 42\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_STR text\nEQ\nASSERT\n"
        "LOAD_GLOBAL 4095\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nTYPE_CHECK 0\nASSERT\n"
        "PUSH_I64 7\nSTORE_GLOBAL 4095\nLOAD_GLOBAL 4095\nPUSH_I64 7\nI64_EQ\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n"
        ".function __init__ 0 0 0 void 0\nLOAD_GLOBAL 0\nTYPE_CHECK 0\nASSERT\n"
        "PUSH_BOOL 1\nSTORE_GLOBAL 0\nRET\n.end\n"
        ".function mutate 0 0 0 void 0\nPUSH_I64 42\nSTORE_GLOBAL 0\nRET\n.end\n"
        ".function get_bool 0 0 0 bool 1\nLOAD_GLOBAL 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "tagged scalar globals");
    if (m) {
        char *c = emit_or_fail(m, "I emit map-free tagged globals with checked scalar consumption");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I retain global tags, initialization, cross-function mutation and saved values");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *consumers[] = {"RET", "BOOL_NOT\nCAST_INT\nRET", "STR_LEN\nRET"};
    for (size_t i = 0; i < sizeof consumers / sizeof consumers[0]; ++i) {
        char program[256];
        snprintf(program, sizeof program, ".entry main\n.function main 0 0 0 int 1\nLOAD_GLOBAL 0\n%s\n.end\n", consumers[i]);
        m = assemble_ok(program, "uninitialized global consumption");
        if (!m) continue;
        char *c = emit_or_fail(m, "I keep uninitialized globals void until consumption");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && status == -1,
                  "I trap void at typed global consumers");
            free(c);
        }
        nvm_module_free(m);
    }
    const uint32_t invalid[] = {4096, 65536, UINT32_MAX};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        for (int store = 0; store < 2; ++store) {
            m = assemble_ok(store ?
                ".entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nSTORE_GLOBAL 0\nPUSH_I64 0\nRET\n.end\n" :
                ".entry main\n.function main 0 0 0 int 1\nLOAD_GLOBAL 0\nPOP\nPUSH_I64 0\nRET\n.end\n", "global bounds fixture");
            if (!m) continue;
            size_t operand = m->functions[0].code_offset + (store ? 10 : 1);
            for (unsigned j = 0; j < 4; ++j) m->code[operand + j] = (uint8_t)(invalid[i] >> (8 * j));
            char error[256] = {0};
            char *c = nvm2c_emit(m, error, sizeof error);
            CHECK(c == NULL && strstr(error, "global"), "I reject full-width out-of-range global operands before emission");
            free(c); nvm_module_free(m);
        }
    }
    m = assemble_ok(
        ".string key \"key\"\n.string text \"kept\"\n.entry main\n"
        ".function main 0 1 0 int 1\nLOAD_GLOBAL 0\nPRINTLN\n"
        "PUSH_BOOL 1\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPRINTLN\n"
        "PUSH_I64 42\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPRINTLN\n"
        "HM_NEW 5 5\nPUSH_STR key\nPUSH_STR text\nHM_SET\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\nSTORE_GLOBAL 0\n"
        "LOAD_LOCAL 0\nPUSH_STR key\nHM_DELETE\nPOP\n"
        "LOAD_GLOBAL 0\nPRINTLN\nLOAD_GLOBAL 0\nSTORE_GLOBAL 1\n"
        "PUSH_I64 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 1\nPUSH_STR text\nEQ\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n", "retained global strings and printing");
    if (m) {
        char *c = emit_or_fail(m, "I retain lookup strings saved in globals after deletion and overwrite");
        if (c) {
            char output[128] = {0};
            int status = -1;
            CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 && status == 0 &&
                  strcmp(output, "void\ntrue\n42\nkept\n") == 0,
                  "I print global scalar tags and preserve saved string aliases");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_map_helpers_compile_without_safepoints(void) {
    NvmModule *m = assemble_ok(
        ".entry main\n"
        ".function main 0 0 0 int 1\n"
        "  HM_NEW 5 1\n"
        "  HM_LEN\n"
        "  RET\n"
        ".end\n",
        "map without collection safepoints");
    if (!m) return;
    char *c = emit_or_fail(m, "I emit map support without requiring a safepoint");
    if (c) {
        CHECK(strstr(c, "(void)nroot_reset; (void)nmap_collect_if_needed;") != NULL,
              "I keep optional root helpers referenced for strict C compilers");
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0 && status == 0,
              "I compile map support without root-reset or collection safepoints");
        free(c);
    }
    nvm_module_free(m);
}

static void test_emitted_map_get(void) {
    test_map_helpers_compile_without_safepoints();
    test_scalar_globals();
    test_tagged_scalar_returns();
    test_mixed_lookup_arguments();
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
        CHECK(c != NULL, "I emit checked boolean consumption of tagged values");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && status == -1,
                  "I trap when a boolean operation consumes a missing value");
        }
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
}

static void test_native_map_runtime(void) {
    test_emitted_map_flow();
    const char *source =
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdio.h>\n#include <assert.h>\n"
        /* The fragment requires its caller's invariant hook. The ordinary
         * generated prelude's diagnostic text has a separate end-to-end gate. */
        "#define NVM2C_ABORT() abort()\n"
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

static void test_record_array_fact_namespaces(void) {
    for (int variant = 0; variant < 6; ++variant) {
        int branch = variant % 3;
        int live_record = variant >= 3;
        char source[2048];
        strcpy(source, ".string text \"preserved\"\n.entry 0\n.function main 0 0 0 int 1\n");
        strcat(source, live_record ? "PUSH_STR text\nAGG_PACK 0 0 0 1\nARR_NEW 8\nPOP\n"
                                  : "ARR_NEW 8\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_PUSH\n"
                                    "PUSH_I64 42\nAGG_PACK 0 0 0 1\nPOP\n");
        if (branch) {
            strcat(source, branch == 1 ? "PUSH_BOOL 1\n" : "PUSH_BOOL 0\n");
            strcat(source, "JMP_FALSE alternate\nDUP\nPOP\nJMP joined\n"
                           "alternate:\nDUP\nPOP\njoined:\n");
        }
        if (!live_record) strcat(source, "PUSH_I64 0\nARR_GET\n");
        strcat(source, "AGG_GET 0\nSTR_LEN\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, "independent record and record-array facts");
        if (!m) continue;
        char *c = emit_or_fail(m, "independent record and record-array facts");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile interleaved record representations");
            CHECK(status == 9, "I preserve string-field facts across the other namespace and branches");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_emitter_many_temporaries(void) {
    const char *pushes[] = {"PUSH_I64 42\n", "PUSH_STR value\n", "ARR_NEW 1\n",
                           "ARR_NEW 5\n", "PUSH_I64 42\nAGG_PACK 0 0 0 1\n", "ARR_NEW 8\n",
                           "PUSH_STR value\nAGG_PACK 0 0 0 1\n"};
    const char *reads[] = {"", "STR_LEN\n", "ARR_LEN\n", "ARR_LEN\n", "AGG_GET 0\n", "ARR_LEN\n",
                          "AGG_GET 0\nSTR_LEN\n"};
    const int expected[] = {42, 5, 0, 0, 42, 0, 5};
    const char *declarations[] = {"int64_t t[300]", "const char *s[300]", "narr_t a[300]",
                                  "nsarr_t sa[300]", "calloc(300, sizeof *r)", "nrarr_t ra[300]", "calloc(300, sizeof *r)"};
    for (int kind = 0; kind < 7; ++kind) {
        for (int condition = 0; condition <= 1; ++condition) {
            char source[32768];
            strcpy(source, ".string value \"hello\"\n.entry 0\n.function main 0 0 0 int 1\n");
            /* The branch condition adds an integer temporary, so that case's
             * total is 301, while every other tested kind has exactly 300. */
            for (int i = 0; i < 300; ++i) {
                strcat(source, pushes[kind]);
                if (i != 299) strcat(source, "POP\n");
            }
            strcat(source, condition ? "PUSH_BOOL 1\n" : "PUSH_BOOL 0\n");
            strcat(source, "JMP_FALSE alternate\nJMP joined\nalternate:\nNOP\njoined:\n");
            strcat(source, reads[kind]);
            strcat(source, "RET\n.end\n");
            NvmModule *m = assemble_ok(source, "many emitter temporaries");
            if (!m) continue;
            char *c = emit_or_fail(m, "many emitter temporaries");
            if (c) {
                CHECK(strstr(c, kind == 0 ? "int64_t t[301]" : declarations[kind]) != NULL,
                      "I size generated storage to the actual temporary high-water count");
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0,
                      "I compile more than 256 temporaries of each supported representation");
                CHECK(status == expected[kind], "I restore late temporary values through either branch");
                free(c);
            }
            nvm_module_free(m);
        }
    }
}

static void test_emitter_deep_stacks(void) {
    for (int condition = 0; condition <= 1; ++condition) {
        char source[8192];
        strcpy(source, ".entry 0\n.function main 0 0 0 int 1\n");
        for (int i = 0; i < 75; ++i) strcat(source, "PUSH_I64 1\n");
        strcat(source, condition ? "PUSH_BOOL 1\n" : "PUSH_BOOL 0\n");
        strcat(source, "JMP_FALSE alternate\nPOP\nPUSH_I64 2\nJMP joined\n"
                       "alternate:\nPOP\nPUSH_I64 3\njoined:\n");
        for (int i = 0; i < 74; ++i) strcat(source, "I64_ADD\n");
        strcat(source, "RET\n.end\n");
        NvmModule *m = assemble_ok(source, "deep emitter branch stack");
        if (!m) continue;
        char *c = emit_or_fail(m, "deep emitter branch stack");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0,
                  "I compile deep stacks with independent branch snapshots");
            CHECK(status == (condition ? 76 : 77),
                  "I preserve all deep operands and the selected branch value");
            free(c);
        }
        nvm_module_free(m);
    }
    for (int strings = 0; strings <= 1; ++strings) {
        char source[8192];
        strcpy(source, ".string value \"hello\"\n.string last \"lastvalue\"\n"
                       ".entry 0\n.function main 0 0 0 int 1\n");
        for (int i = 0; i < 200; ++i) {
            char push[32];
            snprintf(push, sizeof push, "PUSH_I64 %d\n", i);
            strcat(source, strings ? (i == 199 ? "PUSH_STR last\n" : "PUSH_STR value\n") : push);
        }
        strcat(source, strings ? "ARR_LITERAL 5 200\n" : "ARR_LITERAL 1 200\n");
        strcat(source, "PUSH_I64 199\nARR_GET\n");
        if (strings) strcat(source, "STR_LEN\n");
        strcat(source, "RET\n.end\n");
        NvmModule *m = assemble_ok(source, "long emitter array literal");
        if (!m) continue;
        char *c = emit_or_fail(m, "long emitter array literal");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0,
                  "I compile array literals beyond both former fixed buffers");
            CHECK(status == (strings ? 9 : 199), "I retain the last literal element in order");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_classifier_deep_stack(void) {
    /* Wide aggregates must survive deep branch snapshots; inconsistent
     * incoming heights must still be rejected by the direct API. */
    for (int malformed = 0; malformed <= 1; ++malformed) {
        char source[4096];
        strcpy(source, ".entry 0\n.function main 0 0 0 int 1\n");
        for (int i = 0; i < 75; ++i) strcat(source, "PUSH_I64 1\n");
        strcat(source, "PUSH_BOOL 1\nJMP_FALSE joined\nPUSH_I64 2\nPOP\n"
                       "joined:\nAGG_PACK 0 0 0 75\nPOP\nPUSH_I64 0\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, "deep classifier branch stack");
        if (!m) continue;
        if (malformed) {
            /* I remove the balancing pop after assembly so the direct API
             * must detect the unequal incoming heights itself. */
            for (uint32_t pc = 0; pc < m->code_size;) {
                DecodedInstruction ins;
                uint32_t n = isa_decode(m->code + pc, m->code_size - pc, &ins);
                if (!n) break;
                if (ins.opcode == OP_POP) {
                    memset(m->code + pc, OP_NOP, n);
                    break;
                }
                pc += n;
            }
        }
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        if (malformed) {
            CHECK(c == NULL && strstr(error, "incompatible stack heights"),
                  "I reject inconsistent deep branch stacks");
        } else {
            CHECK(c != NULL, "I translate a wide aggregate after a deep stack join");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I execute the wide aggregate after a deep stack join");
            }
        }
        free(c);
        nvm_module_free(m);
    }
}

static void test_shared_code_shape_scopes(void) {
    const char *source =
        ".string text \"hello\"\n.entry main\n"
        ".function read_int 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n"
        ".function read_string 1 1 0 string 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_I64 42\nAGG_PACK 0 0 0 1\nCALL read_int\nPOP\n"
        "PUSH_STR text\nAGG_PACK 0 0 0 1\nCALL read_string\nSTR_LEN\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "shared bytecode shape scopes");
    if (!m) return;
    m->functions[1].code_offset = m->functions[0].code_offset;
    m->functions[1].code_length = m->functions[0].code_length;
    char *c = emit_or_fail(m, "shared bytecode shape scopes");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0 && status == 5,
              "I keep function shapes independent when bytecode ranges overlap");
        free(c);
    }
    nvm_module_free(m);
}

static void test_record_array_alias_shapes(void) {
    for (int conflict = 0; conflict < 2; ++conflict) {
        char source[2048];
        strcpy(source, ".string text \"wrong\"\n.entry main\n.function main 0 2 0 int 1\n"
                       "ARR_NEW 8\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 1\n"
                       "LOAD_LOCAL 0\nPUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_PUSH\nPOP\nLOAD_LOCAL 1\n");
        strcat(source, conflict ? "PUSH_STR text\n" : "PUSH_I64 2\n");
        strcat(source, "AGG_PACK 0 0 0 1\nARR_PUSH\nPOP\nLOAD_LOCAL 0\n"
                       "PUSH_I64 1\nARR_GET\nAGG_GET 0\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, "record-array alias shapes");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        if (conflict) {
            CHECK(c == NULL && strstr(error, "shape"), "I reject incompatible fields inserted through array aliases");
        } else {
            CHECK(c != NULL, "I accept compatible record-array aliases");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 2,
                      "I preserve mutations through compatible aliases");
            }
        }
        free(c);
        nvm_module_free(m);
    }
}

/* I append the first diagnostic through a copied record's empty list field. */
static void test_empty_diagnostic_record_fields(void) {
    const unsigned fields[] = {2, 5};
    for (size_t scenario = 0; scenario < sizeof fields / sizeof fields[0]; ++scenario) {
        unsigned field = fields[scenario];
        char source[8192], fragment[1024];
        strcpy(source, ".types 3 0 0\n.string code \"E_TEST\"\n"
            ".string message \"ordinary diagnostic\"\n.string file \"fixture.nano\"\n"
            ".entry main\n.function make 0 0 0 struct 1\nPUSH_I64 41\nPUSH_BOOL 1\n");
        if (field == 5) strcat(source, "PUSH_STR file\nPUSH_I64 17\nPUSH_BOOL 0\n");
        snprintf(fragment, sizeof fragment,
            "ARR_NEW 8\nAGG_PACK 0 0 0 %u\nRET\n.end\n"
            ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n"
            ".function append 1 1 0 void 0\nLOAD_LOCAL 0\nAGG_GET %u\n"
            "PUSH_I64 1\nPUSH_I64 2\nPUSH_STR code\nPUSH_STR message\n"
            "PUSH_STR file\nPUSH_I64 42\nPUSH_I64 7\nAGG_PACK 0 2 0 3\n"
            "AGG_PACK 0 1 0 5\nARR_PUSH\nPOP\nRET\n.end\n"
            ".function main 0 3 0 int 1\nCALL make\nSTORE_LOCAL 0\n"
            "LOAD_LOCAL 0\nCALL relay\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nAGG_GET %u\n"
            "ARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\nLOAD_LOCAL 1\nCALL append\n",
            field + 1, field, field);
        strcat(source, fragment);
        /* Both record copies still see the first append; scalar neighbors survive. */
        for (unsigned copy = 0; copy < 2; ++copy) {
            snprintf(fragment, sizeof fragment,
                "LOAD_LOCAL %u\nAGG_GET %u\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\n"
                "LOAD_LOCAL %u\nAGG_GET 0\nPUSH_I64 41\nI64_EQ\nASSERT\n"
                "LOAD_LOCAL %u\nAGG_GET 1\nDUP\nTYPE_CHECK 4\nASSERT\nASSERT\n",
                copy, field, copy, copy);
            strcat(source, fragment);
            if (field == 5) {
                snprintf(fragment, sizeof fragment,
                    "LOAD_LOCAL %u\nAGG_GET 2\nPUSH_STR file\nEQ\nASSERT\n"
                    "LOAD_LOCAL %u\nAGG_GET 3\nPUSH_I64 17\nI64_EQ\nASSERT\n"
                    "LOAD_LOCAL %u\nAGG_GET 4\nBOOL_NOT\nASSERT\n", copy, copy, copy);
                strcat(source, fragment);
            }
        }
        snprintf(fragment, sizeof fragment,
            "LOAD_LOCAL 0\nAGG_GET %u\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 2\n"
            "LOAD_LOCAL 2\nAGG_GET 0\nPUSH_I64 1\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 1\nPUSH_I64 2\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 2\nPUSH_STR code\nEQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 3\nPUSH_STR message\nEQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 4\nAGG_GET 0\nPUSH_STR file\nEQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 4\nAGG_GET 1\nPUSH_I64 42\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 2\nAGG_GET 4\nAGG_GET 2\nPUSH_I64 7\nI64_EQ\nASSERT\n"
            "PUSH_I64 0\nRET\n.end\n", field);
        strcat(source, fragment);
        NvmModule *module = assemble_ok(source, "first diagnostic append into an empty record field");
        if (!module) continue;
        char *c = emit_or_fail(module, "empty diagnostic field representation");
        if (c) {
            snprintf(fragment, sizeof fragment, ".ra[%u] = ra[", field);
            CHECK(strstr(c, fragment) != NULL, "I pack the diagnostic field as a record-array handle");
            CHECK(strstr(c, "nrarr_new()") != NULL, "I allocate the empty diagnostic record array explicitly");
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I preserve first diagnostic append, aliases, payload and scalar neighbors");
            free(c);
        }
        nvm_module_free(module);
    }
}

static void test_record_array_fields(void) {
    for (int empty = 0; empty < 2; ++empty) {
        char source[4096];
        snprintf(source, sizeof source, ".string text \"hello\"\n.entry main\n"
            ".function wrap 1 1 0 struct 1\nLOAD_LOCAL 0\nPUSH_BOOL %d\nJMP_FALSE alternate\n"
            "AGG_PACK 0 0 0 1\nJMP joined\nalternate:\nAGG_PACK 0 0 0 1\njoined:\nRET\n.end\n"
            ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n"
            ".function main 0 2 0 int 1\nARR_NEW 8\nSTORE_LOCAL 0\nLOAD_LOCAL 0\n", empty);
        if (!empty) strcat(source, "PUSH_STR text\nPUSH_I64 42\nAGG_PACK 0 0 0 2\nARR_PUSH\n");
        strcat(source, "CALL wrap\nCALL relay\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nAGG_GET 0\n");
        if (empty) strcat(source, "ARR_LEN\nRET\n.end\n");
        else strcat(source, "PUSH_I64 0\nARR_GET\nAGG_GET 0\nSTR_LEN\nPUSH_I64 5\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 1\nAGG_GET 0\nPUSH_STR text\nPUSH_I64 17\nAGG_PACK 0 0 0 2\nARR_PUSH\nPOP\n"
            "LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 2\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 1\nAGG_GET 0\nPUSH_I64 1\nARR_GET\nAGG_GET 1\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, "record-array fields");
        if (!m) continue;
        char *c = emit_or_fail(m, "record-array fields");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile record-array fields through calls");
            CHECK(status == (empty ? 0 : 17), "I preserve mixed element shapes and shared array mutations");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *conflict = ".string text \"wrong\"\n.entry main\n.function main 0 1 0 int 1\n"
        "ARR_NEW 8\nPUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_PUSH\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_PUSH\nPOP\nPUSH_I64 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(conflict, "conflicting nested array element");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I reject conflicting elements through extracted array fields");
        free(c);
        nvm_module_free(m);
    }
}

static void test_delayed_array_element_facts(void) {
    for (int scenario = 0; scenario < 3; ++scenario) {
        int incompatible = scenario == 1;
        char source[4096];
        strcpy(source, ".string text \"hello\"\n.entry main\n");
        if (scenario == 2) strcat(source,
            ".function count 1 1 0 int 1\nLOAD_LOCAL 0\nARR_LEN\nRET\n.end\n");
        strcat(source,
            ".function main 0 1 0 int 1\nARR_NEW 8\nPUSH_STR text\nAGG_PACK 0 0 0 1\n"
            "ARR_PUSH\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nCALL count\nPOP\n"
            "LOAD_LOCAL 0\nCALL make\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nARR_PUSH\nPOP\n"
            "LOAD_LOCAL 0\nCALL count\nRET\n.end\n"
            ".function make 0 0 0 struct 1\n");
        strcat(source, incompatible ? "PUSH_I64 7\nARR_LITERAL 1 1\n" :
            "ARR_NEW 8\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_PUSH\n");
        strcat(source, "AGG_PACK 0 0 0 1\nRET\n.end\n");
        if (scenario != 2) strcat(source,
            ".function count 1 1 0 int 1\nLOAD_LOCAL 0\nARR_LEN\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, "delayed record-array element facts");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        if (incompatible) CHECK(c == NULL, "I reject scalar insertion into an explicit record array");
        else {
            CHECK(c != NULL, "I preserve explicit record arrays while a later callee's fields are unknown");
            if (!c) fprintf(stderr, "    delayed array: %s\n", error);
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 2,
                      "I append an element from a later-inferred nested array");
            }
        }
        free(c);
        nvm_module_free(m);
    }
}

static void test_record_array_return_fields(void) {
    for (int tail = 0; tail < 2; ++tail) {
        char source[4096];
        strcpy(source, ".string text \"hello\"\n.entry main\n"
            ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nSTR_LEN\nRET\n.end\n"
            ".function main 0 1 0 int 1\nPUSH_I64 3\nCALL relay\nSTORE_LOCAL 0\n"
            "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 1\nPUSH_I64 42\nI64_EQ\nASSERT\n"
            "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nCALL length\nRET\n.end\n"
            ".function relay 1 1 0 array 1\nLOAD_LOCAL 0\nPUSH_I64 0\nI64_EQ\nJMP_FALSE recurse\n");
        strcat(source, tail ? "TAIL_CALL make\n" : "CALL make\nRET\n");
        strcat(source, "recurse:\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_SUB\n");
        strcat(source, tail ? "TAIL_CALL relay\n" : "CALL relay\nRET\n");
        strcat(source, ".end\n.function make 0 0 0 array 1\nARR_NEW 8\nPUSH_STR text\n"
            "PUSH_I64 42\nAGG_PACK 0 0 0 2\nARR_PUSH\nRET\n.end\n");
        NvmModule *m = assemble_ok(source, "record-array return field propagation");
        if (!m) continue;
        char *c = emit_or_fail(m, "I infer returned element fields across forward and recursive calls");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 5,
                  "I preserve mixed element fields through normal and tail returns");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *conflict = ".string text \"hello\"\n.entry main\n"
        ".function make 1 1 0 array 1\nLOAD_LOCAL 0\nJMP_FALSE alternate\n"
        "ARR_NEW 8\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_PUSH\nRET\n"
        "alternate:\nARR_NEW 8\nPUSH_I64 7\nAGG_PACK 0 0 0 1\nARR_PUSH\nRET\n.end\n"
        ".function main 0 0 0 int 1\nPUSH_BOOL 1\nCALL make\nARR_LEN\nRET\n.end\n";
    NvmModule *m = assemble_ok(conflict, "conflicting record-array return fields");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "conflicting"), "I reject incompatible element fields across return paths");
        free(c);
        nvm_module_free(m);
    }
}

static void test_record_array_literals(void) {
    const char *delayed = ".string text \"hello\"\n.entry main\n"
        ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nSTR_LEN\nRET\n.end\n"
        ".function main 0 1 0 int 1\nCALL make\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nAGG_GET 0\nCALL length\nRET\n.end\n"
        ".function make 0 0 0 struct 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\n"
        "AGG_PACK 0 0 0 1\nRET\n.end\n";
    NvmModule *delayed_module = assemble_ok(delayed, "delayed nested record local facts");
    if (delayed_module) {
        char *c = emit_or_fail(delayed_module, "I do not invent integer fields for an unresolved local");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 5,
                  "I resolve nested element fields through locals after a forward call");
            free(c);
        }
        nvm_module_free(delayed_module);
    }
    const char *cases[] = {
        "ARR_LITERAL 8 0\nARR_LEN\nRET\n",
        "PUSH_STR text\nPUSH_I64 17\nAGG_PACK 0 0 0 2\n"
        "PUSH_STR text\nPUSH_I64 42\nAGG_PACK 0 0 0 2\nARR_LITERAL 8 2\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 1\nPUSH_I64 17\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\nAGG_GET 0\nSTR_LEN\nRET\n",
        "ARR_LITERAL 8 0\nPUSH_STR text\nAGG_PACK 0 0 0 1\nARR_PUSH\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nSTR_LEN\nRET\n",
        "PUSH_I64 1\nARR_LITERAL 8 1\nPOP\nPUSH_I64 0\nRET\n",
        "PUSH_STR text\nAGG_PACK 0 0 0 1\nPUSH_I64 1\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 2\nPOP\nPUSH_I64 0\nRET\n"
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        char source[4096], error[256] = {0};
        snprintf(source, sizeof source, ".string text \"hello\"\n.entry main\n.function main 0 1 0 int 1\n%s.end\n", cases[i]);
        NvmModule *m = assemble_ok(source, "record-array literal");
        if (!m) continue;
        char *c = nvm2c_emit(m, error, sizeof error);
        if (i >= 3) CHECK(c == NULL && strstr(error, "record-array literal"), "I reject invalid record-array literal elements");
        else {
            CHECK(c != NULL, "I translate explicitly tagged record-array literals");
            if (!c) fprintf(stderr, "    record literal: %s\n", error);
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == (i ? 5 : 0),
                      "I preserve empty literals, record fields and literal element order");
            }
        }
        free(c);
        nvm_module_free(m);
    }
}

static void test_array_valued_record_fields(void) {
    test_record_array_literals();
    test_record_array_return_fields();
    test_delayed_array_element_facts();
    test_record_array_fields();
    test_empty_diagnostic_record_fields();
    for (int strings = 0; strings < 2; ++strings) {
        for (int empty = 0; empty < 2; ++empty) {
            char source[4096], line[256];
            snprintf(source, sizeof source, ".string text \"hello\"\n.entry main\n"
                           ".function wrap 1 1 0 struct 1\nLOAD_LOCAL 0\nPUSH_BOOL %d\nJMP_FALSE alternate\n"
                           "AGG_PACK 0 0 0 1\nJMP joined\nalternate:\nAGG_PACK 0 0 0 1\njoined:\nRET\n.end\n"
                           ".function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n"
                           ".function main 0 2 0 int 1\n", empty);
            if (!empty) strcat(source, strings ? "PUSH_STR text\n" : "PUSH_I64 42\n");
            snprintf(line, sizeof line, "ARR_LITERAL %d %d\n", strings ? 5 : 1, empty ? 0 : 1);
            strcat(source, line);
            strcat(source, "CALL wrap\nCALL relay\nSTORE_LOCAL 0\nLOAD_LOCAL 0\nSTORE_LOCAL 1\n"
                           "LOAD_LOCAL 0\nAGG_GET 0\nARR_LEN\n");
            snprintf(line, sizeof line, "PUSH_I64 %d\nI64_EQ\nASSERT\nLOAD_LOCAL 1\nAGG_GET 0\n",
                     empty ? 0 : 1);
            strcat(source, line);
            if (empty) strcat(source, "ARR_LEN\n");
            else {
                strcat(source, "PUSH_I64 0\nARR_GET\n");
                if (strings) strcat(source, "STR_LEN\n");
            }
            strcat(source, "RET\n.end\n");
            NvmModule *m = assemble_ok(source, "array-valued record fields");
            if (!m) continue;
            char *c = emit_or_fail(m, "array-valued record fields");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0, "I compile array-valued record fields through calls and copies");
                CHECK(status == (empty ? 0 : strings ? 5 : 42), "I preserve the array field's element representation");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    const char *invalid[] = {
        ".entry main\n.function wrap 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n"
        ".function main 0 0 0 int 1\nARR_NEW 1\nCALL wrap\nPOP\nARR_NEW 5\nCALL wrap\nPOP\nPUSH_I64 0\nRET\n.end\n",
        ".entry main\n.function main 0 0 0 int 1\nPUSH_I64 1\nAGG_PACK 0 0 0 1\nAGG_PACK 0 0 0 1\nRET\n.end\n"
    };
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        NvmModule *m = assemble_ok(invalid[i], "unsupported array field shape");
        if (!m) continue;
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, i ? "shape" : "conflicting kinds"),
              "I reject incompatible array representations and unresolved nested field shapes");
        if (!i) CHECK(strstr(error, "function 1, offset") && strstr(error, "parameter 0 of function 0"),
                      "I identify both the conflicting caller and destination parameter");
        free(c);
        nvm_module_free(m);
    }
}

static void test_wide_aggregate_calls(void) {
    const int widths[] = {75, 129, 300};
    for (size_t w = 0; w < sizeof widths / sizeof widths[0]; ++w) {
        for (int condition = 0; condition < 2; ++condition) {
            char source[32768], line[256];
            strcpy(source, ".string yes \"hello\"\n.string no \"alternate\"\n.entry main\n"
                           ".function make 1 1 0 struct 1\nLOAD_LOCAL 0\nJMP_FALSE alternate\n");
            for (int arm = 0; arm < 2; ++arm) {
                if (arm) strcat(source, "alternate:\n");
                for (int f = 0; f < widths[w] - 1; ++f) {
                    snprintf(line, sizeof line, "PUSH_I64 %d\n", f);
                    strcat(source, line);
                }
                snprintf(line, sizeof line, "PUSH_STR %s\nAGG_PACK 0 0 0 %d\nJMP joined\n",
                         arm ? "no" : "yes", widths[w]);
                strcat(source, line);
            }
            strcat(source, "joined:\nRET\n.end\n.function relay 1 1 0 struct 1\nLOAD_LOCAL 0\nRET\n.end\n"
                           ".function main 0 1 0 int 1\n");
            snprintf(line, sizeof line,
                     "PUSH_BOOL %d\nCALL make\nCALL relay\nSTORE_LOCAL 0\n"
                     "LOAD_LOCAL 0\nAGG_GET %d\nPUSH_I64 %d\nI64_EQ\nASSERT\n"
                     "LOAD_LOCAL 0\nAGG_GET %d\nSTR_LEN\nRET\n.end\n",
                     condition, widths[w] - 2, widths[w] - 2, widths[w] - 1);
            strcat(source, line);
            NvmModule *m = assemble_ok(source, "wide aggregate calls");
            if (!m) continue;
            char *c = emit_or_fail(m, "wide aggregate calls");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0, "I compile wide mixed-field aggregates through calls");
                CHECK(status == (condition ? 5 : 9), "I preserve high-index fields through joins, calls and locals");
                free(c);
            }
            nvm_module_free(m);
        }
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

static void test_map_aggregate_fields(void) {
    for (int before = 0; before < 2; ++before) {
        const char *worker = ".function repack 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_GET 0\nAGG_GET 0\nAGG_PACK 0 2 0 1\nRET\n.end\n";
        const char *entry = ".function main 0 0 0 int 1\nPUSH_STR text\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\nCALL repack\nAGG_GET 0\nPUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n";
        char source[2048];
        snprintf(source, sizeof source, ".string text \"late\"\n.types 3 0 0\n.entry main\n%s%s",
                 before ? worker : entry, before ? entry : worker);
        NvmModule *m = assemble_ok(source, "late packed field facts");
        if (!m) continue;
        char *c = emit_or_fail(m, "I resolve packed nested projections after constructing the graph");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I preserve late string field facts through nested projection and repacking");
            free(c);
        }
        nvm_module_free(m);
    }
    NvmModule *unknown = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
        ".function uncalled 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n",
        "truly unresolved packed field");
    if (unknown) {
        char err[512];
        char *c = nvm2c_emit(unknown, err, sizeof err);
        CHECK(c != NULL,
              "I do not require an executable layout for an uncalled parameter");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I execute entry without the uncalled unresolved record");
        }
        free(c);
        nvm_module_free(unknown);
    }
    for (int strings = 0; strings < 2; ++strings) {
        for (int nested = 0; nested < 2; ++nested) {
            for (int variant = 0; variant < 2; ++variant) {
                for (int forward = 0; forward < 2; ++forward) {
                    char source[8192], entry[2048], worker[512];
                    const char *value = strings ? "PUSH_STR text\n" : "PUSH_I64 42\n";
                    snprintf(worker, sizeof worker,
                        ".function wrap 1 1 0 %s 1\nLOAD_LOCAL 0\nAGG_PACK %d 0 0 1\n%sRET\n.end\n",
                        variant && !nested ? "union" : "struct",
                        variant, nested ? "AGG_PACK 0 1 0 1\n" : "");
                    snprintf(entry, sizeof entry,
                        ".function main 0 3 0 int 1\nHM_NEW 5 %d\nSTORE_LOCAL 0\n"
                        "LOAD_LOCAL 0\nCALL wrap\nSTORE_LOCAL 1\n"
                        "LOAD_LOCAL 1\n%sAGG_GET 0\nSTORE_LOCAL 2\n"
                        "LOAD_LOCAL 2\nHM_LEN\nPUSH_I64 0\nEQ\nASSERT\n"
                        "LOAD_LOCAL 2\nPUSH_STR key\n%sHM_SET\nPOP\n"
                        "LOAD_LOCAL 0\nPUSH_STR key\nHM_GET\n%sEQ\nASSERT\n"
                        "LOAD_LOCAL 1\n%sAGG_GET 0\nPUSH_STR key\nHM_GET\n%sEQ\nASSERT\n"
                        "PUSH_I64 0\nRET\n.end\n",
                        strings ? 5 : 1, nested ? "AGG_GET 0\n" : "", value, value,
                        nested ? "AGG_GET 0\n" : "", value);
                    snprintf(source, sizeof source,
                        ".string key \"key\"\n.string text \"value\"\n.types 2 1 1\n.entry main\n%s%s",
                        forward ? entry : worker, forward ? worker : entry);
                    NvmModule *m = assemble_ok(source, "map aggregate fields");
                    if (!m) continue;
                    char *c = emit_or_fail(m, "I retain map fields through aggregate returns");
                    if (c) {
                        int status = -1;
                        CHECK(compile_and_run(c, &status) == 0 && status == 0,
                              "I retain map identity across nested aggregate copies and mutation");
                        free(c);
                    }
                    nvm_module_free(m);
                }
            }
        }
    }
    NvmModule *bad = assemble_ok(
        ".string key \"key\"\n.string text \"wrong\"\n.types 1 0 0\n.entry main\n"
        ".function main 0 0 0 int 1\nHM_NEW 5 1\nAGG_PACK 0 0 0 1\n"
        "AGG_GET 0\nPUSH_STR key\nPUSH_STR text\nHM_SET\nPOP\nPUSH_I64 0\nRET\n.end\n",
        "incompatible nested map values");
    if (bad) {
        char err[512];
        char *c = nvm2c_emit(bad, err, sizeof err);
        CHECK(c == NULL, "I reject incompatible values after map field extraction");
        free(c);
        nvm_module_free(bad);
    }
}

static void test_tagged_host_arguments(void) {
    for (int integer = 0; integer < 2; ++integer) {
        for (int value = 0; value < 4; ++value) {
            for (int forward = 0; forward < 2; ++forward) {
                char source[2048], worker[512], entry[512];
                snprintf(worker, sizeof worker,
                    ".function consume 1 1 0 int 1\nLOAD_LOCAL 0\nCALL_EXTERN 0\nPOP\n"
                    "LOAD_LOCAL 0\nTYPE_CHECK %d\nASSERT\nPUSH_I64 0\nRET\n.end\n", integer ? 1 : 5);
                const char *stored = value == 0 ? "" : value == 3 ? "PUSH_BOOL 1\nSTORE_GLOBAL 0\n" :
                    ((value == 1) == integer) ? "PUSH_I64 65\nSTORE_GLOBAL 0\n" : "PUSH_STR text\nSTORE_GLOBAL 0\n";
                snprintf(entry, sizeof entry,
                    ".function main 0 0 0 int 1\n%sLOAD_GLOBAL 0\nCALL consume\nRET\n.end\n", stored);
                snprintf(source, sizeof source,
                    ".string text \"\"\n.import \"\" \"%s\" %s %s\n.entry main\n%s%s",
                    integer ? "vm_string_from_char" : "vm_file_exists", integer ? "string" : "bool",
                    integer ? "int" : "string", forward ? entry : worker, forward ? worker : entry);
                NvmModule *m = assemble_ok(source, "tagged host argument boundary");
                if (!m) continue;
                char *c = emit_or_fail(m, "I retain tagged argument storage at exact host boundaries");
                if (c) {
                    int status = 0;
                    CHECK(compile_and_run(c, &status) == 0 && status == (value == 1 ? 0 : -1),
                          "I check host argument tags before invocation without changing source locals");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
}

static void test_generic_ordering(void) {
    const char *ops[] = {"LT", "LE", "GT", "GE"};
    const struct { const char *values; int order; } cases[] = {
        {"PUSH_I64 -9223372036854775808\nPUSH_I64 9223372036854775807", -1},
        {"PUSH_I64 42\nPUSH_I64 -1", 1},
        {"PUSH_I64 42\nPUSH_I64 42", 0},
        {"PUSH_BOOL 0\nPUSH_BOOL 1", -1},
        {"PUSH_BOOL 1\nPUSH_BOOL 1", 0},
        {"PUSH_I64 999\nPUSH_BOOL 0", -1},
        {"PUSH_STR high\nPUSH_STR low", 1},
        {"PUSH_STR low\nPUSH_STR low", 0},
        {"PUSH_STR low\nPUSH_STR high", -1},
        {"PUSH_STR low\nPUSH_BOOL 1", 1},
        {"LOAD_GLOBAL 0\nPUSH_I64 -9", -1},
        {"PUSH_I64 42\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_I64 40", 1},
        {"PUSH_STR low\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_STR high", -1},
        {"ARR_LITERAL 1 0\nARR_LITERAL 5 0", 0},
        {"ARR_LITERAL 1 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nARR_LITERAL 5 0", 0},
        {"ARR_LITERAL 1 0\nPUSH_STR high", 1},
        {"HM_NEW 5 1\nHM_NEW 5 5", 0},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        for (int op = 0; op < 4; ++op) {
            int yes = op == 0 ? cases[i].order < 0 : op == 1 ? cases[i].order <= 0 :
                      op == 2 ? cases[i].order > 0 : cases[i].order >= 0;
            char source[1024];
            snprintf(source, sizeof source,
                ".string low \"a\"\n.string high \"z\"\n.entry main\n.function main 0 0 0 int 1\n"
                "%s\n%s\nDUP\nTYPE_CHECK 4\nASSERT\n%sASSERT\nPUSH_I64 0\nRET\n.end\n",
                cases[i].values, ops[op], yes ? "" : "BOOL_NOT\n");
            NvmModule *m = assemble_ok(source, "generic ordered values");
            if (!m) continue;
            char *c = emit_or_fail(m, "I preserve generic comparison tag and value rules");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I order supported native values and retain boolean results");
                free(c);
            }
            nvm_module_free(m);
        }
    }
}

static void test_classifier_local_bounds(void) {
    const unsigned counts[] = {1, 257, 512, 1024};
    for (size_t i = 0; i < sizeof counts / sizeof counts[0]; ++i) {
        char source[768];
        snprintf(source, sizeof source,
            ".string text \"answer\"\n.entry main\n.function main 0 %u 0 int 1\n"
            "PUSH_STR text\nSTORE_LOCAL %u\nLOAD_LOCAL %u\nCALL length\nRET\n.end\n"
            ".function length 1 1 0 int 1\nLOAD_LOCAL 0\nSTR_LEN\nRET\n.end\n",
            counts[i], counts[i] - 1, counts[i] - 1);
        NvmModule *wide = assemble_ok(source, "wide local facts");
        if (!wide) continue;
        char *native = emit_or_fail(wide, "I preserve high-index string locals across calls");
        if (native) {
            int status = -1;
            CHECK(compile_and_run(native, &status) == 0 && status == 6,
                  "I execute module-sized local facts at the supported boundaries");
            free(native);
        }
        nvm_module_free(wide);
    }
    for (int tail = 0; tail < 2; ++tail) {
        char source[32768];
        size_t used = (size_t)snprintf(source, sizeof source,
            ".string text \"answer\"\n.entry main\n.function main 0 0 0 int 1\n");
        for (unsigned i = 0; i < 1023; ++i)
            used += (size_t)snprintf(source + used, sizeof source - used, "PUSH_I64 %u\n", i);
        snprintf(source + used, sizeof source - used,
            "PUSH_STR text\n%s wide\n%s.end\n"
            ".function wide 1024 1024 0 int 1\nLOAD_LOCAL 1023\nSTR_LEN\nRET\n.end\n",
            tail ? "TAIL_CALL" : "CALL", tail ? "" : "RET\n");
        NvmModule *wide = assemble_ok(source, "wide direct argument list");
        if (!wide) continue;
        char *native = emit_or_fail(wide, "I emit checked argument text for 1,024 parameters");
        if (native) {
            int status = -1;
            CHECK(compile_and_run(native, &status) == 0 && status == 6,
                  "I preserve wide ordinary and tail-call argument types");
            free(native);
        }
        nvm_module_free(wide);
    }
    NvmModule *uninitialized = assemble_ok(
        ".entry main\n.function main 0 1024 0 int 1\nLOAD_LOCAL 1023\nRET\n.end\n", "wide uninitialized local");
    if (uninitialized) {
        char error[256] = {0};
        char *native = nvm2c_emit(uninitialized, error, sizeof error);
        CHECK(native != NULL, "I preserve uninitialized high-index locals as void");
        if (native) {
            int status = -1;
            CHECK(compile_and_run(native, &status) == 0 && status != 0,
                  "I reject void as a typed integer return at runtime");
        }
        free(native); nvm_module_free(uninitialized);
    }
    NvmModule *m = assemble_ok(".entry 0\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n",
                              "classifier bounds");
    if (!m) return;
    char err[256];
    m->functions[0].local_count = UINT16_MAX;
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL && strstr(err, "too many locals") != NULL, "I reject oversized locals before writing classifier state");
    free(c);
    m->functions[0].local_count = 1025;
    c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL && strstr(err, "too many locals") != NULL, "I reject one past my supported local ceiling");
    free(c);
    m->functions[0].local_count = 0;
    m->functions[0].arity = 1;
    c = nvm2c_emit(m, err, sizeof err);
    CHECK(c == NULL && strstr(err, "arity exceeds local_count") != NULL, "I reject arity exceeding local storage");
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
        "AGG_PACK 1 0 65535 0\nAGG_TAG\nPUSH_I64 65535\nI64_EQ\nCAST_INT\nRET\n",
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
            CHECK(rejected == NULL && (strstr(error, "RET") != NULL ||
                  strstr(error, "cannot convert aggregate storage string to int") != NULL),
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
        "PUSH_I64 2\nAGG_PACK 0 0 0 1\nCALL take\nRET\n.end\n",
        ".entry make\n.function make 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n"
    };
    for (int i = 0; i < 4; i++) {
        NvmModule *m = assemble_ok(sources[i], "unrepresentable function facts");
        if (!m) continue;
        char error[256];
        char *c = nvm2c_emit(m, error, sizeof error);
        if (i == 1) {
            CHECK(c != NULL, "I omit the uncalled function with unresolved parameter storage");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I preserve entry execution without inventing the uncalled layout");
            }
        } else {
            CHECK(c == NULL && strstr(error, i == 3 ? "AGG_PACK" : "conflicting") != NULL,
                  "I reject conflicting or reachable unresolved field types instead of guessing");
        }
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

static void test_boolean_arrays(void) {
    NvmModule *raw = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nARR_LITERAL 4 1\n"
        "PUSH_I64 0\nARR_GET\nTYPE_CHECK 4\nASSERT\nPUSH_I64 0\nRET\n.end\n",
        "boolean array without tagged runtime");
    if (raw) {
        char *c = emit_or_fail(raw, "I emit raw boolean arrays without a global runtime dependency");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I retain a boolean literal element tag without globals");
            free(c);
        }
        nvm_module_free(raw);
    }
    NvmModule *printed = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nPUSH_BOOL 0\nARR_LITERAL 4 2\n"
        "STORE_GLOBAL 0\nLOAD_GLOBAL 0\nPRINTLN\nPUSH_I64 0\nRET\n.end\n",
        "tagged boolean array printing");
    if (printed) {
        char *c = emit_or_fail(printed, "I emit boolean array printing with retained tags");
        if (c) {
            int status = -1; char output[128];
            CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 &&
                  status == 0 && strcmp(output, "[true, false]\n") == 0,
                  "I print boolean elements as booleans rather than integer storage");
            free(c);
        }
        nvm_module_free(printed);
    }
    const char *constructors[] = {
        "ARR_NEW 4\nPUSH_BOOL 0\nARR_PUSH\n",
        "ARR_LITERAL 4 0\nPUSH_BOOL 0\nARR_PUSH\n",
        "PUSH_BOOL 0\nARR_LITERAL 4 1\n"
    };
    for (int constructor = 0; constructor < 3; ++constructor) {
        for (int before = 0; before < 2; ++before) {
            for (int tail = 0; tail < 2; ++tail) {
                char source[8192], entry[3072], worker[512];
                snprintf(worker, sizeof worker,
                    ".function relay 1 1 0 array 1\nLOAD_LOCAL 0\n%s\n.end\n"
                    ".function identity 1 1 0 array 1\nLOAD_LOCAL 0\nRET\n.end\n",
                    tail ? "TAIL_CALL identity" : "CALL identity\nRET");
                snprintf(entry, sizeof entry,
                    ".function main 0 2 0 int 1\n%sDUP\nSTORE_LOCAL 0\nCALL relay\n"
                    "AGG_PACK 0 0 0 1\nAGG_GET 0\nSTORE_LOCAL 1\n"
                    "LOAD_LOCAL 1\nPUSH_I64 0\nARR_GET\nTYPE_CHECK 4\nASSERT\n"
                    "LOAD_LOCAL 1\nPUSH_I64 0\nPUSH_BOOL 1\nARR_SET\nPOP\n"
                    "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nASSERT\n"
                    "LOAD_LOCAL 0\nSTORE_GLOBAL 0\nLOAD_GLOBAL 0\nPUSH_BOOL 0\nARR_PUSH\nPOP\n"
                    "LOAD_GLOBAL 0\nPUSH_I64 1\nARR_GET\nTYPE_CHECK 4\nASSERT\n"
                    "LOAD_GLOBAL 0\nPUSH_I64 1\nPUSH_BOOL 1\nARR_SET\nPOP\n"
                    "LOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\nASSERT\n"
                    "LOAD_GLOBAL 0\nPUSH_I64 99\nARR_GET\nTYPE_CHECK 0\nASSERT\n"
                    "LOAD_LOCAL 0\nARR_LEN\nPUSH_I64 2\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                    constructors[constructor]);
                snprintf(source, sizeof source, ".types 1 0 0\n.entry main\n%s%s",
                         before ? worker : entry, before ? entry : worker);
                NvmModule *m = assemble_ok(source, "boolean arrays across native boundaries");
                if (!m) continue;
                char *c = emit_or_fail(m, "I preserve distinct boolean array representations");
                if (c) {
                    int status = -1;
                    CHECK(compile_and_run(c, &status) == 0 && status == 0,
                          "I retain boolean tags, aliases and mutation through calls, fields and globals");
                    free(c);
                }
                nvm_module_free(m);
            }
        }
    }
    const char *bad[] = {
        "PUSH_I64 1\nARR_LITERAL 4 1\n",
        "PUSH_BOOL 1\nARR_LITERAL 1 1\n",
        "ARR_NEW 4\nPUSH_I64 1\nARR_PUSH\n",
        "ARR_NEW 1\nPUSH_BOOL 1\nARR_PUSH\n",
        "PUSH_BOOL 1\nARR_LITERAL 4 1\nPUSH_I64 0\nPUSH_I64 0\nARR_SET\n",
        "PUSH_I64 1\nARR_LITERAL 1 1\nPUSH_I64 0\nPUSH_BOOL 0\nARR_SET\n"
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
        char source[1024], error[512];
        snprintf(source, sizeof source, ".entry main\n.function main 0 0 0 int 1\n%sPOP\nPUSH_I64 0\nRET\n.end\n", bad[i]);
        NvmModule *m = assemble_ok(source, "incompatible boolean array elements");
        if (!m) continue;
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL, "I reject integer/boolean array element substitution");
        free(c); nvm_module_free(m);
    }
    for (int set = 0; set < 2; ++set) {
        char source[1024];
        snprintf(source, sizeof source,
            ".entry main\n.function main 0 0 0 int 1\nPUSH_BOOL 1\nARR_LITERAL 4 1\nSTORE_GLOBAL 0\n"
            "LOAD_GLOBAL 0\n%sPUSH_I64 1\n%s\nPOP\nPUSH_I64 0\nRET\n.end\n",
            set ? "PUSH_I64 0\n" : "", set ? "ARR_SET" : "ARR_PUSH");
        NvmModule *m = assemble_ok(source, "tagged boolean array wrong element");
        if (!m) continue;
        char *c = emit_or_fail(m, "I defer dynamic boolean element checks to consumption");
        if (c) {
            int status = 0;
            CHECK(compile_and_run(c, &status) == 0 && (status == -1 || status == 134),
                  "I trap wrong-tag writes through tagged boolean arrays");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_array_set_aliases_bounds_and_types(void) {
    for (int before = 0; before < 2; ++before) {
        for (int wrong = 0; wrong < 2; ++wrong) {
            char worker[1024], source[3072];
            snprintf(worker, sizeof worker,
                ".function update 1 1 0 int 1\nLOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\n%s"
                "AGG_PACK 0 0 0 1\nARR_SET\nPOP\nPUSH_I64 0\nRET\n.end\n",
                wrong ? "PUSH_I64 7\n" : "PUSH_STR updated\n");
            const char *entry = ".function main 0 1 0 int 1\nPUSH_STR old\nAGG_PACK 0 0 0 1\nARR_LITERAL 8 1\nSTORE_LOCAL 0\n"
                "LOAD_LOCAL 0\nAGG_PACK 0 1 0 1\nCALL update\nPOP\n"
                "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nPUSH_STR updated\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n";
            snprintf(source, sizeof source,
                ".string old \"old\"\n.string updated \"new\"\n.types 2 0 0\n.entry main\n%s%s",
                before ? worker : entry, before ? entry : worker);
            NvmModule *m = assemble_ok(source, "late record-array update facts");
            if (!m) continue;
            char err[512];
            char *c = nvm2c_emit(m, err, sizeof err);
            CHECK((c != NULL) == !wrong, "I distinguish unknown fields from incompatible record-array updates");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0 && status == 0,
                      "I preserve alias-visible updates through a projected record array");
                free(c);
            }
            nvm_module_free(m);
        }
    }
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

static void test_wrapper_names_are_ordinary_string_data(void) {
    const char *values[] = {"bin/nano_vm", "nvm_blob", "prefix /* nano_vm */ suffix",
                            "prefix // nvm_blob", "\\\"nano_vm", "nvm_blob\\\\"};
    for (size_t i = 0; i < sizeof values / sizeof values[0]; i++) {
        char assembly[512];
        snprintf(assembly, sizeof assembly,
            ".string text \"%s\"\n.entry main\n.function main 0 0 0 int 1\n"
            "PUSH_STR text\nSTR_LEN\nPUSH_I64 0\nI64_GT_S\nASSERT\n"
            "PUSH_I64 0\nRET\n.end\n", values[i]);
        NvmModule *module = assemble_ok(assembly, "ordinary wrapper-name data");
        CHECK(module != NULL, "wrapper-name string assembles");
        if (!module) continue;
        char error[256];
        char *source = nvm2c_emit(module, error, sizeof error);
        CHECK(source != NULL, "wrapper-name string emits native C");
        if (source) {
            int status = -1;
            CHECK(compile_and_run(source, &status) == 0, "wrapper-name data builds without VM linkage");
            CHECK(status == 0, "wrapper-name data executes as an ordinary string");
            free(source);
        }
        nvm_module_free(module);
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

static void test_local_initialization_tags(void) {
    const char *sources[] = {
        "LOAD_LOCAL 0\nTYPE_CHECK 0\nASSERT\nPUSH_I64 0\nRET\n",
        "PUSH_BOOL 0\nJMP_FALSE joined\nPUSH_BOOL 1\nSTORE_LOCAL 0\njoined:\n"
        "LOAD_LOCAL 0\nTYPE_CHECK 0\nASSERT\nPUSH_I64 0\nRET\n",
        "loop:\nLOAD_LOCAL 0\nPOP\nPUSH_BOOL 1\nSTORE_LOCAL 0\nPUSH_BOOL 0\nJMP_FALSE done\n"
        "JMP loop\ndone:\nPUSH_I64 0\nRET\n"
    };
    for (size_t i = 0; i < sizeof sources / sizeof sources[0]; ++i) {
        char source[1024], error[256] = {0};
        snprintf(source, sizeof source, ".entry main\n.function main 0 1 0 int 1\n%s.end\n", sources[i]);
        NvmModule *m = assemble_ok(source, "possibly uninitialized local");
        if (!m) continue;
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c != NULL, "I preserve void-before-store in native locals");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I execute void-before-store without inventing a scalar tag");
        }
        free(c); nvm_module_free(m);
    }
    for (int arm = 0; arm < 2; ++arm) {
        char source[1024];
        snprintf(source, sizeof source, ".entry main\n.function main 0 2 0 int 1\n"
            "PUSH_BOOL %d\nJMP_FALSE alternate\nPUSH_BOOL 1\nSTORE_LOCAL 0\nJMP joined\n"
            "alternate:\nPUSH_BOOL 0\nSTORE_LOCAL 0\njoined:\nPUSH_I64 0\nSTORE_LOCAL 1\n"
            "loop:\nLOAD_LOCAL 0\nTYPE_CHECK 4\nASSERT\nLOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\n"
            "STORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_I64 3\nI64_LT_S\nJMP_FALSE done\nJMP loop\n"
            "done:\nPUSH_I64 0\nRET\n.end\n", arm);
        NvmModule *m = assemble_ok(source, "definitely initialized branch and loop locals");
        if (!m) continue;
        char *c = emit_or_fail(m, "I accept initialization on every incoming path");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I retain initialized local facts through branch intersections and loop backedges");
            free(c);
        }
        nvm_module_free(m);
    }
    NvmModule *m = assemble_ok(
        ".entry main\n.function main 0 1 0 int 1\nPUSH_I64 0\nRET\nLOAD_LOCAL 0\nPOP\n.end\n",
        "unreachable local read");
    if (m) {
        char *c = emit_or_fail(m, "I ignore unreachable uninitialized local reads");
        free(c); nvm_module_free(m);
    }
}

static void test_scalar_runtime_tags(void) {
    const char *source =
        ".string key \"key\"\n.entry main\n"
        ".function main 0 2 0 int 1\n"
        "PUSH_BOOL 1\nCALL relay\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nTYPE_CHECK 4\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 1\nEQ\nBOOL_NOT\nASSERT\n"
        "HM_NEW 5 1\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nHM_GET\nPUSH_I64 0\nEQ\nBOOL_NOT\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nPUSH_I64 0\nHM_SET\nPUSH_STR key\nHM_GET\n"
        "PUSH_BOOL 0\nEQ\nBOOL_NOT\nASSERT\nPUSH_I64 0\nRET\n.end\n"
        ".function relay 1 1 0 bool 1\nLOAD_LOCAL 0\nTAIL_CALL identity\n.end\n"
        ".function identity 1 1 0 bool 1\nLOAD_LOCAL 0\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "scalar runtime tags");
    CHECK(m != NULL, "scalar runtime-tag fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "I preserve scalar tags through locals, calls, returns and map lookups");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0 && status == 0,
              "I distinguish bool from int and missing map values from integer zero");
        free(c);
    }
    nvm_module_free(m);

}


static void test_boolean_tags(void) {
    test_local_initialization_tags();
    const char *producers[] = {
        "PUSH_I64 1\nPUSH_I64 2\nI64_EQ\n", "PUSH_I64 1\nPUSH_I64 2\nI64_NE\n",
        "PUSH_I64 1\nPUSH_I64 2\nI64_LT_S\n", "PUSH_I64 1\nPUSH_I64 2\nI64_LE_S\n",
        "PUSH_I64 1\nPUSH_I64 2\nI64_GT_S\n", "PUSH_I64 1\nPUSH_I64 2\nI64_GE_S\n",
        "PUSH_BOOL 1\nPUSH_BOOL 0\nBOOL_AND\n", "PUSH_BOOL 1\nPUSH_BOOL 0\nBOOL_OR\n",
        "PUSH_STR text\nPUSH_STR part\nSTR_STARTS_WITH\n",
        "PUSH_STR text\nPUSH_STR part\nSTR_ENDS_WITH\n",
        "PUSH_STR text\nPUSH_STR part\nSTR_CONTAINS\n",
        "PUSH_I64 1\nTYPE_CHECK 1\n"
    };
    char producer_source[4096] = ".string text \"abc\"\n.string part \"a\"\n.entry main\n.function main 0 0 0 int 1\n";
    for (size_t i = 0; i < sizeof producers / sizeof producers[0]; ++i) {
        strcat(producer_source, producers[i]);
        strcat(producer_source, "TYPE_CHECK 4\nASSERT\n");
    }
    strcat(producer_source, "PUSH_I64 0\nRET\n.end\n");
    NvmModule *producer_module = assemble_ok(producer_source, "boolean-producing instructions");
    if (producer_module) {
        char *c = emit_or_fail(producer_module, "I assign boolean tags to predicates and comparisons");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0 && status == 0,
                  "I preserve boolean result tags for integer, boolean and string predicates");
            free(c);
        }
        nvm_module_free(producer_module);
    }
    const char *source =
        ".string key \"key\"\n.string yes \"true\"\n.string no \"false\"\n.entry main\n"
        ".function main 0 2 0 int 1\nPUSH_BOOL 1\nCALL relay\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nTYPE_CHECK 4\nASSERT\nLOAD_LOCAL 0\nTYPE_CHECK 1\nBOOL_NOT\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 1\nEQ\nBOOL_NOT\nASSERT\nPUSH_I64 1\nLOAD_LOCAL 0\nNE\nASSERT\n"
        "LOAD_LOCAL 0\nCAST_STRING\nPUSH_STR yes\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nBOOL_NOT\nCAST_STRING\nPUSH_STR no\nEQ\nASSERT\n"
        "LOAD_LOCAL 0\nCAST_INT\nTYPE_CHECK 1\nASSERT\n"
        "LOAD_LOCAL 0\nCALL pack\nAGG_GET 0\nTYPE_CHECK 4\nASSERT\n"
        "LOAD_LOCAL 0\nPRINTLN\nLOAD_LOCAL 0\nBOOL_NOT\nPRINTLN\n"
        "HM_NEW 5 1\nSTORE_LOCAL 1\nLOAD_LOCAL 1\nPUSH_STR key\nHM_GET\nPUSH_BOOL 0\nEQ\nBOOL_NOT\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nPUSH_I64 1\nHM_SET\nPUSH_STR key\nHM_GET\nDUP\n"
        "PUSH_BOOL 1\nEQ\nBOOL_NOT\nASSERT\nPUSH_I64 1\nEQ\nASSERT\n"
        "LOAD_LOCAL 1\nPUSH_STR key\nHM_HAS\nTYPE_CHECK 4\nASSERT\n"
        "PUSH_BOOL 1\nPUSH_BOOL 0\nJMP_FALSE alternate\nBOOL_NOT\nJMP joined\n"
        "alternate:\nDUP\nSWAP\nPOP\njoined:\nTYPE_CHECK 4\nASSERT\nPUSH_I64 0\nRET\n.end\n"
        ".function relay 1 1 0 bool 1\nLOAD_LOCAL 0\nTAIL_CALL identity\n.end\n"
        ".function identity 1 1 0 bool 1\nLOAD_LOCAL 0\nRET\n.end\n"
        ".function pack 1 1 0 struct 1\nLOAD_LOCAL 0\nAGG_PACK 0 0 0 1\nRET\n.end\n";
    NvmModule *m = assemble_ok(source, "boolean runtime tags");
    if (m) {
        char *c = emit_or_fail(m, "I preserve boolean tags across scalar and aggregate data flow");
        if (c) {
            int status = -1;
            char output[64];
            CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 && status == 0 &&
                  strcmp(output, "true\nfalse\n") == 0,
                  "I distinguish bool from int in equality, casts, printing, calls, joins and record fields");
            free(c);
        }
        nvm_module_free(m);
    }
    const char *bad[] = {
        "PUSH_BOOL 1\nRET\n",
        "PUSH_BOOL 1\nCALL add_int\nRET\n",
        "PUSH_I64 1\nCALL negate_bool\nCAST_INT\nRET\n",
        "PUSH_I64 1\nPUSH_BOOL 1\nCALL both_bool\nCAST_INT\nRET\n",
        "HM_NEW 5 1\nPUSH_STR key\nPUSH_BOOL 1\nHM_SET\nPOP\nPUSH_I64 0\nRET\n"
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
        char assembly[1024], error[256] = {0};
        snprintf(assembly, sizeof assembly, ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n%s.end\n"
            ".function add_int 1 1 0 int 1\nLOAD_LOCAL 0\nPUSH_I64 1\nI64_ADD\nRET\n.end\n"
            ".function negate_bool 1 1 0 bool 1\nLOAD_LOCAL 0\nBOOL_NOT\nRET\n.end\n"
            ".function both_bool 2 2 0 bool 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nBOOL_AND\nRET\n.end\n", bad[i]);
        m = assemble_ok(assembly, "invalid implicit boolean conversion");
        if (!m) continue;
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && error[0], "I reject implicit boolean/integer conversion at typed boundaries");
        free(c); nvm_module_free(m);
    }
}

static void test_module_initializer(void) {
    const char *tags[] = {"void 0", "int 1", "string 1", "hashmap 1"};
    const char *results[] = {"", "PUSH_I64 99\n", "PUSH_STR init\n", "HM_NEW 5 1\n"};
    const char *entry = ".function main 0 0 0 int 1\nPUSH_STR entry\nPRINTLN\nPUSH_I64 7\nRET\n.end\n";
    for (int before = 0; before < 2; ++before) {
        for (size_t kind = 0; kind < sizeof tags / sizeof tags[0]; ++kind) {
            char initializer[256], source[1024];
            snprintf(initializer, sizeof initializer,
                ".function __init__ 0 0 0 %s\nPUSH_STR init\nPRINTLN\n%sRET\n.end\n", tags[kind], results[kind]);
            snprintf(source, sizeof source, ".string init \"init\"\n.string entry \"entry\"\n.entry main\n%s%s",
                     before ? initializer : entry, before ? entry : initializer);
            NvmModule *m = assemble_ok(source, "module initializer ordering");
            if (!m) continue;
            char *c = emit_or_fail(m, "I emit module initializer calls before entry");
            if (c) {
                int status = -1;
                char output[64];
                CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 && status == 7 &&
                      strcmp(output, "init\nentry\n") == 0,
                      "I run the initializer first and discard its result, independently of definition order");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    const char *programs[] = {
        ".string init \"init\"\n.string entry \"entry\"\n.entry main\n"
        ".function main 0 0 0 int 1\nPUSH_STR entry\nPRINTLN\nPUSH_I64 0\nRET\n.end\n"
        ".function __init__ 0 0 0 void 0\nPUSH_STR init\nPRINTLN\nPUSH_BOOL 0\nASSERT\nRET\n.end\n",
        ".string init \"init\"\n.entry __init__\n.function __init__ 0 0 0 int 1\n"
        "PUSH_STR init\nPRINTLN\nPUSH_I64 0\nRET\n.end\n"
    };
    for (int same_entry = 0; same_entry < 2; ++same_entry) {
        NvmModule *m = assemble_ok(programs[same_entry], "initializer failure and entry identity");
        if (!m) continue;
        char *c = emit_or_fail(m, "I preserve initializer failure and entry identity semantics");
        if (c) {
            int status = -1;
            char output[64];
            CHECK(compile_and_run_capture(c, &status, output, sizeof output) == 0 &&
                  (same_entry ? status == 0 : status != 0) &&
                  strcmp(output, same_entry ? "init\ninit\n" : "init\n") == 0,
                  "I do not enter after initializer failure and do not skip a distinct entry invocation");
            free(c);
        }
        nvm_module_free(m);
    }
    NvmModule *m = assemble_ok(
        ".entry main\n.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n"
        ".function __init__ 1 1 0 int 1\nLOAD_LOCAL 0\nRET\n.end\n", "invalid initializer arity");
    if (m) {
        char error[256] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "zero-argument module initializer"),
              "I reject an initializer that requires arguments");
        free(c); nvm_module_free(m);
    }
}

static void test_tagged_string_array_writes(void) {
    const char *values[] = {"PUSH_STR text\nSTORE_GLOBAL 0\n",
                            "PUSH_I64 42\nSTORE_GLOBAL 0\n",
                            "PUSH_BOOL 1\nSTORE_GLOBAL 0\n", ""};
    for (int set = 0; set < 2; ++set) {
        for (int reverse = 0; reverse < 2; ++reverse) {
            for (int tail = 0; tail < 2; ++tail) {
                for (int tag = 0; tag < 4; ++tag) {
                    char main_fn[2048], helpers[1024], source[8192];
                    snprintf(main_fn, sizeof main_fn,
                        ".function main 0 2 0 int 1\n%s"
                        "PUSH_STR old\nARR_LITERAL 5 1\nSTORE_LOCAL 0\n"
                        "LOAD_LOCAL 0\nSTORE_LOCAL 1\nLOAD_LOCAL 0\nLOAD_GLOBAL 0\nCALL relay\nPOP\n"
                        "LOAD_LOCAL 1\nARR_LEN\nPUSH_I64 %d\nI64_EQ\nASSERT\n"
                        "LOAD_LOCAL 1\nPUSH_I64 %d\nARR_GET\nPUSH_STR text\nEQ\nASSERT\n"
                        "LOAD_GLOBAL 0\nTYPE_CHECK 5\nASSERT\n"
                        "LOAD_GLOBAL 0\nPUSH_STR text\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
                        values[tag], set ? 1 : 2, set ? 0 : 1);
                    snprintf(helpers, sizeof helpers,
                        ".function relay 2 2 0 array 1\nLOAD_LOCAL 0\nLOAD_LOCAL 1\n%s\n.end\n"
                        ".function write 2 2 0 array 1\nLOAD_LOCAL 0\n%sLOAD_LOCAL 1\n%s\nRET\n.end\n",
                        tail ? "TAIL_CALL write" : "CALL write\nRET",
                        set ? "PUSH_I64 0\n" : "", set ? "ARR_SET" : "ARR_PUSH");
                    snprintf(source, sizeof source, ".string text \"present\"\n.string old \"old\"\n.entry main\n%s%s",
                             reverse ? helpers : main_fn, reverse ? main_fn : helpers);
                    NvmModule *m = assemble_ok(source, "tagged native string-array write");
                    if (!m) continue;
                    char *c = emit_or_fail(m, "I emit checked tagged string-array writes");
                    if (c) {
                        CHECK(strstr(c, "nvalue_require_string(v[") != NULL,
                              "I check the tag before writing native string storage");
                        int status = 0;
                        CHECK(compile_and_run(c, &status) == 0 && (tag == 0 ? status == 0 : status == -1),
                              "I preserve aliases and source tags, and trap non-string or absent writes");
                        free(c);
                    }
                    nvm_module_free(m);
                }
            }
        }
        char source[1024];
        snprintf(source, sizeof source,
            ".string key \"key\"\n.entry main\n.function main 0 0 0 int 1\n"
            "PUSH_STR key\nARR_LITERAL 5 1\n%sHM_NEW 5 1\nPUSH_STR key\nHM_GET\n%s\nPOP\nPUSH_I64 0\nRET\n.end\n",
            set ? "PUSH_I64 0\n" : "", set ? "ARR_SET" : "ARR_PUSH");
        NvmModule *m = assemble_ok(source, "incompatible known optional payload write");
        if (m) {
            char error[512] = {0};
            char *c = nvm2c_emit(m, error, sizeof error);
            CHECK(c == NULL && strstr(error, "shape"), "I reject known integer payloads at string-array writes");
            free(c);
            nvm_module_free(m);
        }
    }
}

static void test_record_temporary_storage_is_function_sized(void) {
    const char *src =
        ".string seven \"seven\"\n"
        ".entry 0\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  PUSH_STR seven\n"
        "  AGG_PACK 0 0 0 2\n"
        "  AGG_GET 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "record temporary sizing fixture");
    CHECK(m != NULL, "record temporary sizing fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c sizes record temporary storage");
    if (c) {
        CHECK(strstr(c, "nrec_t r[256]") == NULL,
              "generated frames do not reserve the global record temporary limit");
        CHECK(strstr(c, "nrec_t *r = 1 ? calloc(1, sizeof *r) : NULL") != NULL,
              "generated frame reserves only its one record temporary");
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0,
              "function-sized record storage compiles and runs");
        CHECK(status == 7, "function-sized record storage preserves the result");
        free(c);
    }
    nvm_module_free(m);
}

static void test_uncalled_record_parameter_needs_no_invented_shape(void) {
    const char *src =
        ".entry 1\n"
        ".function uncalled 1 1 0 struct 1\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 0\n"
        "  PUSH_I64 1\n"
        "  AGG_PACK 0 0 0 2\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_I64 7\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "uncalled record parameter fixture");
    CHECK(m != NULL, "uncalled record parameter fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c ignores an uncalled record parameter shape");
    if (c) {
        CHECK(strstr(c, "nl_uncalled") == NULL,
              "uncalled function is absent from generated C");
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0,
              "reachable generated C compiles and runs");
        CHECK(status == 7, "entry result survives removal of uncalled code");
        free(c);
    }
    nvm_module_free(m);
}

static void test_cast_int_updates_classifier_stack(void) {
    const char *src =
        ".string value \"42\"\n"
        ".entry 1\n"
        ".function take_int 1 1 0 int 1\n"
        "  LOAD_LOCAL 0\n"
        "  RET\n"
        ".end\n"
        ".function main 0 0 0 int 1\n"
        "  PUSH_STR value\n"
        "  CAST_INT\n"
        "  CALL take_int\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "CAST_INT classifier fixture");
    CHECK(m != NULL, "CAST_INT classifier fixture assembles");
    if (!m) return;
    char err[256];
    char *c = nvm2c_emit(m, err, sizeof err);
    CHECK(c != NULL, "CAST_INT replaces its classifier operand with an integer");
    if (c) {
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0, "CAST_INT generated C compiles and runs");
        CHECK(status == 42, "CAST_INT converts a string before a later CALL");
        free(c);
    } else {
        printf("    nvm2c error: %s\n", err);
    }
    nvm_module_free(m);
}

static void test_array_record_field_keeps_runtime_representation(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_I64 7\n"
        "  ARR_LITERAL 1 1\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  PUSH_I64 0\n"
        "  AGG_PACK 0 0 0 11\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  AGG_GET 0\n"
        "  ARR_LEN\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "array record field fixture");
    CHECK(m != NULL, "array record field fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c preserves an array-valued record field");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "array record field C compiles and runs");
    CHECK(status == 1, "ARR_LEN reads the projected field's preserved array representation");
    free(c);
    nvm_module_free(m);
}

static void test_array_result_kinds_cross_calls(void) {
    const char *src =
        ".string hi \"hi\"\n"
        ".entry 4\n"
        ".function ints 0 0 0 array 1\n"
        "  ARR_NEW 1\n"
        "  RET\n"
        ".end\n"
        ".function strings 0 0 0 array 1\n"
        "  ARR_NEW 5\n"
        "  RET\n"
        ".end\n"
        ".function forward 0 0 0 array 1\n"
        "  TAIL_CALL strings\n"
        ".end\n"
        ".function records 0 0 0 array 1\n"
        "  ARR_NEW 8\n"
        "  RET\n"
        ".end\n"
        ".function main 0 4 0 int 1\n"
        "  CALL ints\n"
        "  STORE_LOCAL 0\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 7\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  CALL forward\n"
        "  STORE_LOCAL 1\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_STR hi\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  STR_LEN\n"
        "  POP\n"
        "  CALL records\n"
        "  STORE_LOCAL 2\n"
        "  LOAD_LOCAL 2\n"
        "  PUSH_I64 1\n"
        "  PUSH_STR hi\n"
        "  AGG_PACK 0 0 0 2\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 2\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  AGG_GET 0\n"
        "  POP\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  ARR_GET\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "array result kinds fixture");
    CHECK(m != NULL, "array result kinds fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c infers array result kinds");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    CHECK(strstr(c, "static narr_t nl_ints") != NULL,
          "integer-array result uses narr_t");
    CHECK(strstr(c, "static nsarr_t nl_strings") != NULL,
          "string-array result uses nsarr_t");
    CHECK(strstr(c, "static nsarr_t nl_forward") != NULL,
          "tail-call array result uses the callee kind");
    CHECK(strstr(c, "static nrarr_t nl_records") != NULL,
          "record-array result keeps nrarr_t");
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0, "array result kinds C compiles and runs");
    CHECK(status == 7, "array result kinds preserve native behavior");
    free(c);
    nvm_module_free(m);
}

static void test_array_growth_has_no_process_wide_arena_limit(void) {
    const char *src =
        ".entry 0\n"
        ".function main 0 2 0 int 1\n"
        "  ARR_NEW 1\n"
        "  STORE_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 1\n"
        "loop:\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 70000\n"
        "  I64_LT_S\n"
        "  JMP_FALSE done\n"
        "  LOAD_LOCAL 0\n"
        "  LOAD_LOCAL 1\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 1\n"
        "  I64_ADD\n"
        "  STORE_LOCAL 1\n"
        "  JMP loop\n"
        "done:\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  PUSH_I64 70000\n"
        "  EQ\n"
        "  ASSERT\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "large array growth fixture");
    CHECK(m != NULL, "large array growth fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits unbounded array growth");
    if (c) {
        int status = -1;
        CHECK(strstr(c, "narr_arena") == NULL,
              "integer arrays do not share a fixed process-wide arena");
        CHECK(compile_and_run(c, &status) == 0,
              "large array growth C compiles and runs");
        CHECK(status == 1, "an array grows past the former 65,536-element limit");
        free(c);
    }
    nvm_module_free(m);
}

static void test_string_array_growth_has_no_process_wide_arena_limit(void) {
    const char *src =
        ".string value \"x\"\n"
        ".entry 0\n"
        ".function main 0 2 0 int 1\n"
        "  ARR_NEW 5\n"
        "  STORE_LOCAL 0\n"
        "  PUSH_I64 0\n"
        "  STORE_LOCAL 1\n"
        "loop:\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 70000\n"
        "  I64_LT_S\n"
        "  JMP_FALSE done\n"
        "  LOAD_LOCAL 0\n"
        "  PUSH_STR value\n"
        "  ARR_PUSH\n"
        "  POP\n"
        "  LOAD_LOCAL 1\n"
        "  PUSH_I64 1\n"
        "  I64_ADD\n"
        "  STORE_LOCAL 1\n"
        "  JMP loop\n"
        "done:\n"
        "  LOAD_LOCAL 0\n"
        "  ARR_LEN\n"
        "  PUSH_I64 70000\n"
        "  EQ\n"
        "  ASSERT\n"
        "  PUSH_I64 1\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "large string array growth fixture");
    CHECK(m != NULL, "large string array growth fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits unbounded string array growth");
    if (c) {
        int status = -1;
        CHECK(strstr(c, "nsarr_arena") == NULL,
              "string arrays do not share a fixed process-wide arena");
        CHECK(compile_and_run(c, &status) == 0,
              "large string array growth C compiles and runs");
        CHECK(status == 1, "a string array grows past the former 65,536-element limit");
        free(c);
    }
    nvm_module_free(m);
}

static void test_uncalled_array_parameter_projection_storage(void) {
    const char *types[] = {"int", "bool", "float", "string"};
    const char *initial[] = {"PUSH_I64 0", "PUSH_BOOL 0", "PUSH_F64 0", "PUSH_STR 0"};
    for (unsigned type = 0; type < 4; ++type) {
        for (unsigned order = 0; order < 2; ++order) {
            char helper[512], source[2 * sizeof helper + 64];
            snprintf(helper, sizeof helper,
                ".function unused 1 3 0 %s 1\n"
                " %s\n STORE_LOCAL 1\n"
                " LOAD_LOCAL 0\n PUSH_I64 0\n ARR_GET\n STORE_LOCAL 2\n"
                " LOAD_LOCAL 2\n STORE_LOCAL 1\n LOAD_LOCAL 1\n RET\n.end\n"
                ".parameters %u array\n", types[type], initial[type], order);
            const char *main = ".function main 0 0 0 int 1\n PUSH_I64 0\n RET\n.end\n";
            snprintf(source, sizeof source, ".string \"\"\n.entry %u\n%s%s", 1 - order,
                     order ? main : helper, order ? helper : main);
            NvmModule *m = assemble_ok(source, "I retain uncalled array parameter projections");
            CHECK(m != NULL, "I assemble both declaration orders for every scalar result");
            if (!m) continue;
            char *c = emit_or_fail(m, "I retain tagged storage for an uncalled array parameter");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0,
                      "I compile all retained helper bodies with strict diagnostics");
                CHECK(status == 0, "I preserve the independent entry result");
                free(c);
            }
            nvm_module_free(m);
        }
    }
}

static void test_uncalled_array_record_consumer_constraints(void) {
    for (unsigned order = 0; order < 2; ++order) {
      for (unsigned copies = 0; copies < 3; ++copies) {
        char reader[512];
        const char *aliases[] = {"", " STORE_LOCAL 1\n LOAD_LOCAL 1\n",
            " STORE_LOCAL 1\n LOAD_LOCAL 1\n STORE_LOCAL 2\n LOAD_LOCAL 2\n"};
        snprintf(reader, sizeof reader,
            ".function reader 1 %u 0 int 1\n"
            " LOAD_LOCAL 0\n PUSH_I64 0\n ARR_GET\n%s CALL consume\n RET\n.end\n",
            copies + 1, aliases[copies]);
        const char *consumer =
            ".function consume 1 1 0 int 1\n"
            " LOAD_LOCAL 0\n AGG_GET 0\n RET\n.end\n";
        char source[2 * sizeof reader + 128];
        snprintf(source, sizeof source,
            ".entry 2\n%s%s.parameters %u array\n"
            ".function main 0 0 0 int 1\n PUSH_I64 0\n RET\n.end\n",
            order ? consumer : reader, order ? reader : consumer, order);
        NvmModule *m = assemble_ok(source, "I retain record consumer constraints before array fallback");
        CHECK(m != NULL, "I assemble both record-consumer declaration orders");
        if (!m) continue;
        char *c = emit_or_fail(m, "I infer a record array from its consumer before choosing tagged storage");
        if (c) {
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "I compile the retained record-array reader");
            CHECK(status == 0, "I preserve its independent entry point");
            const char *prefix = "#define main generated_main\n";
            const char *suffix = "\n#undef main\nint main(void) {\n"
                " nrec_t record = {.n = 1}; record.f[0] = 42;\n"
                " nrarr_s array = {.data = &record, .len = 1};\n"
                " return nl_reader(&array) != 42;\n}\n";
            size_t size = strlen(prefix) + strlen(c) + strlen(suffix) + 1;
            char *probe = malloc(size);
            CHECK(probe != NULL, "I allocate the record-array execution probe");
            if (probe) {
                snprintf(probe, size, "%s%s%s", prefix, c, suffix);
                CHECK(compile_and_run(probe, &status) == 0, "I compile copied record-array consumers");
                CHECK(status == 0, "I execute record reads through each local copy chain");
                free(probe);
            }
            free(c);
        }
        nvm_module_free(m);
      }
    }
}

static void test_void_local_flows_through_branches_loops_and_calls(void) {
    const char *src =
        ".entry 1\n"
        ".function consume 1 1 0 void 0\n"
        "  LOAD_LOCAL 0\n"
        "  TYPE_CHECK 0\n"
        "  ASSERT\n"
        "  RET\n"
        ".end\n"
        ".function main 0 1 0 int 1\n"
        "  PUSH_BOOL 0\n"
        "  JMP_FALSE after_store\n"
        "  PUSH_I64 9\n"
        "  STORE_LOCAL 0\n"
        "after_store:\n"
        "  LOAD_LOCAL 0\n"
        "  TYPE_CHECK 0\n"
        "  ASSERT\n"
        "  LOAD_LOCAL 0\n"
        "  CALL consume\n"
        "loop:\n"
        "  LOAD_LOCAL 0\n"
        "  TYPE_CHECK 0\n"
        "  ASSERT\n"
        "  PUSH_BOOL 0\n"
        "  JMP_FALSE done\n"
        "  JMP loop\n"
        "done:\n"
        "  PUSH_I64 0\n"
        "  RET\n"
        ".end\n";
    NvmModule *m = assemble_ok(src, "void local data-flow fixture");
    CHECK(m != NULL, "void local data-flow fixture assembles");
    if (!m) return;
    char *c = emit_or_fail(m, "nvm2c emits C for void local data flow");
    if (!c) {
        nvm_module_free(m);
        return;
    }
    int status = -1;
    CHECK(compile_and_run(c, &status) == 0,
          "conditional, loop, and call void-local C compiles and runs");
    CHECK(status == 0, "void local can be passed and discarded with VM semantics");
    free(c);
    nvm_module_free(m);
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
    CHECK(strstr(err, "shape") != NULL,
          "string-array payload refusal names the required runtime kind");
    free(c);
    nvm_module_free(m);
}

static void test_mixed_array_push_helpers(void) {
    const char *fixtures[] = {
        ".string text \"hello\"\n.entry main\n.function main 0 0 0 int 1\n"
        "PUSH_I64 9\nARR_LITERAL 1 1\nPOP\nARR_LITERAL 5 0\n"
        "PUSH_STR text\nARR_PUSH\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n",
        ".string text \"hello\"\n.entry main\n.function main 0 0 0 int 1\n"
        "PUSH_STR text\nARR_LITERAL 5 1\nPOP\nARR_LITERAL 1 0\n"
        "PUSH_I64 9\nARR_PUSH\nARR_LEN\nPUSH_I64 1\nI64_EQ\nASSERT\nPUSH_I64 0\nRET\n.end\n"
    };
    for (size_t i = 0; i < sizeof fixtures / sizeof fixtures[0]; ++i) {
        NvmModule *m = assemble_ok(fixtures[i], "mixed array push helpers");
        CHECK(m != NULL, "mixed array push fixture assembles");
        if (!m) continue;
        char error[512] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c != NULL, "mixed array push fixture translates");
        if (c) {
            CHECK(strstr(c, "(void)narr_push;") != NULL,
                  "mixed array output references its integer push helper");
            CHECK(strstr(c, "(void)nsarr_push;") != NULL,
                  "mixed array output references its string push helper");
            int status = -1;
            CHECK(compile_and_run(c, &status) == 0, "mixed array helpers compile with strict warnings");
            CHECK(status == 0, "mixed array push retains its value");
            free(c);
        }
        nvm_module_free(m);
    }
}

static void test_nested_array_field_writes(void) {
    const unsigned tags[] = {TAG_INT, TAG_FLOAT, TAG_BOOL, TAG_STRING};
    const char *values[] = {"PUSH_I64 7\n", "PUSH_F64 2.5\n", "PUSH_BOOL 1\n", "PUSH_STR leaf\n"};
    const char *equal[] = {"I64_EQ\n", "F64_EQ\n", "EQ\n", "EQ\n"};
    const char *view = ".function view 1 1 0 array 1\nLOAD_LOCAL 0\nAGG_GET 0\nRET\n.end\n";
    for (unsigned type = 0; type < 4; ++type) {
        for (unsigned order = 0; order < 2; ++order) {
            char body[4096], source[2 * sizeof body + 64];
            snprintf(body, sizeof body,
                ".function main 0 2 0 int 1\n"
                "%sARR_LITERAL %u 1\nARR_LITERAL 7 1\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n"
                /* I replace through a tagged field and observe the same handle through a call. */
                "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_LITERAL %u 0\nARR_SET\nPOP\n"
                "LOAD_LOCAL 0\nCALL view\nPUSH_I64 0\nARR_GET\nDUP\nSTORE_LOCAL 1\n"
                "ARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n"
                "LOAD_LOCAL 1\n%sARR_PUSH\nPOP\n"
                "LOAD_LOCAL 0\nAGG_GET 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\n%s%sASSERT\n"
                /* I also box a concrete child when appending through the tagged field. */
                "LOAD_LOCAL 0\nAGG_GET 0\n%sARR_LITERAL %u 1\nARR_PUSH\nPOP\n"
                "LOAD_LOCAL 0\nCALL view\nPUSH_I64 1\nARR_GET\nPUSH_I64 0\nARR_GET\n%s%sASSERT\n"
                "PUSH_I64 0\nRET\n.end\n",
                values[type], tags[type], tags[type], values[type], values[type], equal[type],
                values[type], tags[type], values[type], equal[type]);
            snprintf(source, sizeof source, ".string leaf \"leaf\"\n.entry main\n%s%s",
                     order ? body : view, order ? view : body);
            NvmModule *m = assemble_ok(source, "nested array field writes");
            if (!m) continue;
            char *c = emit_or_fail(m, "I retain nested array fields across declaration orders");
            if (c) {
                int status = -1;
                CHECK(compile_and_run(c, &status) == 0, "I compile nested field replacement and append");
                CHECK(status == 0, "I preserve child types and alias-visible nested writes");
                free(c);
            }
            nvm_module_free(m);
        }
    }
    for (unsigned push = 0; push < 2; ++push) {
        char source[1024];
        snprintf(source, sizeof source,
            ".entry main\n.function main 0 1 0 int 1\n"
            "ARR_LITERAL 1 0\nARR_LITERAL 7 1\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 0\n"
            "LOAD_LOCAL 0\nAGG_GET 0\n%sARR_LITERAL 5 0\n%s\nPOP\nPUSH_I64 0\nRET\n.end\n",
            push ? "" : "PUSH_I64 0\n", push ? "ARR_PUSH" : "ARR_SET");
        NvmModule *m = assemble_ok(source, "incompatible nested replacement");
        if (!m) continue;
        char error[512] = {0};
        char *c = nvm2c_emit(m, error, sizeof error);
        CHECK(c == NULL && strstr(error, "shape"), "I reject conflicting nested element types through tagged fields");
        free(c);
        nvm_module_free(m);
    }

}

static void test_nested_scalar_arrays(void) {
    const char *source =
        ".string leaf \"leaf\"\n"
        ".entry main\n"
        ".function rows 0 0 0 array 1\n"
        "PUSH_I64 7\nPUSH_I64 8\nARR_LITERAL 1 2\n"
        "PUSH_I64 9\nARR_LITERAL 1 1\n"
        "ARR_LITERAL 7 2\nRET\n.end\n"
        ".function first 1 1 0 array 1\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nRET\n.end\n"
        ".function stress 0 2 0 int 1\n"
        "ARR_NEW 7\nSTORE_LOCAL 0\nPUSH_I64 0\nSTORE_LOCAL 1\n"
        "loop:\nLOAD_LOCAL 0\nLOAD_LOCAL 1\nARR_LITERAL 1 1\nARR_PUSH\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 1\nPUSH_I64 1\nI64_ADD\nDUP\nSTORE_LOCAL 1\n"
        "PUSH_I64 1200\nI64_LT_S\nJMP_TRUE loop\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 1199\nARR_GET\nPUSH_I64 0\nARR_GET\nPUSH_I64 1199\nI64_EQ\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n"
        ".function main 0 4 0 int 1\n"
        "CALL stress\nPOP\n"
        "CALL rows\nCALL first\nPUSH_I64 1\nARR_GET\n"
        "PUSH_I64 8\nI64_EQ\nASSERT\n"
        "CALL rows\nAGG_PACK 0 0 0 1\nSTORE_LOCAL 3\n"
        "LOAD_LOCAL 3\nAGG_GET 0\nPUSH_I64 1\nARR_GET\n"
        "PUSH_I64 0\nARR_GET\nPUSH_I64 9\nI64_EQ\nASSERT\n"
        "CALL rows\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPUSH_I64 0\nARR_GET\nSTORE_LOCAL 1\n"
        "LOAD_LOCAL 1\nPUSH_I64 1\nARR_GET\nPUSH_I64 8\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nPUSH_I64 1\nARR_GET\nPUSH_I64 0\nARR_GET\n"
        "PUSH_I64 9\nI64_EQ\nASSERT\n"
        "PUSH_I64 17\nPUSH_I64 18\nARR_LITERAL 1 2\n"
        "LOAD_LOCAL 0\nSWAP\nARR_PUSH\nSTORE_LOCAL 0\n"
        "LOAD_LOCAL 0\nPUSH_I64 2\nARR_GET\nSTORE_LOCAL 2\n"
        "LOAD_LOCAL 2\nPUSH_I64 0\nPUSH_I64 19\nARR_SET\nPOP\n"
        "LOAD_LOCAL 0\nPUSH_I64 2\nARR_GET\nPUSH_I64 0\nARR_GET\n"
        "PUSH_I64 19\nI64_EQ\nASSERT\n"
        "LOAD_LOCAL 0\nARR_LITERAL 7 1\nPUSH_I64 0\nARR_GET\n"
        "PUSH_I64 2\nARR_GET\nPUSH_I64 1\nARR_GET\n"
        "PUSH_I64 18\nI64_EQ\nASSERT\n"
        "PUSH_STR leaf\nARR_LITERAL 5 1\nARR_LITERAL 7 1\n"
        "PUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\nPUSH_STR leaf\nEQ\nASSERT\n"
        "PUSH_BOOL 1\nARR_LITERAL 4 1\nARR_LITERAL 7 1\n"
        "PUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\nASSERT\n"
        "PUSH_F64 1.5\nARR_LITERAL 3 1\nARR_LITERAL 7 1\n"
        "PUSH_I64 0\nARR_GET\nPUSH_I64 0\nARR_GET\nPUSH_F64 1.5\nF64_EQ\nASSERT\n"
        "ARR_NEW 7\nARR_LEN\nPUSH_I64 0\nI64_EQ\nASSERT\n"
        "PUSH_I64 0\nRET\n.end\n";
    NvmModule *module = assemble_ok(source, "nested scalar arrays");
    CHECK(module != NULL, "nested scalar-array fixture assembles");
    if (!module) return;
    char error[512] = {0};
    char *c = nvm2c_emit(module, error, sizeof error);
    CHECK(c != NULL, "nested scalar arrays translate to native C");
    if (!c) {
        fprintf(stderr, "    nested scalar arrays: %s\n", error);
    } else {
        CHECK(strstr(c, "naarr_lit") != NULL,
              "nested scalar arrays retain an explicit recursive carrier");
        int status = -1;
        CHECK(compile_and_run(c, &status) == 0,
              "nested scalar arrays compile and run with strict warnings");
        CHECK(status == 0,
              "nested scalar arrays preserve calls, aliases, writes and recursive reads");
        free(c);
    }
    nvm_module_free(module);
}

int main(int argc, char **argv) {
    test_exact_aggregate_callback_provenance();
    test_indirect_target_inference_order();
    test_nested_array_field_writes();
    test_nested_scalar_arrays();
    test_mixed_array_push_helpers();
    test_record_temporary_storage_is_function_sized();
    test_uncalled_record_parameter_needs_no_invented_shape();
    test_cast_int_updates_classifier_stack();
    test_array_record_field_keeps_runtime_representation();
    test_array_result_kinds_cross_calls();
    test_array_growth_has_no_process_wide_arena_limit();
    test_string_array_growth_has_no_process_wide_arena_limit();
    test_uncalled_array_parameter_projection_storage();
    test_uncalled_array_record_consumer_constraints();
    test_void_local_flows_through_branches_loops_and_calls();
    test_tagged_string_array_writes();
    test_boolean_arrays();
    test_map_aggregate_fields();
    test_tagged_host_arguments();
    test_generic_ordering();
    test_boolean_tags();
    test_scalar_runtime_tags();
    test_module_initializer();
    test_self_tail_restart_preserves_values();
    test_self_tail_rejects_malformed_calls();
    test_array_set_aliases_bounds_and_types();
    test_string_edges_run_as_native_c();
    test_wrapper_names_are_ordinary_string_data();
    printf("\n[nvm2c] structured C11 from NanoISA...\n\n");
    test_1024_locals_compile_and_run();
    test_arity_exceeding_locals_is_refused();
    test_record_result_crosses_direct_call();
    test_add_is_structured_c_and_runs();
    test_store_load_local();
    test_generic_comparisons_are_typed();
    test_float_comparison_transport();
    test_globals_cross_functions_and_preserve_identity();
    test_uninitialized_global_result_traps();
    test_tagged_scalar_local_assignments();
    test_projected_global_stores();
    test_builtin_host_imports();
    test_character_host_imports();
    test_builtin_text_reader();
    test_builtin_text_writer();
    test_builtin_filesystem_predicates();
    test_builtin_removal_and_rename();
    test_builtin_identity();
    test_builtin_normalize();
    test_builtin_capture();
    test_builtin_from_char();
    test_builtin_temp_directory();
    test_tagged_record_array();
    test_artifact_array_import_is_not_a_builtin();
    test_owned_artifact_execution();
    test_real_walk_artifact();
    test_call_extern_is_refused();
    test_str_to_upper_is_refused();
    test_push_str_len_runs_without_nano_vm();
    test_str_concat_len_runs_without_nano_vm();
    test_greeting_runs_without_nano_vm();
    test_glue_runs_without_nano_vm();
    test_arr_set_runs_natively();
    test_boxed_array_indices();
    test_boxed_array_arguments();
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
    test_string_array_push_tags_parameter_and_preserves_alias();
    test_string_array_push_rejects_integer_payload();
    test_nested_record_values();
    test_null_module();
    test_choose_then_runs_without_nano_vm();
    test_choose_else_runs_without_nano_vm();
    test_loop_sum_runs_without_nano_vm();
    test_tail_call_runs_without_nano_vm();
    test_classifier_branch_stack();
    test_classifier_deep_stack();
    test_wide_aggregate_calls();
    test_array_valued_record_fields();
    test_record_array_alias_shapes();
    test_shared_code_shape_scopes();
    test_emitter_deep_stacks();
    test_emitter_many_temporaries();
    test_record_array_fact_namespaces();
    test_classifier_unreachable_and_invalid_joins();
    test_classifier_local_bounds();
    test_loop_carried_stack();
    test_variant_tags_and_payloads();
    test_aggregate_runtime_kind_checks();
    test_aggregate_call_facts();
    test_unrepresentable_call_facts();
    test_recursive_and_branch_record_facts();
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
