/*
 * wrapper_gen.c - Generate native binary wrappers for .nvm bytecode
 *
 * Strategy:
 * 1. Serialize the .nvm blob as a C hex byte array
 * 2. Generate a minimal C wrapper that deserializes and runs it
 * 3. Compile the wrapper linking against pre-built .o files
 * 4. Clean up the temp C file
 *
 * The generated wrapper mirrors the --run code path in nanovirt/main.c.
 */

#include "wrapper_gen.h"
#include "../nanoisa/nvm_format.h"
#include "../nanolang.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>

/* ========================================================================
 * Object File Discovery
 * ======================================================================== */

/*
 * Find the directory containing pre-built .o files.
 * Search order:
 * 1. NANO_VIRT_LIB environment variable
 * 2. <binary_dir>/../obj/ (relative to nano_virt binary)
 * 3. ./obj/ from CWD
 *
 * Returns a malloc'd string or NULL.
 */
static char *find_obj_dir(void) {
    /* 1. Environment variable */
    const char *env_dir = getenv("NANO_VIRT_LIB");
    if (env_dir) {
        if (access(env_dir, R_OK) == 0) {
            return strdup(env_dir);
        }
    }

    /* 2. Relative to binary - use /proc/self/exe on Linux, _NSGetExecutablePath on macOS */
    char exe_path[4096];
    memset(exe_path, 0, sizeof(exe_path));

#ifdef __APPLE__
    {
        extern int _NSGetExecutablePath(char *buf, uint32_t *bufsize);
        uint32_t size = sizeof(exe_path);
        if (_NSGetExecutablePath(exe_path, &size) == 0) {
            char *dir = dirname(exe_path);
            char obj_path[4096];
            snprintf(obj_path, sizeof(obj_path), "%s/../obj", dir);
            if (access(obj_path, R_OK) == 0) {
                /* Resolve to canonical path */
                char *real = realpath(obj_path, NULL);
                if (real) return real;
                return strdup(obj_path);
            }
        }
    }
#elif defined(__linux__)
    {
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len > 0) {
            exe_path[len] = '\0';
            char *dir = dirname(exe_path);
            char obj_path[4096];
            snprintf(obj_path, sizeof(obj_path), "%s/../obj", dir);
            if (access(obj_path, R_OK) == 0) {
                char *real = realpath(obj_path, NULL);
                if (real) return real;
                return strdup(obj_path);
            }
        }
    }
#endif

    /* 3. CWD fallback */
    if (access("obj", R_OK) == 0) {
        char *real = realpath("obj", NULL);
        if (real) return real;
        return strdup("obj");
    }

    return NULL;
}

/* ========================================================================
 * C Wrapper Generation
 * ======================================================================== */

static bool write_wrapper_c(FILE *f, const NvmModule *module,
                             const uint8_t *blob, uint32_t blob_size,
                             const ASTNode *program) {
    /* Header includes */
    fprintf(f, "/* Auto-generated NVM wrapper - do not edit */\n");
    fprintf(f, "#include \"nanoisa/nvm_format.h\"\n");
    fprintf(f, "#include \"nanovm/vm.h\"\n");
    fprintf(f, "#include \"nanovm/vm_ffi.h\"\n");
    fprintf(f, "#include \"nanovm/value.h\"\n");
    fprintf(f, "#include <stdio.h>\n");
    fprintf(f, "#include <stdlib.h>\n");
    fprintf(f, "#include <string.h>\n\n");

    /* Globals expected by runtime/cli.c */
    fprintf(f, "int g_argc = 0;\n");
    fprintf(f, "char **g_argv = NULL;\n\n");

    /* Forward declaration for FFI module loading */
    fprintf(f, "extern bool ffi_load_module(const char *module_name, const char *module_path,\n");
    fprintf(f, "                            void *env, bool verbose);\n\n");

    /* Embedded blob as hex array */
    fprintf(f, "static const unsigned char nvm_blob[%u] = {\n", blob_size);
    for (uint32_t i = 0; i < blob_size; i++) {
        if (i % 16 == 0) fprintf(f, "    ");
        fprintf(f, "0x%02x", blob[i]);
        if (i + 1 < blob_size) fprintf(f, ",");
        if (i % 16 == 15 || i + 1 == blob_size) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");

    /* Main function - mirrors main.c:230-267 */
    fprintf(f, "int main(int argc, char **argv) {\n");
    fprintf(f, "    g_argc = argc;\n");
    fprintf(f, "    g_argv = argv;\n\n");

    /* Deserialize */
    fprintf(f, "    NvmModule *module = nvm_deserialize(nvm_blob, %u);\n", blob_size);
    fprintf(f, "    if (!module) {\n");
    fprintf(f, "        fprintf(stderr, \"error: failed to deserialize embedded bytecode\\n\");\n");
    fprintf(f, "        return 1;\n");
    fprintf(f, "    }\n\n");

    /* FFI init + module loading (only if there are imports) */
    if (module->import_count > 0) {
        fprintf(f, "    /* Initialize FFI and load modules */\n");
        fprintf(f, "    vm_ffi_init();\n\n");

        /* Load modules from import table */
        fprintf(f, "    /* Load modules referenced in import table */\n");
        fprintf(f, "    for (uint32_t i = 0; i < module->import_count; i++) {\n");
        fprintf(f, "        const char *mod_name = nvm_get_string(module, module->imports[i].module_name_idx);\n");
        fprintf(f, "        if (mod_name && mod_name[0] != '\\0') {\n");
        fprintf(f, "            vm_ffi_load_module(mod_name);\n");
        fprintf(f, "        }\n");
        fprintf(f, "    }\n\n");

        /* Scan AST_IMPORT nodes for module paths */
        bool has_imports = false;
        if (program) {
            for (int i = 0; i < program->as.program.count; i++) {
                ASTNode *item = program->as.program.items[i];
                if (item->type == AST_IMPORT && item->as.import_stmt.module_path) {
                    if (!has_imports) {
                        fprintf(f, "    /* Load modules by path from source imports */\n");
                        has_imports = true;
                    }
                    fprintf(f, "    vm_ffi_load_module(\"%s\");\n",
                            item->as.import_stmt.module_path);
                }
            }
            if (has_imports) fprintf(f, "\n");
        }

        /* Known-modules table for bare extern functions */
        fprintf(f, "    /* Load well-known standard modules for bare extern fns */\n");
        fprintf(f, "    static const struct { const char *prefix; const char *module; } known_modules[] = {\n");
        fprintf(f, "        {\"path_\",    \"std/fs\"},\n");
        fprintf(f, "        {\"fs_\",      \"std/fs\"},\n");
        fprintf(f, "        {\"file_\",    \"std/fs\"},\n");
        fprintf(f, "        {\"dir_\",     \"std/fs\"},\n");
        fprintf(f, "        {\"regex_\",   \"std/regex\"},\n");
        fprintf(f, "        {\"process_\", \"std/process\"},\n");
        fprintf(f, "        {\"json_\",    \"std/json\"},\n");
        fprintf(f, "        {\"bstr_\",    \"std/bstring\"},\n");
        fprintf(f, "        {NULL, NULL}\n");
        fprintf(f, "    };\n\n");

        fprintf(f, "    for (uint32_t i = 0; i < module->import_count; i++) {\n");
        fprintf(f, "        const char *fn_name = nvm_get_string(module, module->imports[i].function_name_idx);\n");
        fprintf(f, "        const char *mod_name = nvm_get_string(module, module->imports[i].module_name_idx);\n");
        fprintf(f, "        if (fn_name && (!mod_name || mod_name[0] == '\\0')) {\n");
        fprintf(f, "            for (int k = 0; known_modules[k].prefix; k++) {\n");
        fprintf(f, "                if (strncmp(fn_name, known_modules[k].prefix,\n");
        fprintf(f, "                           strlen(known_modules[k].prefix)) == 0) {\n");
        fprintf(f, "                    vm_ffi_load_module(known_modules[k].module);\n");
        fprintf(f, "                    break;\n");
        fprintf(f, "                }\n");
        fprintf(f, "            }\n");
        fprintf(f, "        }\n");
        fprintf(f, "    }\n\n");
    }

    /* VM init */
    fprintf(f, "    VmState vm;\n");
    fprintf(f, "    vm_init(&vm, module);\n\n");

    /* Call __init__ */
    fprintf(f, "    /* Call __init__ to initialize globals before main */\n");
    fprintf(f, "    for (uint32_t i = 0; i < module->function_count; i++) {\n");
    fprintf(f, "        const char *fn_name = nvm_get_string(module, module->functions[i].name_idx);\n");
    fprintf(f, "        if (fn_name && strcmp(fn_name, \"__init__\") == 0) {\n");
    fprintf(f, "            VmResult ir = vm_call_function(&vm, i, NULL, 0);\n");
    fprintf(f, "            if (ir != VM_OK) {\n");
    fprintf(f, "                fprintf(stderr, \"runtime error in __init__: %%s\\n\",\n");
    fprintf(f, "                        vm.error_msg[0] ? vm.error_msg : vm_error_string(ir));\n");
    fprintf(f, "                vm_destroy(&vm);\n");
    fprintf(f, "                nvm_module_free(module);\n");
    if (module->import_count > 0) {
        fprintf(f, "                vm_ffi_shutdown();\n");
    }
    fprintf(f, "                return 1;\n");
    fprintf(f, "            }\n");
    fprintf(f, "            break;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n\n");

    /* Execute */
    fprintf(f, "    int exit_code = 0;\n");
    fprintf(f, "    VmResult r = vm_execute(&vm);\n");
    fprintf(f, "    if (r != VM_OK) {\n");
    fprintf(f, "        fprintf(stderr, \"runtime error: %%s\\n\", vm.error_msg[0] ? vm.error_msg : vm_error_string(r));\n");
    fprintf(f, "        exit_code = 1;\n");
    fprintf(f, "    } else {\n");
    fprintf(f, "        NanoValue result = vm_get_result(&vm);\n");
    fprintf(f, "        if (result.tag == TAG_INT) {\n");
    fprintf(f, "            exit_code = (int)result.as.i64;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n\n");

    /* Cleanup */
    fprintf(f, "    vm_destroy(&vm);\n");
    if (module->import_count > 0) {
        fprintf(f, "    vm_ffi_shutdown();\n");
    }
    fprintf(f, "    nvm_module_free(module);\n");
    fprintf(f, "    return exit_code;\n");
    fprintf(f, "}\n");

    return true;
}

/* ========================================================================
 * Build Object List
 * ======================================================================== */

static bool build_obj_list(char *buf, size_t buf_size, const char *obj_dir) {
    /* These match exactly the objects linked for nano_virt in Makefile.gnu,
     * minus nanovirt/main.o and nanovirt/codegen.o (not needed at runtime) */
    static const char *nanovm_objs[] = {
        "nanovm/value.o", "nanovm/heap.o", "nanovm/vm.o",
        "nanovm/vm_ffi.o", "nanovm/vm_builtins.o", "nanovm/cop_protocol.o", NULL
    };
    static const char *nanoisa_objs[] = {
        "nanoisa/isa.o", "nanoisa/nvm_format.o",
        "nanoisa/assembler.o", "nanoisa/disassembler.o", NULL
    };
    static const char *common_objs[] = {
        "lexer.o", "parser.o", "typechecker.o", "transpiler.o",
        "stdlib_runtime.o", "env.o", "builtins_registry.o",
        "module.o", "module_metadata.o",
        "cJSON.o", "toon_output.o", "module_builder.o",
        "resource_tracking.o", "eval.o", "interpreter_ffi.o",
        "json_diagnostics.o", "reflection.o",
        "eval/eval_hashmap.o", "eval/eval_math.o",
        "eval/eval_string.o", "eval/eval_io.o", NULL
    };
    static const char *runtime_objs[] = {
        "runtime/list_int.o", "runtime/list_string.o",
        "runtime/list_LexerToken.o", "runtime/list_token.o",
        "runtime/list_CompilerDiagnostic.o", "runtime/list_CompilerSourceLocation.o",
        "runtime/list_ASTNumber.o", "runtime/list_ASTFloat.o",
        "runtime/list_ASTString.o", "runtime/list_ASTBool.o",
        "runtime/list_ASTIdentifier.o",
        "runtime/list_ASTBinaryOp.o", "runtime/list_ASTCall.o",
        "runtime/list_ASTModuleQualifiedCall.o",
        "runtime/list_ASTArrayLiteral.o", "runtime/list_ASTLet.o",
        "runtime/list_ASTSet.o", "runtime/list_ASTStmtRef.o",
        "runtime/list_ASTIf.o", "runtime/list_ASTWhile.o",
        "runtime/list_ASTFor.o", "runtime/list_ASTReturn.o",
        "runtime/list_ASTBlock.o", "runtime/list_ASTUnsafeBlock.o",
        "runtime/list_ASTPrint.o", "runtime/list_ASTAssert.o",
        "runtime/list_ASTFunction.o", "runtime/list_ASTShadow.o",
        "runtime/list_ASTStruct.o", "runtime/list_ASTStructLiteral.o",
        "runtime/list_ASTFieldAccess.o", "runtime/list_ASTEnum.o",
        "runtime/list_ASTUnion.o", "runtime/list_ASTUnionConstruct.o",
        "runtime/list_ASTMatch.o", "runtime/list_ASTImport.o",
        "runtime/list_ASTOpaqueType.o", "runtime/list_ASTTupleLiteral.o",
        "runtime/list_ASTTupleIndex.o",
        "runtime/token_helpers.o", "runtime/gc.o", "runtime/dyn_array.o",
        "runtime/gc_struct.o", "runtime/nl_string.o", "runtime/ffi_loader.o",
        "runtime/cli.o", "runtime/regex.o", NULL
    };

    buf[0] = '\0';
    size_t offset = 0;

    const char **groups[] = { nanovm_objs, nanoisa_objs, common_objs, runtime_objs, NULL };
    for (int g = 0; groups[g]; g++) {
        for (int i = 0; groups[g][i]; i++) {
            int n = snprintf(buf + offset, buf_size - offset, "%s/%s ",
                             obj_dir, groups[g][i]);
            if (n < 0 || (size_t)n >= buf_size - offset) {
                return false;
            }
            offset += (size_t)n;
        }
    }

    return true;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

bool wrapper_generate(const NvmModule *module, const uint8_t *blob, uint32_t blob_size,
                      const char *output_path, const char *source_path,
                      const ASTNode *program, bool verbose) {
    (void)source_path;

    /* Find object directory */
    char *obj_dir = find_obj_dir();
    if (!obj_dir) {
        fprintf(stderr, "error: cannot find obj/ directory for linking\n");
        fprintf(stderr, "  Set NANO_VIRT_LIB environment variable or build from project root\n");
        return false;
    }

    /* Verify a key object file exists */
    char test_obj[4096];
    snprintf(test_obj, sizeof(test_obj), "%s/nanovm/vm.o", obj_dir);
    if (access(test_obj, R_OK) != 0) {
        fprintf(stderr, "error: cannot find %s\n", test_obj);
        fprintf(stderr, "  Run 'make -f Makefile.gnu nano_virt' first to build .o files\n");
        free(obj_dir);
        return false;
    }

    /* Generate temp C file */
    char temp_c[256];
    snprintf(temp_c, sizeof(temp_c), "/tmp/nanovirt_%d.c", getpid());

    FILE *f = fopen(temp_c, "w");
    if (!f) {
        fprintf(stderr, "error: cannot create temp file %s\n", temp_c);
        free(obj_dir);
        return false;
    }

    if (!write_wrapper_c(f, module, blob, blob_size, program)) {
        fprintf(stderr, "error: failed to generate wrapper C code\n");
        fclose(f);
        remove(temp_c);
        free(obj_dir);
        return false;
    }
    fclose(f);

    /* Build object file list */
    char obj_list[16384];
    if (!build_obj_list(obj_list, sizeof(obj_list), obj_dir)) {
        fprintf(stderr, "error: object file list too long\n");
        remove(temp_c);
        free(obj_dir);
        return false;
    }

    /* Find the src/ include directory (sibling to obj/) */
    /* obj_dir is like /path/to/project/obj, so src/ is ../src relative to it */
    char src_dir[4096];
    snprintf(src_dir, sizeof(src_dir), "%s/../src", obj_dir);
    char *real_src = realpath(src_dir, NULL);
    if (!real_src) {
        /* Fallback: try ./src */
        real_src = realpath("src", NULL);
        if (!real_src) {
            fprintf(stderr, "error: cannot find src/ include directory\n");
            remove(temp_c);
            free(obj_dir);
            return false;
        }
    }

    /* Select compiler */
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";

    /* Platform-specific flags */
    const char *export_dynamic = "";
#ifdef __linux__
    export_dynamic = "-rdynamic";
#elif defined(__FreeBSD__)
    export_dynamic = "-Wl,-E";
#endif

    /* Compile command */
    char cmd[32768];
    int cmd_len = snprintf(cmd, sizeof(cmd),
            "%s -std=c99 -Wall -Wextra -Werror "
            "-Wno-error=unused-function -Wno-error=unused-parameter "
            "-Wno-error=unused-variable -Wno-error=unused-but-set-variable "
            "%s -I%s -o %s %s %s -lm",
            cc, export_dynamic, real_src, output_path, temp_c, obj_list);

    if (cmd_len >= (int)sizeof(cmd)) {
        fprintf(stderr, "error: compile command too long\n");
        remove(temp_c);
        free(real_src);
        free(obj_dir);
        return false;
    }

    if (verbose) {
        printf("Compiling wrapper: %s\n", cmd);
    }

    int result = system(cmd);

    /* Cleanup */
    remove(temp_c);
    free(real_src);
    free(obj_dir);

    if (result != 0) {
        fprintf(stderr, "error: native compilation failed (exit code %d)\n", result);
        return false;
    }

    return true;
}

/* ========================================================================
 * Daemon-Mode Wrapper Generation
 *
 * Produces a thin binary that embeds the .nvm blob and uses the VMD client
 * library to connect to nano_vmd for execution. Links only:
 *   - vmd_protocol.o
 *   - vmd_client.o
 * ======================================================================== */

static bool write_daemon_wrapper_c(FILE *f, const uint8_t *blob, uint32_t blob_size) {
    fprintf(f, "/* Auto-generated NVM daemon wrapper - do not edit */\n");
    fprintf(f, "#include \"nanovm/vmd_client.h\"\n");
    fprintf(f, "#include <stdio.h>\n");
    fprintf(f, "#include <stdint.h>\n\n");

    /* Embedded blob */
    fprintf(f, "static const unsigned char nvm_blob[%u] = {\n", blob_size);
    for (uint32_t i = 0; i < blob_size; i++) {
        if (i % 16 == 0) fprintf(f, "    ");
        fprintf(f, "0x%02x", blob[i]);
        if (i + 1 < blob_size) fprintf(f, ",");
        if (i % 16 == 15 || i + 1 == blob_size) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");

    fprintf(f, "int main(int argc, char **argv) {\n");
    fprintf(f, "    (void)argc; (void)argv;\n\n");

    fprintf(f, "    VmdClient *client = vmd_connect(5000);\n");
    fprintf(f, "    if (!client) {\n");
    fprintf(f, "        fprintf(stderr, \"error: cannot connect to nano_vmd daemon\\n\");\n");
    fprintf(f, "        return 1;\n");
    fprintf(f, "    }\n\n");

    fprintf(f, "    int exit_code = vmd_execute(client, nvm_blob, %u);\n", blob_size);
    fprintf(f, "    vmd_disconnect(client);\n\n");

    fprintf(f, "    if (exit_code < 0) {\n");
    fprintf(f, "        fprintf(stderr, \"error: communication error with daemon\\n\");\n");
    fprintf(f, "        return 1;\n");
    fprintf(f, "    }\n\n");

    fprintf(f, "    return exit_code;\n");
    fprintf(f, "}\n");

    return true;
}

bool wrapper_generate_daemon(const uint8_t *blob, uint32_t blob_size,
                              const char *output_path, bool verbose) {
    /* Find object directory */
    char *obj_dir = find_obj_dir();
    if (!obj_dir) {
        fprintf(stderr, "error: cannot find obj/ directory for linking\n");
        return false;
    }

    /* Verify the VMD client object exists */
    char test_obj[4096];
    snprintf(test_obj, sizeof(test_obj), "%s/nanovm/vmd_client.o", obj_dir);
    if (access(test_obj, R_OK) != 0) {
        fprintf(stderr, "error: cannot find %s\n", test_obj);
        fprintf(stderr, "  Run 'make -f Makefile.gnu nano_vm' first to build VMD client objects\n");
        free(obj_dir);
        return false;
    }

    /* Generate temp C file */
    char temp_c[256];
    snprintf(temp_c, sizeof(temp_c), "/tmp/nanovirt_daemon_%d.c", getpid());

    FILE *f = fopen(temp_c, "w");
    if (!f) {
        fprintf(stderr, "error: cannot create temp file %s\n", temp_c);
        free(obj_dir);
        return false;
    }

    if (!write_daemon_wrapper_c(f, blob, blob_size)) {
        fprintf(stderr, "error: failed to generate daemon wrapper C code\n");
        fclose(f);
        remove(temp_c);
        free(obj_dir);
        return false;
    }
    fclose(f);

    /* Daemon wrappers need only: vmd_protocol.o + vmd_client.o */
    char obj_list[4096];
    snprintf(obj_list, sizeof(obj_list), "%s/nanovm/vmd_protocol.o %s/nanovm/vmd_client.o",
             obj_dir, obj_dir);

    /* Find the src/ include directory */
    char src_dir[4096];
    snprintf(src_dir, sizeof(src_dir), "%s/../src", obj_dir);
    char *real_src = realpath(src_dir, NULL);
    if (!real_src) {
        real_src = realpath("src", NULL);
        if (!real_src) {
            fprintf(stderr, "error: cannot find src/ include directory\n");
            remove(temp_c);
            free(obj_dir);
            return false;
        }
    }

    /* Select compiler */
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";

    /* Compile command — much simpler than full wrapper */
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
             "%s -std=c99 -Wall -Wextra -Werror "
             "-Wno-error=unused-parameter "
             "-I%s -o %s %s %s",
             cc, real_src, output_path, temp_c, obj_list);

    if (verbose) {
        printf("Compiling daemon wrapper: %s\n", cmd);
    }

    int result = system(cmd);

    remove(temp_c);
    free(real_src);
    free(obj_dir);

    if (result != 0) {
        fprintf(stderr, "error: daemon wrapper compilation failed (exit code %d)\n", result);
        return false;
    }

    return true;
}
