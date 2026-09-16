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
#include "../shell_path.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>

static char *wrapper_path(const char *directory, const char *name) {
    size_t a = strlen(directory), b = strlen(name);
    if (a > SIZE_MAX - b - 2) return NULL;
    char *path = malloc(a + b + 2);
    if (path) snprintf(path, a + b + 2, "%s/%s", directory, name);
    return path;
}

/* I encode path bytes as fixed-width octal C escapes, not source syntax. */
static void write_path_literal(FILE *f, const char *path) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)path; *p; p++)
        fprintf(f, "\\%03o", *p);
    fputc('"', f);
}

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
            if (env_dir[0] == '/') return strdup(env_dir);
            char *cwd = getcwd(NULL, 0);
            if (!cwd) return NULL;
            char *absolute = wrapper_path(cwd, env_dir);
            free(cwd);
            return absolute;
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
            int n = snprintf(obj_path, sizeof(obj_path), "%s/../obj", dir);
            if (n >= 0 && (size_t)n < sizeof(obj_path) && access(obj_path, R_OK) == 0) {
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
            int n = snprintf(obj_path, sizeof(obj_path), "%s/../obj", dir);
            if (n >= 0 && (size_t)n < sizeof(obj_path) && access(obj_path, R_OK) == 0) {
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
    fprintf(f, "#include \"nanoisa/nanoisa.h\"\n");
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
    fprintf(f, "    NanoisaErr load_error;\n");
    fprintf(f, "    NvmModule *module = nanoisa_load_bytes(nvm_blob, %u, &load_error);\n",
            blob_size);
    fprintf(f, "    if (!module) {\n");
    fprintf(f, "        fprintf(stderr, \"error: %%s\\n\", load_error.message);\n");
    fprintf(f, "        return 1;\n");
    fprintf(f, "    }\n\n");

    /* FFI init + module loading (only if there are imports) */
    if (module->import_count > 0) {
        fprintf(f, "    /* Initialize FFI and load modules */\n");
        fprintf(f, "    vm_ffi_init();\n\n");

        /* Load modules from import table */
        fprintf(f, "    /* Load modules referenced in import table */\n");
        fprintf(f, "    for (uint32_t i = 0; i < module->import_count; i++) {\n");
        fprintf(f, "        vm_ffi_load_import(module, i);\n");
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
                    fprintf(f, "    vm_ffi_load_module(");
                    write_path_literal(f, item->as.import_stmt.module_path);
                    fprintf(f, ");\n");
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

static bool build_obj_list(char *buf, size_t buf_size, const char *obj_dir, bool daemon) {
    /* I keep the runtime link closure here and exercise it in wrapper tests. */
    static const char *daemon_objs[] = {
        "nanovm/vmd_protocol.o", "nanovm/vmd_client.o", NULL
    };
    static const char *nanovm_objs[] = {
        "nanovm/value.o", "nanovm/heap.o", "nanovm/heap_cycles.o", "nanovm/vm.o",
        "nanovm/vm_ffi.o", "nanovm/vm_ffi_arrays.o", "nanovm/vm_builtins.o", "nanovm/cop_protocol.o",
        "nanovm/vm_callback.o", "runtime/callback_runtime.o",
        /* asm_assemble() verifies its output, so anything linking the
         * assembler also needs the verifier and the decode/dispatch tables it
         * checks against. */
        "nanovm/vm_decode.o", "nanovm/vm_dispatch.o", NULL
    };
    static const char *nanoisa_objs[] = {
        "nanoisa/isa.o", "nanoisa/nvm_format.o",
        "nanoisa/nvm_format_v2.o", "nanoisa/nvm_v2_cursor.o", "nanoisa/nvm_v2_constants.o", "nanoisa/nvm_v2_signatures.o", "nanoisa/nvm_v2_layouts.o", "nanoisa/nvm_v2_functions.o", "nanoisa/nvm_v2_imports.o", "nanoisa/nvm_v2_module.o", "nanoisa/nvm_v2_convert.o",
        "nanoisa/assembler.o", "nanoisa/disassembler.o",
        "nanoisa/verifier.o", "nanoisa/verifier_types.o", "nanoisa/nanoisa_facade.o", NULL
    };
    static const char *common_objs[] = {
        "lexer.o", "parser.o", "typechecker.o", "transpiler.o",
        "stdlib_runtime.o", "env.o", "builtins_registry.o",
        "module.o", "module_metadata.o", "utf8.o",
        "cJSON.o", "toon_output.o", "module_builder.o",
        "resource_tracking.o", "resource_flow.o", "eval.o", "interpreter_ffi.o",
        "json_diagnostics.o", "reflection.o", "effects.o", "coroutine.o",
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
        "runtime/module_build_dir.o", "runtime/cli.o", "runtime/regex.o", NULL
    };

    buf[0] = '\0';
    const char **groups[] = { daemon ? daemon_objs : nanovm_objs,
                             daemon ? NULL : nanoisa_objs, common_objs, runtime_objs, NULL };
    for (int g = 0; groups[g]; g++) {
        for (int i = 0; groups[g][i]; i++) {
            char *path = wrapper_path(obj_dir, groups[g][i]);
            if (!path) return false;
            bool ok = module_append_path_flag(buf, buf_size, "", path);
            free(path);
            if (!ok) return false;
        }
    }

    return true;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

static bool write_daemon_wrapper_c(FILE *f, const uint8_t *blob, uint32_t blob_size);

/* I stage beside the destination so successful rename is one filesystem
 * operation. Compilers are trusted configuration, not a sandbox boundary. */
#ifndef NANO_WRAPPER_INSTRUMENT_FLAGS
#define NANO_WRAPPER_INSTRUMENT_FLAGS ""
#endif

static bool build_wrapper(const NvmModule *module, const uint8_t *blob,
                          uint32_t blob_size, const char *output_path,
                          const ASTNode *program, bool daemon, bool verbose) {
    bool ok = false, staged = false;
    char *obj_dir = NULL, *parent_input = NULL, *parent = NULL, *output = NULL;
    char *stage = NULL, *source = NULL, *binary = NULL, *wrapper_object = NULL;
    char *src_candidate = NULL, *modules_candidate = NULL;
    char *src = NULL, *modules = NULL;
    FILE *f = NULL;
    if (!output_path || !*output_path || !blob || !blob_size || (!daemon && !module)) {
        fprintf(stderr, "I require a module and a nonempty wrapper output path\n");
        return false;
    }

    parent_input = strdup(output_path);
    if (!parent_input) goto cleanup;
    char *slash = strrchr(parent_input, '/');
    const char *name = slash ? slash + 1 : output_path;
    if (!*name || strcmp(name, ".") == 0 || strcmp(name, "..") == 0) goto cleanup;
    char *base = strdup(name);
    if (!base) goto cleanup;
    if (slash == parent_input) slash[1] = '\0';
    else if (slash) *slash = '\0';
    else strcpy(parent_input, ".");
    parent = realpath(parent_input, NULL);
    if (parent) output = wrapper_path(parent, base);
    free(base);
    if (!parent || !output) goto cleanup;

    obj_dir = find_obj_dir();
    if (!obj_dir) goto cleanup;
    src_candidate = wrapper_path(obj_dir, "../src");
    modules_candidate = wrapper_path(obj_dir, "../modules");
    if (!src_candidate || !modules_candidate) goto cleanup;
    src = realpath(src_candidate, NULL);
    if (!src) src = realpath("src", NULL);
    if (!src) goto cleanup;
    if (!daemon) {
        modules = realpath(modules_candidate, NULL);
        if (!modules) modules = realpath("modules", NULL);
        if (!modules) goto cleanup;
    }

    stage = wrapper_path(parent, ".nano-wrapper-XXXXXX");
    if (!stage || !mkdtemp(stage)) goto cleanup;
    staged = true;
    source = wrapper_path(stage, "source.c");
    binary = wrapper_path(stage, "executable");
    wrapper_object = wrapper_path(stage, "wrapper.o");
    if (!source || !binary || !wrapper_object) goto cleanup;
    f = fopen(source, "wx");
    if (!f) goto cleanup;
    bool written = daemon ? write_daemon_wrapper_c(f, blob, blob_size)
                          : write_wrapper_c(f, module, blob, blob_size, program);
    written = written && !ferror(f);
    if (fclose(f) != 0) written = false;
    f = NULL;
    if (!written) goto cleanup;

    char objects[16384];
    if (!build_obj_list(objects, sizeof(objects), obj_dir, daemon)) goto cleanup;
    const char *cc = getenv("NANO_CC");
    if (!cc) cc = getenv("CC");
    if (!cc) cc = "cc";
    char command[32768];
    int n = snprintf(command, sizeof(command),
                     "%s -std=c99 -Wall -Wextra -Werror "
                     "-Wno-error=unused-function -Wno-error=unused-parameter "
                     "-Wno-error=unused-variable -Wno-error=unused-but-set-variable -c ",
                     cc);
    if (n < 0 || (size_t)n >= sizeof(command)) goto cleanup;
    if (!module_append_include(command, sizeof(command), src) ||
        (modules && !module_append_include(command, sizeof(command), modules)) ||
        !module_append_path_flag(command, sizeof(command), "-o ", wrapper_object) ||
        !module_append_path_flag(command, sizeof(command), "", source)) goto cleanup;
    /* I compile this temporary wrapper without runtime instrumentation, then
     * link the instrumented runtime. Coverage must not create reports in a
     * private staging directory that I remove before execution. */
    if (system(command) != 0) goto cleanup;
    struct stat wrapper_stat;
    if (lstat(wrapper_object, &wrapper_stat) != 0 || !S_ISREG(wrapper_stat.st_mode) ||
        wrapper_stat.st_size <= 0 || wrapper_stat.st_nlink != 1) goto cleanup;
    n = snprintf(command, sizeof(command), "%s ", cc);
    if (n < 0 || (size_t)n >= sizeof(command) ||
        !module_append_path_flag(command, sizeof(command), "-o ", binary) ||
        !module_append_path_flag(command, sizeof(command), "", wrapper_object)) goto cleanup;
    if (!daemon) {
#ifdef NANO_WRAPPER_CRYPTO_DIR
        if (!module_append_path_flag(command, sizeof(command), "-L", NANO_WRAPPER_CRYPTO_DIR))
            goto cleanup;
#endif
    }
    size_t used = strlen(command);
    const char *platform = "";
#if defined(__linux__)
    if (!daemon) platform = "-rdynamic -ldl";
#elif defined(__FreeBSD__)
    if (!daemon) platform = "-Wl,-E";
#endif
    n = snprintf(command + used, sizeof(command) - used, " %s %s %s %s", objects,
                 daemon ? "" : "-lm -pthread -lcrypto -lffi", platform,
                 NANO_WRAPPER_INSTRUMENT_FLAGS);
    if (n < 0 || (size_t)n >= sizeof(command) - used) goto cleanup;
    if (verbose) printf("I compile a private wrapper: %s\n", command);
    if (system(command) != 0) goto cleanup;

    struct stat st;
    if (lstat(binary, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
        !(st.st_mode & 0111) || st.st_nlink != 1) goto cleanup;
    if (rename(binary, output) != 0) goto cleanup;
    ok = true;

cleanup:
    if (f) fclose(f);
    if (staged) {
        if (source) unlink(source);
        if (binary) unlink(binary);
        if (wrapper_object) unlink(wrapper_object);
        /* I never recursively delete compiler-created unknown files. */
        if (rmdir(stage) != 0 && errno != ENOENT)
            fprintf(stderr, "I retained extra compiler artifacts in %s\n", stage);
    }
    if (!ok) fprintf(stderr, "I could not publish the wrapper; I did not replace the destination\n");
    free(obj_dir); free(parent_input); free(parent); free(output);
    free(stage); free(source); free(binary); free(wrapper_object);
    free(src_candidate); free(modules_candidate); free(src); free(modules);
    return ok;
}

bool wrapper_generate(const NvmModule *module, const uint8_t *blob, uint32_t blob_size,
                      const char *output_path, const char *source_path,
                      const ASTNode *program, bool verbose) {
    (void)source_path;
    return build_wrapper(module, blob, blob_size, output_path, program, false, verbose);
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
    return build_wrapper(NULL, blob, blob_size, output_path, NULL, true, verbose);
}
