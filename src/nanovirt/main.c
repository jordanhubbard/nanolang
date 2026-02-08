/*
 * nano_virt main.c - CLI for compiling .nano to .nvm bytecode or native binary
 *
 * Usage: nano_virt input.nano [-o output] [--run] [--emit-nvm] [-v]
 *
 * Pipeline: .nano → lexer → parser → typechecker → codegen → .nvm
 * With -o:        writes .nvm bytecode (if .nvm extension or --emit-nvm)
 *                 or native executable (otherwise, via wrapper_gen)
 * With --run:     executes the .nvm via the embedded VM
 */

#include "nanolang.h"
#include "nanovirt/codegen.h"
#include "nanovirt/wrapper_gen.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/vm.h"
#include "nanovm/vm_ffi.h"
#include "nanovm/value.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration for interpreter FFI module loading (already linked) */
extern bool ffi_load_module(const char *module_name, const char *module_path,
                            Environment *env, bool verbose);

/* Globals expected by runtime/cli.c */
int g_argc = 0;
char **g_argv = NULL;

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len < 0 || len > 10 * 1024 * 1024) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)len, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input.nano> [-o output] [--run] [--emit-nvm] [-v]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -o <path>     Output file (native binary, or .nvm if --emit-nvm)\n");
    fprintf(stderr, "  --run         Execute after compilation (in-process VM)\n");
    fprintf(stderr, "  --emit-nvm    Write raw .nvm bytecode instead of native binary\n");
    fprintf(stderr, "  -v            Verbose output\n");
}

/* Check if path ends with .nvm extension */
static bool has_nvm_extension(const char *path) {
    size_t len = strlen(path);
    return (len >= 4 && strcmp(path + len - 4, ".nvm") == 0);
}

int main(int argc, char **argv) {
    const char *input = NULL;
    const char *output = NULL;
    bool run = false;
    bool emit_nvm = false;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--run") == 0) {
            run = true;
        } else if (strcmp(argv[i], "--emit-nvm") == 0) {
            emit_nvm = true;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (argv[i][0] != '-') {
            input = argv[i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!input) { usage(argv[0]); return 1; }

    /* Read source */
    char *source = read_file(input);
    if (!source) {
        fprintf(stderr, "error: cannot read '%s'\n", input);
        return 1;
    }

    /* Lexer */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        fprintf(stderr, "error: lexer failed\n");
        free(source);
        return 1;
    }

    /* Parser */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        fprintf(stderr, "error: parser failed\n");
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Environment + Module Resolution */
    clear_module_cache();
    Environment *env = create_environment();

    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, input)) {
        fprintf(stderr, "error: module loading failed\n");
        free_ast(program);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Type Checking */
    typecheck_set_current_file(input);
    if (!type_check(program, env)) {
        fprintf(stderr, "error: type check failed\n");
        free_ast(program);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Codegen */
    CodegenResult cg = codegen_compile(program, env, modules, input);
    if (!cg.ok) {
        fprintf(stderr, "error: codegen failed at line %d: %s\n",
                cg.error_line, cg.error_msg);
        free_ast(program);
        free_environment(env);
        free_module_list(modules);
        clear_module_cache();
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Write output if requested */
    if (output) {
        uint32_t size = 0;
        uint8_t *blob = nvm_serialize(cg.module, &size);
        if (!blob) {
            fprintf(stderr, "error: serialization failed\n");
            nvm_module_free(cg.module);
            free_ast(program);
            free_environment(env);
            free_module_list(modules);
            clear_module_cache();
            free_tokens(tokens, token_count);
            free(source);
            return 1;
        }

        bool want_nvm = emit_nvm || has_nvm_extension(output);

        if (want_nvm) {
            /* Write raw .nvm bytecode */
            FILE *f = fopen(output, "wb");
            if (!f) {
                fprintf(stderr, "error: cannot write '%s'\n", output);
                free(blob);
                nvm_module_free(cg.module);
                free_ast(program);
                free_environment(env);
                free_module_list(modules);
                clear_module_cache();
                free_tokens(tokens, token_count);
                free(source);
                return 1;
            }
            fwrite(blob, 1, size, f);
            fclose(f);
            if (verbose) {
                printf("Wrote %u bytes of NVM bytecode to %s\n", size, output);
            }
        } else {
            /* Generate native executable */
            if (verbose) {
                printf("Generating native executable: %s\n", output);
            }
            if (!wrapper_generate(cg.module, blob, size, output, input,
                                  program, verbose)) {
                free(blob);
                nvm_module_free(cg.module);
                free_ast(program);
                free_environment(env);
                free_module_list(modules);
                clear_module_cache();
                free_tokens(tokens, token_count);
                free(source);
                return 1;
            }
            if (verbose) {
                printf("Native executable generated: %s\n", output);
            }
        }
        free(blob);
    }

    /* Preload modules for FFI if the module has imports */
    if (cg.module->import_count > 0) {
        vm_ffi_init();
        vm_ffi_set_env(env);

        /* Load modules referenced in import table */
        for (uint32_t i = 0; i < cg.module->import_count; i++) {
            const char *mod_name = nvm_get_string(cg.module,
                                                   cg.module->imports[i].module_name_idx);
            if (mod_name && mod_name[0] != '\0') {
                vm_ffi_load_module(mod_name);
            }
        }

        /* Also scan for AST_IMPORT nodes to load modules by path.
         * This handles the case where extern fns have empty module names
         * but the .nano file has module/import statements. */
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
            if (item->type == AST_IMPORT) {
                const char *mod_path = item->as.import_stmt.module_path;
                if (mod_path) {
                    vm_ffi_load_module(mod_path);
                }
            }
        }

        /* Try loading well-known standard modules for bare extern fns.
         * Map function name prefixes to module paths. */
        static const struct { const char *prefix; const char *module; } known_modules[] = {
            {"path_",    "std/fs"},
            {"fs_",      "std/fs"},
            {"file_",    "std/fs"},
            {"dir_",     "std/fs"},
            {"regex_",   "std/regex"},
            {"process_", "std/process"},
            {"json_",    "std/json"},
            {"bstr_",    "std/bstring"},
            {NULL, NULL}
        };

        for (uint32_t i = 0; i < cg.module->import_count; i++) {
            const char *fn_name = nvm_get_string(cg.module,
                                                  cg.module->imports[i].function_name_idx);
            const char *mod_name = nvm_get_string(cg.module,
                                                   cg.module->imports[i].module_name_idx);
            if (fn_name && (!mod_name || mod_name[0] == '\0')) {
                for (int k = 0; known_modules[k].prefix; k++) {
                    if (strncmp(fn_name, known_modules[k].prefix,
                               strlen(known_modules[k].prefix)) == 0) {
                        vm_ffi_load_module(known_modules[k].module);
                        break;
                    }
                }
            }
        }
    }

    int exit_code = 0;

    /* Execute if requested */
    if (run) {
        VmState vm;
        vm_init(&vm, cg.module);

        /* Call __init__ to initialize globals before main */
        for (uint32_t i = 0; i < cg.module->function_count; i++) {
            const char *fn_name = nvm_get_string(cg.module,
                                                  cg.module->functions[i].name_idx);
            if (fn_name && strcmp(fn_name, "__init__") == 0) {
                VmResult ir = vm_call_function(&vm, i, NULL, 0);
                if (ir != VM_OK) {
                    fprintf(stderr, "runtime error in __init__: %s\n",
                            vm.error_msg[0] ? vm.error_msg : vm_error_string(ir));
                    vm_destroy(&vm);
                    nvm_module_free(cg.module);
                    free_ast(program);
                    free_environment(env);
                    free_module_list(modules);
                    clear_module_cache();
                    free_tokens(tokens, token_count);
                    free(source);
                    return 1;
                }
                break;
            }
        }

        VmResult r = vm_execute(&vm);
        if (r != VM_OK) {
            fprintf(stderr, "runtime error: %s\n", vm.error_msg[0] ? vm.error_msg : vm_error_string(r));
            exit_code = 1;
        } else {
            NanoValue result = vm_get_result(&vm);
            if (result.tag == TAG_INT) {
                exit_code = (int)result.as.i64;
            }
        }
        vm_destroy(&vm);
    }

    vm_ffi_shutdown();
    nvm_module_free(cg.module);
    free_ast(program);
    free_environment(env);
    free_module_list(modules);
    clear_module_cache();
    free_tokens(tokens, token_count);
    free(source);
    return exit_code;
}
