/*
 * nano_virt main.c - CLI for compiling .nano to .nvm bytecode
 *
 * Usage: nano_virt input.nano [-o output.nvm] [--run]
 *
 * Pipeline: .nano → lexer → parser → typechecker → codegen → .nvm
 * With --run: also executes the .nvm via the embedded VM
 */

#include "nanolang.h"
#include "nanovirt/codegen.h"
#include "nanoisa/nvm_format.h"
#include "nanovm/vm.h"
#include "nanovm/value.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    fprintf(stderr, "Usage: %s <input.nano> [-o output.nvm] [--run]\n", prog);
}

int main(int argc, char **argv) {
    const char *input = NULL;
    const char *output = NULL;
    bool run = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--run") == 0) {
            run = true;
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

    /* Environment + Type Checking */
    Environment *env = create_environment();
    if (!type_check(program, env)) {
        fprintf(stderr, "error: type check failed\n");
        free_ast(program);
        free_environment(env);
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Codegen */
    CodegenResult cg = codegen_compile(program, env);
    if (!cg.ok) {
        fprintf(stderr, "error: codegen failed at line %d: %s\n",
                cg.error_line, cg.error_msg);
        free_ast(program);
        free_environment(env);
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }

    /* Write .nvm if requested */
    if (output) {
        uint32_t size = 0;
        uint8_t *blob = nvm_serialize(cg.module, &size);
        if (!blob) {
            fprintf(stderr, "error: serialization failed\n");
            nvm_module_free(cg.module);
            free_ast(program);
            free_environment(env);
            free_tokens(tokens, token_count);
            free(source);
            return 1;
        }

        FILE *f = fopen(output, "wb");
        if (!f) {
            fprintf(stderr, "error: cannot write '%s'\n", output);
            free(blob);
            nvm_module_free(cg.module);
            free_ast(program);
            free_environment(env);
            free_tokens(tokens, token_count);
            free(source);
            return 1;
        }
        fwrite(blob, 1, size, f);
        fclose(f);
        free(blob);
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

    nvm_module_free(cg.module);
    free_ast(program);
    free_environment(env);
    free_tokens(tokens, token_count);
    free(source);
    return exit_code;
}
