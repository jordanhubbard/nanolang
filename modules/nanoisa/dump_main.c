/*
 * nanoisa - assemble and dump NanoISA modules
 *
 * Usage: nanoisa dump [--pretty] <file.nvm>
 *        nanoisa asm <file.nasm> -o <file.nvm>
 */

#include "nanoisa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *out) {
    fprintf(out,
            "I assemble NanoISA assembly and dump .nvm files back to it.\n"
            "Usage: nanoisa dump [--pretty] <file.nvm>\n"
            "       nanoisa asm <file.nasm> -o <file.nvm>\n");
}

/* Assembling from the command line is what makes a written .nasm file
 * runnable rather than decorative: the assembler already verifies before it
 * returns a module, so a rejected example fails here rather than at load. */
static int assemble_file(const char *src_path, const char *out_path) {
    NanoisaErr err;
    NvmModule *mod = nanoisa_assemble_file(src_path, &err);
    if (!mod) {
        fprintf(stderr, "I cannot assemble '%s': %s\n", src_path, err.message);
        return 1;
    }
    int rc = nanoisa_save_file(mod, out_path, &err);
    nvm_module_free(mod);
    if (rc != NANOISA_OK) {
        fprintf(stderr, "I cannot write '%s': %s\n", out_path, err.message);
        return 1;
    }
    return 0;
}

static int dump_file(const char *path, int pretty) {
    NanoisaErr err;
    NvmModule *mod = nanoisa_load_file(path, &err);
    if (!mod) {
        fprintf(stderr, "I cannot load '%s': %s\n", path, err.message);
        return 1;
    }

    char *text = pretty ? nanoisa_pretty_print(mod) : nanoisa_print(mod);
    nvm_module_free(mod);
    if (!text) {
        fprintf(stderr, "I cannot print '%s'\n", path);
        return 1;
    }

    fputs(text, stdout);
    free(text);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && (strcmp(argv[1], "--help") == 0
            || strcmp(argv[1], "-h") == 0)) {
        usage(stdout);
        return 0;
    }

    if (argc >= 2 && strcmp(argv[1], "asm") == 0) {
        const char *src = NULL, *out = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) { out = argv[++i]; continue; }
            if (argv[i][0] == '-') { usage(stderr); return 2; }
            if (src) { usage(stderr); return 2; }
            src = argv[i];
        }
        if (!src || !out) { usage(stderr); return 2; }
        return assemble_file(src, out);
    }

    if (argc < 3 || strcmp(argv[1], "dump") != 0) {
        usage(stderr);
        return 2;
    }

    int pretty = 0;
    const char *path = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--pretty") == 0 || strcmp(argv[i], "-p") == 0) {
            pretty = 1;
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "I do not recognize '%s'\n", argv[i]);
            usage(stderr);
            return 2;
        }
        if (path) {
            usage(stderr);
            return 2;
        }
        path = argv[i];
    }

    if (!path) {
        usage(stderr);
        return 2;
    }

    return dump_file(path, pretty);
}
