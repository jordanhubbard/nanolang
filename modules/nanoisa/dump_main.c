/*
 * nanoisa - dump NanoISA modules
 *
 * Usage: nanoisa dump [--pretty] <file.nvm>
 */

#include "nanoisa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *out) {
    fprintf(out,
            "I dump .nvm files as NanoISA assembly.\n"
            "Usage: nanoisa dump [--pretty] <file.nvm>\n");
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
