/*
 * nvm2c — command-line NanoISA → structured C11 translator.
 *
 * I am a host tool, not a compiler phase. I read a verified .nvm and write
 * C operators. I do not embed nano_vm. Closed subset: see nvm2c.h.
 *
 * Usage: nvm2c <file.nvm> [-o out.c]
 *        nvm2c --help
 */

#include "nanoisa.h"
#include "nvm2c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *out) {
    fprintf(out,
            "I translate a verified NanoISA module to structured C11.\n"
            "I do not embed nano_vm. I am not a NanoLang compiler phase.\n"
            "Usage: nvm2c <file.nvm> [-o out.c]\n"
            "       nvm2c --help\n"
            "Without -o I write C to stdout.\n");
}

int main(int argc, char **argv) {
    const char *in = NULL;
    const char *out = NULL;
    int i;
    NanoisaErr err;
    NvmModule *mod;
    char emit_err[256];
    char *c;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        }
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out = argv[++i];
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "I do not recognize '%s'\n", argv[i]);
            usage(stderr);
            return 2;
        }
        if (in) {
            usage(stderr);
            return 2;
        }
        in = argv[i];
    }
    if (!in) {
        usage(stderr);
        return 2;
    }

    memset(&err, 0, sizeof err);
    mod = nanoisa_load_file(in, &err);
    if (!mod) {
        fprintf(stderr, "I cannot load '%s': %s\n", in, err.message);
        return 1;
    }

    emit_err[0] = '\0';
    c = nvm2c_emit(mod, emit_err, sizeof emit_err);
    nvm_module_free(mod);
    if (!c) {
        fprintf(stderr, "I cannot translate '%s': %s\n",
                in, emit_err[0] ? emit_err : "refused");
        return 1;
    }

    if (!out) {
        fputs(c, stdout);
        free(c);
        return 0;
    }
    {
        FILE *f = fopen(out, "w");
        if (!f) {
            fprintf(stderr, "I cannot write '%s'\n", out);
            free(c);
            return 1;
        }
        fputs(c, f);
        fclose(f);
    }
    free(c);
    return 0;
}
