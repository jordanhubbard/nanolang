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
#include "file_public.h"
#include "file_cyclic_public.h"
#include "file_cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *out) {
    fprintf(out,
            "I translate a verified NanoISA module to structured C11.\n"
            "I do not embed nano_vm. I am not a NanoLang compiler phase.\n"
            "Usage: nvm2c <file.nvm> [-o out.c]\n"
            "       nvm2c --file-temporary --entry-name IDENT <file.nvm> [-o out.c]\n"
            "       nvm2c --file-temporary --file-cyclic --entry-name IDENT <file.nvm> [-o out.c]\n"
            "       nvm2c --help\n"
            "Without -o I write C to stdout.\n"
            "For native array/GC artifact imports, link bin/nano_aot_runtime.o\n"
            "from make nvm2c-runtime; see docs/AOT_RUNTIME.md for host export flags.\n"
            "For --file-temporary, link the installed libnano_file_runtime.a and pass an explicit host grant.\n"
            "See docs/FILE_HOST_API.md for the bounded File package.\n");
}

int main(int argc, char **argv) {
    const char *in = NULL;
    const char *out = NULL;
    const char *file_entry = NULL;
    bool file_temporary = false;
    bool file_cyclic = false;
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
        if (strcmp(argv[i], "--file-temporary") == 0) {
            if (file_temporary) { usage(stderr); return 2; }
            file_temporary = true;
            continue;
        }
        if (strcmp(argv[i], "--file-cyclic") == 0) {
            if (file_cyclic) { usage(stderr); return 2; }
            file_cyclic = true;
            continue;
        }
        if (strcmp(argv[i], "--entry-name") == 0 && i + 1 < argc) {
            if (file_entry) { usage(stderr); return 2; }
            file_entry = argv[++i];
            continue;
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

    if (file_temporary != (file_entry != NULL) || (file_cyclic && !file_temporary)) {
        fprintf(stderr,"I require --file-temporary and --entry-name together.\n");
        return 2;
    }
    if (file_temporary) {
        uint8_t *bytes = NULL;
        size_t size = 0;
        c = NULL;
        emit_err[0] = '\0';
        if (!nvm_file_cli_read(in,&bytes,&size,emit_err,sizeof emit_err)) {
            fprintf(stderr,"%s\n",emit_err);
            return 1;
        }
        NvmFileRuntimeStatus status = file_cyclic
            ? nvm2c_emit_file_cyclic_bytes(bytes,size,file_entry,&c,emit_err,sizeof emit_err)
            : nvm2c_emit_file_bytes(bytes,size,file_entry,&c,emit_err,sizeof emit_err);
        free(bytes);
        if (status != NVM_FILE_RUNTIME_OK) {
            fprintf(stderr,"I cannot emit the File profile (%u): %s\n",(unsigned)status,emit_err);
            return 1;
        }
        bool published;
        if (out) published = nvm_file_cli_write(out,c,emit_err,sizeof emit_err);
        else published = fputs(c,stdout) != EOF && fflush(stdout) == 0;
        free(c);
        if (!published) {
            fprintf(stderr,"I cannot publish File C output%s%s\n",out ? ": " : "",out ? emit_err : "");
            return 1;
        }
        return 0;
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
